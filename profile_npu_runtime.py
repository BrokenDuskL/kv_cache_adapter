from __future__ import annotations

import argparse
import importlib
import pathlib
from dataclasses import dataclass
from typing import Callable

import torch
from torch.profiler import record_function

try:
    import torch_npu  # noqa: F401
    from torch_npu.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
except Exception as exc:  # pragma: no cover - depends on Ascend runtime
    torch_npu = None
    ProfilerActivity = None
    profile = None
    schedule = None
    tensorboard_trace_handler = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

import adapter as adapter_mod
from adapter import InMemoryBlockStoreBackend, KVCacheAdapter, STATE_RESIDENT


@dataclass(frozen=True)
class RuntimeConfig:
    num_actual_blocks: int
    num_logical_blocks: int
    batch_size: int
    block_size: int
    hit_rate: float
    seed: int


def _device() -> torch.device:
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit(f"NPU is required; torch_npu profiler import failed: {_IMPORT_ERROR!r}")
    return torch.device("npu")


def _synchronize(device: torch.device) -> None:
    if device.type in {"npu", "privateuseone"} and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _pack_slot_meta(pin_counts: torch.Tensor, usage_counts: torch.Tensor) -> torch.Tensor:
    return adapter_mod._pack_slot_meta(pin_counts, usage_counts)


def _make_pop_slot_meta(num_actual_blocks: int, count: int, scenario: str, device: torch.device) -> torch.Tensor:
    pin_counts = torch.zeros((num_actual_blocks,), dtype=adapter_mod.PIN_COUNT_DTYPE, device=device)
    usage_counts = torch.zeros((num_actual_blocks,), dtype=adapter_mod.USAGE_DTYPE, device=device)
    if scenario == "tail":
        if count > num_actual_blocks:
            raise ValueError("count must be <= num_actual_blocks for tail scenario")
        pin_counts[: num_actual_blocks - count] = 1
    elif scenario == "aged":
        usage_counts.fill_(1)
    elif scenario != "dense":
        raise ValueError(f"unknown pop scenario: {scenario}")
    return _pack_slot_meta(pin_counts, usage_counts)


def _make_blocked_slot_ids(
    *,
    num_actual_blocks: int,
    count: int,
    blocked_count: int,
    scenario: str,
    device: torch.device,
) -> torch.Tensor:
    if blocked_count <= 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    if scenario == "tail":
        max_blocked = max(0, count - 1)
        actual_blocked = min(blocked_count, max_blocked)
        begin = num_actual_blocks - count
        return torch.arange(begin, begin + actual_blocked, dtype=torch.int64, device=device)
    actual_blocked = min(blocked_count, max(0, num_actual_blocks - count))
    return torch.arange(actual_blocked, dtype=torch.int64, device=device)


def _resident_unpinned_ids(adapter: KVCacheAdapter) -> torch.Tensor:
    resident_mask = (adapter._slot_state == STATE_RESIDENT) & (adapter._pin_count == 0)
    resident_ids = adapter._physical_to_logical[resident_mask]
    return resident_ids[resident_ids >= 0]


def _cold_ids(all_ids: torch.Tensor, resident_ids: torch.Tensor) -> torch.Tensor:
    cold_mask = torch.ones(all_ids.shape[0], dtype=torch.bool, device=all_ids.device)
    if resident_ids.numel() > 0:
        cold_mask[resident_ids] = False
    return all_ids[cold_mask]


def _sample_unique(pool: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if count == 0:
        return pool[:0]
    if count > pool.numel():
        raise RuntimeError(f"cannot sample {count} unique ids from pool of {pool.numel()}")
    indices = torch.randperm(pool.numel(), generator=generator, device="cpu")[:count]
    return pool.index_select(0, indices.to(device=pool.device))


def _sample_batch_ids(
    adapter: KVCacheAdapter,
    all_ids: torch.Tensor,
    batch_size: int,
    hit_rate: float,
    generator: torch.Generator,
) -> torch.Tensor:
    resident_ids = _resident_unpinned_ids(adapter)
    cold = _cold_ids(all_ids, resident_ids)
    hit_count = int(round(batch_size * hit_rate))
    miss_count = batch_size - hit_count
    parts = []
    if hit_count > 0:
        parts.append(_sample_unique(resident_ids, hit_count, generator))
    if miss_count > 0:
        parts.append(_sample_unique(cold, miss_count, generator))
    merged = torch.cat(parts, dim=0)
    permutation = torch.randperm(merged.numel(), generator=generator, device="cpu")
    return merged.index_select(0, permutation.to(device=merged.device))


def _random_payloads(
    config: RuntimeConfig,
    generator: torch.Generator,
    *,
    device: torch.device,
) -> torch.Tensor:
    cpu_payload = torch.randn(
        (config.batch_size, config.block_size),
        generator=generator,
        dtype=torch.float32,
    )
    return cpu_payload.to(device=device)


def _make_adapter(config: RuntimeConfig, *, device: torch.device) -> KVCacheAdapter:
    payloads = torch.arange(
        config.num_logical_blocks * config.block_size,
        dtype=torch.float32,
        device=device,
    ).reshape(config.num_logical_blocks, config.block_size)
    backend = InMemoryBlockStoreBackend(payloads)
    actual_blocks = torch.zeros(
        (config.num_actual_blocks, config.block_size),
        dtype=torch.float32,
        device=device,
    )
    return KVCacheAdapter(
        num_actual_blocks=config.num_actual_blocks,
        num_logical_blocks=config.num_logical_blocks,
        actual_blocks=actual_blocks,
        backend=backend,
        prefer_native_extension=True,
    )


def _run_runtime_warmup(
    *,
    adapter: KVCacheAdapter,
    all_ids: torch.Tensor,
    config: RuntimeConfig,
    generator: torch.Generator,
    steps: int,
    device: torch.device,
) -> None:
    for _ in range(steps):
        load_ids = _sample_batch_ids(adapter, all_ids, config.batch_size, config.hit_rate, generator)
        adapter.load(load_ids)
        adapter.release(load_ids)
        save_ids = _sample_batch_ids(adapter, all_ids, config.batch_size, config.hit_rate, generator)
        adapter.save(save_ids, _random_payloads(config, generator, device=device))
    _synchronize(device)


def _profile_runtime_step(
    *,
    adapter: KVCacheAdapter,
    all_ids: torch.Tensor,
    config: RuntimeConfig,
    generator: torch.Generator,
    device: torch.device,
    runtime_op: str,
) -> None:
    if runtime_op in {"load", "both"}:
        with record_function("kvca_profile.prepare_load_ids"):
            load_ids = _sample_batch_ids(adapter, all_ids, config.batch_size, config.hit_rate, generator)
        with record_function("kvca_profile.adapter_load"):
            adapter.load(load_ids)
        with record_function("kvca_profile.adapter_release"):
            adapter.release(load_ids)
    if runtime_op in {"save", "both"}:
        with record_function("kvca_profile.prepare_save_ids"):
            save_ids = _sample_batch_ids(adapter, all_ids, config.batch_size, config.hit_rate, generator)
        with record_function("kvca_profile.prepare_save_payload"):
            save_payload = _random_payloads(config, generator, device=device)
        with record_function("kvca_profile.adapter_save"):
            adapter.save(save_ids, save_payload)


def _make_profiler(args: argparse.Namespace, trace_dir: pathlib.Path):
    if profile is None or ProfilerActivity is None or schedule is None or tensorboard_trace_handler is None:
        raise SystemExit(f"torch_npu profiler is unavailable: {_IMPORT_ERROR!r}")
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=args.repeat),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    )


def _print_summary(prof, *, sort_by: str, row_limit: int) -> None:
    try:
        print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))
    except Exception as exc:  # pragma: no cover - profiler-version dependent
        print(f"could not print profiler summary: {exc!r}")


def profile_runtime(args: argparse.Namespace, device: torch.device) -> None:
    config = RuntimeConfig(
        num_actual_blocks=args.num_actual_blocks,
        num_logical_blocks=args.num_logical_blocks,
        batch_size=args.batch_size,
        block_size=args.block_size,
        hit_rate=args.hit_rate,
        seed=args.seed,
    )
    all_ids = torch.arange(config.num_logical_blocks, dtype=torch.int64, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    adapter = _make_adapter(config, device=device)
    trace_dir = pathlib.Path(args.output_dir) / "runtime"
    trace_dir.mkdir(parents=True, exist_ok=True)
    try:
        initial = all_ids[: config.num_actual_blocks]
        adapter.load(initial)
        _synchronize(device)
        adapter.release(initial)
        _run_runtime_warmup(
            adapter=adapter,
            all_ids=all_ids,
            config=config,
            generator=generator,
            steps=args.prewarm_steps,
            device=device,
        )
        total_steps = (args.wait + args.warmup + args.active) * args.repeat
        with _make_profiler(args, trace_dir) as prof:
            for _ in range(total_steps):
                _profile_runtime_step(
                    adapter=adapter,
                    all_ids=all_ids,
                    config=config,
                    generator=generator,
                    device=device,
                    runtime_op=args.runtime_op,
                )
                prof.step()
        _synchronize(device)
        _print_summary(prof, sort_by=args.sort_by, row_limit=args.row_limit)
    finally:
        adapter.shutdown()
        _synchronize(device)
    print(f"runtime trace written to: {trace_dir}")


def profile_pop(args: argparse.Namespace, device: torch.device) -> None:
    ops = importlib.import_module("kv_cache_adapter_npu_custom")
    blocked_slot_ids = _make_blocked_slot_ids(
        num_actual_blocks=args.num_actual_blocks,
        count=args.count,
        blocked_count=args.blocked_count,
        scenario=args.pop_scenario,
        device=device,
    )
    trace_dir = pathlib.Path(args.output_dir) / "pop"
    trace_dir.mkdir(parents=True, exist_ok=True)
    total_steps = (args.wait + args.warmup + args.active) * args.repeat
    total_calls = args.prewarm_steps + total_steps
    base_slot_meta = _make_pop_slot_meta(args.num_actual_blocks, args.count, args.pop_scenario, device)
    slot_meta_batch = base_slot_meta.unsqueeze(0).repeat((total_calls, 1)).contiguous()
    search_start_batch = torch.zeros((total_calls, 1), dtype=torch.int64, device=device)
    for step in range(args.prewarm_steps):
        ops.pop_reusable_slots(slot_meta_batch[step], search_start_batch[step], blocked_slot_ids, args.count)
    _synchronize(device)
    with _make_profiler(args, trace_dir) as prof:
        for step in range(total_steps):
            call_index = args.prewarm_steps + step
            with record_function("kvca_profile.pop_reusable_slots"):
                ops.pop_reusable_slots(
                    slot_meta_batch[call_index],
                    search_start_batch[call_index],
                    blocked_slot_ids,
                    args.count,
                )
            prof.step()
    _synchronize(device)
    _print_summary(prof, sort_by=args.sort_by, row_limit=args.row_limit)
    print(f"pop trace written to: {trace_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile KVCacheAdapter on Ascend NPU")
    parser.add_argument("--target", choices=("runtime", "pop", "both"), default="runtime")
    parser.add_argument("--output-dir", default="./npu_profile")
    parser.add_argument("--wait", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--prewarm-steps", type=int, default=20)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--sort-by", default="self_npu_time_total")
    parser.add_argument("--row-limit", type=int, default=40)

    parser.add_argument("--num-actual-blocks", type=int, default=4096)
    parser.add_argument("--num-logical-blocks", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--hit-rate", type=float, default=0.5)
    parser.add_argument("--runtime-op", choices=("load", "save", "both"), default="both")

    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--blocked-count", type=int, default=0)
    parser.add_argument("--pop-scenario", choices=("dense", "tail", "aged"), default="dense")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _device()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    targets: tuple[Callable[[argparse.Namespace, torch.device], None], ...]
    if args.target == "runtime":
        targets = (profile_runtime,)
    elif args.target == "pop":
        targets = (profile_pop,)
    else:
        targets = (profile_runtime, profile_pop)
    for target in targets:
        target(args, device)


if __name__ == "__main__":
    main()
