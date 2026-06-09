from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

if __package__ in (None, ""):
    from adapter import KVCacheAdapter, LMCacheBackend, STATE_RESIDENT
else:
    from .adapter import KVCacheAdapter, LMCacheBackend, STATE_RESIDENT


@dataclass(frozen=True)
class BenchmarkConfig:
    num_actual_blocks: int = 256
    num_logical_blocks: int = 1024
    batch_size: int = 32
    steps: int = 200
    warmup_steps: int = 20
    block_shape: tuple[int, ...] = (2048,)
    dtype: torch.dtype = torch.float32
    hit_rates: tuple[float, ...] = (0.0, 0.5, 0.9, 1.0)
    seed: int = 1234
    max_local_cpu_size_gb: float = 0.25


@dataclass(frozen=True)
class BenchmarkResult:
    target_hit_rate: float
    achieved_load_hit_rate: float
    achieved_save_hit_rate: float
    avg_load_ms: float
    avg_save_ms: float
    total_seconds: float
    load_ops_per_sec: float
    save_ops_per_sec: float


@dataclass(frozen=True)
class _IterationStats:
    load_seconds: float
    save_seconds: float
    load_hits: int
    save_hits: int


def run_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    _validate_config(config)
    return [_run_single_hit_rate(config, hit_rate) for hit_rate in config.hit_rates]


def format_results(results: list[BenchmarkResult]) -> str:
    lines = [
        "target_hit_rate | achieved_load_hit_rate | achieved_save_hit_rate | avg_load_ms | avg_save_ms | load_ops/s | save_ops/s | total_s",
        "--- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for result in results:
        lines.append(
            f"{result.target_hit_rate:0.2f} | "
            f"{result.achieved_load_hit_rate:0.4f} | "
            f"{result.achieved_save_hit_rate:0.4f} | "
            f"{result.avg_load_ms:0.3f} | "
            f"{result.avg_save_ms:0.3f} | "
            f"{result.load_ops_per_sec:0.1f} | "
            f"{result.save_ops_per_sec:0.1f} | "
            f"{result.total_seconds:0.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    config = BenchmarkConfig(
        num_actual_blocks=args.num_actual_blocks,
        num_logical_blocks=args.num_logical_blocks,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        block_shape=(args.block_size,),
        dtype=_parse_dtype(args.dtype),
        hit_rates=tuple(args.hit_rates),
        seed=args.seed,
        max_local_cpu_size_gb=args.max_local_cpu_size_gb,
    )
    results = run_benchmark(config)
    device = _benchmark_device()
    print(
        "config -> "
        f"actual={config.num_actual_blocks}, logical={config.num_logical_blocks}, "
        f"batch={config.batch_size}, steps={config.steps}, warmup={config.warmup_steps}, "
        f"block_shape={config.block_shape}, dtype={config.dtype}, device={device}"
    )
    print(format_results(results))


def _run_single_hit_rate(config: BenchmarkConfig, target_hit_rate: float) -> BenchmarkResult:
    device = _benchmark_device()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + int(target_hit_rate * 1000))
    all_logical_ids = torch.arange(config.num_logical_blocks, dtype=torch.int64, device=device)

    backend = LMCacheBackend(
        block_shape=config.block_shape,
        block_dtype=config.dtype,
        model_name=f"kv-cache-adapter-bench-{int(target_hit_rate * 1000)}",
        max_local_cpu_size_gb=config.max_local_cpu_size_gb,
    )
    adapter = KVCacheAdapter(
        num_actual_blocks=config.num_actual_blocks,
        num_logical_blocks=config.num_logical_blocks,
        actual_blocks=torch.zeros(
            (config.num_actual_blocks, *config.block_shape),
            dtype=config.dtype,
            device=device,
        ),
        backend=backend,
    )

    try:
        backend.save_blocks(
            all_logical_ids,
            _make_initial_payloads(config, generator, device=device),
        )
        initial_resident = all_logical_ids[: config.num_actual_blocks]
        initial_physical_ids = adapter.load(initial_resident)
        _require_physical_ids_on_device(initial_physical_ids, device)
        adapter.release(initial_resident)

        for _ in range(config.warmup_steps):
            _run_iteration(adapter, all_logical_ids, config, target_hit_rate, generator, measure=False)

        total_load_seconds = 0.0
        total_save_seconds = 0.0
        total_load_hits = 0
        total_save_hits = 0
        total_load_blocks = 0
        total_save_blocks = 0
        _synchronize_device(device)
        total_start = time.perf_counter()
        for _ in range(config.steps):
            stats = _run_iteration(adapter, all_logical_ids, config, target_hit_rate, generator, measure=True)
            total_load_seconds += stats.load_seconds
            total_save_seconds += stats.save_seconds
            total_load_hits += stats.load_hits
            total_save_hits += stats.save_hits
            total_load_blocks += config.batch_size
            total_save_blocks += config.batch_size
        _synchronize_device(device)
        total_seconds = time.perf_counter() - total_start
    finally:
        adapter.shutdown()

    return BenchmarkResult(
        target_hit_rate=target_hit_rate,
        achieved_load_hit_rate=total_load_hits / total_load_blocks,
        achieved_save_hit_rate=total_save_hits / total_save_blocks,
        avg_load_ms=(total_load_seconds / config.steps) * 1000.0,
        avg_save_ms=(total_save_seconds / config.steps) * 1000.0,
        total_seconds=total_seconds,
        load_ops_per_sec=config.steps / total_load_seconds if total_load_seconds > 0 else float("inf"),
        save_ops_per_sec=config.steps / total_save_seconds if total_save_seconds > 0 else float("inf"),
    )


def _run_iteration(
    adapter: KVCacheAdapter,
    all_logical_ids: torch.Tensor,
    config: BenchmarkConfig,
    target_hit_rate: float,
    generator: torch.Generator,
    *,
    measure: bool,
) -> _IterationStats:
    load_ids, load_hits = _sample_batch_ids(adapter, all_logical_ids, config.batch_size, target_hit_rate, generator)
    _synchronize_device(load_ids.device)
    load_start = time.perf_counter()
    loaded_physical_ids = adapter.load(load_ids)
    _synchronize_device(load_ids.device)
    load_seconds = time.perf_counter() - load_start if measure else 0.0
    _require_physical_ids_on_device(loaded_physical_ids, load_ids.device)
    adapter.release(load_ids)

    save_ids, save_hits = _sample_batch_ids(adapter, all_logical_ids, config.batch_size, target_hit_rate, generator)
    save_payloads = _make_random_payloads(config, generator, device=all_logical_ids.device)
    _synchronize_device(save_ids.device)
    save_start = time.perf_counter()
    adapter.save(save_ids, save_payloads)
    _synchronize_device(save_ids.device)
    save_seconds = time.perf_counter() - save_start if measure else 0.0

    return _IterationStats(
        load_seconds=load_seconds,
        save_seconds=save_seconds,
        load_hits=load_hits,
        save_hits=save_hits,
    )


def _sample_batch_ids(
    adapter: KVCacheAdapter,
    all_logical_ids: torch.Tensor,
    batch_size: int,
    target_hit_rate: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    resident_ids = _resident_unpinned_ids(adapter)
    cold_ids = _cold_ids(all_logical_ids, resident_ids)

    hit_count = int(round(batch_size * target_hit_rate))
    miss_count = batch_size - hit_count
    if hit_count > resident_ids.numel():
        raise ValueError("target hit rate requires more resident ids than available")
    if miss_count > cold_ids.numel():
        raise ValueError("target hit rate requires more cold ids than available")

    parts = []
    if hit_count > 0:
        parts.append(_sample_unique_ids(resident_ids, hit_count, generator))
    if miss_count > 0:
        parts.append(_sample_unique_ids(cold_ids, miss_count, generator))

    merged = torch.cat(parts, dim=0)
    permutation = torch.randperm(merged.numel(), generator=generator, device="cpu")
    return merged.index_select(0, permutation.to(device=merged.device)), hit_count


def _resident_unpinned_ids(adapter: KVCacheAdapter) -> torch.Tensor:
    resident_mask = (adapter._slot_state == STATE_RESIDENT) & (adapter._pin_count == 0)
    resident_ids = adapter._physical_to_logical[resident_mask]
    return resident_ids[resident_ids >= 0]


def _cold_ids(all_logical_ids: torch.Tensor, resident_ids: torch.Tensor) -> torch.Tensor:
    cold_mask = torch.ones(all_logical_ids.shape[0], dtype=torch.bool, device=all_logical_ids.device)
    if resident_ids.numel() > 0:
        cold_mask[resident_ids] = False
    return all_logical_ids[cold_mask]


def _sample_unique_ids(pool: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if count == 0:
        return pool[:0]
    indices = torch.randperm(pool.numel(), generator=generator, device="cpu")[:count]
    return pool.index_select(0, indices.to(device=pool.device))


def _make_initial_payloads(
    config: BenchmarkConfig,
    generator: torch.Generator,
    *,
    device: torch.device,
) -> torch.Tensor:
    total_elements = config.num_logical_blocks
    for dim in config.block_shape:
        total_elements *= dim
    payloads = torch.arange(total_elements, dtype=torch.float32).reshape(
        config.num_logical_blocks,
        *config.block_shape,
    )
    permutation = torch.randperm(config.num_logical_blocks, generator=generator, device="cpu")
    return payloads.index_select(0, permutation).to(device=device, dtype=config.dtype)


def _make_random_payloads(
    config: BenchmarkConfig,
    generator: torch.Generator,
    *,
    device: torch.device,
) -> torch.Tensor:
    payloads = torch.randn(
        (config.batch_size, *config.block_shape),
        generator=generator,
        dtype=torch.float32,
    )
    return payloads.to(device=device, dtype=config.dtype)


def _validate_config(config: BenchmarkConfig) -> None:
    if config.num_actual_blocks <= 0:
        raise ValueError("num_actual_blocks must be positive")
    if config.num_logical_blocks <= config.num_actual_blocks:
        raise ValueError("num_logical_blocks must be greater than num_actual_blocks")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.batch_size > config.num_actual_blocks:
        raise ValueError("batch_size must be <= num_actual_blocks so 100% hit batches are possible")
    if config.batch_size > (config.num_logical_blocks - config.num_actual_blocks):
        raise ValueError("batch_size must be <= num_logical_blocks - num_actual_blocks so 0% hit batches are possible")
    for hit_rate in config.hit_rates:
        if hit_rate < 0.0 or hit_rate > 1.0:
            raise ValueError("hit rates must be inside [0, 1]")
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")


def _parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {value}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark KVCacheAdapter with LMCacheBackend")
    parser.add_argument("--num-actual-blocks", type=int, default=256)
    parser.add_argument("--num-logical-blocks", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-local-cpu-size-gb", type=float, default=0.25)
    parser.add_argument("--hit-rates", nargs="+", type=float, default=[0.0, 0.5, 0.9, 1.0])
    return parser.parse_args()


def _benchmark_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type in {"npu", "privateuseone"} and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _require_physical_ids_on_device(physical_ids: torch.Tensor, expected_device: torch.device) -> None:
    if physical_ids.device.type != expected_device.type:
        raise RuntimeError(
            "adapter.load() returned physical ids on the wrong device type: "
            f"expected {expected_device}, got {physical_ids.device}"
        )
    if expected_device.index is not None and physical_ids.device.index != expected_device.index:
        raise RuntimeError(
            "adapter.load() returned physical ids on the wrong device: "
            f"expected {expected_device}, got {physical_ids.device}"
        )


if __name__ == "__main__":
    main()
