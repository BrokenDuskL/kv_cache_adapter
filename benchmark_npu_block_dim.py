from __future__ import annotations

import argparse
import importlib
import os
import time
from dataclasses import dataclass

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import adapter as adapter_mod


@dataclass(frozen=True)
class Result:
    scenario: str
    num_actual_blocks: int
    block_dim: int
    count: int
    avg_us: float


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _pack_slot_meta(pin_counts: torch.Tensor, usage_counts: torch.Tensor) -> torch.Tensor:
    return adapter_mod._pack_slot_meta(pin_counts, usage_counts)


def _make_slot_meta(num_actual_blocks: int, count: int, scenario: str, device: torch.device) -> torch.Tensor:
    pin_counts = torch.zeros((num_actual_blocks,), dtype=adapter_mod.PIN_COUNT_DTYPE, device=device)
    usage_counts = torch.zeros((num_actual_blocks,), dtype=adapter_mod.USAGE_DTYPE, device=device)
    if scenario == "tail":
        if count > num_actual_blocks:
            raise ValueError("count must be <= num_actual_blocks for tail scenario")
        pin_counts[: num_actual_blocks - count] = 1
    elif scenario != "dense":
        raise ValueError(f"unknown scenario: {scenario}")
    return _pack_slot_meta(pin_counts, usage_counts)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type in {"npu", "privateuseone"} and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _benchmark_one(
    *,
    ops: object,
    scenario: str,
    num_actual_blocks: int,
    block_dim: int,
    count: int,
    warmup_steps: int,
    steps: int,
    device: torch.device,
) -> Result:
    os.environ["KVCA_NPU_BLOCK_DIM"] = str(block_dim)
    slot_meta = _make_slot_meta(num_actual_blocks, count, scenario, device)
    search_start = torch.zeros((1,), dtype=torch.int64, device=device)
    blocked_slot_ids = torch.empty((0,), dtype=torch.int64, device=device)

    for _ in range(warmup_steps):
        selected = ops.pop_reusable_slots(slot_meta, search_start, blocked_slot_ids, count)
        del selected
    _synchronize(device)

    start = time.perf_counter()
    for _ in range(steps):
        selected = ops.pop_reusable_slots(slot_meta, search_start, blocked_slot_ids, count)
        del selected
    _synchronize(device)
    elapsed = time.perf_counter() - start
    return Result(
        scenario=scenario,
        num_actual_blocks=num_actual_blocks,
        block_dim=block_dim,
        count=count,
        avg_us=elapsed * 1_000_000.0 / steps,
    )


def _device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    raise SystemExit("NPU is required")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark KVCacheAdapter NPU pop_reusable_slots block_dim choices")
    parser.add_argument("--num-actual-blocks", default="128,256,512,1024,2048,4096")
    parser.add_argument("--block-dims", default="1,2,4,8,16,32")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--scenarios", default="dense,tail")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _device()
    ops = importlib.import_module("kv_cache_adapter_npu_custom")
    num_actual_blocks_values = _parse_int_list(args.num_actual_blocks)
    block_dim_values = _parse_int_list(args.block_dims)
    scenarios = tuple(part.strip() for part in args.scenarios.split(",") if part.strip())

    print("scenario | num_actual_blocks | block_dim | count | avg_us")
    print("--- | --- | --- | --- | ---")
    for scenario in scenarios:
        for num_actual_blocks in num_actual_blocks_values:
            if args.count > num_actual_blocks:
                continue
            best: Result | None = None
            for block_dim in block_dim_values:
                result = _benchmark_one(
                    ops=ops,
                    scenario=scenario,
                    num_actual_blocks=num_actual_blocks,
                    block_dim=block_dim,
                    count=args.count,
                    warmup_steps=args.warmup_steps,
                    steps=args.steps,
                    device=device,
                )
                if best is None or result.avg_us < best.avg_us:
                    best = result
                print(
                    f"{result.scenario} | {result.num_actual_blocks} | {result.block_dim} | "
                    f"{result.count} | {result.avg_us:.3f}",
                    flush=True,
                )
            if best is not None:
                print(
                    f"best:{best.scenario} | {best.num_actual_blocks} | {best.block_dim} | "
                    f"{best.count} | {best.avg_us:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
