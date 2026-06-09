from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from adapter import InMemoryBlockStoreBackend, KVCacheAdapter, STATE_RESIDENT


@dataclass(frozen=True)
class BenchmarkConfig:
    num_actual_blocks: int = 256
    num_logical_blocks: int = 1024
    batch_size: int = 32
    steps: int = 200
    warmup_steps: int = 20
    block_size: int = 128
    hit_rates: tuple[float, ...] = (0.0, 0.5, 1.0)
    seed: int = 1234


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type in {"npu", "privateuseone"} and hasattr(torch, "npu"):
        torch.npu.synchronize()


def resident_unpinned_ids(adapter: KVCacheAdapter) -> torch.Tensor:
    resident_mask = (adapter._slot_state == STATE_RESIDENT) & (adapter._pin_count == 0)
    resident_ids = adapter._physical_to_logical[resident_mask]
    return resident_ids[resident_ids >= 0]


def cold_ids(all_ids: torch.Tensor, resident_ids: torch.Tensor) -> torch.Tensor:
    cold_mask = torch.ones(all_ids.shape[0], dtype=torch.bool, device=all_ids.device)
    if resident_ids.numel() > 0:
        cold_mask[resident_ids] = False
    return all_ids[cold_mask]


def sample_unique(pool: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if count == 0:
        return pool[:0]
    indices = torch.randperm(pool.numel(), generator=generator, device="cpu")[:count]
    return pool.index_select(0, indices.to(device=pool.device))


def sample_batch_ids(
    adapter: KVCacheAdapter,
    all_ids: torch.Tensor,
    batch_size: int,
    target_hit_rate: float,
    generator: torch.Generator,
) -> torch.Tensor:
    resident_ids = resident_unpinned_ids(adapter)
    cold = cold_ids(all_ids, resident_ids)
    hit_count = int(round(batch_size * target_hit_rate))
    miss_count = batch_size - hit_count
    parts = []
    if hit_count > 0:
        parts.append(sample_unique(resident_ids, hit_count, generator))
    if miss_count > 0:
        parts.append(sample_unique(cold, miss_count, generator))
    merged = torch.cat(parts, dim=0)
    permutation = torch.randperm(merged.numel(), generator=generator, device="cpu")
    return merged.index_select(0, permutation.to(device=merged.device))


def random_payloads(
    config: BenchmarkConfig,
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


def make_adapter(
    config: BenchmarkConfig,
    *,
    device: torch.device,
    prefer_native_extension: bool,
) -> KVCacheAdapter:
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
        prefer_native_extension=prefer_native_extension,
    )


def benchmark_runtime_path(
    config: BenchmarkConfig,
    *,
    device: torch.device,
) -> tuple[str, list[tuple[float, float, float]]]:
    all_ids = torch.arange(config.num_logical_blocks, dtype=torch.int64, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    adapter = make_adapter(
        config,
        device=device,
        prefer_native_extension=True,
    )
    results: list[tuple[float, float, float]] = []
    try:
        initial = all_ids[: config.num_actual_blocks]
        adapter.load(initial)
        synchronize(device)
        adapter.release(initial)
        for hit_rate in config.hit_rates:
            for _ in range(config.warmup_steps):
                load_ids = sample_batch_ids(adapter, all_ids, config.batch_size, hit_rate, generator)
                adapter.load(load_ids)
                synchronize(device)
                adapter.release(load_ids)
                save_ids = sample_batch_ids(adapter, all_ids, config.batch_size, hit_rate, generator)
                adapter.save(save_ids, random_payloads(config, generator, device=device))
                synchronize(device)

            total_load = 0.0
            total_save = 0.0
            for _ in range(config.steps):
                load_ids = sample_batch_ids(adapter, all_ids, config.batch_size, hit_rate, generator)
                synchronize(device)
                t0 = time.perf_counter()
                adapter.load(load_ids)
                synchronize(device)
                total_load += time.perf_counter() - t0
                adapter.release(load_ids)

                save_ids = sample_batch_ids(adapter, all_ids, config.batch_size, hit_rate, generator)
                save_payload = random_payloads(config, generator, device=device)
                synchronize(device)
                t0 = time.perf_counter()
                adapter.save(save_ids, save_payload)
                synchronize(device)
                total_save += time.perf_counter() - t0

            results.append(
                (
                    hit_rate,
                    total_load * 1000.0 / config.steps,
                    total_save * 1000.0 / config.steps,
                )
            )
        return adapter.runtime_path, results
    finally:
        adapter.shutdown()
        synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark KVCacheAdapter runtime paths on the current accelerator")
    parser.add_argument("--num-actual-blocks", type=int, default=256)
    parser.add_argument("--num-logical-blocks", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu")
    else:
        raise SystemExit("CUDA or NPU is required")
    args = parse_args()
    config = BenchmarkConfig(
        num_actual_blocks=args.num_actual_blocks,
        num_logical_blocks=args.num_logical_blocks,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        block_size=args.block_size,
        seed=args.seed,
    )
    runtime_path, results = benchmark_runtime_path(
        config,
        device=device,
    )

    print("runtime | hit_rate | avg_load_ms | avg_save_ms")
    print("--- | --- | --- | ---")
    for hit_rate, load_ms, save_ms in results:
        print(f"{runtime_path} | {hit_rate:.1f} | {load_ms:.3f} | {save_ms:.3f}")


if __name__ == "__main__":
    main()
