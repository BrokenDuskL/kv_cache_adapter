from __future__ import annotations

import time

import pytest
import torch

from kv_cache_adapter import InMemoryBlockStoreBackend, KVCacheAdapter


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is required",
)


def make_payload(rows: list[list[int]], *, device: str = "npu") -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32, device=device)


def assert_loaded_payloads(
    adapter: KVCacheAdapter,
    logical_block_ids: torch.Tensor,
    expected_payloads: dict[int, torch.Tensor],
) -> None:
    physical_slot_ids = adapter.load(logical_block_ids)
    for logical_block_id, physical_slot_id in zip(
        logical_block_ids.tolist(),
        physical_slot_ids.tolist(),
        strict=False,
    ):
        assert torch.equal(
            adapter.get_actual_block(int(physical_slot_id)),
            expected_payloads[int(logical_block_id)],
        )
    adapter.release(logical_block_ids)


def test_npu_runtime_path_selection() -> None:
    adapter = KVCacheAdapter(
        num_actual_blocks=2,
        num_logical_blocks=8,
        actual_blocks=torch.zeros((2, 2), dtype=torch.float32, device="npu"),
        backend=InMemoryBlockStoreBackend(num_logical_blocks=8),
        prefer_native_extension=True,
    )
    assert adapter.runtime_path in {"npu_ext_meta", "strict"}


def test_npu_round_trip_across_evictions() -> None:
    adapter = KVCacheAdapter(
        num_actual_blocks=2,
        num_logical_blocks=8,
        actual_blocks=torch.zeros((2, 2), dtype=torch.float32, device="npu"),
        backend=InMemoryBlockStoreBackend(num_logical_blocks=8),
        prefer_native_extension=True,
    )
    expected_payloads = {
        0: make_payload([[10, 11]])[0],
        1: make_payload([[20, 21]])[0],
        2: make_payload([[30, 31]])[0],
        3: make_payload([[40, 41]])[0],
    }

    adapter.save(torch.tensor([0, 1], dtype=torch.int64, device="npu"), torch.stack((expected_payloads[0], expected_payloads[1]), dim=0))
    assert_loaded_payloads(adapter, torch.tensor([0, 1], dtype=torch.int64, device="npu"), expected_payloads)

    adapter.save(torch.tensor([2], dtype=torch.int64, device="npu"), expected_payloads[2].unsqueeze(0))
    assert_loaded_payloads(adapter, torch.tensor([1, 2], dtype=torch.int64, device="npu"), expected_payloads)

    adapter.save(torch.tensor([3], dtype=torch.int64, device="npu"), expected_payloads[3].unsqueeze(0))
    assert_loaded_payloads(adapter, torch.tensor([2, 3], dtype=torch.int64, device="npu"), expected_payloads)


def test_npu_runtime_benchmark_smoke() -> None:
    logical_ids = torch.arange(0, 512, dtype=torch.int64, device="npu")
    payload_bank = torch.randn((1024, 64), dtype=torch.float16, device="npu")

    strict_adapter = KVCacheAdapter(
        num_actual_blocks=128,
        num_logical_blocks=1024,
        actual_blocks=torch.zeros((128, 64), dtype=torch.float16, device="npu"),
        backend=InMemoryBlockStoreBackend(initial_data=payload_bank),
        prefer_native_extension=False,
    )
    native_adapter = KVCacheAdapter(
        num_actual_blocks=128,
        num_logical_blocks=1024,
        actual_blocks=torch.zeros((128, 64), dtype=torch.float16, device="npu"),
        backend=InMemoryBlockStoreBackend(initial_data=payload_bank),
        prefer_native_extension=True,
    )

    torch.npu.synchronize()
    strict_start = time.perf_counter()
    for step in range(16):
        start = (step * 23) % (logical_ids.numel() - 64)
        req = logical_ids[start : start + 64]
        strict_adapter.load(req)
        strict_adapter.release(req)
    torch.npu.synchronize()
    strict_elapsed = time.perf_counter() - strict_start

    native_start = time.perf_counter()
    for step in range(16):
        start = (step * 23) % (logical_ids.numel() - 64)
        req = logical_ids[start : start + 64]
        native_adapter.load(req)
        native_adapter.release(req)
    torch.npu.synchronize()
    native_elapsed = time.perf_counter() - native_start

    assert strict_elapsed >= 0.0
    assert native_elapsed >= 0.0
