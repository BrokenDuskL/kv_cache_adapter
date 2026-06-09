from __future__ import annotations

import pytest
import torch

from kv_cache_adapter.benchmark_lmcache_backend import BenchmarkConfig, run_benchmark
from kv_cache_adapter import (
    BlockNotFoundError,
    InMemoryBlockStoreBackend,
    InsufficientCapacityError,
    KVCacheAdapter,
    LMCacheBackend,
)


def make_adapter(
    *,
    num_actual_blocks: int = 2,
    num_logical_blocks: int = 8,
    backend: InMemoryBlockStoreBackend | None = None,
) -> KVCacheAdapter:
    return KVCacheAdapter(
        num_actual_blocks=num_actual_blocks,
        num_logical_blocks=num_logical_blocks,
        actual_blocks=torch.zeros((num_actual_blocks, 2), dtype=torch.float32),
        backend=backend or InMemoryBlockStoreBackend(num_logical_blocks=num_logical_blocks),
    )


def make_payload(rows: list[list[int]], *, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32, device=device)


def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual, expected)


def assert_loaded_payloads(
    adapter: KVCacheAdapter,
    logical_block_ids: torch.Tensor,
    expected_payloads: dict[int, torch.Tensor],
) -> None:
    physical_slot_ids = adapter.load(logical_block_ids)

    for logical_block_id, physical_slot_id in zip(
        logical_block_ids.detach().cpu().tolist(),
        physical_slot_ids.detach().cpu().tolist(),
    ):
        assert_tensor_equal(
            adapter.get_actual_block(int(physical_slot_id)),
            expected_payloads[logical_block_id],
        )

    adapter.release(logical_block_ids)


def test_free_slots_are_preloaded_into_reusable_lru() -> None:
    adapter = make_adapter(num_actual_blocks=3)

    snapshot = adapter.debug_snapshot()

    assert snapshot["lru_unpinned"] == [0, 1, 2]
    assert snapshot["actual_blocks"].shape == (3, 2)
    adapter.shutdown()


def test_load_preserves_order_without_internal_reordering() -> None:
    backend = InMemoryBlockStoreBackend({
        3: make_payload([[3, 30]])[0],
        5: make_payload([[5, 50]])[0],
    })
    adapter = make_adapter(backend=backend)

    physical_ids = adapter.load(torch.tensor([3, 5], dtype=torch.int64))

    assert physical_ids.tolist() == [0, 1]
    assert backend.load_calls == [3, 5]

    adapter.release(torch.tensor([3, 5], dtype=torch.int64))
    adapter.shutdown()


def test_lru_evicts_only_unpinned_blocks() -> None:
    backend = InMemoryBlockStoreBackend({
        0: make_payload([[0, 10]])[0],
        1: make_payload([[1, 11]])[0],
        2: make_payload([[2, 12]])[0],
    })
    adapter = make_adapter(backend=backend)

    physical_ids = adapter.load(torch.tensor([0, 1], dtype=torch.int64))
    adapter.release(torch.tensor([0], dtype=torch.int64))
    slot_for_2 = adapter.load(torch.tensor([2], dtype=torch.int64))

    snapshot = adapter.debug_snapshot()
    logical_to_physical = snapshot["logical_to_physical"]

    assert physical_ids.tolist() == [0, 1]
    assert slot_for_2.tolist() == [0]
    assert logical_to_physical[1] == 1
    assert logical_to_physical[2] == 0
    assert 0 not in logical_to_physical
    assert backend.save_calls == [0]
    assert backend.operation_log[-2:] == [("save", 0), ("load", 2)]

    adapter.release(torch.tensor([1, 2], dtype=torch.int64))
    adapter.shutdown()


def test_load_raises_when_all_resident_blocks_are_pinned() -> None:
    backend = InMemoryBlockStoreBackend({
        0: make_payload([[0, 10]])[0],
        1: make_payload([[1, 11]])[0],
    })
    adapter = make_adapter(num_actual_blocks=1, backend=backend)

    adapter.load(torch.tensor([0], dtype=torch.int64))

    with pytest.raises(InsufficientCapacityError):
        adapter.load(torch.tensor([1], dtype=torch.int64))

    adapter.release(torch.tensor([0], dtype=torch.int64))
    adapter.shutdown()


def test_updated_payload_is_spilled_when_its_resident_slot_is_eventually_evicted() -> None:
    backend = InMemoryBlockStoreBackend({
        0: make_payload([[0, 10]])[0],
        1: make_payload([[1, 11]])[0],
        2: make_payload([[2, 12]])[0],
        3: make_payload([[3, 13]])[0],
        4: make_payload([[4, 14]])[0],
        5: make_payload([[5, 15]])[0],
    })
    adapter = make_adapter(backend=backend)
    updated_payload = make_payload([[100, 101]])[0]

    adapter.load(torch.tensor([0], dtype=torch.int64))
    adapter.save(torch.tensor([0], dtype=torch.int64), updated_payload.unsqueeze(0))
    adapter.release(torch.tensor([0], dtype=torch.int64))

    for logical_block_id in [1, 2, 3, 4, 5]:
        adapter.load(torch.tensor([logical_block_id], dtype=torch.int64))
        snapshot = adapter.debug_snapshot()
        if 0 not in snapshot["logical_to_physical"]:
            break
        adapter.release(torch.tensor([logical_block_id], dtype=torch.int64))
    else:
        raise AssertionError("logical block 0 never got evicted during the test")

    assert_tensor_equal(backend.snapshot()[0], updated_payload)
    assert ("save", 0) in backend.operation_log

    adapter.shutdown()


def test_save_materializes_cold_block_by_evicting_unpinned_resident() -> None:
    backend = InMemoryBlockStoreBackend({
        0: make_payload([[0, 10]])[0],
        1: make_payload([[1, 11]])[0],
        2: make_payload([[2, 12]])[0],
    })
    adapter = make_adapter(backend=backend)
    updated_payload = make_payload([[100, 101]])[0]
    inserted_payload = make_payload([[200, 201]])[0]

    adapter.load(torch.tensor([0, 1], dtype=torch.int64))
    adapter.save(torch.tensor([0], dtype=torch.int64), updated_payload.unsqueeze(0))
    adapter.release(torch.tensor([0, 1], dtype=torch.int64))

    adapter.save(torch.tensor([2], dtype=torch.int64), inserted_payload.unsqueeze(0))
    snapshot = adapter.debug_snapshot()
    logical_to_physical = snapshot["logical_to_physical"]

    assert 2 in logical_to_physical
    resident_before = {0, 1}
    evicted_residents = resident_before - set(logical_to_physical)
    assert len(evicted_residents) == 1
    evicted_logical = next(iter(evicted_residents))
    assert backend.operation_log[-1] == ("save", evicted_logical)

    inserted_slot = logical_to_physical[2]
    assert_tensor_equal(snapshot["actual_blocks"][inserted_slot], inserted_payload)

    if 0 in logical_to_physical:
        updated_slot = logical_to_physical[0]
        assert_tensor_equal(snapshot["actual_blocks"][updated_slot], updated_payload)
    else:
        assert_tensor_equal(backend.snapshot()[0], updated_payload)

    adapter.shutdown()


def test_load_missing_block_raises_backend_error() -> None:
    adapter = make_adapter(backend=InMemoryBlockStoreBackend(num_logical_blocks=8))

    with pytest.raises(BlockNotFoundError):
        adapter.load(torch.tensor([4], dtype=torch.int64))

    adapter.shutdown()


def test_load_fills_multiple_slots_with_batched_tensor_copy() -> None:
    backend = InMemoryBlockStoreBackend(
        {
            0: make_payload([[0, 10]])[0],
            1: make_payload([[1, 11]])[0],
            2: make_payload([[2, 12]])[0],
            3: make_payload([[3, 13]])[0],
        },
    )
    adapter = make_adapter(
        num_actual_blocks=4,
        backend=backend,
    )

    adapter.load(torch.tensor([0, 1, 2, 3], dtype=torch.int64))
    snapshot = adapter.debug_snapshot()

    assert torch.equal(
        snapshot["actual_blocks"],
        make_payload([[0, 10], [1, 11], [2, 12], [3, 13]]),
    )

    adapter.release(torch.tensor([0, 1, 2, 3], dtype=torch.int64))
    adapter.shutdown()


def test_save_updates_actual_blocks_with_tensor_copy() -> None:
    backend = InMemoryBlockStoreBackend(
        {
            0: make_payload([[0, 10]])[0],
            1: make_payload([[1, 11]])[0],
            2: make_payload([[2, 12]])[0],
            3: make_payload([[3, 13]])[0],
        },
    )
    adapter = make_adapter(
        num_actual_blocks=2,
        backend=backend,
    )

    adapter.load(torch.tensor([0, 1], dtype=torch.int64))
    adapter.save(torch.tensor([0, 1], dtype=torch.int64), make_payload([[100, 101], [110, 111]]))
    adapter.release(torch.tensor([0, 1], dtype=torch.int64))

    adapter.save(torch.tensor([2, 3], dtype=torch.int64), make_payload([[200, 201], [300, 301]]))
    snapshot = adapter.debug_snapshot()

    assert_tensor_equal(backend.snapshot()[0], make_payload([[100, 101]])[0])
    assert_tensor_equal(backend.snapshot()[1], make_payload([[110, 111]])[0])
    assert torch.equal(
        snapshot["actual_blocks"],
        make_payload([[200, 201], [300, 301]]),
    )

    adapter.shutdown()


def _default_test_device() -> torch.device:
    return torch.device("cpu")


@pytest.mark.parametrize("prefer_native_extension", [False, True])
def test_save_and_load_round_trip_across_evictions_uses_public_interface(
    prefer_native_extension: bool,
) -> None:
    device = _default_test_device()
    adapter = KVCacheAdapter(
        num_actual_blocks=2,
        num_logical_blocks=8,
        actual_blocks=torch.zeros((2, 2), dtype=torch.float32, device=device),
        backend=InMemoryBlockStoreBackend(num_logical_blocks=8),
        prefer_native_extension=prefer_native_extension,
    )
    expected_payloads = {
        0: make_payload([[10, 11]], device=device)[0],
        1: make_payload([[20, 21]], device=device)[0],
        2: make_payload([[30, 31]], device=device)[0],
        3: make_payload([[40, 41]], device=device)[0],
        4: make_payload([[50, 51]], device=device)[0],
    }

    adapter.save(
        torch.tensor([0, 1], dtype=torch.int64, device=device),
        torch.stack((expected_payloads[0], expected_payloads[1]), dim=0),
    )
    assert_loaded_payloads(
        adapter,
        torch.tensor([0, 1], dtype=torch.int64, device=device),
        expected_payloads,
    )

    adapter.save(
        torch.tensor([2], dtype=torch.int64, device=device),
        expected_payloads[2].unsqueeze(0),
    )
    assert_loaded_payloads(
        adapter,
        torch.tensor([1, 2], dtype=torch.int64, device=device),
        expected_payloads,
    )

    adapter.save(
        torch.tensor([3], dtype=torch.int64, device=device),
        expected_payloads[3].unsqueeze(0),
    )
    assert_loaded_payloads(
        adapter,
        torch.tensor([3, 0], dtype=torch.int64, device=device),
        expected_payloads,
    )

    expected_payloads[2] = make_payload([[300, 301]], device=device)[0]
    adapter.save(
        torch.tensor([2], dtype=torch.int64, device=device),
        expected_payloads[2].unsqueeze(0),
    )
    assert_loaded_payloads(
        adapter,
        torch.tensor([2, 1], dtype=torch.int64, device=device),
        expected_payloads,
    )

    adapter.save(
        torch.tensor([4], dtype=torch.int64, device=device),
        expected_payloads[4].unsqueeze(0),
    )
    assert_loaded_payloads(
        adapter,
        torch.tensor([4, 3], dtype=torch.int64, device=device),
        expected_payloads,
    )

    adapter.shutdown()


def test_lmcache_backend_loads_into_target_slots_in_place() -> None:
    pytest.importorskip("lmcache")
    backend = LMCacheBackend(
        block_shape=(2,),
        block_dtype=torch.float32,
        model_name="kv-cache-adapter-test",
        max_local_cpu_size_gb=0.01,
    )
    target = torch.full((3, 2), -1.0, dtype=torch.float32)

    backend.save_blocks(
        torch.tensor([1, 3], dtype=torch.int64),
        make_payload([[10, 11], [30, 31]]),
    )
    backend.load_blocks(
        torch.tensor([3, 1], dtype=torch.int64),
        target,
        torch.tensor([0, 2], dtype=torch.int64),
    )

    assert torch.equal(
        target,
        torch.tensor([[30.0, 31.0], [-1.0, -1.0], [10.0, 11.0]], dtype=torch.float32),
    )

    backend.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_lmcache_backend_accepts_cuda_tensors() -> None:
    pytest.importorskip("lmcache")
    backend = LMCacheBackend(
        block_shape=(2,),
        block_dtype=torch.float32,
        model_name="kv-cache-adapter-test-cuda",
        max_local_cpu_size_gb=0.01,
    )
    target = torch.full((3, 2), -1.0, dtype=torch.float32, device="cuda")

    backend.save_blocks(
        torch.tensor([1, 3], dtype=torch.int64, device="cuda"),
        make_payload([[10, 11], [30, 31]], device="cuda"),
    )
    backend.load_blocks(
        torch.tensor([3, 1], dtype=torch.int64, device="cuda"),
        target,
        torch.tensor([0, 2], dtype=torch.int64, device="cuda"),
    )
    torch.cuda.synchronize()

    assert torch.equal(
        target,
        torch.tensor(
            [[30.0, 31.0], [-1.0, -1.0], [10.0, 11.0]],
            dtype=torch.float32,
            device="cuda",
        ),
    )

    backend.shutdown()


def test_lmcache_benchmark_smoke() -> None:
    pytest.importorskip("lmcache")
    results = run_benchmark(
        BenchmarkConfig(
            num_actual_blocks=8,
            num_logical_blocks=32,
            batch_size=4,
            steps=4,
            warmup_steps=1,
            block_shape=(64,),
            hit_rates=(0.0, 0.5, 1.0),
            max_local_cpu_size_gb=0.01,
        )
    )

    assert len(results) == 3
    for result in results:
        assert 0.0 <= result.achieved_load_hit_rate <= 1.0
        assert 0.0 <= result.achieved_save_hit_rate <= 1.0
        assert result.avg_load_ms >= 0.0
        assert result.avg_save_ms >= 0.0
        assert result.total_seconds >= 0.0
