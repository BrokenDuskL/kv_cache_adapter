from __future__ import annotations

import pytest
import torch

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
    max_workers: int = 4,
) -> KVCacheAdapter:
    return KVCacheAdapter(
        num_actual_blocks=num_actual_blocks,
        num_logical_blocks=num_logical_blocks,
        actual_blocks=torch.zeros((num_actual_blocks, 2), dtype=torch.float32),
        backend=backend or InMemoryBlockStoreBackend(num_logical_blocks=num_logical_blocks),
        max_workers=max_workers,
    )


def make_payload(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual, expected)


def test_free_slots_are_preloaded_into_reusable_lru() -> None:
    adapter = make_adapter(num_actual_blocks=3)

    snapshot = adapter.debug_snapshot()

    assert snapshot["lru_unpinned"] == [0, 1, 2]
    assert snapshot["actual_blocks"].shape == (3, 2)
    adapter.shutdown()


def test_load_preserves_order_and_deduplicates_backend_fetches() -> None:
    backend = InMemoryBlockStoreBackend({
        3: make_payload([[3, 30]])[0],
        5: make_payload([[5, 50]])[0],
    })
    adapter = make_adapter(backend=backend)

    physical_ids = adapter.load(torch.tensor([3, 5, 3], dtype=torch.int64))

    assert physical_ids.tolist() == [0, 1, 0]
    assert sorted(backend.load_calls) == [3, 5]

    adapter.release(torch.tensor([3, 5, 3], dtype=torch.int64))
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


def test_save_spills_updated_block_before_later_load() -> None:
    backend = InMemoryBlockStoreBackend({
        0: make_payload([[0, 10]])[0],
        1: make_payload([[1, 11]])[0],
        2: make_payload([[2, 12]])[0],
    })
    adapter = make_adapter(backend=backend)

    adapter.load(torch.tensor([0], dtype=torch.int64))
    adapter.save(torch.tensor([0], dtype=torch.int64), make_payload([[100, 101]]))
    adapter.release(torch.tensor([0], dtype=torch.int64))

    adapter.load(torch.tensor([1], dtype=torch.int64))
    adapter.release(torch.tensor([1], dtype=torch.int64))
    adapter.load(torch.tensor([2], dtype=torch.int64))

    assert_tensor_equal(backend.snapshot()[0], make_payload([[100, 101]])[0])
    assert backend.operation_log[-2:] == [("save", 0), ("load", 2)]

    adapter.release(torch.tensor([2], dtype=torch.int64))
    adapter.shutdown()


def test_save_materializes_cold_block_by_evicting_lru() -> None:
    backend = InMemoryBlockStoreBackend({
        0: make_payload([[0, 10]])[0],
        1: make_payload([[1, 11]])[0],
        2: make_payload([[2, 12]])[0],
    })
    adapter = make_adapter(backend=backend)

    adapter.load(torch.tensor([0, 1], dtype=torch.int64))
    adapter.save(torch.tensor([0], dtype=torch.int64), make_payload([[100, 101]]))
    adapter.release(torch.tensor([0, 1], dtype=torch.int64))

    adapter.save(torch.tensor([2], dtype=torch.int64), make_payload([[200, 201]]))
    snapshot = adapter.debug_snapshot()
    logical_to_physical = snapshot["logical_to_physical"]

    assert 0 not in logical_to_physical
    assert 1 in logical_to_physical
    assert 2 in logical_to_physical
    assert_tensor_equal(backend.snapshot()[0], make_payload([[100, 101]])[0])
    assert_tensor_equal(snapshot["actual_blocks"][0], make_payload([[200, 201]])[0])
    assert backend.operation_log[-1] == ("save", 0)

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
        max_workers=4,
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
        max_workers=2,
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
