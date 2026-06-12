from __future__ import annotations

import importlib

import pytest
import torch

import adapter as adapter_mod


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is required",
)


def _npu_ops():
    return importlib.import_module("kv_cache_adapter_npu_custom")


def _pack_slot_meta(pin_count: int, usage_count: int) -> int:
    return int(
        (pin_count & adapter_mod._PIN_COUNT_MASK)
        | ((usage_count & adapter_mod._USAGE_COUNT_MASK) << adapter_mod._USAGE_COUNT_SHIFT)
    )


def _id_tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device="npu")


def _meta_tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=adapter_mod.SLOT_META_DTYPE, device="npu")


def _expect_tensor(actual: torch.Tensor, expected: list[int], *, dtype: torch.dtype | None = None) -> None:
    torch.npu.synchronize()
    expected_tensor = torch.tensor(expected, dtype=dtype or actual.dtype)
    assert torch.equal(actual.cpu(), expected_tensor)


def test_npu_kernel_inspect_save_requests() -> None:
    ops = _npu_ops()
    logical_to_physical = _id_tensor([-1, 0, 1, -1])
    slot_meta = _meta_tensor([
        _pack_slot_meta(0, 2),
        _pack_slot_meta(1, adapter_mod.USAGE_COUNT_MAX),
    ])
    logical_block_ids = _id_tensor([0, 1, 2, 3])

    current_physical, existing_mask, final_usage_counts = ops.inspect_save_requests(
        logical_to_physical,
        slot_meta,
        logical_block_ids,
    )

    _expect_tensor(current_physical, [-1, 0, 1, -1], dtype=torch.int64)
    _expect_tensor(existing_mask, [0, 1, 1, 0], dtype=torch.uint8)
    _expect_tensor(
        final_usage_counts,
        [1, 3, adapter_mod.USAGE_COUNT_MAX, 1],
        dtype=adapter_mod.SLOT_META_DTYPE,
    )


def test_npu_kernel_inspect_load_requests() -> None:
    ops = _npu_ops()
    logical_to_physical = _id_tensor([-1, 0, 1, -1])
    slot_meta = _meta_tensor([
        _pack_slot_meta(2, 2),
        _pack_slot_meta(1, adapter_mod.USAGE_COUNT_MAX),
    ])
    logical_block_ids = _id_tensor([0, 1, 2, 3])

    current_physical, resident_mask, updated_pin_counts, updated_usage_counts = ops.inspect_load_requests(
        logical_to_physical,
        slot_meta,
        logical_block_ids,
    )

    _expect_tensor(current_physical, [-1, 0, 1, -1], dtype=torch.int64)
    _expect_tensor(resident_mask, [0, 1, 1, 0], dtype=torch.uint8)
    _expect_tensor(updated_pin_counts, [0, 3, 2, 0], dtype=torch.int64)
    _expect_tensor(
        updated_usage_counts,
        [0, 3, adapter_mod.USAGE_COUNT_MAX, 0],
        dtype=adapter_mod.SLOT_META_DTYPE,
    )


def test_npu_kernel_pop_mark_blocked_slots() -> None:
    ops = _npu_ops()

    blocked_mask = ops._debug_mark_blocked_slots(_id_tensor([1, 3]), 5)

    _expect_tensor(blocked_mask, [0, 1, 0, 1, 0], dtype=torch.uint8)


def test_npu_kernel_pop_count_threshold_slots() -> None:
    ops = _npu_ops()
    slot_meta = _meta_tensor([
        _pack_slot_meta(0, 0),
        _pack_slot_meta(0, 1),
        _pack_slot_meta(0, 0),
        _pack_slot_meta(1, 0),
    ])
    blocked_mask = torch.tensor([0, 0, 1, 0], dtype=torch.uint8, device="npu")

    local_count_workspace = ops._debug_count_threshold_slots(
        slot_meta,
        blocked_mask,
        _id_tensor([0]),
        _id_tensor([0, -1]),
        0,
    )

    _expect_tensor(local_count_workspace, [1], dtype=torch.int64)


def test_npu_kernel_pop_plan_threshold_slots() -> None:
    ops = _npu_ops()
    selection_state = _id_tensor([0, -1])

    local_offset_workspace, local_emit_workspace, updated_selection_state = ops._debug_plan_threshold_slots(
        _id_tensor([3]),
        selection_state,
        2,
        0,
    )

    _expect_tensor(local_offset_workspace, [0], dtype=torch.int64)
    _expect_tensor(local_emit_workspace, [2], dtype=torch.int64)
    _expect_tensor(updated_selection_state, [2, 0], dtype=torch.int64)
    _expect_tensor(selection_state, [2, 0], dtype=torch.int64)


def test_npu_kernel_pop_collect_threshold_slots() -> None:
    ops = _npu_ops()
    slot_meta = _meta_tensor([
        _pack_slot_meta(0, 1),
        _pack_slot_meta(0, 0),
        _pack_slot_meta(1, 0),
        _pack_slot_meta(0, 0),
    ])
    blocked_mask = torch.tensor([0, 1, 0, 0], dtype=torch.uint8, device="npu")
    selected_slot_ids = _id_tensor([-1, -1])

    result = ops._debug_collect_threshold_slots(
        slot_meta,
        blocked_mask,
        _id_tensor([0]),
        _id_tensor([1, -1]),
        _id_tensor([0]),
        _id_tensor([1]),
        selected_slot_ids,
        0,
    )

    _expect_tensor(result, [3, -1], dtype=torch.int64)
    _expect_tensor(selected_slot_ids, [3, -1], dtype=torch.int64)


def test_npu_kernel_pop_age_usage() -> None:
    ops = _npu_ops()
    slot_meta = _meta_tensor([
        _pack_slot_meta(0, 3),
        _pack_slot_meta(0, 1),
        _pack_slot_meta(1, 0),
    ])

    ops._debug_age_usage(slot_meta, _id_tensor([2, 1]))

    _expect_tensor(
        slot_meta,
        [_pack_slot_meta(0, 2), _pack_slot_meta(0, 0), _pack_slot_meta(1, 0)],
        dtype=adapter_mod.SLOT_META_DTYPE,
    )


def test_npu_kernel_pop_finalize_selected_slots() -> None:
    ops = _npu_ops()
    search_start = _id_tensor([0])
    selected_slot_ids = _id_tensor([3, 0])

    ops._debug_finalize_selected_slots(_id_tensor([2, 1]), search_start, selected_slot_ids, 4, 2)

    _expect_tensor(search_start, [1], dtype=torch.int64)
    _expect_tensor(selected_slot_ids, [3, 0], dtype=torch.int64)


def test_npu_kernel_pop_finalize_marks_unfilled_slots() -> None:
    ops = _npu_ops()
    search_start = _id_tensor([0])
    selected_slot_ids = _id_tensor([3, 99])

    ops._debug_finalize_selected_slots(_id_tensor([1, 0]), search_start, selected_slot_ids, 4, 2)

    _expect_tensor(search_start, [0], dtype=torch.int64)
    _expect_tensor(selected_slot_ids, [3, -1], dtype=torch.int64)


def test_npu_kernel_pop_reusable_slots_threshold_scan_and_age() -> None:
    ops = _npu_ops()
    slot_meta = _meta_tensor([
        _pack_slot_meta(0, 1),
        _pack_slot_meta(0, 0),
        _pack_slot_meta(1, 0),
        _pack_slot_meta(0, 0),
    ])
    search_start = _id_tensor([0])
    blocked_slot_ids = _id_tensor([1])

    selected_slot_ids = ops.pop_reusable_slots(slot_meta, search_start, blocked_slot_ids, 2)

    _expect_tensor(selected_slot_ids, [3, 0], dtype=torch.int64)
    _expect_tensor(search_start, [1], dtype=torch.int64)
    _expect_tensor(
        slot_meta,
        [
            _pack_slot_meta(0, 0),
            _pack_slot_meta(0, 0),
            _pack_slot_meta(1, 0),
            _pack_slot_meta(0, 0),
        ],
        dtype=adapter_mod.SLOT_META_DTYPE,
    )


def test_npu_kernel_commit_save_metadata() -> None:
    ops = _npu_ops()
    logical_to_physical = _id_tensor([0, 1, 2, -1, -1])
    physical_to_logical = _id_tensor([0, 1, 2])
    slot_meta = _meta_tensor([
        _pack_slot_meta(0, 1),
        _pack_slot_meta(0, 2),
        _pack_slot_meta(0, 3),
    ])

    ops.commit_save_metadata(
        logical_to_physical,
        physical_to_logical,
        slot_meta,
        _id_tensor([1, 2]),
        _id_tensor([3, 4]),
        _id_tensor([1, 2]),
        _meta_tensor([1, 0]),
        _meta_tensor([1, 2]),
    )

    _expect_tensor(logical_to_physical, [0, -1, -1, 1, 2], dtype=torch.int64)
    _expect_tensor(physical_to_logical, [0, 3, 4], dtype=torch.int64)
    _expect_tensor(
        slot_meta,
        [_pack_slot_meta(0, 1), _pack_slot_meta(1, 1), _pack_slot_meta(0, 2)],
        dtype=adapter_mod.SLOT_META_DTYPE,
    )


def test_npu_kernel_commit_load_metadata() -> None:
    ops = _npu_ops()
    logical_to_physical = _id_tensor([0, 1, 2, -1])
    physical_to_logical = _id_tensor([0, 1, 2])
    slot_meta = _meta_tensor([
        _pack_slot_meta(1, 5),
        _pack_slot_meta(1, 4),
        _pack_slot_meta(1, 3),
    ])

    ops.commit_load_metadata(
        logical_to_physical,
        physical_to_logical,
        slot_meta,
        _id_tensor([1]),
        _id_tensor([3]),
        _id_tensor([1]),
        _id_tensor([0]),
        _id_tensor([2]),
        _meta_tensor([6]),
        _meta_tensor([2]),
    )

    _expect_tensor(logical_to_physical, [0, -1, 2, 1], dtype=torch.int64)
    _expect_tensor(physical_to_logical, [0, 3, 2], dtype=torch.int64)
    _expect_tensor(
        slot_meta,
        [_pack_slot_meta(2, 6), _pack_slot_meta(1, 2), _pack_slot_meta(1, 3)],
        dtype=adapter_mod.SLOT_META_DTYPE,
    )


def test_npu_kernel_release_metadata() -> None:
    ops = _npu_ops()
    logical_to_physical = _id_tensor([0, 1, -1])
    slot_meta = _meta_tensor([
        _pack_slot_meta(2, 5),
        _pack_slot_meta(1, 3),
    ])

    ops.release_metadata(logical_to_physical, slot_meta, _id_tensor([0, 1]))

    _expect_tensor(
        slot_meta,
        [_pack_slot_meta(1, 5), _pack_slot_meta(0, 3)],
        dtype=adapter_mod.SLOT_META_DTYPE,
    )
