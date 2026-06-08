from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import torch

ID_DTYPE = torch.int64
USAGE_DTYPE = torch.uint8
USAGE_COUNT_MAX = 127

STATE_FREE = 0
STATE_RESERVED = 1
STATE_LOADING = 2
STATE_RESIDENT = 3
STATE_NAMES = {
    STATE_FREE: "free",
    STATE_RESERVED: "reserved",
    STATE_LOADING: "loading",
    STATE_RESIDENT: "resident",
}


class KVCacheAdapterError(RuntimeError):
    pass


class BlockNotFoundError(KVCacheAdapterError):
    pass


class InsufficientCapacityError(KVCacheAdapterError):
    pass


class BlockStoreBackend(Protocol):
    def save_blocks(self, logical_block_ids: torch.Tensor, block_data: torch.Tensor) -> None:
        ...

    def load_blocks(
        self,
        logical_block_ids: torch.Tensor,
        actual_blocks: torch.Tensor,
        physical_slot_ids: torch.Tensor,
    ) -> None:
        ...


@dataclass(frozen=True)
class _EvictionPlan:
    logical_block_ids: torch.Tensor
    payloads: torch.Tensor


class KVCacheAdapter:
    """Maps n logical ids onto m resident physical slots in `actual_blocks`.

    Contract:
    - `actual_blocks` is the real `(m, ...)` resident tensor
    - `logical_block_ids` must be `torch.int64` on the same device as `actual_blocks`
    - `block_data` must be on the same device/dtype/shape layout as `actual_blocks`
    - callers serialize access; this adapter is not thread-safe
    """

    def __init__(
        self,
        num_actual_blocks: int,
        num_logical_blocks: int,
        actual_blocks: torch.Tensor,
        backend: BlockStoreBackend,
        *,
        max_workers: int | None = None,
        prefer_cuda_extension: bool = True,
    ) -> None:
        del max_workers
        if not isinstance(actual_blocks, torch.Tensor):
            raise TypeError("actual_blocks must be torch.Tensor")
        if actual_blocks.ndim < 1:
            raise ValueError("actual_blocks must have a leading block dimension")
        if num_actual_blocks <= 0:
            raise ValueError("num_actual_blocks must be positive")
        if num_logical_blocks <= 0:
            raise ValueError("num_logical_blocks must be positive")
        if num_actual_blocks > num_logical_blocks:
            raise ValueError("num_actual_blocks must be <= num_logical_blocks")
        if actual_blocks.shape[0] != num_actual_blocks:
            raise ValueError("actual_blocks leading dimension must equal num_actual_blocks")

        self.num_actual_blocks = int(num_actual_blocks)
        self.num_logical_blocks = int(num_logical_blocks)
        self.actual_blocks = actual_blocks
        self.backend = backend

        device = actual_blocks.device
        self._logical_to_physical = torch.full(
            (num_logical_blocks,),
            -1,
            dtype=ID_DTYPE,
            device=device,
        )
        self._physical_to_logical = torch.full(
            (num_actual_blocks,),
            -1,
            dtype=ID_DTYPE,
            device=device,
        )
        self._slot_state = torch.full(
            (num_actual_blocks,),
            STATE_FREE,
            dtype=torch.int64,
            device=device,
        )
        self._pin_count = torch.zeros((num_actual_blocks,), dtype=torch.int64, device=device)
        self._reusable_mask = torch.ones((num_actual_blocks,), dtype=torch.bool, device=device)
        self._usage_count = torch.zeros((num_actual_blocks,), dtype=USAGE_DTYPE, device=device)
        self._search_start = torch.zeros((1,), dtype=ID_DTYPE, device=device)
        self._cuda_ext = self._detect_cuda_extension(prefer_cuda_extension=prefer_cuda_extension)
        self.runtime_path = "cuda_ext_meta" if self._cuda_ext is not None else "strict"

    def save(self, logical_block_ids: torch.Tensor, block_data: torch.Tensor) -> None:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        _require_same_device(logical_block_ids, expected_device=self.actual_blocks.device, name="logical_block_ids")
        self._validate_logical_block_ids(logical_block_ids)
        _require_block_data_tensor(
            block_data,
            expected_blocks=logical_block_ids.shape[0],
            name="block_data",
        )
        _require_same_device(block_data, expected_device=self.actual_blocks.device, name="block_data")
        _require_same_dtype(block_data, reference=self.actual_blocks, name="block_data")
        _require_trailing_shape(block_data, self.actual_blocks, name="block_data")

        if logical_block_ids.numel() == 0:
            return

        current_physical, existing_mask, _ = self._inspect_save_requests(logical_block_ids)
        existing_physical = current_physical[existing_mask]
        missing_ids = logical_block_ids[~existing_mask]
        allocated_physical = self._pop_reusable_slots(missing_ids.shape[0], blocked_slot_ids=existing_physical)

        selected_physical = current_physical.clone()
        if missing_ids.numel() > 0:
            selected_physical[~existing_mask] = allocated_physical

        final_pin_counts = self._pin_count.index_select(0, selected_physical)
        final_usage_counts = torch.ones_like(logical_block_ids, dtype=USAGE_DTYPE)
        if existing_physical.numel() > 0:
            final_usage_counts[existing_mask] = _saturating_increment_usage(
                self._usage_count.index_select(0, existing_physical),
            )
        eviction_plan = self._build_eviction_plan(logical_block_ids, selected_physical)

        if eviction_plan.logical_block_ids.numel() > 0:
            self.backend.save_blocks(eviction_plan.logical_block_ids, eviction_plan.payloads)

        self._copy_into_actual_blocks(selected_physical, block_data)
        self._commit_save_metadata(
            evicted_logical_block_ids=eviction_plan.logical_block_ids,
            logical_block_ids=logical_block_ids,
            physical_slot_ids=selected_physical,
            final_pin_counts=final_pin_counts,
            final_usage_counts=final_usage_counts,
        )

    def load(self, logical_block_ids: torch.Tensor) -> torch.Tensor:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        _require_same_device(logical_block_ids, expected_device=self.actual_blocks.device, name="logical_block_ids")
        self._validate_logical_block_ids(logical_block_ids)
        if logical_block_ids.numel() == 0:
            return torch.empty_like(logical_block_ids)

        unique_ids = self._unique_preserve_order(logical_block_ids)
        current_physical, resident_mask, updated_pin_counts, _ = self._inspect_load_requests(
            unique_ids,
        )
        hit_slot_ids = current_physical[resident_mask]
        miss_logical_ids = unique_ids[~resident_mask]
        miss_physical_slot_ids = self._pop_reusable_slots(miss_logical_ids.shape[0], blocked_slot_ids=hit_slot_ids)
        eviction_plan = self._build_eviction_plan(miss_logical_ids, miss_physical_slot_ids)

        if eviction_plan.logical_block_ids.numel() > 0:
            self.backend.save_blocks(eviction_plan.logical_block_ids, eviction_plan.payloads)
        if miss_logical_ids.numel() > 0:
            self.backend.load_blocks(
                miss_logical_ids,
                self.actual_blocks,
                miss_physical_slot_ids,
            )

        hit_usage_counts = _saturating_increment_usage(
            self._usage_count.index_select(0, hit_slot_ids),
        )
        miss_usage_counts = torch.ones(
            (miss_physical_slot_ids.shape[0],),
            dtype=USAGE_DTYPE,
            device=self.actual_blocks.device,
        )
        touched_slot_ids = torch.cat((hit_slot_ids, miss_physical_slot_ids), dim=0)
        touched_usage_counts = torch.cat((hit_usage_counts, miss_usage_counts), dim=0)
        self._commit_load_metadata(
            evicted_logical_block_ids=eviction_plan.logical_block_ids,
            miss_logical_block_ids=miss_logical_ids,
            miss_physical_slot_ids=miss_physical_slot_ids,
            hit_slot_ids=hit_slot_ids,
            hit_pin_counts=updated_pin_counts[resident_mask],
            touched_slot_ids=touched_slot_ids,
            touched_usage_counts=touched_usage_counts,
        )
        return self._logical_to_physical.index_select(0, logical_block_ids)

    def release(self, logical_block_ids: torch.Tensor) -> None:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        _require_same_device(logical_block_ids, expected_device=self.actual_blocks.device, name="logical_block_ids")
        self._validate_logical_block_ids(logical_block_ids)
        unique_ids = self._unique_preserve_order(logical_block_ids)
        if unique_ids.numel() == 0:
            return

        physical_slot_ids = self._logical_to_physical.index_select(0, unique_ids)
        if torch.any(physical_slot_ids < 0).item():
            raise KeyError("logical block is not resident")

        states = self._slot_state.index_select(0, physical_slot_ids)
        if torch.any(states != STATE_RESIDENT).item():
            raise KVCacheAdapterError("logical block is busy")

        pin_counts = self._pin_count.index_select(0, physical_slot_ids)
        if torch.any(pin_counts <= 0).item():
            raise KVCacheAdapterError("logical block is not pinned")

        updated_pin_counts = pin_counts - 1
        self._pin_count.index_put_((physical_slot_ids,), updated_pin_counts)
        zero_pin_slots = physical_slot_ids[updated_pin_counts == 0]
        if zero_pin_slots.numel() > 0:
            self._reusable_mask.index_fill_(0, zero_pin_slots, True)

    def get_actual_block(self, physical_slot_id: int) -> torch.Tensor:
        self._validate_physical_slot_id(physical_slot_id)
        return self.actual_blocks[physical_slot_id].detach().clone()

    def debug_snapshot(self) -> dict[str, object]:
        reusable_slots = self._ordered_reusable_slots()
        logical_to_physical = self._logical_to_physical.detach().to(device="cpu")
        slot_state = self._slot_state.detach().to(device="cpu")
        return {
            "runtime_path": self.runtime_path,
            "logical_to_physical": {
                logical_block_id: int(physical_slot_id)
                for logical_block_id, physical_slot_id in enumerate(logical_to_physical.tolist())
                if physical_slot_id >= 0
            },
            "lru_unpinned": reusable_slots.detach().to(device="cpu").tolist(),
            "slot_state": [STATE_NAMES[int(state)] for state in slot_state.tolist()],
            "pin_count": self._pin_count.detach().cpu().clone(),
            "actual_blocks": self.actual_blocks.detach().cpu().clone(),
        }

    def shutdown(self) -> None:
        backend_shutdown = getattr(self.backend, "shutdown", None)
        if callable(backend_shutdown):
            backend_shutdown()
            return

        backend_close = getattr(self.backend, "close", None)
        if callable(backend_close):
            backend_close()

    def _detect_cuda_extension(self, *, prefer_cuda_extension: bool) -> Any | None:
        if not prefer_cuda_extension or self.actual_blocks.device.type != "cuda":
            return None
        return _load_cuda_extension_module()

    def _unique_preserve_order(self, values: torch.Tensor) -> torch.Tensor:
        _require_id_tensor(values, name="values")
        if values.numel() <= 1:
            return values
        positions = torch.arange(values.numel(), device=values.device, dtype=ID_DTYPE)
        sentinel = values.new_full((self.num_logical_blocks,), values.numel())
        sentinel.scatter_reduce_(0, values, positions, reduce="amin", include_self=True)
        present_ids = torch.nonzero(sentinel < values.numel(), as_tuple=False).reshape(-1)
        first_positions = sentinel.index_select(0, present_ids)
        order = torch.argsort(first_positions, stable=True)
        return present_ids.index_select(0, order)

    def _inspect_load_requests(
        self,
        logical_block_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cuda_ext is not None:
            return tuple(
                self._cuda_ext.inspect_load_requests(
                    self._logical_to_physical.contiguous(),
                    self._slot_state.contiguous(),
                    self._pin_count.contiguous(),
                    self._usage_count.contiguous(),
                    logical_block_ids.contiguous(),
                ),
            )  # type: ignore[return-value]

        current_physical = self._logical_to_physical.index_select(0, logical_block_ids)
        resident_mask = current_physical >= 0
        updated_pin_counts = torch.zeros_like(logical_block_ids)
        updated_usage_counts = torch.zeros_like(logical_block_ids, dtype=USAGE_DTYPE)

        if resident_mask.numel() == 0 or not torch.any(resident_mask).item():
            return current_physical, resident_mask, updated_pin_counts, updated_usage_counts

        hit_slot_ids = current_physical[resident_mask]
        hit_states = self._slot_state.index_select(0, hit_slot_ids)
        if torch.any(hit_states != STATE_RESIDENT).item():
            raise KVCacheAdapterError("logical block is busy")
        hit_pin_counts = self._pin_count.index_select(0, hit_slot_ids) + 1
        hit_usage_counts = _saturating_increment_usage(self._usage_count.index_select(0, hit_slot_ids))
        updated_pin_counts[resident_mask] = hit_pin_counts
        updated_usage_counts[resident_mask] = hit_usage_counts
        return current_physical, resident_mask, updated_pin_counts, updated_usage_counts

    def _inspect_save_requests(
        self,
        logical_block_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cuda_ext is not None:
            return tuple(
                self._cuda_ext.inspect_save_requests(
                    self._logical_to_physical.contiguous(),
                    self._slot_state.contiguous(),
                    self._usage_count.contiguous(),
                    logical_block_ids.contiguous(),
                ),
            )  # type: ignore[return-value]

        current_physical = self._logical_to_physical.index_select(0, logical_block_ids)
        existing_mask = current_physical >= 0
        final_usage_counts = torch.ones_like(logical_block_ids, dtype=USAGE_DTYPE)

        if existing_mask.numel() == 0 or not torch.any(existing_mask).item():
            return current_physical, existing_mask, final_usage_counts

        existing_physical = current_physical[existing_mask]
        existing_states = self._slot_state.index_select(0, existing_physical)
        if torch.any(existing_states != STATE_RESIDENT).item():
            raise KVCacheAdapterError("logical block is busy")
        final_usage_counts[existing_mask] = _saturating_increment_usage(
            self._usage_count.index_select(0, existing_physical),
        )
        return current_physical, existing_mask, final_usage_counts

    def _pop_reusable_slots(self, count: int, *, blocked_slot_ids: torch.Tensor) -> torch.Tensor:
        if count == 0:
            return self._logical_to_physical[:0]

        blocked_slot_ids = blocked_slot_ids.contiguous()
        if self._cuda_ext is not None:
            return self._cuda_ext.pop_reusable_slots(
                self._usage_count.contiguous(),
                self._reusable_mask.contiguous(),
                self._search_start.contiguous(),
                blocked_slot_ids,
                count,
            )

        available_mask = self._reusable_mask.clone()
        if blocked_slot_ids.numel() > 0:
            available_mask.index_fill_(0, blocked_slot_ids, False)
        available_slot_ids = torch.nonzero(available_mask, as_tuple=False).reshape(-1)
        if available_slot_ids.numel() < count:
            raise InsufficientCapacityError("No reusable actual block is available; all resident blocks are pinned")

        available_usage = self._usage_count.index_select(0, available_slot_ids).to(dtype=torch.int64)
        usage_hist = torch.bincount(available_usage, minlength=USAGE_COUNT_MAX + 1)
        usage_prefix = torch.cumsum(usage_hist, dim=0)
        threshold = int(torch.nonzero(usage_prefix >= count, as_tuple=False)[0, 0].item())

        scan_order = (
            self._search_start[0] + torch.arange(self.num_actual_blocks, device=self.actual_blocks.device, dtype=ID_DTYPE)
        ) % self.num_actual_blocks
        eligible_mask = available_mask.index_select(0, scan_order) & (
            self._usage_count.index_select(0, scan_order).to(dtype=torch.int64) <= threshold
        )
        selected_slot_ids = scan_order[eligible_mask][:count]
        if selected_slot_ids.numel() != count:
            raise InsufficientCapacityError("No reusable actual block is available; all resident blocks are pinned")

        if threshold > 0:
            self._usage_count.copy_(
                torch.clamp(self._usage_count.to(dtype=torch.int16) - threshold, min=0).to(dtype=USAGE_DTYPE),
            )
        self._search_start[0] = (selected_slot_ids[-1] + 1) % self.num_actual_blocks
        return selected_slot_ids

    def _build_eviction_plan(
        self,
        logical_block_ids: torch.Tensor,
        physical_slot_ids: torch.Tensor,
    ) -> _EvictionPlan:
        if physical_slot_ids.numel() == 0:
            return _EvictionPlan(logical_block_ids[:0], self.actual_blocks[:0])

        previous_logical_ids = self._physical_to_logical.index_select(0, physical_slot_ids)
        previous_states = self._slot_state.index_select(0, physical_slot_ids)
        eviction_mask = (
            (previous_states == STATE_RESIDENT)
            & (previous_logical_ids >= 0)
            & (previous_logical_ids != logical_block_ids)
        )
        if not torch.any(eviction_mask).item():
            return _EvictionPlan(logical_block_ids[:0], self.actual_blocks[:0])

        eviction_logical_ids = previous_logical_ids[eviction_mask]
        eviction_slot_ids = physical_slot_ids[eviction_mask]
        eviction_payloads = self.actual_blocks.index_select(0, eviction_slot_ids)
        return _EvictionPlan(eviction_logical_ids, eviction_payloads)

    def _commit_load_metadata(
        self,
        *,
        evicted_logical_block_ids: torch.Tensor,
        miss_logical_block_ids: torch.Tensor,
        miss_physical_slot_ids: torch.Tensor,
        hit_slot_ids: torch.Tensor,
        hit_pin_counts: torch.Tensor,
        touched_slot_ids: torch.Tensor,
        touched_usage_counts: torch.Tensor,
    ) -> None:
        empty_ids = self._logical_to_physical[:0]
        empty_usage = self._usage_count[:0]
        if self._cuda_ext is not None:
            self._cuda_ext.commit_load_metadata(
                self._logical_to_physical.contiguous(),
                self._physical_to_logical.contiguous(),
                self._slot_state.contiguous(),
                self._pin_count.contiguous(),
                self._reusable_mask.contiguous(),
                self._usage_count.contiguous(),
                evicted_logical_block_ids.contiguous(),
                empty_ids.contiguous(),
                empty_usage.contiguous(),
                miss_logical_block_ids.contiguous(),
                miss_physical_slot_ids.contiguous(),
                hit_slot_ids.contiguous(),
                hit_pin_counts.contiguous(),
                touched_slot_ids.contiguous(),
                touched_usage_counts.contiguous(),
            )
            return

        if evicted_logical_block_ids.numel() > 0:
            self._logical_to_physical.index_fill_(0, evicted_logical_block_ids, -1)
        if miss_logical_block_ids.numel() > 0:
            self._logical_to_physical.index_put_((miss_logical_block_ids,), miss_physical_slot_ids)
            self._physical_to_logical.index_put_((miss_physical_slot_ids,), miss_logical_block_ids)
            self._slot_state.index_fill_(0, miss_physical_slot_ids, STATE_RESIDENT)
            self._pin_count.index_fill_(0, miss_physical_slot_ids, 1)
        if hit_slot_ids.numel() > 0:
            self._pin_count.index_put_((hit_slot_ids,), hit_pin_counts)
        if touched_slot_ids.numel() > 0:
            self._reusable_mask.index_fill_(0, touched_slot_ids, False)
            self._usage_count.index_put_((touched_slot_ids,), touched_usage_counts)

    def _commit_save_metadata(
        self,
        *,
        evicted_logical_block_ids: torch.Tensor,
        logical_block_ids: torch.Tensor,
        physical_slot_ids: torch.Tensor,
        final_pin_counts: torch.Tensor,
        final_usage_counts: torch.Tensor,
    ) -> None:
        empty_ids = self._logical_to_physical[:0]
        empty_usage = self._usage_count[:0]
        if self._cuda_ext is not None:
            self._cuda_ext.commit_save_metadata(
                self._logical_to_physical.contiguous(),
                self._physical_to_logical.contiguous(),
                self._slot_state.contiguous(),
                self._pin_count.contiguous(),
                self._reusable_mask.contiguous(),
                self._usage_count.contiguous(),
                evicted_logical_block_ids.contiguous(),
                empty_ids.contiguous(),
                empty_usage.contiguous(),
                logical_block_ids.contiguous(),
                physical_slot_ids.contiguous(),
                final_pin_counts.contiguous(),
                final_usage_counts.contiguous(),
            )
            return

        if evicted_logical_block_ids.numel() > 0:
            self._logical_to_physical.index_fill_(0, evicted_logical_block_ids, -1)
        self._logical_to_physical.index_put_((logical_block_ids,), physical_slot_ids)
        self._physical_to_logical.index_put_((physical_slot_ids,), logical_block_ids)
        self._slot_state.index_fill_(0, physical_slot_ids, STATE_RESIDENT)
        self._pin_count.index_put_((physical_slot_ids,), final_pin_counts)
        self._usage_count.index_put_((physical_slot_ids,), final_usage_counts)
        self._reusable_mask.index_put_((physical_slot_ids,), final_pin_counts == 0)

    def _copy_into_actual_blocks(self, physical_slot_ids: torch.Tensor, block_data: torch.Tensor) -> None:
        if physical_slot_ids.numel() == 0:
            return
        self.actual_blocks.index_copy_(0, physical_slot_ids, block_data)

    def _ordered_reusable_slots(self) -> torch.Tensor:
        if self.num_actual_blocks == 0:
            return self._physical_to_logical[:0]
        scan_order = (
            self._search_start[0] + torch.arange(self.num_actual_blocks, device=self.actual_blocks.device, dtype=ID_DTYPE)
        ) % self.num_actual_blocks
        return scan_order[self._reusable_mask.index_select(0, scan_order)]

    def _validate_logical_block_ids(self, logical_block_ids: torch.Tensor) -> None:
        if logical_block_ids.numel() == 0:
            return
        if torch.any(logical_block_ids < 0).item() or torch.any(logical_block_ids >= self.num_logical_blocks).item():
            raise ValueError(f"logical block ids must be inside [0, {self.num_logical_blocks})")

    def _validate_physical_slot_id(self, physical_slot_id: int) -> None:
        if physical_slot_id < 0 or physical_slot_id >= self.num_actual_blocks:
            raise ValueError(f"physical slot id {physical_slot_id} is outside [0, {self.num_actual_blocks})")


class InMemoryBlockStoreBackend:
    def __init__(
        self,
        initial_data: dict[int, torch.Tensor] | torch.Tensor | None = None,
        *,
        num_logical_blocks: int | None = None,
        present_mask: torch.Tensor | None = None,
        save_delay_s: float = 0.0,
        load_delay_s: float = 0.0,
    ) -> None:
        self._storage: torch.Tensor | None = None
        if isinstance(initial_data, torch.Tensor):
            if initial_data.ndim < 1:
                raise ValueError("initial_data tensor must have a leading block dimension")
            self._storage = initial_data.detach().clone()
            inferred_blocks = int(self._storage.shape[0])
            if num_logical_blocks is None:
                num_logical_blocks = inferred_blocks
            elif num_logical_blocks != inferred_blocks:
                raise ValueError("num_logical_blocks must match initial_data.shape[0]")
            self._present_mask = (
                present_mask.detach().clone().to(device="cpu")
                if present_mask is not None
                else torch.ones((num_logical_blocks,), dtype=torch.bool)
            )
        else:
            if num_logical_blocks is None:
                if initial_data:
                    num_logical_blocks = max(initial_data) + 1
                else:
                    raise ValueError("num_logical_blocks is required when initial_data is empty")
            self._present_mask = (
                present_mask.detach().clone().to(device="cpu")
                if present_mask is not None
                else torch.zeros((num_logical_blocks,), dtype=torch.bool)
            )
            if initial_data:
                first_payload = next(iter(initial_data.values()))
                self._storage = first_payload.new_zeros((num_logical_blocks, *first_payload.shape))
                initial_ids = torch.tensor(list(initial_data.keys()), dtype=ID_DTYPE, device=first_payload.device)
                initial_payloads = torch.stack([initial_data[key] for key in initial_data.keys()], dim=0)
                self._storage.index_copy_(0, initial_ids, initial_payloads)
                self._present_mask.index_fill_(0, initial_ids.cpu(), True)

        self.num_logical_blocks = int(num_logical_blocks)
        self._save_delay_s = save_delay_s
        self._load_delay_s = load_delay_s
        self.save_calls: list[int] = []
        self.load_calls: list[int] = []
        self.operation_log: list[tuple[str, int]] = []
        self.max_concurrent_saves = 0
        self.max_concurrent_loads = 0
        self._concurrent_saves = 0
        self._concurrent_loads = 0

    def save_blocks(self, logical_block_ids: torch.Tensor, block_data: torch.Tensor) -> None:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        _require_block_data_tensor(block_data, expected_blocks=logical_block_ids.shape[0], name="block_data")
        self._enter_save()
        try:
            if self._save_delay_s > 0:
                time.sleep(self._save_delay_s)
            if self._storage is None:
                self._storage = block_data.new_zeros((self.num_logical_blocks, *block_data.shape[1:]))
            else:
                _require_trailing_shape(block_data, self._storage, name="block_data")
                _require_same_dtype(block_data, reference=self._storage, name="block_data")
            self._storage.index_copy_(
                0,
                logical_block_ids.to(device=self._storage.device),
                block_data.to(device=self._storage.device, dtype=self._storage.dtype),
            )
            self._present_mask.index_fill_(0, logical_block_ids.to(device="cpu"), True)
            logical_ids_list = logical_block_ids.detach().to(device="cpu").tolist()
            self.save_calls.extend(logical_ids_list)
            self.operation_log.extend(("save", logical_block_id) for logical_block_id in logical_ids_list)
        finally:
            self._leave_save()

    def load_blocks(
        self,
        logical_block_ids: torch.Tensor,
        actual_blocks: torch.Tensor,
        physical_slot_ids: torch.Tensor,
    ) -> None:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        _require_id_tensor(physical_slot_ids, name="physical_slot_ids")
        if logical_block_ids.shape[0] != physical_slot_ids.shape[0]:
            raise ValueError("physical_slot_ids leading dimension must match logical_block_ids")
        self._enter_load()
        try:
            if self._load_delay_s > 0:
                time.sleep(self._load_delay_s)
            if self._storage is None:
                raise BlockNotFoundError("backend storage is empty")
            hit_mask = self._present_mask.index_select(0, logical_block_ids.to(device="cpu"))
            if not torch.all(hit_mask).item():
                raise BlockNotFoundError("some logical blocks are not in backend")
            _require_trailing_shape(actual_blocks, self._storage, name="actual_blocks")
            loaded_payloads = self._storage.index_select(
                0,
                logical_block_ids.to(device=self._storage.device),
            )
            actual_blocks.index_copy_(
                0,
                physical_slot_ids.to(device=actual_blocks.device),
                loaded_payloads.to(device=actual_blocks.device, dtype=actual_blocks.dtype),
            )
            logical_ids_list = logical_block_ids.detach().to(device="cpu").tolist()
            self.load_calls.extend(logical_ids_list)
            self.operation_log.extend(("load", logical_block_id) for logical_block_id in logical_ids_list)
        finally:
            self._leave_load()

    def snapshot(self) -> dict[int, torch.Tensor]:
        if self._storage is None:
            return {}
        present_ids = torch.nonzero(self._present_mask, as_tuple=False).reshape(-1)
        present_payloads = self._storage.index_select(0, present_ids.to(device=self._storage.device))
        return {
            logical_block_id: payload.detach().clone().cpu()
            for logical_block_id, payload in zip(present_ids.tolist(), present_payloads.unbind(0))
        }

    def _enter_save(self) -> None:
        self._concurrent_saves += 1
        self.max_concurrent_saves = max(self.max_concurrent_saves, self._concurrent_saves)

    def _leave_save(self) -> None:
        self._concurrent_saves -= 1

    def _enter_load(self) -> None:
        self._concurrent_loads += 1
        self.max_concurrent_loads = max(self.max_concurrent_loads, self._concurrent_loads)

    def _leave_load(self) -> None:
        self._concurrent_loads -= 1

    def shutdown(self) -> None:
        return None


class LMCacheBackend:
    def __init__(
        self,
        *,
        block_shape: tuple[int, ...] | torch.Size,
        block_dtype: torch.dtype,
        model_name: str = "kv_cache_adapter",
        world_size: int = 1,
        worker_id: int = 0,
        max_local_cpu_size_gb: float = 1.0,
        lmcache_instance_id: str | None = None,
    ) -> None:
        from lmcache.utils import CacheEngineKey
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

        self._CacheEngineKey = CacheEngineKey
        self._MemoryFormat = MemoryFormat
        self._MemoryObjMetadata = MemoryObjMetadata
        self._TensorMemoryObj = TensorMemoryObj
        self._block_shape = torch.Size(block_shape)
        self._block_dtype = block_dtype
        self._model_name = model_name
        self._world_size = world_size
        self._worker_id = worker_id
        self._config = LMCacheEngineConfig(
            local_cpu=True,
            max_local_cpu_size=max_local_cpu_size_gb,
            lmcache_instance_id=lmcache_instance_id,
        )
        self._backend = LocalCPUBackend(
            config=self._config,
            memory_allocator=_NoopMemoryAllocator(),
        )

    def save_blocks(self, logical_block_ids: torch.Tensor, block_data: torch.Tensor) -> None:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        _require_block_data_tensor(block_data, expected_blocks=logical_block_ids.shape[0], name="block_data")
        self._require_block_layout(block_data, name="block_data")
        if logical_block_ids.numel() == 0:
            return

        logical_block_ids_cpu = logical_block_ids.to(device="cpu")
        payloads_cpu = block_data.detach().to(device="cpu").contiguous()
        keys = self._make_keys(logical_block_ids_cpu)
        self._backend.batched_remove(keys)
        self._backend.batched_submit_put_task(
            keys,
            [self._make_memory_obj(payload) for payload in payloads_cpu.unbind(0)],
        )

    def load_blocks(
        self,
        logical_block_ids: torch.Tensor,
        actual_blocks: torch.Tensor,
        physical_slot_ids: torch.Tensor,
    ) -> None:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        _require_id_tensor(physical_slot_ids, name="physical_slot_ids")
        if logical_block_ids.shape[0] != physical_slot_ids.shape[0]:
            raise ValueError("physical_slot_ids leading dimension must match logical_block_ids")
        self._require_block_layout(actual_blocks, name="actual_blocks")
        if logical_block_ids.numel() == 0:
            return

        logical_block_ids_cpu = logical_block_ids.to(device="cpu")
        memory_objs = self._backend.batched_get_blocking(self._make_keys(logical_block_ids_cpu))
        if any(memory_obj is None for memory_obj in memory_objs):
            raise BlockNotFoundError("some logical blocks are not in LMCache")

        try:
            loaded_payloads = torch.stack(
                [memory_obj.tensor for memory_obj in memory_objs if memory_obj is not None],
                dim=0,
            )
            actual_blocks.index_copy_(
                0,
                physical_slot_ids.to(device=actual_blocks.device),
                loaded_payloads.to(device=actual_blocks.device, dtype=actual_blocks.dtype),
            )
        finally:
            for memory_obj in memory_objs:
                if memory_obj is not None:
                    memory_obj.ref_count_down()

    def shutdown(self) -> None:
        self._backend.clear()
        self._backend.close()

    def _make_keys(self, logical_block_ids: torch.Tensor) -> list[object]:
        return [
            self._CacheEngineKey(
                model_name=self._model_name,
                world_size=self._world_size,
                worker_id=self._worker_id,
                chunk_hash=int(logical_block_id),
                dtype=self._block_dtype,
            )
            for logical_block_id in logical_block_ids.tolist()
        ]

    def _make_memory_obj(self, payload: torch.Tensor) -> object:
        payload = payload.contiguous()
        raw_data = payload.view(torch.uint8).reshape(-1)
        return self._TensorMemoryObj(
            raw_data=raw_data,
            metadata=self._MemoryObjMetadata(
                shape=payload.shape,
                dtype=payload.dtype,
                address=raw_data.data_ptr(),
                phy_size=raw_data.numel(),
                ref_count=1,
                pin_count=0,
                fmt=self._MemoryFormat.UNDEFINED,
                shapes=[payload.shape],
                dtypes=[payload.dtype],
            ),
            parent_allocator=None,
        )

    def _require_block_layout(self, block_data: torch.Tensor, *, name: str) -> None:
        if block_data.dtype != self._block_dtype:
            raise TypeError(f"{name} must have dtype {self._block_dtype}")
        if tuple(block_data.shape[1:]) != tuple(self._block_shape):
            raise ValueError(f"{name} must have trailing shape {tuple(self._block_shape)}")


class _NoopMemoryAllocator:
    def allocate(self, *args: object, **kwargs: object) -> None:
        return None

    def batched_allocate(self, *args: object, **kwargs: object) -> None:
        return None

    def free(self, memory_obj: object, allocator_type: str | None = None) -> None:
        del memory_obj, allocator_type
        return None

    def batched_free(
        self,
        memory_objs: list[object],
        allocator_type: str | None = None,
        update_stats: bool = True,
    ) -> None:
        del memory_objs, allocator_type, update_stats
        return None

    def close(self) -> None:
        return None


@lru_cache(maxsize=1)
def _load_cuda_extension_module() -> Any | None:
    try:
        return importlib.import_module("kv_cache_adapter_cuda")
    except Exception:
        return None


def _saturating_increment_usage(values: torch.Tensor) -> torch.Tensor:
    incremented = values.to(dtype=torch.int16) + 1
    return torch.clamp(incremented, max=USAGE_COUNT_MAX).to(dtype=USAGE_DTYPE)


def _require_id_tensor(values: torch.Tensor, *, name: str) -> None:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if values.dtype != ID_DTYPE:
        raise TypeError(f"{name} must have dtype torch.int64")
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape (k,)")


def _require_same_device(values: torch.Tensor, *, expected_device: torch.device, name: str) -> None:
    if values.device != expected_device:
        raise ValueError(f"{name} must be on device {expected_device}, got {values.device}")


def _require_same_dtype(block_data: torch.Tensor, *, reference: torch.Tensor, name: str) -> None:
    if block_data.dtype != reference.dtype:
        raise TypeError(f"{name} must have dtype {reference.dtype}")


def _require_block_data_tensor(block_data: torch.Tensor, *, expected_blocks: int, name: str) -> None:
    if not isinstance(block_data, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if block_data.ndim < 1:
        raise ValueError(f"{name} must have a leading block dimension")
    if block_data.shape[0] != expected_blocks:
        raise ValueError(f"{name} leading dimension must match logical_block_ids length")


def _require_trailing_shape(block_data: torch.Tensor, reference: torch.Tensor, *, name: str) -> None:
    if block_data.ndim != reference.ndim or block_data.shape[1:] != reference.shape[1:]:
        raise ValueError(f"{name} must have trailing shape {tuple(reference.shape[1:])}")
