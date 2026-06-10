from __future__ import annotations

import ctypes
import importlib
import importlib.machinery
import importlib.util
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import torch

ID_DTYPE = torch.int64
_SLOT_META_BITS = int(os.getenv("KVCA_SLOT_META_BITS", "8"))
if _SLOT_META_BITS == 8:
    SLOT_META_DTYPE = torch.uint8
    PIN_COUNT_BITS = 4
    USAGE_COUNT_BITS = 4
elif _SLOT_META_BITS == 16:
    SLOT_META_DTYPE = torch.uint16
    PIN_COUNT_BITS = 8
    USAGE_COUNT_BITS = 8
else:
    raise ValueError("KVCA_SLOT_META_BITS must be 8 or 16")

USAGE_DTYPE = SLOT_META_DTYPE
PIN_COUNT_DTYPE = SLOT_META_DTYPE
PIN_COUNT_MAX = (1 << PIN_COUNT_BITS) - 1
USAGE_COUNT_MAX = (1 << USAGE_COUNT_BITS) - 1
_PIN_COUNT_MASK = PIN_COUNT_MAX
_USAGE_COUNT_MASK = USAGE_COUNT_MAX
_USAGE_COUNT_SHIFT = PIN_COUNT_BITS

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


@dataclass(frozen=True)
class _MergedKVBlockLayout:
    block_shape: torch.Size
    block_size: int
    hidden_dim: int
    token_major: bool

    @classmethod
    def detect(cls, block_shape: tuple[int, ...] | torch.Size) -> _MergedKVBlockLayout | None:
        shape = torch.Size(block_shape)
        if len(shape) != 4:
            return None
        if shape[1] == 2:
            block_size = int(shape[0])
            hidden_dims = shape[2:]
            token_major = True
        elif shape[0] == 2:
            block_size = int(shape[1])
            hidden_dims = shape[2:]
            token_major = False
        else:
            return None
        hidden_dim = 1
        for dim in hidden_dims:
            hidden_dim *= int(dim)
        return cls(
            block_shape=shape,
            block_size=block_size,
            hidden_dim=hidden_dim,
            token_major=token_major,
        )

    @property
    def transfer_shape(self) -> torch.Size:
        if self.token_major:
            return torch.Size((self.block_size, 2, self.hidden_dim))
        return torch.Size((2, self.block_size, self.hidden_dim))

    def flatten_block_batch(self, block_batch: torch.Tensor) -> torch.Tensor:
        return block_batch.contiguous().view((block_batch.shape[0],) + tuple(self.transfer_shape))

    def restore_block_tensor(self, stored_tensor: torch.Tensor) -> torch.Tensor:
        return stored_tensor.view(self.block_shape)


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
        prefer_native_extension: bool = True,
        prefer_cuda_extension: bool | None = None,
    ) -> None:
        if prefer_cuda_extension is not None:
            prefer_native_extension = prefer_cuda_extension
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
        self._platform = device.type
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
        self._slot_meta = torch.zeros((num_actual_blocks,), dtype=SLOT_META_DTYPE, device=device)
        self._search_start = torch.zeros((1,), dtype=ID_DTYPE, device=device)
        self._native_ext, self.runtime_path = self._detect_native_extension(
            prefer_native_extension=prefer_native_extension)

    @property
    def _pin_count(self) -> torch.Tensor:
        return _slot_meta_pin_counts(self._slot_meta)

    @property
    def _usage_count(self) -> torch.Tensor:
        return _slot_meta_usage_counts(self._slot_meta)

    @property
    def _reusable_mask(self) -> torch.Tensor:
        return self._pin_count == 0

    @property
    def _slot_state(self) -> torch.Tensor:
        return torch.where(
            self._physical_to_logical >= 0,
            torch.full_like(self._physical_to_logical, STATE_RESIDENT),
            torch.full_like(self._physical_to_logical, STATE_FREE),
        )

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

        current_physical, resident_mask, updated_pin_counts, _ = self._inspect_load_requests(
            logical_block_ids,
        )
        hit_slot_ids = current_physical[resident_mask]
        miss_logical_ids = logical_block_ids[~resident_mask]
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
        self._commit_load_metadata(
            evicted_logical_block_ids=eviction_plan.logical_block_ids,
            miss_logical_block_ids=miss_logical_ids,
            miss_physical_slot_ids=miss_physical_slot_ids,
            hit_slot_ids=hit_slot_ids,
            hit_pin_counts=updated_pin_counts[resident_mask],
            hit_usage_counts=hit_usage_counts,
            miss_usage_counts=miss_usage_counts,
        )
        return self._logical_to_physical.index_select(0, logical_block_ids)

    def release(self, logical_block_ids: torch.Tensor) -> None:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        _require_same_device(logical_block_ids, expected_device=self.actual_blocks.device, name="logical_block_ids")
        self._validate_logical_block_ids(logical_block_ids)
        if logical_block_ids.numel() == 0:
            return

        if self._native_ext is not None and hasattr(self._native_ext, "release_metadata"):
            if self._platform == "npu":
                self._native_ext.release_metadata(
                    self._logical_to_physical.contiguous(),
                    self._slot_meta.contiguous(),
                    logical_block_ids.contiguous(),
                )
            else:
                self._native_ext.release_metadata(
                    self._logical_to_physical.contiguous(),
                    self._pin_count.to(dtype=ID_DTYPE).contiguous(),
                    self._reusable_mask.contiguous(),
                    logical_block_ids.contiguous(),
                )
            return

        physical_slot_ids = self._logical_to_physical.index_select(0, logical_block_ids)
        pin_counts = self._pin_count.index_select(0, physical_slot_ids)
        updated_pin_counts = pin_counts - 1
        usage_counts = self._usage_count.index_select(0, physical_slot_ids)
        self._slot_meta.index_put_((physical_slot_ids,), _pack_slot_meta(updated_pin_counts, usage_counts))

    def get_actual_block(self, physical_slot_id: int) -> torch.Tensor:
        self._validate_physical_slot_id(physical_slot_id)
        return self.actual_blocks[physical_slot_id]

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

    def _detect_native_extension(
        self,
        *,
        prefer_native_extension: bool,
    ) -> tuple[Any | None, str]:
        device_type = self.actual_blocks.device.type
        if device_type == "cuda":
            if not prefer_native_extension:
                raise RuntimeError("CUDA actual_blocks require kv_cache_adapter_cuda; strict CUDA path is disabled")
            native_ext = _load_cuda_extension_module()
            if native_ext is None:
                raise RuntimeError("CUDA actual_blocks require kv_cache_adapter_cuda, but the extension is unavailable")
            return native_ext, "cuda_ext_meta"
        if device_type in {"npu", "privateuseone"}:
            if not prefer_native_extension:
                raise RuntimeError("NPU actual_blocks require kv_cache_adapter_npu; strict NPU path is disabled")
            native_ext = _load_npu_extension_module()
            if native_ext is None:
                raise RuntimeError("NPU actual_blocks require kv_cache_adapter_npu, but the extension is unavailable")
            return native_ext, "npu_ext_meta"
        if not prefer_native_extension:
            return None, "strict"
        return None, "strict"

    def _inspect_load_requests(
        self,
        logical_block_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._native_ext is not None:
            if self._platform == "npu":
                return tuple(
                    self._native_ext.inspect_load_requests(
                        self._logical_to_physical.contiguous(),
                        self._slot_meta.contiguous(),
                        logical_block_ids.contiguous(),
                    ),
                )  # type: ignore[return-value]
            return tuple(
                self._native_ext.inspect_load_requests(
                    self._logical_to_physical.contiguous(),
                    self._slot_state.contiguous(),
                    self._pin_count.to(dtype=ID_DTYPE).contiguous(),
                    self._usage_count.contiguous(),
                    logical_block_ids.contiguous(),
                ),
            )  # type: ignore[return-value]

        current_physical = self._logical_to_physical.index_select(0, logical_block_ids)
        resident_mask = current_physical >= 0
        updated_pin_counts = torch.zeros_like(logical_block_ids)
        updated_usage_counts = torch.zeros_like(logical_block_ids, dtype=USAGE_DTYPE)

        if resident_mask.numel() == 0 or not torch.any(resident_mask):
            return current_physical, resident_mask, updated_pin_counts, updated_usage_counts

        hit_slot_ids = current_physical[resident_mask]
        hit_pin_counts = self._pin_count.index_select(0, hit_slot_ids) + 1
        hit_usage_counts = _saturating_increment_usage(self._usage_count.index_select(0, hit_slot_ids))
        updated_pin_counts[resident_mask] = hit_pin_counts.to(dtype=updated_pin_counts.dtype)
        updated_usage_counts[resident_mask] = hit_usage_counts
        return current_physical, resident_mask, updated_pin_counts, updated_usage_counts

    def _inspect_save_requests(
        self,
        logical_block_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._native_ext is not None:
            if self._platform == "npu":
                return tuple(
                    self._native_ext.inspect_save_requests(
                        self._logical_to_physical.contiguous(),
                        self._slot_meta.contiguous(),
                        logical_block_ids.contiguous(),
                    ),
                )  # type: ignore[return-value]
            return tuple(
                self._native_ext.inspect_save_requests(
                    self._logical_to_physical.contiguous(),
                    self._slot_state.contiguous(),
                    self._usage_count.contiguous(),
                    logical_block_ids.contiguous(),
                ),
            )  # type: ignore[return-value]

        current_physical = self._logical_to_physical.index_select(0, logical_block_ids)
        existing_mask = current_physical >= 0
        final_usage_counts = torch.ones_like(logical_block_ids, dtype=USAGE_DTYPE)

        if existing_mask.numel() == 0 or not torch.any(existing_mask):
            return current_physical, existing_mask, final_usage_counts

        existing_physical = current_physical[existing_mask]
        final_usage_counts[existing_mask] = _saturating_increment_usage(
            self._usage_count.index_select(0, existing_physical),
        )
        return current_physical, existing_mask, final_usage_counts

    def _pop_reusable_slots(self, count: int, *, blocked_slot_ids: torch.Tensor) -> torch.Tensor:
        if count == 0:
            return self._logical_to_physical[:0]

        blocked_slot_ids = blocked_slot_ids.contiguous()
        if self._native_ext is not None:
            if self._platform == "npu":
                return self._native_ext.pop_reusable_slots(
                    self._slot_meta.contiguous(),
                    self._search_start.contiguous(),
                    blocked_slot_ids,
                    count,
                )
            return self._native_ext.pop_reusable_slots(
                self._usage_count.contiguous(),
                self._reusable_mask.contiguous(),
                self._search_start.contiguous(),
                blocked_slot_ids,
                count,
            )

        available_mask = self._reusable_mask.clone()
        if blocked_slot_ids.numel() > 0:
            available_mask.index_fill_(0, blocked_slot_ids, False)
        scan_order = (
            self._search_start[0] + torch.arange(self.num_actual_blocks, device=self.actual_blocks.device, dtype=ID_DTYPE)
        ) % self.num_actual_blocks
        scan_available = available_mask.index_select(0, scan_order)
        scan_usage = self._usage_count.index_select(0, scan_order)
        selected_parts: list[torch.Tensor] = []
        selected_count = 0
        selected_threshold = 0
        for threshold in range(USAGE_COUNT_MAX + 1):
            threshold_slots = scan_order[scan_available & (scan_usage == threshold)]
            if threshold_slots.numel() == 0:
                continue
            remaining = count - selected_count
            chosen = threshold_slots[:remaining]
            if chosen.numel() == 0:
                continue
            selected_parts.append(chosen)
            selected_count += int(chosen.numel())
            selected_threshold = threshold
            if selected_count == count:
                selected_slot_ids = torch.cat(selected_parts, dim=0)
                if selected_threshold > 0:
                    aged_usage = torch.clamp(
                        self._usage_count.to(dtype=torch.int32) - selected_threshold,
                        min=0,
                    ).to(dtype=USAGE_DTYPE)
                    self._slot_meta.copy_(_pack_slot_meta(self._pin_count, aged_usage))
                self._search_start[0] = (selected_slot_ids[-1] + 1) % self.num_actual_blocks
                return selected_slot_ids
        raise InsufficientCapacityError("No reusable actual block is available; all resident blocks are pinned")

    def _build_eviction_plan(
        self,
        logical_block_ids: torch.Tensor,
        physical_slot_ids: torch.Tensor,
    ) -> _EvictionPlan:
        if physical_slot_ids.numel() == 0:
            return _EvictionPlan(logical_block_ids[:0], self.actual_blocks[:0])

        previous_logical_ids = self._physical_to_logical.index_select(0, physical_slot_ids)
        eviction_mask = (
            (previous_logical_ids >= 0)
            & (previous_logical_ids != logical_block_ids)
        )
        if not torch.any(eviction_mask):
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
        hit_usage_counts: torch.Tensor,
        miss_usage_counts: torch.Tensor,
    ) -> None:
        if self._native_ext is not None:
            if self._platform == "npu":
                self._native_ext.commit_load_metadata(
                    self._logical_to_physical.contiguous(),
                    self._physical_to_logical.contiguous(),
                    self._slot_meta.contiguous(),
                    evicted_logical_block_ids.contiguous(),
                    miss_logical_block_ids.contiguous(),
                    miss_physical_slot_ids.contiguous(),
                    hit_slot_ids.contiguous(),
                    hit_pin_counts.contiguous(),
                    hit_usage_counts.contiguous(),
                    miss_usage_counts.contiguous(),
                )
            else:
                touched_slot_ids = torch.cat((hit_slot_ids, miss_physical_slot_ids), dim=0)
                touched_usage_counts = torch.cat((hit_usage_counts, miss_usage_counts), dim=0)
                empty_ids = self._logical_to_physical[:0]
                empty_usage = self._usage_count[:0]
                self._native_ext.commit_load_metadata(
                    self._logical_to_physical.contiguous(),
                    self._physical_to_logical.contiguous(),
                    self._slot_state.contiguous(),
                    self._pin_count.to(dtype=ID_DTYPE).contiguous(),
                    self._reusable_mask.contiguous(),
                    self._usage_count.contiguous(),
                    evicted_logical_block_ids.contiguous(),
                    empty_ids.contiguous(),
                    empty_usage.contiguous(),
                    miss_logical_block_ids.contiguous(),
                    miss_physical_slot_ids.contiguous(),
                    hit_slot_ids.contiguous(),
                    hit_pin_counts.to(dtype=ID_DTYPE).contiguous(),
                    touched_slot_ids.contiguous(),
                    touched_usage_counts.contiguous(),
                )
            return

        if evicted_logical_block_ids.numel() > 0:
            self._logical_to_physical.index_fill_(0, evicted_logical_block_ids, -1)
        if miss_logical_block_ids.numel() > 0:
            self._logical_to_physical.index_put_((miss_logical_block_ids,), miss_physical_slot_ids)
            self._physical_to_logical.index_put_((miss_physical_slot_ids,), miss_logical_block_ids)
        if hit_slot_ids.numel() > 0:
            self._slot_meta.index_put_((hit_slot_ids,), _pack_slot_meta(hit_pin_counts, hit_usage_counts))
        if miss_physical_slot_ids.numel() > 0:
            miss_pin_counts = torch.ones_like(miss_physical_slot_ids, dtype=PIN_COUNT_DTYPE)
            self._slot_meta.index_put_((miss_physical_slot_ids,), _pack_slot_meta(miss_pin_counts, miss_usage_counts))

    def _commit_save_metadata(
        self,
        *,
        evicted_logical_block_ids: torch.Tensor,
        logical_block_ids: torch.Tensor,
        physical_slot_ids: torch.Tensor,
        final_pin_counts: torch.Tensor,
        final_usage_counts: torch.Tensor,
    ) -> None:
        if self._native_ext is not None:
            if self._platform == "npu":
                self._native_ext.commit_save_metadata(
                    self._logical_to_physical.contiguous(),
                    self._physical_to_logical.contiguous(),
                    self._slot_meta.contiguous(),
                    evicted_logical_block_ids.contiguous(),
                    logical_block_ids.contiguous(),
                    physical_slot_ids.contiguous(),
                    final_pin_counts.contiguous(),
                    final_usage_counts.contiguous(),
                )
            else:
                empty_ids = self._logical_to_physical[:0]
                empty_usage = self._usage_count[:0]
                self._native_ext.commit_save_metadata(
                    self._logical_to_physical.contiguous(),
                    self._physical_to_logical.contiguous(),
                    self._slot_state.contiguous(),
                    self._pin_count.to(dtype=ID_DTYPE).contiguous(),
                    self._reusable_mask.contiguous(),
                    self._usage_count.contiguous(),
                    evicted_logical_block_ids.contiguous(),
                    empty_ids.contiguous(),
                    empty_usage.contiguous(),
                    logical_block_ids.contiguous(),
                    physical_slot_ids.contiguous(),
                    final_pin_counts.to(dtype=ID_DTYPE).contiguous(),
                    final_usage_counts.contiguous(),
                )
            return

        if evicted_logical_block_ids.numel() > 0:
            self._logical_to_physical.index_fill_(0, evicted_logical_block_ids, -1)
        self._logical_to_physical.index_put_((logical_block_ids,), physical_slot_ids)
        self._physical_to_logical.index_put_((physical_slot_ids,), logical_block_ids)
        self._slot_meta.index_put_((physical_slot_ids,), _pack_slot_meta(final_pin_counts, final_usage_counts))

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
        if logical_block_ids.device.type in {"cuda", "npu", "privateuseone"}:
            return
        if torch.any(logical_block_ids < 0) or torch.any(logical_block_ids >= self.num_logical_blocks):
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
        self._present_mask: torch.Tensor | None = None
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
                present_mask.detach().clone().to(device=self._storage.device)
                if present_mask is not None
                else torch.ones((num_logical_blocks,), dtype=torch.bool, device=self._storage.device)
            )
        else:
            if num_logical_blocks is None:
                if initial_data:
                    num_logical_blocks = max(initial_data) + 1
                else:
                    raise ValueError("num_logical_blocks is required when initial_data is empty")
            self._present_mask = present_mask.detach().clone() if present_mask is not None else None
            if initial_data:
                first_payload = next(iter(initial_data.values()))
                self._storage = first_payload.new_zeros((num_logical_blocks, *first_payload.shape))
                if self._present_mask is None:
                    self._present_mask = torch.zeros((num_logical_blocks,), dtype=torch.bool, device=first_payload.device)
                else:
                    self._present_mask = self._present_mask.to(device=first_payload.device)
                initial_ids = torch.tensor(list(initial_data.keys()), dtype=ID_DTYPE, device=first_payload.device)
                initial_payloads = torch.stack([initial_data[key] for key in initial_data.keys()], dim=0)
                self._storage.index_copy_(0, initial_ids, initial_payloads)
                self._present_mask.index_fill_(0, initial_ids, True)

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
            self._ensure_present_mask(device=self._storage.device)
            self._storage.index_copy_(
                0,
                logical_block_ids.to(device=self._storage.device),
                block_data.to(device=self._storage.device, dtype=self._storage.dtype),
            )
            self._present_mask.index_fill_(0, logical_block_ids.to(device=self._storage.device), True)
            self._record_ids("save", logical_block_ids)
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
            self._ensure_present_mask(device=self._storage.device)
            hit_mask = self._present_mask.index_select(0, logical_block_ids.to(device=self._storage.device))
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
            self._record_ids("load", logical_block_ids)
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

    def _ensure_present_mask(self, *, device: torch.device) -> None:
        if self._present_mask is None:
            self._present_mask = torch.zeros((self.num_logical_blocks,), dtype=torch.bool, device=device)
        elif self._present_mask.device != device:
            self._present_mask = self._present_mask.to(device=device)

    def _record_ids(self, operation: str, logical_block_ids: torch.Tensor) -> None:
        if logical_block_ids.device.type != "cpu":
            return
        logical_ids_list = logical_block_ids.detach().tolist()
        if operation == "save":
            self.save_calls.extend(logical_ids_list)
        else:
            self.load_calls.extend(logical_ids_list)
        self.operation_log.extend((operation, logical_block_id) for logical_block_id in logical_ids_list)


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
        _maybe_enable_lmcache_ascend()
        from lmcache.utils import CacheEngineKey
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.memory_management import (
            MemoryFormat,
            MemoryObjMetadata,
            PinMemoryAllocator,
            TensorMemoryObj,
        )
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

        self._CacheEngineKey = CacheEngineKey
        self._lmcache_memcpy_async_d2h: Any | None = None
        self._lmcache_memcpy_async_h2d: Any | None = None
        self._MemoryFormat = MemoryFormat
        self._MemoryObjMetadata = MemoryObjMetadata
        self._PinMemoryAllocator = PinMemoryAllocator
        self._TensorMemoryObj = TensorMemoryObj
        self._block_shape = torch.Size(block_shape)
        self._block_dtype = block_dtype
        self._model_name = model_name
        self._world_size = world_size
        self._worker_id = worker_id
        self._pin_allocator: Any | None = None
        self._pin_allocator_size_bytes = max(1, int(max_local_cpu_size_gb * (1024**3)))
        self._prefer_pinned_host = torch.cuda.is_available()
        self._npu_fused_transfer = _NPUFusedBlockTransfer(
            block_shape=self._block_shape,
            block_dtype=self._block_dtype,
            memory_format_enum=self._MemoryFormat,
        )
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

        logical_block_ids_cpu = self._logical_ids_to_cpu(logical_block_ids)
        keys = self._make_keys(logical_block_ids_cpu)
        memory_objs = self._make_memory_objs_from_payloads(block_data)
        self._backend.batched_remove(keys)
        try:
            self._backend.batched_submit_put_task(
                keys,
                memory_objs,
            )
        finally:
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()

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

        logical_block_ids_cpu = self._logical_ids_to_cpu(logical_block_ids)
        memory_objs = self._backend.batched_get_blocking(self._make_keys(logical_block_ids_cpu))
        if any(memory_obj is None for memory_obj in memory_objs):
            raise BlockNotFoundError("some logical blocks are not in LMCache")

        try:
            physical_slot_ids_device = (
                physical_slot_ids
                if physical_slot_ids.device == actual_blocks.device
                else physical_slot_ids.to(device=actual_blocks.device)
            )
            if self._npu_fused_transfer.can_load_into(actual_blocks) and self._npu_fused_transfer.can_consume_memory_objs(memory_objs):
                self._npu_fused_transfer.load_into_actual_blocks(
                    memory_objs=memory_objs,
                    actual_blocks=actual_blocks,
                    physical_slot_ids=physical_slot_ids_device,
                )
                return
            if actual_blocks.device.type == "cpu":
                loaded_payloads_cpu = self._stack_payload_tensors(
                    [memory_obj.tensor for memory_obj in memory_objs if memory_obj is not None],
                )
                actual_blocks.index_copy_(
                    0,
                    physical_slot_ids_device,
                    loaded_payloads_cpu.to(dtype=actual_blocks.dtype),
                )
                return

            if actual_blocks.device.type in {"npu", "privateuseone"}:
                self._load_blocks_generic_accelerator(
                    memory_objs=memory_objs,
                    actual_blocks=actual_blocks,
                    physical_slot_ids=physical_slot_ids_device,
                )
                return

            self._ensure_cuda_memcpy_ops_loaded()
            for memory_obj, physical_slot_id in zip(memory_objs, self._logical_ids_to_cpu(physical_slot_ids), strict=False):
                self._lmcache_memcpy_async_h2d(
                    memory_obj,
                    actual_blocks[physical_slot_id],
                )
            _synchronize_device(actual_blocks.device)
        finally:
            for memory_obj in memory_objs:
                if memory_obj is not None:
                    memory_obj.ref_count_down()

    def shutdown(self) -> None:
        self._backend.clear()
        self._backend.close()
        if self._pin_allocator is not None:
            self._pin_allocator.close()
            self._pin_allocator = None

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

    def _logical_ids_to_cpu(self, logical_block_ids: torch.Tensor) -> torch.Tensor:
        if logical_block_ids.device.type == "cpu":
            return logical_block_ids.contiguous()
        return logical_block_ids.to(device="cpu")

    def _make_memory_objs_from_payloads(self, block_data: torch.Tensor) -> list[object]:
        if block_data.device.type == "cpu":
            payloads_cpu = block_data.detach().contiguous()
            return [self._make_memory_obj(payload) for payload in payloads_cpu.unbind(0)]

        if self._npu_fused_transfer.can_save_from(block_data):
            pinned_memory_objs = self._allocate_pinned_memory_objs(
                batch_size=block_data.shape[0],
                payload_shape=self._npu_fused_transfer.transfer_shape,
                payload_dtype=block_data.dtype,
                memory_format=self._npu_fused_transfer.memory_format,
            )
            if pinned_memory_objs is None:
                raise RuntimeError("NPU fused transfer requires pinned host allocation")
            try:
                self._npu_fused_transfer.save_from_block_data(
                    block_data=block_data,
                    memory_objs=pinned_memory_objs,
                )
                return pinned_memory_objs
            except Exception:
                if self._pin_allocator is not None:
                    self._pin_allocator.batched_free(pinned_memory_objs)
                raise

        pinned_memory_objs = self._allocate_pinned_memory_objs(
            batch_size=block_data.shape[0],
            payload_shape=torch.Size(block_data.shape[1:]),
            payload_dtype=block_data.dtype,
            memory_format=self._MemoryFormat.UNDEFINED,
        )
        if pinned_memory_objs is not None:
            try:
                if block_data.device.type in {"npu", "privateuseone"}:
                    self._save_blocks_generic_accelerator(
                        block_data=block_data,
                        memory_objs=pinned_memory_objs,
                    )
                else:
                    self._ensure_cuda_memcpy_ops_loaded()
                    for source_payload, memory_obj in zip(block_data.detach().unbind(0), pinned_memory_objs, strict=False):
                        self._lmcache_memcpy_async_d2h(source_payload, memory_obj)
                _synchronize_device(block_data.device)
                return pinned_memory_objs
            except Exception:
                if self._pin_allocator is not None:
                    self._pin_allocator.batched_free(pinned_memory_objs)
                raise

        payloads_cpu = torch.empty(
            block_data.shape,
            dtype=block_data.dtype,
            device="cpu",
            pin_memory=self._prefer_pinned_host and block_data.device.type == "cuda",
        )
        memory_objs = [self._make_memory_obj(payload) for payload in payloads_cpu.unbind(0)]
        if block_data.device.type in {"npu", "privateuseone"}:
            self._save_blocks_generic_accelerator(
                block_data=block_data,
                memory_objs=memory_objs,
            )
        else:
            self._ensure_cuda_memcpy_ops_loaded()
            for source_payload, memory_obj in zip(block_data.detach().unbind(0), memory_objs, strict=False):
                self._lmcache_memcpy_async_d2h(source_payload, memory_obj)
        _synchronize_device(block_data.device)
        return memory_objs

    def _allocate_pinned_memory_objs(
        self,
        *,
        batch_size: int,
        payload_shape: torch.Size,
        payload_dtype: torch.dtype,
        memory_format: object,
    ) -> list[object] | None:
        if batch_size == 0:
            return []
        if not self._supports_pinned_host_allocator():
            return None
        if self._pin_allocator is None:
            self._pin_allocator = self._PinMemoryAllocator(self._pin_allocator_size_bytes)
        return self._pin_allocator.batched_allocate(
            payload_shape,
            payload_dtype,
            batch_size,
            memory_format,
        )

    def _supports_pinned_host_allocator(self) -> bool:
        return torch.cuda.is_available() or hasattr(torch, "npu")

    def _ensure_cuda_memcpy_ops_loaded(self) -> None:
        if self._lmcache_memcpy_async_d2h is not None and self._lmcache_memcpy_async_h2d is not None:
            return
        _ensure_lmcache_c_ops_ready()
        memcpy_d2h, memcpy_h2d = _load_lmcache_cuda_memcpy_ops()
        self._lmcache_memcpy_async_d2h = memcpy_d2h
        self._lmcache_memcpy_async_h2d = memcpy_h2d

    def _save_blocks_generic_accelerator(
        self,
        *,
        block_data: torch.Tensor,
        memory_objs: list[object],
    ) -> None:
        for source_payload, memory_obj in zip(block_data.detach().unbind(0), memory_objs, strict=False):
            memory_obj.tensor.copy_(source_payload, non_blocking=True)

    def _load_blocks_generic_accelerator(
        self,
        *,
        memory_objs: list[object],
        actual_blocks: torch.Tensor,
        physical_slot_ids: torch.Tensor,
    ) -> None:
        loaded_payloads_cpu = self._stack_payload_tensors(
            [memory_obj.tensor for memory_obj in memory_objs if memory_obj is not None],
        )
        loaded_payloads_device = loaded_payloads_cpu.to(
            device=actual_blocks.device,
            dtype=actual_blocks.dtype,
        )
        actual_blocks.index_copy_(
            0,
            physical_slot_ids,
            loaded_payloads_device,
        )
        _synchronize_device(actual_blocks.device)

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

    def _stack_payload_tensors(self, payload_tensors: list[torch.Tensor]) -> torch.Tensor:
        if self._npu_fused_transfer.layout is None:
            return torch.stack(payload_tensors, dim=0)
        restored = [self._npu_fused_transfer.restore_stored_tensor(tensor) for tensor in payload_tensors]
        return torch.stack(restored, dim=0)


class _NPUFusedBlockTransfer:
    def __init__(
        self,
        *,
        block_shape: torch.Size,
        block_dtype: torch.dtype,
        memory_format_enum: Any,
    ) -> None:
        self.layout = _MergedKVBlockLayout.detect(block_shape)
        self._block_dtype = block_dtype
        self._MemoryFormat = memory_format_enum
        self._c_ops: Any | None = None
        self._staging_cache: torch.Tensor | None = None
        self._staging_capacity_tokens = 0
        self._staging_device: torch.device | None = None
        if self.layout is None:
            return
        try:
            self._c_ops = importlib.import_module("lmcache_ascend.c_ops")
        except Exception:
            self._c_ops = None

    @property
    def transfer_shape(self) -> torch.Size:
        if self.layout is None:
            raise RuntimeError("NPU fused transfer layout is unavailable")
        return self.layout.transfer_shape

    @property
    def memory_format(self) -> object:
        if self.layout is None:
            raise RuntimeError("NPU fused transfer layout is unavailable")
        if self.layout.token_major:
            return self._MemoryFormat.KV_2TD
        return self._MemoryFormat.KV_T2D

    def can_save_from(self, block_data: torch.Tensor) -> bool:
        return self._can_run_on(block_data.device) and tuple(block_data.shape[1:]) == tuple(self.layout.block_shape)

    def can_load_into(self, actual_blocks: torch.Tensor) -> bool:
        return self._can_run_on(actual_blocks.device) and tuple(actual_blocks.shape[1:]) == tuple(self.layout.block_shape)

    def save_from_block_data(
        self,
        *,
        block_data: torch.Tensor,
        memory_objs: list[object],
    ) -> None:
        if self.layout is None or self._c_ops is None:
            raise RuntimeError("NPU fused transfer is unavailable")
        self._c_ops.batched_fused_single_layer_kv_transfer(
            [memory_obj.tensor for memory_obj in memory_objs],
            self._get_staging_cache(block_data.device, block_data.shape[0]),
            block_data.contiguous(),
            self._dense_slot_mapping(block_data.shape[0], device=block_data.device),
            self._chunk_offsets(block_data.shape[0]),
            self._chunk_sizes(block_data.shape[0]),
            True,
            1,
            self.layout.token_major,
            False,
        )
        _synchronize_device(block_data.device)

    def load_into_actual_blocks(
        self,
        *,
        memory_objs: list[object],
        actual_blocks: torch.Tensor,
        physical_slot_ids: torch.Tensor,
    ) -> None:
        if self.layout is None or self._c_ops is None:
            raise RuntimeError("NPU fused transfer is unavailable")
        self._c_ops.batched_fused_single_layer_kv_transfer(
            [memory_obj.tensor for memory_obj in memory_objs if memory_obj is not None],
            self._get_staging_cache(actual_blocks.device, physical_slot_ids.shape[0]),
            actual_blocks,
            self._physical_slot_mapping(physical_slot_ids.contiguous()),
            self._chunk_offsets(physical_slot_ids.shape[0]),
            self._chunk_sizes(physical_slot_ids.shape[0]),
            False,
            1,
            self.layout.token_major,
            False,
        )
        _synchronize_device(actual_blocks.device)

    def restore_stored_tensor(self, stored_tensor: torch.Tensor) -> torch.Tensor:
        if self.layout is None:
            return stored_tensor
        if tuple(stored_tensor.shape) == tuple(self.layout.transfer_shape):
            return self.layout.restore_block_tensor(stored_tensor)
        return stored_tensor

    def can_consume_memory_objs(self, memory_objs: list[object]) -> bool:
        if self.layout is None:
            return False
        expected_shape = tuple(self.layout.transfer_shape)
        return all(
            memory_obj is not None
            and getattr(memory_obj, "tensor", None) is not None
            and tuple(memory_obj.tensor.shape) == expected_shape
            for memory_obj in memory_objs
        )

    def _can_run_on(self, device: torch.device) -> bool:
        return self.layout is not None and self._c_ops is not None and device.type in {"npu", "privateuseone"}

    def _dense_slot_mapping(self, num_blocks: int, *, device: torch.device) -> torch.Tensor:
        if self.layout is None:
            raise RuntimeError("NPU fused transfer layout is unavailable")
        return torch.arange(
            num_blocks * self.layout.block_size,
            dtype=ID_DTYPE,
            device=device,
        )

    def _physical_slot_mapping(self, physical_slot_ids: torch.Tensor) -> torch.Tensor:
        if self.layout is None:
            raise RuntimeError("NPU fused transfer layout is unavailable")
        token_offsets = torch.arange(
            self.layout.block_size,
            dtype=ID_DTYPE,
            device=physical_slot_ids.device,
        )
        return (
            physical_slot_ids.reshape(-1, 1) * self.layout.block_size + token_offsets.reshape(1, -1)
        ).reshape(-1)

    def _chunk_offsets(self, num_blocks: int) -> list[int]:
        if self.layout is None:
            raise RuntimeError("NPU fused transfer layout is unavailable")
        return [block_index * self.layout.block_size for block_index in range(num_blocks)]

    def _chunk_sizes(self, num_blocks: int) -> list[int]:
        if self.layout is None:
            raise RuntimeError("NPU fused transfer layout is unavailable")
        return [self.layout.block_size] * num_blocks

    def _get_staging_cache(self, device: torch.device, num_blocks: int) -> torch.Tensor:
        if self.layout is None:
            raise RuntimeError("NPU fused transfer layout is unavailable")
        required_tokens = num_blocks * self.layout.block_size
        if (
            self._staging_cache is None
            or self._staging_device != device
            or self._staging_capacity_tokens < required_tokens
        ):
            full_shape = (
                (required_tokens, 2, self.layout.hidden_dim)
                if self.layout.token_major
                else (2, required_tokens, self.layout.hidden_dim)
            )
            self._staging_cache = torch.empty(
                full_shape,
                dtype=self._block_dtype,
                device=device,
            )
            self._staging_capacity_tokens = required_tokens
            self._staging_device = device
        if self.layout.token_major:
            return self._staging_cache[:required_tokens]
        return self._staging_cache[:, :required_tokens]


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
    module_dir = pathlib.Path(__file__).resolve().parent
    direct_spec = importlib.machinery.PathFinder.find_spec("kv_cache_adapter_cuda", [str(module_dir)])
    if direct_spec is not None and direct_spec.loader is not None:
        module = importlib.util.module_from_spec(direct_spec)
        sys.modules["kv_cache_adapter_cuda"] = module
        if __package__:
            sys.modules.setdefault(f"{__package__}.kv_cache_adapter_cuda", module)
        direct_spec.loader.exec_module(module)
        return module
    module_names = []
    if __package__:
        module_names.append(f"{__package__}.kv_cache_adapter_cuda")
    module_names.append("kv_cache_adapter_cuda")
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


@lru_cache(maxsize=1)
def _load_npu_extension_module() -> Any | None:
    module_dir = pathlib.Path(__file__).resolve().parent
    wrapper_spec = importlib.machinery.PathFinder.find_spec("kv_cache_adapter_npu_custom", [str(module_dir)])
    if wrapper_spec is not None and wrapper_spec.loader is not None:
        module = importlib.util.module_from_spec(wrapper_spec)
        sys.modules["kv_cache_adapter_npu_custom"] = module
        if __package__:
            sys.modules.setdefault(f"{__package__}.kv_cache_adapter_npu_custom", module)
        wrapper_spec.loader.exec_module(module)
        return module
    legacy_spec = importlib.machinery.PathFinder.find_spec("kv_cache_adapter_npu", [str(module_dir)])
    if legacy_spec is not None and legacy_spec.loader is not None:
        module = importlib.util.module_from_spec(legacy_spec)
        sys.modules["kv_cache_adapter_npu"] = module
        if __package__:
            sys.modules.setdefault(f"{__package__}.kv_cache_adapter_npu", module)
        legacy_spec.loader.exec_module(module)
        return module
    if __package__:
        qualified_custom = f"{__package__}.kv_cache_adapter_npu_custom"
        try:
            return importlib.import_module(qualified_custom)
        except Exception:
            pass
        qualified_legacy = f"{__package__}.kv_cache_adapter_npu"
        try:
            return importlib.import_module(qualified_legacy)
        except Exception:
            return None
    for module_name in ("kv_cache_adapter_npu_custom", "kv_cache_adapter_npu"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


@lru_cache(maxsize=1)
def _load_lmcache_cuda_memcpy_ops() -> tuple[Any, Any]:
    gpu_ops = importlib.import_module("lmcache.v1.gpu_connector.gpu_ops")
    return gpu_ops.lmcache_memcpy_async_d2h, gpu_ops.lmcache_memcpy_async_h2d


def _saturating_increment_usage(values: torch.Tensor) -> torch.Tensor:
    incremented = values.to(dtype=torch.int32) + 1
    return torch.clamp(incremented, max=USAGE_COUNT_MAX).to(dtype=USAGE_DTYPE)


def _slot_meta_pin_counts(slot_meta: torch.Tensor) -> torch.Tensor:
    return (slot_meta.to(dtype=torch.int32) & _PIN_COUNT_MASK).to(dtype=PIN_COUNT_DTYPE)


def _slot_meta_usage_counts(slot_meta: torch.Tensor) -> torch.Tensor:
    return ((slot_meta.to(dtype=torch.int32) >> _USAGE_COUNT_SHIFT) & _USAGE_COUNT_MASK).to(dtype=USAGE_DTYPE)


def _pack_slot_meta(pin_counts: torch.Tensor, usage_counts: torch.Tensor) -> torch.Tensor:
    packed = (
        (pin_counts.to(dtype=torch.int32) & _PIN_COUNT_MASK)
        | ((usage_counts.to(dtype=torch.int32) & _USAGE_COUNT_MASK) << _USAGE_COUNT_SHIFT)
    )
    return packed.to(dtype=SLOT_META_DTYPE)


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


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type in {"npu", "privateuseone"} and hasattr(torch, "npu"):
        torch.npu.synchronize()


@lru_cache(maxsize=1)
def _maybe_enable_lmcache_ascend() -> None:
    if not hasattr(torch, "npu"):
        return
    try:
        importlib.import_module("lmcache_ascend")
    except Exception:
        return


@lru_cache(maxsize=1)
def _ensure_lmcache_c_ops_ready() -> None:
    try:
        importlib.import_module("lmcache.c_ops")
        return
    except ImportError as first_error:
        torch_libdir = pathlib.Path(torch.__file__).resolve().parent / "lib"
        preload_names = (
            "libc10.so",
            "libtorch.so",
            "libtorch_cpu.so",
            "libtorch_python.so",
        )
        cuda_names = (
            "libc10_cuda.so",
            "libtorch_cuda.so",
        )
        for name in preload_names + cuda_names:
            lib_path = torch_libdir / name
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        try:
            importlib.import_module("lmcache.c_ops")
            return
        except ImportError:
            raise first_error
