from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

import torch

ID_DTYPE = torch.int64
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


@dataclass
class _ReservationSnapshot:
    logical_block_ids: torch.Tensor
    physical_slot_ids: torch.Tensor
    previous_logical_ids: torch.Tensor
    previous_states: torch.Tensor
    previous_pin_counts: torch.Tensor
    restore_slot_ids: torch.Tensor
    restore_payloads: torch.Tensor


@dataclass
class _SavePlan:
    logical_block_ids: torch.Tensor
    physical_slot_ids: torch.Tensor
    final_pin_counts: torch.Tensor
    snapshot: _ReservationSnapshot


@dataclass
class _LoadPlan:
    logical_block_ids: torch.Tensor
    physical_slot_ids: torch.Tensor
    snapshot: _ReservationSnapshot


@dataclass
class _EvictionPlan:
    logical_block_ids: torch.Tensor
    payloads: torch.Tensor


class _RuntimeOps(Protocol):
    path: str

    def unique_preserve_order(self, values: torch.Tensor) -> torch.Tensor:
        ...


class _PythonRuntimeOps:
    path = "python"

    def unique_preserve_order(self, values: torch.Tensor) -> torch.Tensor:
        return torch.tensor(list(dict.fromkeys(values.tolist())), dtype=ID_DTYPE)


class _TorchCustomOpsRuntime:
    path = "npu_op"

    def __init__(self, ops_namespace: Any) -> None:
        self._ops_namespace = ops_namespace

    def unique_preserve_order(self, values: torch.Tensor) -> torch.Tensor:
        result = self._ops_namespace.unique_preserve_order(values)
        _require_id_tensor(result, name="torch.ops.kv_cache_adapter.unique_preserve_order(values)")
        return result

    @classmethod
    def try_create(cls) -> _TorchCustomOpsRuntime | None:
        namespace = getattr(torch.ops, "kv_cache_adapter", None)
        if namespace is None or not hasattr(namespace, "unique_preserve_order"):
            return None
        return cls(namespace)


class KVCacheAdapter:
    """Maps n logical block IDs onto m resident blocks stored in `actual_blocks`.

    Contract:
    - `actual_blocks` is the real `(m, ...)` storage tensor
    - `logical_block_ids` must be a CPU `torch.int64` tensor shaped `(k,)`
    - `block_data` must be shaped `(k, ...)`
    - calls are expected to be serialized by the caller; this class is not thread-safe
    """

    def __init__(
        self,
        num_actual_blocks: int,
        num_logical_blocks: int,
        actual_blocks: torch.Tensor,
        backend: BlockStoreBackend,
        *,
        max_workers: int | None = None,
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

        self.num_actual_blocks = num_actual_blocks
        self.num_logical_blocks = num_logical_blocks
        self.actual_blocks = actual_blocks
        self.backend = backend
        self._runtime_ops = self._detect_runtime_ops()
        self.runtime_path = self._runtime_ops.path

        self._logical_to_physical = torch.full((num_logical_blocks,), -1, dtype=ID_DTYPE)
        self._physical_to_logical = torch.full((num_actual_blocks,), -1, dtype=ID_DTYPE)
        self._slot_state = torch.full((num_actual_blocks,), STATE_FREE, dtype=torch.int64)
        self._pin_count = torch.zeros((num_actual_blocks,), dtype=torch.int64)
        self._reusable_slots: OrderedDict[int, None] = OrderedDict(
            (physical_slot_id, None) for physical_slot_id in range(num_actual_blocks)
        )

    def save(self, logical_block_ids: torch.Tensor, block_data: torch.Tensor) -> None:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        self._validate_logical_block_ids(logical_block_ids)
        _require_block_data_tensor(
            block_data,
            expected_blocks=logical_block_ids.shape[0],
            name="block_data",
        )
        _require_trailing_shape(block_data, self.actual_blocks, name="block_data")

        if logical_block_ids.numel() == 0:
            return

        save_plan, eviction_plan = self._plan_save(logical_block_ids)

        try:
            if eviction_plan.logical_block_ids.numel() > 0:
                self.backend.save_blocks(eviction_plan.logical_block_ids, eviction_plan.payloads)
        except Exception:
            self._rollback_save(save_plan)
            raise

        self._commit_save(save_plan, block_data)

    def load(self, logical_block_ids: torch.Tensor) -> torch.Tensor:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        self._validate_logical_block_ids(logical_block_ids)
        if logical_block_ids.numel() == 0:
            return torch.empty_like(logical_block_ids)

        unique_ids = self._runtime_ops.unique_preserve_order(logical_block_ids)
        hit_slot_ids, load_plan, eviction_plan = self._plan_load(unique_ids)

        try:
            if eviction_plan.logical_block_ids.numel() > 0:
                self.backend.save_blocks(eviction_plan.logical_block_ids, eviction_plan.payloads)
            if load_plan.logical_block_ids.numel() > 0:
                self.backend.load_blocks(
                    load_plan.logical_block_ids,
                    self.actual_blocks,
                    load_plan.physical_slot_ids,
                )
        except Exception:
            self._rollback_load(hit_slot_ids, load_plan)
            raise

        self._commit_load(load_plan)
        return self._logical_to_physical.index_select(0, logical_block_ids)

    def release(self, logical_block_ids: torch.Tensor) -> None:
        _require_id_tensor(logical_block_ids, name="logical_block_ids")
        self._validate_logical_block_ids(logical_block_ids)
        unique_ids = self._runtime_ops.unique_preserve_order(logical_block_ids)
        if unique_ids.numel() == 0:
            return

        physical_slot_ids = self._logical_to_physical.index_select(0, unique_ids)
        if torch.any(physical_slot_ids < 0):
            raise KeyError("logical block is not resident")

        states = self._slot_state.index_select(0, physical_slot_ids)
        if torch.any(states != STATE_RESIDENT):
            raise KVCacheAdapterError("logical block is busy")

        pin_counts = self._pin_count.index_select(0, physical_slot_ids)
        if torch.any(pin_counts <= 0):
            raise KVCacheAdapterError("logical block is not pinned")

        updated_pin_counts = pin_counts - 1
        self._pin_count.index_put_((physical_slot_ids,), updated_pin_counts)
        self._touch_reusable_slots(physical_slot_ids[updated_pin_counts == 0])

    def get_actual_block(self, physical_slot_id: int) -> torch.Tensor:
        self._validate_physical_slot_id(physical_slot_id)
        return self.actual_blocks[physical_slot_id].detach().clone()

    def debug_snapshot(self) -> dict[str, object]:
        return {
            "runtime_path": self.runtime_path,
            "logical_to_physical": {
                logical_block_id: int(physical_slot_id)
                for logical_block_id, physical_slot_id in enumerate(self._logical_to_physical.tolist())
                if physical_slot_id >= 0
            },
            "lru_unpinned": list(self._reusable_slots.keys()),
            "slot_state": [STATE_NAMES[int(state)] for state in self._slot_state.tolist()],
            "pin_count": self._pin_count.detach().clone(),
            "actual_blocks": self.actual_blocks.detach().clone(),
        }

    def shutdown(self) -> None:
        backend_shutdown = getattr(self.backend, "shutdown", None)
        if callable(backend_shutdown):
            backend_shutdown()
            return

        backend_close = getattr(self.backend, "close", None)
        if callable(backend_close):
            backend_close()
        return None

    def _detect_runtime_ops(self) -> _RuntimeOps:
        accelerated_runtime = _TorchCustomOpsRuntime.try_create()
        if accelerated_runtime is not None:
            return accelerated_runtime
        return _PythonRuntimeOps()

    def _plan_save(self, logical_block_ids: torch.Tensor) -> tuple[_SavePlan, _EvictionPlan]:
        current_physical = self._logical_to_physical.index_select(0, logical_block_ids)
        existing_mask = current_physical >= 0
        existing_physical = current_physical[existing_mask]
        if existing_physical.numel() > 0:
            existing_states = self._slot_state.index_select(0, existing_physical)
            if torch.any(existing_states != STATE_RESIDENT):
                raise KVCacheAdapterError("logical block is busy")

        missing_ids = logical_block_ids[~existing_mask]
        allocated_physical = self._pop_reusable_slots(missing_ids.shape[0])

        selected_physical = current_physical.clone()
        if missing_ids.numel() > 0:
            selected_physical[~existing_mask] = allocated_physical

        snapshot = self._capture_snapshot(logical_block_ids, selected_physical)
        eviction_plan = self._build_eviction_plan(logical_block_ids, snapshot)

        if eviction_plan.logical_block_ids.numel() > 0:
            self._logical_to_physical.index_put_(
                (eviction_plan.logical_block_ids,),
                torch.full_like(eviction_plan.logical_block_ids, -1),
            )
        self._logical_to_physical.index_put_((logical_block_ids,), selected_physical)
        self._physical_to_logical.index_put_((selected_physical,), logical_block_ids)
        self._slot_state.index_put_((selected_physical,), torch.full_like(selected_physical, STATE_RESERVED))
        final_pin_counts = snapshot.previous_pin_counts.clone()
        self._pin_count.index_put_((selected_physical,), final_pin_counts)
        self._remove_reusable_slots(selected_physical)

        return _SavePlan(logical_block_ids, selected_physical, final_pin_counts, snapshot), eviction_plan

    def _plan_load(self, logical_block_ids: torch.Tensor) -> tuple[torch.Tensor, _LoadPlan, _EvictionPlan]:
        current_physical = self._logical_to_physical.index_select(0, logical_block_ids)
        resident_mask = current_physical >= 0
        hit_slot_ids = current_physical[resident_mask]
        if hit_slot_ids.numel() > 0:
            hit_states = self._slot_state.index_select(0, hit_slot_ids)
            if torch.any(hit_states != STATE_RESIDENT):
                raise KVCacheAdapterError("logical block is busy")

            updated_pin_counts = self._pin_count.index_select(0, hit_slot_ids) + 1
            self._pin_count.index_put_((hit_slot_ids,), updated_pin_counts)
            self._remove_reusable_slots(hit_slot_ids)

        miss_logical_ids = logical_block_ids[~resident_mask]
        if miss_logical_ids.numel() == 0:
            empty_ids = logical_block_ids[:0]
            empty_snapshot = _ReservationSnapshot(
                empty_ids,
                empty_ids,
                empty_ids,
                empty_ids,
                empty_ids,
                empty_ids,
                self.actual_blocks[:0].detach().clone(),
            )
            empty_payloads = self.actual_blocks.new_empty((0, *self.actual_blocks.shape[1:]))
            return hit_slot_ids, _LoadPlan(empty_ids, empty_ids, empty_snapshot), _EvictionPlan(empty_ids, empty_payloads)

        allocated_physical = self._pop_reusable_slots(miss_logical_ids.shape[0])
        snapshot = self._capture_snapshot(miss_logical_ids, allocated_physical)
        eviction_plan = self._build_eviction_plan(miss_logical_ids, snapshot)

        if eviction_plan.logical_block_ids.numel() > 0:
            self._logical_to_physical.index_put_(
                (eviction_plan.logical_block_ids,),
                torch.full_like(eviction_plan.logical_block_ids, -1),
            )
        self._logical_to_physical.index_put_((miss_logical_ids,), allocated_physical)
        self._physical_to_logical.index_put_((allocated_physical,), miss_logical_ids)
        self._slot_state.index_put_((allocated_physical,), torch.full_like(allocated_physical, STATE_LOADING))
        self._pin_count.index_put_((allocated_physical,), torch.zeros_like(allocated_physical))
        self._remove_reusable_slots(allocated_physical)

        return hit_slot_ids, _LoadPlan(miss_logical_ids, allocated_physical, snapshot), eviction_plan

    def _capture_snapshot(
        self,
        logical_block_ids: torch.Tensor,
        physical_slot_ids: torch.Tensor,
    ) -> _ReservationSnapshot:
        previous_logical_ids = self._physical_to_logical.index_select(0, physical_slot_ids)
        previous_states = self._slot_state.index_select(0, physical_slot_ids)
        previous_pin_counts = self._pin_count.index_select(0, physical_slot_ids)
        restore_mask = previous_states == STATE_RESIDENT
        restore_slot_ids = physical_slot_ids[restore_mask]
        if restore_slot_ids.numel() > 0:
            restore_payloads = self.actual_blocks.index_select(
                0,
                restore_slot_ids.to(device=self.actual_blocks.device),
            ).detach().clone()
        else:
            restore_payloads = self.actual_blocks[:0].detach().clone()
        return _ReservationSnapshot(
            logical_block_ids=logical_block_ids.clone(),
            physical_slot_ids=physical_slot_ids.clone(),
            previous_logical_ids=previous_logical_ids,
            previous_states=previous_states,
            previous_pin_counts=previous_pin_counts,
            restore_slot_ids=restore_slot_ids.clone(),
            restore_payloads=restore_payloads,
        )

    def _build_eviction_plan(
        self,
        logical_block_ids: torch.Tensor,
        snapshot: _ReservationSnapshot,
    ) -> _EvictionPlan:
        eviction_mask = (
            (snapshot.previous_states == STATE_RESIDENT)
            & (snapshot.previous_logical_ids >= 0)
            & (snapshot.previous_logical_ids != logical_block_ids)
        )
        if not torch.any(eviction_mask):
            return _EvictionPlan(logical_block_ids[:0], self.actual_blocks[:0])

        eviction_logical_ids = snapshot.previous_logical_ids[eviction_mask]
        eviction_slot_ids = snapshot.physical_slot_ids[eviction_mask]
        eviction_payloads = self.actual_blocks.index_select(
            0,
            eviction_slot_ids.to(device=self.actual_blocks.device),
        )
        return _EvictionPlan(eviction_logical_ids, eviction_payloads)

    def _commit_save(self, save_plan: _SavePlan, block_data: torch.Tensor) -> None:
        if save_plan.logical_block_ids.numel() == 0:
            return

        self._copy_into_actual_blocks(save_plan.physical_slot_ids, block_data)
        self._slot_state.index_put_(
            (save_plan.physical_slot_ids,),
            torch.full_like(save_plan.physical_slot_ids, STATE_RESIDENT),
        )
        self._pin_count.index_put_((save_plan.physical_slot_ids,), save_plan.final_pin_counts)
        self._touch_reusable_slots(save_plan.physical_slot_ids[save_plan.final_pin_counts == 0])

    def _rollback_save(self, save_plan: _SavePlan) -> None:
        self._restore_snapshot(save_plan.snapshot)

    def _commit_load(self, load_plan: _LoadPlan) -> None:
        if load_plan.logical_block_ids.numel() == 0:
            return

        self._slot_state.index_put_(
            (load_plan.physical_slot_ids,),
            torch.full_like(load_plan.physical_slot_ids, STATE_RESIDENT),
        )
        self._pin_count.index_put_(
            (load_plan.physical_slot_ids,),
            torch.ones_like(load_plan.physical_slot_ids),
        )

    def _rollback_load(self, hit_slot_ids: torch.Tensor, load_plan: _LoadPlan) -> None:
        if hit_slot_ids.numel() > 0:
            updated_pin_counts = self._pin_count.index_select(0, hit_slot_ids) - 1
            self._pin_count.index_put_((hit_slot_ids,), updated_pin_counts)
            self._touch_reusable_slots(hit_slot_ids[updated_pin_counts == 0])
        self._restore_snapshot(load_plan.snapshot)

    def _restore_snapshot(self, snapshot: _ReservationSnapshot) -> None:
        if snapshot.logical_block_ids.numel() == 0:
            return

        if snapshot.restore_slot_ids.numel() > 0:
            self._copy_into_actual_blocks(snapshot.restore_slot_ids, snapshot.restore_payloads)

        self._logical_to_physical.index_put_(
            (snapshot.logical_block_ids,),
            torch.full_like(snapshot.logical_block_ids, -1),
        )
        self._physical_to_logical.index_put_((snapshot.physical_slot_ids,), snapshot.previous_logical_ids)
        self._slot_state.index_put_((snapshot.physical_slot_ids,), snapshot.previous_states)
        self._pin_count.index_put_((snapshot.physical_slot_ids,), snapshot.previous_pin_counts)

        restore_mask = snapshot.previous_logical_ids >= 0
        if torch.any(restore_mask):
            self._logical_to_physical.index_put_(
                (snapshot.previous_logical_ids[restore_mask],),
                snapshot.physical_slot_ids[restore_mask],
            )

        reusable_mask = (
            ((snapshot.previous_states == STATE_FREE) | (snapshot.previous_states == STATE_RESIDENT))
            & (snapshot.previous_pin_counts == 0)
        )
        self._touch_reusable_slots(snapshot.physical_slot_ids[reusable_mask])
        self._remove_reusable_slots(snapshot.physical_slot_ids[~reusable_mask])

    def _copy_into_actual_blocks(self, physical_slot_ids: torch.Tensor, block_data: torch.Tensor) -> None:
        if physical_slot_ids.numel() == 0:
            return
        self.actual_blocks.index_copy_(
            0,
            physical_slot_ids.to(device=self.actual_blocks.device),
            block_data.to(device=self.actual_blocks.device, dtype=self.actual_blocks.dtype),
        )

    def _pop_reusable_slots(self, count: int) -> torch.Tensor:
        if count == 0:
            return self._logical_to_physical[:0]
        if len(self._reusable_slots) < count:
            raise InsufficientCapacityError("No reusable actual block is available; all resident blocks are pinned")

        slot_ids = [self._reusable_slots.popitem(last=False)[0] for _ in range(count)]
        return torch.tensor(slot_ids, dtype=ID_DTYPE)

    def _touch_reusable_slots(self, physical_slot_ids: torch.Tensor) -> None:
        for physical_slot_id in physical_slot_ids.tolist():
            self._reusable_slots.pop(physical_slot_id, None)
            self._reusable_slots[physical_slot_id] = None

    def _remove_reusable_slots(self, physical_slot_ids: torch.Tensor) -> None:
        for physical_slot_id in physical_slot_ids.tolist():
            self._reusable_slots.pop(physical_slot_id, None)

    def _validate_logical_block_ids(self, logical_block_ids: torch.Tensor) -> None:
        if logical_block_ids.numel() == 0:
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
                present_mask.detach().clone()
                if present_mask is not None
                else torch.ones((num_logical_blocks,), dtype=torch.bool)
            )
        else:
            if num_logical_blocks is None:
                if initial_data:
                    num_logical_blocks = max(initial_data) + 1
                else:
                    raise ValueError("num_logical_blocks is required when initial_data is empty")
            self._storage = None
            self._present_mask = (
                present_mask.detach().clone()
                if present_mask is not None
                else torch.zeros((num_logical_blocks,), dtype=torch.bool)
            )
            if initial_data:
                first_payload = next(iter(initial_data.values()))
                self._storage = first_payload.new_zeros((num_logical_blocks, *first_payload.shape))
                initial_ids = torch.tensor(list(initial_data.keys()), dtype=ID_DTYPE)
                initial_payloads = torch.stack([initial_data[key] for key in initial_data.keys()], dim=0)
                self._storage.index_copy_(0, initial_ids, initial_payloads)
                self._present_mask.index_put_((initial_ids,), torch.ones_like(initial_ids, dtype=torch.bool))

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
            self._storage.index_copy_(
                0,
                logical_block_ids.to(device=self._storage.device),
                block_data.to(device=self._storage.device, dtype=self._storage.dtype),
            )
            self._present_mask.index_put_(
                (logical_block_ids,),
                torch.ones_like(logical_block_ids, dtype=torch.bool),
            )
            logical_ids_list = logical_block_ids.tolist()
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
            hit_mask = self._present_mask.index_select(0, logical_block_ids)
            if not torch.all(hit_mask):
                raise BlockNotFoundError("some logical blocks are not in backend")
            logical_ids_list = logical_block_ids.tolist()
            self.load_calls.extend(logical_ids_list)
            self.operation_log.extend(("load", logical_block_id) for logical_block_id in logical_ids_list)
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
        finally:
            self._leave_load()

    def snapshot(self) -> dict[int, torch.Tensor]:
        if self._storage is None:
            return {}
        present_ids = torch.nonzero(self._present_mask, as_tuple=False).reshape(-1)
        present_payloads = self._storage.index_select(0, present_ids.to(device=self._storage.device))
        return {
            logical_block_id: payload.detach().clone()
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

        payloads_cpu = block_data.detach().to(device="cpu").contiguous()
        keys = self._make_keys(logical_block_ids)
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

        memory_objs = self._backend.batched_get_blocking(self._make_keys(logical_block_ids))
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


def _require_id_tensor(values: torch.Tensor, *, name: str) -> None:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if values.dtype != ID_DTYPE:
        raise TypeError(f"{name} must have dtype torch.int64")
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape (k,)")
    if values.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")


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
