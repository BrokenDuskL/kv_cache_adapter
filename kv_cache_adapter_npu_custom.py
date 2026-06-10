from __future__ import annotations

import importlib


_EXPORTS = (
    "inspect_load_requests",
    "inspect_save_requests",
    "pop_reusable_slots",
    "commit_load_metadata",
    "commit_save_metadata",
    "release_metadata",
)


def _load_ops_module():
    try:
        module = importlib.import_module("kv_cache_adapter_npu_custom_ops")
    except Exception as exc:  # pragma: no cover - depends on Ascend runtime
        raise ImportError("kv_cache_adapter_npu_custom_ops is not available") from exc
    missing_exports = [name for name in _EXPORTS if not hasattr(module, name)]
    if missing_exports:  # pragma: no cover - depends on Ascend runtime
        raise ImportError(
            "kv_cache_adapter_npu_custom_ops is missing exports: " + ", ".join(missing_exports),
        )
    return module, ""


_c_ops, _prefix = _load_ops_module()


def _dispatch(name: str, *args):
    return getattr(_c_ops, f"{_prefix}{name}")(*args)


def inspect_load_requests(*args):
    return _dispatch("inspect_load_requests", *args)


def inspect_save_requests(*args):
    return _dispatch("inspect_save_requests", *args)


def pop_reusable_slots(*args):
    return _dispatch("pop_reusable_slots", *args)


def commit_load_metadata(*args):
    return _dispatch("commit_load_metadata", *args)


def commit_save_metadata(*args):
    return _dispatch("commit_save_metadata", *args)


def release_metadata(*args):
    return _dispatch("release_metadata", *args)
