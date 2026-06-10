from __future__ import annotations

import ctypes
import importlib
import importlib.machinery
import importlib.util
import pathlib
import sys


_EXPORTS = (
    "inspect_load_requests",
    "inspect_save_requests",
    "pop_reusable_slots",
    "commit_load_metadata",
    "commit_save_metadata",
    "release_metadata",
)


def _load_ops_module():
    module_dir = pathlib.Path(__file__).resolve().parent
    support_lib = module_dir / "libkv_cache_adapter_npu_custom_kernels.so"
    if support_lib.exists():
        ctypes.CDLL(str(support_lib), mode=ctypes.RTLD_GLOBAL)
    direct_spec = importlib.machinery.PathFinder.find_spec("kv_cache_adapter_npu_custom_ops", [str(module_dir)])
    if direct_spec is not None and direct_spec.loader is not None:
        module = importlib.util.module_from_spec(direct_spec)
        sys.modules["kv_cache_adapter_npu_custom_ops"] = module
        if __package__:
            sys.modules.setdefault(f"{__package__}.kv_cache_adapter_npu_custom_ops", module)
        direct_spec.loader.exec_module(module)
    elif __package__:
        qualified_name = f"{__package__}.kv_cache_adapter_npu_custom_ops"
        module = importlib.import_module(qualified_name)
    else:
        raise ImportError(
            f"kv_cache_adapter_npu_custom_ops not found next to {__file__}; rebuild with "
            "python setup.py build_ext --inplace or install the package",
        )
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
