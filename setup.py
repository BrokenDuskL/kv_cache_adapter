import configparser
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


ROOT_DIR = Path(__file__).resolve().parent
_TRUE_VALUES = {"1", "true", "yes", "on"}
KVCA_SLOT_META_BITS = os.getenv("KVCA_SLOT_META_BITS", "8").strip() or "8"


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def _get_ascend_home_path() -> str:
    return os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")


def _get_ascend_env_path() -> str:
    ascend_home = Path(_get_ascend_home_path())
    candidates = (
        ascend_home / "set_env.sh",
        ascend_home.parent / "set_env.sh",
    )
    for path in candidates:
        if path.exists():
            return str(path)
    raise ValueError(
        "Unable to locate Ascend set_env.sh; set ASCEND_HOME_PATH to the CANN toolkit root.",
    )


def _get_npu_soc() -> str:
    env_soc = os.getenv("SOC_VERSION", "").strip()
    if env_soc:
        return env_soc

    for npu_id in range(8):
        try:
            output = subprocess.check_output(
                [
                    "npu-smi",
                    "info",
                    "-t",
                    "board",
                    "-i",
                    str(npu_id),
                    "-c",
                    "0",
                ],
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

        info: dict[str, str] = {}
        for line in output.strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
        chip_name = info.get("Chip Name")
        if not chip_name:
            continue
        npu_name = info.get("NPU Name")
        if npu_name:
            return f"{chip_name}_{npu_name}"
        return chip_name if chip_name.startswith("Ascend") else f"Ascend{chip_name}"

    raise RuntimeError("Unable to determine SOC_VERSION; set SOC_VERSION or make npu-smi available.")


def _get_aicore_arch_number(ascend_path: str, soc_version: str, host_arch: str) -> str | None:
    ini_path = Path(ascend_path) / f"{host_arch}-linux" / "data" / "platform_config" / f"{soc_version}.ini"
    if not ini_path.exists():
        return None
    config = configparser.ConfigParser()
    config.read(ini_path)
    aic_version = config.get("version", "AIC_version", fallback="")
    if not aic_version:
        return None
    return aic_version.split("-")[-1]


def _should_build_ascend_extension() -> bool:
    return _env_enabled("BUILD_KV_CACHE_ADAPTER_ASCEND") or importlib.util.find_spec("torch_npu") is not None


class CMakeExtension(Extension):
    def __init__(self, name: str, *, cmake_lists_dir: str) -> None:
        super().__init__(name=name, sources=[])
        self.cmake_lists_dir = cmake_lists_dir


_BaseBuildExtension = BuildExtension.with_options(
    no_python_abi_suffix=True,
    use_ninja=True,
)


class KVCacheAdapterBuildExtension(_BaseBuildExtension):
    def build_extension(self, ext: Extension) -> None:
        if isinstance(ext, CMakeExtension):
            self._build_cmake_extension(ext)
            return
        super().build_extension(ext)

    def _build_cmake_extension(self, ext: CMakeExtension) -> None:
        module_basename = ext.name.rsplit(".", 1)[-1]
        build_root = ROOT_DIR / "build" / module_basename
        install_root = build_root / "install"
        if build_root.exists():
            shutil.rmtree(build_root)
        build_root.mkdir(parents=True, exist_ok=True)
        install_root.mkdir(parents=True, exist_ok=True)

        pybind11_cmake_dir = subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            text=True,
        ).strip()
        python_include_path = sysconfig.get_path("include", scheme="posix_prefix")

        import torch
        import torch_npu

        ascend_home = _get_ascend_home_path()
        arch = platform.machine()
        soc_version = _get_npu_soc()
        aicore_arch = _get_aicore_arch_number(ascend_home, soc_version, arch)
        torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        torch_path = os.path.dirname(os.path.abspath(torch.__file__))
        torch_cxx11_abi = int(torch.compiled_with_cxx11_abi())
        torch_cmake_dir = os.path.join(torch.utils.cmake_prefix_path, "Torch")

        cmake_parts = [
            f". {_get_ascend_env_path()}",
            "&&",
            "cmake",
            "-S",
            ext.cmake_lists_dir,
            "-B",
            str(build_root),
            f"-DSOC_VERSION={soc_version}",
            f"-DARCH={arch}",
            "-DUSE_ASCEND=1",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={pybind11_cmake_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_root}",
            f"-DPYTHON_INCLUDE_PATH={python_include_path}",
            f"-DASCEND_CANN_PACKAGE_PATH={ascend_home}",
            f"-DTORCH_NPU_PATH={torch_npu_path}",
            f"-DTORCH_PATH={torch_path}",
            f"-DGLIBCXX_USE_CXX11_ABI={torch_cxx11_abi}",
            f"-DTorch_DIR={torch_cmake_dir}",
            f"-DKVCA_SLOT_META_BITS={KVCA_SLOT_META_BITS}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
        ]
        if aicore_arch is not None:
            cmake_parts.append(f"-DASCEND_AICORE_ARCH={aicore_arch}")
        if os.getenv("CC"):
            cmake_parts.append(f"-DCMAKE_C_COMPILER={os.environ['CC']}")
        if os.getenv("CXX"):
            cmake_parts.append(f"-DCMAKE_CXX_COMPILER={os.environ['CXX']}")
        cmake_parts.extend(
            [
                "&&",
                "cmake",
                "--build",
                str(build_root),
                "-j",
                "--verbose",
                "&&",
                "cmake",
                "--install",
                str(build_root),
            ],
        )
        subprocess.run(" ".join(cmake_parts), cwd=ROOT_DIR, shell=True, check=True, text=True)

        expected_ext_path = Path(self.get_ext_fullpath(ext.name))
        output_dir = expected_ext_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        built_modules = sorted(install_root.glob(f"{module_basename}*.so"))
        if not built_modules:
            raise RuntimeError(f"Failed to locate built extension for {ext.name} under {install_root}")
        built_extension = built_modules[0]
        shutil.copy2(built_extension, expected_ext_path)
        if self.inplace:
            shutil.copy2(built_extension, ROOT_DIR / expected_ext_path.name)

        support_lib = install_root / "lib" / "libkv_cache_adapter_npu_custom_kernels.so"
        if not support_lib.exists():
            raise RuntimeError(
                f"Failed to locate required support library {support_lib} for {ext.name}; "
                "the build must place this sidecar next to the Python extension output",
            )
        dst_path = output_dir / support_lib.name
        if os.path.abspath(support_lib) != os.path.abspath(dst_path):
            shutil.copy2(support_lib, dst_path)
        if self.inplace:
            source_dst = ROOT_DIR / support_lib.name
            if os.path.abspath(support_lib) != os.path.abspath(source_dst):
                shutil.copy2(support_lib, source_dst)


ext_modules: list[Extension] = [
    CppExtension(
        name="kv_cache_adapter.kv_cache_adapter_npu",
        sources=["csrc/kv_adapter_npu.cpp"],
        extra_compile_args=[f"-DKVCA_SLOT_META_BITS={KVCA_SLOT_META_BITS}", "-O3"],
    ),
]

if CUDA_HOME is not None and os.path.exists("csrc/kv_adapter_cuda.cu"):
    ext_modules.append(
        CUDAExtension(
            name="kv_cache_adapter.kv_cache_adapter_cuda",
            sources=[
                "csrc/binding.cpp",
                "csrc/kv_adapter_cuda.cu",
            ],
            extra_compile_args={
                "cxx": [f"-DKVCA_SLOT_META_BITS={KVCA_SLOT_META_BITS}", "-O3"],
                "nvcc": [f"-DKVCA_SLOT_META_BITS={KVCA_SLOT_META_BITS}", "-O3", "--use_fast_math"],
            },
        ),
    )

if _should_build_ascend_extension():
    ext_modules.append(
        CMakeExtension(
            name="kv_cache_adapter.kv_cache_adapter_npu_custom_ops",
            cmake_lists_dir=str(ROOT_DIR / "csrc" / "ascend"),
        ),
    )


setup(
    name="kv_cache_adapter",
    packages=["kv_cache_adapter"],
    package_dir={"kv_cache_adapter": "."},
    package_data={"kv_cache_adapter": ["*.so"]},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": KVCacheAdapterBuildExtension},
)
