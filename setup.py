import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


ext_modules = [
    CppExtension(
        name="kv_cache_adapter_npu",
        sources=["csrc/kv_adapter_npu.cpp"],
        extra_compile_args=["-O3"],
    ),
]

if CUDA_HOME is not None and os.path.exists("csrc/kv_adapter_cuda.cu"):
    ext_modules.append(
        CUDAExtension(
            name="kv_cache_adapter_cuda",
            sources=[
                "csrc/binding.cpp",
                "csrc/kv_adapter_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    )


setup(
    name="kv_cache_adapter",
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension.with_options(
            no_python_abi_suffix=True,
            use_ninja=True,
        ),
    },
)
