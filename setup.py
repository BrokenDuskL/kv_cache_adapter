from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="kv_cache_adapter_cuda",
    ext_modules=[
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
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            no_python_abi_suffix=True,
            use_ninja=True,
        ),
    },
)
