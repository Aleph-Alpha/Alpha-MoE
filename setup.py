import os
import subprocess
import setuptools
import importlib
import importlib.resources
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
torch.utils.cpp_extension.COMMON_NVCC_FLAGS = []


if __name__ == '__main__':
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    sources = ['./csrc/torch_interface.cpp',
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_up_down_acc.cu",
               ]

    setuptools.setup(
        name='alpha_kernel_python',
        packages=setuptools.find_packages(include=["alpha_kernel_python"]),
        version='0.0.1',
        ext_modules=[
            CUDAExtension(
                name='alpha_kernel',
                sources=sources,
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )

