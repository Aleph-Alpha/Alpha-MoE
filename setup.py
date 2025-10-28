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
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_prefetching.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_smem.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_db.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_tb.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_mb.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_sacc.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_pc.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_ast.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_wgmma.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_wgmma_tma.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_wgmma_swiglu.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_wgmma_tma_swiglu.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_mow_w8a8_up_down.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_up_down_ast.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_up_down_tma.cu",
               "./csrc/kernels/fused_moe_w8a8/fused_moe_w8a8_up_down_acc.cu",
               ]

    setuptools.setup(
        name='alpha-kernel-python',
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

