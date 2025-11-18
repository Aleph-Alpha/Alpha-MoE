#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

using fp8 = __nv_fp8_e4m3;

void fused_moe_w8a8_wgmma_up_down_acc(
        const fp8* x,
        const float* x_scale,
        fp8* w, const float* w_scale,
        fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const float* topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        int num_experts,
        int sorted_num,
        int block_m,
        int block_n,
        int warp_n,
        int stages,
        float scaling_factor,
        cudaStream_t stream
);

// CUDA implementation
void fused_moe_w8a8_up_down(
        const torch::Tensor& x,
        const torch::Tensor& x_scale,
        const torch::Tensor& w,
        const torch::Tensor& w_scale,
        const torch::Tensor& w2,
        const torch::Tensor& w2_scale,
        const torch::Tensor& sorted_token_ids,
        const torch::Tensor& expert_ids,
        const torch::Tensor& num_tokens_post_padded,
        const torch::Tensor& topk_weights,
        torch::Tensor& out,
        int64_t top_k,
        int64_t block_m,
        int64_t block_n,
        int64_t warp_n,
        int64_t stages,
        double scaling_factor
) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    TORCH_CHECK(x.size(0));
    TORCH_CHECK(x.size(1) == w.size(2));
    TORCH_CHECK(x.size(1) % 128 == 0, "K dimension must be a multiple of 128");
    TORCH_CHECK((block_n == 64 && warp_n == 4) || (block_n == 32 && warp_n == 8));
    TORCH_CHECK(stages > 0 && stages < 6);
    TORCH_CHECK(block_m > 0 && block_m <= 128 && block_m%8 == 0);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(w.is_contiguous());
    TORCH_CHECK(w2.is_contiguous());

    TORCH_CHECK(x_scale.is_contiguous());
    TORCH_CHECK(w_scale.is_contiguous());
    TORCH_CHECK(w2_scale.is_contiguous());

    TORCH_CHECK(sorted_token_ids.is_contiguous());
    TORCH_CHECK(expert_ids.is_contiguous());
    TORCH_CHECK(topk_weights.is_contiguous());

    fused_moe_w8a8_wgmma_up_down_acc(
            static_cast<__nv_fp8_e4m3*>(x.data_ptr()),
            static_cast<float*>(x_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w.data_ptr()),
            static_cast<float*>(w_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w2.data_ptr()),
            static_cast<float*>(w2_scale.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int*>(sorted_token_ids.data_ptr()),
            static_cast<int*>(expert_ids.data_ptr()),
            static_cast<int*>(num_tokens_post_padded.data_ptr()),
            static_cast<float*>(topk_weights.data_ptr()),
            static_cast<int>(top_k),
            x.size(0),
            x.size(1),
            w.size(1),
            w.size(0),
            sorted_token_ids.size(0),
            static_cast<int>(block_m),
            static_cast<int>(block_n),
            static_cast<int>(warp_n),
            static_cast<int>(stages),
            static_cast<float>(scaling_factor),
            stream
    );
}
// Register the custom operator
TORCH_LIBRARY_FRAGMENT(alpha_moe, m) {
    m.def("fused_moe_w8a8_up_down("
            "Tensor x, "
            "Tensor x_scale, "
            "Tensor w, "
            "Tensor w_scale, "
            "Tensor w2, "
            "Tensor w2_scale, "
            "Tensor sorted_token_ids, "
            "Tensor expert_ids, "
            "Tensor num_tokens_post_padded, "
            "Tensor topk_weights, "
            "Tensor(a!) out, "
            "int top_k, "
            "int block_m, "
            "int block_n, "
            "int warp_n, "
            "int stages, "
            "float scaling_factor"
            ") -> ()"
         );
    m.impl("fused_moe_w8a8_up_down", torch::kCUDA, &fused_moe_w8a8_up_down);
}

PYBIND11_MODULE(alpha_moe, m) {
  // Empty module - operators are registered via TORCH_LIBRARY
}
