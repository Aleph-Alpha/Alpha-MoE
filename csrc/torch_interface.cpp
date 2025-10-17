#include <pybind11/functional.h>
#include <torch/python.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

using fp8 = __nv_fp8_e4m3;

#define MOE_ARGS const __nv_fp8_e4m3* x,\
        const float* x_scale,\
        const __nv_fp8_e4m3* w,\
        const float* w_scale,\
        __nv_bfloat16* out,\
        const int* sorted_token_ids,\
        const int* expert_ids,\
        const int* num_tokens_post_padded,\
        const int top_k,\
        int M,\
        int K,\
        int N,\
        int sorted_num

#define MOE_CALL static_cast<__nv_fp8_e4m3*>(x.data_ptr()), \
            static_cast<float*>(x_scale.data_ptr()), \
            static_cast<__nv_fp8_e4m3*>(w.data_ptr()), \
            static_cast<float*>(w_scale.data_ptr()), \
            static_cast<__nv_bfloat16*>(out.data_ptr()), \
            static_cast<int*>(sorted_token_ids.data_ptr()), \
            static_cast<int*>(expert_ids.data_ptr()), \
            static_cast<int*>(num_tokens_post_padded.data_ptr()), \
            top_k, \
            x.size(0), \
            x.size(1), \
            w.size(1), \
            sorted_token_ids.size(0)

void fused_moe_w8a8(MOE_ARGS);
void fused_moe_w8a8_regtiling(MOE_ARGS);
void fused_moe_w8a8_prefetching(MOE_ARGS);
void fused_moe_w8a8_smem(MOE_ARGS);
void fused_moe_w8a8_db(MOE_ARGS);
void fused_moe_w8a8_tb(MOE_ARGS);
void fused_moe_w8a8_mb(MOE_ARGS);
void fused_moe_w8a8_sacc(MOE_ARGS);
void fused_moe_w8a8_pc(MOE_ARGS);
void fused_moe_w8a8_ast(MOE_ARGS);
void fused_moe_w8a8_wgmma(MOE_ARGS, int block_m);
void fused_moe_w8a8_wgmma_tma(const __nv_fp8_e4m3* x,
        const float* x_scale,
        __nv_fp8_e4m3* w,
        const float* w_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int top_k,
        int M, int K, int N,
        int sorted_num, int block_m);
void fused_moe_w8a8_wgmma_swiglu(MOE_ARGS, int block_m);
void fused_moe_w8a8_wgmma_tma_swiglu(const __nv_fp8_e4m3* x,
        const float* x_scale,
        __nv_fp8_e4m3* w,
        const float* w_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int top_k,
        int M, int K, int N,
        int sorted_num, int block_m);

void fused_moe_w8a8_wgmma_up_down(const fp8* x,
        const float* x_scale,
        const fp8* w, const float* w_scale,
        const fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const __nv_bfloat16* topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        int sorted_num,
        int block_m
);

void fused_moe_w8a8_wgmma_up_down_ast(const fp8* x,
        const float* x_scale,
        const fp8* w, const float* w_scale,
        const fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const __nv_bfloat16* topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        int sorted_num,
        int block_m
);

void fused_moe_w8a8_wgmma_up_down_tma(const fp8* x,
        const float* x_scale,
        fp8* w, const float* w_scale,
        fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const __nv_bfloat16* topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        int sorted_num,
        int block_m
);

void fused_moe_w8a8_wgmma_up_down_acc(const fp8* x,
        const float* x_scale,
        fp8* w, const float* w_scale,
        fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const __nv_bfloat16* topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        int sorted_num,
        int block_m,
        float scaling_factor
);

torch::Tensor fused_moe_launcher(
        torch::Tensor& x,
        torch::Tensor& x_scale,
        torch::Tensor& w,
        torch::Tensor& w_scale,
        torch::Tensor& sorted_token_ids,
        torch::Tensor& expert_ids,
        torch::Tensor& num_tokens_post_padded,
        int top_k,
        int kernel_variant,
        int BM
        )
{
    auto options = torch::TensorOptions().dtype(at::ScalarType::BFloat16).device(w.device());
    torch::Tensor out = torch::empty({x.size(0) * top_k, w.size(1)}, options);
    switch (kernel_variant)
    {
        case 0:
            fused_moe_w8a8(MOE_CALL);
            break;
        case 1:
            fused_moe_w8a8_prefetching(MOE_CALL);
            break;
        case 2:
            fused_moe_w8a8_smem(MOE_CALL);
            break;
        case 3:
            fused_moe_w8a8_db(MOE_CALL);
            break;
        case 4:
            fused_moe_w8a8_tb(MOE_CALL);
            break;
        case 5:
            fused_moe_w8a8_mb(MOE_CALL);
            break;
        case 6:
            fused_moe_w8a8_sacc(MOE_CALL);
            break;
        case 7:
            fused_moe_w8a8_pc(MOE_CALL);
            break;
        case 8:
            fused_moe_w8a8_ast(MOE_CALL);
            break;
        case 9:
            fused_moe_w8a8_wgmma(MOE_CALL, BM);
            break;
        case 10:
            fused_moe_w8a8_wgmma_tma(MOE_CALL, BM);
            break;
        case 11:
            out = torch::empty({x.size(0) * top_k, w.size(1)/2}, options);
            fused_moe_w8a8_wgmma_swiglu(MOE_CALL, BM);
            break;
        case 12:
            out = torch::empty({x.size(0) * top_k, w.size(1)/2}, options);
            fused_moe_w8a8_wgmma_tma_swiglu(MOE_CALL, BM);
            break;
    }
    return out;
}

torch::Tensor fused_moe_launcher_up_down(
        torch::Tensor& x,
        torch::Tensor& x_scale,
        torch::Tensor& w,
        torch::Tensor& w_scale,
        torch::Tensor& w2,
        torch::Tensor& w2_scale,
        torch::Tensor& sorted_token_ids,
        torch::Tensor& expert_ids,
        torch::Tensor& num_tokens_post_padded,
        torch::Tensor& topk_weights,
        int top_k,
        int kernel_variant,
        int BM,
        float scaling_factor
        )
{
    auto options = torch::TensorOptions().dtype(at::ScalarType::BFloat16).device(w.device());
    torch::Tensor out = torch::zeros({x.size(0)*top_k, x.size(1)}, options);
    switch (kernel_variant)
    {
        case 0:
            fused_moe_w8a8_wgmma_up_down(static_cast<__nv_fp8_e4m3*>(x.data_ptr()),
            static_cast<float*>(x_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w.data_ptr()),
            static_cast<float*>(w_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w2.data_ptr()),
            static_cast<float*>(w2_scale.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int*>(sorted_token_ids.data_ptr()),
            static_cast<int*>(expert_ids.data_ptr()),
            static_cast<int*>(num_tokens_post_padded.data_ptr()),
            static_cast<__nv_bfloat16*>(topk_weights.data_ptr()),
            top_k,
            x.size(0),
            x.size(1),
            w.size(1),
            sorted_token_ids.size(0),
            BM);
            break;
        case 1:
            fused_moe_w8a8_wgmma_up_down_ast(static_cast<__nv_fp8_e4m3*>(x.data_ptr()),
            static_cast<float*>(x_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w.data_ptr()),
            static_cast<float*>(w_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w2.data_ptr()),
            static_cast<float*>(w2_scale.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int*>(sorted_token_ids.data_ptr()),
            static_cast<int*>(expert_ids.data_ptr()),
            static_cast<int*>(num_tokens_post_padded.data_ptr()),
            static_cast<__nv_bfloat16*>(topk_weights.data_ptr()),
            top_k,
            x.size(0),
            x.size(1),
            w.size(1),
            sorted_token_ids.size(0),
            BM);
            break;
        case 2:
            fused_moe_w8a8_wgmma_up_down_tma(static_cast<__nv_fp8_e4m3*>(x.data_ptr()),
            static_cast<float*>(x_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w.data_ptr()),
            static_cast<float*>(w_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w2.data_ptr()),
            static_cast<float*>(w2_scale.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int*>(sorted_token_ids.data_ptr()),
            static_cast<int*>(expert_ids.data_ptr()),
            static_cast<int*>(num_tokens_post_padded.data_ptr()),
            static_cast<__nv_bfloat16*>(topk_weights.data_ptr()),
            top_k,
            x.size(0),
            x.size(1),
            w.size(1),
            sorted_token_ids.size(0),
            BM);
            break;
        case 3:
            out = torch::zeros({x.size(0), x.size(1)}, options);
            fused_moe_w8a8_wgmma_up_down_acc(static_cast<__nv_fp8_e4m3*>(x.data_ptr()),
            static_cast<float*>(x_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w.data_ptr()),
            static_cast<float*>(w_scale.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(w2.data_ptr()),
            static_cast<float*>(w2_scale.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<int*>(sorted_token_ids.data_ptr()),
            static_cast<int*>(expert_ids.data_ptr()),
            static_cast<int*>(num_tokens_post_padded.data_ptr()),
            static_cast<__nv_bfloat16*>(topk_weights.data_ptr()),
            top_k,
            x.size(0),
            x.size(1),
            w.size(1),
            sorted_token_ids.size(0),
            BM,
            scaling_factor
            );
            break;
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_moe_w8a8", &fused_moe_launcher);
    m.def("fused_moe_w8a8_up_down", &fused_moe_launcher_up_down);
}
