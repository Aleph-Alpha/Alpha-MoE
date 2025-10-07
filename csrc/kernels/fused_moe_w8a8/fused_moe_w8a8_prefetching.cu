

#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

template <int BM, int BK, int BN, int PF>
__global__ void fused_moe_w8a8_prefetching_kernel(
        const fp8* __restrict__ x,
        const float* __restrict__ x_scale,
        const fp8* __restrict__ w,
        const float* __restrict__ w_scale,
        __nv_bfloat16* __restrict__ out,
        const int* __restrict__ sorted_token_ids,
        const int* __restrict__ expert_ids,
        const int* __restrict__ num_tokens_post_padded,
        const int top_k,
        int M,
        int K,
        int N
        )
{
    const int32_t warpN = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpM = blockIdx.y*blockDim.y+threadIdx.y;

    //TODO should not be hardcoded
    constexpr int block_shape[2] = {128, 128};

    const int exp_idx = expert_ids[warpM];
    const fp8* exp_w = w + exp_idx * K * N;
    const int lane_id = threadIdx.x%32;
    const int w_row = warpN * BN + (lane_id>>2);

    if(warpM * BM >= num_tokens_post_padded[0])
        return;

    // if(exp_idx < 0 || exp_idx >= 257)
    //     printf("INVALID IDX %d, %d, %d\n",blockIdx.y, exp_idx, num_tokens_post_padded[0]);


    int token_dest[2];
    token_dest[0] = sorted_token_ids[warpM*BM + (lane_id>>2)];
    token_dest[1] = sorted_token_ids[warpM*BM + (lane_id>>2) + 8];
    int token_src[2];
    token_src[0] = sorted_token_ids[warpM*BM + (lane_id>>2)] / top_k;
    token_src[1] = sorted_token_ids[warpM*BM + (lane_id>>2) + 8] / top_k;

    float scale_x[PF][2];
    float scale_w[PF];
    const int scale_cols_x = K/block_shape[1];
    const int scale_rows_w = N/block_shape[1];
    const int scale_cols_w = K/block_shape[0];

    uint32_t tile_x[PF][4];
    uint32_t tile_w[PF][2];
    float f_acc[4] = {0.f};
    int tc_stage=0;
    int scale_stage=0;
    // bool p = blockIdx.x == 1 && blockIdx.y == 5 && threadIdx.x == 0;
    auto load_tiles = [&](int off, int stage)
    {
            if (token_src[0] < M)
            {
                tile_x[stage][0] = reinterpret_cast<const uint32_t*>(x + token_src[0]*K + off)[lane_id%4];
                tile_x[stage][2] = reinterpret_cast<const uint32_t*>(x + token_src[0]*K + off + 16)[lane_id%4];
            }
            if (token_src[1] < M)
            {
                tile_x[stage][1] = reinterpret_cast<const uint32_t*>(x + token_src[1]*K + off)[lane_id%4];
                tile_x[stage][3] = reinterpret_cast<const uint32_t*>(x + token_src[1]*K + off + 16)[lane_id%4];
            }

            const int w_col = (lane_id%4)*4 + off;
            tile_w[stage][0] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col);
            tile_w[stage][1] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col + 16);
    };

    auto load_scales = [&](int off, int stage)
    {
        if (token_src[0] < M)
        {
            scale_x[stage][0] = x_scale[(token_src[0])*scale_cols_x + off];
        }
        if (token_src[1] < M)
        {
            scale_x[stage][1] = x_scale[(token_src[1])*scale_cols_x + off];
        }
        scale_w[stage] = w_scale[exp_idx * scale_rows_w * scale_cols_w + (w_row/block_shape[1])*scale_cols_w + off];
    };

    for(int stage = 0; stage < PF && stage*BK < K; stage++)
    {
        load_tiles(stage*BK, stage);
    }

    for(int stage = 0; stage < PF && stage < K/block_shape[0]; stage++)
    {
        load_scales(stage, stage);
    }

    for (int block=0; block < K/block_shape[0]; block += 1)
    {
        int b_off = block * block_shape[0];
        float acc[4] = {0.f};
        for(int k = 0; k < block_shape[0]; k += BK)
        {
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                    : "r"(tile_x[tc_stage][0]), "r"(tile_x[tc_stage][1]), "r"(tile_x[tc_stage][2]), "r"(tile_x[tc_stage][3]), "r"(tile_w[tc_stage][0]), "r"(tile_w[tc_stage][1]));
            if(b_off + k + tc_stage*BK < K)
                load_tiles(b_off + k + PF*BK, tc_stage);
            tc_stage = (tc_stage+1)%PF;

        }
        if (token_src[0] < M)
        {
            f_acc[0] += scale_x[scale_stage][0] * scale_w[scale_stage] * acc[0];
            f_acc[1] += scale_x[scale_stage][0] * scale_w[scale_stage] * acc[1];
        }
        if (token_src[1] < M)
        {
            f_acc[2] += scale_x[scale_stage][1] * scale_w[scale_stage] * acc[2];
            f_acc[3] += scale_x[scale_stage][1] * scale_w[scale_stage] * acc[3];
        }
        if(block + PF < K/block_shape[0])
        {
            load_scales(block + PF, scale_stage);
        }
        scale_stage = (scale_stage+1)%PF;
    }
    if (token_src[0] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[0], f_acc[1]);;
    }
    if (token_src[1] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[2], f_acc[3]);;
    }
}

void fused_moe_w8a8_prefetching(
        const fp8* x,
        const float* x_scale,
        const fp8* w, const float* w_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int top_k,
        int M,
        int K,
        int N,
        int sorted_num
        )
{
    constexpr int BM = 16;
    constexpr int BK = 32;
    constexpr int BN = 8;
    constexpr int PF = 4;
    constexpr int num_warps_x = 4;
    constexpr int num_warps_y = 2;
    dim3 dimBlock(32*num_warps_x, num_warps_y, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*num_warps_x)), std::ceil((float)sorted_num/(BM*num_warps_y)), 1);
    // TODO get some JIT mechanism instead of hard coding
    if (top_k == 1)
    {
        fused_moe_w8a8_prefetching_kernel<BM, BK, BN, 2><<<dimGrid, dimBlock>>>(
                x,
                x_scale,
                w,
                w_scale,
                out,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                M,
                K,
                N
                );
    }
    else
    {
        fused_moe_w8a8_prefetching_kernel<BM, BK, BN, PF><<<dimGrid, dimBlock>>>(
                x,
                x_scale,
                w,
                w_scale,
                out,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                M,
                K,
                N
                );
    }
}
