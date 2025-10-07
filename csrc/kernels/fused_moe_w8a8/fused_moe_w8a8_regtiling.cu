
#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

template <int BM, int BK, int BN>
__global__ void fused_moe_w8a8_regtiling_kernel(
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

    uint32_t tile_x[4];
    uint32_t tile_w[2];
    float f_acc[4] = {0.f};
    // bool p = blockIdx.x == 1 && blockIdx.y == 5 && threadIdx.x == 0;

    for (int block=0; block < K/block_shape[0]; block += 1)
    {
        const int scale_cols_x = K/block_shape[1];
        const int scale_rows_w = N/block_shape[1];
        const int scale_cols_w = K/block_shape[0];

        float scale_x[2];
        if (token_src[0] < M)
        {
            scale_x[0] = x_scale[(token_src[0])*scale_cols_x + block];
        }
        if (token_src[1] < M)
        {
            scale_x[1] = x_scale[(token_src[1])*scale_cols_x + block];
        }

        float scale_w = w_scale[exp_idx * scale_rows_w * scale_cols_w + (w_row/block_shape[1])*scale_cols_w + block];

        int b_off = block * block_shape[0];
        float acc[4] = {0.f};
        for(int k = 0; k < block_shape[0]; k += BK)
        {
            if (token_src[0] < M)
            {
                tile_x[0] = reinterpret_cast<const uint32_t*>(x + token_src[0]*K + k + b_off)[lane_id%4];
                tile_x[2] = reinterpret_cast<const uint32_t*>(x + token_src[0]*K + k + b_off + 16)[lane_id%4];
            }
            if (token_src[1] < M)
            {
                tile_x[1] = reinterpret_cast<const uint32_t*>(x + token_src[1]*K + k + b_off)[lane_id%4];
                tile_x[3] = reinterpret_cast<const uint32_t*>(x + token_src[1]*K + k + b_off + 16)[lane_id%4];
            }

            const int w_col = (lane_id%4)*4 + k + b_off;
            tile_w[0] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col);
            tile_w[1] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col + 16);
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                    : "r"(tile_x[0]), "r"(tile_x[1]), "r"(tile_x[2]), "r"(tile_x[3]), "r"(tile_w[0]), "r"(tile_w[1]));

            // fp8 tmp[4];
            // for (int i = 0; i<4; i++)
            // {
            //     tmp[i] = exp_w[w_row*K + w_col];
            //     // if(p)
            //     //     printf("loading w row %d, w col %d, %f, %f \n", w_row, w_col, float(tmp[i]), float(tmp[i]) * scale_w);
            // }
            // fp8 tmp2[4];
            // for (int i = 0; i<4; i++)
            // {
            //     const int w_col = (lane_id%4)*4 + i + k + b_off + 16;
            //     tmp2[i] = exp_w[w_row*K + w_col];
            // }
            // tile_w[1] = *reinterpret_cast<uint32_t*>(&tmp2);

            // float x_dq[8];
            // float w_dq[8];
            // for (int i = 0; i < 4; i++)
            // {
            //     x_dq[i] = float(reinterpret_cast<fp8*>(&tile_x[0])[i]) * scale_x[0];
            //     x_dq[4 + i] = float(reinterpret_cast<fp8*>(&tile_x[2])[i]) * scale_x[0];
            // }
            // for (int i = 0; i < 4; i++)
            // {
            //     w_dq[i] = float(tmp[i]) * scale_w;
            //     w_dq[i+4] = float(tmp2[i]) * scale_w;
            // }
            // if(p)
            //     printf("M %d, K %d, N %d, mma %d, %d with %f,%f,%f,%f ||| %f, %f, %f, %f , w %f,%f,%f,%f ||| %f, %f, %f, %f acc %f, %f, %f, %f, scale x %f,%f scale_w %f\n",
            //             M, K, N,
            //             k,
            //             token_src[0],
            //             x_dq[0],
            //             x_dq[1],
            //             x_dq[2],
            //             x_dq[3],
            //             x_dq[4],
            //             x_dq[5],
            //             x_dq[6],
            //             x_dq[7],
            //             w_dq[0],
            //             w_dq[1],
            //             w_dq[2],
            //             w_dq[3],
            //             float(tmp[0]),
            //             float(tmp[1]),
            //             float(tmp[2]),
            //             float(tmp[3]),
            //             acc[0],
            //             acc[1],
            //             acc[2],
            //             acc[3],
            //             scale_x[0],
            //             scale_x[1],
            //             scale_w
            //             );
        }
        f_acc[0] += scale_x[0] * scale_w * acc[0];
        f_acc[1] += scale_x[0] * scale_w * acc[1];
        f_acc[2] += scale_x[1] * scale_w * acc[2];
        f_acc[3] += scale_x[1] * scale_w * acc[3];
    }
    if (token_src[0] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[0], f_acc[1]);;
        // out[token_dest[0]*N + warpN * BN + (lane_id%4)*2] = f_acc[0];
        // out[token_dest[0]*N + warpN * BN + (lane_id%4)*2 + 1] = f_acc[1];
    }
    if (token_src[1] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[2], f_acc[3]);;
        // out[token_dest[1]*N + warpN * BN + (lane_id%4)*2] = f_acc[2];
        // out[token_dest[1]*N + warpN * BN + (lane_id%4)*2 + 1] = f_acc[3];
    }
    // if(p)
    //     printf("finished with src %d/%d, dest %d/%d, off %d, exp %d, exp_off %d, %f,%f,%f,%f\n", token_src[0], token_src[1],
    //             token_dest[0], token_dest[1],
    //             blockIdx.y*BM + (lane_id>>2),
    //             exp_idx, exp_idx * K * N,
    //             f_acc[0],
    //             f_acc[1],
    //             f_acc[2],
    //             f_acc[3]);
}

void fused_moe_w8a8_regtiling(
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
    constexpr int num_warps_x = 4;
    constexpr int num_warps_y = 2;
    dim3 dimBlock(32*num_warps_x, num_warps_y, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*num_warps_x)), std::ceil((float)sorted_num/(BM*num_warps_y)), 1);
    fused_moe_w8a8_regtiling_kernel<BM, BK, BN><<<dimGrid, dimBlock>>>(
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
