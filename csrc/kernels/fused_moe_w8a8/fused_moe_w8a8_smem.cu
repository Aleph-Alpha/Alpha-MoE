#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

__device__ __forceinline__ void ld_matrix_x2(uint32_t* tile, uint32_t mat)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
            : "=r"(tile[0]), "=r"(tile[1]) : "r"(mat));
}

__device__ __forceinline__ void ld_matrix_x4(uint32_t* tile, uint32_t mat)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(tile[0]), "=r"(tile[1]), "=r"(tile[2]), "=r"(tile[3]) : "r"(mat));
}

#define S_BITS 3
#define S_MASK 0b1110000

template <int BM, int BK, int BN, int PF, int WM, int WN>
__global__ void fused_moe_w8a8_smem_kernel(
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
    const int warp_id = threadIdx.x/32;
    const int w_row = warpN * BN + (lane_id>>2);

    if(warpM * BM >= num_tokens_post_padded[0])
        return;

    // if(exp_idx < 0 || exp_idx >= 257)
    //     printf("INVALID IDX %d, %d, %d\n",blockIdx.y, exp_idx, num_tokens_post_padded[0]);


    int token_dest[2];
    token_dest[0] = sorted_token_ids[warpM*BM + (lane_id>>2)];
    token_dest[1] = sorted_token_ids[warpM*BM + (lane_id>>2) + 8];

    //SMEM sizes
    constexpr int WS = WN*PF*BK*BN;
    constexpr int XS = PF*BK*BM;
    // how many bytes we transfer per CP_ASYNC
    constexpr int TB = 16;
    // Thread offset per transfer
    constexpr int TO = TB/sizeof(fp8);
    __shared__ alignas(128) fp8 s_w[WS];
    __shared__ alignas(128) fp8 s_x[XS];

    uint32_t tile_x[4];
    uint32_t tile_w[2];
    float f_acc[4] = {0.f};
    int compute_stage=0;
    // bool p = blockIdx.x == 1 && blockIdx.y == 128 && lane_id == 14;
    // bool p = blockIdx.x == 0 && blockIdx.y == 0 && lane_id == 0;
    auto load_tiles = [&](int off, int stage)
    {
            int xs_row = (lane_id>>2);
            if (token_dest[0]/top_k < M)
            {
                tile_x[0] = reinterpret_cast<const uint32_t*>(s_x + xs_row*PF*BK + stage*BK)[lane_id%4];
                tile_x[2] = reinterpret_cast<const uint32_t*>(s_x + xs_row*PF*BK + stage*BK + 16)[lane_id%4];
            }
            if (token_dest[1]/top_k < M)
            {
                xs_row += 8;
                tile_x[1] = reinterpret_cast<const uint32_t*>(s_x + xs_row*PF*BK + stage*BK)[lane_id%4];
                tile_x[3] = reinterpret_cast<const uint32_t*>(s_x + xs_row*PF*BK + stage*BK + 16)[lane_id%4];
            }
            // const int xs_col = (lane_id/16)*(BK/2) + stage*BK;
            // uint32_t sm_x = __cvta_generic_to_shared(s_x + xs_row*PF*BK + xs_col);
            // ld_matrix_x4(tile_x, sm_x);
            // if(p)
            //     printf("reading %d tile %d/%d, stage %d\n", token_dest[0], xs_row, xs_col, stage);

            const int ws_row = warp_id*BN + (lane_id%8);
            const int ws_col = (lane_id/8)*(BK/2) + stage*BK;
            int i = ws_row*PF*BK + ws_col;

            int swizzled = i^((i&(S_MASK<<S_BITS))>>S_BITS);
            // int swizzled=i;
            uint32_t sm_w = __cvta_generic_to_shared(s_w + swizzled);
            ld_matrix_x2(tile_w, sm_w);
            // if(p)
            //     printf("reading %d tile %d/%d, stage %d\n", token_dest[0], ws_row, ws_col, stage);
    };

    for (int block=0; block < K/block_shape[0]; block += 1)
    {
        const int scale_cols_x = K/block_shape[1];
        const int scale_rows_w = N/block_shape[1];
        const int scale_cols_w = K/block_shape[0];
        int b_off = block * block_shape[0];

        for(int i = (threadIdx.y*blockDim.x + threadIdx.x)*TO;
                i < WS;
                i += blockDim.x*blockDim.y*TO)
        {
            int row = blockIdx.x*WN*BN + i/(BK*PF);
            int col = b_off + i%(BK*PF);
            int swizzled = i^((i&(S_MASK<<S_BITS))>>S_BITS);
            // int swizzled=i;
            uint32_t sm = __cvta_generic_to_shared(s_w + swizzled);
            // if(p)
            //     printf("loading %d to tile %d %d, i %d\n", exp_idx, row, col, i);
            CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(exp_w + row*K + col), TB);
        }
        for(int i = (threadIdx.y*blockDim.x + threadIdx.x)*TO;
                i < XS;
                i += blockDim.x*blockDim.y*TO)
        {
            int r = i/(BK*PF);
            int tok_dst =  i/(XS/2) == 0 ? token_dest[0] : token_dest[1];
            int row = __shfl_sync(0xFFFFFFFF, tok_dst/top_k, (r*4));
            if(row < M)
            {
                int col = b_off + i%(BK*PF);
                uint32_t sm = __cvta_generic_to_shared(s_x + i);
                CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row*K + col), TB);
            }
        }
        CP_ASYNC_COMMIT_GROUP();
        float scale_x[2];
        if (token_dest[0]/top_k < M)
        {
            scale_x[0] = x_scale[(token_dest[0]/top_k)*scale_cols_x + block];
        }
        if (token_dest[1]/top_k < M)
        {
            scale_x[1] = x_scale[(token_dest[1]/top_k)*scale_cols_x + block];
        }

        float scale_w = w_scale[exp_idx * scale_rows_w * scale_cols_w + (w_row/block_shape[1])*scale_cols_w + block];

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();

        float acc[4] = {0.f};
        for(int k = 0; k < block_shape[0]; k += BK)
        {
            load_tiles(b_off + k, k/BK);
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                    : "r"(tile_x[0]), "r"(tile_x[1]), "r"(tile_x[2]), "r"(tile_x[3]), "r"(tile_w[0]), "r"(tile_w[1]));
            __syncthreads();

            // float x_dq[8];
            // float w_dq[8];
            // fp8* tmp = reinterpret_cast<fp8*>(&tile_w[0]);
            // fp8* tmp2 = reinterpret_cast<fp8*>(&tile_w[1]);
            // for (int i = 0; i < 4; i++)
            // {
            //     // x_dq[i] = float(reinterpret_cast<fp8*>(&tile_x[0])[i]) * scale_x[0];
            //     // x_dq[4 + i] = float(reinterpret_cast<fp8*>(&tile_x[2])[i]) * scale_x[0];
            //     x_dq[i] = float(reinterpret_cast<fp8*>(&tile_x[1])[i]) * scale_x[1];
            //     x_dq[4 + i] = float(reinterpret_cast<fp8*>(&tile_x[3])[i]) * scale_x[1];
            // }
            // for (int i = 0; i < 4; i++)
            // {
            //     w_dq[i] = float(tmp[i]) * scale_w;
            //     w_dq[i+4] = float(tmp2[i]) * scale_w;
            // }
            // if(p && block == 0)
            //     printf("M %d, K %d, N %d, mma %d, %d with %f,%f,%f,%f ||| %f, %f, %f, %f , w %f,%f,%f,%f ||| %f, %f, %f, %f acc %f, %f, %f, %f, scale x %f,%f scale_w %f\n",
            //             M, K, N,
            //             k,
            //             token_dest[1],
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
                        // );
        }
        if (token_dest[0]/top_k < M)
        {
            f_acc[0] += scale_x[0] * scale_w * acc[0];
            f_acc[1] += scale_x[0] * scale_w * acc[1];
        }
        if (token_dest[1]/top_k < M)
        {
            f_acc[2] += scale_x[1] * scale_w * acc[2];
            f_acc[3] += scale_x[1] * scale_w * acc[3];
        }
    }
    if (token_dest[0]/top_k < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[0], f_acc[1]);;
    }
    if (token_dest[1]/top_k < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[2], f_acc[3]);;
    }
    // if((token_dest[0] == 8 || token_dest[1] == 8) && (warpN*BN + (lane_id%4)*2) == 20)
    //     printf("finished with src %d/%d, dest %d/%d, off %d, exp %d, exp_off %d, %f,%f,%f,%f, b %d/%d, t %d/%d\n", token_dest[0], token_dest[1],
    //             token_dest[0], token_dest[1],
    //             warpN * BN + (lane_id%4)*2,
    //             exp_idx, exp_idx * K * N,
    //             f_acc[0],
    //             f_acc[1],
    //             f_acc[2],
    //             f_acc[3],
    //             blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

void fused_moe_w8a8_smem(
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
    constexpr int WN = 4;
    // TODO this will only work for num_warps_y = 1
    constexpr int WM = 1;
    dim3 dimBlock(32*WN, WM, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*WN)), std::ceil((float)sorted_num/(BM*WM)), 1);

    // CUtensorMap tensor_map{};
    // constexpr uint32_t rank = 3;
    // uint64_t size[rank] = {};

    fused_moe_w8a8_smem_kernel<BM, BK, BN, PF, WM, WN><<<dimGrid, dimBlock>>>(
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
