#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG4(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

__device__ static __forceinline__ void init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count+transaction_count)
    );
}

__device__ static __forceinline__ void cp_async_mbarrier_arrive(uint64_t* bar) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n"
        :: "r"(bar_ptr)
    );
}

__device__ static __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    // asm volatile (
    //     "{\n"
    //     ".reg .pred                P1;\n"
    //     "LAB_WAIT:\n"
    //     "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n"
    //     "@P1                       bra.uni DONE;\n"
    //     "bra.uni                   LAB_WAIT;\n"
    //     "DONE:\n"
    //     "}\n"
    //     :: "r"(mbar_ptr),
    //     "r"(kPhaseBit)
    // );
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
        "@!P1                       bra.uni WAIT;\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ static __forceinline__ void arrive(uint64_t* bar, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.arrive.shared.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

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

template <int BM, int BK, int BN, int PF, int WM, int WN, int STAGES>
__global__ __launch_bounds__(128) void fused_moe_w8a8_mb_kernel(
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

    __shared__ uint64_t bar[STAGES];
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < STAGES; i++)
        {
            init_barrier(&bar[i], blockDim.x, 0);
        }
    }
    __syncthreads();


    // if(exp_idx < 0 || exp_idx >= 257)
    //     printf("INVALID IDX %d, %d, %d\n",blockIdx.y, exp_idx, num_tokens_post_padded[0]);


    int token_dest[2];
    token_dest[0] = sorted_token_ids[warpM*BM + (lane_id>>2)];
    token_dest[1] = sorted_token_ids[warpM*BM + (lane_id>>2) + 8];

    int token_src[2];

    //SMEM sizes
    constexpr int WS = WN*PF*BK*BN;
    constexpr int XS = PF*BK*BM;
    // how many bytes we transfer per CP_ASYNC
    constexpr int TB = 16;
    // Thread offset per transfer
    constexpr int TO = TB/sizeof(fp8);
    __shared__ alignas(128) fp8 s_w[STAGES*WS];
    __shared__ alignas(128) fp8 s_x[STAGES*XS];

    // N tiles in memory
    constexpr int TN = BN/8;
    constexpr int STG = block_shape[0]/32;

    uint32_t tile_x[STG][4];
    uint32_t tile_w[TN][STG/2][4];
    float f_acc[TN][4] = {0.f};
    int compute_stage = 0;
    int load_stage = 0;
    int n_stages = K/block_shape[0];

    auto load_tiles_x = [&](int stage)
    {
        const int smem_stage = compute_stage%STAGES;
        int xs_row = (lane_id%16);
        const int xs_col = (lane_id/16)*(BK/2) + stage*BK;

        int i = xs_row*PF*BK + xs_col;
        int swizzled = i^((i&(S_MASK<<S_BITS))>>S_BITS);

        uint32_t sm_x = __cvta_generic_to_shared(s_x + smem_stage*XS + swizzled);
        ld_matrix_x4(tile_x[stage], sm_x);
    };

    auto load_tiles_w = [&](int stage, int w_off)
    {
        const int ws_row = warp_id*BN + (lane_id%8) + w_off*8;
        const int ws_col = (lane_id/8)*(BK/2) + stage*BK;
        const int smem_stage = compute_stage%STAGES;

        int i = ws_row*PF*BK + ws_col;
        int swizzled = i^((i&(S_MASK<<S_BITS))>>S_BITS);
        uint32_t sm_w = __cvta_generic_to_shared(s_w + smem_stage*WS + swizzled);
        ld_matrix_x4(tile_w[w_off][stage/2], sm_w);
    };


    auto async_load = [&]()
    {
        const int off = load_stage * block_shape[0];
        int smem_stage = load_stage%STAGES;
        for(int i = (threadIdx.y*blockDim.x + threadIdx.x)*TO;
                i < WS;
                i += blockDim.x*blockDim.y*TO)
        {
            int row = blockIdx.x*WN*BN + i/(BK*PF);
            int col = off + i%(BK*PF);
            int swizzled = i^((i&(S_MASK<<S_BITS))>>S_BITS);
            uint32_t sm = __cvta_generic_to_shared(s_w + smem_stage*WS + swizzled);
            CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(exp_w + row*K + col), TB);
        }

        for(int i = (threadIdx.y*blockDim.x + threadIdx.x)*TO;
                i < XS;
                i += blockDim.x*blockDim.y*TO)
        {
            int r = i/(BK*PF);
            int tok_src = i/(XS/2) == 0 ? token_src[0] : token_src[1];
            int row = __shfl_sync(0xFFFFFFFF, tok_src, (r*4));
            if(row < M)
            {
                int swizzled = i^((i&(S_MASK<<S_BITS))>>S_BITS);
                int col = off + i%(BK*PF);
                uint32_t sm = __cvta_generic_to_shared(s_x + smem_stage*XS + swizzled);
                CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row*K + col), TB);
            }
        }
        cp_async_mbarrier_arrive(bar + smem_stage);
        load_stage++;
    };

    token_src[0] = token_dest[0]/top_k;
    token_src[1] = token_dest[1]/top_k;
    for (int i = 0; i<STAGES; i++)
    {
        async_load();
    }
    int p = 0;

    for (int block=0; block < n_stages; block += 1)
    {
        compute_stage = block;

        wait(bar + (compute_stage%STAGES), p);
        if ((compute_stage%STAGES) == STAGES-1)
            p^=1;

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

        float tile_acc[TN][4] = {0.f};
        for(int w_off = 0; w_off<TN; w_off++)
        {
            for (int s = 0; s<STG; s+=2)
            {
                load_tiles_x(s);
                load_tiles_x(s+1);
                load_tiles_w(s, w_off);
            }
        }
        for(int w_off = 0; w_off<TN; w_off++)
        {
            float* ta = tile_acc[w_off];
            uint32_t* tx = tile_x[0];
            uint32_t* tw = tile_w[w_off][0];
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(ta[0]), "+f"(ta[1]), "+f"(ta[2]), "+f"(ta[3])
                    : "r"(tx[0]), "r"(tx[1]), "r"(tx[2]), "r"(tx[3]), "r"(tw[0]), "r"(tw[1]));
            tx = tile_x[1];
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(ta[0]), "+f"(ta[1]), "+f"(ta[2]), "+f"(ta[3])
                    : "r"(tx[0]), "r"(tx[1]), "r"(tx[2]), "r"(tx[3]), "r"(tw[2]), "r"(tw[3]));
            tx = tile_x[2];
            tw = tile_w[w_off][1];
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(ta[0]), "+f"(ta[1]), "+f"(ta[2]), "+f"(ta[3])
                    : "r"(tx[0]), "r"(tx[1]), "r"(tx[2]), "r"(tx[3]), "r"(tw[0]), "r"(tw[1]));
            tx = tile_x[3];
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(ta[0]), "+f"(ta[1]), "+f"(ta[2]), "+f"(ta[3])
                    : "r"(tx[0]), "r"(tx[1]), "r"(tx[2]), "r"(tx[3]), "r"(tw[2]), "r"(tw[3]));

            if (token_src[0] < M)
            {
                f_acc[w_off][0] += scale_x[0] * scale_w * ta[0];
                f_acc[w_off][1] += scale_x[0] * scale_w * ta[1];
            }
            if (token_src[1] < M)
            {
                f_acc[w_off][2] += scale_x[1] * scale_w * ta[2];
                f_acc[w_off][3] += scale_x[1] * scale_w * ta[3];
            }
        }

        if(load_stage < n_stages)
        {
            __syncthreads();
            async_load();
        }
    }
    for(int i = 0; i<TN; i++)
    {
        if (token_src[0] < M)
        {
            *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + i*8 + (lane_id%4)*2) = __nv_bfloat162(f_acc[i][0], f_acc[i][1]);;
        }
        if (token_src[1] < M)
        {
            *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + i*8 + (lane_id%4)*2) = __nv_bfloat162(f_acc[i][2], f_acc[i][3]);;
        }
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

void fused_moe_w8a8_mb(
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
    constexpr int BN = 16;
    constexpr int PF = 4;
    constexpr int WN = 4;
    // TODO this will only work for num_warps_y = 1
    constexpr int WM = 1;
    dim3 dimBlock(32*WN, WM, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*WN)), std::ceil((float)sorted_num/(BM*WM)), 1);

    fused_moe_w8a8_mb_kernel<BM, BK, BN, PF, WM, WN, 3><<<dimGrid, dimBlock>>>(
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
            //     printf("M %d, K %d, N %d cs %d, ls %d, mma %d, %d with %f,%f,%f,%f ||| %f, %f, %f, %f , w %f,%f,%f,%f ||| %f, %f, %f, %f acc %f, %f, %f, %f, scale x %f,%f scale_w %f\n",
            //             M, K, N,
            //             compute_stage, load_stage,
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
            //             );
