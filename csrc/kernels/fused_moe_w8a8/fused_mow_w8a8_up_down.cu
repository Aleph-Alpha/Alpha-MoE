#include <cuda/std/limits>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <cuda/ptx>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// not gonna type all that
using fp8 = __nv_fp8_e4m3;
constexpr __device__ __forceinline__ int32_t const_ceil(float num)
{
    return (static_cast<float>(static_cast<int32_t>(num)) == num)
        ? static_cast<int32_t>(num)
        : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 4);
}

template <int COL_BITS, int VAL_BITS = 4, int ROW_BITS = 3>
__device__ __forceinline__ int32_t swizzle(const int32_t i)
{
    constexpr uint32_t S_MASK = 0b111 << VAL_BITS;
    return i^((i&(S_MASK<<COL_BITS))>>COL_BITS);
}

// Descriptor for a shared memory matrix.
// Implementation is derived from PTX guide: https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format
template<int SDO, int LDO = 1, int64_t S_MODE = 1>
__device__ __forceinline__ uint64_t make_smem_desc(fp8* ptr) {
    // Convert shared memory pointer to integer
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)LDO) << 16;
    desc |= ((uint64_t)SDO) << 32;
    // I don't think we need this anymore
    desc |= (((uint64_t)addr >> 0x7) & 0x7) << 49;
    desc |= S_MODE << 62;
    return desc;
}

__device__ __forceinline__ float swiglu_mul(float x, float w)
{
    return (x/(1+__expf(-x)))*w;
}

__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma8(float d[1][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3},   "
        " %4,"
        " %5,"
        " %6, %7, %8;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma16(float d[2][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
        " %8,"
        " %9,"
        " %10, %11, %12;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma24(float d[3][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11},  "
        " %12,"
        " %13,"
        " %14, %15, %16;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma32(float d[4][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},  "
        " %16,"
        " %17,"
        " %18, %19, %20;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma40(float d[5][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n40k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19},  "
        " %20,"
        " %21,"
        " %22, %23, %24;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma48(float d[6][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23},  "
        " %24,"
        " %25,"
        " %26, %27, %28;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma56(float d[7][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n56k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27},  "
        " %28,"
        " %29,"
        " %30, %31, %32;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma64(float d[8][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},  "
        " %32,"
        " %33,"
        " %34, %35, %36;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma72(float d[9][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35},  "
        " %36,"
        " %37,"
        " %38, %39, %40;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma80(float d[10][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39},  "
        " %40,"
        " %41,"
        " %42, %43, %44;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma88(float d[11][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n88k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43},  "
        " %44,"
        " %45,"
        " %46, %47, %48;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma96(float d[12][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n96k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},  "
        " %48,"
        " %49,"
        " %50, %51, %52;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
         "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma104(float d[13][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n104k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51},  "
        " %52,"
        " %53,"
        " %54, %55, %56;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
         "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
         "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma112(float d[14][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n112k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55},  "
        " %56,"
        " %57,"
        " %58, %59, %60;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
         "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
         "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
         "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma120(float d[15][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n120k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  %56,  %57,  %58,  %59},  "
        " %60,"
        " %61,"
        " %62, %63, %64;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
         "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
         "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
         "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
         "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmma128(float d[16][4], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},  "
        " %64,"
        " %65,"
        " %66, %67, %68;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
         "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
         "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
         "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
         "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
         "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
         "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
         "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
         "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
         "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
         "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
         "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
         "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
         "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
         "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]),
         "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int BM, int SDO_A = 64, int LDO_A = 1, int SDO_B = 64, int LDO_B = 1, int S_MODE_A = 1, int S_MODE_B = 1>
__device__ void wgmma(float d[BM/8][4], fp8* sA, fp8* sB)
{
    uint64_t desc_a = make_smem_desc<SDO_A, LDO_A, S_MODE_A>(&sA[0]);
    uint64_t desc_b = make_smem_desc<SDO_B, LDO_B, S_MODE_B>(&sB[0]);

    if constexpr (BM == 8)
        wgmma8<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 16)
        wgmma16<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 24)
        wgmma24<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 32)
        wgmma32<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 40)
        wgmma40<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 48)
        wgmma48<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 56)
        wgmma56<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 64)
        wgmma64<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 72)
        wgmma72<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 80)
        wgmma80<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 88)
        wgmma88<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 96)
        wgmma96<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 104)
        wgmma104<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 112)
        wgmma112<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 120)
        wgmma120<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
    else if constexpr (BM == 128)
        wgmma128<ScaleD, ScaleA, ScaleB>(d, desc_a, desc_b);
}

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG4(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

__device__ __forceinline__ void init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count+transaction_count)
    );
}

__device__ __forceinline__ void cp_async_mbarrier_arrive(uint64_t* bar) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n"
        :: "r"(bar_ptr)
    );
}

__device__ __forceinline__ void arrive(uint64_t* bar, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.arrive.shared.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

__device__ __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
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

__device__ __forceinline__ void st_matrix_x4(uint32_t* tile, uint32_t mat)
{
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
            :
            : "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
            : "memory"
            );
}

__device__ __forceinline__ void st_matrix_x4_trans(uint32_t* tile, uint32_t mat)
{
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
            :
            : "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
            : "memory"
            );
}

__device__ __forceinline__ void st_matrix_x2_trans(uint32_t* tile, uint32_t mat)
{
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
            :
            : "r"(mat), "r"(tile[0]), "r"(tile[1])
            : "memory"
            );
}

__device__ __forceinline__ void st_matrix_x1_trans(uint32_t* tile, uint32_t mat)
{
    asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};"
            :
            : "r"(mat), "r"(tile[0])
            : "memory"
            );
}

template <uint32_t RegCount>
__device__ void reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}


template<int STAGES, int WN, int BM, int BK, int BN>
struct smem_up
{
    alignas(1024) fp8 w[STAGES*WN*BK*BN];
    alignas(1024) fp8 x[STAGES*BK*BM];
};

template<int STAGES, int WN, int BM, int BK, int BN>
struct smem_down
{
    alignas(1024) fp8 w[STAGES*BK*WN*BN];
    alignas(1024) fp8 x[BM*BN/2];
};


template <int BM, int BK, int BN, int WN, int STAGES, int PRODUCER_THREADS>
__global__ __launch_bounds__(WN*32 + PRODUCER_THREADS) void fused_moe_w8a8_wgmma_up_down_kernel(
        const fp8* __restrict__ x,
        const float* __restrict__ x_scale,
        const fp8* __restrict__ w,
        const float* __restrict__ w_scale,
        const fp8* __restrict__ w2,
        const float* __restrict__ w2_scale,
        __nv_bfloat16* __restrict__ out,
        const int* __restrict__ sorted_token_ids,
        const int* __restrict__ expert_ids,
        const int* __restrict__ num_tokens_post_padded,
        const __nv_bfloat16* __restrict__ topk_weights,
        const int top_k,
        int M,
        int K,
        int N
        )
{
    constexpr int CONSUMER_THREADS = WN*32;
    const int32_t warpM = blockIdx.y;
    const int exp_idx = expert_ids[warpM];
    if(warpM * BM >= num_tokens_post_padded[0])
        return;

    const int32_t warpN = (blockIdx.x*CONSUMER_THREADS + (threadIdx.x - PRODUCER_THREADS))/32;

    //TODO should not be hardcoded
    constexpr int block_shape[2] = {128, 128};

    const fp8* exp_w = w + exp_idx * K * N;

    constexpr uint32_t S_BITS_UP = 3;
    constexpr uint32_t S_MODE_UP = 1;

    constexpr uint32_t S_BITS_DOWN = 3;
    constexpr uint32_t S_MODE_DOWN = 1;

    const int K2 = N/2;
    const int N2 = K;
    const fp8* exp_w2 = w2 + exp_idx * K2 * N2;
    const int lane_id = threadIdx.x%32;
    const bool is_producer = threadIdx.x < PRODUCER_THREADS;
    const int warp_id = is_producer ? threadIdx.x/32 : (threadIdx.x-PRODUCER_THREADS)/32;
    const int w_row = warpN * BN + (lane_id>>2);

    constexpr int BK2 = WN*BN/2;
    constexpr int BN2 = BK*2;

    //SMEM sizes
    constexpr int WS = WN*BK*BN;
    constexpr int XS = BK*BM;
    // how many bytes we transfer per CP_ASYNC
    constexpr int TB = 16;
    // Thread offset per transfer
    constexpr int TO = TB/sizeof(fp8);
    //Tokens per thread
    constexpr int TPT = BM/4;

    extern __shared__ __align__(1024) uint8_t sh[];
    smem_up<STAGES, WN, BM, BK, BN>& s = *reinterpret_cast<smem_up<STAGES, WN, BM, BK, BN>*>(sh);

    __shared__ __align__(8) uint64_t bar[2*STAGES];
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < STAGES; i++)
        {
            init_barrier(&bar[i], PRODUCER_THREADS, 0);
            init_barrier(&bar[i + STAGES], CONSUMER_THREADS, 0);
        }
    }
    __syncthreads();

    int n_stages_up = K/block_shape[0];
    int n_stages_down = N2/BN2;

    constexpr int TM = BM/8;
    constexpr int TN = BN/16;
    float f_acc[TN][TM][4];
    memset(f_acc, 0, sizeof(f_acc));

    int token_dest[TPT];
    int token_src[TPT];
    int p = 0;
    // PRODUCER
    if (is_producer)
    {
        // TODO does it really matter?
        // reg_dealloc<32>();
        constexpr int X_IT = const_ceil(XS/float(PRODUCER_THREADS*TO));
        constexpr int W_IT = const_ceil(WS/float(PRODUCER_THREADS*TO));
        int tsrc[X_IT];
        for(int r = 0; r < X_IT; r += 1)
        {
            tsrc[r] = sorted_token_ids[warpM*BM + r*(PRODUCER_THREADS/8) + threadIdx.x/8] / top_k;
        }
        int smem_stage = 0;
        for (int load_stage = 0; load_stage<n_stages_up; load_stage++)
        {
            if (smem_stage == STAGES)
            {
                p^=1;
                smem_stage = 0;
            }
            const int off = load_stage * block_shape[0];
            int i = (threadIdx.x)*TO;
            int row = blockIdx.x*WN*BN + i/BK;
            int col = off + i%BK;
            int swizzled = swizzle<S_BITS_UP>(i);

            wait(bar + STAGES + smem_stage, p);
            for(int r = 0; r < W_IT; r += 1)
            {
                uint32_t sm = __cvta_generic_to_shared(s.w + smem_stage*WS + swizzled);
                CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(exp_w + row*K + col), TB);
                i += PRODUCER_THREADS*TO;
                swizzled += PRODUCER_THREADS*TO;
                row += PRODUCER_THREADS*TO/BK;
            }

            i = (threadIdx.x)*TO;
            swizzled = swizzle<S_BITS_UP>(i);
            for(int r = 0; r < X_IT; r += 1)
            {
                int row = tsrc[r];
                if(row < M && i < XS)
                {
                    uint32_t sm = __cvta_generic_to_shared(s.x + smem_stage*XS + swizzled);
                    CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row*K + col), TB);
                }
                i += PRODUCER_THREADS*TO;
                swizzled += PRODUCER_THREADS*TO;
            }
            cp_async_mbarrier_arrive(bar + smem_stage);
            smem_stage++;
        }
        __syncthreads();
        __syncthreads();

        smem_down<STAGES, WN, BM, BK, BN>& s_d = *reinterpret_cast<smem_down<STAGES, WN, BM, BK, BN>*>(sh);

        for (int load_stage = 0; load_stage<n_stages_down; load_stage++)
        {
            if (smem_stage == STAGES)
            {
                p^=1;
                smem_stage = 0;
            }
            const int off = load_stage * block_shape[0];
            int i = (threadIdx.x)*TO;
            int row = load_stage*BN2 + i/BK2;
            int col = blockIdx.x*BK2 + i%BK2;
            int swizzled = swizzle<S_BITS_DOWN>(i);
            wait(bar + STAGES + smem_stage, p);
            for(int r = 0; r < W_IT; r += 1)
            {
                uint32_t sm = __cvta_generic_to_shared(s.w + smem_stage*WS + swizzled);
                CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(exp_w2 + row*K2 + col), TB);
                i += PRODUCER_THREADS*TO;
                swizzled += PRODUCER_THREADS*TO;
                row += PRODUCER_THREADS*TO/BK;
            }
            cp_async_mbarrier_arrive(bar + smem_stage);
            smem_stage++;
        }
    }
    // CONSUMER
    else
    {
        for(int t = 0; t < TPT; t+=2)
        {
            int2 tdst = *reinterpret_cast<const int2*>(&sorted_token_ids[warpM*BM + t*4 + (lane_id%4)*2]);
            token_dest[t] = tdst.x;
            token_dest[t+1] = tdst.y;

            token_src[t] = token_dest[t]/top_k;
            token_src[t+1] = token_dest[t+1]/top_k;
        }
        // reg_alloc<128>();
        // Empty barriers arrive instantly
        for (int i = 0; i < STAGES; i++)
            arrive(&bar[STAGES + i]);


        int smem_stage = 0;
        for (int compute_stage = 0; compute_stage < n_stages_up; compute_stage += 1)
        {
            if (smem_stage == STAGES)
            {
                p^=1;
                smem_stage = 0;
            }
            wait(bar + smem_stage, p);

            const int scale_cols_x = K/block_shape[1];
            const int scale_rows_w = N/block_shape[1];
            const int scale_cols_w = K/block_shape[0];

            float scale_x[TPT];
            for(int t = 0; t < TPT; t++)
            {
                if (token_src[t] < M)
                {
                    scale_x[t] = x_scale[token_src[t]*scale_cols_x + compute_stage];
                }
            }
            float scale_w[2];
            int s_r = w_row/(block_shape[1]*2);
            scale_w[0] = w_scale[exp_idx * scale_rows_w * scale_cols_w + (s_r)*scale_cols_w + compute_stage];
            scale_w[1] = w_scale[exp_idx * scale_rows_w * scale_cols_w + (s_r + 1)*scale_cols_w + compute_stage];


            float tile_acc[TN][TM][4];
            memset(tile_acc, 0, sizeof(f_acc));
            warpgroup_arrive();
            for(int tn = 0; tn<TN; tn++)
            {
                fp8* sw = s.w + smem_stage*WS + (warp_id/4)*(BN*4)*BK + tn*64*BK;
                fp8* sx = s.x + smem_stage*XS;
                wgmma<1,1,1, BM, 64, 1, 64, 1>(tile_acc[tn], sw, sx);
                wgmma<1,1,1, BM, 64, 1, 64, 1>(tile_acc[tn], sw+1*32, sx+1*32);
                wgmma<1,1,1, BM, 64, 1, 64, 1>(tile_acc[tn], sw+2*32, sx+2*32);
                wgmma<1,1,1, BM, 64, 1, 64, 1>(tile_acc[tn], sw+3*32, sx+3*32);
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(bar + STAGES + smem_stage);

            for(int tm = 0; tm<TM; tm++)
            {
                float x_sc;
                x_sc = scale_x[tm*2];
                for(int tn = 0; tn<TN; tn++)
                {
                    f_acc[tn][tm][0] += scale_w[0] * x_sc * tile_acc[tn][tm][0];
                    f_acc[tn][tm][2] += scale_w[1] * x_sc * tile_acc[tn][tm][2];
                }

                x_sc = scale_x[tm*2 + 1];
                for(int tn = 0; tn<TN; tn++)
                {
                    f_acc[tn][tm][1] += scale_w[0] * x_sc * tile_acc[tn][tm][1];
                    f_acc[tn][tm][3] += scale_w[1] * x_sc * tile_acc[tn][tm][3];
                }
            }
            smem_stage++;
        }
        __syncthreads();
        smem_down<STAGES, WN, BM, BK, BN>& s_d = *reinterpret_cast<smem_down<STAGES, WN, BM, BK, BN>*>(sh);
        float4* block_max = reinterpret_cast<float4*>(s_d.x);
        for(int tn = 0; tn<TN; tn++)
        {
            for(int tm = 0; tm<TM; tm++)
            {
                // if(threadIdx.x == PRODUCER_THREADS && blockIdx.x == 0 && blockIdx.y == 0)
                // {
                //     printf("swiglu %f, %f, %f\n", f_acc[tn][tm][0], f_acc[tn][tm][2], swiglu_mul(f_acc[tn][tm][0], f_acc[tn][tm][2]));
                // }
                f_acc[tn][tm][0] = swiglu_mul(f_acc[tn][tm][0], f_acc[tn][tm][2]);
                f_acc[tn][tm][1] = swiglu_mul(f_acc[tn][tm][1], f_acc[tn][tm][3]);
            }
        }
        // Holds max firts, scale later
        float token_max[TN][TM][2];
        for(int tn = 0; tn<TN; tn++)
        {
            for(int tm = 0; tm<TM; tm++)
            {
                for (int t = 0; t < 2; t++)
                {
                    token_max[tn][tm][t] = fabsf(f_acc[tn][tm][t]);
                    token_max[tn][tm][t] = fmaxf(__shfl_xor_sync(0xFFFFFFFF, token_max[tn][tm][t], 16), token_max[tn][tm][t]);
                    token_max[tn][tm][t] = fmaxf(__shfl_xor_sync(0xFFFFFFFF, token_max[tn][tm][t], 8), token_max[tn][tm][t]);
                    token_max[tn][tm][t] = fmaxf(__shfl_xor_sync(0xFFFFFFFF, token_max[tn][tm][t], 4), token_max[tn][tm][t]);
                    int off = tn*BM + tm*8 + (lane_id%4)*2 + t;
                    reinterpret_cast<float*>(block_max + off)[warp_id] = token_max[tn][tm][t];
                }
            }
        }
        __syncthreads();
        // It's kida sad that it's not constexpr compatible
        // constexpr float fp8_max = static_cast<float>(::cuda::std::numeric_limits<fp8>::max());
        // constexpr float fp8_min = static_cast<float>(::cuda::std::numeric_limits<fp8>::min());

        constexpr float fp8_max = 448.0;
        constexpr float fp8_min = -448.0;

        for(int tn = 0; tn<TN; tn++)
        {
            for(int tm = 0; tm<TM; tm++)
            {
                for (int t = 0; t < 2; t++)
                {
                    int off = tn*BM + tm*8 + (lane_id%4)*2 + t;
                    float4 bmax = block_max[off];
                    token_max[tn][tm][t] = fmaxf(bmax.x, token_max[tn][tm][t]);
                    token_max[tn][tm][t] = fmaxf(bmax.y, token_max[tn][tm][t]);
                    token_max[tn][tm][t] = fmaxf(bmax.z, token_max[tn][tm][t]);
                    token_max[tn][tm][t] = fmaxf(bmax.w, token_max[tn][tm][t]);
                    float scale = token_max[tn][tm][t] / fp8_max;
                    token_max[tn][tm][t] = scale;
                    float val = f_acc[tn][tm][t];
                    float q = val / scale;
                    f_acc[tn][tm][t] = fminf(fmaxf(q, fp8_min), fp8_max);
                    int x_row = tm*8 + (lane_id%4)*2 + t;
                    int x_col = tn*32 + warp_id*8 + lane_id/4;
                    int i = x_row*BK2 + x_col;
                    int swizzled = swizzle<S_BITS_DOWN>(i);
                    // swizzled = i;
                    s_d.x[swizzled] = fp8(f_acc[tn][tm][t]);
                    // if(x_row == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                    // {
                    //     printf("saving %f to %d, %d, acc %d, %f, %f\n", f_acc[tn][tm][t], x_row, x_col, token_dest[0], val, q);
                    // }
                }
            }
        }

        float scale_x[TPT];
        for(int t = 0; t < TPT; t++)
        {
            if (token_src[t] < M)
            {
                scale_x[t] = topk_weights[token_src[t]];
            }
        }

        for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage += 1)
        {
            if (smem_stage == STAGES)
            {
                p^=1;
                smem_stage = 0;
            }
            wait(bar + smem_stage, p);

            const int scale_rows_w = N2/block_shape[1];
            const int scale_cols_w = K2/block_shape[0];
            float scale_w;
            int s_r = w_row/block_shape[1];
            scale_w = w2_scale[exp_idx * scale_rows_w * scale_cols_w + s_r*scale_cols_w + compute_stage];


            constexpr int TN2 = BN2/64;
            float tile_acc[TN2][TN][TM][4];
            memset(tile_acc, 0, sizeof(tile_acc));
            warpgroup_arrive();
            fp8* sx = s_d.x;
            for(int tn2 = 0; tn2<TN2; tn2++)
            {
                fp8* sw = s_d.w + smem_stage*WS + tn2*64*BK2;
                if(threadIdx.x == PRODUCER_THREADS && blockIdx.x == 0 && blockIdx.y == 0 && compute_stage == 0)
                {
                    for(int r = 0; r<1; r++)
                    {
                        for(int c = 0; c<BK2; c++)
                        {
                            int i = r*BK2 + c;
                            int swizzled = swizzle<S_BITS_DOWN>(i);
                            printf("%f,", float(sx[swizzled]) * token_max[0][0][0]);
                        }
                        printf("\n");
                    }

                }
                for(int tn = 0; tn<TN; tn++)
                {
                    wgmma<1,1,1, BM, 16, 1, 16, 1, S_MODE_DOWN, S_MODE_DOWN>(tile_acc[tn2][tn], sw + (tn*32), sx + (tn*32));
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(bar + STAGES + smem_stage);
            // ATOMIC ADD OUTPUT HERE
            for(int tn2 = 0; tn2<TN2; tn2++)
            {
                for(int tm = 0; tm<TM; tm++)
                {
                    for (int t = 0; t<4; t++)
                    {
                        float acc = 0;
                        int s = t%2;
                        for(int tn = 0; tn<TN; tn++)
                        {
                            acc += token_max[tn][tm][s]*tile_acc[tn2][tn][tm][t];
                        }
                        int out_row = token_src[tm*2 + t%2];
                        int out_col = compute_stage*TN2*64 + tn2*64 + (warp_id%4)*16 + (t/2)*8 + (lane_id/4);
                        if(out_row < M)
                        {

                            // out[out_row*K + out_col] = acc;
                            atomicAdd(out + out_row*K + out_col, acc*scale_w*scale_x[tm*2 + t%2]);
                            if(threadIdx.x == PRODUCER_THREADS && blockIdx.x == 0 && blockIdx.y == 0 && compute_stage == 0)
                            {
                                fp8* sw = s_d.w + smem_stage*WS + tn2*64*BK2;
                                printf("adding acc %f, w_s %f, x_s %f, tile_ acc %f, token_max %f, %f, x %f, %f, w %f, %f\n",
                                        acc,
                                        scale_w,
                                        scale_x[t],
                                        tile_acc[tn2][0][tm][t],
                                        token_max[0][tm][0],
                                        token_max[0][tm][1],
                                        float(sw[0]),
                                        float(sw[1]),
                                        float(sx[0]),
                                        float(sx[1])
                                      );
                            }
                        }
                    }
                }
            }
            smem_stage++;
        }
    }
}

template<int BM>
void launch_fused_moe_kernel_up_down(
        const fp8* x,
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
        )
{
    constexpr int BK = 128;
    constexpr int BN = 16;
    constexpr int WN = 4;
    constexpr int STAGES = 3;
    constexpr int PRODUCER_THREADS = 128;
    dim3 dimBlock(32*WN + PRODUCER_THREADS, 1, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*WN)), std::ceil((float)sorted_num/(block_m)), 1);

    static_assert(sizeof(smem_up<STAGES, WN, BM, BK, BN>) > sizeof(smem_down<STAGES, WN, BM, BK, BN>));
    size_t sMemSize = sizeof(smem_up<STAGES, WN, BM, BK, BN>);
    gpuErrchk(cudaFuncSetAttribute(
                fused_moe_w8a8_wgmma_up_down_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    fused_moe_w8a8_wgmma_up_down_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS><<<dimGrid, dimBlock, sMemSize>>>(
            x,
            x_scale,
            w,
            w_scale,
            w2,
            w2_scale,
            out,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            top_k,
            M,
            K,
            N
            );
}

void fused_moe_w8a8_wgmma_up_down(
        const fp8* x,
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
        )
{
    switch(block_m) {
        case 8:   launch_fused_moe_kernel_up_down<8>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,   num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 16:  launch_fused_moe_kernel_up_down<16>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 24:  launch_fused_moe_kernel_up_down<24>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 32:  launch_fused_moe_kernel_up_down<32>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 40:  launch_fused_moe_kernel_up_down<40>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 48:  launch_fused_moe_kernel_up_down<48>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 56:  launch_fused_moe_kernel_up_down<56>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 64:  launch_fused_moe_kernel_up_down<64>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 72:  launch_fused_moe_kernel_up_down<72>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 80:  launch_fused_moe_kernel_up_down<80>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 88:  launch_fused_moe_kernel_up_down<88>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 96:  launch_fused_moe_kernel_up_down<96>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 104: launch_fused_moe_kernel_up_down<104>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 112: launch_fused_moe_kernel_up_down<112>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 120: launch_fused_moe_kernel_up_down<120>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        case 128: launch_fused_moe_kernel_up_down<128>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m); break;
        default:
                  fprintf(stderr, "Unsupported block_m value: %d. Supported values are: 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128\n", block_m);
                  break;
    }
}
