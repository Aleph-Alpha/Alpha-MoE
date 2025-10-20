#include <cuda/std/limits>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <cuda/ptx>
#include <cassert>
#include <cudaTypedefs.h>


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

namespace aa4
{
    PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
        // Get pointer to cuTensorMapEncodeTiled
        cudaDriverEntryPointQueryResult driver_status;
        void* cuTensorMapEncodeTiled_ptr = nullptr;
        gpuErrchk(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
        ASSERT(driver_status == cudaDriverEntryPointSuccess, "Failed driver status %d", 0);

        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
    }
}
using namespace aa4;

constexpr __device__ __forceinline__ int32_t const_ceil(float num)
{
    return (static_cast<float>(static_cast<int32_t>(num)) == num)
        ? static_cast<int32_t>(num)
        : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 4);
}

template <int BITS, int BASE = 4, int SHIFT = 3>
__device__ __forceinline__ int32_t swizzle(const int32_t i)
{
    if constexpr(BITS == 0)
        return i;
    constexpr uint32_t S_MASK = ((1 << BITS) - 1) << (BASE + SHIFT);
    return i^((i&S_MASK)>>SHIFT);
}

// Descriptor for a shared memory matrix.
// Implementation is derived from PTX guide: https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format
template<int SDO, int LDO, uint64_t S_MODE>
__device__ __forceinline__ uint64_t make_smem_descriptor(fp8* ptr) {
    // Convert shared memory pointer to integer
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = matrix_descriptor_encode(addr);
    desc |= ((uint64_t)LDO) << 16;
    desc |= ((uint64_t)SDO) << 32;
    // I don't think we need this anymore
    // desc |= (((uint64_t)addr >> 0x7) & 0x7) << 49;
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

template<int ScaleD, int ScaleA, int ScaleB, int BM, int SDO_A, int LDO_A, int SDO_B, int LDO_B, int S_MODE_A, int S_MODE_B>
__device__ void wgmma(float d[BM/8][4], fp8* sA, fp8* sB)
{
    uint64_t desc_a = make_smem_descriptor<SDO_A, LDO_A, S_MODE_A>(&sA[0]);
    uint64_t desc_b = make_smem_descriptor<SDO_B, LDO_B, S_MODE_B>(&sB[0]);

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

__device__ inline void load_async(fp8 *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile (
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
        "r"(global_row_idx), "r"(global_col_idx)
        : "memory"
    );
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

__device__ __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile ("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "r"(bytes));
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
    alignas(1024) fp8 w[STAGES*WN*BK*BN];
    alignas(1024) fp8 x[BM*WN*BN/2];
    alignas(1024) __nv_bfloat16 out[BM*(BK*2 + 8)];
};


template <int BM, int BK, int BN, int WN, int STAGES, int PRODUCER_THREADS>
__global__ __launch_bounds__(WN*32 + PRODUCER_THREADS) void fused_moe_w8a8_wgmma_up_down_acc_kernel(
        const fp8* __restrict__ x,
        const float* __restrict__ x_scale,
        const __grid_constant__ CUtensorMap tensor_map_w,
        const float* __restrict__ w_scale,
        const __grid_constant__ CUtensorMap tensor_map_w2,
        const float* __restrict__ w2_scale,
        __nv_bfloat16* __restrict__ out,
        const int* __restrict__ sorted_token_ids,
        const int* __restrict__ expert_ids,
        const int* __restrict__ num_tokens_post_padded,
        const __nv_bfloat16* __restrict__ topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        float scaling_factor
        )
{
    constexpr int CONSUMER_THREADS = WN*32;
    constexpr int WARPGROUPS = WN / 4;
    const int32_t warpM = blockIdx.y;
    const int exp_idx = expert_ids[warpM];
    if(warpM * BM >= num_tokens_post_padded[0])
        return;

    const int32_t warpN = (blockIdx.x*CONSUMER_THREADS + (threadIdx.x - PRODUCER_THREADS))/32;

    //TODO should not be hardcoded
    constexpr int block_shape[2] = {128, 128};

    constexpr uint32_t S_BITS_UP = 3;
    constexpr uint32_t S_MODE_UP = 4 - S_BITS_UP;

    constexpr uint32_t S_BITS_DOWN = WN*BN == 256 ? 3 :
                                     WN*BN == 128 ? 2 :
                                                    1;

    constexpr uint32_t S_MODE_DOWN = 4 - S_BITS_DOWN;

    const int K2 = N/2;
    const int N2 = K;
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
            init_barrier(&bar[i], PRODUCER_THREADS + 1, 0);
            init_barrier(&bar[i + STAGES], CONSUMER_THREADS, 0);
        }
    }
    __syncthreads();
    auto consumer_sync = [&]()
    {
        asm volatile("bar.sync 0, %0;\n" :: "n"(CONSUMER_THREADS));
    };

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
        int tsrc[X_IT];
        for(int r = 0; r < X_IT; r += 1)
        {
            tsrc[r] = sorted_token_ids[warpM*BM + r*(PRODUCER_THREADS/8) + threadIdx.x/8] / top_k;
        }
        int smem_stage = 0;
        const int w_row_up = exp_idx * N + (blockIdx.x)*WN*BN;
        for (int load_stage = 0; load_stage<n_stages_up; load_stage++)
        {
            if (smem_stage == STAGES)
            {
                p^=1;
                smem_stage = 0;
            }
            const int off = load_stage * block_shape[0];
            int i = (threadIdx.x)*TO;
            int col = off + i%BK;
            int swizzled = swizzle<S_BITS_UP>(i);

            wait(bar + STAGES + smem_stage, p);

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

            if(threadIdx.x == 0)
            {
                expect_bytes(bar+smem_stage, WS*sizeof(fp8));
                load_async(s.w + smem_stage*WS, &tensor_map_w, bar + smem_stage, w_row_up, off);
            }
            smem_stage++;
        }

        smem_down<STAGES, WN, BM, BK, BN>& s_d = *reinterpret_cast<smem_down<STAGES, WN, BM, BK, BN>*>(sh);

        for (int load_stage = 0; load_stage<n_stages_down; load_stage++)
        {
            if (smem_stage == STAGES)
            {
                p^=1;
                smem_stage = 0;
            }
            wait(bar + STAGES + smem_stage, p);

            if(threadIdx.x == 0)
            {
                const int w_row_down = blockIdx.x*BK2;
                const int w_col_down = exp_idx * N2 + load_stage*BN2;
                expect_bytes(bar+smem_stage, WS*sizeof(fp8));
                load_async(s_d.w + smem_stage*WS, &tensor_map_w2, bar + smem_stage, w_col_down, w_row_down);
            }
            arrive(bar + smem_stage);
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
            memset(tile_acc, 0, sizeof(tile_acc));
            warpgroup_arrive();
            for(int tn = 0; tn<TN; tn++)
            {
                fp8* sw = s.w + smem_stage*WS + (warp_id/4)*(BN*4)*BK + tn*64*BK;
                fp8* sx = s.x + smem_stage*XS;
                wgmma<1,1,1, BM, 64, 1, 64, 1, S_MODE_UP, S_MODE_UP>(tile_acc[tn], sw, sx);
                wgmma<1,1,1, BM, 64, 1, 64, 1, S_MODE_UP, S_MODE_UP>(tile_acc[tn], sw+1*32, sx+1*32);
                wgmma<1,1,1, BM, 64, 1, 64, 1, S_MODE_UP, S_MODE_UP>(tile_acc[tn], sw+2*32, sx+2*32);
                wgmma<1,1,1, BM, 64, 1, 64, 1, S_MODE_UP, S_MODE_UP>(tile_acc[tn], sw+3*32, sx+3*32);
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
        consumer_sync();
        smem_down<STAGES, WN, BM, BK, BN>& s_d = *reinterpret_cast<smem_down<STAGES, WN, BM, BK, BN>*>(sh);
        float4* block_max = reinterpret_cast<float4*>(s_d.out);
        for(int tn = 0; tn<TN; tn++)
        {
            for(int tm = 0; tm<TM; tm++)
            {
                f_acc[tn][tm][0] = swiglu_mul(f_acc[tn][tm][0], f_acc[tn][tm][2]);
                f_acc[tn][tm][1] = swiglu_mul(f_acc[tn][tm][1], f_acc[tn][tm][3]);
            }
        }
        // Holds max firts, scale later
        float token_max[TM][2];
        for(int tm = 0; tm<TM; tm++)
        {
            for (int t = 0; t < 2; t++)
            {
                token_max[tm][t] = 1e-10;
                for(int tn = 0; tn<TN; tn++)
                {
                    token_max[tm][t] = fmaxf(fabsf(f_acc[tn][tm][t]),token_max[tm][t]);
                }
                token_max[tm][t] = fmaxf(__shfl_xor_sync(0xFFFFFFFF, token_max[tm][t], 16), token_max[tm][t]);
                token_max[tm][t] = fmaxf(__shfl_xor_sync(0xFFFFFFFF, token_max[tm][t], 8), token_max[tm][t]);
                token_max[tm][t] = fmaxf(__shfl_xor_sync(0xFFFFFFFF, token_max[tm][t], 4), token_max[tm][t]);
                int off = tm*8 + (lane_id%4)*2 + t;
                reinterpret_cast<float*>(block_max + off * WARPGROUPS)[warp_id] = token_max[tm][t];
            }
        }
        consumer_sync();
        // It's kida sad that it's not constexpr compatible
        // constexpr float fp8_max = static_cast<float>(::cuda::std::numeric_limits<fp8>::max());
        // constexpr float fp8_min = static_cast<float>(::cuda::std::numeric_limits<fp8>::min());

        constexpr float fp8_max = 448.0;
        constexpr float fp8_min = -448.0;

        for(int tm = 0; tm<TM; tm++)
        {
            for (int t = 0; t < 2; t++)
            {
                int off = tm*8 + (lane_id%4)*2 + t;
                // Reduce max across all warp groups using float4 loads
                for(int wg = 0; wg < WARPGROUPS; wg++)
                {
                    float4 bmax = block_max[off * WARPGROUPS + wg];
                    token_max[tm][t] = fmaxf(bmax.x, token_max[tm][t]);
                    token_max[tm][t] = fmaxf(bmax.y, token_max[tm][t]);
                    token_max[tm][t] = fmaxf(bmax.z, token_max[tm][t]);
                    token_max[tm][t] = fmaxf(bmax.w, token_max[tm][t]);
                }
                float m = token_max[tm][t];
                float scale = token_max[tm][t] / fp8_max;
                token_max[tm][t] = scale;
                for(int tn = 0; tn<TN; tn++)
                {
                    float val = f_acc[tn][tm][t];
                    float q = val / scale;
                    f_acc[tn][tm][t] = fminf(fmaxf(q, fp8_min), fp8_max);
                    int x_row = tm*8 + (lane_id%4)*2 + t;
                    int x_col = (warp_id/4)*(TN*32) + tn*32 + (warp_id%4)*8 + lane_id/4;
                    int i = x_row*BK2 + x_col;
                    int swizzled = swizzle<S_BITS_DOWN>(i);
                    s_d.x[swizzled] = fp8(f_acc[tn][tm][t]);
                }
            }
        }

        for(int t = 0; t < TPT; t++)
        {
            if (token_dest[t] < M * top_k)
            {
                const float topk_w = topk_weights[token_dest[t]];
                token_max[t/2][t%2] *= topk_w * scaling_factor;
            }
        }

        constexpr int TN2 = BN2/(64*WARPGROUPS);
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
            const int scales_per_stage = BN2 / block_shape[0];
            const int scales_per_wg = (TN2 * 64) / block_shape[0];
            float scale_w[TN2/2];
            int s_r = (w_row/2)/block_shape[1];
            for (int i = 0; i<(TN2/2); i++)
                scale_w[i] = w2_scale[exp_idx * scale_rows_w * scale_cols_w + s_r*scale_cols_w + compute_stage*scales_per_stage + (warp_id/4)*scales_per_wg + i];


            float tile_acc[TN2][TM][4];
            memset(tile_acc, 0, sizeof(tile_acc));
            fp8* sx = s_d.x;
            warpgroup_arrive();

            for(int tn2 = 0; tn2 < TN2; tn2++)
            {
                fp8* sw = s_d.w + smem_stage*WS + ((warp_id/4)*TN2 + tn2)*64*BK2;
                for(int tk = 0; tk < BK2/32; tk++)
                {
                    wgmma<1,1,1, BM, BK2/2, 1, BK2/2, 1, S_MODE_DOWN, S_MODE_DOWN>(tile_acc[tn2], sw + (tk*32), sx + (tk*32));
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(bar + STAGES + smem_stage);


            constexpr int PAD = BN2+8;
            asm volatile("cp.async.bulk.wait_group 0;");
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
            consumer_sync();
            for(int tn2 = 0; tn2<TN2; tn2+=2)
            {
                for(int tm = 0; tm<TM; tm++)
                {
                    __nv_bfloat16 tile[8];
                    int out_col = (warp_id/4)*TN2*64 + (warp_id%4)*16 + (lane_id&8) + tn2*64 + (lane_id/16)*64;
                    for (int t = 0; t<8; t++)
                    {
                        int out_row = token_src[tm*2 + t%2];
                        if(out_row < M)
                        {
                            int s = t%2;
                            tile[t] = token_max[tm][s]*tile_acc[tn2 + t/4][tm][t%4]*scale_w[tn2/2];
                        }
                    }
                    int out_row = tm * 8 + lane_id%8;
                    st_matrix_x4_trans(reinterpret_cast<uint32_t*>(tile),
                            __cvta_generic_to_shared(s_d.out + out_row*PAD + out_col));
                }
            }
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
            consumer_sync();
            if(warp_id == 0 && lane_id < 4)
            {
                for(int t = 0; t < TPT; t++)
                {
                    if (token_src[t] < M)
                    {
                        int row = (t/2)*8 + t%2 + lane_id*2;
                        cuda::ptx::cp_reduce_async_bulk(
                                cuda::ptx::space_global,
                                cuda::ptx::space_shared,
                                cuda::ptx::op_add,
                                out + token_src[t]*N2 + compute_stage*BN2,
                                s_d.out + row*PAD,
                                BN2*sizeof(__nv_bfloat16));
                    }
                }
                cuda::ptx::cp_async_bulk_commit_group();
            }
            smem_stage++;
        }
    }
}

template<int BM>
void launch_fused_moe_kernel_up_down_acc(
        const fp8* x,
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
        )
{
    constexpr int BK = 128;
    constexpr int BN = 32;
    constexpr int WN = 8;
    constexpr int STAGES = 2;
    constexpr int PRODUCER_THREADS = 128;
    dim3 dimBlock(32*WN + PRODUCER_THREADS, 1, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*WN)), std::ceil((float)sorted_num/(block_m)), 1);

    size_t sMemSize = std::max(sizeof(smem_up<STAGES, WN, BM, BK, BN>),
                               sizeof(smem_down<STAGES, WN, BM, BK, BN>));

    gpuErrchk(cudaFuncSetAttribute(
                fused_moe_w8a8_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    // Create TMA descriptor for w (up projection)
    CUtensorMap tensor_map_w{};
    {
        constexpr uint32_t rank = 2;
        uint64_t size[rank] = {static_cast<uint64_t>(K), static_cast<uint64_t>(N*257)};
        uint64_t stride[rank] = {static_cast<uint64_t>(K) * sizeof(fp8)};
        uint32_t box_size[rank] = {BK, BN*WN};
        uint32_t elem_stride[rank] = {1, 1};
        auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
        CUresult res = cuTensorMapEncodeTiled(
                &tensor_map_w,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                rank, w, size, stride, box_size, elem_stride,
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                );
    }

    // Create TMA descriptor for w2 (down projection)
    constexpr int BK2 = WN*BN/2;
    constexpr int BN2 = BK*2;
    const int K2 = N/2;
    const int N2 = K;
    CUtensorMap tensor_map_w2{};
    {
        constexpr uint32_t rank = 2;
        uint64_t size[rank] = {static_cast<uint64_t>(K2), static_cast<uint64_t>(N2*257)};
        uint64_t stride[rank] = {static_cast<uint64_t>(K2) * sizeof(fp8)};
        uint32_t box_size[rank] = {BK2, BN2};
        uint32_t elem_stride[rank] = {1, 1};
        auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
        CUresult res = cuTensorMapEncodeTiled(
                &tensor_map_w2,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                rank, w2, size, stride, box_size, elem_stride,
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                WN*BN == 256 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B :
                WN*BN == 128 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B :
                               CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                );
    }

    fused_moe_w8a8_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS><<<dimGrid, dimBlock, sMemSize>>>(
            x,
            x_scale,
            tensor_map_w,
            w_scale,
            tensor_map_w2,
            w2_scale,
            out,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            top_k,
            M,
            K,
            N,
            scaling_factor
            );
}

void fused_moe_w8a8_wgmma_up_down_acc(
        const fp8* x,
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
        )
{
    switch(block_m) {
        case 8:   launch_fused_moe_kernel_up_down_acc<8>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,   num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 16:  launch_fused_moe_kernel_up_down_acc<16>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 24:  launch_fused_moe_kernel_up_down_acc<24>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 32:  launch_fused_moe_kernel_up_down_acc<32>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 40:  launch_fused_moe_kernel_up_down_acc<40>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 48:  launch_fused_moe_kernel_up_down_acc<48>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 56:  launch_fused_moe_kernel_up_down_acc<56>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 64:  launch_fused_moe_kernel_up_down_acc<64>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 72:  launch_fused_moe_kernel_up_down_acc<72>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 80:  launch_fused_moe_kernel_up_down_acc<80>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 88:  launch_fused_moe_kernel_up_down_acc<88>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 96:  launch_fused_moe_kernel_up_down_acc<96>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,  num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 104: launch_fused_moe_kernel_up_down_acc<104>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 112: launch_fused_moe_kernel_up_down_acc<112>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 120: launch_fused_moe_kernel_up_down_acc<120>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        case 128: launch_fused_moe_kernel_up_down_acc<128>(x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, K, N, sorted_num, block_m, scaling_factor); break;
        default:
                  fprintf(stderr, "Unsupported block_m value: %d. Supported values are: 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128\n", block_m);
                  break;
    }
}

