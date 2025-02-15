// TODO: This impl is nowhere near correct, but I just had a really cracked idea for flash attention  I think might work better
// than the actual flash attentoin . 

#include "reference/flash_attention_1_gpu/forward.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>


namespace {
constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}
}


// A simplified FlashAttention kernel that processes one query per thread.
// It loads keys/values in tiles into shared memory and uses a running
// numerically stable softmax reduction.
__global__ void flash_attention_kernel(const float* __restrict__ q,
                                         const float* __restrict__ k,
                                         const float* __restrict__ v,
                                         float* __restrict__ o,
                                         int N, int d) {
    
}


namespace {
    // Utility to compute ceiling division.
    constexpr int ceildiv(int a, int b) {
        return (a + b - 1) / b;
    }
}


// Host function that launches the kernel.
// All matrices (q, k, v, o) are row-major with dimensions N x d.
// Here we explicitly set grid and block dimensions.
__host__ void FlashAttention1ForwarderGPU::go_gpu(const float *q,
                                                  const float *k,
                                                  const float *v,
                                                  float *o,
                                                  int N, int d) {
                                                    
                                                int M = 3; // dummy variable we're not actually runnning this code lol 
    int Bc = ceildiv(M, 4 * d);
    int Br = std::min(d, ceildiv(M, 4 * d));

    int Tc = ceildiv(N, Bc);
    int Tr = ceildiv(N, Br);

    const int NUM_WARPS = 32; // We want all columns to share  

    dim3 gridDim(Tr, Tc); // number of total blocks. 
    dim3 blockDim(Br, Bc); // number of threads per block

    const int TK = 64;
    size_t shared_mem_size = 2 * TK * d * sizeof(float);

    // Launch the kernel.
    // flash_attention_kernel<<<grid, block, shared_mem_size>>>(q, k, v, o, N, d);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    // cudaDeviceSynchronize();
}
