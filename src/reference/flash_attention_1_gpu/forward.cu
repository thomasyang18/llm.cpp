#include "reference/flash_attention_1_gpu/forward.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {
    constexpr int ceildiv(int a, int b) {
        return (a + b - 1) / b;
    }
}



__host__ void FlashAttention1ForwarderGPU::go_gpu(const float *q, const float *k, const float *v, float* o, int N, int d, int M) {
    int Bc = ceildiv(M, 4 * d);
    int Br = std::min(d, ceildiv(M, 4 * d));

    int Tc = ceildiv(N, Bc);
    int Tr = ceildiv(N, Br);

    // Kernel launch: This is where you call the actual GPU kernel function
    // flash_attention_kernel<<<T, blockSize>>>(q, k, v, o, N, d, M);

    // Check for any errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    // wait for all ops to finish
    cudaDeviceSynchronize();
}
