#include "reference/flash_attention_1_gpu/forward.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {
    constexpr int ceildiv(int a, int b) {
        return (a + b - 1) / b;
    }
}

__global__ void flash_attention_kernel(const float *q, const float *k, const float *v, float* o, int N, int d, int M) {
    int Bc = ceildiv(M, 4 * d);
    int Br = std::min(d, ceildiv(M, 4 * d));

    int Tc = ceildiv(N, Bc);
    int Tr = ceildiv(N, Br);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    // Initialize local variables
    float l = 0.0f;
    float m = -std::numeric_limits<float>::infinity();

    for (int kv_start = 0; kv_start < N; kv_start += Bc) {
        int kv_end = std::min(kv_start + Bc, N);

        for (int q_start = 0; q_start < N; q_start += Br) {
            int q_end = std::min(q_start + Br, N);

            if (tid >= q_start && tid < q_end) {
                float q_row[d];
                float o_row[d];
                float temp[d];

                for (int i = 0; i < d; ++i) {
                    q_row[i] = q[tid * d + i];
                    o_row[i] = o[tid * d + i];
                }

                for (int j = kv_start; j < kv_end; ++j) {
                    float k_row[d];
                    float v_row[d];

                    for (int i = 0; i < d; ++i) {
                        k_row[i] = k[j * d + i];
                        v_row[i] = v[j * d + i];
                    }

                    float S = 0.0f;
                    for (int i = 0; i < d; ++i) {
                        S += q_row[i] * k_row[i];
                    }
                    S *= 1.0f / sqrtf(d);

                    if (j > tid) {
                        S = -std::numeric_limits<float>::infinity();
                    }

                    float tilde_m = S;
                    float tilde_p = exp(S - tilde_m);
                    float tilde_L = tilde_p;

                    float m_new = fmaxf(m, tilde_m);
                    float L_new = exp(m - m_new) * l + exp(tilde_m - m_new) * tilde_L;

                    for (int i = 0; i < d; ++i) {
                        temp[i] = tilde_p * v_row[i];
                    }

                    float scale_prev = exp(m - m_new);
                    float scale_new = exp(tilde_m - m_new);

                    for (int i = 0; i < d; ++i) {
                        o_row[i] = (scale_prev * l * o_row[i] + scale_new * temp[i]) / L_new;
                    }

                    l = L_new;
                    m = m_new;
                }

                for (int i = 0; i < d; ++i) {
                    o[tid * d + i] = o_row[i];
                }
            }
        }
    }
}

__host__ void FlashAttention1ForwarderGPU::go_gpu(const float *q, const float *k, const float *v, float* o, int N, int d, int M) {
    int Bc = ceildiv(M, 4 * d);
    int Br = std::min(d, ceildiv(M, 4 * d));

    int Tc = ceildiv(N, Bc);
    int Tr = ceildiv(N, Br);

    // Kernel launch: This is where you call the actual GPU kernel function
    flash_attention_kernel<<<Tc, Br>>>(q, k, v, o, N, d, M);

    // Check for any errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // wait for all ops to finish
    cudaDeviceSynchronize();
}
