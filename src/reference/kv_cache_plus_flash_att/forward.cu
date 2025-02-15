#pragma once 
#include "reference/kv_cache_plus_flash_att/forward.cuh"
#include "utils/cuda_utils.cuh"
#include <cooperative_groups.h>

KV_Cache_Manager_GPU::KV_Cache_Manager_GPU(const GPTConfig& _p_config, int size): _config(_p_config) {
    _cache.resize(config().n_layer);
    for (int i = 0; i < _cache.size(); ++i) {
        _cache[i].resize(config().n_head);
        for (int j = 0; j < _cache[i].size(); ++j) {
            user_cuda::cudaMallocDevice(_cache[i][j].k, size * config().d());
            user_cuda::cudaMallocDevice(_cache[i][j].v, size * config().d());
        }
    }
}

KV_Cache_Manager_GPU::~KV_Cache_Manager_GPU() {
    for (int i = 0; i < _cache.size(); ++i) {
        for (int j = 0; j < _cache[i].size(); ++j) {
            user_cuda::cudaFreeDevice(_cache[i][j].k);
            user_cuda::cudaFreeDevice(_cache[i][j].v);
        }
    }
}


/*
    START OF INTERESTING CODE

    I mean, the high level idea is simple: 

    We already know we don't have to recompute the whole thing because KV cache.

    Just split your data into blocks 

    [   k,v   ][    k,v    ][   k,v    ][    k,v     ]

    And then apply softmax in parallel.
    
    Note that q, and o are __constant__ and __global__, respectively. 

    Because the online softmax algorithm forms a monoid (e.g. data type T with associativity), we can apply parallel reduction.

    This will be like... acutally 0 speedup, probably. But it might be interesting to code.
*/

/*
    Now that I think about it, I'm not sure if this is even necessarily better; 

    KV cache is already so memory hungry, you could argue that the parallel work approach, which needs O(n) memory concurrently,
    might be even worse. Whereas just a simple linear scan is O(1) extra memory. 
    
    Materializing all that extra memory... ew. But whatever.
*/

/*
    BALANCING MEMORY INITIALIZATION WITH PARALLELISM

    (We're going to allocate O(n) memory anyways.)
    
    Okay, right, so recall:

    A *block* has 1024 threads, max.

    A warp is 32 threads working in lockstep => dude like 

    I can't help but feel like this idea is doomed. Now that I'm working out the details, it's just so memory bound. 

    Like locally, for every thread you're just doing an O(d) computation. I guess the softmax reduction might be faster. 

    I want to see what other people have done.

    https://pytorch.org/blog/accelerating-generative-ai-2/

    These guys seem like they know what they're talking about. 

*/



/* The following code is all generated by o3. It looks fine in principle (I like how it generated SoftMax like a monoid, like I was thinking lmao)
    but this will not work in practice, like, at all. Rip. 

namespace {
    namespace constants {
        //  HIGHLY UNRIGOROUS PERF ENGINEERING

        //    ChatGPT said:

        //    The Tesla T4 GPU has the following cache details:

        //        L1 Cache: 64 KB per SM (Streaming Multiprocessor)
        //        so... M = 64KB :smiley:
        
        constexpr int M = (1 << 16);
        // maximum, single-headed attention embedding value.
        constexpr int MAX_D = 64;
    }

    constexpr int ceildiv(int a, int b) {
        return (a + b - 1) / b;
    }

    // Our softmax accumulator type.
    struct SoftmaxAccum {
        float m;                              // running max (initialized to -INFINITY)
        float s;                              // running sum of exponentials (in the shifted space)
        float v_sum[constants::MAX_D];        // running weighted sum for v
    };

    // // Global accumulator and a lock variable for atomic updates.
    // __device__ SoftmaxAccum globalAccum;
    // __device__ int accumLock;

    // q is stored in constant memory.
    __constant__ float q[constants::MAX_D];

    // Combine two accumulators (the “monoid operation”)
    __device__ inline SoftmaxAccum combine(const SoftmaxAccum &a, const SoftmaxAccum &b, int d) {
        // If one accumulator is “empty” (s==0) then return the other.
        if (a.s == 0.0f) return b;
        if (b.s == 0.0f) return a;
        SoftmaxAccum out;
        out.m = fmaxf(a.m, b.m);
        float factor_a = expf(a.m - out.m);
        float factor_b = expf(b.m - out.m);
        out.s = a.s * factor_a + b.s * factor_b;
        for (int j = 0; j < d; j++) {
            out.v_sum[j] = a.v_sum[j] * factor_a + b.v_sum[j] * factor_b;
        }
        return out;
    }

    // Initialize an accumulator to the identity (no contribution)
    __device__ inline void initAccum(SoftmaxAccum &acc, int d) {
        acc.m = -INFINITY;
        acc.s = 0.0f;
        for (int j = 0; j < d; j++) {
            acc.v_sum[j] = 0.0f;
        }
    }

    
        // One new kernel that implements KV Cache + Flash Attention Forward.
        // It uses a grid–wide barrier (via cooperative groups) to (a) initialize the global accumulator,
        // (b) wait for all block–level reductions, and (c) have a single thread compute the final output.
    
    __global__ void kv_cache_flash_kernel(const float *k, const float *v, float *o, int N, int d) {
        // One thread initializes the global accumulator and lock.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            globalAccum.m = -INFINITY;
            globalAccum.s = 0.0f;
            for (int j = 0; j < d; j++) {
                globalAccum.v_sum[j] = 0.0f;
            }
            accumLock = 0;
        }

        // Each thread computes a local accumulator over its portion of the KV cache.
        SoftmaxAccum local;
        initAccum(local, d);

        // Grid-stride loop over KV pairs.
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = gridDim.x * blockDim.x;
        for (int i = tid; i < N; i += stride) {
            // Compute dot = q · k[i]
            float dot = 0.0f;
            for (int j = 0; j < d; j++) {
                dot += q[j] * k[i * d + j];
            }
            // Update local accumulator with the new element.
            if (local.s == 0.0f) {
                // First element: initialize
                local.m = dot;
                local.s = 1.0f;
                for (int j = 0; j < d; j++) {
                    local.v_sum[j] = v[i * d + j];
                }
            } else {
                float new_m = fmaxf(local.m, dot);
                float factor_a = expf(local.m - new_m);
                float factor_b = expf(dot - new_m);
                local.s = local.s * factor_a + factor_b;
                for (int j = 0; j < d; j++) {
                    local.v_sum[j] = local.v_sum[j] * factor_a + v[i * d + j] * factor_b;
                }
                local.m = new_m;
            }
        }

        // Allocate shared memory for block-level reduction.
        extern __shared__ SoftmaxAccum sdata[];
        sdata[threadIdx.x] = local;
        __syncthreads();

        // Block-level reduction using the combine operation.
        for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                sdata[threadIdx.x] = combine(sdata[threadIdx.x], sdata[threadIdx.x + offset], d);
            }
            __syncthreads();
        }
        // sdata[0] now holds the block’s accumulator.

        // One thread per block (here thread 0) atomically updates the global accumulator.
        if (threadIdx.x == 0) {
            // Spin–lock to update globalAccum
            while (atomicCAS(&accumLock, 0, 1) != 0) {  }
            globalAccum = combine(globalAccum, sdata[0], d);
            atomicExch(&accumLock, 0);
        }
        grid.sync(); // Wait until all blocks have updated the global accumulator.

        // Finally, one thread computes the output.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (int j = 0; j < d; j++) {
                // The final attention output is the weighted sum divided by the sum.
                o[j] = globalAccum.v_sum[j] / globalAccum.s;
            }
        }
    }
} // end anonymous namespace

//-----------------------------------------------------------------------------
// Existing function (with only added kernel launch code)
void KV_Cache_Plus_Flash_Forwarder::go_gpu(const __host__ float *q, 
                                             const __device__ float *k, 
                                             const __device__ float *v, 
                                             __host__ float* o, 
                                             int N, int d)
{
    int Bc = ceildiv(::constants::M, 4 * d);
    int Tc = ceildiv(N, Bc);

    // Copy query vector into constant memory.
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(::q, q, d * sizeof(float)));

    // Launch the new kernel.
    // (You may choose different block/grid dimensions.)
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    size_t shared_mem_size = blockDim.x * sizeof(SoftmaxAccum);
    kv_cache_flash_kernel<<<gridDim, blockDim, shared_mem_size>>>(k, v, o, N, d);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
*/

void KV_Cache_Plus_Flash_Forwarder::go_gpu(const __host__ float *q, 
                                             const __device__ float *k, 
                                             const __device__ float *v, 
                                             __host__ float* o, 
                                             int N, int d)
{
    return; // boilerplate here, idk what to do with this code lol
}