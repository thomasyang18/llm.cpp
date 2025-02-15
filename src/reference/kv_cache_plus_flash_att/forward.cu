#pragma once 
#include "reference/kv_cache_plus_flash_att/forward.cuh"
#include "utils/cuda_utils.cuh"

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



namespace {
    namespace constants {
        /* HIGHLY UNRIGOROUS PERF ENGINEERING

        ChatGPT said:

        The Tesla T4 GPU has the following cache details:

            L1 Cache: 64 KB per SM (Streaming Multiprocessor)
            so... M = 64KB :smiley:
        */

        constexpr int M = (1<<16);
        // this is maximum, single-headed attention embedding value. 
        // I mean we can theoretically make it larger to support arbitrary gpt2 models but...
        // Let's just work with the small for now.
        constexpr int MAX_D = 64;  
    }

    constexpr int ceildiv(int a, int b) {
        return (a + b - 1) / b;
    }
}

namespace { // nice that anonymous namespaces work in CUDA still    
   __constant__ float q[MAX_D]; // global cache for q


}   


// q is a RowVector(d).
// k, v are Matrix(N, d), row major. 
// o is RowVector(d).
void KV_Cache_Plus_Flash_Forwarder::go_gpu(const __host__ float *q, 
            const __device__ float *k, 
            const __device__ float *v, 
            __host__ float* o, 
            int N, int d)
{
    int Bc = ceildiv(::constants::M, 4 * d);
    int Tc = ceildiv(N, Bc);

    

}