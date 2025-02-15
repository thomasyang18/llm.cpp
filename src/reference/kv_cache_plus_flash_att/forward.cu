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

template<KV_Cache_Manager_GPU::KV_Type T>
void KV_Cache_Manager_GPU::push_back(int layer, const __host__ float* data) {
    // https://stackoverflow.com/questions/41011900/equivalent-ternary-operator-for-constexpr-if
    // this seems interesting
    // but just do it the "dumb" way 
    
    // copy [head1, 64][head2, 64][head3, 64] etc.
    if constexpr (T == K) {
        for (int j = 0; j < config().n_head; ++j) 
            memcpy(_cache[layer][j].k, data + j * config().d(), config().d());
    } else if constexpr (T == V) {
        for (int j = 0; j < config().n_head; ++j) 
            memcpy(_cache[layer][j].v, data + j * config().d(), config().d());
    } else static_assert(T < 0); // because static_assert(false) won't work...
}


