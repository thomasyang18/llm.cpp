#pragma once

#include "inference.hpp"
#include <optional>
#include <cuda_runtime.h>
#include <utility>

class KV_Cache_Manager_GPU {
    struct kv_pair {
        __device__ float * k, * v;
        size_t size = 0; // increment by bits of config().d()
    };

    std::vector<std::vector<kv_pair>> _cache;

    const GPTConfig& _config;

public:
    KV_Cache_Manager_GPU(const GPTConfig& config, int max_tokens);
    ~KV_Cache_Manager_GPU(); // manages some CUDA memory

    // CUDA is yelling at me idk why 
    struct const_kv_pair_return {
        const __device__ float* k, * v;
    };

    inline const_kv_pair_return ask(int i, int j) const {
        // no bounds checking, care
        return const_kv_pair_return{.k = _cache[i][j].k, 
                                    .v = _cache[i][j].v};
    }

    inline const GPTConfig& config() {return _config;}

    enum class KV_Type { K, V };  

    // We mark as *host*, because for a simplified implementation we ONLY want to do the attention
    // step on GPU. I have some nice passes I want to do here, potentially. 

    // The KV_Cache is entirely on GPU, the rest of the model weights... should be fine? I mean it's less than a GB, at least. 
    template<KV_Type T>
    void push_back(int layer, const __host__ float* data) {
        // https://stackoverflow.com/questions/41011900/equivalent-ternary-operator-for-constexpr-if
        // this seems interesting
        // but just do it the "dumb" way 
        
        // copy [head1, 64][head2, 64][head3, 64] etc.
        if constexpr (T == KV_Type::K) {
            for (int j = 0; j < config().n_head; ++j) 
                memcpy(_cache[layer][j].k, data + j * config().d(), config().d());
        } else if constexpr (T == KV_Type::V) {
            for (int j = 0; j < config().n_head; ++j) 
                memcpy(_cache[layer][j].v, data + j * config().d(), config().d());
        } else static_assert(T == KV_Type::K); // because static_assert(false) won't work...
    }
};

// This is a use once class. 
// Kind of dumb, but 
class KV_Cache_Plus_Flash_Forwarder : public Forwarder {
    // Row major appendable vector for KV cache, representing a dynamic matrix.
    // Just going to make this 

    // oh god, horrible API, living with my decisions

    struct state_that_i_wish_i_had_the_foresight_to_encode_properly {
        int layer = 0; 
        int tokens = 0;
    } glob_state;

    // The idea here is to repeatedly de_allocate and allocate this data structure :/ 
    // Again, hacks, but the idea is more important. 
    std::optional<KV_Cache_Manager_GPU> kv_cache;

public:
    KV_Cache_Plus_Flash_Forwarder(const ModelWeights& model);

    Eigen::MatrixXf causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention) override;

    // uhh, this is ugly but whatever. If you call this, you will get a runtime error.
    int forward(std::vector<int> tokens);

    // q is a RowVector(d).
    // k, v are Matrix(N, d), row major. 
    // o is RowVector(d).
    void go_gpu(const __host__ float *q, 
                const __device__ float *k, 
                const __device__ float *v, 
                __host__ float*  o, 
                int N, int d);

    // Call this instead.
    // We currently do not support "sliding window" kv-caching, so tokens.size() + THRESH <= block_size.
    std::vector<int> forward_until(std::vector<int> tokens, int THRESH); 
};
