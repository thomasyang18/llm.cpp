// One of the most intuitive observations, once you understand that 
// the transformer architecture doesn't *really* depend on the sequence length at all.
// All the weight matrices operate on independent tokens, and the "kernel smoothing" bit just works.
// This is a really creative architecture and algorithm.

/*
also, if the weight position embeddings are just some sine wave and not trained alongside the model, 
then a model can have infinite context length can it not. ig hardware concerns tho.ADJ_OFFSET_SINGLESHOT

plus no way infintie context length is good lol too much context = bad.(RoPE?)
*/

#pragma once

#include "inference.hpp"

class KVCachingForwarder : public Forwarder {
    struct kv_pair {
        Eigen::RowVectorXf k;
        Eigen::RowVectorXf v;
    };

    std::vector<std::vector<kv_pair>> kv_cache; 

public:
    KVCachingForwarder(const ModelWeights& model);

    Eigen::MatrixXf causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention) override;

    // uhh, this is ugly but whatever. If you call this, you will get a runtime error.
    int forward(std::vector<int> tokens);

    // Call this instead.
    // We currently do not support "sliding window" kv-caching, so tokens.size() + THRESH <= block_size.
    std::vector<int> forward_until(std::vector<int> tokens, int THRESH); 
};
