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
