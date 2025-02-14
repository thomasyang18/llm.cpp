/*
    We can definitely build on this, many of the ops here are extermely unoptimized, mainly for the sake of readability. 
    Gonna explore more operations.
*/

#pragma once

#include "inference.hpp"

class FlashAttention1Forwarder : public Forwarder {
public:
    FlashAttention1Forwarder(const ModelWeights& model);

    Eigen::MatrixXf causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention) override;

    int forward(std::vector<int> tokens) override;
};
