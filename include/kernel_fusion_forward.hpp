#pragma once

#include "inference.hpp"

class KernelFusionForwarder : public Forwarder {
public:
    KernelFusionForwarder(const ModelWeights& model);

    Eigen::MatrixXf causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention) override;

    int forward(std::vector<int> tokens) override;
};
