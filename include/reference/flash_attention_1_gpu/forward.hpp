#pragma once

#include "inference.hpp"

class FlashAttention1ForwarderGPU : public Forwarder {
public:
    FlashAttention1ForwarderGPU(const ModelWeights& model);

    Eigen::MatrixXf causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention) override;

    void go_gpu(const float *q, const float *k, const float *v, float* o, int N, int d, int M);

    int forward(std::vector<int> tokens) override;
};
