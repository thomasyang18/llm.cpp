#pragma once
#include <Eigen/Dense>
#include "model_weights.hpp"

#include <vector> 
#include <optional>

// We're only optimized for inference; so we can pass r-values everywhere 

class ForwardNaive {
public:
    ForwardNaive(const ModelWeights& model);

    int forward(std::vector<int> tokens);

    const ModelWeights& model() { return _model; }

private:
    const ModelWeights& _model;

    Eigen::MatrixXf causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention, std::optional<size_t> seq_length);
    Eigen::MatrixXf mlp(Eigen::MatrixXf x, const MLPWeights& mlp);
    Eigen::MatrixXf layer_norm(Eigen::MatrixXf x, const LayerNormWeights& ln);
    Eigen::MatrixXf gelu(Eigen::MatrixXf x);
};
