#pragma once
#include <Eigen/Dense>
#include "model_weights.hpp"

class ForwardNaive {
public:
    ForwardNaive(const ModelWeights& weights);

    Eigen::MatrixXf forward(const Eigen::MatrixXi& idx);

private:
    const ModelWeights& _weights;

    Eigen::MatrixXf causal_self_attention(const Eigen::MatrixXf& x);
    Eigen::MatrixXf mlp(const Eigen::MatrixXf& x);
    Eigen::MatrixXf layer_norm(const Eigen::MatrixXf& x, const LayerNormWeights& ln);
    Eigen::MatrixXf gelu(const Eigen::MatrixXf& x);
};
