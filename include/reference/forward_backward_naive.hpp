// Reference implementation for a correct pass (this took a lot of wrangling to get)
// Will **not** intergrate with inference.{hpp, cpp} and the rest of the common infra; this is a baseline impl
// that should be referenced, that's why it's in its seperate directory. 
// Can be integrated easily for standalone testing. 

#pragma once
#include <Eigen/Dense>
#include "utils/model_weights.hpp"

#include <vector> 

class Forward_BackwardNaive {
public:
    Forward_BackwardNaive(const ModelWeights& model);

    int forward(std::vector<int> tokens);

    const ModelWeights& model() { return _model; }

private:
    const ModelWeights& _model;

    Eigen::MatrixXf causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention);
    Eigen::MatrixXf mlp(Eigen::MatrixXf x, const MLPWeights& mlp);
    Eigen::MatrixXf layer_norm(Eigen::MatrixXf x, const LayerNormWeights& ln);
    Eigen::MatrixXf gelu(Eigen::MatrixXf x);
};
