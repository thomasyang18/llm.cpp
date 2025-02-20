#pragma once
#include <Eigen/Dense>
#include "utils/model_weights.hpp"

#include <vector>

class Forward_BackwardNaive {
public:
    Forward_BackwardNaive(ModelWeights& model);
    
    void backward(std::vector<int> tokens, float temp);

    const ModelWeights& model() { return _model; }

private:
    ModelWeights& _model;

    Eigen::MatrixXf causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention);
    Eigen::MatrixXf mlp(Eigen::MatrixXf x, const MLPWeights& mlp);
    Eigen::MatrixXf layer_norm(Eigen::MatrixXf x, const LayerNormWeights& ln);
    Eigen::MatrixXf gelu(Eigen::MatrixXf x);

    Eigen::MatrixXf backward_layer_norm(const Eigen::MatrixXf& gradient, const Eigen::MatrixXf& input, LayerNormWeights& ln);
    Eigen::MatrixXf backward_mlp(const Eigen::MatrixXf& gradient, const Eigen::MatrixXf& input, MLPWeights& mlp);
    Eigen::MatrixXf backward_causal_self_attention(const Eigen::MatrixXf& gradient, const Eigen::MatrixXf& input, AttentionWeights& attn);    
};
