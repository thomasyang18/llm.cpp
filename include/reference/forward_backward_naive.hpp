#pragma once
#include <Eigen/Dense>
#include "utils/model_weights.hpp"

#include <vector>

class Forward_BackwardNaive {
public:
    Forward_BackwardNaive(ModelWeights& model);

    int forward(std::vector<int> tokens);

    void backward(std::vector<int> tokens);

    const ModelWeights& model() { return _model; }

private:
    ModelWeights& _model;

    Eigen::MatrixXf causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention);
    Eigen::MatrixXf mlp(Eigen::MatrixXf x, const MLPWeights& mlp);
    Eigen::MatrixXf layer_norm(Eigen::MatrixXf x, const LayerNormWeights& ln);
    Eigen::MatrixXf gelu(Eigen::MatrixXf x);

    // Backward pass functions
    void backward_causal_self_attention(Eigen::MatrixXf& dx, const Eigen::MatrixXf& x, const AttentionWeights& attn, const Eigen::MatrixXf& attn_out, const Eigen::MatrixXf& attn_scores, const Eigen::MatrixXf& attn_values);
    void backward_mlp(Eigen::MatrixXf& dx, const Eigen::MatrixXf& x, const MLPWeights& mlp, const Eigen::MatrixXf& mlp_out);

    Eigen::MatrixXf backward_layer_norm(const Eigen::MatrixXf& gradient, const Eigen::MatrixXf& input, LayerNormWeights& ln);

    void backward_gelu(Eigen::MatrixXf& dx, const Eigen::MatrixXf& x, const Eigen::MatrixXf& gelu_out);
};
