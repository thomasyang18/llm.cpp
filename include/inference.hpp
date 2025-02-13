#pragma once
#include <Eigen/Dense>
#include "utils/model_weights.hpp"

#include <vector> 

// forward_naive.{cpp, hpp}

// I think K-V cache and kernel fusion operate on different layers, so we could define mixins for optimizations. 
// At the same time... I don't know the space well enough. Should first just implement some passes, and then 
// if it gets too cancer, THEN refactor. 
class Forwarder {
public:
    Forwarder(const ModelWeights& model);
    
    virtual ~Forwarder() = default;  // Virtual destructor for proper cleanup in derived classes

    const ModelWeights& model() { return _model; }

    // Attention is the main shenanigan we modify 
    virtual Eigen::MatrixXf causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention) = 0;
    
    // We can also do some optimizations here... k/v cache etc. 
    virtual int forward(std::vector<int> tokens) = 0;

    // Useful repeat aux util functions to make it more explicit 
    Eigen::MatrixXf forward_linear(Eigen::MatrixXf x, const Linear& linear);
    Eigen::RowVectorXf softmax(const Eigen::RowVectorXf& x);

    // These don't really change at all; these are not the bottleneck, ever. 
    Eigen::MatrixXf mlp(Eigen::MatrixXf x, const MLPWeights& mlp);
    Eigen::MatrixXf layer_norm(Eigen::MatrixXf x, const LayerNormWeights& ln);
    Eigen::MatrixXf gelu(Eigen::MatrixXf x);

private:
    const ModelWeights& _model;
};
