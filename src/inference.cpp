#include "inference.hpp"
#include <cmath>
#include <cassert>
#include <numeric>
#include <random>


#include <queue>
#include <utility>


#include <iostream>

Forwarder::Forwarder(const ModelWeights& model) : _model(model) {}

Eigen::RowVectorXf Forwarder::softmax(const Eigen::RowVectorXf& x) {
    Eigen::RowVectorXf exp_x = (x.array() - x.maxCoeff()).exp();  // for numerical stability
    return exp_x / exp_x.sum();
}

Eigen::MatrixXf Forwarder::forward_linear(Eigen::MatrixXf x, const Linear& linear) {
    x *= linear.weight;
    x.rowwise() += linear.bias;
    return x;
};

Eigen::MatrixXf Forwarder::mlp(Eigen::MatrixXf x, const MLPWeights& mlp) {
    x = forward_linear(x, mlp.to_up);
    x = gelu(x);
    x = forward_linear(x, mlp.back_down);
    return x;
}

Eigen::MatrixXf Forwarder::layer_norm(Eigen::MatrixXf x, const LayerNormWeights& ln) {
    // Explicitly write this independently, since I'm not that familiar with layernorm... matrix operations scary
    constexpr float eps = 1e-5;

    for (int i = 0; i < x.rows(); ++i) { // iterate over all tokens
        float mean = x.row(i).mean();
        float variance = (x.row(i).array() - mean).square().sum() / x.cols();
        float denom = 1.0f / std::sqrt(variance + eps);
        x.row(i) = ((x.row(i).array() - mean) * denom) * ln.gamma.array() + ln.beta.array();
    }
    return x;
}

namespace {
    // This is the exact gelu used by GPT2
    float _gelu(float x) {
        const float GELU_SCALE = std::sqrt(2.0f / static_cast<float>(M_PI));
        float cube = 0.044715f * x * x * x;
        return 0.5f * x * (1.0f + tanhf(GELU_SCALE * (x + cube)));
    }
}

Eigen::MatrixXf Forwarder::gelu(Eigen::MatrixXf x) { return x.unaryExpr(&_gelu); }

int Forwarder::sampler(Eigen::RowVectorXf logits) {
    // Multiply the last token's state with the transposed lm_head to get the logits.
    logits = softmax(logits);    
    
    // slightly more memory efficient (i think?) implementation of sampling than before
    std::priority_queue<std::pair<float, int>> lowest;
    std::vector<std::pair<float, int>> highest;

    const int THRESH = 10;
    for (int i = 0; i < model().config().vocab_size; ++i) {
        lowest.emplace(-logits[i], i); 
        while (lowest.size() > THRESH) lowest.pop();
    }

    float cumsum = 0; // maintains the total probabilities
    highest.resize(lowest.size());
    for (int i = int(highest.size()) - 1; i >= 0; --i) {
        highest[i] = lowest.top(); lowest.pop();
        highest[i].first = -highest[i].first;
        cumsum += highest[i].first;
    } 

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, cumsum);
    float rand_value = dist(gen);

    int last = 0;
    for (auto [prob, token]: highest) {
        last = token; 
        if ((rand_value -= prob) < 0) break;
    }
    return last;
}