#include "reference/forward_backward_naive.hpp"
#include <cmath>
#include <cassert>
#include <numeric>
#include <random>

#include <iostream>

Forward_BackwardNaive::Forward_BackwardNaive(ModelWeights& model) : _model(model) {}

// Forwards a single stream of tokens, returns a single token.
int Forward_BackwardNaive::forward(std::vector<int> tokens) {assert(false);} // do not go down this route

// =============== START OF MODEL ===============

Eigen::RowVectorXf softmax(const Eigen::RowVectorXf& logits) {
    Eigen::RowVectorXf exp_logits = (logits.array() - logits.maxCoeff()).exp();  // for numerical stability
    return exp_logits / exp_logits.sum();
}


void Forward_BackwardNaive::backward(std::vector<int> tokens) {
    assert(tokens.size() > 1 && "Need at least two tokens for backward pass");

    std::vector<int> input_tokens(tokens.begin(), tokens.end() - 1);
    std::vector<int> loss_tokens(tokens.begin() + 1, tokens.end());

    // Forward pass to get intermediate activations
    Eigen::MatrixXf x(input_tokens.size(), model().config().n_embd);

    for (int i = 0; i < input_tokens.size(); ++i) {
        x.row(i) = model().wte().row(input_tokens[i]) + model().wpe().row(i);
    }

    // Eigen::MatrixXf x_embed = x;

    std::vector<std::vector<Eigen::MatrixXf>> activations;

    for (int i = 0; i < model().blocks().size(); ++i) {
        activations.emplace_back(5); // vector of length 5

        const auto& block = model().blocks()[i];

        Eigen::MatrixXf x_ln1 = layer_norm(x, block.ln_1);
        Eigen::MatrixXf x_attn = causal_self_attention(x_ln1, block.attn);
        Eigen::MatrixXf x_ln2 = layer_norm(x_attn, block.ln_2);
        Eigen::MatrixXf x_mlp = mlp(x_ln2, block.mlp);

        activations[i][0] = x;
        activations[i][1] = x_ln1;
        activations[i][2] = x_attn;
        activations[i][3] = x_ln2;
        activations[i][4] = x_mlp;

        x = x_mlp;
    }

    Eigen::MatrixXf x_ln_f = layer_norm(x, model().ln_f());

    // Calculate loss
    float loss = 0.0f;
    for (int i = 0; i < loss_tokens.size(); ++i) {
        Eigen::RowVectorXf logits = x_ln_f.row(i) * model().lm_head().transpose();
        Eigen::RowVectorXf probs = softmax(logits);
        loss -= std::log(probs(loss_tokens[i]));
    }
    loss /= loss_tokens.size();

    // Backward pass
    Eigen::MatrixXf dx = Eigen::MatrixXf::Zero(x_ln_f.rows(), x_ln_f.cols());
    for (int i = 0; i < loss_tokens.size(); ++i) {
        Eigen::RowVectorXf logits = x_ln_f.row(i) * model().lm_head().transpose();
        Eigen::RowVectorXf probs = softmax(logits);
        Eigen::RowVectorXf dlogits = probs;
        dlogits(loss_tokens[i]) -= 1.0f;
        dx.row(i) = dlogits * model().lm_head();
    }
    dx /= loss_tokens.size();

    for (int i = model().blocks().size() - 1; i >= 0; --i) {
        const auto& block = model().blocks()[i];

        backward_layer_norm(dx, activations[i][4], block.ln_2, activations[i][3]);
        backward_mlp(dx, activations[i][3], block.mlp, activations[i][4]);

        backward_layer_norm(dx, activations[i][2], block.ln_1, activations[i][3]);
        backward_causal_self_attention(dx, activations[i][1], block.attn, activations[i][2], _dattn_scores[i], _dattn_values[i]);
    }
}

Eigen::MatrixXf forward_linear(Eigen::MatrixXf x, const Linear& linear) {
    x = x * linear.weight;
    x.rowwise() += linear.bias;
    return x;
}

Eigen::MatrixXf Forward_BackwardNaive::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attn) {
    int T = x.rows();
    int C = x.cols();

    int n_head = model().config().n_head;

    assert(C % n_head == 0);
    int head_dim = C / n_head; // Each head gets C/n_head features

    Eigen::MatrixXf qkv = forward_linear(x, attn.qkv);

    Eigen::MatrixXf q = qkv.leftCols(C);
    Eigen::MatrixXf k = qkv.middleCols(C, C);
    Eigen::MatrixXf v = qkv.rightCols(C);

    Eigen::MatrixXf y(T, C);

    for (int h = 0; h < n_head; h++) {
        Eigen::MatrixXf q_h = q.middleCols(h * head_dim, head_dim);
        Eigen::MatrixXf k_h = k.middleCols(h * head_dim, head_dim);
        Eigen::MatrixXf v_h = v.middleCols(h * head_dim, head_dim);

        Eigen::MatrixXf att_h = q_h * k_h.transpose();

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        att_h *= scale;

        for (int i = 0; i < T; i++) {
            for (int j = i + 1; j < T; j++) {
                att_h(i, j) = -std::numeric_limits<float>::infinity();
            }
        }

        for (int i = 0; i < T; i++) att_h.row(i) = softmax(att_h.row(i));
        Eigen::MatrixXf out_h = att_h * v_h;
        y.middleCols(h * head_dim, head_dim) = out_h;
    }

    y = forward_linear(y, attn.c_proj);
    return y;
}

Eigen::MatrixXf Forward_BackwardNaive::mlp(Eigen::MatrixXf x, const MLPWeights& mlp) {
    x = forward_linear(x, mlp.to_up);
    x = gelu(x);
    x = forward_linear(x, mlp.back_down);
    return x;
}

Eigen::MatrixXf Forward_BackwardNaive::layer_norm(Eigen::MatrixXf x, const LayerNormWeights& ln) {
    constexpr float eps = 1e-5;

    for (int i = 0; i < x.rows(); ++i) { // iterate over all tokens
        float mean = x.row(i).mean();
        float variance = (x.row(i).array() - mean).square().sum() / x.cols();
        float denom = 1.0f / std::sqrt(variance + eps);
        x.row(i) = ((x.row(i).array() - mean) * denom) * ln.gamma.array() + ln.beta.array();
    }
    return x;
}

float _gelu(float x) {
    const float GELU_SCALE = std::sqrt(2.0f / static_cast<float>(M_PI));
    float cube = 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + tanhf(GELU_SCALE * (x + cube)));
}

Eigen::MatrixXf Forward_BackwardNaive::gelu(Eigen::MatrixXf x) { return x.unaryExpr(&_gelu); }

void Forward_BackwardNaive::backward_causal_self_attention(Eigen::MatrixXf& dx, const Eigen::MatrixXf& x, const AttentionWeights& attn, const Eigen::MatrixXf& attn_out, const Eigen::MatrixXf& attn_scores, const Eigen::MatrixXf& attn_values) {
    // TODO: Implement backward pass for causal self-attention
}

void Forward_BackwardNaive::backward_mlp(Eigen::MatrixXf& dx, const Eigen::MatrixXf& x, const MLPWeights& mlp, const Eigen::MatrixXf& mlp_out) {
    // TODO: Implement backward pass for MLP
}

void Forward_BackwardNaive::backward_layer_norm(Eigen::MatrixXf& dx, const Eigen::MatrixXf& x, const LayerNormWeights& ln, const Eigen::MatrixXf& ln_out) {
    // TODO: Implement backward pass for layer normalization
}

void Forward_BackwardNaive::backward_gelu(Eigen::MatrixXf& dx, const Eigen::MatrixXf& x, const Eigen::MatrixXf& gelu_out) {
    // TODO: Implement backward pass for GELU activation
}
