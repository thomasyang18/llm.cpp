#include "reference/forward_backward_naive.hpp"
#include <cmath>
#include <cassert>
#include <numeric>
#include <random>

#include <iostream>

template<typename T>
bool assert_not_nan(T container) {
#ifdef DEBUG
    for (int i = 0; i < container.rows(); ++i) {
        for (int j = 0; j < container.cols(); ++j) {
            if (container(i, j) != container(i, j)) return false;
        }
    }
#endif
    return true;
}

Eigen::RowVectorXf softmax(const Eigen::RowVectorXf& logits) {
    Eigen::RowVectorXf exp_logits = (logits.array() - logits.maxCoeff()).exp();  // for numerical stability
    return exp_logits / exp_logits.sum();
}

namespace sampling {

    int top_k_sample(const Eigen::RowVectorXf& logits, int k) {
        // Step 1: Apply softmax
        Eigen::RowVectorXf probs = softmax(logits);

        // Step 2: Create a vector of indices sorted by probability (descending)
        std::vector<int> sorted_indices(probs.cols());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);  // Create a list of indices [0, 1, 2, ..., vocab_size-1]

        // Sort indices based on their corresponding probability values in descending order
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
            return probs[a] > probs[b];
        });

        // Step 3: Top-k random sampling
        // Take the top-k highest probabilities
        std::vector<int> top_k_indices(sorted_indices.begin(), sorted_indices.begin() + k);

        // Create a discrete distribution based on the top-k probabilities
        std::vector<float> top_k_probs(top_k_indices.size());
        for (size_t i = 0; i < top_k_indices.size(); ++i) {
            top_k_probs[i] = probs[top_k_indices[i]];
        }

        std::discrete_distribution<> dist(top_k_probs.begin(), top_k_probs.end());

        std::mt19937 gen(std::random_device{}());
        int sampled_index = dist(gen);

        int sampled_token = top_k_indices[sampled_index];

        return sampled_token;
    }
}

Forward_BackwardNaive::Forward_BackwardNaive(ModelWeights& model) : _model(model) {}

// Forwards a single stream of tokens, returns a single token.
int Forward_BackwardNaive::forward(std::vector<int> tokens) {assert(false);} // do not go down this route 

void Forward_BackwardNaive::backward(std::vector<int> tokens) {
    assert(tokens.size() > 1 && "Need at least two tokens for backward pass");

    std::vector<int> input_tokens(tokens.begin(), tokens.end() - 1);
    int target_token = tokens.back();

    // Forward pass to get intermediate activations
    Eigen::MatrixXf x(input_tokens.size(), model().config().n_embd);
    for (int i = 0; i < input_tokens.size(); ++i) {
        x.row(i) = model().wte().row(input_tokens[i]) + model().wpe().row(i);
    }

    std::vector<Eigen::MatrixXf> activations;
    activations.push_back(x);

    for (const auto& block : model().blocks()) {
        x = layer_norm(x, block.ln_1);
        activations.push_back(x);
        x = causal_self_attention(x, block.attn);
        activations.push_back(x);
        x = layer_norm(x, block.ln_2);
        activations.push_back(x);
        x = mlp(x, block.mlp);
        activations.push_back(x);
    }

    x = layer_norm(x, model().ln_f());
    activations.push_back(x);

    // Backward pass
    Eigen::MatrixXf dx = (x * model().lm_head().transpose()).row(input_tokens.size() - 1);
    dx = dx - model().wte().row(target_token);

    for (int i = model().blocks().size() - 1; i >= 0; --i) {
        const auto& block = model().blocks()[i];

        backward_layer_norm(dx, activations[4 * i + 3], block.ln_2, activations[4 * i + 4]);
        backward_mlp(dx, activations[4 * i + 2], block.mlp, activations[4 * i + 3]);

        backward_layer_norm(dx, activations[4 * i + 1], block.ln_1, activations[4 * i + 2]);
        backward_causal_self_attention(dx, activations[4 * i], block.attn, activations[4 * i + 1], _dattn_scores[i], _dattn_values[i]);
    }
}

Eigen::MatrixXf forward_linear(Eigen::MatrixXf x, const Linear& linear) {
    x = x * linear.weight;
    x.rowwise() += linear.bias;
    return x;
};

Eigen::MatrixXf Forward_BackwardNaive::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attn) {
    // x: [T, C] where T = sequence length and C = embedding dimension (n_embd)
    int T = x.rows();
    int C = x.cols();
    // assert(T == model().config().block_size);
    int n_head = model().config().n_head;

    assert(C % n_head == 0);
    int head_dim = C / n_head; // Each head gets C/n_head features

    // 1. Compute qkv: project x with weight and add bias.
    //    attn.c_attn_weight should have shape [C, 3*C] and attn.c_attn_bias shape [3*C].
    //    The result qkv is of shape [T, 3*C].

    Eigen::MatrixXf qkv = forward_linear(x, attn.qkv);

    // 2. Split qkv into q, k, v. Each will have shape [T, C].
    Eigen::MatrixXf q = qkv.leftCols(C);
    Eigen::MatrixXf k = qkv.middleCols(C, C);
    Eigen::MatrixXf v = qkv.rightCols(C);

    // 3. Prepare an output accumulator for all heads.
    Eigen::MatrixXf y(T, C);
    y.setZero();

    assert(assert_not_nan(q) && " q matrix");
    assert(assert_not_nan(k) && " k matrix");
    assert(assert_not_nan(v) && " v matrix");

    /*
        Okay I should probably try to understand the actual transpose magic going on when implementing attention, but for a first pass
        this isn't so bad.

        From what I understand, the "actual" embeddings are just C / n_heads long.
        Partition into blocks of [T, actual_embd].

        Then operate on those indivdually. Am I correct?
    */

    for (int h = 0; h < n_head; h++) {
        // Extract block corresponding to head h.
        // Each block has shape [T, head_dim].
        Eigen::MatrixXf q_h = q.middleCols(h * head_dim, head_dim);
        Eigen::MatrixXf k_h = k.middleCols(h * head_dim, head_dim);
        Eigen::MatrixXf v_h = v.middleCols(h * head_dim, head_dim);

        // 5. Compute the attention scores for head h.
        //    att_h = q_h * k_h^T, so att_h has shape [T, T].
        Eigen::MatrixXf att_h = q_h * k_h.transpose();

        // Scale the attention scores by sqrt(head_dim)
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        att_h *= scale;

        // // 6. Apply a causal mask: for each row i, zero out (or set to -infinity) any column j > i.
        // //    Here we loop over rows and columns.
        // recall that att(i, j) is how much token i attends to token j.
        for (int i = 0; i < T; i++) {
            for (int j = i + 1; j < T; j++) {
                att_h(i, j) = -std::numeric_limits<float>::infinity();
            }
        }

        // 7. Apply softmax to each row of att_h.
        //    For numerical stability, subtract the max in each row before exponentiating.
        for (int i = 0; i < T; i++) att_h.row(i) = softmax(att_h.row(i));

        // 8. Compute the weighted sum of the values: out_h = att_h * v_h.
        //    out_h has shape [T, head_dim].
        Eigen::MatrixXf out_h = att_h * v_h;

        // 9. Write out_h into the appropriate middleCols of the output y.
        y.middleCols(h * head_dim, head_dim) = out_h;
    }

    assert(assert_not_nan(y) && " before attention projection ");

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
