#include "forward_naive.hpp"
#include <cmath>

ForwardNaive::ForwardNaive(const ModelWeights& weights) : _weights(weights) {}

Eigen::MatrixXf ForwardNaive::forward(const Eigen::MatrixXi& idx) {
    int B = idx.rows(); // batch size
    int T = idx.cols(); // sequence length

    // Token and position embeddings
    Eigen::MatrixXf tok_emb = _weights.wte() * idx.cast<float>();
    Eigen::MatrixXf pos_emb = _weights.wpe().topRows(T);
    Eigen::MatrixXf x = tok_emb + pos_emb.replicate(B, 1);

    // Forward through transformer blocks
    for (const auto& block : _weights.blocks()) {
        x = x + causal_self_attention(layer_norm(x, block.ln_1));
        x = x + mlp(layer_norm(x, block.ln_2));
    }

    // Final layer norm
    x = layer_norm(x, _weights.ln_f());

    return x;
}

Eigen::MatrixXf ForwardNaive::causal_self_attention(const Eigen::MatrixXf& x) {
    int B = x.rows(); // batch size
    int T = x.cols(); // sequence length
    int C = x.rows(); // embedding dimensionality (n_embd)

    // Calculate query, key, values for all heads in batch and move head forward to be the batch dim
    Eigen::MatrixXf qkv = x * _weights.attn.c_attn_weight + _weights.attn.c_attn_bias.transpose().replicate(B, 1);
    Eigen::MatrixXf q = qkv.leftCols(C);
    Eigen::MatrixXf k = qkv.middleCols(C, C);
    Eigen::MatrixXf v = qkv.rightCols(C);

    // Reshape and transpose for multi-head attention
    q = q.reshape(B, _weights.n_head, T, C / _weights.n_head).transpose(1, 2);
    k = k.reshape(B, _weights.n_head, T, C / _weights.n_head).transpose(1, 2);
    v = v.reshape(B, _weights.n_head, T, C / _weights.n_head).transpose(1, 2);

    // Manual implementation of attention
    Eigen::MatrixXf att = (q * k.transpose(2, 3)) * (1.0 / std::sqrt(k.cols()));
    att = att.array().select(att.array() + 1, att.array() < 0);
    att = att.array().exp() / att.array().exp().rowwise().sum();
    Eigen::MatrixXf y = att * v;

    // Re-assemble all head outputs side by side
    y = y.transpose(1, 2).reshape(B, T, C);

    // Output projection
    y = y * _weights.attn.c_proj_weight + _weights.attn.c_proj_bias.transpose().replicate(B, 1);

    return y;
}

Eigen::MatrixXf ForwardNaive::mlp(const Eigen::MatrixXf& x) {
    Eigen::MatrixXf x1 = x * _weights.mlp.c_fc_weight + _weights.mlp.c_fc_bias.transpose().replicate(x.rows(), 1);
    x1 = gelu(x1);
    x1 = x1 * _weights.mlp.c_proj_weight + _weights.mlp.c_proj_bias.transpose().replicate(x.rows(), 1);
    return x1;
}

Eigen::MatrixXf ForwardNaive::layer_norm(const Eigen::MatrixXf& x, const LayerNormWeights& ln) {
    Eigen::MatrixXf mean = x.rowwise().mean();
    Eigen::MatrixXf variance = ((x.array() - mean.array()).square().rowwise().sum() / x.cols()).sqrt();
    return (x.array() - mean.array()) / variance.array() * ln.weight.array() + ln.bias.array();
}

Eigen::MatrixXf ForwardNaive::gelu(const Eigen::MatrixXf& x) {
    return 0.5 * x.array() * (1.0 + (x.array() + 0.044715 * x.array().pow(3)).tanh());
}
