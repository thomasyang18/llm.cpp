#pragma once
#include <Eigen/Dense>

// For now, we just want to work in raw weights.
// Because in the future, we might write *cracked* passes!

struct AttentionWeights {
    // Combined QKV weights and bias
    Eigen::MatrixXf c_attn_weight;  // [n_embd, 3 * n_embd]
    Eigen::VectorXf c_attn_bias;    // [3 * n_embd]
    
    // Output projection
    Eigen::MatrixXf c_proj_weight;  // [n_embd, n_embd]
    Eigen::VectorXf c_proj_bias;    // [n_embd]
};

struct MLPWeights {
    // First linear layer
    Eigen::MatrixXf c_fc_weight;    // [n_embd, 4 * n_embd]
    Eigen::VectorXf c_fc_bias;      // [4 * n_embd]
    
    // Second linear layer
    Eigen::MatrixXf c_proj_weight;  // [4 * n_embd, n_embd]
    Eigen::VectorXf c_proj_bias;    // [n_embd]
};

struct LayerNormWeights {
    Eigen::VectorXf weight;         // [n_embd] this is gamma. (768, 1)
    Eigen::VectorXf bias;           // [n_embd] this is beta. (768, 1)
};

struct TransformerBlockWeights {
    LayerNormWeights ln_1;
    AttentionWeights attn;
    LayerNormWeights ln_2;
    MLPWeights mlp;
};
