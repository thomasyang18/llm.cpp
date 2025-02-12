#pragma once
#include <Eigen/Dense>

// For now, we just want to work in raw weights.
// Because in the future, we might write *cracked* passes!

// Stores weights and biases 
// Projects from [weight_dim1 => weight_dim 2]
// Bias must be of weight_dim2 
struct Linear {
    Eigen::MatrixXf weight;
    Eigen::RowVectorXf bias;

    // By default, Eigen vectors are *column wise*. 
    // Store as a 1xn row vector. 
    Linear(Eigen::MatrixXf weight, Eigen::VectorXf bias);

    // Empty default allowed.
    Linear();
};

struct AttentionWeights {
    // Combined QKV weights and bias
    // Recall that these were transposed from the original documentation (so they do not need to be transposed in the code).

    // [n_embd, 3 * n_embd] => q, k, v, then break it up from there. 
    Linear qkv; 

    // Output projection [n_embd -> n_embd]
    Linear c_proj;
};

struct MLPWeights {
    // Should project [n_embd -> 4 * n_embd -> n_embd]
    Linear to_up; 
    Linear back_down;
};

struct LayerNormWeights {
    // Recall that these were transposed from the original documentaiton.
    // (although we just manually implemented this lol...)

    Eigen::VectorXf gamma;         // [n_embd] this is gamma. (768, 1)
    Eigen::VectorXf beta;           // [n_embd] this is beta. (768, 1)
};

struct TransformerBlockWeights {
    LayerNormWeights ln_1;
    AttentionWeights attn;
    LayerNormWeights ln_2;
    MLPWeights mlp;
};
