#pragma once
#include <Eigen/Dense>

// For now, we just want to work in raw weights.
// Because in the future, we might write *cracked* passes!
// And we don't want to write too tightly coupled logic with the classes. 
// So these are just all dataclasses. 

// Stores weights and biases 
// Projects from [weight_dim1 => weight_dim 2]
// Bias must be of weight_dim2 
struct Linear {
    Eigen::MatrixXf weight;
    Eigen::RowVectorXf bias;
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
    Eigen::RowVectorXf gamma;
    Eigen::RowVectorXf beta;           
};

struct TransformerBlockWeights {
    LayerNormWeights ln_1;
    AttentionWeights attn;
    LayerNormWeights ln_2;
    MLPWeights mlp;
};
