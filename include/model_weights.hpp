#pragma once
#include "config.hpp"
#include "layer_weights.hpp"
#include <vector>
#include <stdexcept>

class ModelWeights {
public:
    explicit ModelWeights(const GPTConfig& config);

    // Load all weights from directory
    void load_weights(const std::string& dir_path);

    // Getters for different components
    const Eigen::MatrixXf& get_token_embedding() const { return wte; }
    const Eigen::MatrixXf& get_position_embedding() const { return wpe; }
    const std::vector<TransformerBlockWeights>& get_transformer_blocks() const { return h; }
    const LayerNormWeights& get_final_layer_norm() const { return ln_f; }

private:
    GPTConfig config;

    // Token and position embeddings
    Eigen::MatrixXf wte;    // [vocab_size, n_embd]
    Eigen::MatrixXf wpe;    // [block_size, n_embd]

    // Transformer blocks
    std::vector<TransformerBlockWeights> h;

    // Final layer norm
    LayerNormWeights ln_f;

    // Helper methods for loading specific components
    void load_embeddings();
    void load_transformer_block(int layer_idx);
    void load_final_layer_norm();
};
