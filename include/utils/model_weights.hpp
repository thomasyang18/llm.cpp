#pragma once
#include "utils/config.hpp"
#include "utils/layer_weights.hpp"
#include <vector>
#include <stdexcept>
#include <filesystem>

// greate c++ code
class Forward_BackwardNaive;

class ModelWeights {
public:
    explicit ModelWeights(const GPTConfig& config);

    // Load all weights from directory
    void load_weights(const std::filesystem::path& dir_path);

    void init_data_random();

    // Getters for different components
    const Eigen::MatrixXf& wte() const { return _wte; }
    const Eigen::MatrixXf& wpe() const { return _wpe; }
    const std::vector<TransformerBlockWeights>& blocks() const { return _h; }
    const LayerNormWeights& ln_f() const { return _ln_f; }

    // weight tying
    const Eigen::MatrixXf& lm_head() const {return _wte; }

    const GPTConfig& config() const {return _config; }

    friend class FriendClass; // this is so that I can train weights (e..g modify)
private:
    const GPTConfig& _config;

    // Token and position embeddings
    Eigen::MatrixXf _wte;    // [vocab_size, n_embd]
    Eigen::MatrixXf _wpe;    // [block_size, n_embd]

    // Transformer blocks
    std::vector<TransformerBlockWeights> _h;

    // Final layer norm
    LayerNormWeights _ln_f;

    // Helper methods for loading specific components
    void load_embeddings(const std::filesystem::path& dir_path);
    void load_transformer_block(int layer_idx, const std::filesystem::path& dir_path);
    void load_final_layer_norm(const std::filesystem::path& dir_path);
};
