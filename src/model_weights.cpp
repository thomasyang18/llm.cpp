#include "model_weights.hpp"
#include "weight_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;

ModelWeights::ModelWeights(const GPTConfig& config) : _config(config) {
    _h.resize(config.n_layer);
}

void ModelWeights::load_weights(const std::filesystem::path& dir_path) {
    load_embeddings(dir_path);

    for (int i = 0; i < _config.n_layer; i++) {
        load_transformer_block(i, dir_path);
    }

    load_final_layer_norm(dir_path);
}

void ModelWeights::load_embeddings(const std::filesystem::path& dir_path) {
    try {
        _wte = weight_utils::load_2d_tensor(
            dir_path / "transformer.wte.weight.npy"
        );
        _wpe = weight_utils::load_2d_tensor(
            dir_path / "transformer.wpe.weight.npy"
        );

        if (!weight_utils::verify_tensor_shape(_wte, _config.vocab_size, _config.n_embd)) {
            throw std::runtime_error("Invalid shape for wte");
        }
        if (!weight_utils::verify_tensor_shape(_wpe, _config.block_size, _config.n_embd)) {
            throw std::runtime_error("Invalid shape for wpe");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load embeddings: " + std::string(e.what()));
    }
}

// internal hack to make this this nicer...
template <typename T>
std::filesystem::path operator+(std::filesystem::path path, T&& data)
// (accepting by value, we're going to modify!)
{
    path += std::forward<T>(data);
    return path;
}

void ModelWeights::load_transformer_block(int layer_idx, const std::filesystem::path& dir_path) {
    try {
        auto base_path = dir_path / "transformer.h.";
        base_path += std::to_string(layer_idx);
        auto& block = _h[layer_idx];

        // Load attention weights
        block.attn.c_attn_weight = weight_utils::load_2d_tensor(
            base_path + ".attn.c_attn.weight.npy"
        );
        block.attn.c_attn_bias = weight_utils::load_1d_tensor(
            base_path + ".attn.c_attn.bias.npy"
        );
        block.attn.c_proj_weight = weight_utils::load_2d_tensor(
            base_path + ".attn.c_proj.weight.npy"
        );
        block.attn.c_proj_bias = weight_utils::load_1d_tensor(
            base_path + ".attn.c_proj.bias.npy"
        );

        // Load MLP weights
        block.mlp.c_fc_weight = weight_utils::load_2d_tensor(
            base_path + ".mlp.c_fc.weight.npy"
        );
        block.mlp.c_fc_bias = weight_utils::load_1d_tensor(
            base_path + ".mlp.c_fc.bias.npy"
        );
        block.mlp.c_proj_weight = weight_utils::load_2d_tensor(
            base_path + ".mlp.c_proj.weight.npy"
        );
        block.mlp.c_proj_bias = weight_utils::load_1d_tensor(
            base_path + ".mlp.c_proj.bias.npy"
        );

        // Load layer norm weights
        block.ln_1.weight = weight_utils::load_1d_tensor(
            base_path + ".ln_1.weight.npy"
        );
        block.ln_1.bias = weight_utils::load_1d_tensor(
            base_path + ".ln_1.bias.npy"
        );
        block.ln_2.weight = weight_utils::load_1d_tensor(
            base_path + ".ln_2.weight.npy"
        );
        block.ln_2.bias = weight_utils::load_1d_tensor(
            base_path + ".ln_2.bias.npy"
        );
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load transformer block " + std::to_string(layer_idx) + ": " + std::string(e.what()));
    }
}

void ModelWeights::load_final_layer_norm(const std::filesystem::path& dir_path) {
    try {
        _ln_f.weight = weight_utils::load_1d_tensor(
            dir_path / "transformer.ln_f.weight.npy"
        );
        _ln_f.bias = weight_utils::load_1d_tensor(
            dir_path / "transformer.ln_f.bias.npy"
        );

        if (!weight_utils::verify_tensor_shape(_ln_f.weight, _config.n_embd)) {
            throw std::runtime_error("Invalid shape for ln_f.weight");
        }
        if (!weight_utils::verify_tensor_shape(_ln_f.bias, _config.n_embd)) {
            throw std::runtime_error("Invalid shape for ln_f.bias");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load final layer norm: " + std::string(e.what()));
    }
}
