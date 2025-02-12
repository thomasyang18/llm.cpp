#include "model_weights.hpp"
#include "weight_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;

ModelWeights::ModelWeights(const GPTConfig& config) : config(config) {
    h.resize(config.n_layer);
}

void ModelWeights::load_weights(const std::string& dir_path) {
    config.weights_path = dir_path;

    load_embeddings();

    for (int i = 0; i < config.n_layer; i++) {
        load_transformer_block(i);
    }

    load_final_layer_norm();
}

void ModelWeights::load_embeddings() {
    try {
        wte = weight_utils::load_2d_tensor(
            fs::path(config.weights_path) / "transformer.wte.weight.npy"
        );
        wpe = weight_utils::load_2d_tensor(
            fs::path(config.weights_path) / "transformer.wpe.weight.npy"
        );

        if (!weight_utils::verify_tensor_shape(wte, config.vocab_size, config.n_embd)) {
            throw std::runtime_error("Invalid shape for wte");
        }
        if (!weight_utils::verify_tensor_shape(wpe, config.block_size, config.n_embd)) {
            throw std::runtime_error("Invalid shape for wpe");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load embeddings: " + std::string(e.what()));
    }
}

void ModelWeights::load_transformer_block(int layer_idx) {
    try {
        auto base_path = fs::path(config.weights_path) / "transformer.h." / std::to_string(layer_idx);
        auto& block = h[layer_idx];

        // Load attention weights
        block.attn.c_attn_weight = weight_utils::load_2d_tensor(
            base_path / "attn.c_attn.weight.npy"
        );
        block.attn.c_attn_bias = weight_utils::load_1d_tensor(
            base_path / "attn.c_attn.bias.npy"
        );
        block.attn.c_proj_weight = weight_utils::load_2d_tensor(
            base_path / "attn.c_proj.weight.npy"
        );
        block.attn.c_proj_bias = weight_utils::load_1d_tensor(
            base_path / "attn.c_proj.bias.npy"
        );

        // Load MLP weights
        block.mlp.c_fc_weight = weight_utils::load_2d_tensor(
            base_path / "mlp.c_fc.weight.npy"
        );
        block.mlp.c_fc_bias = weight_utils::load_1d_tensor(
            base_path / "mlp.c_fc.bias.npy"
        );
        block.mlp.c_proj_weight = weight_utils::load_2d_tensor(
            base_path / "mlp.c_proj.weight.npy"
        );
        block.mlp.c_proj_bias = weight_utils::load_1d_tensor(
            base_path / "mlp.c_proj.bias.npy"
        );

        // Load layer norm weights
        block.ln_1.weight = weight_utils::load_1d_tensor(
            base_path / "ln_1.weight.npy"
        );
        block.ln_1.bias = weight_utils::load_1d_tensor(
            base_path / "ln_1.bias.npy"
        );
        block.ln_2.weight = weight_utils::load_1d_tensor(
            base_path / "ln_2.weight.npy"
        );
        block.ln_2.bias = weight_utils::load_1d_tensor(
            base_path / "ln_2.bias.npy"
        );
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load transformer block " + std::to_string(layer_idx) + ": " + std::string(e.what()));
    }
}

void ModelWeights::load_final_layer_norm() {
    try {
        ln_f.weight = weight_utils::load_1d_tensor(
            fs::path(config.weights_path) / "transformer.ln_f.weight.npy"
        );
        ln_f.bias = weight_utils::load_1d_tensor(
            fs::path(config.weights_path) / "transformer.ln_f.bias.npy"
        );

        if (!weight_utils::verify_tensor_shape(ln_f.weight, config.n_embd)) {
            throw std::runtime_error("Invalid shape for ln_f.weight");
        }
        if (!weight_utils::verify_tensor_shape(ln_f.bias, config.n_embd)) {
            throw std::runtime_error("Invalid shape for ln_f.bias");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load final layer norm: " + std::string(e.what()));
    }
}
