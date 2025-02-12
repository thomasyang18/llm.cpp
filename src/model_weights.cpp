// src/model_weights.cpp
#include "model_weights.hpp"
#include "weight_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;

ModelWeights::ModelWeights(const GPTConfig& config) : config(config) {
    h.resize(config.n_layer);
}

bool ModelWeights::load_weights(const std::string& dir_path) {
    config.weights_path = dir_path;
    
    if (!load_embeddings()) return false;
    
    for (int i = 0; i < config.n_layer; i++) {
        if (!load_transformer_block(i)) return false;
    }
    
    if (!load_final_layer_norm()) return false;
    
    return true;
}

bool ModelWeights::load_embeddings() {
    try {
        wte = weight_utils::load_2d_tensor(
            fs::path(config.weights_path) / "transformer.wte.weight.npy"
        );
        wpe = weight_utils::load_2d_tensor(
            fs::path(config.weights_path) / "transformer.wpe.weight.npy"
        );
        
        return weight_utils::verify_tensor_shape(wte, config.vocab_size, config.n_embd) &&
               weight_utils::verify_tensor_shape(wpe, config.block_size, config.n_embd);
    } catch (const std::exception& e) {
        return false;
    }
}

bool ModelWeights::load_transformer_block(int layer_idx) {
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
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool ModelWeights::load_final_layer_norm() {
    try {
        ln_f.weight = weight_utils::load_1d_tensor(
            fs::path(config.weights_path) / "transformer.ln_f.weight.npy"
        );
        ln_f.bias = weight_utils::load_1d_tensor(
            fs::path(config.weights_path) / "transformer.ln_f.bias.npy"
        );
        
        return weight_utils::verify_tensor_shape(ln_f.weight, config.n_embd) &&
               weight_utils::verify_tensor_shape(ln_f.bias, config.n_embd);
    } catch (const std::exception& e) {
        return false;
    }
}