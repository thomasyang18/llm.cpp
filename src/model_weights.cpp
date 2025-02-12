#include "model_weights.hpp"
#include "weight_utils.hpp"
#include <filesystem>

namespace fs = std::filesystem;


/*
    SOME TENSORS RANDOMLY NEED TRANSPOSITION!!!

    transposed_layers = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}

    They didn't transpose in the python script, so doing this manually....
*/

// internal hack to make this this nicer...
template <typename T>
fs::path operator+(fs::path path, T&& data)
// (accepting by value, we're going to modify!)
{
    path += std::forward<T>(data);
    return path;
}

ModelWeights::ModelWeights(const GPTConfig& config) : _config(config) {
    _h.resize(config.n_layer);
}

void ModelWeights::load_weights(const fs::path& dir_path) {
    load_embeddings(dir_path);

    for (int i = 0; i < _config.n_layer; i++) {
        load_transformer_block(i, dir_path);
    }

    load_final_layer_norm(dir_path);
}

void ModelWeights::load_embeddings(const fs::path& dir_path) {
    try {
        _wte = weight_utils::load_2d_tensor(
            dir_path / "transformer.wte.weight.npy"
        );
        _wpe = weight_utils::load_2d_tensor(
            dir_path / "transformer.wpe.weight.npy"
        );

        weight_utils::assert_tensor_shape(_wte, _config.vocab_size, _config.n_embd, "wte");
        weight_utils::assert_tensor_shape(_wpe, _config.block_size, _config.n_embd, "wpe");
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load embeddings: " + std::string(e.what()));
    }
}

void assert_linear(const Linear& input, int in_dim, int out_dim, std::string name) {
    weight_utils::assert_tensor_shape(input.weight, in_dim, out_dim, name);
    weight_utils::assert_vector_shape(input.bias, out_dim, name);
}

void assert_layer_norm(const LayerNormWeights& input, int size, std::string name) {
    weight_utils::assert_vector_shape(input.beta, size, name);
    weight_utils::assert_vector_shape(input.gamma, size, name);
}

void ModelWeights::load_transformer_block(int layer_idx, const fs::path& dir_path) {
    try {
        auto base_path = dir_path / "transformer.h.";
        base_path += std::to_string(layer_idx);
        auto& block = _h[layer_idx];

        // Load attention weights

        block.attn.qkv = Linear(
            weight_utils::load_2d_tensor(base_path + ".attn.c_attn.weight.npy"),
            weight_utils::load_1d_tensor(base_path + ".attn.c_attn.bias.npy")
        );

        // ===TRANSPOSED====
        // "attn.c_attn.weight"

        // block.attn.qkv.weight.transposeInPlace();

        assert_linear(block.attn.qkv, config().n_embd, config().n_embd * 3, "attention block #" + std::to_string(layer_idx));


        // The previous embedding space, logically speaking, is [n_heads, smol_embd = n_embd / n_heads] dimensions each. They each learn some slice.
        // Now we sort of "mix" them all back together into the "original" embedding. 
        // This is the TRUE POWER of multi-headed attention - this mixing happens HERE. 

        block.attn.c_proj = Linear(
            weight_utils::load_2d_tensor(base_path + ".attn.c_proj.weight.npy"),
            weight_utils::load_1d_tensor(base_path + ".attn.c_proj.bias.npy")
        );

        // ===TRANSPOSED===
        // "attn.c_proj.weight"

        /*
            This is the only one I'm not sure about, but it immidiately broke and started printing "42067" over and over
            when I uncommented it out. So chances are this ain't it chief.
        */
        // block.attn.c_proj.weight.transposeInPlace();

        assert_linear(block.attn.c_proj, config().n_embd, config().n_embd, "post-attention linear soup #" + std::to_string(layer_idx));

        // ===========================Load MLP weights===================================

        // These two project up to 4 * n_embd

        block.mlp.to_up = Linear(
            weight_utils::load_2d_tensor(base_path + ".mlp.c_fc.weight.npy"),
            weight_utils::load_1d_tensor(base_path + ".mlp.c_fc.bias.npy")
        );

        // ===TRANSPOSED===
        // "mlp.c_fc.weight"

        // block.mlp.to_up.weight.transposeInPlace();

        assert_linear(block.mlp.to_up, config().n_embd, config().n_embd * 4, "linear block up #" + std::to_string(layer_idx));

        // Project back down to n_embd

        block.mlp.back_down = Linear(
            weight_utils::load_2d_tensor(base_path + ".mlp.c_proj.weight.npy"),
            weight_utils::load_1d_tensor(base_path + ".mlp.c_proj.bias.npy")
        );

        // ===TRANSPOSED===
        // "mlp.c_proj.weight"

        // block.mlp.back_down.weight.transposeInPlace();

        assert_linear(block.mlp.back_down, config().n_embd * 4, config().n_embd, "linear block down #" + std::to_string(layer_idx));


        // Load layer norm weights

        block.ln_1 = LayerNormWeights{
            .gamma = weight_utils::load_1d_tensor(base_path + ".ln_1.weight.npy"),
            .beta = weight_utils::load_1d_tensor(base_path + ".ln_1.bias.npy")
        };
        
        block.ln_2 = LayerNormWeights{
            .gamma = weight_utils::load_1d_tensor(base_path + ".ln_2.weight.npy"),
            .beta = weight_utils::load_1d_tensor(base_path + ".ln_2.bias.npy")
        };

        assert_layer_norm(block.ln_1, config().n_embd, "ln_1 #" + std::to_string(layer_idx));
        assert_layer_norm(block.ln_2, config().n_embd, "ln_2 #" + std::to_string(layer_idx));
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load transformer block " + std::to_string(layer_idx) + ": " + std::string(e.what()));
    }
}

void ModelWeights::load_final_layer_norm(const fs::path& dir_path) {
    try {
        _ln_f = LayerNormWeights{
            .gamma = weight_utils::load_1d_tensor(dir_path / "transformer.ln_f.weight.npy"),
            .beta = weight_utils::load_1d_tensor(dir_path / "transformer.ln_f.bias.npy")
        };

        assert_layer_norm(_ln_f, config().n_embd, "ln_f");
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load final layer norm: " + std::string(e.what()));
    }
}
