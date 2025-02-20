#include "utils/model_weights.hpp"
#include "utils/weight_utils.hpp"
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

ModelWeights::ModelWeights(const GPTConfig& config) : _config(config) {
    _h.resize(config.n_layer);
}

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


void assert_linear(const Linear& input, int in_dim, int out_dim, std::string name) {
    weight_utils::assert_tensor_shape(input.weight, in_dim, out_dim, name);
    weight_utils::assert_vector_shape(input.bias, out_dim, name);
}

void assert_layer_norm(const LayerNormWeights& input, int size, std::string name) {
    weight_utils::assert_vector_shape(input.gamma, size, name);
    weight_utils::assert_vector_shape(input.beta, size, name);
}

void ModelWeights::verifySizes() {

    weight_utils::assert_tensor_shape(_wte, _config.vocab_size, _config.n_embd, "wte");
    weight_utils::assert_tensor_shape(_wpe, _config.block_size, _config.n_embd, "wpe");

    for (int layer_idx = 0; layer_idx < config().n_layer; ++layer_idx) {
        auto &block = _h[layer_idx];

        assert_linear(block.attn.qkv, config().n_embd, config().n_embd * 3, "attention block #" + std::to_string(layer_idx));
        assert_linear(block.attn.c_proj, config().n_embd, config().n_embd, "post-attention linear soup #" + std::to_string(layer_idx));
        assert_linear(block.mlp.to_up, config().n_embd, config().n_embd * 4, "linear block up #" + std::to_string(layer_idx));
        assert_linear(block.mlp.back_down, config().n_embd * 4, config().n_embd, "linear block down #" + std::to_string(layer_idx));

        assert_layer_norm(block.ln_1, config().n_embd, "ln_1 #" + std::to_string(layer_idx));
        assert_layer_norm(block.ln_2, config().n_embd, "ln_2 #" + std::to_string(layer_idx));
    }
        assert_layer_norm(_ln_f, config().n_embd, "ln_f");

}

#define MEAN 0
#define STDDEV 0.02

Eigen::MatrixXf random_2d(int rows, int cols, float mean = 0, float stddev = STDDEV) {
    Eigen::MatrixXf res(rows, cols);

    std::mt19937 rng;
    std::normal_distribution<> nd(mean, stddev);

    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        res(i, j) = nd(rng);
    }

    return res;
}

Eigen::RowVectorXf random_1d(int size, float mean = 0, float stddev = STDDEV) {
    Eigen::RowVectorXf res(size);

    std::mt19937 rng;
    std::normal_distribution<> nd(mean, stddev);

    for (int i = 0; i < size; ++i) {
        res(i) = nd(rng);
    }

    return res;
}

Linear random_linear(int rows, int cols, float mean = 0, float stddev = STDDEV) {
    return Linear{
        .weight = random_2d(rows, cols, mean, stddev),
        .bias = random_1d(cols, mean, stddev)
    };
}

LayerNormWeights random_ln(int size, float mean = 0, float stddev = STDDEV) {
    return LayerNormWeights{
        .gamma = random_1d(size, mean, stddev),
        .beta = random_1d(size, mean, stddev)
    };
}

void ModelWeights::init_data_random() {
    _wte = random_2d(config().vocab_size, config().n_embd);
    _wpe = random_2d(config().block_size, config().n_embd);

    
    /*
        https://youtu.be/l8pRSuU81PU?si=b7uo4niS4_77t_R-&t=4704

        Cancel out variance via 1/sqrt(layers * 2 ) for residuals

        for residuals

        Interesting reason why we still use float16s, and not int8s for training (well aside from precision issues it intuitively maeks sense w/ derivatives but)

        https://youtu.be/l8pRSuU81PU?si=rCD_cgdLn2MkEBvU&t=5177 every neuron is a normal distribution so it sort of 'matches that' thats insane
    */
    const float cancel = 1.0 / std::sqrt(config().n_layer * 2);

    for (int layer_idx = 0; layer_idx < _config.n_layer; ++layer_idx) {
        auto& block = _h[layer_idx];

        block.attn.qkv = random_linear(config().n_embd, config().n_embd * 3);
        block.attn.c_proj = random_linear(config().n_embd, config().n_embd, MEAN, STDDEV * cancel);

        block.mlp.to_up = random_linear(config().n_embd, config().n_embd * 4);
        block.mlp.back_down = random_linear(config().n_embd * 4, config().n_embd, MEAN, STDDEV * cancel);

        block.ln_1 = random_ln(config().n_embd);
        block.ln_2 = random_ln(config().n_embd);
    }

    _ln_f = random_ln(config().n_embd);
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
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load embeddings: " + std::string(e.what()));
    }
}


void ModelWeights::load_transformer_block(int layer_idx, const fs::path& dir_path) {
    try {
        auto base_path = dir_path / "transformer.h.";
        base_path += std::to_string(layer_idx);
        auto& block = _h[layer_idx];

        // Load attention weights

        block.attn.qkv = Linear{
            .weight = weight_utils::load_2d_tensor(base_path + ".attn.c_attn.weight.npy"),
            .bias = weight_utils::load_1d_tensor(base_path + ".attn.c_attn.bias.npy")
        };

        // ===TRANSPOSED====
        // "attn.c_attn.weight"

        // block.attn.qkv.weight.transposeInPlace();



        // The previous embedding space, logically speaking, is [n_heads, smol_embd = n_embd / n_heads] dimensions each. They each learn some slice.
        // Now we sort of "mix" them all back together into the "original" embedding. 
        // This is the TRUE POWER of multi-headed attention - this mixing happens HERE. 

        block.attn.c_proj = Linear{
            .weight = weight_utils::load_2d_tensor(base_path + ".attn.c_proj.weight.npy"),
            .bias = weight_utils::load_1d_tensor(base_path + ".attn.c_proj.bias.npy")
        };

        // ===TRANSPOSED===
        // "attn.c_proj.weight"

        /*
            This is the only one I'm not sure about, but it immidiately broke and started printing "42067" over and over
            when I uncommented it out. So chances are this ain't it chief.
        */
        // block.attn.c_proj.weight.transposeInPlace();


        // ===========================Load MLP weights===================================

        // These two project up to 4 * n_embd

        block.mlp.to_up = Linear{
            .weight = weight_utils::load_2d_tensor(base_path + ".mlp.c_fc.weight.npy"),
            .bias = weight_utils::load_1d_tensor(base_path + ".mlp.c_fc.bias.npy")
        };

        // ===TRANSPOSED===
        // "mlp.c_fc.weight"

        // block.mlp.to_up.weight.transposeInPlace();


        // Project back down to n_embd

        block.mlp.back_down = Linear{
            .weight = weight_utils::load_2d_tensor(base_path + ".mlp.c_proj.weight.npy"),
            .bias = weight_utils::load_1d_tensor(base_path + ".mlp.c_proj.bias.npy")
        };

        // ===TRANSPOSED===
        // "mlp.c_proj.weight"

        // block.mlp.back_down.weight.transposeInPlace();



        // Load layer norm weights

        block.ln_1 = LayerNormWeights{
            .gamma = weight_utils::load_1d_tensor(base_path + ".ln_1.weight.npy"),
            .beta = weight_utils::load_1d_tensor(base_path + ".ln_1.bias.npy")
        };
        
        block.ln_2 = LayerNormWeights{
            .gamma = weight_utils::load_1d_tensor(base_path + ".ln_2.weight.npy"),
            .beta = weight_utils::load_1d_tensor(base_path + ".ln_2.bias.npy")
        };

        
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

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load final layer norm: " + std::string(e.what()));
    }
}
