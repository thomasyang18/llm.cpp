// main.cpp
#include "model_weights.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

void print_tensor_info(const std::string& name, const Eigen::MatrixXf& tensor) {
    std::cout << name << ": " << tensor.rows() << " x " << tensor.cols() << "\n";
    // Print first few values as sanity check
    std::cout << "First few values: " << tensor.topLeftCorner(1, std::min(5, (int)tensor.cols())) << "\n\n";
}

void print_tensor_info(const std::string& name, const Eigen::VectorXf& tensor) {
    std::cout << name << ": size = " << tensor.size() << "\n";
    // Print first few values as sanity check
    std::cout << "First few values: " << tensor.head(std::min(5, (int)tensor.size())).transpose() << "\n\n";
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_weights_dir>\n";
        return 1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Initialize config with default values
        GPTConfig config;
        std::cout << "Initializing GPT-2 with config:\n"
                  << "vocab_size: " << config.vocab_size << "\n"
                  << "n_layer: " << config.n_layer << "\n"
                  << "n_head: " << config.n_head << "\n"
                  << "n_embd: " << config.n_embd << "\n"
                  << "block_size: " << config.block_size << "\n\n";

        // Create model weights instance
        ModelWeights weights(config);
        
        // Load weights
        std::cout << "Loading weights from: " << argv[1] << std::endl;
        if (!weights.load_weights(argv[1])) {
            std::cerr << "Failed to load weights\n";
            return 1;
        }

        // Print timing information
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Loading completed in " << duration.count() << "ms\n\n";

        // Print some tensor information as verification
        std::cout << "=== Weight Tensor Information ===\n\n";
        
        // Token embeddings
        print_tensor_info("Token embedding (wte)", weights.get_token_embedding());
        print_tensor_info("Position embedding (wpe)", weights.get_position_embedding());

        // Sample from first transformer block
        const auto& first_block = weights.get_transformer_blocks()[0];
        std::cout << "First Transformer Block:\n";
        print_tensor_info("  Attention QKV weights", first_block.attn.c_attn_weight);
        print_tensor_info("  Attention QKV bias", first_block.attn.c_attn_bias);
        print_tensor_info("  MLP fc weights", first_block.mlp.c_fc_weight);
        
        // Final layer norm
        const auto& final_ln = weights.get_final_layer_norm();
        print_tensor_info("Final LayerNorm weight", final_ln.weight);
        print_tensor_info("Final LayerNorm bias", final_ln.bias);

        std::cout << "All weights loaded successfully!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
