#include "reference/kv_caching_forward.hpp"
#include "reference/flash_attention_1_forward.hpp"
#include "reference/forward_naive.hpp"

#include "reference/forward_backward_naive.hpp"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <filesystem>

const int MAX_INFERENCE_TOKENS = 50;

void print_tensor_info(const std::string& name, const Eigen::MatrixXf& tensor) {
    std::cout << name << ": " << tensor.rows() << " x " << tensor.cols() << "\n";
    // Print first few values as sanity check
    std::cout << "First few values: " << std::endl;

    for (int i = 0 ; i < std::min((int)tensor.rows(), 5); ++i ) {
        for (int j = 0 ; j < std::min((int)tensor.cols(), 5); ++j ) {
            std::cout << tensor(i, j) << " ";
        }
        std::cout << std::endl;
        // tensor.topLeftCorner(1, std::min(5, (int)tensor.cols())) << "\n\n";
    }

    for (int i = 0; i < tensor.rows(); ++i) for (int j = 0; j < tensor.cols(); ++j) {
        if (abs(-0.04861503 - tensor(i, j)) < 1e-7) std::cout << " Found match at " << i << " " << j << std::endl;
    }
}

void print_tensor_info(const std::string& name, const Eigen::RowVectorXf& tensor) {
    std::cout << name << ": size = " << tensor.size() << "\n";
    // Print first few values as sanity check
    std::cout << "First few values: " << tensor.head(std::min(5, (int)tensor.size())).transpose() << "\n\n";
}

template<typename T>
std::vector<int> one_by_one(T& forwarder, std::vector<int> tokens ) {
    for (int i = 0; i < MAX_INFERENCE_TOKENS; ++i) {
        if (tokens.size() > forwarder.model().config().block_size) tokens.erase(tokens.begin());
        int next_token = forwarder.forward(tokens);
        // std::cerr << " ADDING TOKEN " << next_token << std::endl;
        tokens.push_back(next_token);

        // Print resulting token array
        std::cout << "Resulting token array: [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i];
            if (i < tokens.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";

        if (next_token == forwarder.model().config().EOT_TOKEN) { // EOT token
            break;
        }
    }
    return tokens;
}

int main(int argc, char** argv) {
    int mode; 
    std::cout << "type 0 for pre-loaded weight inference, type 1 for training " << std::endl;
    std::cin >> mode;

    if (mode == 1) {
        // training loop here 
        GPTConfig config;
        ModelWeights model(config);
        model.init_data_random();

        model.verifySizes();

        std::vector<int> tokens = {43, 29625, 220, 2419, 388, 288, 45621, 1650, 716, 316, 11, 369, 8831, 316, 333, 31659, 271, 2259, 1288, 270, 11, 10081, 466, 304, 3754, 4666, 10042, 753, 312, 312, 2797, 3384, 2248, 382, };
        tokens.push_back(config.EOT_TOKEN);
        // 2123, 288, 349, 382, 2153, 2616, 435, 1557, 64, 13, 7273, 551, 320, 512, 10356, 8710, 1789, 11, 627, 271, 18216, 81, 463, 4208, 3780, 334, 297, 321, 1073, 4827, 271, 299, 23267, 3384, 435, 1557, 541, 409, 304, 64, 13088, 78, 4937, 265, 13, 10343, 271, 257, 1133, 4173, 495, 288, 45621, 287, 1128, 260, 258, 681, 270, 287, 2322, 37623, 378, 11555, 270, 1658, 325, 269, 359, 388, 288, 349, 382, 304, 84, 31497, 5375, 9242, 64, 1582, 72, 2541, 13, 18181, 23365, 264, 600, 1609, 64, 721, 265, 6508, 312, 265, 265, 1729, 386, 738, 11, 264, 2797, 287, 10845, 8957, 45567, 1163, 544, 748, 263, 2797, 285, 692, 270, 2355, 4686, 1556, 4827, 388, 13, 50256};
        
        Forward_BackwardNaive trainer(model);
        int iter = 0;
        for (float temp = 1.0; temp > 0; temp -= 0.1) {
            std::cout << "ITeration " << iter++ << " : " << temp << std::endl;
            trainer.backward(tokens, temp);
        }

        KVCachingForwarder forward(model);
        std::vector<int> tokens2; for (int i = 0; i < 5; ++i) tokens2.push_back(tokens[i]); // fill first 5 for free
        
        auto result = forward.forward_until(tokens2, tokens.size());
        std::cout << "[";
        for (auto x: result) std::cout << x << ", ";
        std::cout << "]\n";

        return 0;
    } 

    try {
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <path_to_weights_dir>\n";
            return 1;
        }
        // Initialize config with default values
        GPTConfig config;
        // Create model weights instance
        ModelWeights weights(config);
        // Load weights
        std::cout << "Loading weights from: " << argv[1] << std::endl;
        weights.load_weights(argv[1]);

        weights.verifySizes();
        std::cout << "All weights loaded successfully!\n";

// lol this will come later 
#ifdef DO_QUANTIZE_DEBUG
        {
            // DEBUG SCRIPT FOR AIDER

            // We might use this in quantization later. 

            // Aggregate min and max values for all weights
            float min_weight = std::numeric_limits<float>::max();
            float max_weight = std::numeric_limits<float>::min();
            float min_bias = std::numeric_limits<float>::max();
            float max_bias = std::numeric_limits<float>::min();

            // Token and position embeddings
            min_weight = std::min(min_weight, weights.wte().minCoeff());
            max_weight = std::max(max_weight, weights.wte().maxCoeff());
            min_weight = std::min(min_weight, weights.wpe().minCoeff());
            max_weight = std::max(max_weight, weights.wpe().maxCoeff());

            // Transformer blocks
            for (const auto& block : weights.blocks()) {
                // LayerNormWeights ln_1
                min_weight = std::min(min_weight, block.ln_1.gamma.minCoeff());
                max_weight = std::max(max_weight, block.ln_1.gamma.maxCoeff());
                min_weight = std::min(min_weight, block.ln_1.beta.minCoeff());
                max_weight = std::max(max_weight, block.ln_1.beta.maxCoeff());

                // AttentionWeights attn
                min_weight = std::min(min_weight, block.attn.qkv.weight.minCoeff());
                max_weight = std::max(max_weight, block.attn.qkv.weight.maxCoeff());
                min_bias = std::min(min_bias, block.attn.qkv.bias.minCoeff());
                max_bias = std::max(max_bias, block.attn.qkv.bias.maxCoeff());
                min_weight = std::min(min_weight, block.attn.c_proj.weight.minCoeff());
                max_weight = std::max(max_weight, block.attn.c_proj.weight.maxCoeff());
                min_bias = std::min(min_bias, block.attn.c_proj.bias.minCoeff());
                max_bias = std::max(max_bias, block.attn.c_proj.bias.maxCoeff());

                // LayerNormWeights ln_2
                min_weight = std::min(min_weight, block.ln_2.gamma.minCoeff());
                max_weight = std::max(max_weight, block.ln_2.gamma.maxCoeff());
                min_weight = std::min(min_weight, block.ln_2.beta.minCoeff());
                max_weight = std::max(max_weight, block.ln_2.beta.maxCoeff());

                // MLPWeights mlp
                min_weight = std::min(min_weight, block.mlp.to_up.weight.minCoeff());
                max_weight = std::max(max_weight, block.mlp.to_up.weight.maxCoeff());
                min_bias = std::min(min_bias, block.mlp.to_up.bias.minCoeff());
                max_bias = std::max(max_bias, block.mlp.to_up.bias.maxCoeff());
                min_weight = std::min(min_weight, block.mlp.back_down.weight.minCoeff());
                max_weight = std::max(max_weight, block.mlp.back_down.weight.maxCoeff());
                min_bias = std::min(min_bias, block.mlp.back_down.bias.minCoeff());
                max_bias = std::max(max_bias, block.mlp.back_down.bias.maxCoeff());
            }

            // Final layer norm
            min_weight = std::min(min_weight, weights.ln_f().gamma.minCoeff());
            max_weight = std::max(max_weight, weights.ln_f().gamma.maxCoeff());
            min_weight = std::min(min_weight, weights.ln_f().beta.minCoeff());
            max_weight = std::max(max_weight, weights.ln_f().beta.maxCoeff());

            // Print aggregated min and max values
            std::cout << "Aggregated min and max values for all weights:\n";
            std::cout << "Min weight: " << min_weight << ", Max weight: " << max_weight << "\n";
            std::cout << "Min bias: " << min_bias << ", Max bias: " << max_bias << "\n";

            // return 0;
        }
#endif 

        int N = 8;
        std::vector<int> tokens = {
            15496, 11, 314, 1101, 257, 3303, 2746, 13
        };
        int MODE;
        std::cin >> MODE;

        // debug add
        config.top_k_sample = 1;

        // lol this wasn't a reference
        assert(weights.config().top_k_sample == 1);

        auto start_time = std::chrono::high_resolution_clock::now();

        if (MODE == 0) {
            ForwardNaive
                forward_naive(weights);

            tokens = one_by_one(forward_naive, tokens);

        } else if (MODE == 1) {
            KVCachingForwarder
                forwarder(weights);

            tokens = forwarder.forward_until(tokens, MAX_INFERENCE_TOKENS);
        } else if (MODE == 2) {
            FlashAttention1Forwarder
                forward_naive(weights);

            tokens = one_by_one(forward_naive, tokens);
        } else assert(false);

        // Print timing information
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Inference done in in " << duration.count() << "ms\n\n";

        // Print resulting token array
        std::cout << "Resulting token array:\n [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i];
            if (i < tokens.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
