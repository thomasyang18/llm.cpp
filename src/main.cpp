#include "reference/kv_caching_forward.hpp"
#include "reference/flash_attention_1_forward.hpp"
#include "reference/forward_naive.hpp"

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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_weights_dir>\n";
        return 1;
    }

    try {
        // Initialize config with default values
        GPTConfig config;
        // Create model weights instance
        ModelWeights weights(config);
        // Load weights
        std::cout << "Loading weights from: " << argv[1] << std::endl;
        weights.load_weights(argv[1]);
        std::cout << "All weights loaded successfully!\n";


        {
            // DEBUG SCRIPT FOR AIDER


            return 0;
        }


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
