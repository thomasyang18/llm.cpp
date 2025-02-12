#include "model_weights.hpp"
#include "forward_naive.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <filesystem>

const int MAX_INFERENCE_TOKENS = 100;

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
        weights.load_weights(argv[1]);

        // Print timing information
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Loading completed in " << duration.count() << "ms\n\n";

        std::cout << "All weights loaded successfully!\n";

        // Create ForwardNaive instance
        ForwardNaive forward_naive(weights);

        // Read input tokens

        // int N;
        // std::cout << "Enter the number of tokens: ";
        // std::cin >> N;

        // std::vector<int> tokens(N);
        // std::cout << "Enter the tokens: ";
        // for (int i = 0; i < N; ++i) {
        //     std::cin >> tokens[i];
        // }

        /*
            Hardcoded in input tokens for debugging purposes
        */

        int N = 8;
        std::vector<int> tokens = {
            15496, 11, 314, 1101, 257, 3303, 2746, 13
        };

        // Token generation loop
        for (int i = 0; i < MAX_INFERENCE_TOKENS; ++i) {
            if (tokens.size() > weights.config().block_size) tokens.erase(tokens.begin());
            int next_token = forward_naive.forward(tokens);
            std::cerr << " ADDING TOKEN " << next_token << std::endl;
            tokens.push_back(next_token);
            if (next_token == 50256) { // EOT token
                break;
            }
        }

        // Print resulting token array
        std::cout << "Resulting token array: [";
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
