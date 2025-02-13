#include "kv_caching_forward.hpp"
// #include "kernel_fusion_forward.hpp"
// #include "reference/forward_naive.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <filesystem>

const int MAX_INFERENCE_TOKENS = 100;

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

        // Okay, I need to just print out the weights here.
        // *surely* I'm not going insane, right? My algorithm seems correct now.
        // The only thing that can be wrong is the data but I serialized it striaght from colab....
        // print_tensor_info("wte", weights.wte());

        /*
            There's simply no way dooing this fails. I'm literally just gonna write back what I think is the tensor to memory. 

            cnpy::NpyArray arr = cnpy::npy_load(path.string());
        */

        // Row vectors seem to be fine. 
        // print_tensor_info("dev/model_weights/transformer.ln_f.bias.npy", weights.ln_f().beta);

        // exit(0);


        // Create ForwardNaive instance
        // KernelFusionForwarder 
        // ForwardNaive
        KVCachingForwarder
            forward_naive(weights);

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

        tokens = forward_naive.forward_until(tokens, 50);

        /*

        // Token generation loop
        for (int i = 0; i < MAX_INFERENCE_TOKENS; ++i) {
            if (tokens.size() > weights.config().block_size) tokens.erase(tokens.begin());
            int next_token = forward_naive.forward(tokens);
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

            if (next_token == 50256) { // EOT token
                break;
            }
        }
        */

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
