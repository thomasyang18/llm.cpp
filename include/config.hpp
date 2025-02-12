#pragma once
#include <string>

struct GPTConfig {
    int block_size = 1024;
    int vocab_size = 50257;
    int n_layer = 12;
    int n_head = 12;
    int n_embd = 768;
    std::string weights_path;
};

