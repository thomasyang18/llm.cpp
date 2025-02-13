#pragma once
#include <string>
#include <cassert>

struct GPTConfig {
    int block_size = 1024;
    int vocab_size = 50257;
    int n_layer = 12;
    int n_head = 12;
    int n_embd = 768;

    int EOT_TOKEN = 50256;

    GPTConfig() {
        assert(n_embd % n_head == 0);
        // actual number of embeddings is n_embd/n_head, but GPT batches them since
        // they're mostly independent until attention
    }
};
