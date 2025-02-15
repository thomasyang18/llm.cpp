#pragma once
#include <string>
#include <cassert>

// Oh my god, I think the issue is 
// that because this dataclass isn't backed by an object file I don't recompile this? 
// And it leads to segfaults because this technically is data or something? idrk, but 
// I modified this, ran make again, and basically everything was fine for some reason, but then 
// I segfaulted and this was the culprit. 

struct GPTConfig {
    int block_size = 1024;
    int vocab_size = 50257;
    int n_layer = 12;
    int n_head = 12;
    int n_embd = 768;

    int EOT_TOKEN = 50256;

    int top_k_sample = 10; // Set this to 1 when debugging.

    GPTConfig() {
        assert(n_embd % n_head == 0);
        // actual number of embeddings is n_embd/n_head, but GPT batches them since
        // they're mostly independent until attention
    }

    inline int d() const {
        return n_embd / n_head;
    }
};
