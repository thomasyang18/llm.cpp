#include "reference/kv_cache_plus_flash_att/forward.cuh"
#include <memory>

KV_Cache_Plus_Flash_Forwarder::KV_Cache_Plus_Flash_Forwarder(const ModelWeights& model) : Forwarder(model) {}

int KV_Cache_Plus_Flash_Forwarder::forward(std::vector<int> tokens) {
    throw std::runtime_error("Do not call forward in KV_Cache_Plus_Flash_Forwarder; call forward_until");
}

Eigen::MatrixXf KV_Cache_Plus_Flash_Forwarder::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attn) {
    int C = model().config().n_embd;
    int d = model().config().d();

    Eigen::RowVectorXf qkv = forward_linear(x, attn.qkv);

    kv_cache->push_back<KV_Cache_Manager_GPU::KV_Type::K>(glob_state.layer, 
        qkv.segment(C, C).data()
    ); // K
    kv_cache->push_back<KV_Cache_Manager_GPU::KV_Type::V>(glob_state.layer, 
        qkv.segment(2 * C, C).data()
    ); // V

    // We also operate on this as pointers 
    std::unique_ptr<float[]> q_ptr(new float[C]); 
    memcpy(q_ptr.get(), qkv.data(), C);

    // Mainly because it cleans up after itself once I pass it back into Eigen
    // (Eigen only will take references/copy; never own/move AFAIK)

    std::unique_ptr<float[]> o(new float[C]);

    for (int h = 0; h < model().config().n_head; h++) {
        auto [k, v] = kv_cache->ask(glob_state.layer, h);
        go_gpu(q_ptr.get() + d * h, 
                k, v, 
                o.get() + d * h,
                glob_state.tokens, d);
    }

    return forward_linear(
        Eigen::Map<Eigen::RowVectorXf>(o.get(), C)    
    , attn.c_proj);
}

std::vector<int> KV_Cache_Plus_Flash_Forwarder::forward_until(std::vector<int> tokens, int THRESH) {
    if (!
            (0 < std::min((int)tokens.size(), THRESH) 
            && tokens.size() + THRESH <= model().config().block_size)
        ) {
        throw std::runtime_error("Either you passed in an empty vector, or you passed in too many tokens. size=" + 
            std::to_string(tokens.size()) + " THRESH=" + std::to_string(THRESH));
    }

    // TODO: implement some re-usable data structure that scales with O(tokens) inference memory,
    // not using the entire block size.
    // For now, we just re-allocate every time... this can definitely be re-used.
    // this ties a little bit into k/v caching I guess? not sure. 

    // init to max number of tokens possible; yes PagedAttention says this is bad, but we want simple 
    kv_cache.emplace(model().config(), tokens.size() + THRESH);

    std::vector<int> result = tokens;
    // We're gonna do it this way - 
    // we're going to just fill the kv cache 1 by 1, so we don't have to deal with like, "n-batch kv cache" 
    // and weird edge cases. This makes it much easier to reason about.
    // IG this is what people mean by "hydrate the cache".

    // BAD_CLASS: manual reset
    glob_state.layer = 0;
    glob_state.tokens = 0;

    for (int i = 0; i < tokens.size() + THRESH; ++i) {
        int cur_token = (i < tokens.size() ? tokens[i] : result.back());
        if (cur_token == model().config().EOT_TOKEN) break;

        glob_state.layer = 0; // BAD_CLASS: reset 

        Eigen::RowVectorXf x = model().wte().row(cur_token) + model().wpe().row(i);

        // Forward through transformer blocks
        for (const auto& block : model().blocks()) {
            x = x + causal_self_attention(layer_norm(x, block.ln_1), block.attn);

            glob_state.layer++; // BAD_CLASS: incr 

            x = x + mlp(layer_norm(x, block.ln_2), block.mlp);
        }

        // Only start pushing back samples if we're generating new tokens.
        if (i >= tokens.size()) {
            x = layer_norm(x, model().ln_f());
            result.push_back(
                sampler(x * model().lm_head().transpose())
            );
        }

        glob_state.tokens++; // BAD_CLASS: incr
    }

    kv_cache.reset();

    return result;
}