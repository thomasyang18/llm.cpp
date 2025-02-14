#include "reference/kv_caching_forward.hpp"

KVCachingForwarder::KVCachingForwarder(const ModelWeights& model) : Forwarder(model) {}


int KVCachingForwarder::forward(std::vector<int> tokens) {
    throw std::runtime_error("Do not call forward in KVCachingForwarder; call forward_until");
}

Eigen::MatrixXf KVCachingForwarder::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attn) {
    // This is such a bad hackaround, I designed the API badly.
    // But basically, recall that kv_cache is [num_tokens, num_layer, kv_pair]. 
    // The last token in our vector is the current attention cache we're building to.  

    int tok_idx = int(kv_cache.size()) - 1;
    int layer = kv_cache.back().size(); 
    int total_tokens = int(kv_cache.size());

    assert(tok_idx >= 0);
    assert(layer < model().config().n_layer);
    assert(total_tokens <= model().config().block_size); 

    // x: [T, C] where T = sequence length and C = embedding dimension (n_embd)
    int C = x.cols();
    int n_head = model().config().n_head;

    assert(C % n_head == 0);
    int head_dim = C / n_head; // Each head gets C/n_head features

    // implicit cast to row vector, because at compile time we know that x is (1, _)
    Eigen::RowVectorXf qkv = forward_linear(x, attn.qkv);

    // 2. Split qkv into q, k, v. Each will have shape [T, C].
    Eigen::RowVectorXf q = qkv.segment(0, C);
    Eigen::RowVectorXf k = qkv.segment(C, C);
    Eigen::RowVectorXf v = qkv.segment(2 * C, C);

    // cast to row_vec
    kv_cache.back().push_back({.k = k, .v = v});

    // 3. Prepare an output accumulator for all heads.
    Eigen::RowVectorXf y(1, C);

    for (int h = 0; h < n_head; h++) {
        Eigen::RowVectorXf q_h = q.segment(h * head_dim, head_dim);

        Eigen::RowVectorXf att_h(total_tokens);

        /*
            https://jalammar.github.io/illustrated-transformer/
            We use our q[tok_idx], to compute the dot product w.r.t all {v[i] | i <= tok_idx}
        */


        /*
            Also, seeing the loops explicitly laid out like this (u can def make these matrices tho), yeah. 
            I see bactra's point now.
            http://bactra.org/notebooks/nn-attention-and-transformers.html

            If you have a function that tells you "how close" something is to something else, 
            you can run that through the weighted average.

            We split it up here as softmax -> mult on value function y[i]

            but you can view it as 

            compute "relevant distance metric" -> (kernel smooth with distance function and value functions)

            These loops make that pretty explicit, I think. 

            I guess the one weird thing is that y[i] is also an embedding space, and you can combine embedding spaces 
            just like you can scalars, theoretically? idk people always handwave that, seems legit, ML is magic..

            Kinda crazy orzocity.
        */

        for (int i = 0; i < total_tokens; ++i) {
            const Eigen::RowVectorXf& ki = kv_cache[i][layer].k.segment(h * head_dim, head_dim);
            att_h(i) = q_h.dot(ki);
        }

        // Scale the attention scores by sqrt(head_dim)
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        att_h *= scale;

        att_h = softmax(att_h);

        Eigen::RowVectorXf out_h(head_dim);
        out_h.setZero();

        for (int i = 0; i < total_tokens; ++i) {
            const Eigen::RowVectorXf& vi = kv_cache[i][layer].v.segment(h * head_dim, head_dim);
            out_h += vi * att_h(i);
        }
        // 9. Write out_h into the appropriate block of the output y.
        y.segment(h * head_dim, head_dim) = out_h;
    }
    y = forward_linear(y, attn.c_proj);

    // cast back up to a (1, _) matrix should be fine? hopefully Eigen's templates 
    // are powerful enough to test this
    return y;
}



std::vector<int> KVCachingForwarder::forward_until(std::vector<int> tokens, int THRESH) {
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

    // [tokens, n_layer = 12, (k, v)]
    // Note we don't need to store q! 
    kv_cache.clear();
    kv_cache.reserve(tokens.size() + THRESH);

    std::vector<int> result = tokens;
    // We're gonna do it this way - 
    // we're going to just fill the kv cache 1 by 1, so we don't have to deal with like, "n-batch kv cache" 
    // and weird edge cases. This makes it much easier to reason about.
    // IG this is what people mean by "hydrate the cache".

    for (int i = 0; i < tokens.size() + THRESH; ++i) {
        int cur_token = (i < tokens.size() ? tokens[i] : result.back());
        if (cur_token == model().config().EOT_TOKEN) break;

        kv_cache.emplace_back(); // add cache spot for next token 

        // Keeping this matrix API for ease of use (even though it technically should be RowVector ig)

        Eigen::RowVectorXf x = model().wte().row(cur_token) + model().wpe().row(i);

        // Forward through transformer blocks
        for (const auto& block : model().blocks()) {
            x = x + causal_self_attention(layer_norm(x, block.ln_1), block.attn);
            x = x + mlp(layer_norm(x, block.ln_2), block.mlp);
        }

        // Only start pushing back samples if we're generating new tokens.
        if (i >= tokens.size()) {
            x = layer_norm(x, model().ln_f());
            result.push_back(
                sampler(x * model().lm_head().transpose())
            );
        }
    }

    return result;
}