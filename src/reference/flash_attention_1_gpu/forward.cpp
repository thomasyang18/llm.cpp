#include "reference/flash_attention_1_gpu/forward.hpp"

FlashAttention1ForwarderGPU::FlashAttention1ForwarderGPU(const ModelWeights& model) : Forwarder(model) {}

Eigen::MatrixXf FlashAttention1ForwarderGPU::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attn) {
    int N = x.rows();
    int d = model().config().n_embd / model().config().n_head;

    assert(model().config().n_embd % model().config().n_head == 0);

    // TODO: I am worried about multi-headed attention, but perhaps because we're breaking up by blocks of size D 
    // it just doesn't matter.
    
    // we could fix this with kv caching, maybe? but let's continue.
    Eigen::MatrixXf qkv = forward_linear(x, attn.qkv);

    // 2. Split qkv into q, k, v. Each will have shape [T, C].

    // For simplicitly, let's just think about these all as row major.
    // But I'm pretty sure, optimally, we like, have some be column major and some row
    // (like I think v and k matrices have to be opposites, since in one we transpose, while in the other we don't....)
    using row_major_t = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    row_major_t global_q = qkv.leftCols(model().config().n_embd);
    row_major_t global_k = qkv.middleCols(model().config().n_embd, model().config().n_embd);
    row_major_t global_v = qkv.rightCols(model().config().n_embd);

    // 3. Prepare an output accumulator for all heads.
    row_major_t global_o(N, model().config().n_embd);

    // Alright, screw this, the transposition stuff makes 0 sense lowkey, idk how you can just do that. 
    // Dumb but correct: We can load all the q, k, v things into actual matrices, individually, per head. 
    // THEN do the flash_attention stuff using the correct notation. 
    // Again, slow, but correct. 

    for (int h = 0; h < model().config().n_head; h++) {
        // Really inaccurate model, q, k, v, o are "HBM". Lol, wrong for so many reasons
        // But whatever, educational. 
        row_major_t hbm_q = global_q.middleCols(h * d, d);
        row_major_t hbm_k = global_k.middleCols(h * d, d);
        row_major_t hbm_v = global_v.middleCols(h * d, d);

        float *o = new float[N * d];
        std::fill(o, o + N * d, 0);

        // Recall that *all eigen vectors are column major by default*. 
        go_gpu(hbm_q.data(), hbm_k.data(), hbm_v.data(), o, N, d);

        delete o;
        
        // write back out to true global slice. Be careful about transposes and stuff. 
        global_o.middleCols(h * d, d) = Eigen::Map<row_major_t>(o, N, d);
    }

    global_o = forward_linear(global_o, attn.c_proj);
    return global_o;
}

int FlashAttention1ForwarderGPU::forward(std::vector<int> tokens) {
    assert(0 < tokens.size() && tokens.size() <= model().config().block_size &&
        "Passing more tokens than max sequence length.");

    // TODO: implement some re-usable data structure that scales with O(tokens) inference memory,
    // not using the entire block size.
    // For now, we just re-allocate every time... this can definitely be re-used.
    // this ties a little bit into k/v caching I guess? not sure. 

    Eigen::MatrixXf x(tokens.size(), model().config().n_embd);
    
    for (int i = 0; i < tokens.size(); ++i) {
        x.row(i) =
            model().wte().row(tokens[i]) +
            model().wpe().row(i);
    }

    // Forward through transformer blocks
    for (const auto& block : model().blocks()) {
        x = x + causal_self_attention(layer_norm(x, block.ln_1), block.attn);
        x = x + mlp(layer_norm(x, block.ln_2), block.mlp);
    }

    // Final layer norm
    x = layer_norm(x, model().ln_f());

    // Project the last token, sample it
    return sampler(x.row(tokens.size() - 1) * model().lm_head().transpose());
}
