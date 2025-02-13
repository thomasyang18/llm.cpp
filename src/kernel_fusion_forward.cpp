#include "kernel_fusion_forward.hpp"

KernelFusionForwarder::KernelFusionForwarder(const ModelWeights& model) : Forwarder(model) {}

Eigen::MatrixXf KernelFusionForwarder::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attn) {
    // TODO: This is just the lazy attention impl lol 

    // x: [T, C] where T = sequence length and C = embedding dimension (n_embd)
    int T = x.rows();
    int C = x.cols();
    int n_head = model().config().n_head;

    assert(C % n_head == 0);
    int head_dim = C / n_head; // Each head gets C/n_head features

    // 1. Compute qkv: project x with weight and add bias.
    //    attn.c_attn_weight should have shape [C, 3*C] and attn.c_attn_bias shape [3*C].
    //    The result qkv is of shape [T, 3*C].

    Eigen::MatrixXf qkv = forward_linear(x, attn.qkv);

    // 2. Split qkv into q, k, v. Each will have shape [T, C].
    Eigen::MatrixXf q = qkv.leftCols(C);
    Eigen::MatrixXf k = qkv.middleCols(C, C);
    Eigen::MatrixXf v = qkv.rightCols(C);

    // 3. Prepare an output accumulator for all heads.
    Eigen::MatrixXf y(T, C);
    y.setZero();
    /*
        Okay I should probably try to understand the actual transpose magic going on when implementing attention, but for a first pass
        this isn't so bad. 

        From what I understand, the "actual" embeddings are just C / n_heads long.
        Partition into blocks of [T, actual_embd]. 

        Then operate on those indivdually. Am I correct? 
    */

    for (int h = 0; h < n_head; h++) {
        // Extract block corresponding to head h.
        // Each block has shape [T, head_dim].
        Eigen::MatrixXf q_h = q.block(0, h * head_dim, T, head_dim);
        Eigen::MatrixXf k_h = k.block(0, h * head_dim, T, head_dim);
        Eigen::MatrixXf v_h = v.block(0, h * head_dim, T, head_dim);

        // 5. Compute the attention scores for head h.
        //    att_h = q_h * k_h^T, so att_h has shape [T, T].
        Eigen::MatrixXf att_h = q_h * k_h.transpose();

        // Scale the attention scores by sqrt(head_dim)
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        att_h *= scale;

        // // 6. Apply a causal mask: for each row i, zero out (or set to -infinity) any column j > i.
        // //    Here we loop over rows and columns.
        // recall that att(i, j) is how much token i attends to token j. 
        for (int i = 0; i < T; i++) {
            for (int j = i + 1; j < T; j++) {
                att_h(i, j) = -std::numeric_limits<float>::infinity();
            }
        }

        // 7. Apply softmax to each row of att_h.
        //    For numerical stability, subtract the max in each row before exponentiating.
        for (int i = 0; i < T; i++) att_h.row(i) = softmax(att_h.row(i));

        // 8. Compute the weighted sum of the values: out_h = att_h * v_h.
        //    out_h has shape [T, head_dim].
        Eigen::MatrixXf out_h = att_h * v_h;

        // 9. Write out_h into the appropriate block of the output y.
        y.block(0, h * head_dim, T, head_dim) = out_h;
    }

    y = forward_linear(y, attn.c_proj);
    return y;
}

int KernelFusionForwarder::forward(std::vector<int> tokens) {
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
