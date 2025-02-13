#include "kernel_fusion_forward.hpp"

KernelFusionForwarder::KernelFusionForwarder(const ModelWeights& model) : Forwarder(model) {}

Eigen::MatrixXf KernelFusionForwarder::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attention) {
    // Implement causal self-attention using kernel fusion
    // This is a placeholder implementation
    return x;
}

int KernelFusionForwarder::forward(std::vector<int> tokens) {
    assert(0 < tokens.size() && tokens.size() <= model().config().block_size &&
        "Passing more tokens than max sequence length.");

    Eigen::MatrixXf x(model().config().block_size, model().config().n_embd);
    // Matrices are not default initialized, unfortunately... Have to intiialize allto zeroes to avoid NAN.
    // I think masking should save this though? idk.
    // x.setZero();

    for (int i = 0; i < tokens.size(); ++i) {
        // We simply index into the tokens[i] vector to retrieve the embedding
        // (ig more formally, we have a 1 hot vector and do matrix multiply here?)
        // But this is the easiest way to think about it
        x.row(i) =
            model().wte().row(tokens[i]) +
            model().wpe().row(i);
    }

    // Karpathy does this I ll do the same
    // Okay.... adding these for some reason drastically changed inference.
    // I feel like that obviously means something is wrong, no? but what do I know.
    for (int i = tokens.size(); i < model().config().block_size; ++i) {
        x.row(i) =
            model().wte().row(50256) +
            model().wpe().row(i);
    }

    // At this step, **X** is a vector representing [# tokens, n_embed]

    // Padded tokens don't matter; TODO specify the proper attention mask

    // Forward through transformer blocks
    for (const auto& block : model().blocks()) {
        x = x + causal_self_attention(layer_norm(x, block.ln_1), block.attn);
        x = x + mlp(layer_norm(x, block.ln_2), block.mlp);
    }

    // Final layer norm
    x = layer_norm(x, model().ln_f());

    // One detail I did not realize:
    // As the model is trained, each position $j$ predicts the position $j + 1$.
    // So we only want the logits vector at position tokens.size() - 1,
    // e.g. the last token in out sequence.

    // std::cout << model().lm_head().transpose().rows() << " " <<
    // model().lm_head().transpose().cols() << std::endl;

    // std:: cout << x.rows() << " " << x.cols() << std::endl;

    // Eigen::RowVectorXf logits(model().config().vocab_size);

    // Surely we can speed this up by being smart but whatever

    // auto res = (x * model().lm_head().transpose());

    // for (int i = 0; i < tokens.size(); ++i) {
    //     std::cout << "yo mama!" << res.row(i).mean() << std:: endl;
    // }

    Eigen::RowVectorXf logits = (x * model().lm_head().transpose()).row(tokens.size() - 1);

    int sampled_token = sampling::top_k_sample(logits, 10);

    return sampled_token;
}
