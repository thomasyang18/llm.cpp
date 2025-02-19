#include "reference/forward_backward_naive.hpp"
#include <cmath>
#include <cassert>
#include <numeric>
#include <random>

#include <iostream>

Forward_BackwardNaive::Forward_BackwardNaive(ModelWeights& model) : _model(model) {}

// Forwards a single stream of tokens, returns a single token.
int Forward_BackwardNaive::forward(std::vector<int> tokens) {assert(false);} // do not go down this route

// =============== START OF ALL BACKWARDS FUNCTIONS ===========

Eigen::MatrixXf Forward_BackwardNaive::backward_layer_norm(const Eigen::MatrixXf& gradient, const Eigen::MatrixXf& x, LayerNormWeights& ln) {
    constexpr float eps = 1e-5;

    Eigen::MatrixXf result(x.rows(), x.cols());

    // Again break it down per token so its explicit, IMO.
    for (int i = 0; i < x.rows(); ++i) {
        float n = x.cols();

        float mean = x.row(i).mean();
        float variance = (x.row(i).array() - mean).square().sum() / n;
        float denom = 1.0f / std::sqrt(variance + eps);

        float dvar = 
        float dmean = 

        // this derivative is given by GPT 
        result.row(i) = gradient.array() * denom + 
                        (x.row(i).array() - mean) * dvar * (2 / n) + 
                        dmean / n;
    }

    for (int i = 0; i < x.rows(); ++i) {
        // For a single element, it's just 1; so this just works 
        ln.beta += gradient.row(i);
        ln.gamma += gradient.row(i).array() * x.row(i).array();
    }

    return result;
}

// =============== START OF MODEL ===============

Eigen::RowVectorXf softmax(const Eigen::RowVectorXf& logits) {
    Eigen::RowVectorXf exp_logits = (logits.array() - logits.maxCoeff()).exp();  // for numerical stability
    return exp_logits / exp_logits.sum();
}


void Forward_BackwardNaive::backward(std::vector<int> tokens) {
    // We assume tokens.size() >= 2.
    // For training we use the first N-1 tokens as input and tokens[1...N-1] as targets.
    assert(tokens.size() >= 2 && "Need at least two tokens for training.");
    int L = tokens.size() - 1; // number of predictions

    // -------------------------------
    // 1. Forward Pass (Training Mode)
    // -------------------------------
    // Compute embeddings and add position embeddings.
    Eigen::MatrixXf x(L, model().config().n_embd);
    for (int i = 0; i < L; ++i) {
        x.row(i) = model().wte().row(tokens[i]) + model().wpe().row(i);
    }

    // We save all necessary intermediate activations per transformer block.
    struct BlockActivations {
        Eigen::MatrixXf x_in;    // Input to block.
        Eigen::MatrixXf ln1_out; // Output of first layer norm.
        Eigen::MatrixXf attn_out;// Output of causal self-attention.
        Eigen::MatrixXf res1;    // After first residual: x_in + attn_out.
        Eigen::MatrixXf ln2_out; // Output of second layer norm.
        Eigen::MatrixXf mlp_out; // Output of the MLP.
        Eigen::MatrixXf x_out;   // Block output: res1 + mlp_out.
    };
    std::vector<BlockActivations> acts;
    acts.reserve(model().blocks().size());

    // Loop over each transformer block.
    for (const auto& block : model().blocks()) {
        BlockActivations act;
        act.x_in = x;  // Save the input to the block.

        // ---- First branch: LN1 + Attention ----
        act.ln1_out = layer_norm(x, block.ln_1);
        act.attn_out = causal_self_attention(act.ln1_out, block.attn);
        // Residual connection: add the attention output back.
        act.res1 = x + act.attn_out;

        // ---- Second branch: LN2 + MLP ----
        act.ln2_out = layer_norm(act.res1, block.ln_2);
        act.mlp_out = mlp(act.ln2_out, block.mlp);
        // Residual connection.
        x = act.res1 + act.mlp_out;
        act.x_out = x;

        acts.push_back(act);
    }

    // Final layer norm.
    Eigen::MatrixXf x_final = layer_norm(x, model().ln_f());

    float loss = 0.0f;


    /*
        For Unsloth HuggingFace stuff, we want to explicitly forbid O(NV) memory.
        This both clarifies that:

        1 - This is possible
        2 - Explicitly breaks up the derivatives so that it makes sense.

        (I only proved the apply_wte_deriv, but I'll just take apply_x_deriv as true... "seems legit" kinda argument lmfao)
    */

    /*
        So again, to avoid materializing O(N x V) memory, we will do a 'complete skip' of the backwards layer.

        Skip size = 1.
    */

    Eigen::MatrixXf current_gradient(L, model().config().n_embd);

    {
        // we apply this to wte deriv
        Eigen::MatrixXf apply_wte_deriv(model().config().n_embd, model().config().n_layer);
        apply_wte_deriv.setZero();

        // Again, we can do any sliding window # of iterations to materialize at most O(slide * V) memory.
        // Probably slide = n_embd should be good, since the max vector size anywyas is O(e * V)? Or anything smaller. 
        // But **for clarity**, slide = 1.

        for (int i = 0; i < L; ++i) {
            // For token i, the target is tokens[i+1].
            int target = tokens[i + 1];

            Eigen::RowVectorXf loss_vec = softmax(
                x_final.row(i) *
                model().lm_head().transpose());

            loss -= std::log(loss_vec(target));
            
            // this is just how softmax works
            loss_vec(target) -= 1.0f;

            // Now, vector has to be averaged, before adding to saved_weight_app
            loss_vec /= L; 

            apply_wte_deriv += x_final.row(i).transpose() * loss_vec; // becomes [e x V vector]. All are scaled down implicitly.
            current_gradient.row(i) = x_final.row(i) * model().wte(); // (N x V) x (V x e). All are scaled down implicitly too. 
        }
        loss /= L;
    }

    current_gradient = Forward_BackwardNaive::backward_layer_norm(current_gradient, x_final, _model._ln_f);



    // -------------------------------
    // 4. Backpropagation through Transformer Blocks (in reverse)
    // -------------------------------
    // Process blocks in reverse order.
    for (int bi = static_cast<int>(model().blocks().size()) - 1; bi >= 0; --bi) {
        const auto& block = model().blocks()[bi];
        const auto& act = acts[bi];

        // --- Block structure reminder ---
        //   x_in      : input to block.
        //   ln1_out   : = layer_norm(x_in, ln_1)
        //   attn_out  : = causal_self_attention(ln1_out, attn)
        //   res1      : = x_in + attn_out
        //   ln2_out   : = layer_norm(res1, ln_2)
        //   mlp_out   : = mlp(ln2_out, mlp)
        //   x_out     : = res1 + mlp_out
        //
        // In the forward pass, the residual additions mean that the gradient flowing into x_out
        // splits equally into the two “branches.”

        // Start with the gradient d_x coming in for x_out.
        // Because x_out = res1 + mlp_out, we have:
        //   d_res1 (from residual addition) = d_x,
        //   d_mlp_out = d_x.
        Eigen::MatrixXf d_res1 = d_x;
        Eigen::MatrixXf d_mlp_out = d_x;

        // ---- Backprop through the MLP branch ----
        // mlp_out = mlp(ln2_out, block.mlp)
        Eigen::MatrixXf d_ln2 = backward_mlp(d_mlp_out, act.ln2_out, block.mlp, act.mlp_out);

        // Backprop through LN2.
        Eigen::MatrixXf d_res1_from_ln2 = backward_layer_norm(d_ln2, act.res1, block.ln_2, act.ln2_out);
        // Combine gradients arriving at res1.
        d_res1 += d_res1_from_ln2;

        // ---- Backprop through the residual that formed res1 ----
        // Since res1 = x_in + attn_out, the gradient splits:
        //   d_x_in (from the addition) = d_res1,
        //   d_attn (from the attention branch) = d_res1.
        Eigen::MatrixXf d_x_in = d_res1;
        Eigen::MatrixXf d_attn = d_res1;

        // ---- Backprop through the causal self-attention branch ----
        // attn_out = causal_self_attention(ln1_out, block.attn)
        Eigen::MatrixXf d_ln1 = backward_causal_self_attention(d_attn, act.ln1_out, block.attn, act.attn_out /*, plus any extra buffers */);

        // ---- Backprop through LN1 ----
        // ln1_out = layer_norm(x_in, block.ln_1)
        Eigen::MatrixXf d_x_in_from_ln1 = backward_layer_norm(d_ln1, act.x_in, block.ln_1, act.ln1_out);

        // Total gradient for block input.
        d_x = d_x_in + d_x_in_from_ln1;
    }

    // -------------------------------
    // 5. Backpropagation to Embeddings
    // -------------------------------
    // Now d_x has shape (L, n_embd) – one gradient per input token.
    // Here you would accumulate gradients for the token embedding (wte) and positional embedding (wpe) matrices.
    // For example:
    // for (int i = 0; i < L; ++i) {
    //     accumulate_grad(model().wte(), tokens[i], d_x.row(i));
    //     accumulate_grad(model().wpe(), i, d_x.row(i));
    // }
    // (The accumulation mechanism is not shown here.)
}


// START OF ALL FORWARD FUNCTIONS

Eigen::MatrixXf forward_linear(Eigen::MatrixXf x, const Linear& linear) {
    x = x * linear.weight;
    x.rowwise() += linear.bias;
    return x;
}

Eigen::MatrixXf Forward_BackwardNaive::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attn) {
    int T = x.rows();
    int C = x.cols();

    int n_head = model().config().n_head;

    assert(C % n_head == 0);
    int head_dim = C / n_head; // Each head gets C/n_head features

    Eigen::MatrixXf qkv = forward_linear(x, attn.qkv);

    Eigen::MatrixXf q = qkv.leftCols(C);
    Eigen::MatrixXf k = qkv.middleCols(C, C);
    Eigen::MatrixXf v = qkv.rightCols(C);

    Eigen::MatrixXf y(T, C);

    for (int h = 0; h < n_head; h++) {
        Eigen::MatrixXf q_h = q.middleCols(h * head_dim, head_dim);
        Eigen::MatrixXf k_h = k.middleCols(h * head_dim, head_dim);
        Eigen::MatrixXf v_h = v.middleCols(h * head_dim, head_dim);

        Eigen::MatrixXf att_h = q_h * k_h.transpose();

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        att_h *= scale;

        for (int i = 0; i < T; i++) {
            for (int j = i + 1; j < T; j++) {
                att_h(i, j) = -std::numeric_limits<float>::infinity();
            }
        }

        for (int i = 0; i < T; i++) att_h.row(i) = softmax(att_h.row(i));
        Eigen::MatrixXf out_h = att_h * v_h;
        y.middleCols(h * head_dim, head_dim) = out_h;
    }

    y = forward_linear(y, attn.c_proj);
    return y;
}

Eigen::MatrixXf Forward_BackwardNaive::mlp(Eigen::MatrixXf x, const MLPWeights& mlp) {
    x = forward_linear(x, mlp.to_up);
    x = gelu(x);
    x = forward_linear(x, mlp.back_down);
    return x;
}

Eigen::MatrixXf Forward_BackwardNaive::layer_norm(Eigen::MatrixXf x, const LayerNormWeights& ln) {
    constexpr float eps = 1e-5;

    for (int i = 0; i < x.rows(); ++i) { // iterate over all tokens
        float mean = x.row(i).mean();
        float variance = (x.row(i).array() - mean).square().sum() / x.cols();
        float denom = 1.0f / std::sqrt(variance + eps);
        x.row(i) = ((x.row(i).array() - mean) * denom) * ln.gamma.array() + ln.beta.array();
    }
    return x;
}

float _gelu(float x) {
    const float GELU_SCALE = std::sqrt(2.0f / static_cast<float>(M_PI));
    float cube = 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + tanhf(GELU_SCALE * (x + cube)));
}

Eigen::MatrixXf Forward_BackwardNaive::gelu(Eigen::MatrixXf x) { return x.unaryExpr(&_gelu); }
