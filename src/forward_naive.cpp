#include "forward_naive.hpp"
#include <cmath>
#include <cassert>

#include <iostream>

ForwardNaive::ForwardNaive(const ModelWeights& model) : _model(model) {}

// Forwards a single stream of tokens, returns a single token.
int ForwardNaive::forward(std::vector<int> tokens) {
    assert(0 < tokens.size() && tokens.size() <= model().config().block_size && 
        "Passing more tokens than max sequence length.");

    Eigen::MatrixXf x(model().config().block_size, model().config().n_embd); 
    
    for (int i = 0; i < tokens.size(); ++i) {
        // We simply index into the tokens[i] vector to retrieve the embedding
        // (ig more formally, we have a 1 hot vector and do matrix multiply here?)
        // But this is the easiest way to think about it 
        x.row(i) = 
            model().wte().row(tokens[i]) + 
            model().wpe().row(i); 
    }

    // At this step, **X** is a vector representing [# tokens, n_embed]

    // Padded tokens don't matter; TODO specify the proper attention mask

    // Forward through transformer blocks
    for (const auto& block : model().blocks()) {
        x = x + causal_self_attention(layer_norm(x, block.ln_1), block.attn, tokens.size());
        x = x + mlp(layer_norm(x, block.ln_2), block.mlp);
    }

    // Final layer norm
    x = layer_norm(x, model().ln_f());

    // One detail I did not realize:
    // As the model is trained, each position $j$ predicts the position $j + 1$. 
    // So we only want the logits vector at position tokens.size() - 1, 
    // e.g. the last token in out sequence.
    Eigen::RowVectorXf logits = x * model().lm_head().transpose().row(tokens.size() - 1);  



    // Full greedy for now; consider other schemes later. 
    float prob = 0; int argmax = 0;
    assert(logits.rows() == 1); assert(logits.cols() == model().config().vocab_size);
    for (int token = 0; token < model().config().vocab_size; ++token) 
    if (logits[token] > prob) {
        prob = logits[token]; argmax = token; 
    }
    return argmax;
}

/*
I always found it weird that you batched attention. 

Furthermore, the embedding vectors themselves scale scale as a factor of like, n_heads 
with this implementation. 

I would've thought that passing the same embedding vector through the thing, with same number of dimensions
(64), would be far more conductive to it?

Oh wait, nvm. I see. 

> does wte and the final reverse projection, do they like, both learn as well? 
> or are they given from some basic  token -> projection model e.g. are they constant. 
> cuz i find it weird that they are [vocab_size x (n_embed * n_heads)] implicitly, 
> but if they learn alongside the model then it makes more sense. 
> if so, interesting how they weight tie 
*/

// I'm on ubuntu 20.04, so I do not have access to Eigen::reshaped or Eigen::transpose(axis1, axis2).
// I'm upgrading to a new computer soon anyways... sad. Not really gonna try bother manually
// updating to Eigen 3.4 

/*
    All 3d>= matrix multiplication is defined like so .

    Matrix(a, b, c, ... X, Y).
    Matrix(a, b, c, ... Y, Z). 

    The first n-2 axes must match up - those are treated as actual "indexing" dimensions. 

    E.g. you have a * b * c ... independent matrices. 

    Then you just do standard MM on (x, y) (y, z).
*/

Eigen::MatrixXf ForwardNaive::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attn, std::optional<size_t> seq_length ) {
    // x: [T, C] where T = sequence length and C = embedding dimension (n_embd)
    int T = x.rows();
    int C = x.cols();
    assert(T == model().config().block_size);
    int n_head = model().config().n_head;
    int head_dim = C / n_head; // Each head gets C/n_head features

    // 1. Compute qkv: project x with weight and add bias.
    //    attn.c_attn_weight should have shape [C, 3*C] and attn.c_attn_bias shape [3*C].
    //    The result qkv is of shape [T, 3*C].
    Eigen::MatrixXf qkv = x * attn.c_attn_weight + attn.c_attn_bias.transpose();
    
    // 2. Split qkv into q, k, v. Each will have shape [T, C].
    Eigen::MatrixXf q = qkv.leftCols(C);
    Eigen::MatrixXf k = qkv.middleCols(C, C);
    Eigen::MatrixXf v = qkv.rightCols(C);
    
    // 3. Prepare an output accumulator for all heads.
    Eigen::MatrixXf y(T, C);
    y.setZero();

    // Determine the sequence length to use for masking
    size_t max_seq_length = seq_length.value_or(T); // Use seq_length if specified, otherwise use T.

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

        // 6. Apply a causal mask: for each row i, zero out (or set to -infinity) any column j > i.
        //    Here we loop over rows and columns.
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                if (j > i || std::max(i, j) >= max_seq_length) 
                    att_h(i, j) = -std::numeric_limits<float>::infinity();
            }
        }

        // 7. Apply softmax to each row of att_h.
        //    For numerical stability, subtract the max in each row before exponentiating.
        for (int i = 0; i < T; i++) {
            float row_max = att_h.row(i).maxCoeff();
            // Compute exponentials
            Eigen::VectorXf exp_row = (att_h.row(i).array() - row_max).exp();
            float sum_exp = exp_row.sum();
            // Replace the row with the normalized softmax values
            att_h.row(i) = exp_row.transpose() / sum_exp;
        }

        // 8. Compute the weighted sum of the values: out_h = att_h * v_h.
        //    out_h has shape [T, head_dim].
        Eigen::MatrixXf out_h = att_h * v_h;

        // 9. Write out_h into the appropriate block of the output y.
        y.block(0, h * head_dim, T, head_dim) = out_h;
    }

    // 10. Apply the final output projection.
    //     attn.c_proj_weight should have shape [C, C] and attn.c_proj_bias shape [C].
    y = y * attn.c_proj_weight + attn.c_proj_bias.transpose();

    return y;
}

/*
Apparently 4*x MLP internal layer is just a normal thing. I thought it was special; 
no, you just want to map to some higher dimension, and 4*x was good I guess. 

Importantly, each token is *independent*. What the MLP actually does is a contract that says
f(attended token) -> even better token.
1 token input, 1 token output (recall that a token can be represented by a 1d vector of size [n_embd])
This way, you alternate between all-pairs learning and a simple MLP that tries to understand words
sort of just normally. 

THEREFORE, what this means is: 

Multiplying a matrix of [sequence_length, n_embd] x [n_embd, internal_proj] x [internal_proj, n_embd]
is a well defined operation that operates **pairwise independently** on all tokens. 

To anyone with only a basic understanding of ML (me), this computation might seem weird at first, 
since we usually think of MLPs operating on a single layer of neurons. 

We are here, we just add an extra row dimension, and everything works out :) 
*/


Eigen::MatrixXf ForwardNaive::mlp(Eigen::MatrixXf x, const MLPWeights& mlp) {
    // x is our sequence of context-augmented tokens. 
    Eigen::MatrixXf x1 = x * mlp.c_fc_weight + 
        // Note that this is what we're doing when we replicate the bias too!
        mlp.c_fc_bias.transpose().replicate(x.rows(), 1);

    x1 = gelu(x1);
    x1 = x1 * mlp.c_proj_weight + mlp.c_proj_bias.transpose().replicate(x.rows(), 1);
    return x1;
}

// Some nice Eigen code here, but overall nothing too scary. 
Eigen::MatrixXf ForwardNaive::layer_norm(Eigen::MatrixXf x, const LayerNormWeights& ln) {
    // Explicitly write this independently, since I'm not that familiar with layernorm... matrix operations scary
    constexpr float eps = 1e-5;

    auto result = x;

    for (int i = 0; i < x.rows(); ++i) { // iterate over all tokens
        float mean = x.row(i).mean();
        float variance = (x.row(i).array() - mean).square().sum() / x.cols(); 
        auto denom = std::sqrt(variance + eps); 

        // Okay bro I'm just gonna manually implement this for now, Eigen operations just don't make sense

        for (int j = 0; j < x.cols(); ++j) {
            // The weights and biases for layernorm are stored as [n_ebd, 1] vector; this is just a dot product basically.
            result(i, j) = (x(i, j) - mean) / denom * ln.weight(j, 0) + ln.bias(j, 0);
        }
    }
    return result;
}

// aight buddy 
Eigen::MatrixXf ForwardNaive::gelu(Eigen::MatrixXf x) {
    return 0.5 * x.array() * (1.0 + (x.array() + 0.044715 * x.array().pow(3)).tanh());
}
