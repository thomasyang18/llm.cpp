#include "forward_naive.hpp"
#include <cmath>
#include <cassert>
#include <numeric>

#include <iostream>

template<typename T> 
bool assert_not_nan(T container) {
    for (int i = 0; i < container.rows(); ++i) {
        for (int j = 0; j < container.cols(); ++j) {
            if (container(i, j) != container(i, j)) return false;
        }
    }
    return true;
};


namespace sampling {

    Eigen::RowVectorXf softmax(const Eigen::RowVectorXf& logits) {
        Eigen::RowVectorXf exp_logits = (logits.array() - logits.maxCoeff()).exp();  // for numerical stability
        return exp_logits / exp_logits.sum();
    }

    int top_k_sample(const Eigen::RowVectorXf& logits, int k) {
        // Step 1: Apply softmax
        Eigen::RowVectorXf probs = softmax(logits);
        
        // Step 2: Create a vector of indices sorted by probability (descending)
        std::vector<int> sorted_indices(probs.cols());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);  // Create a list of indices [0, 1, 2, ..., vocab_size-1]

        // Sort indices based on their corresponding probability values in descending order
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
            return probs[a] > probs[b];
        });

        // Step 3: Top-k greedy sampling
        // Take the top-k highest probabilities
        std::vector<int> top_k_indices(sorted_indices.begin(), sorted_indices.begin() + k);

        // Optionally, you can sample randomly from the top-k tokens (here, we're just choosing the token with max prob)
        int sampled_token = top_k_indices[0];  // Greedily pick the token with the highest probability in the top-k

        return sampled_token;
    }
}

ForwardNaive::ForwardNaive(const ModelWeights& model) : _model(model) {}

// Forwards a single stream of tokens, returns a single token.
int ForwardNaive::forward(std::vector<int> tokens) {
    assert(0 < tokens.size() && tokens.size() <= model().config().block_size && 
        "Passing more tokens than max sequence length.");

    Eigen::MatrixXf x(model().config().block_size, model().config().n_embd); 
    // Matrices are not default initialized, unfortunately... Have to intiialize allto zeroes to avoid NAN.
    // I think masking should save this though? idk.
    x.setZero();

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
        // Before start of transformer block...
        assert(assert_not_nan(x) && "Before transfomer block");

        x = x + causal_self_attention(layer_norm(x, block.ln_1), block.attn, tokens.size());

        assert(assert_not_nan(x) && "Right after attention block...");

        x = x + mlp(layer_norm(x, block.ln_2), block.mlp);


        assert(assert_not_nan(x) && "After transfomer block");
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
    Eigen::MatrixXf qkv = x * attn.c_attn_weight.transpose();
    for (int i = 0; i < T; i++) qkv.row(i) += attn.c_attn_bias; // broadcast bias vector to every single projected token 
    
    // 2. Split qkv into q, k, v. Each will have shape [T, C].
    Eigen::MatrixXf q = qkv.leftCols(C);
    Eigen::MatrixXf k = qkv.middleCols(C, C);
    Eigen::MatrixXf v = qkv.rightCols(C);
    
    // 3. Prepare an output accumulator for all heads.
    Eigen::MatrixXf y(T, C);
    y.setZero();

    // Determine the sequence length to use for masking
    size_t max_seq_length = seq_length.value_or(T); // Use seq_length if specified, otherwise use T.

    assert(assert_not_nan(q) && " q matrix");
    assert(assert_not_nan(k) && " k matrix");
    assert(assert_not_nan(v) && " v matrix");
    

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
                // I guess we already imply with the (j > i) mask here, so we don't NEED to care if our tokens 
                // attend to some useless ones? idk.
                // if (j > i || std::max(i, j) >= max_seq_length) 
                if (j > i)
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

    assert(assert_not_nan(y) && " before attention projection ");

    // 10. Apply the final output projection.
    //     attn.c_proj_weight should have shape [C, C] and attn.c_proj_bias shape [C].

    // I'm not sure if I should transpose here, actually, the attention matrices are all weird and funky bro. 
    y = y * attn.c_proj_weight.transpose();
    
    // broadcast
    for (int i = 0; i < T; ++i) y.row(i) += attn.c_proj_bias;

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

    assert(assert_not_nan(x) && "Before MLP");

    Eigen::MatrixXf x1 = x * mlp.c_fc_weight.transpose();
    // broadcast. Really need to figure out these broadcasting operators huh. 
    // to avoid these hacks...

    assert(mlp.c_fc_bias.transpose().cols() == x1.cols());

    assert(assert_not_nan(x) && "After first mult");

    for (int i = 0; i < x.rows(); ++i) x1.row(i) += mlp.c_fc_bias.transpose().row(0);  

    assert(assert_not_nan(x) && "After first bias");

    x1 = gelu(x1);

    x1 = x1 * mlp.c_proj_weight.transpose();

    assert(assert_not_nan(x) && "After second mult");

    for (int i = 0; i < x.rows(); ++i) x1.row(i) += mlp.c_proj_bias.transpose().row(0);  

    assert(assert_not_nan(x) && "After second bias");
    
    return x1;
}

// Some nice Eigen code here, but overall nothing too scary. 
Eigen::MatrixXf ForwardNaive::layer_norm(Eigen::MatrixXf x, const LayerNormWeights& ln) {
    // Explicitly write this independently, since I'm not that familiar with layernorm... matrix operations scary


    // sanity check that before layernorm, my program is sane.
    assert(assert_not_nan(x) && "Before layernorm");

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
    // sanity check that my layernorm didn't kill the program
    assert(assert_not_nan(result) && "After layernorm");
    return result;
}

// aight buddy 
Eigen::MatrixXf ForwardNaive::gelu(Eigen::MatrixXf x) {
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {

            assert(x(i ,j) == x(i, j) && "Before gelu");

            auto v = x(i, j);

            x(i, j) = 0.5 * v * (1.0 + tanhf(
                v + 0.044715 * v * v * v
            )) ;

            assert(x(i ,j) == x(i, j) && "Afte gelu");
        }
    }
    return x;
}
