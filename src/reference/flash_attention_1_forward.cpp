#include "reference/flash_attention_1_forward.hpp"

FlashAttention1Forwarder::FlashAttention1Forwarder(const ModelWeights& model) : Forwarder(model) {}

/*
    M = Arbitrary constant, not really that interested in fine-tuning/benchmarking but 

    floats (our dtype) are 4 bytes 
    core {0, 1} are on the same chip, have the same L2 cache (I think)
    so disable core 1 
    to get a reasonable estimate of 2^20 floats. 

    worried about thrashing but whatever, seems like a clean enough #. dont know enough about real
    perf engineering to reason about this 
*/
// none of these constants worked, makes a decent amount of sense?
// my program isn't doing any specialized high performance matrix math, not taking advantage of simd etc
// well, maybe eigen does. even if it does, i doubt some of my loops are optimal lmao.

const int M = 2<<20; 

namespace {
constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}
}

/* Whenever we want to debug something stupid, drop this in. Works wonders.

    int Bc = N; int Br = N; 

    Eigen::MatrixXf attn = hbm_q * hbm_k.transpose();

    assert(attn.rows() == N && attn.cols() == N);

    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    attn *= scale;

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            attn(i, j) = -std::numeric_limits<float>::infinity();
        }
    }
    for (int i = 0; i < attn.rows(); ++i) attn.row(i) = softmax(attn.row(i));

    global_o.middleCols(h * d, d) = attn * hbm_v;

    continue;

*/

Eigen::MatrixXf FlashAttention1Forwarder::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attn) {
    int N = x.rows();
    int d = model().config().n_embd / model().config().n_head;

    assert(model().config().n_embd % model().config().n_head == 0);

    // TODO: I am worried about multi-headed attention, but perhaps because we're breaking up by blocks of size D 
    // it just doesn't matter.
    
    // we could fix this with kv caching, maybe? but let's continue.
    Eigen::MatrixXf qkv = forward_linear(x, attn.qkv);

    // 2. Split qkv into q, k, v. Each will have shape [T, C].
    Eigen::MatrixXf global_q = qkv.leftCols(model().config().n_embd);
    Eigen::MatrixXf global_k = qkv.middleCols(model().config().n_embd, model().config().n_embd);
    Eigen::MatrixXf global_v = qkv.rightCols(model().config().n_embd);

    // 3. Prepare an output accumulator for all heads.
    Eigen::MatrixXf global_o(N, model().config().n_embd);
    global_o.setZero();

    // Alright, screw this, the transposition stuff makes 0 sense lowkey, idk how you can just do that. 
    // Dumb but correct: We can load all the q, k, v things into actual matrices, individually, per head. 
    // THEN do the flash_attention stuff using the correct notation. 
    // Again, slow, but correct. 

    for (int h = 0; h < model().config().n_head; h++) {
        // Really inaccurate model, q, k, v, o are "HBM". Lol, wrong for so many reasons
        // But whatever, educational. 
        Eigen::MatrixXf hbm_q = global_q.middleCols(h * d, d);
        Eigen::MatrixXf hbm_k = global_k.middleCols(h * d, d);
        Eigen::MatrixXf hbm_v = global_v.middleCols(h * d, d);

        Eigen::MatrixXf hbm_o = global_o.middleCols(h * d, d);

        // These guys are actually column vectors! Since they compute statistics, per each row; hence, column vectors.
        Eigen::VectorXf hbm_l(N); hbm_l.setZero(); // norm sum state
        Eigen::VectorXf hbm_m(N); hbm_m.setConstant(-std::numeric_limits<float>::infinity()); // max state

        int Bc = ceildiv(M, 4 * d);
        int Br = std::min(d, ceildiv(M, 4 * d));


        // ==== START OF ACTUAL FLASH ATTENTION ====
        
        for (int kv_start = 0; kv_start < N; kv_start += Bc) {
            // [HBM -> SRAM] Batch load this slice into SRAM 
            Eigen::MatrixXf k = hbm_k.middleRows(kv_start, std::min(Bc, N - kv_start));
            Eigen::MatrixXf v = hbm_v.middleRows(kv_start, std::min(Bc, N - kv_start));

            // Inner loop: iterate over blocks of Q (row blocks)
            for (int q_start = 0; q_start < N; q_start += Br) {
                // [HBM -> SRAM] Load Q, O, L, and m blocks (for block i) from HBM into SRAM
                Eigen::MatrixXf q = hbm_q.middleRows(q_start, std::min(Br, N - q_start)); 
                Eigen::MatrixXf o = hbm_o.middleRows(q_start, std::min(Br, N - q_start)); 
                Eigen::VectorXf l = hbm_l.segment(q_start, std::min(Br, N - q_start));
                Eigen::VectorXf m = hbm_m.segment(q_start, std::min(Br, N - q_start));

                // [On-chip] Compute the intermediate attention scores:
                // S = Q_block * (K_block)^T, done in fast on-chip SRAM
                Eigen::MatrixXf S = q*k.transpose();

                S *= 1.0 / sqrtf(d); // scaling attention is important...

                // Erm. We can recover the causal mask explicitly here. It's a little bit scuffed, but we 
                // know all the indices this matrix should correspond to, so we can still mask out
                // the "real" indicies (j > i).

                for (int i = 0; i < S.rows(); ++i){
                    for (int j = 0; j < S.cols(); ++j) {
                        if (j + kv_start > i + q_start) 
                        S(i, j) = 
                            -std::numeric_limits<float>::infinity();
                    }
                }
                
                // [On-chip] Compute row-wise maximum of S: tilde_m = rowMax(S)
                Eigen::VectorXf tilde_m = S.rowwise().maxCoeff();

                // [On-chip] Compute the exponentiated, shifted scores:
                // tilde_P = exp(S - tilde_m) applied row-wise
                // insane broadcasting stuff wtf
                Eigen::MatrixXf tilde_p = S; 
                // (S.array().colwise() - tilde_m.array()).exp();
                for (int i = 0 ;i < S.rows(); ++i) {
                    for (int j = 0 ; j < S.cols(); ++j) {
                        tilde_p(i, j) = std::exp(S(i, j) - tilde_m(i)); 
                    }
                }

                // [On-chip] Compute row-wise sum of tilde_P: tilde_L = rowSum(tilde_P)
                Eigen::VectorXf tilde_L = tilde_p.rowwise().sum();

                // [On-chip] Update normalization statistics:
                // m_new = max(m_block, tilde_m) elementwise
                Eigen::VectorXf m_new = m.cwiseMax(tilde_m);
                // L_new = exp(m_block - m_new) * L_block + exp(tilde_m - m_new) * tilde_L, elementwise
                Eigen::VectorXf L_new = 
                        (m.array() - m_new.array()).exp() * l.array() + 
                        (tilde_m.array() - m_new.array()).exp() * tilde_L.array();

                // [On-chip] Compute the block output update:
                // temp = tilde_P * V_block (matrix multiplication in SRAM)
                Eigen::MatrixXf temp = tilde_p * v;

                // Update each row of the current O_block:
                for (int r = 0; r < q.rows(); ++r) {
                    // Compute scaling factors for row r
                    float scale_prev = std::exp(m[r] - m_new[r]);  // scale for previous output
                    float scale_new  = std::exp(tilde_m[r] - m_new[r]);     // scale for current computation

                    for (int c = 0; c < d; ++c) {
                        // Combine previous and new results, then normalize:
                        // O_block[r][c] = ( scale_prev * L_block[r] * O_block[r][c] + scale_new * temp[r][c] ) / L_new[r]
                        o(r,c) = (scale_prev * l[r] * o(r,c) +
                                        scale_new  * temp(r,c)) / L_new[r];
                    }
                }

                // [SRAM -> HBM] Write updated O_block, L_new, and m_new back to HBM
                hbm_o.middleRows(q_start, std::min(Br, N - q_start)) = o;
                hbm_l.segment(q_start, std::min(Br, N - q_start)) = L_new;
                hbm_m.segment(q_start, std::min(Br, N - q_start)) = m_new;
            }
        }

        // ==== END OF ACTUAL FLASH ATTENTION ====
        
        // write back out to true global slice
        global_o.middleCols(h * d, d) = hbm_o;
    }

    global_o = forward_linear(global_o, attn.c_proj);
    return global_o;
}

int FlashAttention1Forwarder::forward(std::vector<int> tokens) {
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
