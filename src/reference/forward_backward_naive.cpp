/*
    This shit is so doomed dude

    Like I'm very sure my math and interpretations are correct.
    
    All you need is actual_value(input), and gradient(output) to recompute all intermediaries, and 
    you can keep breaking down this logic over and over. 
    
    but minus signs plus signs everywhere and etc.

    I guess this is why people fall back on ML frameworks 

*/


#include "reference/forward_backward_naive.hpp"
#include <cmath>
#include <cassert>
#include <numeric>
#include <random>

#include <iostream>


bool is_normal(float input) {
    if (input != input) return false;

    if (input == -std::numeric_limits<float>::infinity()) return false;
    if (input == std::numeric_limits<float>::infinity()) return false;
    return true;
}

template<typename T>

// ALSO checks that we didn't hit inf or -inf or something


bool assert_not_nan(T container) {
// #ifdef DEBUG
    for (int i = 0; i < container.rows(); ++i) {
        for (int j = 0; j < container.cols(); ++j) {
            if (!is_normal(container(i, j))) return false;
        }
    }
// #endif
    return true;
};

// =============== AUX HELPER FORWARD FUNCTIONS (USEFUL FOR BACKWARDS RECOMP TOO) ===============
namespace {
Eigen::RowVectorXf softmax(const Eigen::RowVectorXf& logits, bool debug = false) {
    if (debug) {
        // This proves that EVEN BEFOER we passed to softmax it became nan. WTF???
        std::cout << "WTF " << std::endl;
        for (int i = 0; i < 10; ++i) std::cout << logits(i) << " "; 
        std::cout << std::endl;
    }

    Eigen::RowVectorXf exp_logits = (logits.array() - logits.maxCoeff()).exp();  // for numerical stability

    assert(exp_logits.sum() > 0);

    return exp_logits / exp_logits.sum();
}

Eigen::MatrixXf forward_linear(Eigen::MatrixXf x, const Linear& linear) {
    x = x * linear.weight;
    x.rowwise() += linear.bias;
    return x;
}
}

Forward_BackwardNaive::Forward_BackwardNaive(ModelWeights& model) : _model(model) {}

// =============== START OF ALL BACKWARDS FUNCTIONS ===========

/*
    The contract of backward_X functions is that it updates some Weights, and passes down the gradient, given the input to the function x.
*/

Eigen::MatrixXf Forward_BackwardNaive::backward_layer_norm(const Eigen::MatrixXf& gradient, const Eigen::MatrixXf& x, LayerNormWeights& ln) {
    constexpr float eps = 1e-5;

    Eigen::MatrixXf result(x.rows(), x.cols());

    // Again break it down per token so it's explicit.
    for (int i = 0; i < x.rows(); ++i) {
        float n = x.cols();

        float mean = x.row(i).mean();
        float variance = (x.row(i).array() - mean).square().sum() / n;
        float denom = 1.0f / std::sqrt(variance + eps);

        // Compute dvar and dmean using the backward formulas
        float dvar = (gradient.row(i).array() * (x.row(i).array() - mean) * -0.5 * std::pow(variance + eps, -1.5)).sum();
        float dmean = (gradient.row(i).array() * -denom).sum() + dvar * (-2.0 / n) * (x.row(i).array() - mean).sum();

        // Compute the gradient w.r.t. x
        result.row(i) = gradient.row(i).array() * denom +
                        (x.row(i).array() - mean) * dvar * (2.0 / n) +
                        dmean / n;

        assert(is_normal(dvar));
        assert(is_normal(dmean));
        
        assert ( assert_not_nan(result.row(i)) );
    }

    // Update layer norm parameters gamma and beta
    for (int i = 0; i < x.rows(); ++i) {
        ln.beta += gradient.row(i);
        ln.gamma.array() += gradient.row(i).array() * x.row(i).array();
    }

    return result;
}

namespace {
Eigen::MatrixXf backward_linear(const Eigen::MatrixXf& gradient, const Eigen::MatrixXf& input, Linear& linear) {
    // This is inefficeint, but again, for matrices, let's walk step by step.
    // We basically have a projection of vectors here from (a => b).
    // Again, anything dependent on N, magically goes away because invert, invert.
    // Don't ask me how, don't ask me why. (Okay, I can derive it from scratch ,I can. But it's still genuinely so crazy and unintuitive).
    // Handwavy "oh when you look at 2nd matrix its transposed colwise iter" like nah bro this shit's just insanely clean somehow

    assert(linear.weight.rows() == input.cols());

    // So now we enforce result gradient must be (N x a). 
    // Or, we just make it equal to input.cols()... dumbfuck.
    // But still, WHY is linear.weight.rows() NOT JUST input.cols(), like what????
    Eigen::MatrixXf result(input.rows(), input.cols()); 
    
    Eigen::MatrixXf weight_grad = Eigen::MatrixXf::Zero(linear.weight.rows(), linear.weight.cols());
    Eigen::RowVectorXf bias_grad = Eigen::RowVectorXf::Zero(linear.bias.size());

    for (int i = 0; i < input.rows(); ++i) {
        // This shit is still magic to me honestly, even though I've derived the matrix dependencies directly. 
        // I have 'intuition' but no I don't really have intuition lmao. 
        // TODO: This is GPT slop, understand (tho it makes more sense since I've done a similar derivation tho)
        // Okay... GPT had some transpose issue because in its trained code its format wasn't
        // the same as my weight format.
        // Sigh.... I'm gonna have to re-derive this from scratch, aren't I?

        // Correct weight gradient: ∇W = input^T * gradient, for each row:
        weight_grad += input.row(i).transpose() * gradient.row(i);
        // Correct bias gradient: ∇b = sum over gradient rows
        bias_grad += gradient.row(i);
        // Propagate gradient backwards: ∇x = gradient * weight^T
        result.row(i) = gradient.row(i) * linear.weight.transpose();
    }

    // Update weights & bias (simulating SGD step, though you'd typically scale this in real training)
    // gradient descent
    linear.weight -= weight_grad;
    linear.bias -= bias_grad;

    return result; // Pass gradient to previous layer
}

float _gelu_derivative(float x) {
    // TRUSTED CHAT GPT CODE. could just be big wrong who knows.
    const float GELU_SCALE = std::sqrt(2.0f / static_cast<float>(M_PI));
    float cube = 0.044715f * x * x * x;
    float tanh_arg = GELU_SCALE * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float sech2 = 1.0f - tanh_out * tanh_out;  // sech^2(x) = 1 - tanh^2(x)
    
    float term1 = 0.5f * (1.0f + tanh_out);
    float term2 = 0.5f * x * sech2 * GELU_SCALE * (1.0f + 3.0f * 0.044715f * x * x);
    
    float res = 
    term1 + term2;

    assert(is_normal(res));

    return res;
}   
}

Eigen::MatrixXf Forward_BackwardNaive::backward_mlp(const Eigen::MatrixXf& gradient, const Eigen::MatrixXf& input, MLPWeights& mlp) {
    // Recompute partially, because we need inputs here. 
    Eigen::MatrixXf x = forward_linear(input, mlp.to_up);
    x = gelu(x);

    // Apply backward linear transformation through the second layer.
    Eigen::MatrixXf temp_grad = backward_linear(gradient, x, mlp.back_down);

    // Apply GELU derivative element-wise.
    // TODO: Idk what cWiseProduct is doning but seems legit. 
    temp_grad = temp_grad.cwiseProduct(x.unaryExpr(&_gelu_derivative));

    // Backpropagate through first linear layer.
    return backward_linear(temp_grad, input, mlp.to_up);
}

namespace{ 
// TODO: GPT Pasted slop idk it 
Eigen::MatrixXf softmax_backward(const Eigen::MatrixXf& dA, const Eigen::MatrixXf& A) {
    Eigen::MatrixXf dS = Eigen::MatrixXf::Zero(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i) {
        // For each row, let A_i be softmax and dA_i be gradient wrt A.
        // dS_i = A_i .* (dA_i - (A_i dot dA_i))
        float dot = (A.row(i).cwiseProduct(dA.row(i))).sum();
        dS.row(i) = A.row(i).cwiseProduct(dA.row(i) - Eigen::RowVectorXf::Constant(A.cols(), dot));
    }
    assert( assert_not_nan(dS) );
    return dS;
}
}

Eigen::MatrixXf Forward_BackwardNaive::backward_causal_self_attention(
    const Eigen::MatrixXf& gradient, 
    const Eigen::MatrixXf& input, 
    AttentionWeights& attn) 
{
    // Get configuration parameters.
    const auto& cfg = model().config();
    const int n      = input.rows();          // number of tokens
    const int n_embd = cfg.n_embd;              // full embedding dim, e.g. 768
    const int n_head = cfg.n_head;              // number of heads, e.g. 12
    const int d      = cfg.d();                // head dimension = n_embd / n_head

    // === Forward Recompute for QKV ===
    // Compute combined QKV from input; shape: (n, 3*n_embd)

    // std::cout << "WTF " << input.rows() << " " << input.cols() << " " << std::endl;

    // std::cout << "WTF " << attn.qkv.weight.rows() << " " << attn.qkv.weight.cols() << " " << std::endl;
    
    Eigen::MatrixXf QKV = forward_linear(input, attn.qkv);

    // Split QKV into Q, K, V; each of shape: (n, n_embd)
    Eigen::MatrixXf Q = QKV.leftCols(n_embd);
    Eigen::MatrixXf K = QKV.middleCols(n_embd, n_embd);
    Eigen::MatrixXf V = QKV.rightCols(n_embd);

    // === Reshape into heads (simulate 3D tensor using vectors) ===
    std::vector<Eigen::MatrixXf> Q_heads(n_head), K_heads(n_head), V_heads(n_head);
    for (int h = 0; h < n_head; ++h) {
        Q_heads[h] = Q.block(0, h * d, n, d);
        K_heads[h] = K.block(0, h * d, n, d);
        V_heads[h] = V.block(0, h * d, n, d);
    }

    // === Recompute Attention Output for Each Head ===
    // We'll need the softmax outputs and raw scores for backprop.
    std::vector<Eigen::MatrixXf> scores_heads(n_head), A_heads(n_head), O_heads(n_head);
    const float scale = 1.0 / std::sqrt(static_cast<float>(d));


    assert(d == 64);

    for (int h = 0; h < n_head; ++h) {
        // std::cout << "Loop # " << h << std::endl;
        // scores = (Q * K^T)/scale  → shape: (n, n)
        Eigen::MatrixXf scores = (Q_heads[h] * K_heads[h].transpose()) * scale;

        // std::cout << "causal mask " << std::endl;
        // Apply causal mask: for each token i, force scores(i,j) = -∞ for j > i.
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                scores(i, j) = -std::numeric_limits<float>::infinity();
            }
        }

        // std::cout << "scores[h]" << std::endl;
        scores_heads[h] = scores;
        // A = softmax(scores) row-wise.
        A_heads[h] = Eigen::MatrixXf(n, n);
        for (int i =0 ; i < n; ++i) {

        // std::cout << "soft maxwell " << i << std::endl;
            A_heads[h].row(i) = softmax(scores.row(i));
        }

        // std::cout << "about to end " << std::endl;
        // Output for head h: O = A * V, shape: (n, d)
        O_heads[h] = A_heads[h] * V_heads[h];

        // std::cout << "end loop " << h << std::endl;
    }

    // === Combine Heads: Concatenate O_heads into O of shape (n, n_embd) ===
    Eigen::MatrixXf O(n, n_embd);
    for (int h = 0; h < n_head; ++h) {
        // std::cout << "loop nhead o " << h << std::endl;
        O.block(0, h * d, n, d) = O_heads[h];
    }

    // std::cout << "BACKPROP STARTS HERE " << std::endl;

    /*
        BACKPROP STARTS HERE

        TODO: DID NOT VERIFY ANYTHING. but still seems like it should be 'simple' backprop idk. with the matrix magic transpose 
        just works out (yes ik I can derive contributions, but still 0 intuition)

        This time, I can't break up the matrix parallel-ly - so I can't even get that kind of tiny intuition. Rip. 
    */

    // === Backprop Through the Output Projection ===
    // 'gradient' is the gradient from later layers wrt the final output of c_proj.
    // Backpropagate through c_proj (assumes forward pass used O as input).
    Eigen::MatrixXf dO = backward_linear(gradient, O, attn.c_proj); // (n, n_embd)


    // std::cout << "DONE dO " << std::endl;

    // === Split dO into Heads ===
    std::vector<Eigen::MatrixXf> dO_heads(n_head);
    for (int h = 0; h < n_head; ++h) {
        dO_heads[h] = dO.block(0, h * d, n, d);
    }


    // std::cout << "DONE SPLIT " << std::endl;

    // === Backprop Through Attention per Head ===
    // For each head, we backprop through:
    //    O = A * V   ⇒  dA = dO * V^T,  dV = A^T * dO.
    // Then through the softmax: given A and dA, compute dScores.
    // Finally, for the scaled dot-product: dQ = (dScores * K)/scale and dK = (dScores^T * Q)/scale.
    std::vector<Eigen::MatrixXf> dQ_heads(n_head), dK_heads(n_head), dV_heads(n_head);
    for (int h = 0; h < n_head; ++h) {

        // std::cout << "dA " << std::endl;

        Eigen::MatrixXf dA = dO_heads[h] * V_heads[h].transpose();  // (n, n)

        // std::cout << "dV " << std::endl;

        Eigen::MatrixXf dV = A_heads[h].transpose() * dO_heads[h];      // (n, d)
        // Backprop through softmax.

        // std::cout << "dScores " << std::endl;
        Eigen::MatrixXf dScores = softmax_backward(dA, A_heads[h]);       // (n, n)
        // Backprop through the scaled dot-product: scores = (Q * K^T)/scale.


            // std::cout << "dq" << std::endl;
        Eigen::MatrixXf dQ = (dScores * K_heads[h]) / scale;              // (n, d)

        // std::cout << "dk " << std::endl;
        Eigen::MatrixXf dK = (dScores.transpose() * Q_heads[h]) / scale;    // (n, d)

        dQ_heads[h] = dQ;
        dK_heads[h] = dK;
        dV_heads[h] = dV;
    }

    // std::cout << "BACKPROP POST ATTENTION HEAD  " << std::endl;

    // === Concatenate dQ, dK, dV for all heads into dQKV of shape (n, 3*n_embd) ===
    Eigen::MatrixXf dQKV(n, 3 * n_embd);
    for (int h = 0; h < n_head; ++h) {
        dQKV.block(0, h * d, n, d) = dQ_heads[h];
        dQKV.block(0, n_embd + h * d, n, d) = dK_heads[h];
        dQKV.block(0, 2 * n_embd + h * d, n, d) = dV_heads[h];
    }


    // std::cout << " dInput " << std::endl;

    // === Backprop Through the QKV Projection ===
    // Pass the concatenated gradient back through the qkv linear layer.
    Eigen::MatrixXf dInput = backward_linear(dQKV, input, attn.qkv);

    return dInput;
}


void Forward_BackwardNaive::backward(std::vector<int> tokens, float temp) {
    // We assume tokens.size() >= 2.
    // For training we use the first N-1 tokens as input and tokens[1...N-1] as targets.
    assert(tokens.size() >= 2 && "Need at least two tokens for training.");
    int L = tokens.size() - 1; // number of predictions

    /*
        assert that model weights no fucked
    */

    assert( assert_not_nan(model().wpe()) );
    assert( assert_not_nan(model().wte()) );
    assert( assert_not_nan(model().ln_f().gamma) );
    assert( assert_not_nan(model().ln_f().beta) );

    for (auto &b: model().blocks()) {
        assert( assert_not_nan(b.attn.qkv.bias) );
        assert( assert_not_nan(b.attn.qkv.weight) );
        
        assert( assert_not_nan(b.attn.c_proj.bias) );
        assert( assert_not_nan(b.attn.c_proj.weight) );
        assert( assert_not_nan(b.ln_1.gamma) );
        assert( assert_not_nan(b.ln_1.beta) );
        assert( assert_not_nan(b.ln_2.gamma) );
        assert( assert_not_nan(b.ln_2.beta) );
        
        assert( assert_not_nan(b.mlp.to_up.bias) );
        assert( assert_not_nan(b.mlp.to_up.weight) );
        assert( assert_not_nan(b.mlp.back_down.bias) );
        assert( assert_not_nan(b.mlp.back_down.weight) );
    }

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
        // I don't know why GPT generated "out". We need INPUTS for all functions to compute gradients, not outputs.
        Eigen::MatrixXf ln1_in;  
        Eigen::MatrixXf attn_in;
        Eigen::MatrixXf ln2_in; 
        Eigen::MatrixXf mlp_in; 
    };
    std::vector<BlockActivations> acts;
    acts.reserve(model().blocks().size());

    for (const auto& block : model().blocks()) {
        // Saving residuals and x is pointless; we know they get split up. 
        // The ONLY thing that we need, for every single function, is just the INPUT (not result!!!)! 
        // That's a very nice property of backpropagation. You only get this intuition if you 
        // do it yourself (or just know the math like a smart person ig). 
        BlockActivations act;

        Eigen::MatrixXf residual = x;

        x = layer_norm(act.ln1_in = x, block.ln_1);

        assert(assert_not_nan(x) && "check 1");

        x = causal_self_attention(act.attn_in = x, block.attn);

        assert(assert_not_nan(x) && "check 2");
        

        x += residual; 

        residual = x;
        assert(assert_not_nan(x) && "check 3");
        

        x = layer_norm(act.ln2_in = x, block.ln_2);
        assert(assert_not_nan(x) && "check 4");
        
        x = mlp(act.mlp_in = x, block.mlp);
        assert(assert_not_nan(x) && "check 5");
        

        x += residual;
        acts.push_back(act);

        assert( assert_not_nan(act.ln1_in) );
        assert( assert_not_nan(act.attn_in) );
        assert( assert_not_nan(act.ln2_in) );
        assert( assert_not_nan(act.mlp_in) );

        assert( assert_not_nan(x) );
    }

    // Final layer norm.
    Eigen::MatrixXf x_final = layer_norm(x, model().ln_f());

    assert( assert_not_nan(x_final) );

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

    // We don't start from [N x V]; again we start frmo [N x e] and do a "fast skip" to avoid materializing [N x V] memory.
    Eigen::MatrixXf current_gradient(L, model().config().n_embd);
    current_gradient.setZero();

    
    

    {
        // we apply this to wte deriv
        Eigen::MatrixXf apply_wte_deriv(model().config().n_embd, model().config().vocab_size);
        apply_wte_deriv.setZero();

        // Again, we can do any sliding window # of iterations to materialize at most O(slide * V) memory.
        // Probably slide = n_embd should be good, since the max vector size anywyas is O(e * V)? Or anything smaller.
        // But **for clarity**, slide = 1.

        for (int i = 0; i < L; ++i) {
            // For token i, the target is tokens[i+1].
            int target = tokens[i + 1];

            assert_not_nan(model().lm_head());
            assert_not_nan(x_final.row(i));

            Eigen::RowVectorXf loss_vec = softmax(
                x_final.row(i) *
                model().lm_head().transpose());

            const float eps = 1e-9;
            if (!(loss_vec(target) + eps > 0)) {
                std::cout << "WTF? " << i << " | " << loss_vec(target) << std::endl;
            }
            assert(loss_vec(target) + eps > 0);

            loss -= std::log(loss_vec(target) + eps); // TODO: should I do this? this is a hack

            // this is just how softmax works
            loss_vec(target) -= 1.0f;

            // Now, vector has to be averaged, before adding to saved_weight_app
            loss_vec /= L;

            // this is still insane
            apply_wte_deriv += x_final.row(i).transpose() * loss_vec; // becomes [e x V vector]. All are scaled down implicitly.
            
            current_gradient.row(i) = loss_vec * model().wte(); // (N x V) x (V x e). All are scaled down implicitly too.
        }
        loss /= L;

        std::cout << "Loss: " << loss << std::endl;

        // Remember to subtract in all gradient applications, since it's gradient DESCENT not ASCENT. 
        // But wecan't just straight up multiply gradient by -1, thats not correct :P. 
        // Just replace all the plusses with minuses (put the fries in the bag)

        _model._wte.transpose() -= apply_wte_deriv * temp; 
    }

    // std::cout << "GOT PAST THE HARD PART!" << std::endl;

    // COMPLETE BULLSHIT SIMPLE LEARNING FOR NOW: Just scale gradients gradually down to 0
    current_gradient *= temp; 

    current_gradient = Forward_BackwardNaive::backward_layer_norm(current_gradient, x_final, _model._ln_f);

    assert( assert_not_nan(current_gradient) );

    for (int bi = static_cast<int>(model().blocks().size()) - 1; bi >= 0; --bi) {
        const auto& act = acts[bi];

        // std::cout << "Block " << bi << " skip layer " << 2 << std::endl;

        Eigen::MatrixXf skip_layer = current_gradient; // residual skips 

        // std::cout << " Backward mlp " << std::endl;

        current_gradient = Forward_BackwardNaive::backward_mlp(current_gradient, act.mlp_in, _model._h[bi].mlp);


    assert( assert_not_nan(current_gradient) );
        // std::cout << " Backward layer norm " << std::endl;

        current_gradient = Forward_BackwardNaive::backward_layer_norm(current_gradient, act.ln2_in, _model._h[bi].ln_2);


        // std::cout << " add resisaul 2" << std::endl;

        current_gradient += skip_layer; // add back the residual

        // std::cout << "Block " << bi << " skip layer " << 1 << std::endl;

        skip_layer = current_gradient; // residual skips 

        // std::cout << "Attention computation " << std::endl;

        current_gradient = Forward_BackwardNaive::backward_causal_self_attention(current_gradient, act.attn_in, _model._h[bi].attn);

        // std::cout << "backward layer norm" << std::endl;

        current_gradient = Forward_BackwardNaive::backward_layer_norm(current_gradient, act.ln1_in, _model._h[bi].ln_1);

    assert( assert_not_nan(current_gradient) );

        // std::cout << "Block " << bi << " skip layer " << 2 << std::endl;

        current_gradient += skip_layer; // add back the residual
    }

    // TODO: Okay yeah this part makes sense, it's just a simple addition pass
    // And finally we're out of matrix magic transpose land - this independent backprop makes sense. 
    // So orz....

    for (int i = 0; i < L; ++i) {
        _model._wte.row(tokens[i]) += current_gradient.row(i);
        _model._wpe.row(i) += current_gradient.row(i);
    }
}


// START OF ALL FORWARD FUNCTIONS

Eigen::MatrixXf Forward_BackwardNaive::causal_self_attention(Eigen::MatrixXf x, const AttentionWeights& attn) {
    int T = x.rows();
    int C = x.cols();

    int n_head = model().config().n_head;

    assert(C % n_head == 0);
    int head_dim = C / n_head; // Each head gets C/n_head features

    assert_not_nan(x);
    assert_not_nan(attn.qkv.weight);
    assert_not_nan(attn.qkv.bias);
    

    Eigen::MatrixXf qkv = forward_linear(x, attn.qkv);

    Eigen::MatrixXf q = qkv.leftCols(C);
    Eigen::MatrixXf k = qkv.middleCols(C, C);
    Eigen::MatrixXf v = qkv.rightCols(C);

    Eigen::MatrixXf y(T, C);

    for (int h = 0; h < n_head; h++) {
        Eigen::MatrixXf q_h = q.middleCols(h * head_dim, head_dim);
        Eigen::MatrixXf k_h = k.middleCols(h * head_dim, head_dim);
        Eigen::MatrixXf v_h = v.middleCols(h * head_dim, head_dim);

        assert(assert_not_nan(q_h));
        assert(assert_not_nan(k_h));
        assert(assert_not_nan(v_h));

        
        

        Eigen::MatrixXf att_h = q_h * k_h.transpose();

        assert(assert_not_nan(att_h));
        
        assert(head_dim == 64);

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        att_h *= scale;

        assert(assert_not_nan(att_h));

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

namespace{ 
float _gelu(float x) {
    const float GELU_SCALE = std::sqrt(2.0f / static_cast<float>(M_PI));
    float cube = 0.044715f * x * x * x;
    float res = 0.5f * x * (1.0f + tanhf(GELU_SCALE * (x + cube)));

    assert(is_normal(res));
    return res;
}
}

Eigen::MatrixXf Forward_BackwardNaive::gelu(Eigen::MatrixXf x) { return x.unaryExpr(&_gelu); }
