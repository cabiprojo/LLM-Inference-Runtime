#include <iostream>
#include <string>
#include <vector>

#include "io/tensor_io.h"

#include <cmath>

namespace {

// debug helper, prints a tensor's shape, not part of the model
void print_shape(const std::unordered_map<std::string, Tensor>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        std::cout << "  " << name << ": NOT FOUND\n";
        return;
    }
    std::cout << "  " << name << ": [";
    const auto& shape = it->second.shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    std::cout << "]\n";
}

}  // namespace


// normalizes each token's vector to mean 0 variance 1, then applies a learned scale and shift
// x - input tensor of shape (seq_len, n_embd)
// gamma - scale tensor of shape (n_embd)
// beta - shift tensor of shape (n_embd)
// out - output tensor of shape (seq_len, n_embd)
// seq_len - number of sequences (batch size)
// n_embd - embedding dimension
// eps - small constant for numerical stability
void layer_norm(const float* x, const float* gamma, const float* beta, 
    float* out, int seq_len, int n_embd, float eps) {
        for (int i = 0; i < seq_len; ++i) {
            float mean = 0.0f;
            float variance = 0.0f;

            // compute mean
            for (int j = 0; j < n_embd; ++j) {
                mean += x[i * n_embd + j];
            }
            mean /= n_embd;

            // compute variance
            for (int j = 0; j < n_embd; ++j) {
                float diff = x[i * n_embd + j] - mean;
                variance += diff * diff;
            }
            variance /= n_embd;

            // normalize and apply scale and shift
            for (int j = 0; j < n_embd; ++j) {
                out[i * n_embd + j] = gamma[j] * ((x[i * n_embd + j] - mean) / std::sqrt(variance + eps)) + beta[j];
            }
        }
        
}

// turns token ids into vectors by summing token and position embeddings
// token_ids: seq_len token ids
// wte: token embedding table, shape (vocab_size, n_embd)
// wpe: position embedding table, shape (n_ctx, n_embd)
// out: shape (seq_len, n_embd), out[i] = wte[token_ids[i]] + wpe[i]
void embed(const int* token_ids, const float* wte, const float* wpe,
           float* out, int seq_len, int n_embd) {
    for (int i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        for (int j = 0; j < n_embd; ++j) {
            out[i * n_embd + j] = wte[token_id * n_embd + j] + wpe[i * n_embd + j];
        }
    }
}

// nonlinearity applied inside the mlp
// GPT-2's tanh-approximation gelu, applied elementwise
// n = total element count
void gelu(const float* x, float* out, int n) {
    constexpr float k = 0.7978845608f;  // sqrt(2/pi)
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + std::tanh(k * (v + 0.044715f * v * v * v)));
    }
}

// turns raw attention scores into per-row probabilities, blocking future positions
// in-place causal softmax over a (seq_len, seq_len) attention score matrix
// row i may only attend to columns 0..i (GPT-2 is autoregressive)
// each visible row is turned into a probability distribution via softmax
// columns j > i are zeroed since they should never contribute to attn_weights @ V
void causal_softmax(float* scores, int seq_len) {
    for (int i = 0; i < seq_len; ++i) {
        float* row = scores + i * seq_len;

        // numerically stable softmax, subtract the row's max before exp()
        // so it never overflows
        // restricted to the visible columns 0..i
        float max_val = row[0];
        for (int j = 1; j <= i; ++j) {
            if (row[j] > max_val) max_val = row[j];
        }

        float sum = 0.0f;
        for (int j = 0; j <= i; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }
        for (int j = 0; j <= i; ++j) {
            row[j] /= sum;
        }
        for (int j = i + 1; j < seq_len; ++j) {
            row[j] = 0.0f;  // masked out
        }
    }
}

// picks the most likely next token during generation, not part of the forward pass
// index of the largest value in logits[0..vocab_size)
// greedy next-token pick
int argmax(const float* logits, int vocab_size) {
    int best = 0;
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

// measures how far a computed buffer is from the reference
// validation harness
// compares a computed buffer against a reference checkpoint and reports max abs diff
// anything above ~1e-3 usually means a real bug, not just float rounding
float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

// true if every element is within atol + rtol * |reference value| (numpy's allclose rule)
// absolute tolerance alone breaks once activation magnitudes grow into the
// hundreds/thousands in later blocks, since fp32 rounding grows proportionally
bool all_close(const float* computed, const float* reference, int n, float atol, float rtol) {
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(computed[i] - reference[i]);
        float tol = atol + rtol * std::fabs(reference[i]);
        if (diff > tol) return false;
    }
    return true;
}

// prints pass or fail by comparing computed output against a reference checkpoint
void check(const std::string& label, const float* computed, const float* reference, int n) {
    float diff = max_abs_diff(computed, reference, n);
    bool pass = all_close(computed, reference, n, 1e-3f, 1e-3f);
    std::cout << (pass ? "[PASS] " : "[FAIL] ") << label
               << " max abs diff = " << diff << "\n";
}

// A, B are the two input matrices, C is the output matrix
// M = number of rows in A, K = number of columns in A (and rows in B), N = number of columns in B
void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// out = x @ W^T + b
// x: (seq_len, in_features), W: (out_features, in_features), b: (out_features)
// out: (seq_len, out_features)
void linear(const float* x, const float* W, const float* b,
            float* out, int seq_len, int in_features, int out_features) {
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                sum += x[i * in_features + k] * W[j * in_features + k];
            }
            out[i * out_features + j] = sum + b[j];
        }
    }
}

// multi-head causal self-attention for one block
// x: (seq_len, n_embd), layer-normed input (ln_1 output)
// c_attn_w: (3*n_embd, n_embd), c_attn_b: (3*n_embd), combined qkv projection
// c_proj_w: (n_embd, n_embd), c_proj_b: (n_embd), output projection
// out: (seq_len, n_embd)
void attention(const float* x, const float* c_attn_w, const float* c_attn_b,
               const float* c_proj_w, const float* c_proj_b,
               float* out, int seq_len, int n_embd, int n_head) {
    int head_dim = n_embd / n_head;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // combined q, k, v projection in one linear call
    std::vector<float> qkv(seq_len * 3 * n_embd);
    linear(x, c_attn_w, c_attn_b, qkv.data(), seq_len, n_embd, 3 * n_embd);

    std::vector<float> concat(seq_len * n_embd);
    std::vector<float> Qh(seq_len * head_dim);
    std::vector<float> Kh(seq_len * head_dim);
    std::vector<float> Vh(seq_len * head_dim);
    std::vector<float> scores(seq_len * seq_len);
    std::vector<float> head_out(seq_len * head_dim);

    for (int h = 0; h < n_head; ++h) {
        // pull this head's 64-wide slice out of the packed qkv buffer
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                Qh[i * head_dim + d] = qkv[i * 3 * n_embd + h * head_dim + d];
                Kh[i * head_dim + d] = qkv[i * 3 * n_embd + n_embd + h * head_dim + d];
                Vh[i * head_dim + d] = qkv[i * 3 * n_embd + 2 * n_embd + h * head_dim + d];
            }
        }

        // scores = Qh @ Kh^T * scale
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    sum += Qh[i * head_dim + d] * Kh[j * head_dim + d];
                }
                scores[i * seq_len + j] = sum * scale;
            }
        }

        causal_softmax(scores.data(), seq_len);

        // head_out = scores @ Vh
        matmul(scores.data(), Vh.data(), head_out.data(), seq_len, seq_len, head_dim);

        // write this head's output into its slice of the concat buffer
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                concat[i * n_embd + h * head_dim + d] = head_out[i * head_dim + d];
            }
        }
    }

    linear(concat.data(), c_proj_w, c_proj_b, out, seq_len, n_embd, n_embd);
}

// one transformer block, x is updated in place to become this block's output
// ln_1 -> attention -> residual add -> ln_2 -> mlp -> residual add
// prefix selects this block's weights, e.g. "h.0."
void transformer_block(float* x, const std::unordered_map<std::string, Tensor>& weights,
                        const std::string& prefix, int seq_len, int n_embd, int n_head, float eps) {
    std::vector<float> ln1_out(seq_len * n_embd);
    layer_norm(x, weights.at(prefix + "ln_1.weight").data.data(),
               weights.at(prefix + "ln_1.bias").data.data(),
               ln1_out.data(), seq_len, n_embd, eps);

    std::vector<float> attn_out(seq_len * n_embd);
    attention(ln1_out.data(),
              weights.at(prefix + "attn.c_attn.weight").data.data(),
              weights.at(prefix + "attn.c_attn.bias").data.data(),
              weights.at(prefix + "attn.c_proj.weight").data.data(),
              weights.at(prefix + "attn.c_proj.bias").data.data(),
              attn_out.data(), seq_len, n_embd, n_head);

    for (int i = 0; i < seq_len * n_embd; ++i) {
        x[i] += attn_out[i];
    }

    std::vector<float> ln2_out(seq_len * n_embd);
    layer_norm(x, weights.at(prefix + "ln_2.weight").data.data(),
               weights.at(prefix + "ln_2.bias").data.data(),
               ln2_out.data(), seq_len, n_embd, eps);

    int mlp_hidden = weights.at(prefix + "mlp.c_fc.weight").shape[0];
    std::vector<float> fc_out(seq_len * mlp_hidden);
    linear(ln2_out.data(), weights.at(prefix + "mlp.c_fc.weight").data.data(),
           weights.at(prefix + "mlp.c_fc.bias").data.data(),
           fc_out.data(), seq_len, n_embd, mlp_hidden);
    gelu(fc_out.data(), fc_out.data(), seq_len * mlp_hidden);

    std::vector<float> mlp_out(seq_len * n_embd);
    linear(fc_out.data(), weights.at(prefix + "mlp.c_proj.weight").data.data(),
           weights.at(prefix + "mlp.c_proj.bias").data.data(),
           mlp_out.data(), seq_len, mlp_hidden, n_embd);

    for (int i = 0; i < seq_len * n_embd; ++i) {
        x[i] += mlp_out[i];
    }
}


// loads real GPT-2 weights and reference activations, then tests pieces of the model against them
int main() {
    const std::string weights_path = "../data/gpt2_weights.bin";
    const std::string activations_path = "../data/reference_activations.bin";

    auto weights = load_tensors(weights_path);
    auto activations = load_tensors(activations_path);

    std::cout << "Loaded " << weights.size() << " weight tensors and "
              << activations.size() << " reference activation tensors.\n\n";

    const int seq_len = activations.at("input_ids").shape[0];
    const int n_embd = weights.at("wte").shape[1];

    // input_ids were stored as float32 (see tensor_io's format)
    // cast back to int
    std::vector<int> input_ids(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        input_ids[i] = static_cast<int>(activations.at("input_ids").data[i]);
    }

    // test 1: embed() against the "embeddings" checkpoint
    std::vector<float> computed_embeddings(seq_len * n_embd);
    embed(input_ids.data(), weights.at("wte").data.data(), weights.at("wpe").data.data(),
          computed_embeddings.data(), seq_len, n_embd);
    check("embed()", computed_embeddings.data(), activations.at("embeddings").data.data(),
          seq_len * n_embd);

    // test 2: layer_norm() against the "block_0_ln1" checkpoint
    std::vector<float> computed_ln1(seq_len * n_embd);
    layer_norm(computed_embeddings.data(),
               weights.at("h.0.ln_1.weight").data.data(),
               weights.at("h.0.ln_1.bias").data.data(),
               computed_ln1.data(), seq_len, n_embd, 1e-5f);
    check("layer_norm() [block 0, ln_1]", computed_ln1.data(),
          activations.at("block_0_ln1").data.data(), seq_len * n_embd);

    // test 3: matmul() against a hand-computed 2x2 example
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]] -> A@B = [[19,22],[43,50]]
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float expected[4] = {19, 22, 43, 50};
    float computed_matmul[4];
    matmul(A, B, computed_matmul, 2, 2, 2);
    check("matmul() [hand-computed 2x2]", computed_matmul, expected, 4);

    // test 4: linear() + gelu() chained through the real MLP, against "block_0_mlp"
    // uses the reference block_0_ln2 checkpoint as input, isolating this test from
    // block loop wiring (not written yet) so a failure here means linear/gelu, not wiring
    const int mlp_hidden = weights.at("h.0.mlp.c_fc.weight").shape[0];
    std::vector<float> fc_out(seq_len * mlp_hidden);
    linear(activations.at("block_0_ln2").data.data(),
           weights.at("h.0.mlp.c_fc.weight").data.data(),
           weights.at("h.0.mlp.c_fc.bias").data.data(),
           fc_out.data(), seq_len, n_embd, mlp_hidden);
    gelu(fc_out.data(), fc_out.data(), seq_len * mlp_hidden);

    std::vector<float> computed_mlp(seq_len * n_embd);
    linear(fc_out.data(),
           weights.at("h.0.mlp.c_proj.weight").data.data(),
           weights.at("h.0.mlp.c_proj.bias").data.data(),
           computed_mlp.data(), seq_len, mlp_hidden, n_embd);
    check("linear()+gelu() [block 0 mlp]", computed_mlp.data(),
          activations.at("block_0_mlp").data.data(), seq_len * n_embd);

    // test 5: attention() against the "block_0_attn" checkpoint
    // uses the reference block_0_ln1 checkpoint as input, isolating this test
    // from block loop wiring (not written yet)
    const int n_head = 12;
    std::vector<float> computed_attn(seq_len * n_embd);
    attention(activations.at("block_0_ln1").data.data(),
              weights.at("h.0.attn.c_attn.weight").data.data(),
              weights.at("h.0.attn.c_attn.bias").data.data(),
              weights.at("h.0.attn.c_proj.weight").data.data(),
              weights.at("h.0.attn.c_proj.bias").data.data(),
              computed_attn.data(), seq_len, n_embd, n_head);
    check("attention() [block 0]", computed_attn.data(),
          activations.at("block_0_attn").data.data(), seq_len * n_embd);

    // test 6: full forward pass, embed -> 12 blocks -> final ln -> logits
    const int n_layer = 12;
    const float eps = 1e-5f;

    std::vector<float> x(seq_len * n_embd);
    embed(input_ids.data(), weights.at("wte").data.data(), weights.at("wpe").data.data(),
          x.data(), seq_len, n_embd);

    for (int layer = 0; layer < n_layer; ++layer) {
        std::string prefix = "h." + std::to_string(layer) + ".";
        transformer_block(x.data(), weights, prefix, seq_len, n_embd, n_head, eps);
        std::string ref_name = "block_" + std::to_string(layer) + "_out";
        check("  block_" + std::to_string(layer) + "_out", x.data(),
              activations.at(ref_name).data.data(), seq_len * n_embd);
    }

    std::vector<float> final_ln_out(seq_len * n_embd);
    layer_norm(x.data(), weights.at("ln_f.weight").data.data(),
               weights.at("ln_f.bias").data.data(),
               final_ln_out.data(), seq_len, n_embd, eps);
    check("full forward [final_ln]", final_ln_out.data(),
          activations.at("final_ln").data.data(), seq_len * n_embd);

    // logits = final_ln_out @ wte^T, no bias -- weight tying reuses the token
    // embedding matrix, so linear() works here with a zero bias vector
    const int vocab_size = weights.at("wte").shape[0];
    std::vector<float> zero_bias(vocab_size, 0.0f);
    std::vector<float> computed_logits(seq_len * vocab_size);
    linear(final_ln_out.data(), weights.at("wte").data.data(), zero_bias.data(),
           computed_logits.data(), seq_len, n_embd, vocab_size);
    check("full forward [logits]", computed_logits.data(),
          activations.at("logits").data.data(), seq_len * vocab_size);

    return 0;
}
