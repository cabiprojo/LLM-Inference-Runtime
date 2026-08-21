#include "ops.h"

#include <cmath>
#include <string>
#include <vector>

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

void embed(const int* token_ids, const float* wte, const float* wpe,
           float* out, int seq_len, int n_embd) {
    for (int i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        for (int j = 0; j < n_embd; ++j) {
            out[i * n_embd + j] = wte[token_id * n_embd + j] + wpe[i * n_embd + j];
        }
    }
}

void gelu(const float* x, float* out, int n) {
    constexpr float k = 0.7978845608f;  // sqrt(2/pi)
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + std::tanh(k * (v + 0.044715f * v * v * v)));
    }
}

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

int argmax(const float* logits, int vocab_size) {
    int best = 0;
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

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
