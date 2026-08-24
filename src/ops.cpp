#include "ops.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <string>
#include <thread>
#include <unordered_set>
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

void embed_one(int token_id, int position, const float* wte, const float* wpe,
               float* out, int n_embd) {
    for (int j = 0; j < n_embd; ++j) {
        out[j] = wte[token_id * n_embd + j] + wpe[position * n_embd + j];
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

int pick_next_token(const float* logits, int vocab_size,
                     const std::vector<int>& ids, int ngram_size) {
    int prefix_len = ngram_size - 1;
    if (ngram_size <= 0 || static_cast<int>(ids.size()) < prefix_len) {
        return argmax(logits, vocab_size);
    }

    // the most recent (ngram_size - 1) tokens -- the "prefix" about to be extended
    const int* prefix = ids.data() + ids.size() - prefix_len;

    // every token that has already followed this exact prefix before -- picking
    // any of them again would recreate an n-gram already generated earlier
    std::unordered_set<int> banned;
    for (int i = 0; i + ngram_size <= static_cast<int>(ids.size()); ++i) {
        bool matches = true;
        for (int k = 0; k < prefix_len; ++k) {
            if (ids[i + k] != prefix[k]) { matches = false; break; }
        }
        if (matches) banned.insert(ids[i + prefix_len]);
    }

    if (banned.empty()) return argmax(logits, vocab_size);

    int best = -1;
    for (int i = 0; i < vocab_size; ++i) {
        if (banned.count(i)) continue;
        if (best == -1 || logits[i] > logits[best]) best = i;
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


// same computation as matmul(), reordered into TILE x TILE blocks so a chunk
// of A, B, and C stay resident in cache while they're being reused, instead
// of re-fetching from RAM for every single output element
void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N, int TILE) {
    for (int idx = 0; idx < M * N; ++idx) C[idx] = 0.0f;

    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int i_max = std::min(i0 + TILE, M);
                int j_max = std::min(j0 + TILE, N);
                int k_max = std::min(k0 + TILE, K);

                for (int i = i0; i < i_max; ++i) {
                    for (int j = j0; j < j_max; ++j) {
                        float sum = C[i * N + j];
                        for (int k = k0; k < k_max; ++k) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

// same result as matmul_tiled, but the inner loop is vectorized across j
// (8 output columns at once) instead of k, since B[k*N+j] is contiguous in j
// but strided in k -- SIMD wants the contiguous direction
void matmul_simd(const float* A, const float* B, float* C, int M, int K, int N, int TILE) {
    for (int idx = 0; idx < M * N; ++idx) C[idx] = 0.0f;

    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int i_max = std::min(i0 + TILE, M);
                int j_max = std::min(j0 + TILE, N);
                int k_max = std::min(k0 + TILE, K);

                for (int i = i0; i < i_max; ++i) {
                    int j = j0;

                    // 8 columns at a time with AVX2
                    for (; j + 8 <= j_max; j += 8) {
                        __m256 acc = _mm256_loadu_ps(&C[i * N + j]);

                        for (int k = k0; k < k_max; ++k) {
                            __m256 a_bcast = _mm256_set1_ps(A[i * K + k]);
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            acc = _mm256_fmadd_ps(a_bcast, b_vec, acc);
                        }

                        _mm256_storeu_ps(&C[i * N + j], acc);
                    }

                    // scalar fallback for leftover columns, when (j_max - j0) isn't a multiple of 8
                    for (; j < j_max; ++j) {
                        float sum = C[i * N + j];
                        for (int k = k0; k < k_max; ++k) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

void matmul_threaded(const float* A, const float* B, float* C, int M, int K, int N,
                      int TILE, int num_threads) {
    std::vector<std::thread> threads;
    int rows_per_thread = (M + num_threads - 1) / num_threads;  // ceiling division

    for (int t = 0; t < num_threads; ++t) {
        int row_start = t * rows_per_thread;
        int row_end = std::min(row_start + rows_per_thread, M);
        if (row_start >= row_end) break;  // more threads than rows

        // each thread computes rows [row_start, row_end) of C -- a disjoint
        // slice, so no two threads ever touch the same memory
        threads.emplace_back([=]() {
            matmul_simd(A + row_start * K, B, C + row_start * N,
                        row_end - row_start, K, N, TILE);
        });
    }

    for (auto& t : threads) t.join();
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

// same tiling idea as matmul_tiled, applied to linear()'s x @ W^T + b.
// unlike matmul, both x and W are already contiguously accessed here -- but the
// full W matrix (e.g. 9.4 MB for the mlp c_fc weight) still doesn't fit in
// cache, so it still gets evicted and re-fetched from RAM once per row of x
// without tiling. this fixes that the same way: work on one small chunk of
// W at a time, fully, before moving to the next chunk.
void linear_tiled(const float* x, const float* W, const float* b,
                   float* out, int seq_len, int in_features, int out_features, int TILE) {
    // start each output at its bias, then accumulate x @ W^T contributions across k0 chunks
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < out_features; ++j) {
            out[i * out_features + j] = b[j];
        }
    }

    for (int i0 = 0; i0 < seq_len; i0 += TILE) {
        for (int j0 = 0; j0 < out_features; j0 += TILE) {
            for (int k0 = 0; k0 < in_features; k0 += TILE) {
                int i_max = std::min(i0 + TILE, seq_len);
                int j_max = std::min(j0 + TILE, out_features);
                int k_max = std::min(k0 + TILE, in_features);

                for (int i = i0; i < i_max; ++i) {
                    for (int j = j0; j < j_max; ++j) {
                        float sum = out[i * out_features + j];
                        for (int k = k0; k < k_max; ++k) {
                            sum += x[i * in_features + k] * W[j * in_features + k];
                        }
                        out[i * out_features + j] = sum;
                    }
                }
            }
        }
    }
}

void linear_threaded(const float* x, const float* W, const float* b,
                      float* out, int seq_len, int in_features, int out_features,
                      int TILE, int num_threads) {
    std::vector<std::thread> threads;
    int rows_per_thread = (seq_len + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int row_start = t * rows_per_thread;
        int row_end = std::min(row_start + rows_per_thread, seq_len);
        if (row_start >= row_end) break;

        // each thread computes rows [row_start, row_end) of out -- a disjoint
        // slice, W and b are shared read-only across all threads
        threads.emplace_back([=]() {
            linear_tiled(x + row_start * in_features, W, b, out + row_start * out_features,
                         row_end - row_start, in_features, out_features, TILE);
        });
    }

    for (auto& t : threads) t.join();
}

namespace {

// dispatches to linear(), linear_tiled(), or linear_threaded() depending on
// use_tiled/use_threaded -- lets attention()/transformer_block() share one
// code path for all three variants. use_threaded implies tiled internally
// (linear_threaded calls linear_tiled per thread), so it takes priority.
void call_linear(bool use_tiled, bool use_threaded, const float* x, const float* W, const float* b,
                  float* out, int seq_len, int in_features, int out_features) {
    if (use_threaded) {
        linear_threaded(x, W, b, out, seq_len, in_features, out_features, 16, 4);
    } else if (use_tiled) {
        linear_tiled(x, W, b, out, seq_len, in_features, out_features, 16);
    } else {
        linear(x, W, b, out, seq_len, in_features, out_features);
    }
}

}  // namespace

void attention(const float* x, const float* c_attn_w, const float* c_attn_b,
               const float* c_proj_w, const float* c_proj_b,
               float* out, int seq_len, int n_embd, int n_head, bool use_tiled,
               LayerKVCache* cache, bool use_threaded) {
    int head_dim = n_embd / n_head;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // combined q, k, v projection in one linear call
    std::vector<float> qkv(seq_len * 3 * n_embd);
    call_linear(use_tiled, use_threaded, x, c_attn_w, c_attn_b, qkv.data(), seq_len, n_embd, 3 * n_embd);

    // prefill: populate the cache with every position's K/V so decode steps
    // afterward don't need to recompute them
    if (cache != nullptr) {
        cache->K.reserve(cache->K.size() + seq_len * n_embd);
        cache->V.reserve(cache->V.size() + seq_len * n_embd);
        for (int i = 0; i < seq_len; ++i) {
            const float* k_row = qkv.data() + i * 3 * n_embd + n_embd;
            const float* v_row = qkv.data() + i * 3 * n_embd + 2 * n_embd;
            cache->K.insert(cache->K.end(), k_row, k_row + n_embd);
            cache->V.insert(cache->V.end(), v_row, v_row + n_embd);
        }
        cache->cached_len += seq_len;
    }

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

    call_linear(use_tiled, use_threaded, concat.data(), c_proj_w, c_proj_b, out, seq_len, n_embd, n_embd);
}

void attention_decode(const float* x_new, const float* c_attn_w, const float* c_attn_b,
                       const float* c_proj_w, const float* c_proj_b,
                       float* out, int n_embd, int n_head, LayerKVCache& cache, bool use_tiled) {
    int head_dim = n_embd / n_head;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // q, k, v for just this one new token
    std::vector<float> qkv_new(3 * n_embd);
    call_linear(use_tiled, false, x_new, c_attn_w, c_attn_b, qkv_new.data(), 1, n_embd, 3 * n_embd);
    const float* q_new = qkv_new.data();

    // append this token's own K, V onto the cache -- it can attend to itself too
    const float* k_new = qkv_new.data() + n_embd;
    const float* v_new = qkv_new.data() + 2 * n_embd;
    cache.K.insert(cache.K.end(), k_new, k_new + n_embd);
    cache.V.insert(cache.V.end(), v_new, v_new + n_embd);
    cache.cached_len += 1;

    int total_len = cache.cached_len;  // includes the new token, just appended
    std::vector<float> concat(n_embd);
    std::vector<float> scores(total_len);

    for (int h = 0; h < n_head; ++h) {
        // scores = q_new_h @ K_cache_h^T * scale -- no causal mask needed,
        // every cached position is <= this token's own position by construction
        for (int j = 0; j < total_len; ++j) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                sum += q_new[h * head_dim + d] * cache.K[j * n_embd + h * head_dim + d];
            }
            scores[j] = sum * scale;
        }

        // plain softmax over all total_len cached positions (numerically stable)
        float max_val = scores[0];
        for (int j = 1; j < total_len; ++j) {
            if (scores[j] > max_val) max_val = scores[j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < total_len; ++j) {
            scores[j] = std::exp(scores[j] - max_val);
            sum_exp += scores[j];
        }
        for (int j = 0; j < total_len; ++j) {
            scores[j] /= sum_exp;
        }

        // this head's output = weighted sum of every cached V, weighted by scores
        for (int d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            for (int j = 0; j < total_len; ++j) {
                sum += scores[j] * cache.V[j * n_embd + h * head_dim + d];
            }
            concat[h * head_dim + d] = sum;
        }
    }

    call_linear(use_tiled, false, concat.data(), c_proj_w, c_proj_b, out, 1, n_embd, n_embd);
}

void transformer_block(float* x, const std::unordered_map<std::string, Tensor>& weights,
                        const std::string& prefix, int seq_len, int n_embd, int n_head, float eps,
                        bool use_tiled, LayerKVCache* cache, bool use_threaded) {
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
              attn_out.data(), seq_len, n_embd, n_head, use_tiled, cache, use_threaded);

    for (int i = 0; i < seq_len * n_embd; ++i) {
        x[i] += attn_out[i];
    }

    std::vector<float> ln2_out(seq_len * n_embd);
    layer_norm(x, weights.at(prefix + "ln_2.weight").data.data(),
               weights.at(prefix + "ln_2.bias").data.data(),
               ln2_out.data(), seq_len, n_embd, eps);

    int mlp_hidden = weights.at(prefix + "mlp.c_fc.weight").shape[0];
    std::vector<float> fc_out(seq_len * mlp_hidden);
    call_linear(use_tiled, use_threaded, ln2_out.data(), weights.at(prefix + "mlp.c_fc.weight").data.data(),
                weights.at(prefix + "mlp.c_fc.bias").data.data(),
                fc_out.data(), seq_len, n_embd, mlp_hidden);
    gelu(fc_out.data(), fc_out.data(), seq_len * mlp_hidden);

    std::vector<float> mlp_out(seq_len * n_embd);
    call_linear(use_tiled, use_threaded, fc_out.data(), weights.at(prefix + "mlp.c_proj.weight").data.data(),
                weights.at(prefix + "mlp.c_proj.bias").data.data(),
                mlp_out.data(), seq_len, mlp_hidden, n_embd);

    for (int i = 0; i < seq_len * n_embd; ++i) {
        x[i] += mlp_out[i];
    }
}

void transformer_block_decode(float* x_new, const std::unordered_map<std::string, Tensor>& weights,
                               const std::string& prefix, int n_embd, int n_head, float eps,
                               LayerKVCache& cache, bool use_tiled) {
    std::vector<float> ln1_out(n_embd);
    layer_norm(x_new, weights.at(prefix + "ln_1.weight").data.data(),
               weights.at(prefix + "ln_1.bias").data.data(),
               ln1_out.data(), 1, n_embd, eps);

    std::vector<float> attn_out(n_embd);
    attention_decode(ln1_out.data(),
                      weights.at(prefix + "attn.c_attn.weight").data.data(),
                      weights.at(prefix + "attn.c_attn.bias").data.data(),
                      weights.at(prefix + "attn.c_proj.weight").data.data(),
                      weights.at(prefix + "attn.c_proj.bias").data.data(),
                      attn_out.data(), n_embd, n_head, cache, use_tiled);

    for (int i = 0; i < n_embd; ++i) {
        x_new[i] += attn_out[i];
    }

    std::vector<float> ln2_out(n_embd);
    layer_norm(x_new, weights.at(prefix + "ln_2.weight").data.data(),
               weights.at(prefix + "ln_2.bias").data.data(),
               ln2_out.data(), 1, n_embd, eps);

    // note: no use_threaded here -- this is always a single row (seq_len=1)
    // during decode, so there's nothing to split across threads
    int mlp_hidden = weights.at(prefix + "mlp.c_fc.weight").shape[0];
    std::vector<float> fc_out(mlp_hidden);
    call_linear(use_tiled, false, ln2_out.data(), weights.at(prefix + "mlp.c_fc.weight").data.data(),
                weights.at(prefix + "mlp.c_fc.bias").data.data(),
                fc_out.data(), 1, n_embd, mlp_hidden);
    gelu(fc_out.data(), fc_out.data(), mlp_hidden);

    std::vector<float> mlp_out(n_embd);
    call_linear(use_tiled, false, fc_out.data(), weights.at(prefix + "mlp.c_proj.weight").data.data(),
                weights.at(prefix + "mlp.c_proj.bias").data.data(),
                mlp_out.data(), 1, mlp_hidden, n_embd);

    for (int i = 0; i < n_embd; ++i) {
        x_new[i] += mlp_out[i];
    }
}

std::vector<int> generate(const std::vector<int>& prompt_ids,
                           const std::unordered_map<std::string, Tensor>& weights,
                           int n_embd, int n_head, int n_layer, float eps,
                           int num_new_tokens, bool use_tiled, bool use_threaded,
                           int no_repeat_ngram_size) {
    std::vector<int> ids = prompt_ids;
    const int vocab_size = weights.at("wte").shape[0];
    std::vector<float> zero_bias(vocab_size, 0.0f);

    for (int step = 0; step < num_new_tokens; ++step) {
        int seq_len = static_cast<int>(ids.size());

        std::vector<float> x(seq_len * n_embd);
        embed(ids.data(), weights.at("wte").data.data(), weights.at("wpe").data.data(),
              x.data(), seq_len, n_embd);

        for (int layer = 0; layer < n_layer; ++layer) {
            std::string prefix = "h." + std::to_string(layer) + ".";
            transformer_block(x.data(), weights, prefix, seq_len, n_embd, n_head, eps,
                               use_tiled, nullptr, use_threaded);
        }

        std::vector<float> final_ln_out(seq_len * n_embd);
        layer_norm(x.data(), weights.at("ln_f.weight").data.data(),
                   weights.at("ln_f.bias").data.data(),
                   final_ln_out.data(), seq_len, n_embd, eps);

        // only the last position's logits are needed to pick the next token,
        // so linear() is called on just that one row instead of the whole sequence
        const float* last_row = final_ln_out.data() + (seq_len - 1) * n_embd;
        std::vector<float> logits(vocab_size);
        linear(last_row, weights.at("wte").data.data(), zero_bias.data(),
               logits.data(), 1, n_embd, vocab_size);

        int next_id = pick_next_token(logits.data(), vocab_size, ids, no_repeat_ngram_size);
        ids.push_back(next_id);
    }

    return ids;
}

std::vector<int> generate_cached(const std::vector<int>& prompt_ids,
                                  const std::unordered_map<std::string, Tensor>& weights,
                                  int n_embd, int n_head, int n_layer, float eps,
                                  int num_new_tokens, bool use_tiled, bool use_threaded,
                                  int no_repeat_ngram_size) {
    std::vector<int> ids = prompt_ids;
    if (num_new_tokens <= 0) return ids;

    const int vocab_size = weights.at("wte").shape[0];
    std::vector<float> zero_bias(vocab_size, 0.0f);
    const int prompt_len = static_cast<int>(prompt_ids.size());
    std::vector<LayerKVCache> cache(n_layer);

    // prefill: process the whole prompt in one pass, populating every layer's
    // cache, then pick the first new token from the prompt's last position.
    // use_threaded only matters here -- decode below is always a single row
    std::vector<float> x(prompt_len * n_embd);
    embed(ids.data(), weights.at("wte").data.data(), weights.at("wpe").data.data(),
          x.data(), prompt_len, n_embd);

    for (int layer = 0; layer < n_layer; ++layer) {
        std::string prefix = "h." + std::to_string(layer) + ".";
        transformer_block(x.data(), weights, prefix, prompt_len, n_embd, n_head, eps,
                           use_tiled, &cache[layer], use_threaded);
    }

    std::vector<float> final_ln_out(prompt_len * n_embd);
    layer_norm(x.data(), weights.at("ln_f.weight").data.data(),
               weights.at("ln_f.bias").data.data(),
               final_ln_out.data(), prompt_len, n_embd, eps);

    const float* last_row = final_ln_out.data() + (prompt_len - 1) * n_embd;
    std::vector<float> logits(vocab_size);
    linear(last_row, weights.at("wte").data.data(), zero_bias.data(),
           logits.data(), 1, n_embd, vocab_size);
    ids.push_back(pick_next_token(logits.data(), vocab_size, ids, no_repeat_ngram_size));

    // decode: one new token at a time, reusing the cache instead of recomputing
    // the whole sequence -- this is the loop that's actually fast
    for (int step = 1; step < num_new_tokens; ++step) {
        int position = static_cast<int>(ids.size()) - 1;  // this new token's absolute position
        int token_id = ids.back();

        std::vector<float> x_new(n_embd);
        embed_one(token_id, position, weights.at("wte").data.data(), weights.at("wpe").data.data(),
                  x_new.data(), n_embd);

        for (int layer = 0; layer < n_layer; ++layer) {
            std::string prefix = "h." + std::to_string(layer) + ".";
            transformer_block_decode(x_new.data(), weights, prefix, n_embd, n_head, eps,
                                      cache[layer], use_tiled);
        }

        std::vector<float> final_ln_new(n_embd);
        layer_norm(x_new.data(), weights.at("ln_f.weight").data.data(),
                   weights.at("ln_f.bias").data.data(),
                   final_ln_new.data(), 1, n_embd, eps);

        std::vector<float> logits_new(vocab_size);
        linear(final_ln_new.data(), weights.at("wte").data.data(), zero_bias.data(),
               logits_new.data(), 1, n_embd, vocab_size);
        ids.push_back(pick_next_token(logits_new.data(), vocab_size, ids, no_repeat_ngram_size));
    }

    return ids;
}
