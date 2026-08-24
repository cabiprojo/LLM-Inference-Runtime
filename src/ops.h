#pragma once
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "io/tensor_io.h"

// normalizes each token's vector to mean 0 variance 1, then applies a learned scale and shift
// x - input tensor of shape (seq_len, n_embd)
// gamma - scale tensor of shape (n_embd)
// beta - shift tensor of shape (n_embd)
// out - output tensor of shape (seq_len, n_embd)
// seq_len - number of sequences (batch size)
// n_embd - embedding dimension
// eps - small constant for numerical stability
void layer_norm(const float* x, const float* gamma, const float* beta,
                 float* out, int seq_len, int n_embd, float eps);

// turns token ids into vectors by summing token and position embeddings
// token_ids: seq_len token ids
// wte: token embedding table, shape (vocab_size, n_embd)
// wpe: position embedding table, shape (n_ctx, n_embd)
// out: shape (seq_len, n_embd), out[i] = wte[token_ids[i]] + wpe[i]
void embed(const int* token_ids, const float* wte, const float* wpe,
           float* out, int seq_len, int n_embd);

// same as embed(), but for exactly one token at an arbitrary position --
// needed for KV-cache decode, where the new token isn't at position 0
void embed_one(int token_id, int position, const float* wte, const float* wpe,
                float* out, int n_embd);

// nonlinearity applied inside the mlp
// GPT-2's tanh-approximation gelu, applied elementwise
// n = total element count
void gelu(const float* x, float* out, int n);

// turns raw attention scores into per-row probabilities, blocking future positions
// in-place causal softmax over a (seq_len, seq_len) attention score matrix
// row i may only attend to columns 0..i (GPT-2 is autoregressive)
// each visible row is turned into a probability distribution via softmax
// columns j > i are zeroed since they should never contribute to attn_weights @ V
void causal_softmax(float* scores, int seq_len);

// picks the most likely next token during generation, not part of the forward pass
// index of the largest value in logits[0..vocab_size)
// greedy next-token pick
int argmax(const float* logits, int vocab_size);

// same as argmax(), but never picks a token that would recreate an n-gram
// (ngram_size consecutive tokens) already present earlier in ids. greedy
// decoding alone tends to loop into repeated phrases with no randomness to
// break out -- this blocks the exact repeats without adding any randomness,
// so it stays deterministic. ngram_size <= 0 disables this and behaves
// exactly like plain argmax()
int pick_next_token(const float* logits, int vocab_size,
                     const std::vector<int>& ids, int ngram_size);

// randomly samples a next token instead of always taking the top pick --
// this is what makes production LLMs (ChatGPT, Claude) give different
// answers to the same prompt on different runs, real greedy decoding never
// varies. restricts sampling to the top_k highest-probability tokens first
// (so it never picks something absurd from the long tail), then samples
// from those weighted by softmax(logits / temperature). higher temperature
// = more random, lower = closer to greedy. same n-gram repeat block as
// pick_next_token(). rng is passed in so the caller controls the seed --
// seed it once per generation, not per token, or draws become correlated
int sample_next_token(const float* logits, int vocab_size, const std::vector<int>& ids,
                       int ngram_size, float temperature, int top_k, std::mt19937& rng);

// A, B are the two input matrices, C is the output matrix
// M = number of rows in A, K = number of columns in A (and rows in B), N = number of columns in B
void matmul(const float* A, const float* B, float* C, int M, int K, int N);

// same result as matmul(), computed TILE x TILE block at a time for cache reuse
void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N, int TILE);

// same result as matmul_tiled(), inner loop vectorized 8-wide with AVX2
void matmul_simd(const float* A, const float* B, float* C, int M, int K, int N, int TILE);

// same result as matmul_simd(), with the M output rows split across
// num_threads threads (each running matmul_simd on its own disjoint chunk of
// rows) -- no locks needed since no two threads ever write the same row of C
void matmul_threaded(const float* A, const float* B, float* C, int M, int K, int N,
                      int TILE, int num_threads);

// out = x @ W^T + b
// x: (seq_len, in_features), W: (out_features, in_features), b: (out_features)
// out: (seq_len, out_features)
void linear(const float* x, const float* W, const float* b,
            float* out, int seq_len, int in_features, int out_features);

// same result as linear(), computed TILE x TILE block at a time for cache reuse
void linear_tiled(const float* x, const float* W, const float* b,
                   float* out, int seq_len, int in_features, int out_features, int TILE);

// same result as linear_tiled(), with the seq_len rows split across
// num_threads threads, same disjoint-output-rows approach as matmul_threaded
void linear_threaded(const float* x, const float* W, const float* b,
                      float* out, int seq_len, int in_features, int out_features,
                      int TILE, int num_threads);

// per-layer K/V history for KV-cache generation. K and V grow by one row
// (n_embd floats) each time a new token is processed, whether via prefill
// (many rows at once) or decode (one row at a time)
struct LayerKVCache {
    std::vector<float> K;  // (cached_len, n_embd), flat, row-major
    std::vector<float> V;  // (cached_len, n_embd), flat, row-major
    int cached_len = 0;
};

// multi-head causal self-attention for one block
// x: (seq_len, n_embd), layer-normed input (ln_1 output)
// c_attn_w: (3*n_embd, n_embd), c_attn_b: (3*n_embd), combined qkv projection
// c_proj_w: (n_embd, n_embd), c_proj_b: (n_embd), output projection
// out: (seq_len, n_embd)
// use_tiled: if true, internal linear() calls use linear_tiled(TILE=16) instead
// use_threaded: if true, internal linear() calls use linear_threaded() instead
//   (takes priority over use_tiled, since threaded already runs tiled internally)
// cache: if non-null, this call's K/V (for every position in x) are appended
//   to it -- used for prefill, populating the cache from the initial prompt
void attention(const float* x, const float* c_attn_w, const float* c_attn_b,
               const float* c_proj_w, const float* c_proj_b,
               float* out, int seq_len, int n_embd, int n_head, bool use_tiled = false,
               LayerKVCache* cache = nullptr, bool use_threaded = false);

// attention for exactly one new token, using and extending a KV-cache instead
// of recomputing K/V for the whole sequence. x_new: (1, n_embd). appends this
// token's K/V onto cache, then attends its Q against the full (old + new) cache.
// out: (1, n_embd)
void attention_decode(const float* x_new, const float* c_attn_w, const float* c_attn_b,
                       const float* c_proj_w, const float* c_proj_b,
                       float* out, int n_embd, int n_head, LayerKVCache& cache, bool use_tiled = false);

// one transformer block, x is updated in place to become this block's output
// ln_1 -> attention -> residual add -> ln_2 -> mlp -> residual add
// prefix selects this block's weights, e.g. "h.0."
// use_tiled: if true, every internal linear() call uses linear_tiled(TILE=16) instead
// use_threaded: if true, every internal linear() call uses linear_threaded() instead
void transformer_block(float* x, const std::unordered_map<std::string, Tensor>& weights,
                        const std::string& prefix, int seq_len, int n_embd, int n_head, float eps,
                        bool use_tiled = false, LayerKVCache* cache = nullptr, bool use_threaded = false);

// one transformer block for exactly one new token, using and extending cache.
// x_new: (1, n_embd), updated in place to become this block's output for the new token
void transformer_block_decode(float* x_new, const std::unordered_map<std::string, Tensor>& weights,
                               const std::string& prefix, int n_embd, int n_head, float eps,
                               LayerKVCache& cache, bool use_tiled = false);

// greedy autoregressive generation: repeatedly runs the full forward pass,
// picks the most likely next token, appends it, and repeats. no KV-cache yet,
// so every step recomputes the whole sequence from scratch -- correct, not fast.
// prompt_ids: starting tokens
// weights: full weight map, as loaded by load_tensors()
// n_embd, n_head, n_layer, eps: model config (768, 12, 12, 1e-5 for GPT-2 small)
// num_new_tokens: how many additional tokens to generate
// returns: prompt_ids followed by num_new_tokens generated ids
// no_repeat_ngram_size: 0 disables (pure greedy, matches HF exactly); a
//   positive value (e.g. 3) blocks repeated n-grams, see pick_next_token()
// temperature: 0 disables sampling entirely (pure greedy, deterministic,
//   matches HF exactly); > 0 samples instead (see sample_next_token()) --
//   different every run, seeded internally from a random device each call
// top_k: only used when temperature > 0, restricts sampling to the top_k
//   highest-probability tokens
std::vector<int> generate(const std::vector<int>& prompt_ids,
                           const std::unordered_map<std::string, Tensor>& weights,
                           int n_embd, int n_head, int n_layer, float eps,
                           int num_new_tokens, bool use_tiled = false, bool use_threaded = false,
                           int no_repeat_ngram_size = 0, float temperature = 0.0f, int top_k = 40);

// same generation, but KV-cached: the prompt is processed once (prefill,
// populating the cache), then each new token only computes its own Q/K/V and
// attends against the accumulated cache instead of recomputing the whole
// sequence from scratch every step. Must produce identical output to
// generate() for the same inputs -- this is a speed optimization only.
// use_threaded speeds up the prefill pass only (decode is a single row --
// nothing to split across threads)
// no_repeat_ngram_size: 0 disables (pure greedy, matches generate() exactly);
//   a positive value (e.g. 3) blocks repeated n-grams, see pick_next_token()
// temperature/top_k: same as generate(), see above
std::vector<int> generate_cached(const std::vector<int>& prompt_ids,
                                  const std::unordered_map<std::string, Tensor>& weights,
                                  int n_embd, int n_head, int n_layer, float eps,
                                  int num_new_tokens, bool use_tiled = false, bool use_threaded = false,
                                  int no_repeat_ngram_size = 0, float temperature = 0.0f, int top_k = 40);
