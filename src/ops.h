#pragma once
#include <string>
#include <unordered_map>

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

// A, B are the two input matrices, C is the output matrix
// M = number of rows in A, K = number of columns in A (and rows in B), N = number of columns in B
void matmul(const float* A, const float* B, float* C, int M, int K, int N);

// same result as matmul(), computed TILE x TILE block at a time for cache reuse
void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N, int TILE);

// same result as matmul_tiled(), inner loop vectorized 8-wide with AVX2
void matmul_simd(const float* A, const float* B, float* C, int M, int K, int N, int TILE);

// out = x @ W^T + b
// x: (seq_len, in_features), W: (out_features, in_features), b: (out_features)
// out: (seq_len, out_features)
void linear(const float* x, const float* W, const float* b,
            float* out, int seq_len, int in_features, int out_features);

// same result as linear(), computed TILE x TILE block at a time for cache reuse
void linear_tiled(const float* x, const float* W, const float* b,
                   float* out, int seq_len, int in_features, int out_features, int TILE);

// multi-head causal self-attention for one block
// x: (seq_len, n_embd), layer-normed input (ln_1 output)
// c_attn_w: (3*n_embd, n_embd), c_attn_b: (3*n_embd), combined qkv projection
// c_proj_w: (n_embd, n_embd), c_proj_b: (n_embd), output projection
// out: (seq_len, n_embd)
// use_tiled: if true, internal linear() calls use linear_tiled(TILE=16) instead
void attention(const float* x, const float* c_attn_w, const float* c_attn_b,
               const float* c_proj_w, const float* c_proj_b,
               float* out, int seq_len, int n_embd, int n_head, bool use_tiled = false);

// one transformer block, x is updated in place to become this block's output
// ln_1 -> attention -> residual add -> ln_2 -> mlp -> residual add
// prefix selects this block's weights, e.g. "h.0."
// use_tiled: if true, every internal linear() call uses linear_tiled(TILE=16) instead
void transformer_block(float* x, const std::unordered_map<std::string, Tensor>& weights,
                        const std::string& prefix, int seq_len, int n_embd, int n_head, float eps,
                        bool use_tiled = false);
