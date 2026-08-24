# Transformer Inference Engine

I built a GPT-2 small (124M) inference engine from scratch in C++. No PyTorch, no ML framework, nothing at runtime except the C++ standard library and a binary weight file I export ahead of time.

PyTorch only shows up once, offline, in `python/dump_reference.py`. It loads the real trained GPT-2 weights, dumps them into my own binary format, and generates per-layer reference activations so I can check my C++ output against the real model. It never runs during actual inference.

## What's in here

Four stages, each one checked against the real model before moving on to the next:

1. **Correct forward pass.** Embeddings, LayerNorm, multi-head causal self-attention, MLP, the full 12-block residual stream, final projection to logits. Validated against PyTorch's own output at every checkpoint, not just the final answer.
2. **Cache-aware tiling.** Restructured the matrix multiplications so they reuse data while it's still sitting in CPU cache, instead of hitting RAM over and over.
3. **SIMD vectorization.** Hand-wrote AVX2 intrinsics for the one function that actually needed them, after checking what the compiler already handled on its own.
4. **KV-cache and a real generation loop.** Added greedy decoding so the model can actually generate new text, then cached the attention K/V vectors so each new token doesn't require recomputing the whole sequence from scratch.

## Results

| What | Naive | Optimized | Speedup |
|---|---|---|---|
| `matmul`, isolated (M=64, K=768, N=3072) | 921-1327 ms | 87 ms tiled, 25 ms tiled+SIMD | ~15x, then ~37-53x |
| `linear`, one block (real weight shapes) | 308-448 ms | 161-195 ms tiled | ~1.9-2.3x |
| Full 12-block forward pass, real engine | 5601 ms | 2459 ms | ~2.28x |
| Generation, 20 new tokens | 20663 ms | 2246 ms with KV-cache | ~9.2x |

The forward pass row and the generation row are the ones that actually matter. They're not projections pulled from some isolated benchmark, they're the real binary, timed directly, with every path checked for correctness before I trusted the speed numbers.

**Honest comparison against plain PyTorch:** running the same 20-token generation through PyTorch on CPU (`python/benchmark_pytorch.py`, single-threaded, same prompt) takes about 751 ms, versus 2246 ms for my KV-cached engine. PyTorch is roughly 3x faster overall. That's expected, not a failure. PyTorch's CPU backend runs on Intel's oneDNN/MKL-DNN, production BLAS kernels with years of expert tuning (register blocking, prefetching, hand-tuned assembly) well beyond one pass at tiling and one hand-written SIMD loop, plus a memory allocator that avoids the repeated buffer allocations my engine does on every call. The point of this project was understanding and applying the techniques (cache locality, vectorization, KV-cache) from scratch, each one validated and each one measurably faster than the version before it, not beating a decade of production engineering.

## Architecture

```
token ids → embed (token + position embeddings)
          → 12x [ layer_norm → attention → +residual → layer_norm → mlp → +residual ]
          → layer_norm
          → logits (projection, weight-tied to the embedding matrix)
```

Every tensor is a flat `std::vector<float>` with manual row-major indexing (`data[row * cols + col]`). No tensor abstraction, no framework. Every weight matrix is stored `(out_features, in_features)`, matching the `nn.Linear` convention, so every linear layer computes `y = x @ W^T + b` the same way.

## How it's built, stage by stage

### Stage 1: getting it correct

I wrote `dump_reference.py` to hand-implement the GPT-2 forward pass in PyTorch instead of just calling `model.forward()`, so it doubles as both a checkpoint generator and a readable spec for what the C++ side needs to do. It dumps activations after every sub-step of every block, 76 checkpoints total, so if something breaks I can trace it to the exact function responsible instead of guessing across 12 blocks of math.

One thing that actually threw me off: after wiring up the full 12-block loop, blocks 2 through 11 started failing a fixed tolerance check while blocks 0 and 1 passed fine. My first instinct was that something was broken. Instead of assuming that, I checked every block's error individually and saw a plateau, not runaway growth. Then I checked the actual size of the activation values, which grow into the thousands by block 2, and realized the "failures" were relative errors around `2e-6`. That's just normal fp32 rounding drift (my summation order doesn't match PyTorch's BLAS), not a bug. Fixed it by switching to a relative tolerance check instead of a fixed absolute one.

### Stage 2: cache-aware tiling

The naive `matmul`'s inner loop reads `B[k*N+j]` with a stride of `N`. For the MLP's `768x3072` weight matrix, that's 9.4 MB, way bigger than any cache level, so almost every access is a fresh trip out to RAM. I fixed this by restructuring the loop into `TILE x TILE` blocks, processing one small chunk fully while it's still cache-resident before moving to the next one. Validated it bit-exact against the naive version on random data first.

`linear()` got the same treatment. Its access pattern is already contiguous for both operands, so tiling there only fixes the "working set too big for cache" problem, not a wasted-bandwidth problem too. Smaller win, but still real: about 1.9-2.3x on the actual per-block linear layer cost.

One honest thing worth calling out: `TILE=16` consistently beat `TILE=32` and `TILE=64`, confirmed across 7 repeated runs so it wasn't a fluke. My back-of-envelope L1 capacity math said 32 should have been the better choice. My best guess is cache associativity or conflict misses that a simple capacity calculation doesn't account for, but I couldn't actually confirm that. This environment (WSL2 on Windows) doesn't expose real hardware performance counters, and I couldn't install `perf` or `valgrind` without sudo access I didn't have. So the *what* is solid (16 wins, repeatedly), the *why* is a reasoned guess I never got to verify.

### Stage 3: SIMD

I checked GCC's vectorization report before writing any intrinsics by hand, and it turned out the compiler already auto-vectorizes `linear()`'s inner loop to full 8-wide AVX2 on its own, because its access pattern is exactly what the auto-vectorizer wants. `matmul`'s inner loop, still limited by that strided `B` access even after tiling, only got 4-wide SSE.

The fix wasn't to force-vectorize the strided version. It was to vectorize a different loop entirely. For a fixed `i` and `k`, walking across `j` in `B[k*N+j]` is actually contiguous, so `matmul_simd` vectorizes 8 output columns at once instead of the reduction dimension, using `_mm256_loadu_ps`, `_mm256_set1_ps`, and `_mm256_fmadd_ps`. Checked it against `matmul` (differences stayed within normal fp32 rounding), then benchmarked it in the same run as the tiled-only version: about 8x faster on top of tiling, roughly 139x combined versus the original naive `matmul`.

I deliberately didn't hand-vectorize `linear()`. The compiler was already getting the full 8-wide win there, so writing intrinsics by hand would have just reproduced free compiler work while adding real code complexity and less portability.

### Stage 4: generation loop and KV-cache

Up to this point the engine could only score a fixed sentence, it couldn't actually generate anything new. I added `generate()`, a plain greedy decoding loop: run the forward pass, take the logits at the last position, pick the most likely next token, append it, repeat. Validated this against HuggingFace's own `model.generate()` with greedy decoding from the same prompt, and it matched exactly, token for token, across 10 generated steps. That's a much stronger correctness signal than any single-step test, since it means every forward pass has to be correct across 10 compounding steps built on top of each other.

The obvious problem with that loop: every new token reruns the entire forward pass over the whole sequence so far, from scratch. So I added a KV-cache. Each layer keeps a running buffer of every previous token's K and V vectors. The prompt gets processed once (prefill), which populates the cache, then every new token after that only computes its own Q, K, and V, appends its K/V onto the cache, and attends against everything accumulated so far instead of recomputing the whole history. No causal mask is even needed in that step, since by definition every cached position already happened before the new one.

Before trusting the speedup I checked that `generate_cached()` produces byte-for-byte identical output to the non-cached version, since caching should only ever change speed, never the answer. It did. Then I benchmarked both on 20 new tokens from the same prompt: 20663 ms naive versus 2246 ms cached, about 9.2x faster. The gap grows the longer the generation runs, since it's removing recomputation that scales with sequence length.

I also put together a small end-to-end demo. `python/chat.py` tokenizes a text prompt with the real GPT-2 tokenizer, calls the C++ `generate` binary (which uses the cached path by default), and decodes the result back into text. Typed "Once upon a time" in and got "Once upon a time, the world was a place of great beauty and great danger" back out.

## Build and run

Requires g++ with C++17 and AVX2/FMA support (checked via `-march=native`).

```bash
# generate weights + reference activations (one time, requires python/venv with torch+transformers)
cd python && python3 dump_reference.py

# build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# correctness (validates every function against real PyTorch output, plus generation vs HF ground truth)
./build/gpt2
./build/generate_demo

# performance (isolated matmul/linear benchmarks, naive vs tiled vs SIMD)
./build/bench_matmul

# the actual demo: type a prompt, get generated text back
cd python && python3 chat.py "Once upon a time" --num-new 20

# how does it compare to plain PyTorch on CPU?
cd python && python3 benchmark_pytorch.py
```

## What's not here yet

INT8 quantization and multithreading are natural next steps but I didn't get to them here. This project ended up being about cache-aware tiling, SIMD, and getting a real generation loop with KV-cache working end to end, not maximum feature coverage.
