# Transformer Inference Engine

A GPT-2 small (124M) inference engine built from scratch in C++ — no PyTorch, no ML framework, no external math library at runtime. The only dependency at inference time is a hand-rolled binary weight file and the C++ standard library.

PyTorch is used exactly once, offline, to export real trained GPT-2 weights into that binary format and to generate per-layer reference activations for correctness validation (`python/dump_reference.py`). It never runs at inference time.

## What this is

Three phases, each validated against the real model before moving to the next:

1. **Correct forward pass** — embeddings, LayerNorm, multi-head causal self-attention, MLP, the full 12-block residual stream, final projection to logits. Validated end-to-end against PyTorch's own output at every checkpoint, not just the final answer.
2. **Cache-aware tiling** — restructuring the matrix multiplications to reuse data while it's still in CPU cache, instead of re-fetching from RAM repeatedly.
3. **SIMD vectorization** — hand-written AVX2 intrinsics for the one function that needed them, after the compiler's own auto-vectorization was checked and found lacking.

## Results

| What | Naive | Optimized | Speedup |
|---|---|---|---|
| `matmul`, isolated (M=64,K=768,N=3072) | 921–1327 ms | 87 ms (tiled) → 25 ms (tiled + SIMD) | ~15x → ~37–53x |
| `linear`, one block (4 real weight shapes) | 308–448 ms | 161–195 ms (tiled) | ~1.9–2.3x |
| **Full 12-block forward pass, real engine** | **5601 ms** | **2459 ms** | **~2.28x** |

The last row is the one that matters most — it's not a projection from an isolated benchmark, it's the actual `gpt2` binary's full forward pass, timed directly, with both the naive and tiled paths verified correct against real PyTorch reference activations first.

## Architecture

```
token ids → embed (token + position embeddings)
          → 12x [ layer_norm → attention → +residual → layer_norm → mlp → +residual ]
          → layer_norm
          → logits (projection, weight-tied to the embedding matrix)
```

Every tensor is a flat `std::vector<float>` with manual row-major indexing (`data[row * cols + col]`) — no tensor abstraction, no framework. Every weight matrix is stored `(out_features, in_features)`, matching the `nn.Linear` convention, so every linear layer computes `y = x @ W^T + b` uniformly.

## Phase-by-phase notes

### Phase 1 — correctness

`python/dump_reference.py` hand-implements the GPT-2 forward pass in PyTorch (not `model.forward()`) so it doubles as both a checkpoint generator and a readable spec for the C++ side. It dumps activations after every sub-step of every block — 76 checkpoints total — so a mismatch anywhere can be traced to the exact function responsible, not just "somewhere in 12 blocks of math."

**The validation methodology caught something interesting**: after wiring up the full 12-block loop, blocks 2 through 11 initially failed a fixed absolute-tolerance check (`1e-3`), while block 0 and 1 passed. Rather than assume either "it's fine" or "it's broken," checking every block's error individually showed a plateau, not runaway growth — and checking the actual activation magnitudes (which grow into the thousands by block 2) showed the "failures" were relative errors around `2e-6`. Normal fp32 non-associativity (this engine's summation order vs. PyTorch's BLAS), not a bug. Fixed by switching to `numpy.allclose`-style relative tolerance (`atol + rtol * |reference|`).

### Phase 2 — cache-aware tiling

The naive `matmul`'s inner loop accesses `B[k*N+j]` with a stride of `N` — for the MLP's `768×3072` weight matrix (9.4 MB), that's far bigger than any cache level, so almost every access is a fresh trip to RAM. Restructuring into `TILE × TILE` blocks (processing one small, cache-resident chunk fully before moving to the next) fixed this without changing the answer — validated bit-exact against the naive version on random data.

`linear()` got the same treatment. Its access pattern (`x[i*in+k]`, `W[j*in+k]`) is already contiguous for both operands — no strided-access penalty — so tiling there only fixes the "working set too big for cache" problem, not a bandwidth-waste problem too. Smaller, but still real: ~1.9–2.3x on the actual per-block linear-layer cost.

**One honest finding**: `TILE=16` consistently beat `TILE=32` and `TILE=64` (confirmed across 7 repeated runs, not a one-off measurement), despite back-of-envelope L1-capacity math suggesting `32` should be the better choice. The likely explanation is cache associativity/conflict misses that a simple capacity calculation doesn't account for — but this project's environment (WSL2 on Windows) doesn't expose real hardware performance counters, and `perf`/`valgrind` weren't installable without `sudo` access that wasn't available. So the *what* (16 wins, reproducibly) is solid; the *why* (associativity) is a reasoned but unconfirmed theory. Worth being precise about that distinction rather than overclaiming.

### Phase 3 — SIMD

GCC's vectorization report (`-fopt-info-vec`) showed the compiler already auto-vectorizes `linear()`'s inner loop to full 8-wide AVX2 on its own — its contiguous access pattern is exactly what the auto-vectorizer wants. `matmul`'s inner loop, still constrained by the strided `B[k*N+j]` access even after tiling, only reached 4-wide SSE.

The fix wasn't to force-vectorize the strided version — it was to vectorize a *different* loop. For a fixed `i` and `k`, walking across `j` in `B[k*N+j]` **is** contiguous, so `matmul_simd` vectorizes 8 output columns at once instead of the reduction dimension, using `_mm256_loadu_ps`/`_mm256_set1_ps`/`_mm256_fmadd_ps`. Validated against `matmul` (differences within normal fp32 rounding), then benchmarked in the same run as the tiled-only version: **~8x faster on top of tiling**, ~139x combined versus the original naive `matmul`.

Deliberately *not* hand-vectorized: `linear()`. The compiler was already extracting the full 8-wide win there; hand intrinsics would just reproduce free compiler work while adding real code complexity and portability cost.

## Build & run

Requires g++ with C++17 and AVX2/FMA support (checked via `-march=native`).

```bash
# generate weights + reference activations (one time, requires python/venv with torch+transformers)
cd python && python3 dump_reference.py

# build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# correctness (validates every function against real PyTorch output)
./build/gpt2

# performance (isolated matmul/linear benchmarks, naive vs tiled vs SIMD)
./build/bench_matmul
```

## What's not here (yet)

INT8 quantization, KV-cache, multithreading/CUDA, and batching are natural next phases but weren't pursued here — the project's goal was depth on cache-aware tiling and SIMD specifically, not maximum feature coverage.
