#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ops.h"

namespace {

void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

using MatmulFn = std::function<void(const float*, const float*, float*, int, int, int)>;

// times a matmul-shaped function over several iterations, reports average time and GFLOPS
void benchmark(const std::string& label, const MatmulFn& fn, int M, int K, int N, int iterations) {
    std::mt19937 rng(42);
    std::vector<float> A(M * K), B(K * N), C(M * N);
    fill_random(A, rng);
    fill_random(B, rng);

    fn(A.data(), B.data(), C.data(), M, K, N);  // warm up, not timed

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        fn(A.data(), B.data(), C.data(), M, K, N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_seconds = std::chrono::duration<double>(end - start).count();
    double avg_seconds = total_seconds / iterations;
    double flops = 2.0 * M * K * N;  // one multiply and one add per inner loop step
    double gflops = flops / avg_seconds / 1e9;

    std::cout << label << " (M=" << M << ", K=" << K << ", N=" << N << "): "
               << avg_seconds * 1000.0 << " ms/call, "
               << gflops << " GFLOPS\n";
}

// confirms matmul_tiled produces the same result as matmul, just reordered
void check_correctness(int M, int K, int N, int tile) {
    std::mt19937 rng(7);
    std::vector<float> A(M * K), B(K * N), C_naive(M * N), C_tiled(M * N);
    fill_random(A, rng);
    fill_random(B, rng);

    matmul(A.data(), B.data(), C_naive.data(), M, K, N);
    matmul_tiled(A.data(), B.data(), C_tiled.data(), M, K, N, tile);

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_diff = std::max(max_diff, std::fabs(C_naive[i] - C_tiled[i]));
    }
    std::cout << (max_diff < 1e-3f ? "[PASS] " : "[FAIL] ")
               << "matmul_tiled matches matmul, tile=" << tile
               << ", max abs diff = " << max_diff << "\n";
}

// confirms matmul_simd produces the same result as matmul, just reordered + vectorized
void check_simd_correctness(int M, int K, int N, int tile) {
    std::mt19937 rng(13);
    std::vector<float> A(M * K), B(K * N), C_naive(M * N), C_simd(M * N);
    fill_random(A, rng);
    fill_random(B, rng);

    matmul(A.data(), B.data(), C_naive.data(), M, K, N);
    matmul_simd(A.data(), B.data(), C_simd.data(), M, K, N, tile);

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_diff = std::max(max_diff, std::fabs(C_naive[i] - C_simd[i]));
    }
    std::cout << (max_diff < 1e-3f ? "[PASS] " : "[FAIL] ")
               << "matmul_simd matches matmul, tile=" << tile
               << ", max abs diff = " << max_diff << "\n";
}

// confirms linear_tiled produces the same result as linear, just reordered
void check_linear_correctness(int seq_len, int in_features, int out_features, int tile) {
    std::mt19937 rng(11);
    std::vector<float> x(seq_len * in_features), W(out_features * in_features), b(out_features);
    std::vector<float> out_naive(seq_len * out_features), out_tiled(seq_len * out_features);
    fill_random(x, rng);
    fill_random(W, rng);
    fill_random(b, rng);

    linear(x.data(), W.data(), b.data(), out_naive.data(), seq_len, in_features, out_features);
    linear_tiled(x.data(), W.data(), b.data(), out_tiled.data(), seq_len, in_features, out_features, tile);

    float max_diff = 0.0f;
    for (int i = 0; i < seq_len * out_features; ++i) {
        max_diff = std::max(max_diff, std::fabs(out_naive[i] - out_tiled[i]));
    }
    std::cout << (max_diff < 1e-3f ? "[PASS] " : "[FAIL] ")
               << "linear_tiled matches linear, tile=" << tile
               << ", max abs diff = " << max_diff << "\n";
}

using LinearFn = std::function<void(const float*, const float*, const float*, float*, int, int, int)>;

// times a linear-shaped function, returns average seconds per call (no printing --
// used to sum up a projected full-model total across several shapes)
double time_linear(const LinearFn& fn, int seq_len, int in_features, int out_features, int iterations) {
    std::mt19937 rng(42);
    std::vector<float> x(seq_len * in_features), W(out_features * in_features), b(out_features);
    std::vector<float> out(seq_len * out_features);
    fill_random(x, rng);
    fill_random(W, rng);
    fill_random(b, rng);

    fn(x.data(), W.data(), b.data(), out.data(), seq_len, in_features, out_features);  // warm up

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        fn(x.data(), W.data(), b.data(), out.data(), seq_len, in_features, out_features);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_seconds = std::chrono::duration<double>(end - start).count();
    double avg_seconds = total_seconds / iterations;
    double gflops = (2.0 * seq_len * in_features * out_features) / avg_seconds / 1e9;

    std::cout << "  (in=" << in_features << ", out=" << out_features << "): "
               << avg_seconds * 1000.0 << " ms/call, " << gflops << " GFLOPS\n";
    return avg_seconds;
}

}  // namespace

int main() {
    // correctness first -- tiling must produce the identical answer, just faster
    check_correctness(64, 768, 3072, 32);
    check_correctness(64, 768, 3072, 64);
    std::cout << "\n";

    // baseline: the naive matmul numbers from before
    benchmark("matmul               ", matmul, 256, 256, 256, 20);
    const int seq_len = 64;
    benchmark("matmul               ", matmul, seq_len, 768, 768, 50);
    benchmark("matmul               ", matmul, seq_len, 768, 3072, 20);

    std::cout << "\n";

    // tiled comparison on the slow shape, a couple of tile sizes
    const int M = seq_len, K = 768, N = 3072;
    benchmark("matmul_tiled(TILE=16)", [](const float* A, const float* B, float* C, int M, int K, int N) {
        matmul_tiled(A, B, C, M, K, N, 16);
    }, M, K, N, 20);
    benchmark("matmul_tiled(TILE=32)", [](const float* A, const float* B, float* C, int M, int K, int N) {
        matmul_tiled(A, B, C, M, K, N, 32);
    }, M, K, N, 20);
    benchmark("matmul_tiled(TILE=64)", [](const float* A, const float* B, float* C, int M, int K, int N) {
        matmul_tiled(A, B, C, M, K, N, 64);
    }, M, K, N, 20);

    std::cout << "\n";

    // SIMD, on top of tiling -- must still match matmul exactly
    check_simd_correctness(64, 768, 3072, 16);
    benchmark("matmul_simd(TILE=16) ", [](const float* A, const float* B, float* C, int M, int K, int N) {
        matmul_simd(A, B, C, M, K, N, 16);
    }, M, K, N, 20);

    std::cout << "\n";

    // linear() is what actually does the heavy work in the real model --
    // this checks correctness, then projects a full 12-block forward pass
    // worth of linear() calls, naive vs tiled, using the real GPT-2 small shapes
    check_linear_correctness(seq_len, 768, 2304, 16);   // c_attn shape
    check_linear_correctness(seq_len, 3072, 768, 16);   // mlp c_proj shape
    std::cout << "\n";

    const LinearFn linear_naive = linear;
    const LinearFn linear_16 = [](const float* x, const float* W, const float* b, float* out,
                                    int seq_len, int in_features, int out_features) {
        linear_tiled(x, W, b, out, seq_len, in_features, out_features, 16);
    };

    // the four linear() calls inside one transformer_block, in order
    struct Shape { int in_features, out_features; const char* name; };
    Shape shapes[] = {
        {768, 2304, "attn.c_attn"},
        {768, 768,  "attn.c_proj"},
        {768, 3072, "mlp.c_fc"},
        {3072, 768, "mlp.c_proj"},
    };

    std::cout << "naive linear() per shape:\n";
    double naive_block_seconds = 0.0;
    for (const auto& s : shapes) {
        std::cout << "  " << s.name;
        naive_block_seconds += time_linear(linear_naive, seq_len, s.in_features, s.out_features, 20);
    }

    std::cout << "tiled linear_tiled(TILE=16) per shape:\n";
    double tiled_block_seconds = 0.0;
    for (const auto& s : shapes) {
        std::cout << "  " << s.name;
        tiled_block_seconds += time_linear(linear_16, seq_len, s.in_features, s.out_features, 20);
    }

    std::cout << "\none block: naive " << naive_block_seconds * 1000.0 << " ms, tiled "
               << tiled_block_seconds * 1000.0 << " ms\n";
    std::cout << "projected 12 blocks: naive " << naive_block_seconds * 12 * 1000.0
               << " ms, tiled " << tiled_block_seconds * 12 * 1000.0 << " ms\n";
    std::cout << "speedup: " << naive_block_seconds / tiled_block_seconds << "x\n";

    return 0;
}
