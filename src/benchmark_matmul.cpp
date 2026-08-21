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

    return 0;
}
