#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "ops.h"

namespace {

void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) x = dist(rng);
}

// times matmul(M,K,N) over several iterations, reports average time and GFLOPS
void benchmark(int M, int K, int N, int iterations) {
    std::mt19937 rng(42);
    std::vector<float> A(M * K), B(K * N), C(M * N);
    fill_random(A, rng);
    fill_random(B, rng);

    // warm up once, not timed -- avoids counting one-time costs like page faults
    matmul(A.data(), B.data(), C.data(), M, K, N);

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        matmul(A.data(), B.data(), C.data(), M, K, N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_seconds = std::chrono::duration<double>(end - start).count();
    double avg_seconds = total_seconds / iterations;
    double flops = 2.0 * M * K * N;  // one multiply and one add per inner loop step
    double gflops = flops / avg_seconds / 1e9;

    std::cout << "matmul(M=" << M << ", K=" << K << ", N=" << N << "): "
               << avg_seconds * 1000.0 << " ms/call, "
               << gflops << " GFLOPS\n";
}

}  // namespace

int main() {
    // small square case, easy sanity check on timing
    benchmark(256, 256, 256, 20);

    // shapes lifted from real GPT-2 small, seq_len is a plausible short prompt length
    const int seq_len = 64;
    benchmark(seq_len, 768, 768, 50);    // attention-projection-style shape
    benchmark(seq_len, 768, 3072, 50);   // mlp c_fc shape, the biggest matmul in the network

    return 0;
}
