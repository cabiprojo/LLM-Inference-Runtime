#include <chrono>
#include <iostream>
#include <vector>

#include "io/tensor_io.h"
#include "ops.h"

// loads real GPT-2 weights, runs greedy generation from the real reference
// prompt, and checks the result against HuggingFace's own greedy generation
// (dump_reference.py's generated_ids) -- both use the identical deterministic
// argmax decoding, so they should match token for token
int main() {
    auto weights = load_tensors("../data/gpt2_weights.bin");
    auto activations = load_tensors("../data/reference_activations.bin");

    const int prompt_len = activations.at("input_ids").shape[0];
    std::vector<int> prompt(prompt_len);
    for (int i = 0; i < prompt_len; ++i) {
        prompt[i] = static_cast<int>(activations.at("input_ids").data[i]);
    }

    const auto& ref_tensor = activations.at("generated_ids");
    const int total_len = ref_tensor.shape[0];
    const int num_new_tokens = total_len - prompt_len;
    std::vector<int> expected(total_len);
    for (int i = 0; i < total_len; ++i) {
        expected[i] = static_cast<int>(ref_tensor.data[i]);
    }

    const int n_embd = weights.at("wte").shape[1];
    const int n_head = 12;
    const int n_layer = 12;
    const float eps = 1e-5f;

    std::cout << "prompt (" << prompt_len << " tokens): ";
    for (int id : prompt) std::cout << id << " ";
    std::cout << "\n";

    std::vector<int> generated = generate(prompt, weights, n_embd, n_head, n_layer, eps, num_new_tokens);

    std::cout << "generated: ";
    for (int id : generated) std::cout << id << " ";
    std::cout << "\n";

    std::cout << "expected:  ";
    for (int id : expected) std::cout << id << " ";
    std::cout << "\n";

    bool match = (generated == expected);
    std::cout << (match ? "[PASS] " : "[FAIL] ")
               << "generate() matches HuggingFace's greedy generation\n";

    // KV-cache must be a pure speed optimization -- same prompt, same
    // num_new_tokens, must produce the exact same ids as the non-cached version
    std::vector<int> cached = generate_cached(prompt, weights, n_embd, n_head, n_layer, eps, num_new_tokens);
    bool cache_match = (cached == generated);
    std::cout << (cache_match ? "[PASS] " : "[FAIL] ")
               << "generate_cached() matches generate() exactly\n";

    // the repetition guard is a separate feature from caching -- confirm
    // generate() and generate_cached() still agree with each other once
    // it's turned on, same discipline as every other optimization here
    std::vector<int> no_repeat = generate(prompt, weights, n_embd, n_head, n_layer, eps,
                                           num_new_tokens, false, false, /*no_repeat_ngram_size=*/3);
    std::vector<int> no_repeat_cached = generate_cached(prompt, weights, n_embd, n_head, n_layer, eps,
                                                         num_new_tokens, false, false, 3);
    bool no_repeat_match = (no_repeat == no_repeat_cached);
    std::cout << (no_repeat_match ? "[PASS] " : "[FAIL] ")
               << "generate_cached() with no_repeat_ngram_size=3 matches generate() with the same setting\n";

    // real timing comparison, on a longer generation where the difference
    // between "recompute everything" and "reuse the cache" actually shows up
    const int bench_new_tokens = 20;
    std::cout << "\ntiming: " << bench_new_tokens << " new tokens from the same "
              << prompt_len << "-token prompt\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    generate(prompt, weights, n_embd, n_head, n_layer, eps, bench_new_tokens);
    auto t1 = std::chrono::high_resolution_clock::now();
    generate_cached(prompt, weights, n_embd, n_head, n_layer, eps, bench_new_tokens, /*use_tiled=*/false);
    auto t2 = std::chrono::high_resolution_clock::now();
    generate_cached(prompt, weights, n_embd, n_head, n_layer, eps, bench_new_tokens, /*use_tiled=*/true);
    auto t3 = std::chrono::high_resolution_clock::now();

    double naive_ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0;
    double cached_ms = std::chrono::duration<double>(t2 - t1).count() * 1000.0;
    double cached_tiled_ms = std::chrono::duration<double>(t3 - t2).count() * 1000.0;
    std::cout << "  naive (recompute every step):     " << naive_ms << " ms\n";
    std::cout << "  cached, no tiling in decode:       " << cached_ms << " ms\n";
    std::cout << "  cached, tiled decode (chat.py uses this): " << cached_tiled_ms << " ms\n";
    std::cout << "  speedup, cached vs naive: " << naive_ms / cached_ms << "x\n";
    std::cout << "  speedup, cached+tiled vs naive: " << naive_ms / cached_tiled_ms << "x\n";
    std::cout << "  speedup, tiled decode vs untiled decode: " << cached_ms / cached_tiled_ms << "x\n";

    return (match && cache_match) ? 0 : 1;
}
