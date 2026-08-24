#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "io/tensor_io.h"
#include "ops.h"

namespace {

// debug helper, prints a tensor's shape, not part of the model
void print_shape(const std::unordered_map<std::string, Tensor>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        std::cout << "  " << name << ": NOT FOUND\n";
        return;
    }
    std::cout << "  " << name << ": [";
    const auto& shape = it->second.shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    std::cout << "]\n";
}

}  // namespace

// measures how far a computed buffer is from the reference
// validation harness
// compares a computed buffer against a reference checkpoint and reports max abs diff
// anything above ~1e-3 usually means a real bug, not just float rounding
float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

// true if every element is within atol + rtol * |reference value| (numpy's allclose rule)
// absolute tolerance alone breaks once activation magnitudes grow into the
// hundreds/thousands in later blocks, since fp32 rounding grows proportionally
bool all_close(const float* computed, const float* reference, int n, float atol, float rtol) {
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(computed[i] - reference[i]);
        float tol = atol + rtol * std::fabs(reference[i]);
        if (diff > tol) return false;
    }
    return true;
}

// prints pass or fail by comparing computed output against a reference checkpoint
void check(const std::string& label, const float* computed, const float* reference, int n) {
    float diff = max_abs_diff(computed, reference, n);
    bool pass = all_close(computed, reference, n, 1e-3f, 1e-3f);
    std::cout << (pass ? "[PASS] " : "[FAIL] ") << label
               << " max abs diff = " << diff << "\n";
}

// loads real GPT-2 weights and reference activations, then tests pieces of the model against them
int main() {
    const std::string weights_path = "../data/gpt2_weights.bin";
    const std::string activations_path = "../data/reference_activations.bin";

    auto weights = load_tensors(weights_path);
    auto activations = load_tensors(activations_path);

    std::cout << "Loaded " << weights.size() << " weight tensors and "
              << activations.size() << " reference activation tensors.\n\n";

    const int seq_len = activations.at("input_ids").shape[0];
    const int n_embd = weights.at("wte").shape[1];

    // input_ids were stored as float32 (see tensor_io's format)
    // cast back to int
    std::vector<int> input_ids(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        input_ids[i] = static_cast<int>(activations.at("input_ids").data[i]);
    }

    // test 1: embed() against the "embeddings" checkpoint
    std::vector<float> computed_embeddings(seq_len * n_embd);
    embed(input_ids.data(), weights.at("wte").data.data(), weights.at("wpe").data.data(),
          computed_embeddings.data(), seq_len, n_embd);
    check("embed()", computed_embeddings.data(), activations.at("embeddings").data.data(),
          seq_len * n_embd);

    // test 2: layer_norm() against the "block_0_ln1" checkpoint
    std::vector<float> computed_ln1(seq_len * n_embd);
    layer_norm(computed_embeddings.data(),
               weights.at("h.0.ln_1.weight").data.data(),
               weights.at("h.0.ln_1.bias").data.data(),
               computed_ln1.data(), seq_len, n_embd, 1e-5f);
    check("layer_norm() [block 0, ln_1]", computed_ln1.data(),
          activations.at("block_0_ln1").data.data(), seq_len * n_embd);

    // test 3: matmul() against a hand-computed 2x2 example
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]] -> A@B = [[19,22],[43,50]]
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float expected[4] = {19, 22, 43, 50};
    float computed_matmul[4];
    matmul(A, B, computed_matmul, 2, 2, 2);
    check("matmul() [hand-computed 2x2]", computed_matmul, expected, 4);

    // test 4: linear() + gelu() chained through the real MLP, against "block_0_mlp"
    // uses the reference block_0_ln2 checkpoint as input, isolating this test from
    // block loop wiring (not written yet) so a failure here means linear/gelu, not wiring
    const int mlp_hidden = weights.at("h.0.mlp.c_fc.weight").shape[0];
    std::vector<float> fc_out(seq_len * mlp_hidden);
    linear(activations.at("block_0_ln2").data.data(),
           weights.at("h.0.mlp.c_fc.weight").data.data(),
           weights.at("h.0.mlp.c_fc.bias").data.data(),
           fc_out.data(), seq_len, n_embd, mlp_hidden);
    gelu(fc_out.data(), fc_out.data(), seq_len * mlp_hidden);

    std::vector<float> computed_mlp(seq_len * n_embd);
    linear(fc_out.data(),
           weights.at("h.0.mlp.c_proj.weight").data.data(),
           weights.at("h.0.mlp.c_proj.bias").data.data(),
           computed_mlp.data(), seq_len, mlp_hidden, n_embd);
    check("linear()+gelu() [block 0 mlp]", computed_mlp.data(),
          activations.at("block_0_mlp").data.data(), seq_len * n_embd);

    // test 5: attention() against the "block_0_attn" checkpoint
    // uses the reference block_0_ln1 checkpoint as input, isolating this test
    // from block loop wiring (not written yet)
    const int n_head = 12;
    std::vector<float> computed_attn(seq_len * n_embd);
    attention(activations.at("block_0_ln1").data.data(),
              weights.at("h.0.attn.c_attn.weight").data.data(),
              weights.at("h.0.attn.c_attn.bias").data.data(),
              weights.at("h.0.attn.c_proj.weight").data.data(),
              weights.at("h.0.attn.c_proj.bias").data.data(),
              computed_attn.data(), seq_len, n_embd, n_head);
    check("attention() [block 0]", computed_attn.data(),
          activations.at("block_0_attn").data.data(), seq_len * n_embd);

    // test 6: full forward pass, embed -> 12 blocks -> final ln -> logits
    // run three ways -- naive, tiled, threaded -- all checked against the
    // real reference checkpoints, proving each path is correct in the actual
    // model, not just on the synthetic random data bench_matmul used
    const int n_layer = 12;
    const float eps = 1e-5f;
    const int vocab_size = weights.at("wte").shape[0];
    std::vector<float> zero_bias(vocab_size, 0.0f);

    struct Mode { const char* tag; bool use_tiled; bool use_threaded; };
    Mode modes[] = {
        {"[naive] ", false, false},
        {"[tiled] ", true, false},
        {"[threaded] ", true, true},
    };

    for (const auto& mode : modes) {
        const std::string tag = mode.tag;

        std::vector<float> x(seq_len * n_embd);
        embed(input_ids.data(), weights.at("wte").data.data(), weights.at("wpe").data.data(),
              x.data(), seq_len, n_embd);

        for (int layer = 0; layer < n_layer; ++layer) {
            std::string prefix = "h." + std::to_string(layer) + ".";
            transformer_block(x.data(), weights, prefix, seq_len, n_embd, n_head, eps,
                               mode.use_tiled, nullptr, mode.use_threaded);
        }
        std::string ref_name = "block_" + std::to_string(n_layer - 1) + "_out";
        check(tag + "block_11_out", x.data(), activations.at(ref_name).data.data(), seq_len * n_embd);

        std::vector<float> final_ln_out(seq_len * n_embd);
        layer_norm(x.data(), weights.at("ln_f.weight").data.data(),
                   weights.at("ln_f.bias").data.data(),
                   final_ln_out.data(), seq_len, n_embd, eps);
        check(tag + "final_ln", final_ln_out.data(), activations.at("final_ln").data.data(), seq_len * n_embd);

        std::vector<float> computed_logits(seq_len * vocab_size);
        linear(final_ln_out.data(), weights.at("wte").data.data(), zero_bias.data(),
               computed_logits.data(), seq_len, n_embd, vocab_size);
        check(tag + "logits", computed_logits.data(), activations.at("logits").data.data(),
              seq_len * vocab_size);
    }

    // test 7: real timing of the actual engine, naive vs tiled vs threaded --
    // the real test sentence is only 6 tokens, too short to show a meaningful
    // timing difference, so this uses a longer synthetic input (real weights,
    // made-up token ids) purely to measure wall-clock speed, not correctness
    std::cout << "\ntiming: full 12-block forward pass, seq_len=64 (synthetic input, real weights)\n";
    const int bench_seq_len = 64;
    std::vector<int> bench_ids(bench_seq_len);
    for (int i = 0; i < bench_seq_len; ++i) bench_ids[i] = i % 1000;

    for (const auto& mode : modes) {
        std::vector<float> x(bench_seq_len * n_embd);
        embed(bench_ids.data(), weights.at("wte").data.data(), weights.at("wpe").data.data(),
              x.data(), bench_seq_len, n_embd);

        auto start = std::chrono::high_resolution_clock::now();
        for (int layer = 0; layer < n_layer; ++layer) {
            std::string prefix = "h." + std::to_string(layer) + ".";
            transformer_block(x.data(), weights, prefix, bench_seq_len, n_embd, n_head, eps,
                               mode.use_tiled, nullptr, mode.use_threaded);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double>(end - start).count() * 1000.0;

        std::cout << "  " << mode.tag << ms << " ms\n";
    }

    return 0;
}
