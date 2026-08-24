#include <iostream>
#include <string>
#include <vector>

#include "io/tensor_io.h"
#include "ops.h"

// usage: ./generate <num_new_tokens> <temperature> <id1> <id2> ... <idN>
// temperature 0 = deterministic greedy (matches HF exactly); > 0 = sampling,
// varies every run, higher = more random. runs KV-cached generation from the
// given prompt token ids, prints the full resulting sequence (prompt +
// generated), space-separated, to stdout. no tokenization here -- that's the
// Python wrapper's job (python/chat.py)
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <num_new_tokens> <temperature> <id1> <id2> ...\n";
        return 1;
    }

    int num_new_tokens = std::stoi(argv[1]);
    float temperature = std::stof(argv[2]);
    std::vector<int> prompt;
    for (int i = 3; i < argc; ++i) {
        prompt.push_back(std::stoi(argv[i]));
    }

    auto weights = load_tensors("../data/gpt2_weights.bin");
    const int n_embd = weights.at("wte").shape[1];
    const int n_head = 12;
    const int n_layer = 12;
    const float eps = 1e-5f;

    // use_tiled=true here matters a lot: decode runs once per generated
    // token and dominates total generation time, so this is what actually
    // turns on the cache-aware linear() path for the demo, not just the
    // isolated benchmarks.
    // no_repeat_ngram_size=3: blocks exact repeated phrases either way,
    // sampling alone usually avoids loops but this is a cheap extra guard
    std::vector<int> result = generate_cached(prompt, weights, n_embd, n_head, n_layer, eps,
                                               num_new_tokens, /*use_tiled=*/true,
                                               /*use_threaded=*/false, /*no_repeat_ngram_size=*/3,
                                               temperature, /*top_k=*/40);

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << (i + 1 < result.size() ? " " : "\n");
    }

    return 0;
}
