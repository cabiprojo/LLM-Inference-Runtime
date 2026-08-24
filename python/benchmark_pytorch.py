"""
Times plain PyTorch's own generate() on CPU, for the same prompt and same
number of new tokens as the C++ engine's benchmark, as a reference point.

Usage:
    python3 benchmark_pytorch.py
"""
import time

import torch
from transformers import GPT2LMHeadModel

MODEL_NAME = "gpt2"
INPUT_IDS = [464, 3797, 3332, 319, 262, 2603]  # matches dump_reference.py
NUM_NEW_TOKENS = 20  # matches the C++ generate_demo benchmark


def main():
    torch.set_num_threads(1)  # fair comparison -- the C++ side is single-threaded too
    device = torch.device("cpu")

    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    input_ids = torch.tensor([INPUT_IDS], device=device)

    # warm up, not timed -- avoids counting one-time costs like lazy init
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=1, do_sample=False,
                        pad_token_id=model.config.eos_token_id)

    start = time.perf_counter()
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=NUM_NEW_TOKENS, do_sample=False,
                        pad_token_id=model.config.eos_token_id)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"PyTorch (CPU, 1 thread), {NUM_NEW_TOKENS} new tokens: {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()
