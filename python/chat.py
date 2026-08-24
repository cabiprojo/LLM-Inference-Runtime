"""
Typeable demo for the C++ engine: encodes a text prompt with GPT-2's real
tokenizer, calls the C++ generate binary to do the actual inference and
generation, decodes the result back to text.

Tokenization/detokenization happens here, in Python -- the C++ side never
sees text, only token ids, and does zero ML-framework work at inference time.

Usage:
    python3 chat.py "Once upon a time" --num-new 20
    python3 chat.py "Once upon a time" --num-new 20 --temperature 0.8   # varies each run
    python3 chat.py "Once upon a time" --num-new 20 --temperature 0    # deterministic (default)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# skip the network "check for updates" call on every run -- the tokenizer
# files are already cached locally from dump_reference.py, no need to phone
# home just to load them. this is what was causing the slow startup.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# this venv is deliberately torch-free (chat.py only ever uses the
# tokenizer, never the actual model) -- silence transformers' notice about
# that, it's expected, not a problem
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import GPT2Tokenizer

BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
GENERATE_BIN = BUILD_DIR / "generate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="text prompt to continue")
    parser.add_argument("--num-new", type=int, default=20, help="number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                         help="0 = deterministic greedy (matches PyTorch exactly every time); "
                              "> 0 = random sampling, varies each run, higher = more random")
    args = parser.parse_args()

    if not GENERATE_BIN.exists():
        print(f"'generate' binary not found at {GENERATE_BIN} -- build the project first", file=sys.stderr)
        sys.exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt_ids = tokenizer.encode(args.prompt)

    cmd = [str(GENERATE_BIN), str(args.num_new), str(args.temperature)] + [str(i) for i in prompt_ids]
    result = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print("generate binary failed:", result.stderr, file=sys.stderr)
        sys.exit(1)

    output_ids = [int(x) for x in result.stdout.split()]
    text = tokenizer.decode(output_ids)
    print(text)


if __name__ == "__main__":
    main()
