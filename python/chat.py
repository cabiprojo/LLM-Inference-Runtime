"""
Typeable demo for the C++ engine: encodes a text prompt with GPT-2's real
tokenizer, calls the C++ generate binary to do the actual inference and
generation, decodes the result back to text.

Tokenization/detokenization happens here, in Python -- the C++ side never
sees text, only token ids, and does zero ML-framework work at inference time.

Usage:
    python3 chat.py "Once upon a time" --num-new 20
"""
import argparse
import subprocess
import sys
from pathlib import Path

from transformers import GPT2Tokenizer

BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
GENERATE_BIN = BUILD_DIR / "generate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="text prompt to continue")
    parser.add_argument("--num-new", type=int, default=20, help="number of new tokens to generate")
    args = parser.parse_args()

    if not GENERATE_BIN.exists():
        print(f"'generate' binary not found at {GENERATE_BIN} -- build the project first", file=sys.stderr)
        sys.exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt_ids = tokenizer.encode(args.prompt)

    cmd = [str(GENERATE_BIN), str(args.num_new)] + [str(i) for i in prompt_ids]
    result = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print("generate binary failed:", result.stderr, file=sys.stderr)
        sys.exit(1)

    output_ids = [int(x) for x in result.stdout.split()]
    text = tokenizer.decode(output_ids)
    print(text)


if __name__ == "__main__":
    main()
