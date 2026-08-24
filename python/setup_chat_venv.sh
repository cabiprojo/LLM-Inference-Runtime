#!/usr/bin/env bash
# Sets up a small, fast venv for chat.py, on the native Linux filesystem
# (WSL home directory), not the Windows-mounted drive this project lives on.
#
# Why this exists: chat.py only ever uses transformers' tokenizer, never the
# actual model, so it doesn't need torch at all. But even just `import
# transformers` took ~43 seconds when the venv lived under /mnt/c/... -- WSL2's
# bridge to Windows drives is slow for the "open thousands of small files"
# access pattern that importing a large Python package involves. The same
# import takes ~1 second from a venv on native WSL filesystem instead.
#
# Run once: bash setup_chat_venv.sh
# Then run chat.py with: ~/chat_demo_venv/bin/python chat.py "..." --num-new 20

set -e

VENV_DIR="$HOME/chat_demo_venv"

if [ -d "$VENV_DIR" ]; then
    echo "$VENV_DIR already exists, skipping setup."
    exit 0
fi

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet transformers

echo "Done. Run chat.py with: $VENV_DIR/bin/python chat.py \"your prompt\" --num-new 20"
