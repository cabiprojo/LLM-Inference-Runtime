"""
Loads real GPT-2 small weights from HuggingFace, runs a hand-written forward
pass (NOT model.forward()) so every intermediate step matches what you'll
implement in C++, and dumps weights + per-block activation checkpoints to
binary files the C++ side reads via src/io/tensor_io.h.

Run from the python/ directory:
    python3 dump_reference.py
"""
import json
import math
import struct
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MODEL_NAME = "gpt2"  # the 124M "small" checkpoint

# Fixed, hardcoded token ids -- BPE tokenization is out of scope for phase 1
# (correct forward pass only). These happen to be the GPT-2 tokenization of
# "The cat sat on the mat", but all that matters is C++ and Python agree.
INPUT_IDS = [464, 3797, 3332, 319, 262, 2603]


def save_tensors(path: Path, tensors: dict):
    """Binary format read by src/io/tensor_io.h:
        int32   num_tensors
        repeat num_tensors times:
          int32   name_len
          char    name[name_len]
          int32   ndim
          int32   shape[ndim]
          float32 data[product(shape)]
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(tensors)))
        for name, t in tensors.items():
            t = t.detach().contiguous().to(torch.float32)
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<i", t.dim()))
            if t.dim() > 0:
                f.write(struct.pack(f"<{t.dim()}i", *t.shape))
            f.write(t.numpy().tobytes())


def layer_norm(x, weight, bias, eps):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * weight + bias


def gelu_new(x):
    # GPT-2's tanh-approximation GELU (not the exact erf-based version).
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))


def attention(x, w, n_head):
    seq_len, n_embd = x.shape
    head_dim = n_embd // n_head

    qkv = x @ w["attn.c_attn.weight"].T + w["attn.c_attn.bias"]
    q, k, v = qkv.split(n_embd, dim=-1)

    def split_heads(t):
        return t.view(seq_len, n_head, head_dim).transpose(0, 1)  # (n_head, seq_len, head_dim)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)  # (n_head, seq_len, seq_len)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    out = weights @ v  # (n_head, seq_len, head_dim)
    out = out.transpose(0, 1).contiguous().view(seq_len, n_embd)
    out = out @ w["attn.c_proj.weight"].T + w["attn.c_proj.bias"]
    return out


def mlp(x, w):
    h = x @ w["mlp.c_fc.weight"].T + w["mlp.c_fc.bias"]
    h = gelu_new(h)
    h = h @ w["mlp.c_proj.weight"].T + w["mlp.c_proj.bias"]
    return h


def transformer_block(x, w, n_head, eps, checkpoints, i):
    ln1_out = layer_norm(x, w["ln_1.weight"], w["ln_1.bias"], eps)
    checkpoints[f"block_{i}_ln1"] = ln1_out

    attn_out = attention(ln1_out, w, n_head)
    checkpoints[f"block_{i}_attn"] = attn_out

    resid1 = x + attn_out
    checkpoints[f"block_{i}_resid1"] = resid1

    ln2_out = layer_norm(resid1, w["ln_2.weight"], w["ln_2.bias"], eps)
    checkpoints[f"block_{i}_ln2"] = ln2_out

    mlp_out = mlp(ln2_out, w)
    checkpoints[f"block_{i}_mlp"] = mlp_out

    resid2 = resid1 + mlp_out
    checkpoints[f"block_{i}_out"] = resid2

    return resid2


def main():
    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    cfg = model.config
    sd = model.state_dict()

    n_layer, n_head, n_embd = cfg.n_layer, cfg.n_head, cfg.n_embd
    eps = cfg.layer_norm_epsilon
    print(f"n_layer={n_layer} n_head={n_head} n_embd={n_embd}")

    # Build the weight dict, standardized to the nn.Linear convention.
    # HuggingFace's GPT2 uses Conv1D layers, which store weight as
    # (in_features, out_features) and compute y = x @ W + b. We transpose
    # everything here to (out_features, in_features) so C++ always computes
    # y = x @ W^T + b, uniformly, for every linear layer in the network.
    weights = {
        "wte": sd["transformer.wte.weight"],   # (vocab_size, n_embd)
        "wpe": sd["transformer.wpe.weight"],   # (n_ctx, n_embd)
        "ln_f.weight": sd["transformer.ln_f.weight"],
        "ln_f.bias": sd["transformer.ln_f.bias"],
    }

    layer_weights = []  # per-layer dict, used for the manual forward pass below
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        w = {
            "ln_1.weight": sd[p + "ln_1.weight"],
            "ln_1.bias": sd[p + "ln_1.bias"],
            "attn.c_attn.weight": sd[p + "attn.c_attn.weight"].T.contiguous(),
            "attn.c_attn.bias": sd[p + "attn.c_attn.bias"],
            "attn.c_proj.weight": sd[p + "attn.c_proj.weight"].T.contiguous(),
            "attn.c_proj.bias": sd[p + "attn.c_proj.bias"],
            "ln_2.weight": sd[p + "ln_2.weight"],
            "ln_2.bias": sd[p + "ln_2.bias"],
            "mlp.c_fc.weight": sd[p + "mlp.c_fc.weight"].T.contiguous(),
            "mlp.c_fc.bias": sd[p + "mlp.c_fc.bias"],
            "mlp.c_proj.weight": sd[p + "mlp.c_proj.weight"].T.contiguous(),
            "mlp.c_proj.bias": sd[p + "mlp.c_proj.bias"],
        }
        layer_weights.append(w)
        for name, t in w.items():
            weights[f"h.{i}.{name}"] = t

    # --- Manual forward pass, mirroring exactly what you'll write in C++ ---
    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long)
    seq_len = input_ids.shape[0]

    tok_emb = weights["wte"][input_ids]              # (seq_len, n_embd)
    pos_emb = weights["wpe"][torch.arange(seq_len)]  # (seq_len, n_embd)
    x = tok_emb + pos_emb

    checkpoints = {"embeddings": x}

    for i in range(n_layer):
        x = transformer_block(x, layer_weights[i], n_head, eps, checkpoints, i)

    x = layer_norm(x, weights["ln_f.weight"], weights["ln_f.bias"], eps)
    checkpoints["final_ln"] = x

    logits = x @ weights["wte"].T  # weight tying: reuse the token embedding matrix
    checkpoints["logits"] = logits

    # --- Sanity check: does our manual forward pass match HF's own forward? ---
    with torch.no_grad():
        ref_logits = model(input_ids.unsqueeze(0)).logits.squeeze(0)
    max_diff = (logits - ref_logits).abs().max().item()
    print(f"Manual forward vs HF forward, max abs logit diff: {max_diff:.2e}")
    assert max_diff < 1e-3, "Manual forward pass doesn't match HF's own output -- reference is untrustworthy!"
    print("Manual forward pass matches HuggingFace's forward pass. Reference is trustworthy.")

    # --- Save everything ---
    save_tensors(DATA_DIR / "gpt2_weights.bin", weights)

    activations = {"input_ids": torch.tensor(INPUT_IDS, dtype=torch.int32), **checkpoints}
    save_tensors(DATA_DIR / "reference_activations.bin", activations)

    manifest = {
        "model": MODEL_NAME,
        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
        "layer_norm_eps": eps,
        "input_ids": INPUT_IDS,
        "weight_shapes": {k: list(v.shape) for k, v in weights.items()},
        "activation_shapes": {k: list(v.shape) for k, v in checkpoints.items()},
    }
    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(weights)} weight tensors -> {DATA_DIR / 'gpt2_weights.bin'}")
    print(f"Wrote {len(activations)} activation tensors -> {DATA_DIR / 'reference_activations.bin'}")
    print(f"Wrote manifest -> {DATA_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
