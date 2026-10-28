#!/usr/bin/env python3
"""Writes the goldens of tests/test_t5_encoder.cpp (i#308): outputs of
transformers' T5EncoderModel in F32.

synth: a tiny T5 v1.1 encoder (width 16, 4 heads of 4, FFN 24, 2
blocks, 8 relative buckets up to distance 16, vocab 32) whose weights
come from the LCG the test uses, in the test's tensor order, run on 20
fixed ids. Writes tests/golden/t5_synth.f32 (20 x 16 floats). Needs no
download.

    python3 -I scripts/make-t5-goldens.py synth --out tests/golden

real: PixArt-alpha/PixArt-XL-2-512x512's encoder (fp16 weights upcast
to F32, about 20 GB of RAM) on three inputs:

- "prompt": the ids PixArt's pipeline feeds T5 for the golden prompt
  of scripts/make-pixart-goldens.py (cleaned and lowercased);
- "empty": "", the negative prompt (just </s>);
- "long": a caption of 107 tokens, so the relative position buckets
  for distances 64 to 106 are used.

The encoder runs on the valid tokens only (no padding), which gives the
same rows as the padded, masked run. t5_encoder.f32 holds, for each
case in order, the output rows listed in t5_encoder.json ("rows"),
4096 floats each: every row for "prompt" and "empty", every 8th row
for "long".

    python3 -I scripts/make-t5-goldens.py real --model DIR/text_encoder \\
        --tokenizer DIR/tokenizer --out tests/golden

Tested with Python 3.12, torch 2.5.1 (CPU), transformers 4.46.3,
revision 50f7021.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import T5Config, T5EncoderModel, T5Tokenizer

PROMPT_IDS = [3, 9, 422, 1131, 8947, 30, 3, 9, 5726, 953, 6, 1835, 1379,
              659, 5, 1]

LONG = ("a watercolor painting of an old stone lighthouse on a rocky "
        "cliff above a stormy sea at dusk, with seagulls circling the "
        "lantern room, waves breaking white against the dark rocks, a "
        "narrow path winding up from a small fishing village with red "
        "roofs, warm light in the windows, heavy grey clouds parting to "
        "show a thin band of orange sky near the horizon, soft brush "
        "strokes, muted blues and greens, high detail")

# The synthetic model; must match tests/test_t5_encoder.cpp.
SYN = dict(vocab=32, width=16, heads=4, head_dim=4, ffn=24, blocks=2,
           buckets=8, max_distance=16, tokens=20)


def synth_tensors():
    """(GGUF name, torch parameter name, torch shape), in LCG order."""
    s = SYN
    w, inner = s["width"], s["heads"] * s["head_dim"]
    out = [("token_embd.weight", "shared.weight", [s["vocab"], w]),
           ("enc.blk.0.attn_rel_b.weight",
            "encoder.block.0.layer.0.SelfAttention."
            "relative_attention_bias.weight", [s["buckets"], s["heads"]])]
    for n in range(s["blocks"]):
        p = "encoder.block.%d.layer." % n
        b = "enc.blk.%d." % n
        out += [
            (b + "attn_norm.weight", p + "0.layer_norm.weight", [w]),
            (b + "attn_q.weight", p + "0.SelfAttention.q.weight", [inner, w]),
            (b + "attn_k.weight", p + "0.SelfAttention.k.weight", [inner, w]),
            (b + "attn_v.weight", p + "0.SelfAttention.v.weight", [inner, w]),
            (b + "attn_o.weight", p + "0.SelfAttention.o.weight", [w, inner]),
            (b + "ffn_norm.weight", p + "1.layer_norm.weight", [w]),
            (b + "ffn_gate.weight", p + "1.DenseReluDense.wi_0.weight",
             [s["ffn"], w]),
            (b + "ffn_up.weight", p + "1.DenseReluDense.wi_1.weight",
             [s["ffn"], w]),
            (b + "ffn_down.weight", p + "1.DenseReluDense.wo.weight",
             [w, s["ffn"]]),
        ]
    out.append(("enc.output_norm.weight", "encoder.final_layer_norm.weight",
                [w]))
    return out


def synth(out_dir):
    s = SYN
    cfg = T5Config(vocab_size=s["vocab"], d_model=s["width"],
                   d_kv=s["head_dim"], d_ff=s["ffn"], num_layers=s["blocks"],
                   num_heads=s["heads"],
                   relative_attention_num_buckets=s["buckets"],
                   relative_attention_max_distance=s["max_distance"],
                   feed_forward_proj="gated-gelu", layer_norm_epsilon=1e-6,
                   dropout_rate=0.0)
    model = T5EncoderModel(cfg).eval()
    params = dict(model.named_parameters())
    lcg = 146
    for _, pname, shape in synth_tensors():
        n = int(np.prod(shape))
        v = np.empty(n, dtype=np.float32)
        for i in range(n):
            lcg = (lcg * 1664525 + 1013904223) & 0xffffffff
            v[i] = (np.float32(lcg >> 8) / np.float32(16777216.0)
                    - np.float32(0.5)) * np.float32(0.5)
        params[pname].data.copy_(torch.from_numpy(v.reshape(shape)))
    ids = [(i * 7 + 3) % s["vocab"] for i in range(s["tokens"] - 1)] + [1]
    h = model(input_ids=torch.tensor([ids])).last_hidden_state[0]
    h.numpy().astype("<f4").tofile(os.path.join(out_dir, "t5_synth.f32"))
    print("synth", ids)


def real(model_dir, tok_dir, out_dir):
    tok = T5Tokenizer.from_pretrained(tok_dir, legacy=True)
    model = T5EncoderModel.from_pretrained(
        model_dir, variant="fp16", torch_dtype=torch.float32).eval()
    cases = [("prompt", PROMPT_IDS), ("empty", tok("").input_ids),
             ("long", tok(LONG).input_ids)]
    index = {"revision": "50f7021", "width": model.config.d_model,
             "cases": []}
    with open(os.path.join(out_dir, "t5_encoder.f32"), "wb") as f:
        for name, ids in cases:
            x = torch.tensor([ids], dtype=torch.long)
            h = model(input_ids=x).last_hidden_state[0].numpy()
            step = 8 if name == "long" else 1
            rows = list(range(0, len(ids), step))
            f.write(np.ascontiguousarray(h[rows].astype("<f4")).tobytes())
            index["cases"].append({
                "name": name, "ids": ids, "rows": rows,
                "rms": float(np.sqrt(np.mean(h.astype(np.float64) ** 2)))})
            print(name, len(ids), "tokens", flush=True)
    with open(os.path.join(out_dir, "t5_encoder.json"), "w") as f:
        json.dump(index, f, indent=1)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "real"])
    ap.add_argument("--model")
    ap.add_argument("--tokenizer")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    if a.mode == "synth":
        synth(a.out)
    else:
        real(a.model, a.tokenizer, a.out)


if __name__ == "__main__":
    main()
