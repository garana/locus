#!/usr/bin/env python3
"""Writes the goldens of tests/test_clip_text.cpp (i#372): transformers'
CLIPTokenizer and CLIPTextModel in F32.

tok: CLIPTokenizer (with ftfy, as FluxPipeline loads it) on the
prompts below, padded to 77 with truncation. Writes
tests/golden/clip_tokenizer.jsonl, one {"text", "ids"} per line.

    python3 -I scripts/make-clip-goldens.py tok \\
        --tokenizer DIR/tokenizer --out tests/golden

synth: a tiny CLIPTextModel (vocab 40, width 16, 4 heads of 4, FFN 24,
2 layers, 10 positions, legacy eos_token_id 2) whose weights come from
the LCG the test uses, in the test's tensor order, run on 10 fixed ids.
Writes tests/golden/clip_synth.f32: the last hidden state (10 x 16),
then the pooled output (16). Needs no download.

    python3 -I scripts/make-clip-goldens.py synth --out tests/golden

real: FLUX.1's CLIP-L (text_encoder, BF16 upcast to F32) on the
tokenizer's first four prompts. Writes, into --out (tests/golden/flux):
clip_pooled.f32 (4 x 768) and clip_pooled.json (the ids).

    python3 -I scripts/make-clip-goldens.py real --model DIR/text_encoder \\
        --tokenizer DIR/tokenizer --out tests/golden/flux

Tested with Python 3.12, torch 2.5.1 (CPU), transformers 4.46.3,
ftfy 6.3.1, on black-forest-labs/FLUX.1-Kontext-dev at 24e9ded.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

PROMPTS = [
    "A red fox standing in the snow",
    "a photo of an astronaut riding a horse on the moon, 4k, detailed",
    "",
    "Portrait of a young woman, oil painting, soft light, by Vermeer",
    "it's a dog's life, isn't it? They're here; we'll see. I'd go!",
    "   lots   of\tspace\nand\r\nnewlines   ",
    "1024x768 HDR!!! (masterpiece:1.2) [best quality] {sharp}",
    "café naïve résumé Ñandú Ærø",
    "Ελληνικά γράμματα and Русский текст",
    "emoji 🦊🔥 and symbols — “quotes” … ©®™ €100",
    "<|startoftext|>already special<|endoftext|> tokens",
    "UPPER lower MiXeD CaSe 3.14159 1,000,000",
    "ﬁne ﬂour, ＦＵＬＬ　ｗｉｄｔｈ！ and ￥500",
    "Tom &amp; Jerry &#38; &#x41;BC &quot;cartoon&quot;",
    "it’s ‘single’ and ʼmodifier quotes",
    "bell\x07 and \x1b[31mred\x1b[0m escapes",
    "ΟΔΟΣ ΣΟΦΟΣ and ΣΑ",
    "wow!<|endoftext|>after and a￣macron",
    ("a very long prompt " * 20).strip(),
]

# The synthetic model; must match tests/test_clip_text.cpp.
SYN = dict(vocab=40, width=16, heads=4, ffn=24, layers=2, context=10)


def synth_tensors():
    """(parameter name, shape), in LCG order."""
    s = SYN
    w, f = s["width"], s["ffn"]
    m = "text_model."
    out = [(m + "embeddings.token_embedding.weight", [s["vocab"], w]),
           (m + "embeddings.position_embedding.weight", [s["context"], w])]
    for n in range(s["layers"]):
        p = m + "encoder.layers.%d." % n
        out += [(p + "layer_norm1.weight", [w]), (p + "layer_norm1.bias", [w])]
        for x in ("q", "k", "v", "out"):
            out += [(p + "self_attn.%s_proj.weight" % x, [w, w]),
                    (p + "self_attn.%s_proj.bias" % x, [w])]
        out += [(p + "layer_norm2.weight", [w]), (p + "layer_norm2.bias", [w]),
                (p + "mlp.fc1.weight", [f, w]), (p + "mlp.fc1.bias", [f]),
                (p + "mlp.fc2.weight", [w, f]), (p + "mlp.fc2.bias", [w])]
    out += [(m + "final_layer_norm.weight", [w]),
            (m + "final_layer_norm.bias", [w])]
    return out


def tok(tok_dir, out_dir):
    t = CLIPTokenizer.from_pretrained(tok_dir)
    with open(os.path.join(out_dir, "clip_tokenizer.jsonl"), "w") as f:
        for p in PROMPTS:
            ids = t(p, padding="max_length", max_length=77,
                    truncation=True).input_ids
            f.write(json.dumps({"text": p, "ids": ids},
                               ensure_ascii=False) + "\n")


def synth(out_dir):
    s = SYN
    cfg = CLIPTextConfig(vocab_size=s["vocab"], hidden_size=s["width"],
                         intermediate_size=s["ffn"],
                         num_hidden_layers=s["layers"],
                         num_attention_heads=s["heads"],
                         max_position_embeddings=s["context"],
                         hidden_act="quick_gelu", layer_norm_eps=1e-5,
                         eos_token_id=2, attention_dropout=0.0)
    model = CLIPTextModel(cfg).eval()
    params = dict(model.named_parameters())
    lcg = 372
    for name, shape in synth_tensors():
        n = int(np.prod(shape))
        v = np.empty(n, dtype=np.float32)
        for i in range(n):
            lcg = (lcg * 1664525 + 1013904223) & 0xffffffff
            v[i] = (np.float32(lcg >> 8) / np.float32(16777216.0)
                    - np.float32(0.5)) * np.float32(0.5)
        params[name].data.copy_(torch.from_numpy(v.reshape(shape)))
    # The largest id (39) first appears at position 6: the pooled row.
    ids = [38, 5, 17, 3, 22, 9, 39, 39, 39, 39]
    o = model(input_ids=torch.tensor([ids]))
    with open(os.path.join(out_dir, "clip_synth.f32"), "wb") as f:
        f.write(o.last_hidden_state[0].numpy().astype("<f4").tobytes())
        f.write(o.pooler_output[0].numpy().astype("<f4").tobytes())


def real(model_dir, tok_dir, out_dir):
    t = CLIPTokenizer.from_pretrained(tok_dir)
    model = CLIPTextModel.from_pretrained(
        model_dir, torch_dtype=torch.float32).eval()
    cases = []
    with open(os.path.join(out_dir, "clip_pooled.f32"), "wb") as f:
        for p in PROMPTS[:4]:
            ids = t(p, padding="max_length", max_length=77,
                    truncation=True).input_ids
            pooled = model(input_ids=torch.tensor([ids])).pooler_output[0]
            f.write(pooled.numpy().astype("<f4").tobytes())
            cases.append({"text": p, "ids": ids})
    with open(os.path.join(out_dir, "clip_pooled.json"), "w") as f:
        json.dump({"revision": "24e9ded", "width": 768, "cases": cases}, f,
                  indent=1, ensure_ascii=False)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tok", "synth", "real"])
    ap.add_argument("--model")
    ap.add_argument("--tokenizer")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    if a.mode == "tok":
        tok(a.tokenizer, a.out)
    elif a.mode == "synth":
        synth(a.out)
    else:
        real(a.model, a.tokenizer, a.out)


if __name__ == "__main__":
    main()
