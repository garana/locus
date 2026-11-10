#!/usr/bin/env python3
"""Writes a float64 reference of a BERT GGUF's embeddings (i#502): the
GGUF's own weights, dequantized by the gguf package, through the BERT
forward pass in torch float64, with no rounding of the activations.

It is the reference for a model whose original weights are not on
hand as safetensors (bge-m3): it measures how far locus and llama.cpp
each are from exact arithmetic on the same quantized weights. The ids
come from llama-tokenize (special tokens added, special-token text not
parsed), which tests/test_wpm_tokenizer.cpp and test_t5_tokenizer.cpp
check locus's tokenizers against.

For a reranker (pooling_type 4, i#503) --texts is a rerank golden
(scripts/make-rerank-golden.py) and the output holds its queries with
the float64 scores.

Needs gguf 0.19.0 and torch 2.5.1 (the golden venv) and llama.cpp's
llama-tokenize (build 12110, commit cd959bae9):

    python3 -I scripts/make-bert-q8-ref.py \\
        --model tests/models/bge-m3-q8_0.gguf \\
        --llama-tokenize /opt/homebrew/bin/llama-tokenize \\
        --texts tests/golden/bge_m3_embd.json \\
        --out tests/golden/bge_m3_embd_q8ref.json
"""
import argparse
import json
import os
import subprocess
import tempfile

import gguf
import numpy as np
import torch


def ids_of(tokenize, model, text):
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(text.encode("utf-8"))
        path = f.name
    try:
        out = subprocess.run(
            [tokenize, "-m", model, "-f", path, "--ids", "--log-disable",
             "--no-parse-special"],
            check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    return json.loads(out.strip().splitlines()[-1])


def load(path):
    """@returns (hparams, {name: float64 tensor}) with torch's [out, in]
    layout (GGUF's ne reversed)."""
    r = gguf.GGUFReader(path)
    hp = {k: r.get_field(k).contents() for k in r.fields
          if k.startswith("bert.")}
    w = {}
    for t in r.tensors:
        a = gguf.quants.dequantize(t.data, t.tensor_type)
        shape = [int(x) for x in reversed(t.shape)]
        w[t.name] = torch.from_numpy(
            np.asarray(a, dtype=np.float64).reshape(shape))
    return hp, w


def forward(hp, w, ids):
    eps = float(hp["bert.attention.layer_norm_epsilon"])
    heads = int(hp["bert.attention.head_count"])
    n = len(ids)
    ln = torch.nn.functional.layer_norm

    def norm(x, p):
        return ln(x, x.shape[-1:], w[p + ".weight"], w[p + ".bias"], eps)

    def lin(x, p):
        return x @ w[p + ".weight"].T + w[p + ".bias"]

    types = w["token_types.weight"].reshape(-1, w["token_embd.weight"]
                                            .shape[1])
    x = (w["token_embd.weight"][ids] + w["position_embd.weight"][:n] +
         types[0])
    x = norm(x, "token_embd_norm")
    d = x.shape[1]
    hd = d // heads
    for i in range(int(hp["bert.block_count"])):
        p = f"blk.{i}."
        q = lin(x, p + "attn_q").view(n, heads, hd).transpose(0, 1)
        k = lin(x, p + "attn_k").view(n, heads, hd).transpose(0, 1)
        v = lin(x, p + "attn_v").view(n, heads, hd).transpose(0, 1)
        s = torch.softmax(q @ k.transpose(1, 2) / hd ** 0.5, dim=-1)
        a = (s @ v).transpose(0, 1).reshape(n, d)
        x = norm(x + lin(a, p + "attn_output"), p + "attn_output_norm")
        f = lin(x, p + "ffn_up")
        f = 0.5 * f * (1.0 + torch.erf(f / 2.0 ** 0.5))
        x = norm(x + lin(f, p + "ffn_down"), p + "layer_output_norm")
    pooling = int(hp["bert.pooling_type"])
    if pooling == 4:
        # Rank: the CLS state through tanh(cls) and cls.output.
        h = x[0]
        if "cls.weight" in w:
            h = torch.tanh(lin(h, "cls"))
        return lin(h, "cls.output")
    v = {1: x.mean(dim=0), 2: x[0], 3: x[-1]}[pooling]
    return torch.nn.functional.normalize(v, dim=0)


def rerank(a, hp, w):
    """Scores the pairs of a rerank golden: <s> q </s></s> d </s>, the
    XLM-R layout llama-server builds, from llama-tokenize's ids of q and
    d (whose own <s> and </s> are dropped)."""
    with open(a.texts, encoding="utf-8") as f:
        data = json.load(f)
    for q in data["queries"]:
        qi = ids_of(a.llama_tokenize, a.model, q["query"])[1:-1]
        scores = []
        for d in q["documents"]:
            di = ids_of(a.llama_tokenize, a.model, d)[1:-1]
            ids = [0] + qi + [2, 2] + di + [2]
            scores.append(float(forward(hp, w, ids)[0]))
        q["scores"] = scores
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-tokenize", required=True)
    ap.add_argument("--texts", required=True,
                    help="a golden JSON whose texts (or queries) to use")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hp, w = load(a.model)
    if int(hp["bert.pooling_type"]) == 4:
        with torch.no_grad():
            data = rerank(a, hp, w)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.write("\n")
        return
    with open(a.texts, encoding="utf-8") as f:
        texts = json.load(f)["texts"]
    vectors = []
    with torch.no_grad():
        for t in texts:
            ids = ids_of(a.llama_tokenize, a.model, t)
            vectors.append([float(x) for x in forward(hp, w, ids)])
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "embeddings": vectors}, f,
                  ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
