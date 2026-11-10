#!/usr/bin/env python3
"""Writes the embeddings of a few sentences with bge-small-en-v1.5 (CLS
pooling, L2-normalized) for tests/test_bert.cpp (i#502), from one of:

- llama.cpp's llama-embedding (tested with build 12110, commit
  cd959bae9; run on the CPU) on bge-small-en-v1.5-q8_0.gguf
  (ggml-org/bge-small-en-v1.5-Q8_0-GGUF, commit f2068ed), Python's
  standard library only:

    python3 -I scripts/make-bert-embd-golden.py \\
        --model tests/models/bge-small-en-v1.5-q8_0.gguf \\
        --llama-embedding /opt/homebrew/bin/llama-embedding \\
        --out tests/golden/bge_small_embd.json

- transformers' BertModel (4.46.3, torch 2.5.1, float32) on the
  original weights, BAAI/bge-small-en-v1.5 at commit 5c38ec7
  (model.safetensors sha256 3c9f3166...e21ad), the reference both
  quantized runs are measured against:

    python3 -I scripts/make-bert-embd-golden.py \\
        --hf DIR --out tests/golden/bge_small_embd_hf.json
"""
import argparse
import json
import os
import subprocess
import tempfile

# One line each: llama-embedding splits its prompt at newlines.
TEXTS = [
    "Hello, World!",
    "The capital of France is Paris.",
    "what is the capital of china?",
    "Beijing is the capital of the People's Republic of China.",
    "A small red apple on a wooden table, soft morning light.",
    "Café naïve résumé, 東京 and Привет.",
    "tokenization of subwords: walking, walked, walks",
]


def embedding(tool, model, text):
    """@returns llama-embedding's vector for text, on the CPU."""
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(text.encode("utf-8"))
        path = f.name
    try:
        out = subprocess.run(
            [tool, "-m", model, "-f", path, "--embd-output-format", "json",
             "-ngl", "0", "--device", "none"],
            check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    return json.loads(out)["data"][0]["embedding"]


def hf_embeddings(path):
    """@returns transformers' CLS vectors for TEXTS, L2-normalized, from
    the safetensors weights in path (no pickle)."""
    import torch
    from transformers import AutoTokenizer, BertModel

    tok = AutoTokenizer.from_pretrained(path)
    model = BertModel.from_pretrained(path, use_safetensors=True,
                                      torch_dtype=torch.float32).eval()
    out = []
    with torch.no_grad():
        for t in TEXTS:
            h = model(**tok(t, return_tensors="pt")).last_hidden_state
            v = torch.nn.functional.normalize(h[0, 0], dim=0)
            out.append([float(x) for x in v])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--llama-embedding")
    ap.add_argument("--hf")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.hf:
        vectors = hf_embeddings(a.hf)
    elif a.model and a.llama_embedding:
        vectors = [embedding(a.llama_embedding, a.model, t) for t in TEXTS]
    else:
        ap.error("need --hf, or --model and --llama-embedding")
    data = {"texts": TEXTS, "embeddings": vectors}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
