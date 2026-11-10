#!/usr/bin/env python3
"""Writes the rerank goldens of tests/test_rerank.cpp (i#503): the
relevance score (the classifier's raw logit) of a few query-document
pairs, from one of:

- llama.cpp's llama-server --rerank (tested with build 12110, commit
  cd959bae9; run on the CPU) on a GGUF from
  scripts/convert-xlmr-reranker.py; Python's standard library only:

    python3 -I scripts/make-rerank-golden.py \\
        --model tests/models/bge-reranker-v2-m3-q8_0.gguf \\
        --llama-server /opt/homebrew/bin/llama-server \\
        --out tests/golden/bge_rerank.json

- transformers' XLMRobertaForSequenceClassification (4.46.3, torch
  2.5.1, float32) on the original weights, BAAI/bge-reranker-v2-m3 at
  commit 953dc6f (model.safetensors sha256 d9e3e081...edb5286):

    python3 -I scripts/make-rerank-golden.py --hf DIR \\
        --out tests/golden/bge_rerank_hf.json
"""
import argparse
import json
import subprocess
import time
import urllib.request

QUERIES = [
    ("what is panda?", [
        "hi",
        "The giant panda (Ailuropoda melanoleuca), sometimes called a "
        "panda bear or simply panda, is a bear species endemic to China.",
        "Pandas eat bamboo.",
        "Paris is the capital of France.",
    ]),
    ("¿Cuál es la capital de Francia?", [
        "Paris est la capitale de la France.",
        "Berlin ist die Hauptstadt von Deutschland.",
        "法国的首都是巴黎。",
    ]),
]


def llama_scores(server, model, port=18191):
    """@returns the scores per query, in document order, from a
    llama-server --rerank started here on the CPU."""
    p = subprocess.Popen(
        [server, "-m", model, "--rerank", "-ngl", "0", "--device", "none",
         "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            try:
                urllib.request.urlopen(url + "/health")
                break
            except OSError:
                time.sleep(0.25)
        out = []
        for q, docs in QUERIES:
            req = urllib.request.Request(
                url + "/v1/rerank",
                json.dumps({"query": q, "documents": docs}).encode(),
                {"Content-Type": "application/json"})
            res = json.load(urllib.request.urlopen(req))["results"]
            s = [0.0] * len(docs)
            for r in res:
                s[r["index"]] = r["relevance_score"]
            out.append(s)
        return out
    finally:
        p.terminate()
        p.wait()


def hf_scores(path):
    """@returns transformers' logits per query, in document order."""
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, use_safetensors=True, torch_dtype=torch.float32).eval()
    out = []
    with torch.no_grad():
        for q, docs in QUERIES:
            s = []
            for d in docs:
                x = tok(q, d, return_tensors="pt")
                s.append(float(model(**x).logits[0, 0]))
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--llama-server")
    ap.add_argument("--hf")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.hf:
        scores = hf_scores(a.hf)
    elif a.model and a.llama_server:
        scores = llama_scores(a.llama_server, a.model)
    else:
        ap.error("need --hf, or --model and --llama-server")
    data = {"queries": [{"query": q, "documents": d, "scores": s}
                        for (q, d), s in zip(QUERIES, scores)]}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
