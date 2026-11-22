#!/usr/bin/env python3
"""Trains a linear classifier head on a locus embedding model's vectors
and writes it as a "linear-head" GGUF, for the embedding-head guard
(i#1160). SetFit stores its head only as a pickle (model_head.pkl,
joblib), which locus never loads, so this is the safe path: the same
logistic regression, in a GGUF.

1. Embed the labeled examples with locus-embed, the server's own
   embedding code (pooled as the GGUF says, L2-normalized), so the
   vectors are the ones the guard will score:

       build/locus-embed EMBED.gguf examples.jsonl embedded.jsonl

   Each line of examples.jsonl is {"text": ..., "label": "NAME"}, or
   with --multi-label {"text": ..., "labels": ["A", "B"]} (possibly
   empty).
2. Fit and write the head (offline venv; torch and gguf, no
   scikit-learn):

       .local-dev/vl-env/run-offline.sh scripts/text/train-linear-head.py \\
           --embeddings embedded.jsonl --embedding-model EMBED.gguf \\
           --out head.gguf

The fit is scikit-learn's LogisticRegression objective: multinomial
(softmax over the labels) by default, or one sigmoid per label with
--multi-label, with an L2 penalty 1/(2C) on the weights (not the
biases), C = 1 by default, minimized in float64 by L-BFGS. Labels
are sorted by name. With --holdout-every N, every Nth example is held
out of the fit and only scored; --predictions writes those examples'
vectors and probabilities (a test golden).

The GGUF (keys under "linear-head."):
- embedding_length (d), classifier.output_labels,
  classifier.problem_type (single_label_classification or
  multi_label_classification), temperature (1);
- the embedding model it was trained on: embedding.name (its
  general.name, else the file name), embedding.width,
  embedding.pooling (its pooling_type), embedding.file_size, and as
  information embedding.sha256, embedding.source.url and
  embedding.source.revision when the GGUF has them;
- tensors cls.output.weight (labels x d) and cls.output.bias, F32.
"""
import argparse
import hashlib
import json
import os

import gguf
import numpy as np
import torch

ARCH = "linear-head"


def embedding_identity(path):
    """@returns The keys that name the embedding model at path."""
    r = gguf.GGUFReader(path)

    def field(key):
        f = r.get_field(key)
        return None if f is None else f.contents()

    arch = field("general.architecture")
    width = field(f"{arch}.embedding_length")
    pooling = field(f"{arch}.pooling_type")
    if width is None or pooling is None:
        raise SystemExit(f"{path}: not an embedding model (no "
                         f"{arch}.embedding_length or pooling_type)")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {
        "name": field("general.name") or os.path.basename(path),
        "width": int(width),
        "pooling": int(pooling),
        "file_size": os.path.getsize(path),
        "sha256": h.hexdigest(),
        "source.url": field("general.source.url"),
        "source.revision": field("general.source.revision"),
    }


def fit(x, y, multi, c, iters):
    """@returns (W [K, d], b [K]) minimizing the summed log loss plus
    ||W||^2 / (2c), as scikit-learn's LogisticRegression does.
    y is class indices (multinomial) or a 0/1 [n, K] matrix."""
    n, d = x.shape
    k = y.shape[1] if multi else int(y.max()) + 1
    w = torch.zeros(k, d, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, tolerance_grad=1e-10,
                            tolerance_change=1e-12, history_size=50,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = x @ w.T + b
        if multi:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z, y, reduction="sum")
        else:
            loss = torch.nn.functional.cross_entropy(z, y, reduction="sum")
        loss = loss + (w * w).sum() / (2.0 * c)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def scores(x, w, b, multi):
    z = x @ w.T + b
    return torch.sigmoid(z) if multi else torch.softmax(z, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True,
                    help="locus-embed's output (JSONL with embedding)")
    ap.add_argument("--embedding-model", required=True,
                    help="the embedding GGUF the vectors came from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--multi-label", action="store_true",
                    help="one sigmoid per label; rows carry \"labels\"")
    ap.add_argument("--c", type=float, default=1.0,
                    help="inverse L2 strength, as scikit-learn's C")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--holdout-every", type=int, default=0)
    ap.add_argument("--predictions",
                    help="write the held-out examples and their scores")
    a = ap.parse_args()
    if not a.c > 0:
        raise SystemExit("--c must be positive")

    ident = embedding_identity(a.embedding_model)
    rows = []
    with open(a.embeddings, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no examples")
    for r in rows:
        if len(r["embedding"]) != ident["width"]:
            raise SystemExit("an embedding is not the model's width; embed "
                             "with the same GGUF as --embedding-model")
    if a.multi_label:
        labels = sorted({l for r in rows for l in r["labels"]})
    else:
        labels = sorted({r["label"] for r in rows})
    if len(labels) < 2 and not a.multi_label:
        raise SystemExit("a single-label head needs at least two labels")
    index = {l: i for i, l in enumerate(labels)}

    x = torch.tensor([r["embedding"] for r in rows], dtype=torch.float64)
    if a.multi_label:
        y = torch.zeros(len(rows), len(labels), dtype=torch.float64)
        for i, r in enumerate(rows):
            for l in r["labels"]:
                y[i, index[l]] = 1.0
    else:
        y = torch.tensor([index[r["label"]] for r in rows])
    held = [a.holdout_every > 0 and i % a.holdout_every ==
            a.holdout_every - 1 for i in range(len(rows))]
    train = torch.tensor([not h for h in held])
    w, b = fit(x[train], y[train], a.multi_label, a.c, a.iters)

    def accuracy(mask):
        if not mask.any():
            return None
        s = scores(x[mask], w, b, a.multi_label)
        if a.multi_label:
            return float(((s >= 0.5).double() == y[mask]).all(1)
                         .double().mean())
        return float((s.argmax(1) == y[mask]).double().mean())

    test = ~train
    print(f"{len(labels)} labels, {int(train.sum())} train / "
          f"{int(test.sum())} held out; accuracy train "
          f"{accuracy(train)}, held out {accuracy(test)}")

    wr = gguf.GGUFWriter(a.out, ARCH)
    wr.add_name(f"linear head on {ident['name']}")
    wr.add_uint32(f"{ARCH}.embedding_length", ident["width"])
    wr.add_classifier_output_labels(labels)
    wr.add_string(f"{ARCH}.classifier.problem_type",
                  "multi_label_classification" if a.multi_label
                  else "single_label_classification")
    wr.add_float32(f"{ARCH}.temperature", 1.0)
    wr.add_string(f"{ARCH}.embedding.name", ident["name"])
    wr.add_uint32(f"{ARCH}.embedding.width", ident["width"])
    wr.add_uint32(f"{ARCH}.embedding.pooling", ident["pooling"])
    wr.add_uint64(f"{ARCH}.embedding.file_size", ident["file_size"])
    wr.add_string(f"{ARCH}.embedding.sha256", ident["sha256"])
    for k in ("source.url", "source.revision"):
        if ident[k]:
            wr.add_string(f"{ARCH}.embedding.{k}", ident[k])
    wr.add_tensor("cls.output.weight", w.numpy().astype(np.float32))
    wr.add_tensor("cls.output.bias", b.numpy().astype(np.float32))
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()

    if a.predictions:
        # The head as written (float32), so a test can match it closely.
        w32 = torch.from_numpy(w.numpy().astype(np.float32)).double()
        b32 = torch.from_numpy(b.numpy().astype(np.float32)).double()
        s = scores(x[test], w32, b32, a.multi_label)
        out = {"labels": labels,
               "problem_type": "multi_label_classification"
               if a.multi_label else "single_label_classification",
               "items": [{"text": r["text"], "embedding": r["embedding"],
                          "scores": [float(v) for v in s[j]]}
                         for j, r in enumerate(
                             [r for r, h in zip(rows, held) if h])]}
        with open(a.predictions, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
