#!/usr/bin/env python3
"""Writes a sequence-classifier golden for i#1157 from transformers
(4.56.1, torch 2.5.1, float32) on the safetensors weights: per text,
the token ids (special tokens added, no truncation), the windows the
classifier guard cuts (510 body tokens between [CLS] and [SEP], each
starting 64 tokens before the last ends), each window's logits and
scores (sigmoid or softmax as transformers' pipeline applies them),
and each label's highest score over the windows.

Tested on unitary/toxic-bert at commit 4d6c22e (model.safetensors
sha256 2c272885...4f6ad09; multi_label_classification):

    .local-dev/vl-env/run-offline.sh make-classifier-golden.py \\
        --hf SNAPSHOT_DIR --out toxic_bert_hf.json
"""
import argparse
import json

import torch

LONG = (
    "The committee met on Tuesday to review the quarterly budget, the "
    "hiring plan and the schedule for the new office. Each department "
    "presented its numbers, and the discussion moved on to travel, "
    "equipment and training costs for the coming year. ")

TEXTS = [
    "Hello, how are you today?",
    "Thank you for the detailed and helpful answer.",
    "You are a stupid idiot and everyone hates you.",
    "I will find you and hurt you.",
    "What a load of crap, this is garbage.",
    # Past 512 tokens: the insult sits in the last window only.
    LONG * 14 + "You are a stupid idiot and everyone hates you.",
]

WIDTH, OVERLAP = 510, 64


def windows(body, cls, sep):
    """The classifier guard's windows over body (ids without specials)."""
    out, stride, at = [], WIDTH - OVERLAP, 0
    while at < len(body):
        n = min(WIDTH, len(body) - at)
        out.append([cls] + body[at:at + n] + [sep])
        if at + n == len(body):
            break
        at += stride
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.hf)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.hf, use_safetensors=True, dtype=torch.float32).eval()
    cfg = model.config
    pt = cfg.problem_type
    sigmoid = pt == "multi_label_classification" or (
        pt is None and cfg.num_labels == 1)
    if pt == "regression":
        raise SystemExit("a regression model has no label scores")
    items = []
    with torch.no_grad():
        for t in TEXTS:
            ids = tok(t, truncation=False)["input_ids"]
            body = tok(t, add_special_tokens=False,
                       truncation=False)["input_ids"]
            assert ids == [tok.cls_token_id] + body + [tok.sep_token_id]
            ws = windows(body, tok.cls_token_id, tok.sep_token_id)
            logits, scores = [], []
            for w in ws:
                z = model(input_ids=torch.tensor([w])).logits[0]
                s = torch.sigmoid(z) if sigmoid else torch.softmax(z, -1)
                logits.append([float(x) for x in z])
                scores.append([float(x) for x in s])
            top = [max(s[i] for s in scores) for i in range(cfg.num_labels)]
            items.append({"text": t, "ids": ids, "windows": ws,
                          "logits": logits, "scores": scores, "top": top})
            print(len(ids), len(ws), [round(x, 4) for x in top], t[:40])
    data = {
        "model": cfg.name_or_path,
        "problem_type": pt,
        "labels": [cfg.id2label[i] for i in range(cfg.num_labels)],
        "window_body_tokens": WIDTH,
        "window_overlap": OVERLAP,
        "items": items,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
