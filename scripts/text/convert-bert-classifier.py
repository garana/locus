#!/usr/bin/env python3
"""Converts a BERT sequence classifier (transformers'
BertForSequenceClassification, such as unitary/toxic-bert) to a GGUF
of arch "bert" in llama.cpp's layout, for locus's encoder classifiers
(i#1157). Needs only safetensors, numpy and gguf (no transformers),
so it runs in the offline venv.

The layout is what llama.cpp's converter writes for BERT
(conversion/bert.py at b1b770d90), which locus's BertEncoder already
reads, plus what a classifier needs:
- keys: context_length, embedding_length, feed_forward_length,
  block_count, attention.head_count, attention.layer_norm_epsilon,
  attention.causal (false), pooling_type 4 (rank: the classifier on
  the CLS state), classifier.output_labels (by integer id) and
  classifier.problem_type (config.json's, else what transformers'
  text-classification pipeline applies without one: softmax,
  single_label_classification, over several labels and a sigmoid,
  multi_label_classification, on one);
- tensors: token_embd, position_embd, token_types (every row),
  token_embd_norm; per block attn_q, attn_k, attn_v, attn_output,
  attn_output_norm, ffn_up, ffn_down, layer_output_norm; cls.output
  (the classifier);
- cls: the pooler's dense layer, which BertForSequenceClassification
  runs (with tanh) on the CLS state before the classifier. llama.cpp's
  converter drops it, so its GGUF of such a model scores the raw CLS
  state; locus's BertEncoder::rank applies cls and tanh when present,
  as transformers does.

The tokenizer is WordPiece from vocab.txt, written as llama.cpp does
("bert" keys): a word-start piece as "\\u2581piece", a continuation
("##piece") as "piece", the special tokens ([PAD], [UNK], [CLS],
[SEP], [MASK]) unchanged as control tokens, and their ids. Only an
uncased vocab is taken (do_lower_case), as llama.cpp's WordPiece
lowercases.

The weights come from model.safetensors (never a pickle). The
matrices and token_embd are Q8_0 (F32 with --f32); norms, biases,
positions, token types and the classifier stay F32. Any config this
does not handle is refused rather than converted wrong.

    .local-dev/vl-env/run-offline.sh scripts/text/convert-bert-classifier.py \\
        --model SNAPSHOT_DIR --out tests/models/toxic-bert-q8_0.gguf
"""
import argparse
import json
import os
import pathlib
import sys

import gguf
import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import hf_source  # noqa: E402

Q = gguf.GGMLQuantizationType
ARCH = "bert"
SPECIAL = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def problem_type(cfg):
    """@returns How the logits become scores, as transformers names
    it: config.json's problem_type, else what its text-classification
    pipeline applies without one (softmax over several labels, a
    sigmoid on one)."""
    kinds = ("single_label_classification", "multi_label_classification",
             "regression")
    p = cfg.get("problem_type")
    if p is None:
        p = kinds[0] if len(cfg["id2label"]) > 1 else kinds[1]
    if p not in kinds:
        raise SystemExit(f"problem_type {p!r} is not one of {kinds}")
    return p


def check_config(cfg, tcfg):
    """Exits unless cfg is the BERT classifier this layout holds."""
    want = {
        "architectures": ["BertForSequenceClassification"],
        "model_type": "bert",
        "hidden_act": "gelu",
    }
    for k, v in want.items():
        if cfg.get(k) != v:
            raise SystemExit(f"config {k} is {cfg.get(k)!r}, not {v!r}")
    if cfg.get("position_embedding_type", "absolute") != "absolute":
        raise SystemExit("only absolute position embeddings are handled")
    if cfg["hidden_size"] % cfg["num_attention_heads"] != 0:
        raise SystemExit("hidden_size is not a multiple of the heads")
    if not tcfg.get("do_lower_case", True):
        raise SystemExit("a cased vocab is not handled (llama.cpp's "
                         "WordPiece lowercases)")
    if not cfg.get("id2label"):
        raise SystemExit("config.json has no id2label")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="folder with config.json, model.safetensors and "
                         "vocab.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--f32", action="store_true",
                    help="keep the matrices F32 (a reference model)")
    a = ap.parse_args()
    with open(os.path.join(a.model, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tcfg = {}
    tpath = os.path.join(a.model, "tokenizer_config.json")
    if os.path.exists(tpath):
        with open(tpath, encoding="utf-8") as f:
            tcfg = json.load(f)
    check_config(cfg, tcfg)
    labels = [v for _, v in sorted(cfg["id2label"].items(),
                                   key=lambda kv: int(kv[0]))]
    layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]

    with open(os.path.join(a.model, "vocab.txt"), encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f]
    if len(vocab) != cfg["vocab_size"]:
        raise SystemExit(f"vocab.txt has {len(vocab)} pieces, config "
                         f"{cfg['vocab_size']}")
    ids = {t: i for i, t in enumerate(vocab)}
    for t in SPECIAL:
        if t not in ids:
            raise SystemExit(f"vocab.txt has no {t}")
    tokens, types = [], []
    for t in vocab:
        if t in SPECIAL:
            tokens.append(t)
            types.append(gguf.TokenType.CONTROL)
        else:
            tokens.append(t[2:] if t.startswith("##") else "▁" + t)
            types.append(gguf.TokenType.NORMAL)

    w = gguf.GGUFWriter(a.out, ARCH)
    w.add_name(hf_source.model_name(a.model))
    hf_source.add_source(w, path=a.model)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(hidden)
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_block_count(layers)
    w.add_head_count(cfg["num_attention_heads"])
    w.add_layer_norm_eps(cfg["layer_norm_eps"])
    w.add_causal_attention(False)
    w.add_pooling_type(gguf.PoolingType.RANK)
    w.add_classifier_output_labels(labels)
    w.add_string(f"{ARCH}.classifier.problem_type", problem_type(cfg))
    w.add_file_type(gguf.LlamaFileType.ALL_F32 if a.f32
                    else gguf.LlamaFileType.MOSTLY_Q8_0)
    w.add_token_type_count(cfg.get("type_vocab_size", 1))
    w.add_tokenizer_model("bert")
    w.add_token_list(tokens)
    w.add_token_types(types)
    # [CLS] as BOS, as llama.cpp stores it (gguf has no CLS key).
    w.add_bos_token_id(ids["[CLS]"])
    w.add_sep_token_id(ids["[SEP]"])
    w.add_unk_token_id(ids["[UNK]"])
    w.add_pad_token_id(ids["[PAD]"])
    w.add_mask_token_id(ids["[MASK]"])

    st = safe_open(os.path.join(a.model, "model.safetensors"), "np")
    read = set()
    # Tensors a classifier never uses (the masked-LM and next-sentence
    # heads of some checkpoints, the position ids buffer).
    unused = ("bert.embeddings.position_ids",)
    unused_prefix = ("cls.predictions.", "cls.seq_relationship.")

    def put(name, src, quant=False):
        read.add(src)
        x = st.get_tensor(src).astype(np.float32)
        if quant and not a.f32 and x.ndim == 2 and x.shape[1] % 32 == 0:
            w.add_tensor(name, gguf.quants.quantize(x, Q.Q8_0),
                         raw_dtype=Q.Q8_0)
        else:
            w.add_tensor(name, x)

    def put_norm(dst, src):
        put(dst + ".weight", src + ".weight")
        put(dst + ".bias", src + ".bias")

    e = "bert.embeddings."
    put("token_embd.weight", e + "word_embeddings.weight", quant=True)
    put("position_embd.weight", e + "position_embeddings.weight")
    put("token_types.weight", e + "token_type_embeddings.weight")
    put_norm("token_embd_norm", e + "LayerNorm")
    for i in range(layers):
        p = f"bert.encoder.layer.{i}."
        b = f"blk.{i}."
        for dst, src in (("attn_q", "attention.self.query"),
                         ("attn_k", "attention.self.key"),
                         ("attn_v", "attention.self.value"),
                         ("attn_output", "attention.output.dense"),
                         ("ffn_up", "intermediate.dense"),
                         ("ffn_down", "output.dense")):
            put(b + dst + ".weight", p + src + ".weight", quant=True)
            put(b + dst + ".bias", p + src + ".bias")
        put_norm(b + "attn_output_norm", p + "attention.output.LayerNorm")
        put_norm(b + "layer_output_norm", p + "output.LayerNorm")
    put("cls.weight", "bert.pooler.dense.weight")
    put("cls.bias", "bert.pooler.dense.bias")
    put("cls.output.weight", "classifier.weight")
    put("cls.output.bias", "classifier.bias")
    left = sorted(k for k in set(st.keys()) - read
                  if k not in unused and not k.startswith(unused_prefix))
    if left:
        raise SystemExit(f"tensors not converted: {left}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
