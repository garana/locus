#!/usr/bin/env python3
"""Converts a DeBERTa v3 sequence classifier (transformers'
DebertaV2ForSequenceClassification, such as
protectai/deberta-v3-base-prompt-injection-v2 or Prompt Guard) to a
GGUF with arch "deberta-v2", for locus's DeBERTa encoder (i#1158).
llama.cpp's converter has no DeBERTa architecture.

The layout (keys under "deberta-v2."):
- context_length (max_position_embeddings), embedding_length,
  feed_forward_length, block_count, attention.head_count,
  attention.layer_norm_epsilon, position_buckets,
  max_relative_positions (as the model resolves -1), pooling_type 4
  (a classifier head), classifier.output_labels and
  classifier.problem_type (config.json's, else what transformers'
  text-classification pipeline does without one: softmax over several
  labels, single_label_classification, and a sigmoid on one,
  multi_label_classification);
- token_embd, token_embd_norm; rel_embd (the 2 * position_buckets
  relative embeddings) and rel_embd_norm (norm_rel_ebd);
- per block attn_q, attn_k, attn_v, attn_output, attn_output_norm,
  ffn_up, ffn_down, layer_output_norm, as BERT's names;
- cls (the pooler's dense, then GELU) and cls.output (one logit per
  label).

Only the DeBERTa v3 shape is taken: relative attention with both
c2p and p2c terms, share_att_key, log position buckets, a layer
norm on the relative embeddings, GELU, no absolute positions, no
token types, no convolution layer and no embedding projection. Any
other config is refused rather than converted wrong.

The tokenizer is SentencePiece unigram (spm.model), written as
llama.cpp's "t5" (UGM) tokenizer keys: the pieces, scores and types,
the normalizer's precompiled charsmap and flags, [CLS] (1) as BOS and
[SEP] (2) as EOS, both added. spm.model is read with a small protobuf
reader below, so no sentencepiece or protobuf package is needed.
[MASK] (128000 for v3) is a control token; the rows past the
tokenizer up to vocab_size are unused tokens.

The weights come from model.safetensors (never a pickle). The
matrices are Q8_0 (F32 with --f32); norms, biases, the relative
embeddings and the classifier stay F32.

Needs gguf, safetensors and numpy (the offline venv):

    .local-dev/vl-env/run-offline.sh scripts/text/convert-deberta.py \\
        --model SNAPSHOT_DIR \\
        --out tests/models/deberta-v3-base-injection-q8_0.gguf
"""
import argparse
import json
import os
import pathlib
import struct
import sys

import gguf
import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import hf_source  # noqa: E402

Q = gguf.GGMLQuantizationType
ARCH = "deberta-v2"


def varint(b, i):
    """@returns (value, next index) of the protobuf varint at b[i]."""
    v = shift = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << shift
        if c < 0x80:
            return v, i
        shift += 7


def fields(b):
    """Yields (field number, wire type, value) of a protobuf message:
    an int for varints, bytes for length-delimited fields, and the raw
    4 or 8 bytes for fixed ones."""
    i = 0
    while i < len(b):
        key, i = varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            n, i = varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise SystemExit(f"spm.model: wire type {wt} not handled")
        yield num, wt, v


def read_spm(path):
    """@returns (pieces [(text, score, type)], normalizer dict) from a
    SentencePiece ModelProto: field 1 the pieces (1 text, 2 score, 3
    type, NORMAL when absent), field 3 the normalizer spec (2 charsmap,
    3 add_dummy_prefix, 4 remove_extra_whitespaces, 5
    escape_whitespaces; each flag true when absent)."""
    with open(path, "rb") as f:
        b = f.read()
    pieces = []
    norm = {"charsmap": b"", "add_dummy_prefix": True,
            "remove_extra_whitespaces": True, "escape_whitespaces": True}
    for num, _, v in fields(b):
        if num == 1:
            text, score, typ = None, 0.0, 1
            for n2, _, v2 in fields(v):
                if n2 == 1:
                    text = bytes(v2).decode("utf-8")
                elif n2 == 2:
                    score = struct.unpack("<f", v2)[0]
                elif n2 == 3:
                    typ = v2
            pieces.append((text, score, typ))
        elif num == 3:
            names = {3: "add_dummy_prefix", 4: "remove_extra_whitespaces",
                     5: "escape_whitespaces"}
            for n2, _, v2 in fields(v):
                if n2 == 2:
                    norm["charsmap"] = bytes(v2)
                elif n2 in names:
                    norm[names[n2]] = bool(v2)
    return pieces, norm


def check_config(cfg):
    """Exits unless cfg is the DeBERTa v3 shape this layout holds."""
    want = {
        "architectures": ["DebertaV2ForSequenceClassification"],
        "relative_attention": True,
        "share_att_key": True,
        "position_biased_input": False,
        "type_vocab_size": 0,
        "hidden_act": "gelu",
        "pooler_hidden_act": "gelu",
    }
    for k, v in want.items():
        if cfg.get(k) != v:
            raise SystemExit(f"config {k} is {cfg.get(k)!r}, not {v!r}")
    if sorted(cfg.get("pos_att_type") or []) != ["c2p", "p2c"]:
        raise SystemExit("pos_att_type must be c2p and p2c")
    if "layer_norm" not in cfg.get("norm_rel_ebd", "none"):
        raise SystemExit("norm_rel_ebd must hold layer_norm")
    if cfg.get("conv_kernel_size", 0) > 0:
        raise SystemExit("a convolution layer (DeBERTa v2 xlarge) is not "
                         "handled")
    if cfg.get("embedding_size", cfg["hidden_size"]) != cfg["hidden_size"]:
        raise SystemExit("an embedding projection is not handled")
    if cfg.get("position_buckets", -1) <= 0:
        raise SystemExit("position_buckets must be positive")
    if cfg.get("pooler_hidden_size", cfg["hidden_size"]) != cfg["hidden_size"]:
        raise SystemExit("the pooler must be hidden_size wide")
    if cfg.get("attention_head_size",
               cfg["hidden_size"] // cfg["num_attention_heads"]) * \
            cfg["num_attention_heads"] != cfg["hidden_size"]:
        raise SystemExit("heads x head size must be hidden_size")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="folder with config.json, model.safetensors and "
                         "spm.model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--f32", action="store_true",
                    help="keep the matrices F32 (a reference model)")
    a = ap.parse_args()
    with open(os.path.join(a.model, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    check_config(cfg)
    labels = [v for _, v in sorted(cfg["id2label"].items(),
                                   key=lambda kv: int(kv[0]))]
    layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    max_rel = cfg.get("max_relative_positions", -1)
    if max_rel < 1:
        max_rel = cfg["max_position_embeddings"]

    pieces, norm = read_spm(os.path.join(a.model, "spm.model"))
    special = {1: "[CLS]", 2: "[SEP]", 3: "[UNK]", 0: "[PAD]"}
    for i, text in special.items():
        if pieces[i][0] != text:
            raise SystemExit(f"spm piece {i} is {pieces[i][0]!r}, not {text}")
    with open(os.path.join(a.model, "tokenizer_config.json"),
              encoding="utf-8") as f:
        tcfg = json.load(f)
    if tcfg.get("do_lower_case") or tcfg.get("split_by_punct"):
        raise SystemExit("do_lower_case and split_by_punct must be off")
    # Tokens added past the SentencePiece pieces ([MASK]), by id.
    added = {int(i): t["content"]
             for i, t in tcfg.get("added_tokens_decoder", {}).items()
             if int(i) >= len(pieces)}
    tokens = [p[0] for p in pieces]
    scores = [p[1] for p in pieces]
    types = [p[2] for p in pieces]
    vocab = cfg["vocab_size"]
    for i in range(len(pieces), vocab):
        if i in added:
            tokens.append(added[i])
            types.append(gguf.TokenType.CONTROL)
        else:
            tokens.append(f"[UNUSED_{i}]")
            types.append(gguf.TokenType.UNUSED)
        scores.append(0.0)

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
    w.add_uint32(f"{ARCH}.position_buckets", cfg["position_buckets"])
    w.add_uint32(f"{ARCH}.max_relative_positions", max_rel)
    w.add_pooling_type(gguf.PoolingType.RANK)
    w.add_classifier_output_labels(labels)
    w.add_string(f"{ARCH}.classifier.problem_type", problem_type(cfg))
    w.add_file_type(gguf.LlamaFileType.ALL_F32 if a.f32
                    else gguf.LlamaFileType.MOSTLY_Q8_0)

    w.add_tokenizer_model("t5")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(3)
    w.add_pad_token_id(0)
    w.add_add_bos_token(True)
    w.add_add_eos_token(True)
    w.add_add_space_prefix(norm["add_dummy_prefix"])
    w.add_remove_extra_whitespaces(norm["remove_extra_whitespaces"])
    if norm["charsmap"]:
        w.add_precompiled_charsmap(norm["charsmap"])

    st = safe_open(os.path.join(a.model, "model.safetensors"), "np")
    read = set()

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

    rows = st.get_slice("deberta.embeddings.word_embeddings.weight")
    if rows.get_shape() != [vocab, hidden]:
        raise SystemExit("word_embeddings is not vocab_size x hidden_size")
    put("token_embd.weight", "deberta.embeddings.word_embeddings.weight",
        quant=True)
    put_norm("token_embd_norm", "deberta.embeddings.LayerNorm")
    rel = st.get_slice("deberta.encoder.rel_embeddings.weight")
    if rel.get_shape() != [2 * cfg["position_buckets"], hidden]:
        raise SystemExit("rel_embeddings is not 2 * position_buckets rows")
    put("rel_embd.weight", "deberta.encoder.rel_embeddings.weight")
    put_norm("rel_embd_norm", "deberta.encoder.LayerNorm")
    for i in range(layers):
        p = f"deberta.encoder.layer.{i}."
        b = f"blk.{i}."
        for dst, src in (("attn_q", "attention.self.query_proj"),
                         ("attn_k", "attention.self.key_proj"),
                         ("attn_v", "attention.self.value_proj"),
                         ("attn_output", "attention.output.dense"),
                         ("ffn_up", "intermediate.dense"),
                         ("ffn_down", "output.dense")):
            put(b + dst + ".weight", p + src + ".weight", quant=True)
            put(b + dst + ".bias", p + src + ".bias")
        put_norm(b + "attn_output_norm", p + "attention.output.LayerNorm")
        put_norm(b + "layer_output_norm", p + "output.LayerNorm")
    put("cls.weight", "pooler.dense.weight")
    put("cls.bias", "pooler.dense.bias")
    put("cls.output.weight", "classifier.weight")
    put("cls.output.bias", "classifier.bias")
    left = sorted(set(st.keys()) - read)
    if left:
        raise SystemExit(f"tensors not converted: {left}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
