#!/usr/bin/env python3
"""Converts an XLM-R cross-encoder reranker (transformers'
XLMRobertaForSequenceClassification, such as BAAI/bge-reranker-v2-m3)
to a GGUF in llama.cpp's layout: arch "bert" with pooling_type 4
(rank), the classifier as "cls" (dense, then tanh) and "cls.output"
(one logit per query-document pair), for locus's /v1/rerank (i#503).

The weights come from model.safetensors (never a pickle). The
tokenizer keys are copied from a GGUF of the same SentencePiece model,
so no sentencepiece package is needed: bge-reranker-v2-m3 and bge-m3
share sentencepiece.bpe.model (sha256 cfc8146a...0db2865), and
ggml-org/bge-m3-Q8_0-GGUF (commit 9eba04c) has its vocab. As
llama.cpp's converter does, the position rows XLM-R skips (the first
pad_token_id + 1) are dropped, and the matrices are Q8_0 (F32 with
--f32); norms, biases, positions, token types and the classifier stay
F32.

Needs gguf 0.19.0, safetensors 0.4.5 and numpy (the golden venv):

    python3 -I scripts/convert-xlmr-reranker.py \\
        --model DIR/bge-reranker-v2-m3 \\
        --vocab tests/models/bge-m3-q8_0.gguf \\
        --out tests/models/bge-reranker-v2-m3-q8_0.gguf
"""
import argparse
import json
import os

import gguf
import numpy as np
from safetensors import safe_open

Q = gguf.GGMLQuantizationType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="directory with config.json and model.safetensors")
    ap.add_argument("--vocab", required=True,
                    help="a GGUF with the same SentencePiece vocab")
    ap.add_argument("--out", required=True)
    ap.add_argument("--f32", action="store_true",
                    help="keep the matrices F32 (a reference model)")
    a = ap.parse_args()
    with open(os.path.join(a.model, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("architectures") != ["XLMRobertaForSequenceClassification"]:
        raise SystemExit("not an XLMRobertaForSequenceClassification")
    labels = [v for _, v in sorted(cfg["id2label"].items(),
                                   key=lambda kv: int(kv[0]))]
    offset = cfg["pad_token_id"] + 1
    layers = cfg["num_hidden_layers"]

    vocab = gguf.GGUFReader(a.vocab)
    if vocab.get_field("tokenizer.ggml.model").contents() != "t5":
        raise SystemExit("the vocab GGUF has no SentencePiece vocab")
    n_vocab = len(vocab.get_field("tokenizer.ggml.tokens").contents())
    if n_vocab != cfg["vocab_size"]:
        raise SystemExit(f"vocab sizes differ: {n_vocab} {cfg['vocab_size']}")

    w = gguf.GGUFWriter(a.out, "bert")
    w.add_name("bge-reranker-v2-m3")
    w.add_context_length(cfg["max_position_embeddings"] - offset)
    w.add_embedding_length(cfg["hidden_size"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_block_count(layers)
    w.add_head_count(cfg["num_attention_heads"])
    w.add_layer_norm_eps(cfg["layer_norm_eps"])
    w.add_causal_attention(False)
    w.add_pooling_type(gguf.PoolingType.RANK)
    w.add_classifier_output_labels(labels)
    w.add_file_type(gguf.LlamaFileType.ALL_F32 if a.f32
                    else gguf.LlamaFileType.MOSTLY_Q8_0)
    for name, field in vocab.fields.items():
        if not name.startswith("tokenizer."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(name, field.contents(), vtype, sub_type=sub)

    st = safe_open(os.path.join(a.model, "model.safetensors"), "np")

    def put(name, src, quant=False, rows=None):
        x = st.get_tensor(src).astype(np.float32)
        if rows is not None:
            x = x[rows]
        if quant and not a.f32 and x.ndim == 2 and x.shape[1] % 32 == 0:
            # The shape comes from the quantized bytes' shape.
            w.add_tensor(name, gguf.quants.quantize(x, Q.Q8_0),
                         raw_dtype=Q.Q8_0)
        else:
            w.add_tensor(name, x)

    e = "roberta.embeddings."
    put("token_embd.weight", e + "word_embeddings.weight", quant=True)
    put("position_embd.weight", e + "position_embeddings.weight",
        rows=slice(offset, None))
    put("token_types.weight", e + "token_type_embeddings.weight",
        rows=0 if cfg["type_vocab_size"] == 1 else None)
    put("token_embd_norm.weight", e + "LayerNorm.weight")
    put("token_embd_norm.bias", e + "LayerNorm.bias")
    for i in range(layers):
        p = f"roberta.encoder.layer.{i}."
        b = f"blk.{i}."
        for dst, src in (("attn_q", "attention.self.query"),
                         ("attn_k", "attention.self.key"),
                         ("attn_v", "attention.self.value"),
                         ("attn_output", "attention.output.dense"),
                         ("ffn_up", "intermediate.dense"),
                         ("ffn_down", "output.dense")):
            put(b + dst + ".weight", p + src + ".weight", quant=True)
            put(b + dst + ".bias", p + src + ".bias")
        for dst, src in (("attn_output_norm", "attention.output.LayerNorm"),
                         ("layer_output_norm", "output.LayerNorm")):
            put(b + dst + ".weight", p + src + ".weight")
            put(b + dst + ".bias", p + src + ".bias")
    put("cls.weight", "classifier.dense.weight")
    put("cls.bias", "classifier.dense.bias")
    put("cls.output.weight", "classifier.out_proj.weight")
    put("cls.output.bias", "classifier.out_proj.bias")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
