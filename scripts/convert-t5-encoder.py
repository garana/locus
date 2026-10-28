#!/usr/bin/env python3
"""Converts a T5 v1.1 encoder (transformers T5EncoderModel) and its
SentencePiece tokenizer to a "t5encoder" GGUF (i#308), with llama.cpp's
names and keys, so llama.cpp's T5 encoder GGUFs load the same way.

- Tensors keep the checkpoint dtype (F16 for PixArt's text_encoder):
  shared.weight -> token_embd.weight; per block N
  layer.0.SelfAttention.{q,k,v,o} -> enc.blk.N.attn_{q,k,v,o},
  layer.0.SelfAttention.relative_attention_bias (block 0 only) ->
  enc.blk.0.attn_rel_b, layer.0.layer_norm -> enc.blk.N.attn_norm,
  layer.1.DenseReluDense.{wi_0,wi_1,wo} -> enc.blk.N.ffn_{gate,up,down},
  layer.1.layer_norm -> enc.blk.N.ffn_norm; encoder.final_layer_norm
  -> enc.output_norm.
- Tokenizer: tokenizer.ggml.model "t5"; the SentencePiece pieces with
  their scores and types, then the added tokens (<extra_id_N>) as
  user-defined, then "[PADn]" unused tokens up to the embedding rows;
  the normalizer's precompiled_charsmap; add_space_prefix and
  remove_extra_whitespaces from the normalizer spec.

--vocab-only writes the metadata and tokenizer without tensors (about
1 MB), for tokenizer tests.

Tested with Python 3.12, gguf 0.19.0, sentencepiece 0.2.0, protobuf
and safetensors, on PixArt-alpha/PixArt-XL-2-512x512 at 50f7021:

    python3 -I scripts/convert-t5-encoder.py --model DIR/text_encoder \\
        --tokenizer DIR/tokenizer --variant fp16 --out t5-xxl-enc.gguf
"""

import argparse
import json
import pathlib
import sys

import gguf
import numpy as np
from safetensors.numpy import load_file
from sentencepiece import sentencepiece_model_pb2 as spb

# SentencePiece piece types -> GGUF token types.
SP_TYPES = {1: gguf.TokenType.NORMAL, 2: gguf.TokenType.UNKNOWN,
            3: gguf.TokenType.CONTROL, 4: gguf.TokenType.USER_DEFINED,
            5: gguf.TokenType.UNUSED, 6: gguf.TokenType.BYTE}


def tensor_name(name):
    """The GGUF name for a checkpoint tensor, or None to drop it."""
    if name == "shared.weight":
        return "token_embd.weight"
    if name == "encoder.final_layer_norm.weight":
        return "enc.output_norm.weight"
    parts = name.split(".")
    if parts[:2] != ["encoder", "block"]:
        return None
    n = parts[2]
    rest = ".".join(parts[3:])
    table = {
        "layer.0.SelfAttention.q.weight": "attn_q",
        "layer.0.SelfAttention.k.weight": "attn_k",
        "layer.0.SelfAttention.v.weight": "attn_v",
        "layer.0.SelfAttention.o.weight": "attn_o",
        "layer.0.SelfAttention.relative_attention_bias.weight":
            "attn_rel_b",
        "layer.0.layer_norm.weight": "attn_norm",
        "layer.1.DenseReluDense.wi_0.weight": "ffn_gate",
        "layer.1.DenseReluDense.wi_1.weight": "ffn_up",
        "layer.1.DenseReluDense.wo.weight": "ffn_down",
        "layer.1.layer_norm.weight": "ffn_norm",
    }
    if rest not in table:
        return None
    return "enc.blk.%s.%s.weight" % (n, table[rest])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="T5EncoderModel directory (config.json, shards)")
    ap.add_argument("--tokenizer", required=True,
                    help="directory with spiece.model, added_tokens.json")
    ap.add_argument("--variant", default="",
                    help="shard name variant, e.g. fp16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-only", action="store_true")
    a = ap.parse_args()

    mdir = pathlib.Path(a.model)
    cfg = json.loads((mdir / "config.json").read_text())
    if cfg.get("feed_forward_proj") != "gated-gelu":
        sys.exit("only gated-gelu (T5 v1.1) encoders are supported")

    w = gguf.GGUFWriter(a.out, "t5encoder")
    w.add_name(cfg.get("_name_or_path", "t5").split("/")[-1] or "t5")
    w.add_context_length(512)
    w.add_embedding_length(cfg["d_model"])
    w.add_feed_forward_length(cfg["d_ff"])
    w.add_block_count(cfg["num_layers"])
    w.add_head_count(cfg["num_heads"])
    w.add_key_length(cfg["d_kv"])
    w.add_value_length(cfg["d_kv"])
    w.add_layer_norm_rms_eps(cfg["layer_norm_epsilon"])
    w.add_relative_attn_buckets_count(cfg["relative_attention_num_buckets"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    tdir = pathlib.Path(a.tokenizer)
    sp = spb.ModelProto()
    sp.ParseFromString((tdir / "spiece.model").read_bytes())
    if sp.trainer_spec.model_type != 1:
        sys.exit("expected a unigram SentencePiece model")
    tokens = [p.piece.encode("utf-8") for p in sp.pieces]
    scores = [float(p.score) for p in sp.pieces]
    types = [SP_TYPES[p.type] for p in sp.pieces]
    added = {}
    added_path = tdir / "added_tokens.json"
    if added_path.exists():
        added = {int(i): t for t, i in
                 json.loads(added_path.read_text()).items()}
    vocab_rows = cfg["vocab_size"]
    for i in range(len(tokens), vocab_rows):
        if i in added:
            tokens.append(added[i].encode("utf-8"))
            types.append(gguf.TokenType.USER_DEFINED)
        else:
            tokens.append(("[PAD%d]" % i).encode("utf-8"))
            types.append(gguf.TokenType.UNUSED)
        scores.append(0.0)
    w.add_tokenizer_model("t5")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    n = sp.normalizer_spec
    w.add_add_space_prefix(n.add_dummy_prefix)
    w.add_remove_extra_whitespaces(n.remove_extra_whitespaces)
    w.add_precompiled_charsmap(n.precompiled_charsmap)
    w.add_eos_token_id(sp.trainer_spec.eos_id)
    w.add_unk_token_id(sp.trainer_spec.unk_id)
    w.add_pad_token_id(sp.trainer_spec.pad_id)
    w.add_add_bos_token(False)
    w.add_add_eos_token(True)

    if not a.vocab_only:
        suffix = ("." + a.variant) if a.variant else ""
        shards = sorted(mdir.glob("model%s*.safetensors" % suffix))
        if not shards:
            sys.exit("no model%s*.safetensors in %s" % (suffix, mdir))
        seen = 0
        for shard in shards:
            for name, t in load_file(str(shard)).items():
                gname = tensor_name(name)
                if gname is None:
                    continue
                w.add_tensor(gname, t)
                seen += 1
        want = 2 + 9 * cfg["num_layers"] + 1
        if seen != want:
            sys.exit("wrote %d tensors, expected %d" % (seen, want))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
