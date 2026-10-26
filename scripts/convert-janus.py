#!/usr/bin/env python3
"""Converts Janus-Pro's language model and image-generation heads to a
"llama" GGUF that locus loads (i#221 section 5), so that the text
encoder, the backbone and the image-token heads come from one file.

- language_model.model.* / lm_head -> the standard llama GGUF names, with
  llama.cpp's Q / K row permutation (locus's llama path ropes
  interleaved pairs, HF's checkpoint is rotate-half).
- The BPE tokenizer from tokenizer.json: tokenizer.ggml.model "gpt2",
  pre "deepseek-llm", tokens padded to the embedding size, the added
  tokens as control tokens, bos 100000, eos 100001 and add_bos_token
  written explicitly (locus defaults it off for this pre-tokenizer).
- The image-generation extras, F32, under janus.*: gen_embed (the image
  token embedding table), gen_aligner (Linear, GELU, Linear into the
  residual stream) and gen_head (Linear, GELU, Linear to the image
  codes), plus janus.image_tokens / image_grid / begin_of_image / pad
  / cfg_scale.
- The image-understanding encoder (i#241) under vit.*: the SigLIP-L/16
  vision tower (patch_embd conv, pos_embd, 24 pre-norm blocks of
  ln1 / attn_qkv / attn_out / ln2 / ffn_up / ffn_down, post_ln) and the
  aligner (Linear, GELU, Linear, 1024 -> 2048), plus the janus.vision.*
  keys (sizes, eps, and the placeholder / end_of_image ids).

Matrices are written as BF16 by default, the checkpoint's own type,
so every weight is exact (F16 would round the smallest ones); --outtype
f16 / f32 are the alternatives. Norms, biases, the patch-embed conv,
the position table and the janus.* tensors are F32.

Dev-time tool: needs the Janus-Pro snapshot (pytorch_model.bin,
loaded with torch's weights_only default, tokenizer.json, config.json).
Tested with Python 3.10, torch 2.14.1, gguf 0.19.0, numpy 2.2.6,
Janus-Pro-1B revision 960ab33191f61342a4c60ae74d8dc356a39fafcb:

    python3 scripts/convert-janus.py --model DIR \\
        --out tests/models/janus-pro-1b-f16.gguf

With those versions the F16 output (3.5 GB, written before the vit.*
tensors) has sha256
977b8a1ced63a2bdf2c20f00d6530b8f43f3d3bacc06907d6e02d27396cf7fc5,
and the default BF16 output with vit.* (4.1 GB,
tests/models/janus-pro-1b-bf16.gguf) has sha256
2145ae9c090dfbac33e78df79a438e9ff27c46d24ff44c780bef3f467666e51a.
Check: locus-run on "The capital of France is" continues with the
same 12 tokens as HF's f32 greedy decode.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

LM = "language_model."


def permute(w, n_head):
    """llama.cpp's HF -> GGUF Q / K row order (rotate-half to pairs)."""
    return (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
            .swapaxes(1, 2).reshape(w.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Janus-Pro snapshot")
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--outtype", choices=["bf16", "f16", "f32"],
                    default="bf16")
    args = ap.parse_args()
    d = pathlib.Path(args.model)
    cfg = json.loads((d / "config.json").read_text())["language_config"]
    tok = json.loads((d / "tokenizer.json").read_text())
    if tok["model"]["type"] != "BPE":
        sys.exit("tokenizer is not BPE")
    pre = tok["pre_tokenizer"]["pretokenizers"]
    # The deepseek-llm split (i#226): refuse a different tokenizer
    # rather than tag it with the wrong pre-tokenizer name.
    if [p["type"] for p in pre] != ["Split"] * 5 + ["Digits", "ByteLevel"]:
        sys.exit("pre-tokenizer is not the deepseek-llm sequence")

    sd = torch.load(d / "pytorch_model.bin", map_location="cpu")
    n_vocab = cfg["vocab_size"]
    n_head = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_head)
    n_layer = cfg["num_hidden_layers"]
    big = {"bf16": "bf16", "f16": np.float16, "f32": np.float32}[
        args.outtype]

    from gguf import GGUFWriter, GGMLQuantizationType, TokenType, quants
    out = GGUFWriter(args.out, "llama")
    out.add_name("Janus-Pro-1B")
    out.add_context_length(cfg["max_position_embeddings"])
    out.add_embedding_length(cfg["hidden_size"])
    out.add_block_count(n_layer)
    out.add_feed_forward_length(cfg["intermediate_size"])
    out.add_head_count(n_head)
    out.add_head_count_kv(n_kv)
    out.add_rope_dimension_count(cfg["hidden_size"] // n_head)
    out.add_rope_freq_base(10000.0)      # transformers LlamaConfig default
    out.add_layer_norm_rms_eps(1e-6)     # likewise (absent from config)
    out.add_vocab_size(n_vocab)

    # Tokenizer: vocab + added tokens, padded to the embedding rows.
    tokens = [""] * n_vocab
    types = [TokenType.UNUSED] * n_vocab
    for s, i in tok["model"]["vocab"].items():
        tokens[i] = s
        types[i] = TokenType.NORMAL
    for a in tok["added_tokens"]:
        tokens[a["id"]] = a["content"]
        types[a["id"]] = (TokenType.CONTROL if a["special"]
                          else TokenType.USER_DEFINED)
    for i in range(n_vocab):
        if tokens[i] == "":
            tokens[i] = f"[PAD{i}]"
    merges = tok["model"]["merges"]
    if merges and not isinstance(merges[0], str):
        merges = [" ".join(m) for m in merges]
    out.add_tokenizer_model("gpt2")
    out.add_tokenizer_pre("deepseek-llm")
    out.add_token_list(tokens)
    out.add_token_types(types)
    out.add_token_merges(merges)
    out.add_bos_token_id(100000)
    out.add_eos_token_id(100001)
    out.add_add_bos_token(True)
    out.add_add_eos_token(False)

    out.add_uint32("janus.image_tokens", 16384)
    out.add_uint32("janus.image_grid", 24)
    out.add_uint32("janus.begin_of_image", 100003)
    out.add_uint32("janus.pad", 100002)
    out.add_float32("janus.cfg_scale", 5.0)

    def t(name):
        if name not in sd:
            sys.exit(f"missing {name}")
        return sd[name].detach().float().numpy()

    def put(name, arr, dtype):
        if dtype == "bf16":
            # Exact: the values came from bf16 (bf16 -> f32 -> bf16).
            raw = quants.quantize(np.ascontiguousarray(arr, np.float32),
                                  GGMLQuantizationType.BF16)
            out.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.BF16)
        else:
            out.add_tensor(name, np.ascontiguousarray(arr.astype(dtype)))

    put("token_embd.weight", t(LM + "model.embed_tokens.weight"), big)
    for i in range(n_layer):
        p = f"{LM}model.layers.{i}."
        b = f"blk.{i}."
        put(b + "attn_norm.weight", t(p + "input_layernorm.weight"),
            np.float32)
        put(b + "attn_q.weight",
            permute(t(p + "self_attn.q_proj.weight"), n_head), big)
        put(b + "attn_k.weight",
            permute(t(p + "self_attn.k_proj.weight"), n_kv), big)
        put(b + "attn_v.weight", t(p + "self_attn.v_proj.weight"), big)
        put(b + "attn_output.weight", t(p + "self_attn.o_proj.weight"), big)
        put(b + "ffn_norm.weight", t(p + "post_attention_layernorm.weight"),
            np.float32)
        put(b + "ffn_gate.weight", t(p + "mlp.gate_proj.weight"), big)
        put(b + "ffn_up.weight", t(p + "mlp.up_proj.weight"), big)
        put(b + "ffn_down.weight", t(p + "mlp.down_proj.weight"), big)
    put("output_norm.weight", t(LM + "model.norm.weight"), np.float32)
    put("output.weight", t(LM + "lm_head.weight"), big)

    # Image-generation extras (F32; small next to the backbone).
    extras = {
        "janus.gen_embed.weight": "gen_embed.weight",
        "janus.gen_aligner.0.weight": "gen_aligner.layers.0.weight",
        "janus.gen_aligner.0.bias": "gen_aligner.layers.0.bias",
        "janus.gen_aligner.2.weight": "gen_aligner.layers.2.weight",
        "janus.gen_aligner.2.bias": "gen_aligner.layers.2.bias",
        "janus.gen_head.proj.weight": "gen_head.output_mlp_projector.weight",
        "janus.gen_head.proj.bias": "gen_head.output_mlp_projector.bias",
        "janus.gen_head.out.weight": "gen_head.vision_head.weight",
        "janus.gen_head.out.bias": "gen_head.vision_head.bias",
    }
    for dst, src in extras.items():
        put(dst, t(src), np.float32)

    # Image understanding (i#241): the SigLIP-L/16 tower and the aligner.
    vcfg = json.loads((d / "config.json").read_text())["vision_config"]
    if vcfg["params"]["model_name"] != "siglip_large_patch16_384":
        sys.exit("vision tower is not siglip_large_patch16_384")
    ids = {s: i for i, s in enumerate(tokens)}
    for tag in ("<image_placeholder>", "<begin_of_image>", "<end_of_image>"):
        if tag not in ids:
            sys.exit(f"tokenizer has no {tag}")
    V = "vision_model.vision_tower."
    n_vlayer = 24
    out.add_uint32("janus.vision.image_size", 384)
    out.add_uint32("janus.vision.patch_size", 16)
    out.add_uint32("janus.vision.embedding_length", 1024)
    out.add_uint32("janus.vision.block_count", n_vlayer)
    out.add_uint32("janus.vision.head_count", 16)
    out.add_uint32("janus.vision.feed_forward_length", 4096)
    out.add_float32("janus.vision.layer_norm_eps", 1e-6)
    out.add_uint32("janus.vision.image_placeholder", ids["<image_placeholder>"])
    out.add_uint32("janus.vision.end_of_image", ids["<end_of_image>"])
    put("vit.patch_embd.weight", t(V + "patch_embed.proj.weight"), np.float32)
    put("vit.patch_embd.bias", t(V + "patch_embed.proj.bias"), np.float32)
    put("vit.pos_embd", t(V + "pos_embed")[0], np.float32)
    for i in range(n_vlayer):
        p = f"{V}blocks.{i}."
        b = f"vit.blk.{i}."
        if f"{p}ls1.gamma" in sd or f"{p}attn.q_norm.weight" in sd:
            sys.exit("vision block has layer scale or qk norm")
        for dst, src, typ in (
                ("ln1.weight", "norm1.weight", np.float32),
                ("ln1.bias", "norm1.bias", np.float32),
                ("attn_qkv.weight", "attn.qkv.weight", big),
                ("attn_qkv.bias", "attn.qkv.bias", np.float32),
                ("attn_out.weight", "attn.proj.weight", big),
                ("attn_out.bias", "attn.proj.bias", np.float32),
                ("ln2.weight", "norm2.weight", np.float32),
                ("ln2.bias", "norm2.bias", np.float32),
                ("ffn_up.weight", "mlp.fc1.weight", big),
                ("ffn_up.bias", "mlp.fc1.bias", np.float32),
                ("ffn_down.weight", "mlp.fc2.weight", big),
                ("ffn_down.bias", "mlp.fc2.bias", np.float32)):
            put(b + dst, t(p + src), typ)
    if f"{V}blocks.{n_vlayer}.norm1.weight" in sd:
        sys.exit(f"vision tower has more than {n_vlayer} blocks")
    put("vit.post_ln.weight", t(V + "norm.weight"), np.float32)
    put("vit.post_ln.bias", t(V + "norm.bias"), np.float32)
    put("vit.aligner.0.weight", t("aligner.layers.0.weight"), big)
    put("vit.aligner.0.bias", t("aligner.layers.0.bias"), np.float32)
    put("vit.aligner.2.weight", t("aligner.layers.2.weight"), big)
    put("vit.aligner.2.bias", t("aligner.layers.2.bias"), np.float32)

    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print(f"wrote {args.out}: {n_layer} layers, vocab {n_vocab}, "
          f"{len(merges)} merges")


if __name__ == "__main__":
    main()
