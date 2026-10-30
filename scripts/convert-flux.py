#!/usr/bin/env python3
"""Converts a diffusers FLUX.1 pipeline directory (FluxPipeline or
FluxKontextPipeline: black-forest-labs/FLUX.1-{dev,schnell,Kontext-dev})
to GGUFs (i#372):

- a "flux" GGUF with the transformer (FluxTransformer2DModel): the
  checkpoint names, 2D weights kept BF16 (the matvec kernels read
  BF16), every other tensor F32; the model keys from
  transformer/config.json and the sampler keys from
  scheduler/scheduler_config.json;
- a "clip_text" GGUF with the CLIP-L text encoder (CLIPTextModel), all
  F32, and its BPE tokenizer (vocab.json, merges.txt);
- an "autoencoder_kl" GGUF with the VAE encoder and decoder, all F32,
  with shift_factor and the quant-conv flags (FLUX has neither
  quant_conv nor post_quant_conv).

The T5-XXL text encoder (text_encoder_2) goes through
scripts/convert-t5-encoder.py. Each output may be left out.

The checkpoints are BF16, which safetensors.numpy cannot load, so this
reads the safetensors headers and tensor bytes itself.

Tested with Python 3.12, gguf 0.19.0, numpy, on
black-forest-labs/FLUX.1-Kontext-dev at 24e9ded:

    python3 -I scripts/convert-flux.py --model FLUX.1-Kontext-dev \\
        --dit-out flux-kontext-dev.gguf --clip-out clip-l.gguf \\
        --vae-out flux-vae.gguf
"""

import argparse
import json
import pathlib
import struct
import sys

import gguf
import numpy as np


def read_header(path):
    """@returns (header dict without metadata, data offset)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def load_dir(d, prefix):
    """@returns [(name, array)] of every prefix*.safetensors file in d,
    sorted by name; BF16 as uint16 bits, F16 and F32 as floats."""
    files = sorted(pathlib.Path(d).glob(prefix + "*.safetensors"))
    if not files:
        sys.exit("no %s*.safetensors in %s" % (prefix, d))
    out = []
    for path in files:
        hdr, base = read_header(path)
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        for name, t in hdr.items():
            b0, b1 = t["data_offsets"]
            raw = mm[base + b0:base + b1]
            shape = t["shape"] or [1]
            dt = {"BF16": np.uint16, "F16": np.float16,
                  "F32": np.float32}.get(t["dtype"])
            if dt is None:
                sys.exit("%s: unsupported dtype %s" % (name, t["dtype"]))
            out.append((name, raw.view(dt).reshape(shape)))
    out.sort(key=lambda x: x[0])
    return out


def to_f32(a):
    """@returns a as float32 (BF16 bits widened exactly)."""
    if a.dtype == np.uint16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def config(d):
    return json.loads((pathlib.Path(d) / "config.json").read_text())


def finish(w):
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def write_dit(path, mdir):
    t = config(mdir / "transformer")
    s = json.loads((mdir / "scheduler" /
                    "scheduler_config.json").read_text())
    if t.get("_class_name") != "FluxTransformer2DModel":
        sys.exit("expected a FluxTransformer2DModel")
    if t.get("patch_size", 1) != 1:
        sys.exit("expected patch_size 1 (the pipeline packs 2x2 itself)")
    heads, hd = t["num_attention_heads"], t["attention_head_dim"]
    axes = [int(x) for x in t["axes_dims_rope"]]
    if sum(axes) != hd or any(x % 2 for x in axes):
        sys.exit("the RoPE axes must be even and sum to the head width")
    out_ch = t.get("out_channels") or t["in_channels"]
    if out_ch != t["in_channels"]:
        sys.exit("expected out_channels == in_channels")
    if s.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        sys.exit("expected a FlowMatchEulerDiscreteScheduler")
    if s.get("time_shift_type", "exponential") != "exponential":
        sys.exit("expected the exponential time shift")
    if s.get("shift_terminal") or s.get("stochastic_sampling") or \
            s.get("invert_sigmas") or s.get("use_karras_sigmas") or \
            s.get("use_exponential_sigmas") or s.get("use_beta_sigmas"):
        sys.exit("unsupported scheduler option")
    inner = heads * hd
    guidance = bool(t["guidance_embeds"])

    tensors = load_dir(mdir / "transformer", "diffusion_pytorch_model")
    # The keys below that are not in the config: check them against the
    # tensor shapes (torch [out, in]).
    shapes = {name: a.shape for name, a in tensors}
    ff = shapes.get("transformer_blocks.0.ff.net.0.proj.weight", (0,))[0]
    mlp = shapes.get("single_transformer_blocks.0.proj_mlp.weight",
                     (0,))[0]
    if ff != 4 * inner or mlp != 4 * inner:
        sys.exit("expected feed-forward widths of 4 x inner")
    if shapes.get("time_text_embed.timestep_embedder.linear_1.weight") != \
            (inner, 256):
        sys.exit("expected 256 timestep channels")
    if ("time_text_embed.guidance_embedder.linear_1.weight" in shapes) != \
            guidance:
        sys.exit("guidance_embeds disagrees with the tensors")

    w = gguf.GGUFWriter(str(path), "flux")
    w.add_name(mdir.name)
    k = "flux."
    w.add_uint32(k + "block_count", t["num_layers"])
    w.add_uint32(k + "single_block_count", t["num_single_layers"])
    w.add_uint32(k + "embedding_length", inner)
    w.add_uint32(k + "feed_forward_length", ff)
    w.add_uint32(k + "attention.head_count", heads)
    w.add_uint32(k + "attention.key_length", hd)
    w.add_float32(k + "norm_epsilon", 1e-6)
    w.add_uint32(k + "in_channels", t["in_channels"])
    w.add_uint32(k + "joint_attention_dim", t["joint_attention_dim"])
    w.add_uint32(k + "pooled_projection_dim", t["pooled_projection_dim"])
    w.add_bool(k + "guidance_embeds", guidance)
    w.add_uint32(k + "timestep.channels", 256)
    w.add_float32(k + "rope.theta", 10000.0)
    w.add_array(k + "rope.axes", axes)
    p = k + "sampler."
    w.add_uint32(p + "num_train_timesteps", s["num_train_timesteps"])
    w.add_bool(p + "use_dynamic_shifting",
               bool(s.get("use_dynamic_shifting", False)))
    w.add_float32(p + "shift", float(s.get("shift", 1.0)))
    w.add_float32(p + "base_shift", float(s.get("base_shift", 0.5)))
    w.add_float32(p + "max_shift", float(s.get("max_shift", 1.15)))
    w.add_uint32(p + "base_image_seq_len", s.get("base_image_seq_len", 256))
    w.add_uint32(p + "max_image_seq_len", s.get("max_image_seq_len", 4096))

    n = 0
    for name, a in tensors:
        if a.ndim == 2 and a.dtype == np.uint16:
            w.add_tensor(name, np.ascontiguousarray(a),
                         raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            w.add_tensor(name, to_f32(a))
        n += 1
    # Per double block: the attention 20 (q, k, v, out and their added
    # twins, 2 each; four q/k norms), ff 4, ff_context 4, norm1 2,
    # norm1_context 2. Per single block: q/k norms 2, q/k/v 6, norm 2,
    # proj_mlp 2, proj_out 2. Outside: context_embedder 2, x_embedder 2,
    # the timestep and text embedders 8 (guidance 4 more), norm_out 2,
    # proj_out 2.
    want = 32 * t["num_layers"] + 14 * t["num_single_layers"] + 16 + \
        (4 if guidance else 0)
    if n != want:
        sys.exit("flux: wrote %d tensors, expected %d" % (n, want))
    finish(w)


def write_clip(path, mdir):
    c = config(mdir / "text_encoder")
    if c.get("model_type") != "clip_text_model":
        sys.exit("expected a CLIPTextModel")
    if c.get("hidden_act") != "quick_gelu":
        sys.exit("expected quick_gelu")
    tdir = mdir / "tokenizer"
    vocab = json.loads((tdir / "vocab.json").read_text())
    if len(vocab) != c["vocab_size"] or \
            sorted(vocab.values()) != list(range(len(vocab))):
        sys.exit("vocab.json must hold ids 0..vocab_size-1")
    tokens = [None] * len(vocab)
    for tok, i in vocab.items():
        tokens[i] = tok
    # CLIPTokenizer keeps the lines [1, 49152 - 256 - 2] of merges.txt
    # (the first is "#version"): 48894 merges.
    lines = (tdir / "merges.txt").read_text().strip().split("\n")
    merges = lines[1:49152 - 256 - 2 + 1]
    tc = json.loads((tdir / "tokenizer_config.json").read_text())
    special = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        v = tc.get(key)
        tok = v.get("content") if isinstance(v, dict) else v
        if tok is not None:
            if tok not in vocab:
                sys.exit("%s %r is not in the vocab" % (key, tok))
            special[key] = vocab[tok]

    w = gguf.GGUFWriter(str(path), "clip_text")
    w.add_name("clip-l")
    k = "clip_text."
    w.add_uint32(k + "block_count", c["num_hidden_layers"])
    w.add_uint32(k + "embedding_length", c["hidden_size"])
    w.add_uint32(k + "feed_forward_length", c["intermediate_size"])
    w.add_uint32(k + "attention.head_count", c["num_attention_heads"])
    w.add_uint32(k + "context_length", c["max_position_embeddings"])
    w.add_float32(k + "norm_epsilon", c["layer_norm_eps"])
    # transformers pools at input_ids.argmax() when eos_token_id is the
    # legacy 2, else at the first eos_token_id.
    w.add_uint32(k + "config_eos_token_id", c["eos_token_id"])
    w.add_tokenizer_model("clip")
    w.add_token_list(tokens)
    w.add_token_merges(merges)
    w.add_bos_token_id(special["bos_token"])
    w.add_eos_token_id(special["eos_token"])
    w.add_pad_token_id(special.get("pad_token", special["eos_token"]))
    w.add_unk_token_id(special.get("unk_token", special["eos_token"]))

    n = 0
    for name, a in load_dir(mdir / "text_encoder", "model"):
        if not name.startswith("text_model."):
            continue
        w.add_tensor(name, to_f32(a))
        n += 1
    # Embeddings 2, final norm 2; per layer: 2 norms of 2, q/k/v/out of 2,
    # fc1 and fc2 of 2.
    want = 4 + 16 * c["num_hidden_layers"]
    if n != want:
        sys.exit("clip_text: wrote %d tensors, expected %d" % (n, want))
    finish(w)


def write_vae(path, mdir):
    v = config(mdir / "vae")
    if v.get("_class_name") != "AutoencoderKL":
        sys.exit("expected an AutoencoderKL")
    if v.get("act_fn") != "silu":
        sys.exit("expected act_fn silu")
    if any(b != "UpDecoderBlock2D" for b in v["up_block_types"]) or \
            any(b != "DownEncoderBlock2D" for b in v["down_block_types"]):
        sys.exit("expected DownEncoderBlock2D / UpDecoderBlock2D blocks")
    if not v.get("mid_block_add_attention", True):
        sys.exit("expected the mid-block attention")
    chans = v["block_out_channels"]
    if any(c % v["norm_num_groups"] != 0 for c in chans):
        sys.exit("channels must be multiples of norm_num_groups")

    w = gguf.GGUFWriter(str(path), "autoencoder_kl")
    w.add_name("flux-vae")
    k = "autoencoder_kl."
    w.add_uint32(k + "latent_channels", v["latent_channels"])
    w.add_uint32(k + "out_channels", v["out_channels"])
    w.add_uint32(k + "in_channels", v["in_channels"])
    w.add_array(k + "block_out_channels", [int(c) for c in chans])
    w.add_uint32(k + "layers_per_block", v["layers_per_block"])
    w.add_uint32(k + "norm_groups", v["norm_num_groups"])
    w.add_float32(k + "norm_epsilon", 1e-6)
    w.add_float32(k + "scaling_factor", float(v["scaling_factor"]))
    w.add_float32(k + "shift_factor", float(v.get("shift_factor") or 0.0))
    w.add_bool(k + "mid_block_attention", True)
    w.add_bool(k + "use_quant_conv", bool(v.get("use_quant_conv", True)))
    w.add_bool(k + "use_post_quant_conv",
               bool(v.get("use_post_quant_conv", True)))

    n = 0
    for name, a in load_dir(mdir / "vae", "diffusion_pytorch_model"):
        w.add_tensor(name, to_f32(a))
        n += 1
    if n == 0:
        sys.exit("no VAE tensors")
    finish(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="the diffusers pipeline directory")
    ap.add_argument("--dit-out", help="the transformer GGUF")
    ap.add_argument("--clip-out", help="the CLIP-L text encoder GGUF")
    ap.add_argument("--vae-out", help="the VAE GGUF")
    a = ap.parse_args()
    if not (a.dit_out or a.clip_out or a.vae_out):
        sys.exit("give --dit-out, --clip-out, --vae-out or several")

    mdir = pathlib.Path(a.model)
    idx = json.loads((mdir / "model_index.json").read_text())
    if idx.get("_class_name") not in ("FluxPipeline", "FluxKontextPipeline"):
        sys.exit("expected a FluxPipeline or FluxKontextPipeline directory")
    if a.dit_out:
        write_dit(pathlib.Path(a.dit_out), mdir)
    if a.clip_out:
        write_clip(pathlib.Path(a.clip_out), mdir)
    if a.vae_out:
        write_vae(pathlib.Path(a.vae_out), mdir)


if __name__ == "__main__":
    main()
