#!/usr/bin/env python3
"""Converts an LTX-Video single-file checkpoint (DiT + VAE in one
safetensors, the format of Lightricks/LTX-Video's ltxv-2b-*.safetensors)
to two GGUFs (i#357):

- an "ltxv" GGUF with the DiT: the model.diffusion_model.* tensors
  without that prefix; 2D weights stay BF16 (the matvec kernels read
  BF16), every other tensor becomes F32; the model and sampler keys
  come from the checkpoint's "config" metadata;
- an "ltxv_vae" GGUF with the VAE decoder: vae.decoder.* and
  vae.per_channel_statistics.* without the "vae." prefix, all F32 (the
  conv kernels read F32); the encoder is left out. Without --vae-out
  only the DiT is written (the synthetic goldens have no VAE).

The checkpoint is BF16, which safetensors.numpy cannot load, so this
reads the safetensors header and tensor bytes itself.

Tested with Python 3.12, gguf 0.19.0, numpy, on
ltxv-2b-0.9.8-distilled.safetensors (sha256 76aa8c47...2cad8f):

    python3 -I scripts/convert-ltxv.py \\
        --ckpt ltxv-2b-0.9.8-distilled.safetensors \\
        --dit-out ltxv-2b-dit.gguf --vae-out ltxv-2b-vae.gguf
"""

import argparse
import json
import struct
import sys

import gguf
import numpy as np

DIT_PREFIX = "model.diffusion_model."
VAE_KEEP = ("vae.decoder.", "vae.per_channel_statistics.")


def read_header(path):
    """@returns (header dict without metadata, metadata, data offset)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", {})
    return hdr, meta, 8 + n


def load(mm, base, t):
    """@returns the tensor as uint16 (BF16 bits) or float32, its shape."""
    b0, b1 = t["data_offsets"]
    raw = mm[base + b0:base + b1]
    shape = t["shape"] or [1]
    if t["dtype"] == "BF16":
        return raw.view(np.uint16).reshape(shape)
    if t["dtype"] == "F32":
        return raw.view(np.float32).reshape(shape)
    sys.exit("unsupported dtype %s" % t["dtype"])


def to_f32(a):
    """@returns a as float32 (BF16 bits widened exactly)."""
    if a.dtype == np.uint16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def write_dit(path, cfg, sch, steps, tensors):
    t = cfg
    if t.get("positional_embedding_type") != "rope":
        sys.exit("expected rope positions")
    if t.get("qk_norm") != "rms_norm" or \
            t.get("standardization_norm") != "rms_norm":
        sys.exit("expected rms_norm q/k and block norms")
    if t.get("activation_fn") != "gelu-approximate":
        sys.exit("expected activation_fn gelu-approximate")
    if not t.get("attention_bias") or t.get("norm_elementwise_affine"):
        sys.exit("expected attention biases and norms without affine")
    heads, hd = t["num_attention_heads"], t["attention_head_dim"]
    inner = heads * hd
    if t["cross_attention_dim"] != inner:
        sys.exit("expected cross_attention_dim == inner dim")
    if sch.get("_class_name") != "RectifiedFlowScheduler":
        sys.exit("expected a RectifiedFlowScheduler")
    # The keys below that are not in the config: check them against the
    # tensor shapes (torch [out, in]).
    shapes = {name: a.shape for name, a in tensors}
    if shapes.get("transformer_blocks.0.ff.net.0.proj.weight") != \
            (4 * inner, inner):
        sys.exit("expected a feed-forward width of 4 x inner")
    if shapes.get("adaln_single.emb.timestep_embedder.linear_1.weight") != \
            (inner, 256):
        sys.exit("expected 256 timestep channels")

    w = gguf.GGUFWriter(path, "ltxv")
    w.add_name("ltxv")
    k = "ltxv."
    w.add_uint32(k + "block_count", t["num_layers"])
    w.add_uint32(k + "embedding_length", inner)
    w.add_uint32(k + "feed_forward_length", 4 * inner)
    w.add_uint32(k + "attention.head_count", heads)
    w.add_uint32(k + "attention.key_length", hd)
    w.add_float32(k + "norm_epsilon", t["norm_eps"])
    w.add_uint32(k + "in_channels", t["in_channels"])
    w.add_uint32(k + "out_channels", t["out_channels"])
    w.add_uint32(k + "caption_channels", t["caption_channels"])
    w.add_float32(k + "rope.theta", t["positional_embedding_theta"])
    w.add_array(k + "rope.max_pos",
                [int(x) for x in t["positional_embedding_max_pos"]])
    w.add_float32(k + "timestep.scale_multiplier",
                  float(t["timestep_scale_multiplier"]))
    w.add_uint32(k + "timestep.channels", 256)
    s = k + "sampler."
    w.add_uint32(s + "num_train_timesteps", sch["num_train_timesteps"])
    w.add_string(s + "sampler", sch.get("sampler") or "")
    w.add_array(s + "allowed_timesteps", [float(x) for x in steps])

    n = 0
    for name, a in tensors:
        if a.ndim == 2 and a.dtype == np.uint16:
            w.add_tensor(name, np.ascontiguousarray(a),
                         raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            w.add_tensor(name, to_f32(a))
        n += 1
    # adaln_single 6, caption_projection 4, patchify_proj 2, proj_out 2,
    # the final table 1; per block attn1 10, attn2 10, ff 4, table 1.
    want = 15 + 25 * t["num_layers"]
    if n != want:
        sys.exit("ltxv: wrote %d tensors, expected %d" % (n, want))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def write_vae(path, cfg, tensors):
    v = cfg
    if v.get("dims") != 3 or v.get("causal_decoder"):
        sys.exit("expected a 3D, non-causal decoder")
    if v.get("norm_layer") != "pixel_norm":
        sys.exit("expected pixel_norm")
    if not v.get("timestep_conditioning"):
        sys.exit("expected a timestep-conditioned decoder")
    kinds, params = [], []
    for kind, p in v["decoder_blocks"]:
        if kind == "res_x":
            if p.get("inject_noise"):
                sys.exit("inject_noise is not supported")
            kinds.append(kind)
            params.append(int(p["num_layers"]))
        elif kind == "compress_all":
            if not p.get("residual"):
                sys.exit("expected residual compress_all blocks")
            kinds.append(kind)
            params.append(int(p.get("multiplier", 1)))
        else:
            sys.exit("unsupported decoder block %s" % kind)

    w = gguf.GGUFWriter(path, "ltxv_vae")
    w.add_name("ltxv-vae")
    k = "ltxv_vae."
    w.add_uint32(k + "latent_channels", v["latent_channels"])
    w.add_uint32(k + "out_channels", v["out_channels"])
    w.add_uint32(k + "patch_size", v["patch_size"])
    w.add_float32(k + "scaling_factor", float(v["scaling_factor"]))
    w.add_array(k + "decoder.block_kinds", kinds)
    w.add_array(k + "decoder.block_params", params)
    # GGUF tensors have at most 4 dims, so a conv3d weight
    # [out, in, 3, 3, 3] is stored as [out, in, 27] (kt, kh, kw order).
    w.add_uint32(k + "decoder.kernel_size", 3)

    n = 0
    for name, a in tensors:
        if a.ndim == 5:
            if a.shape[2:] != (3, 3, 3):
                sys.exit("%s: expected a 3x3x3 kernel" % name)
            a = a.reshape(a.shape[0], a.shape[1], 27)
        w.add_tensor(name, to_f32(a))
        n += 1
    # conv_in 2, conv_out 2, the last table 1, last_time_embedder 4,
    # timestep_scale_multiplier 1, per_channel_statistics 5; a res_x
    # group has 5 per layer (conv1 2, conv2 2, table 1) plus its
    # time_embedder 4; a compress_all block has its conv 2.
    want = 15 + sum(5 * p + 4 if kind == "res_x" else 2
                    for kind, p in zip(kinds, params))
    if n != want:
        sys.exit("ltxv_vae: wrote %d tensors, expected %d" % (n, want))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ltxv-*.safetensors")
    ap.add_argument("--dit-out", required=True)
    ap.add_argument("--vae-out",
                    help="the decoder GGUF; omitted, only the DiT is written")
    a = ap.parse_args()

    hdr, meta, base = read_header(a.ckpt)
    if "config" not in meta:
        sys.exit("no config metadata: not an LTX-Video single file")
    cfg = json.loads(meta["config"])
    mm = np.memmap(a.ckpt, dtype=np.uint8, mode="r")

    dit, vae = [], []
    for name in sorted(hdr):
        if name.startswith(DIT_PREFIX):
            dit.append((name[len(DIT_PREFIX):], load(mm, base, hdr[name])))
        elif name.startswith(VAE_KEEP):
            vae.append((name[len("vae."):], load(mm, base, hdr[name])))
    write_dit(a.dit_out, cfg["transformer"], cfg["scheduler"],
              cfg.get("allowed_inference_steps", []), dit)
    if a.vae_out:
        write_vae(a.vae_out, cfg["vae"], vae)


if __name__ == "__main__":
    main()
