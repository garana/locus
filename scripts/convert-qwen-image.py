#!/usr/bin/env python3
"""Converts the diffusers Qwen-Image-Edit pipeline directory
(Qwen/Qwen-Image-Edit, QwenImageEditPipeline) to GGUFs (i#531):

- a "qwen_image" GGUF with the transformer
  (QwenImageTransformer2DModel): the checkpoint names, 2D weights kept
  BF16 or, with --dit-type q8_0, quantized to Q8_0 (about 21 GB instead
  of 40.9 GB), every other tensor F32; the model keys from
  transformer/config.json, the constants the diffusers class fixes
  (RoPE theta 10000 with scale_rope, norm epsilon 1e-6, 256 timestep
  channels, tanh GELU) and the sampler keys from
  scheduler/scheduler_config.json;
- a "qwen_image_vae" GGUF with the VAE (AutoencoderKLQwenImage)
  encoder and decoder, all F32, under diffusers' names. A Conv3d weight
  [out, in, kt, kh, kw] is stored [out, in, kt * kh * kw] (GGUF holds
  at most 4 dims); the RMS_norm gammas as the checkpoint holds them.
  This is the layout image::QwenVae reads (PR#556).

The text encoder (Qwen2.5-VL) is not converted here: llama.cpp's
converter writes its language model and its mmproj.

The checkpoints are BF16, which safetensors.numpy cannot load, so this
reads the safetensors headers and tensor bytes itself. Tensors are
written one at a time (the tensor table first, then each tensor's data
as it is converted), so a 21 GB Q8_0 transformer is never held in RAM.

Run it only in the hash-locked venv, offline:

    .local-dev/vl-env/run-offline.sh scripts/convert-qwen-image.py \\
        --model Qwen-Image-Edit --dit-out qwen-image-edit-q8_0.gguf \\
        --dit-type q8_0 --vae-out qwen-image-vae.gguf

Tested with Python 3.12, gguf 0.17.1, numpy 2.1.3, on
Qwen/Qwen-Image-Edit at ac7f931.
"""

import argparse
import json
import pathlib
import struct
import sys

import gguf
import numpy as np

Q8 = gguf.GGMLQuantizationType.Q8_0
BF16 = gguf.GGMLQuantizationType.BF16


def read_header(path):
    """@returns (header dict without metadata, data offset)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def load_dir(d, prefix):
    """@returns [(name, array)] of every prefix*.safetensors file in d,
    sorted by name, memory-mapped; BF16 as uint16 bits, F16 and F32 as
    floats."""
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
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        sys.exit("a tensor name repeats across the shards")
    return out


def to_f32(a):
    """@returns a as float32 (BF16 bits widened exactly)."""
    if a.dtype == np.uint16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def config(d):
    return json.loads((pathlib.Path(d) / "config.json").read_text())


class Out:
    """A GGUF written tensor by tensor: add() records each tensor's
    type and size and a function that makes its bytes; finish() writes
    the header, the keys and the tensor table, then calls each function
    in turn and writes its result, so only one tensor is in memory at a
    time."""

    def __init__(self, path, arch):
        self.w = gguf.GGUFWriter(str(path), arch)
        self.make = []

    def add(self, name, shape, kind, make):
        """kind: "f32", "bf16" or "q8_0"; shape in torch order."""
        rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        cols = shape[-1]
        if kind == "f32":
            self.w.add_tensor_info(name, shape, np.dtype(np.float32),
                                   4 * rows * cols)
        elif kind == "bf16":
            self.w.add_tensor_info(name, shape, np.dtype(np.uint16),
                                   2 * rows * cols, raw_dtype=BF16)
        else:
            if cols % 32 != 0:
                sys.exit("%s: Q8_0 needs a multiple of 32 columns" % name)
            byte_shape = list(shape[:-1]) + [cols // 32 * 34]
            self.w.add_tensor_info(name, byte_shape, np.dtype(np.uint8),
                                   rows * (cols // 32) * 34, raw_dtype=Q8)
        self.make.append(make)

    def finish(self):
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_ti_data_to_file()
        for make in self.make:
            self.w.write_tensor_data(np.ascontiguousarray(make()))
        self.w.close()


def write_dit(path, mdir, q8):
    t = config(mdir / "transformer")
    s = json.loads((mdir / "scheduler" /
                    "scheduler_config.json").read_text())
    if t.get("_class_name") != "QwenImageTransformer2DModel":
        sys.exit("expected a QwenImageTransformer2DModel")
    if t.get("guidance_embeds"):
        sys.exit("guidance_embeds is not supported")
    heads, hd = t["num_attention_heads"], t["attention_head_dim"]
    axes = [int(x) for x in t["axes_dims_rope"]]
    if sum(axes) != hd or any(x % 2 for x in axes) or len(axes) != 3:
        sys.exit("expected three even RoPE axes summing to the head width")
    patch = t["patch_size"]
    out_ch = t["out_channels"]
    if t["in_channels"] != out_ch * patch * patch:
        sys.exit("expected in_channels == out_channels * patch_size^2")
    if s.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        sys.exit("expected a FlowMatchEulerDiscreteScheduler")
    if s.get("time_shift_type", "exponential") != "exponential":
        sys.exit("expected the exponential time shift")
    if s.get("stochastic_sampling") or s.get("invert_sigmas") or \
            s.get("use_karras_sigmas") or s.get("use_exponential_sigmas") or \
            s.get("use_beta_sigmas"):
        sys.exit("unsupported scheduler option")
    inner = heads * hd

    tensors = load_dir(mdir / "transformer", "diffusion_pytorch_model")
    shapes = {name: tuple(a.shape) for name, a in tensors}
    # The widths the config does not hold, checked against the tensor
    # shapes (torch [out, in]).
    ff = shapes.get("transformer_blocks.0.img_mlp.net.0.proj.weight",
                    (0,))[0]
    if ff != 4 * inner or \
            shapes.get("transformer_blocks.0.txt_mlp.net.0.proj.weight",
                       (0,))[0] != ff:
        sys.exit("expected feed-forward widths of 4 x inner")
    if shapes.get("time_text_embed.timestep_embedder.linear_1.weight") != \
            (inner, 256):
        sys.exit("expected 256 timestep channels")
    if shapes.get("img_in.weight") != (inner, t["in_channels"]) or \
            shapes.get("txt_in.weight") != (inner, t["joint_attention_dim"]):
        sys.exit("unexpected img_in / txt_in shapes")
    if shapes.get("transformer_blocks.0.img_mod.1.weight") != \
            (6 * inner, inner):
        sys.exit("expected six modulation vectors per stream")
    if shapes.get("proj_out.weight") != (patch * patch * out_ch, inner):
        sys.exit("unexpected proj_out shape")

    o = Out(path, "qwen_image")
    w = o.w
    w.add_name(mdir.name)
    k = "qwen_image."
    w.add_uint32(k + "block_count", t["num_layers"])
    w.add_uint32(k + "embedding_length", inner)
    w.add_uint32(k + "feed_forward_length", ff)
    w.add_uint32(k + "attention.head_count", heads)
    w.add_uint32(k + "attention.key_length", hd)
    w.add_float32(k + "norm_epsilon", 1e-6)
    w.add_uint32(k + "in_channels", t["in_channels"])
    w.add_uint32(k + "out_channels", out_ch)
    w.add_uint32(k + "patch_size", patch)
    w.add_uint32(k + "joint_attention_dim", t["joint_attention_dim"])
    w.add_uint32(k + "timestep.channels", 256)
    w.add_float32(k + "rope.theta", 10000.0)
    w.add_array(k + "rope.axes", axes)
    w.add_bool(k + "rope.scale", True)
    w.add_string(k + "activation", "gelu_tanh")
    p = k + "sampler."
    w.add_uint32(p + "num_train_timesteps", s["num_train_timesteps"])
    w.add_bool(p + "use_dynamic_shifting",
               bool(s.get("use_dynamic_shifting", False)))
    # Float64: the pipeline computes the shift in double from these.
    w.add_float64(p + "shift", float(s.get("shift", 1.0)))
    w.add_float64(p + "base_shift", float(s.get("base_shift", 0.5)))
    w.add_float64(p + "max_shift", float(s.get("max_shift", 1.15)))
    w.add_float64(p + "shift_terminal",
                  float(s.get("shift_terminal") or 0.0))
    w.add_uint32(p + "base_image_seq_len", s.get("base_image_seq_len", 256))
    w.add_uint32(p + "max_image_seq_len", s.get("max_image_seq_len", 4096))

    n = 0
    for name, a in tensors:
        if a.ndim == 2 and a.dtype == np.uint16 and q8 and \
                a.shape[1] % 32 == 0:
            o.add(name, a.shape, "q8_0",
                  lambda a=a: gguf.quants.quantize(to_f32(a), Q8))
        elif a.ndim == 2 and a.dtype == np.uint16:
            o.add(name, a.shape, "bf16", lambda a=a: a)
        else:
            o.add(name, a.shape, "f32", lambda a=a: to_f32(a))
        n += 1
    # Per block: the attention 20 (q, k, v, out and their add_ twins, 2
    # each; four q/k norms), img_mlp 4, txt_mlp 4, img_mod 2, txt_mod 2.
    # Outside: img_in 2, txt_in 2, txt_norm 1, the timestep embedder 4,
    # norm_out 2, proj_out 2.
    want = 32 * t["num_layers"] + 13
    if n != want:
        sys.exit("qwen_image: %d tensors, expected %d" % (n, want))
    o.finish()


def write_vae(path, mdir):
    v = config(mdir / "vae")
    if v.get("_class_name") != "AutoencoderKLQwenImage":
        sys.exit("expected an AutoencoderKLQwenImage")
    z = v["z_dim"]
    if len(v["latents_mean"]) != z or len(v["latents_std"]) != z:
        sys.exit("latents_mean / latents_std must have z_dim values")
    if len(v["temperal_downsample"]) != len(v["dim_mult"]) - 1:
        sys.exit("temperal_downsample must have len(dim_mult) - 1 values")

    o = Out(path, "qwen_image_vae")
    w = o.w
    w.add_name("qwen-image-vae")
    k = "qwen_image_vae."
    for key in ("base_dim", "z_dim", "num_res_blocks"):
        w.add_uint32(k + key, v[key])
    w.add_array(k + "dim_mult", [int(x) for x in v["dim_mult"]])
    # The real model's attn_scales is empty and so left out (the writer
    # cannot type an empty array); QwenVae treats a missing key as empty.
    if v.get("attn_scales"):
        w.add_array(k + "attn_scales", [float(x) for x in v["attn_scales"]])
    w.add_array(k + "temperal_downsample",
                [int(bool(x)) for x in v["temperal_downsample"]])
    w.add_array(k + "latents_mean", [float(x) for x in v["latents_mean"]])
    w.add_array(k + "latents_std", [float(x) for x in v["latents_std"]])

    n = 0
    for name, a in load_dir(mdir / "vae", "diffusion_pytorch_model"):
        shape = list(a.shape)
        if len(shape) == 5:  # Conv3d: [out, in, kt * kh * kw]
            shape = [shape[0], shape[1], shape[2] * shape[3] * shape[4]]
        o.add(name, shape, "f32",
              lambda a=a, shape=shape: to_f32(a).reshape(shape))
        n += 1
    if n == 0:
        sys.exit("no VAE tensors")
    o.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="the diffusers pipeline directory")
    ap.add_argument("--dit-out", help="the transformer GGUF")
    ap.add_argument("--dit-type", choices=("bf16", "q8_0"), default="bf16",
                    help="the transformer's 2D weight type (default bf16)")
    ap.add_argument("--vae-out", help="the VAE GGUF")
    a = ap.parse_args()
    if not (a.dit_out or a.vae_out):
        sys.exit("give --dit-out, --vae-out or both")

    mdir = pathlib.Path(a.model)
    idx = json.loads((mdir / "model_index.json").read_text())
    if idx.get("_class_name") not in ("QwenImageEditPipeline",
                                      "QwenImagePipeline"):
        sys.exit("expected a QwenImageEditPipeline or QwenImagePipeline")
    if a.dit_out:
        write_dit(pathlib.Path(a.dit_out), mdir, a.dit_type == "q8_0")
    if a.vae_out:
        write_vae(pathlib.Path(a.vae_out), mdir)


if __name__ == "__main__":
    main()
