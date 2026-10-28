#!/usr/bin/env python3
"""Converts a PixArt-alpha DiT (diffusers PixArtTransformer2DModel) and its
DPM-Solver++ scheduler config to a "pixart" GGUF (i#146), as
docs/models/pixart-alpha.md section 7.3 lays out:

- tensors keep their checkpoint names and dtype (F32), except
  caption_projection.y_embedding, which diffusers does not use;
- the model keys come from transformer/config.json plus the values
  PixArtTransformer2DModel.__init__ derives (base_size, interpolation
  scale); the sampler keys from scheduler/scheduler_config.json.

Only the 256 and 512 models (sample_size up to 64, no size
embedders) are supported.

Tested with Python 3.12, gguf 0.19.0, safetensors, on
PixArt-alpha/PixArt-XL-2-512x512 at 50f7021:

    python3 -I scripts/convert-pixart.py --transformer DIR/transformer \\
        --scheduler DIR/scheduler --out pixart-xl-2-512.gguf
"""

import argparse
import json
import pathlib
import sys

import gguf
from safetensors.numpy import load_file

ARCH = "pixart"
SKIP = {"caption_projection.y_embedding"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer", required=True,
                    help="directory with config.json and safetensors")
    ap.add_argument("--scheduler", required=True,
                    help="directory with scheduler_config.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tdir = pathlib.Path(a.transformer)
    cfg = json.loads((tdir / "config.json").read_text())
    sch = json.loads(
        (pathlib.Path(a.scheduler) / "scheduler_config.json").read_text())
    if cfg.get("norm_type") != "ada_norm_single":
        sys.exit("expected norm_type ada_norm_single")
    if cfg["sample_size"] == 128 or cfg.get("use_additional_conditions"):
        sys.exit("the 1024 model (size embedders) is not supported")
    if cfg.get("activation_fn") != "gelu-approximate":
        sys.exit("expected activation_fn gelu-approximate")
    if not cfg.get("attention_bias") or cfg.get("norm_elementwise_affine"):
        sys.exit("expected attention biases and norms without affine")
    heads, hd = cfg["num_attention_heads"], cfg["attention_head_dim"]
    inner = heads * hd
    if cfg["cross_attention_dim"] != inner:
        sys.exit("expected cross_attention_dim == inner dim")
    if sch.get("algorithm_type") != "dpmsolver++" or \
            sch.get("prediction_type") != "epsilon":
        sys.exit("expected a dpmsolver++ epsilon scheduler")
    interp = cfg.get("interpolation_scale")
    if interp is None:
        interp = max(cfg["sample_size"] // 64, 1)

    w = gguf.GGUFWriter(a.out, ARCH)
    w.add_name("pixart")
    k = ARCH + "."
    w.add_uint32(k + "block_count", cfg["num_layers"])
    w.add_uint32(k + "embedding_length", inner)
    w.add_uint32(k + "feed_forward_length", 4 * inner)
    w.add_uint32(k + "attention.head_count", heads)
    w.add_uint32(k + "attention.key_length", hd)
    w.add_float32(k + "norm_epsilon", cfg["norm_eps"])
    w.add_uint32(k + "in_channels", cfg["in_channels"])
    w.add_uint32(k + "out_channels", cfg["out_channels"])
    w.add_uint32(k + "patch_size", cfg["patch_size"])
    w.add_uint32(k + "sample_size", cfg["sample_size"])
    w.add_uint32(k + "caption_channels", cfg["caption_channels"])
    w.add_uint32(k + "caption_length", 120)
    w.add_uint32(k + "pos_embed.base_size",
                 cfg["sample_size"] // cfg["patch_size"])
    w.add_float32(k + "pos_embed.interpolation_scale", float(interp))
    w.add_uint32(k + "timestep.channels", 256)
    w.add_bool(k + "timestep.flip_sin_to_cos", True)
    w.add_float32(k + "timestep.freq_shift", 0.0)
    w.add_float32(k + "guidance_scale", 4.5)
    s = k + "sampler."
    w.add_uint32(s + "num_train_timesteps", sch["num_train_timesteps"])
    w.add_float32(s + "beta_start", sch["beta_start"])
    w.add_float32(s + "beta_end", sch["beta_end"])
    w.add_string(s + "beta_schedule", sch["beta_schedule"])
    w.add_string(s + "algorithm", sch["algorithm_type"])
    w.add_uint32(s + "solver_order", sch["solver_order"])
    w.add_string(s + "solver_type", sch["solver_type"])
    w.add_string(s + "timestep_spacing", sch["timestep_spacing"])
    w.add_bool(s + "lower_order_final", sch["lower_order_final"])
    w.add_string(s + "final_sigmas_type",
                 sch.get("final_sigmas_type") or "zero")

    files = sorted(tdir.glob("diffusion_pytorch_model*.safetensors"))
    if not files:
        sys.exit("no diffusion_pytorch_model*.safetensors in %s" % tdir)
    n = 0
    for f in files:
        for name, t in sorted(load_file(str(f)).items()):
            if name in SKIP:
                continue
            w.add_tensor(name, t)
            n += 1
    # pos_embed 2, adaln_single 6, caption_projection 4, the final
    # table 1, proj_out 2; per block attn1 8, attn2 8, ff 4, table 1.
    want = 15 + 21 * cfg["num_layers"]
    if n != want:
        sys.exit("wrote %d tensors, expected %d" % (n, want))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
