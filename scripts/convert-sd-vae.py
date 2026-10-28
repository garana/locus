#!/usr/bin/env python3
"""Converts the decoder half of a diffusers AutoencoderKL (the Stable
Diffusion VAE that PixArt-alpha uses) to an "autoencoder_kl" GGUF
(i#146), as docs/models/pixart-alpha.md section 7.4 lays out:

- tensors: post_quant_conv.* and decoder.*, with their checkpoint names
  and dtype (F32); the encoder and quant_conv are left out;
- keys from vae/config.json.

Only the plain decoder is supported: UpDecoderBlock2D blocks, SiLU,
the mid-block attention.

Tested with Python 3.12, gguf 0.19.0, safetensors, on
PixArt-alpha/PixArt-XL-2-512x512 at 50f7021:

    python3 -I scripts/convert-sd-vae.py --vae DIR/vae --out sd-vae.gguf
"""

import argparse
import json
import pathlib
import sys

import gguf
from safetensors.numpy import load_file

ARCH = "autoencoder_kl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True,
                    help="directory with config.json and safetensors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vdir = pathlib.Path(a.vae)
    cfg = json.loads((vdir / "config.json").read_text())
    if cfg.get("act_fn") != "silu":
        sys.exit("expected act_fn silu")
    if any(t != "UpDecoderBlock2D" for t in cfg["up_block_types"]):
        sys.exit("expected UpDecoderBlock2D up blocks")
    if not cfg.get("mid_block_add_attention", True):
        sys.exit("expected the mid-block attention")
    chans = cfg["block_out_channels"]
    if any(c % cfg["norm_num_groups"] != 0 for c in chans):
        sys.exit("channels must be multiples of norm_num_groups")

    w = gguf.GGUFWriter(a.out, ARCH)
    w.add_name("sd-vae")
    k = ARCH + "."
    w.add_uint32(k + "latent_channels", cfg["latent_channels"])
    w.add_uint32(k + "out_channels", cfg["out_channels"])
    w.add_array(k + "block_out_channels", [int(c) for c in chans])
    w.add_uint32(k + "layers_per_block", cfg["layers_per_block"])
    w.add_uint32(k + "norm_groups", cfg["norm_num_groups"])
    w.add_float32(k + "norm_epsilon", 1e-6)
    w.add_float32(k + "scaling_factor", cfg["scaling_factor"])
    w.add_bool(k + "mid_block_attention", True)

    files = sorted(vdir.glob("diffusion_pytorch_model*.safetensors"))
    if not files:
        sys.exit("no diffusion_pytorch_model*.safetensors in %s" % vdir)
    n = 0
    for f in files:
        for name, t in sorted(load_file(str(f)).items()):
            if not (name.startswith("decoder.") or
                    name.startswith("post_quant_conv.")):
                continue
            w.add_tensor(name, t)
            n += 1
    if n == 0:
        sys.exit("no decoder tensors")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
