#!/usr/bin/env python3
"""Writes the goldens of tests/test_pixart_vae.cpp (i#146).

synth: a tiny diffusers AutoencoderKL (block_out_channels [8, 16], one
layer per block, 4 norm groups, 4 latent channels) with seeded
weights, converted with scripts/convert-sd-vae.py, and its decode of a
seeded [1, 4, 8, 8] latent (divided by the scaling factor first, as
the pipeline does). Writes, into --out:

- vae_synth.gguf: the converted decoder;
- vae_synth_in.f32: the latent [4, 8, 8], before the division;
- vae_synth_out.f32: the decoder output [3, 16, 16], before clamping.

    python3 -I scripts/make-pixart-vae-goldens.py synth --out tests/golden

real: from the pipeline goldens of scripts/make-pixart-goldens.py
(--goldens DIR), the reference image [512, 512, 3] (0..255 floats,
clamped) sampled at every 4th row and column, into
--out/vae_image_s4.f32 ([128, 128, 3]). The test decodes the golden
latent_final and compares those pixels.

    python3 -I scripts/make-pixart-vae-goldens.py real \\
        --goldens DIR/goldens --out tests/golden/pixart

Tested with Python 3.12, torch 2.5.1 (CPU), diffusers 0.31.0, gguf
0.19.0.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def synth(out):
    import torch
    from diffusers import AutoencoderKL

    torch.manual_seed(146)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4,
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(8, 16), layers_per_block=1, act_fn="silu",
        norm_num_groups=4, sample_size=16, scaling_factor=0.18215).eval()
    g = torch.Generator().manual_seed(146)
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_((torch.rand(p.shape, generator=g) - 0.5) * 0.5)
    z = torch.randn(1, 4, 8, 8, generator=g)
    with torch.no_grad():
        y = vae.decode(z / vae.config.scaling_factor,
                       return_dict=False)[0]
    with tempfile.TemporaryDirectory() as tmp:
        vdir = os.path.join(tmp, "vae")
        vae.save_pretrained(vdir, safe_serialization=True)
        subprocess.run([sys.executable, "-I", str(HERE / "convert-sd-vae.py"),
                        "--vae", vdir,
                        "--out", os.path.join(out, "vae_synth.gguf")],
                       check=True)
    z[0].numpy().astype("<f4").tofile(os.path.join(out, "vae_synth_in.f32"))
    y[0].numpy().astype("<f4").tofile(os.path.join(out, "vae_synth_out.f32"))


def real(gdir, out):
    with open(os.path.join(gdir, "index.json")) as f:
        idx = json.load(f)
    shape = idx["image"]["shape"]
    img = np.fromfile(os.path.join(gdir, "image.f32"),
                      dtype="<f4").reshape(shape)
    os.makedirs(out, exist_ok=True)
    np.ascontiguousarray(img[::4, ::4, :]).astype("<f4").tofile(
        os.path.join(out, "vae_image_s4.f32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "real"])
    ap.add_argument("--goldens")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "synth":
        synth(a.out)
    else:
        real(a.goldens, a.out)


if __name__ == "__main__":
    main()
