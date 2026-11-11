#!/usr/bin/env python3
"""Makes tests/golden/qwen_image_vae.f32, the real Qwen-Image VAE's
encode and decode on one small frame, that the GGUF written by
scripts/convert-qwen-image.py is checked against through
image::QwenVae (i#531).

diffusers' AutoencoderKLQwenImage, from the pipeline's vae/ directory
(Qwen/Qwen-Image-Edit at ac7f931), in float32 on the CPU:

  the encoder's mean for a [3][32][32] LCG image ([16][4][4]: three
  downsamples, 8 pixels per latent), then the decoder's output for a
  [16][4][4] LCG latent ([3][32][32]). The LCG is the one in
  tests/test_qwen_vae.cpp, scaled by 4 as there.

    run-offline.sh scripts/make-qwen-image-vae-golden.py \\
        --vae Qwen-Image-Edit/vae

Tested with Python 3.12, torch 2.5.1, diffusers 0.35.1 and numpy
2.1.3. With those, the output's sha256 (on an M4 Max, CPU):

  qwen_image_vae.f32
    17c24221c58339dd7ff9aa44339c89770137ab93066cb5e9df7f539f555167f4
"""
import argparse
import hashlib
import pathlib

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden"
SIZE = 32


def lcg(state, n):
    """n values of the test's LCG. @returns (values, state)."""
    vals = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        vals[i] = (np.float32(state >> 8) / np.float32(16777216.0) -
                   np.float32(0.5)) * np.float32(0.5)
    return vals, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True,
                    help="the pipeline's vae/ directory")
    a = ap.parse_args()
    vae = AutoencoderKLQwenImage.from_pretrained(
        a.vae, torch_dtype=torch.float32).eval()
    z_dim = vae.config.z_dim
    down = 2 ** (len(vae.config.dim_mult) - 1)
    zs = SIZE // down

    img, _ = lcg(777, 3 * SIZE * SIZE)
    img = torch.from_numpy(img * np.float32(4.0)).view(1, 3, 1, SIZE, SIZE)
    z, _ = lcg(999, z_dim * zs * zs)
    z = torch.from_numpy(z * np.float32(4.0)).view(1, z_dim, 1, zs, zs)
    with torch.no_grad():
        mean = vae.encode(img).latent_dist.mode()
        out = vae.decode(z).sample
    assert mean.shape == (1, z_dim, 1, zs, zs), mean.shape
    assert out.shape == (1, 3, 1, SIZE, SIZE), out.shape
    blob = np.concatenate([mean.numpy().ravel(), out.numpy().ravel()])
    data = blob.astype("<f4").tobytes()
    path = GOLDEN / "qwen_image_vae.f32"
    path.write_bytes(data)
    print(path, hashlib.sha256(data).hexdigest(),
          float(mean.pow(2).mean().sqrt()), float(out.pow(2).mean().sqrt()))


if __name__ == "__main__":
    main()
