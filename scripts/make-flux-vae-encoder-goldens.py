#!/usr/bin/env python3
"""Writes the golden of the FLUX VAE encoder test in
tests/test_flux_sampler.cpp (i#372): AutoencoderKL.encode's mode, then
(mean - shift_factor) * scaling_factor, as FluxKontextPipeline turns its
reference image into latents, in F32.

The image is a seeded smooth pattern in [-1, 1], [3, 128, 128] (a sum
of sines, so the encoder sees structure rather than noise). Writes, into
--out (tests/golden/flux): vae_enc_in.f32 (the image) and
vae_enc_out.f32 (the latent [16, 16, 16]).

    python3 -I scripts/make-flux-vae-encoder-goldens.py \\
        --vae DIR/FLUX.1-Kontext-dev/vae --out tests/golden/flux

Tested with Python 3.12, torch 2.5.1 (CPU), diffusers 0.31.0, on
black-forest-labs/FLUX.1-Kontext-dev at 24e9ded.
"""
import argparse
import os

import torch
from diffusers import AutoencoderKL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    g = torch.Generator().manual_seed(3723)
    yy, xx = torch.meshgrid(torch.arange(128.0), torch.arange(128.0),
                            indexing="ij")
    chans = []
    for c in range(3):
        f = 0.02 + 0.1 * torch.rand(4, generator=g)
        ph = 6.28 * torch.rand(4, generator=g)
        v = sum(torch.sin(f[i] * (xx if i % 2 else yy) + ph[i])
                for i in range(4)) / 4
        chans.append(v)
    img = torch.stack(chans).clamp(-1, 1)[None]
    vae = AutoencoderKL.from_pretrained(a.vae,
                                        torch_dtype=torch.float32).eval()
    mean = vae.encode(img).latent_dist.mode()
    lat = (mean - vae.config.shift_factor) * vae.config.scaling_factor
    img[0].numpy().astype("<f4").tofile(os.path.join(a.out, "vae_enc_in.f32"))
    lat[0].numpy().astype("<f4").tofile(
        os.path.join(a.out, "vae_enc_out.f32"))


if __name__ == "__main__":
    main()
