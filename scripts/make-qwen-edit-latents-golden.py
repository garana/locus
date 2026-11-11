#!/usr/bin/env python3
"""Makes tests/golden/qwen_edit_latents.f32, the fixture that locus's
image::qwen_edit_reference_latents is compared against (i#614). No
model needed.

The synthetic Qwen-Image VAE of make-qwen-vae-goldens.py (the same
LCG-filled AutoencoderKLQwenImage that tests/golden/qwen_vae_synth.gguf
holds) on a [3][32][24] LCG image in [-1, 1], through diffusers'
QwenImageEditPipeline._encode_vae_image (the posterior's mode, then
(z - latents_mean) / latents_std) and _pack_latents (2 x 2 patches into
one token each, channel-major within it). The image is not square, so
the token order (rows, then columns) shows.

Writes, little-endian float32: the packed tokens, (8 / 2) * (6 / 2) = 12
tokens of 4 * 4 = 16 values.

    run-offline.sh scripts/make-qwen-edit-latents-golden.py

Tested with Python 3.12, torch 2.5.1, diffusers 0.35.1 and numpy 2.1.3.
"""
import importlib.util
import pathlib
import sys
import types

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage, QwenImageEditPipeline

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE.parent / "tests" / "golden"

# Keep in sync with tests/test_qwen_edit_image.cpp.
H, W = 32, 24


def load(name):
    """@returns The sibling script `name` as a module (no .pyc)."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"),
                                                  HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    vae_mod = load("make-qwen-vae-goldens")
    torch.manual_seed(0)
    vae = AutoencoderKLQwenImage(**vae_mod.CFG).float().eval()
    sd = vae.state_dict()
    state = 12345
    for name in sorted(sd):
        vals, state = vae_mod.lcg(state, sd[name].numel())
        sd[name].copy_(torch.from_numpy(vals).view(sd[name].shape))

    img, _ = vae_mod.lcg(777, 3 * H * W)
    img = torch.from_numpy(img * np.float32(4.0)).view(1, 3, 1, H, W)
    z_dim = vae_mod.CFG["z_dim"]
    # The pipeline's two steps, called on a stand-in for the pipeline
    # that has only what they read.
    pipe = types.SimpleNamespace(vae=vae, latent_channels=z_dim)
    with torch.no_grad():
        z = QwenImageEditPipeline._encode_vae_image(pipe, img, None)
    _, c, _, zh, zw = z.shape
    tokens = QwenImageEditPipeline._pack_latents(z, 1, c, zh, zw)
    assert tokens.shape == (1, (zh // 2) * (zw // 2), c * 4), tokens.shape
    path = GOLDEN / "qwen_edit_latents.f32"
    path.write_bytes(tokens.numpy().astype("<f4").tobytes())
    print(path, tuple(tokens.shape), float(tokens.pow(2).mean().sqrt()))


if __name__ == "__main__":
    main()
