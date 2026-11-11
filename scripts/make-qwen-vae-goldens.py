#!/usr/bin/env python3
"""Makes the Qwen-Image VAE fixtures under tests/golden that locus's
image::QwenVae is compared against (i#533). No model needed.

A tiny diffusers AutoencoderKLQwenImage (base_dim 4, z_dim 4, dim_mult
[1, 2, 2], one residual block per level, no encoder attention as in the
real model, a temporal down and up sample, so the first-frame path of
the 3D resamplers is covered), its parameters filled from the test's
LCG in SORTED name order. Writes:

  qwen_vae_synth.gguf  the model in the qwen_image_vae layout (diffusers
                       names, F32, Conv3d kernels as [out][in][kt*kh*kw]);
  qwen_vae_synth.f32   the encoder's mean for a [3][32][32] LCG image
                       ([4][8][8]; two downsamples, so 4 pixels per
                       latent), then the decoder's output for a
                       [4][8][8] LCG latent ([3][32][32]).

    run-offline.sh scripts/make-qwen-vae-goldens.py

Tested with Python 3.12, torch 2.5.1, diffusers 0.35.1, gguf 0.17.1 and
numpy 2.1.3. With those, the outputs' sha256 (on an M2 Pro, CPU):

  qwen_vae_synth.gguf
    d2afdb51abbdcb64a9374aa3ee94d71ffc7312999cb6c5966e34b8d15a44d48d
  qwen_vae_synth.f32
    4a5e5fd07c01db74285ab7bb8b7db15ce6c37fb8d632a573becad9834c60b4e9
"""
import pathlib

import gguf
import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden"

# Keep in sync with tests/test_qwen_vae.cpp.
CFG = dict(base_dim=4, z_dim=4, dim_mult=[1, 2, 2], num_res_blocks=1,
           attn_scales=[], temperal_downsample=[False, True],
           latents_mean=[0.1, -0.2, 0.3, -0.4],
           latents_std=[1.5, 0.5, 2.0, 1.0])
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
    torch.manual_seed(0)
    vae = AutoencoderKLQwenImage(**CFG).float().eval()
    sd = vae.state_dict()
    state = 12345
    for name in sorted(sd):
        vals, state = lcg(state, sd[name].numel())
        sd[name].copy_(torch.from_numpy(vals).view(sd[name].shape))

    w = gguf.GGUFWriter(str(GOLDEN / "qwen_vae_synth.gguf"), "qwen_image_vae")
    k = "qwen_image_vae."
    for key in ("base_dim", "z_dim", "num_res_blocks"):
        w.add_uint32(k + key, CFG[key])
    w.add_array(k + "dim_mult", CFG["dim_mult"])
    # attn_scales is empty, as the real model's, and so left out (the
    # writer cannot type an empty array).
    w.add_array(k + "temperal_downsample",
                [int(b) for b in CFG["temperal_downsample"]])
    w.add_array(k + "latents_mean", [float(x) for x in CFG["latents_mean"]])
    w.add_array(k + "latents_std", [float(x) for x in CFG["latents_std"]])
    for name in sorted(sd):
        a = sd[name].numpy().astype(np.float32)
        if a.ndim == 5:  # GGUF holds at most 4 dims
            a = a.reshape(a.shape[0], a.shape[1], -1)
        w.add_tensor(name, np.ascontiguousarray(a))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    img, _ = lcg(777, 3 * SIZE * SIZE)
    img = torch.from_numpy(img * np.float32(4.0)).view(1, 3, 1, SIZE, SIZE)
    zs = SIZE // 2 ** (len(CFG["dim_mult"]) - 1)
    z, _ = lcg(999, CFG["z_dim"] * zs * zs)
    z = torch.from_numpy(z * np.float32(4.0)).view(1, CFG["z_dim"], 1, zs, zs)
    with torch.no_grad():
        mean = vae.encode(img).latent_dist.mode()
        out = vae.decode(z).sample
    assert mean.shape == (1, CFG["z_dim"], 1, zs, zs), mean.shape
    assert out.shape == (1, 3, 1, SIZE, SIZE), out.shape
    blob = np.concatenate([mean.numpy().ravel(), out.numpy().ravel()])
    path = GOLDEN / "qwen_vae_synth.f32"
    path.write_bytes(blob.astype("<f4").tobytes())
    print(path, float(mean.pow(2).mean().sqrt()),
          float(out.pow(2).mean().sqrt()), float(out.abs().max()))


if __name__ == "__main__":
    main()
