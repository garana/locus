#!/usr/bin/env python3
"""Writes the goldens of tests/test_ltxv_vae.cpp (i#357).

synth: a tiny LTX-Video Decoder (16 latent channels, base 8, patch 2,
pixel norm, timestep-conditioned, not causal) with config blocks
[res_x(1), compress_all(2), res_x(2)], so up_blocks run res_x(2) at 16
channels, the 2x upsample to 8 channels, then res_x(1): the reversal is
pinned. Seeded F32 weights and latent statistics, saved as a single-file
checkpoint, converted with scripts/convert-ltxv.py (decoder only), and
its torch output for a 3 x 2 x 3 latent at timestep 0.05. Writes, into
--out:

- ltxv_vae_synth.gguf: the converted decoder;
- ltxv_vae_synth.json: the timestep and shapes;
- ltxv_vae_synth_in.f32: the latent [16, 3, 2, 3] (before the
  per-channel un-normalization);
- ltxv_vae_synth_out.f32: the video [3, 5, 8, 12].

real: the 2B checkpoint's decoder (--ckpt) in float32 on a seeded
2 x 2 x 2 latent at timestep 0.05. Writes, into --out
(tests/golden/ltxv):

- vae.json: the timestep and shapes;
- vae_in.f32: the latent [128, 2, 2, 2];
- vae_out.f32: the video [3, 9, 64, 64].

encode: the 2B checkpoint's encoder (--ckpt) in float32: the latent
mean, normalized per channel, of a seeded 9-frame 64 x 64 video and of
its first frame alone. Writes, into --out (tests/golden/ltxv):
vae_enc.json (the shapes), vae_enc_in.f32 (the video [3, 9, 64, 64],
then the image [3, 1, 64, 64]) and vae_enc_out.f32 (the latents
[128, 2, 2, 2], then [128, 1, 2, 2]).

The reference is the official package at a pinned commit, given as a
source checkout (--ltx-src, the directory holding ltx_video/):

    python3 -I scripts/make-ltxv-vae-goldens.py synth \\
        --ltx-src LTX-Video --out tests/golden
    python3 -I scripts/make-ltxv-vae-goldens.py real \\
        --ltx-src LTX-Video --ckpt ltxv-2b-0.9.8-distilled.safetensors \\
        --out tests/golden/ltxv

Tested with Python 3.12, torch 2.5.1 (CPU), diffusers 0.31.0, einops,
gguf 0.19.0, on Lightricks/LTX-Video at 4b2d053.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
TIMESTEP = 0.05


def decode(decoder, z, std, mean, scale_t, scale_s):
    """@returns the reference decoder's output for the normalized z, as
    vae_decode does with vae_per_channel_normalize."""
    import torch
    zu = z * std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
    _, _, f, h, w = z.shape
    with torch.no_grad():
        return decoder(zu, target_shape=(1, 3, (f - 1) * scale_t + 1,
                                         h * scale_s, w * scale_s),
                       timestep=torch.tensor([TIMESTEP]))


def synth(ltx_src, out):
    sys.path.insert(0, str(pathlib.Path(ltx_src).resolve()))
    import torch
    from safetensors.torch import save_file
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        Decoder)

    blocks = [["res_x", {"num_layers": 1, "inject_noise": False}],
              ["compress_all", {"residual": True, "multiplier": 2}],
              ["res_x", {"num_layers": 2, "inject_noise": False}]]
    torch.manual_seed(357)
    dec = Decoder(dims=3, in_channels=16, out_channels=3,
                  blocks=[tuple(b) for b in blocks], base_channels=8,
                  patch_size=2, norm_layer="pixel_norm", causal=False,
                  timestep_conditioning=True).eval()
    g = torch.Generator().manual_seed(357)
    with torch.no_grad():
        for p in dec.parameters():
            p.copy_((torch.rand(p.shape, generator=g) - 0.5) * 0.5)
        dec.timestep_scale_multiplier.fill_(1000.0)
    std = torch.rand(16, generator=g) + 0.5
    mean = (torch.rand(16, generator=g) - 0.5) * 0.4
    z = torch.randn(1, 16, 3, 2, 3, generator=g)
    o = decode(dec, z, std, mean, 2, 4)

    cfg = {"vae": {"_class_name": "CausalVideoAutoencoder", "dims": 3,
                   "in_channels": 3, "out_channels": 3,
                   "latent_channels": 16, "decoder_blocks": blocks,
                   "decoder_base_channels": 8, "scaling_factor": 1.0,
                   "norm_layer": "pixel_norm", "patch_size": 2,
                   "causal_decoder": False, "timestep_conditioning": True}}
    sd = {"vae.decoder." + k: v.contiguous()
          for k, v in dec.state_dict().items()}
    stats = {"channel": torch.arange(16).float(), "mean-of-means": mean,
             "mean-of-stds": torch.ones(16),
             "mean-of-stds_over_std-of-means": torch.ones(16),
             "std-of-means": std}
    for k, v in stats.items():
        sd["vae.per_channel_statistics." + k] = v.contiguous()
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "synth.safetensors")
        save_file(sd, ckpt, metadata={"config": json.dumps(cfg)})
        subprocess.run([sys.executable, "-I",
                        str(HERE / "convert-ltxv.py"), "--ckpt", ckpt,
                        "--vae-out",
                        os.path.join(out, "ltxv_vae_synth.gguf")],
                       check=True)
    z[0].numpy().astype("<f4").tofile(
        os.path.join(out, "ltxv_vae_synth_in.f32"))
    o[0].numpy().astype("<f4").tofile(
        os.path.join(out, "ltxv_vae_synth_out.f32"))
    with open(os.path.join(out, "ltxv_vae_synth.json"), "w") as f:
        json.dump({"timestep": TIMESTEP, "latent": list(z.shape[1:]),
                   "out": list(o.shape[1:])}, f, indent=1)
        f.write("\n")


def real(ltx_src, ckpt, out):
    sys.path.insert(0, str(pathlib.Path(ltx_src).resolve()))
    import torch
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder)

    vae = CausalVideoAutoencoder.from_pretrained(ckpt).float().eval()
    g = torch.Generator().manual_seed(357)
    z = torch.randn(1, 128, 2, 2, 2, generator=g)
    o = decode(vae.decoder, z, vae.std_of_means, vae.mean_of_means, 8, 32)
    z[0].numpy().astype("<f4").tofile(os.path.join(out, "vae_in.f32"))
    o[0].numpy().astype("<f4").tofile(os.path.join(out, "vae_out.f32"))
    with open(os.path.join(out, "vae.json"), "w") as f:
        json.dump({"timestep": TIMESTEP, "latent": list(z.shape[1:]),
                   "out": list(o.shape[1:])}, f, indent=1)
        f.write("\n")


def encode(ltx_src, ckpt, out):
    """The real encoder (i#372): the latent distribution's mean of a
    seeded smooth video [3, 9, 64, 64] and of its first frame as a still
    image [3, 1, 64, 64], normalized per channel as vae_encode does
    (vae_encode samples the distribution; locus takes its mean)."""
    sys.path.insert(0, str(pathlib.Path(ltx_src).resolve()))
    import torch
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder)

    vae = CausalVideoAutoencoder.from_pretrained(ckpt).float().eval()
    g = torch.Generator().manual_seed(3725)
    t, yy, xx = torch.meshgrid(torch.arange(9.0), torch.arange(64.0),
                               torch.arange(64.0), indexing="ij")
    chans = []
    for c in range(3):
        f = 0.05 + 0.15 * torch.rand(3, generator=g)
        ph = 6.28 * torch.rand(3, generator=g)
        chans.append((torch.sin(f[0] * xx + 0.3 * t + ph[0]) +
                      torch.sin(f[1] * yy - 0.2 * t + ph[1]) +
                      torch.sin(f[2] * (xx + yy) + ph[2])) / 3)
    video = torch.stack(chans)[None]
    cases = {}
    with open(os.path.join(out, "vae_enc_in.f32"), "wb") as fi, \
            open(os.path.join(out, "vae_enc_out.f32"), "wb") as fo:
        for name, x in (("video", video), ("image", video[:, :, :1])):
            with torch.no_grad():
                m = vae.encoder(x)[:, :128]
            z = (m - vae.mean_of_means.view(1, -1, 1, 1, 1)) / \
                vae.std_of_means.view(1, -1, 1, 1, 1)
            fi.write(x[0].numpy().astype("<f4").tobytes())
            fo.write(z[0].numpy().astype("<f4").tobytes())
            cases[name] = {"in": list(x.shape[1:]), "out": list(z.shape[1:])}
    with open(os.path.join(out, "vae_enc.json"), "w") as f:
        json.dump(cases, f, indent=1)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "real", "encode"])
    ap.add_argument("--ltx-src", required=True,
                    help="a Lightricks/LTX-Video checkout")
    ap.add_argument("--ckpt", help="real: the 2B single-file checkpoint")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "synth":
        synth(a.ltx_src, a.out)
    else:
        if not a.ckpt:
            sys.exit(a.mode + " needs --ckpt")
        (real if a.mode == "real" else encode)(a.ltx_src, a.ckpt, a.out)


if __name__ == "__main__":
    main()
