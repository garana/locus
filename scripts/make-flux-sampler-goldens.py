#!/usr/bin/env python3
"""Writes the goldens of tests/test_flux_sampler.cpp (i#372).

sampler: diffusers' FlowMatchEulerDiscreteScheduler from a FLUX.1
pipeline's scheduler config, set up as FluxPipeline does
(np.linspace(1, 1 / steps, steps) and calculate_shift), for each
(--model, steps, image tokens) case below; plus FluxPipeline's
_pack_latents and _prepare_latent_image_ids on a seeded [16, 6, 8]
latent. Writes, into --out (tests/golden/flux): sampler.json.

vae: the pipeline's decode (latents / scaling_factor + shift_factor,
then AutoencoderKL.decode in F32) of a seeded [16, 16, 16] latent.
Writes vae_in.f32 (the latent, before the scale and shift) and
vae_out.f32 (the image [3, 128, 128]).

    python3 -I scripts/make-flux-sampler-goldens.py \\
        --model DIR/FLUX.1-Kontext-dev --schnell DIR/FLUX.1-schnell \\
        --out tests/golden/flux

Tested with Python 3.12, torch 2.5.1 (CPU), diffusers 0.31.0, on
black-forest-labs/FLUX.1-Kontext-dev at 24e9ded and FLUX.1-schnell at
741f7c3.
"""
import argparse
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift

CASES = [("kontext", 28, 4096), ("kontext", 28, 1024), ("kontext", 1, 256),
         ("kontext", 50, 64), ("schnell", 4, 4096), ("schnell", 1, 1024)]


def schedule(sched_dir, steps, seq):
    s = FlowMatchEulerDiscreteScheduler.from_pretrained(sched_dir)
    c = s.config
    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu = calculate_shift(seq, c.base_image_seq_len, c.max_image_seq_len,
                         c.base_shift, c.max_shift)
    s.set_timesteps(sigmas=sigmas, mu=mu)
    return [float(x) for x in s.sigmas], [float(x) for x in s.timesteps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="FLUX.1 Kontext dev")
    ap.add_argument("--schnell", required=True, help="FLUX.1 schnell")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    dirs = {"kontext": a.model, "schnell": a.schnell}
    out = {"cases": []}
    for name, steps, seq in CASES:
        sig, ts = schedule(os.path.join(dirs[name], "scheduler"), steps, seq)
        out["cases"].append({"model": name, "steps": steps, "tokens": seq,
                             "sigmas": sig, "timesteps": ts})
    g = torch.Generator().manual_seed(3721)
    lat = torch.randn(1, 16, 6, 8, generator=g)
    out["pack_in"] = [float(x) for x in lat.flatten()]
    out["pack_out"] = [float(x) for x in
                       FluxPipeline._pack_latents(lat, 1, 16, 6, 8).flatten()]
    ids = FluxPipeline._prepare_latent_image_ids(1, 6, 8, "cpu",
                                                torch.float32)
    out["ids"] = [float(x) for x in ids.flatten()]
    with open(os.path.join(a.out, "sampler.json"), "w") as f:
        json.dump(out, f)
        f.write("\n")

    vae = AutoencoderKL.from_pretrained(os.path.join(a.model, "vae"),
                                        torch_dtype=torch.float32).eval()
    z = torch.randn(1, 16, 16, 16, generator=g)
    x = z / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.decode(x, return_dict=False)[0]
    z[0].numpy().astype("<f4").tofile(os.path.join(a.out, "vae_in.f32"))
    img[0].numpy().astype("<f4").tofile(os.path.join(a.out, "vae_out.f32"))


if __name__ == "__main__":
    main()
