#!/usr/bin/env python3
"""Writes tests/golden/pixart_sampler.{json,f32}: diffusers'
DPMSolverMultistepScheduler with PixArt-alpha's scheduler config
(docs/models/pixart-alpha.md section 5.1), set to 8 steps, fed seeded
random noise predictions, so tests/test_pixart_sampler.cpp can check
locus's sampler step by step without the DiT (i#146).

The json holds the timesteps and the 9 float32 sigmas. The .f32 file
holds, as [1, 4, 16, 16] float32 arrays in order: the start latent,
then for each of the 8 steps the noise prediction fed in and the
latent that step() returned.

    python3 -I scripts/make-pixart-sampler-goldens.py --out tests/golden

Tested with Python 3.12, torch 2.5.1 (CPU), diffusers 0.31.0.
"""
import argparse
import json
import os

import torch
from diffusers import DPMSolverMultistepScheduler

CONFIG = {"algorithm_type": "dpmsolver++", "beta_end": 0.02,
          "beta_schedule": "linear", "beta_start": 0.0001,
          "dynamic_thresholding_ratio": 0.995, "euler_at_final": False,
          "lambda_min_clipped": -float("inf"), "lower_order_final": True,
          "num_train_timesteps": 1000, "prediction_type": "epsilon",
          "sample_max_value": 1.0, "solver_order": 2,
          "solver_type": "midpoint", "steps_offset": 0,
          "thresholding": False, "timestep_spacing": "linspace",
          "trained_betas": None, "use_karras_sigmas": False,
          "use_lu_lambdas": False, "variance_type": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sch = DPMSolverMultistepScheduler(**CONFIG)
    sch.set_timesteps(8)
    g = torch.Generator().manual_seed(146)
    x = torch.randn(1, 4, 16, 16, generator=g)
    arrays = [x.clone()]
    for t in sch.timesteps:
        eps = torch.randn(1, 4, 16, 16, generator=g)
        x = sch.step(eps, t, x, return_dict=False)[0]
        arrays += [eps, x.clone()]
    with open(os.path.join(a.out, "pixart_sampler.f32"), "wb") as f:
        for v in arrays:
            f.write(v.numpy().astype("<f4").tobytes())
    with open(os.path.join(a.out, "pixart_sampler.json"), "w") as f:
        json.dump({"timesteps": [int(t) for t in sch.timesteps],
                   "sigmas": [float(s) for s in sch.sigmas],
                   "shape": [1, 4, 16, 16]}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
