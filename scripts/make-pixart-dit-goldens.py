#!/usr/bin/env python3
"""Writes the goldens of tests/test_pixart_dit.cpp (i#146).

synth: a tiny PixArtTransformer2DModel (8 x 8 latent, 16 tokens, 2
blocks, 2 heads of 8, caption width 32) with seeded weights, converted
with scripts/convert-pixart.py, and its torch output on two batch rows
with different timesteps and caption masks. Writes, into --out:

- pixart_synth.gguf: the converted model;
- pixart_synth.json: timesteps, masks and shapes;
- pixart_synth_in.f32: the latent [2, 4, 8, 8], then the caption
  states [2, 6, 32];
- pixart_synth_out.f32: the output [2, 8, 8, 8].

    python3 -I scripts/make-pixart-dit-goldens.py synth --out tests/golden

real: trims the pipeline goldens of scripts/make-pixart-goldens.py
(--goldens DIR, holding index.json) to what the DiT test reads, into
--out (tests/golden/pixart):

- dit.json: the timestep and the valid caption rows per batch row;
- dit_text.f32: T5 rows with mask 1, row 0 (the "" negative) first,
  then row 1 (the prompt), 4096 floats each;
- dit_in.f32: the step-0 latent [4, 64, 64] (both batch rows are the
  same latent);
- dit_out.f32: the DiT output [2, 8, 64, 64].

    python3 -I scripts/make-pixart-dit-goldens.py real \\
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
    from diffusers import PixArtTransformer2DModel

    torch.manual_seed(146)
    model = PixArtTransformer2DModel(
        sample_size=8, num_layers=2, num_attention_heads=2,
        attention_head_dim=8, in_channels=4, out_channels=8, patch_size=2,
        caption_channels=32, cross_attention_dim=16,
        norm_type="ada_norm_single", norm_elementwise_affine=False,
        norm_eps=1e-6, activation_fn="gelu-approximate",
        attention_bias=True, num_embeds_ada_norm=1000).eval()
    g = torch.Generator().manual_seed(146)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_((torch.rand(p.shape, generator=g) - 0.5) * 0.5)
    x = torch.randn(2, 4, 8, 8, generator=g)
    y = torch.randn(2, 6, 32, generator=g)
    t = torch.tensor([999.0, 500.0])
    mask = torch.tensor([[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        o = model(x, encoder_hidden_states=y, encoder_attention_mask=mask,
                  timestep=t, added_cond_kwargs={"resolution": None,
                                                 "aspect_ratio": None},
                  return_dict=False)[0]

    with tempfile.TemporaryDirectory() as tmp:
        tdir = os.path.join(tmp, "transformer")
        sdir = os.path.join(tmp, "scheduler")
        model.save_pretrained(tdir, safe_serialization=True)
        os.makedirs(sdir)
        with open(os.path.join(sdir, "scheduler_config.json"), "w") as f:
            json.dump({"algorithm_type": "dpmsolver++",
                       "prediction_type": "epsilon",
                       "num_train_timesteps": 1000, "beta_start": 0.0001,
                       "beta_end": 0.02, "beta_schedule": "linear",
                       "solver_order": 2, "solver_type": "midpoint",
                       "timestep_spacing": "linspace",
                       "lower_order_final": True}, f)
        subprocess.run([sys.executable, "-I",
                        str(HERE / "convert-pixart.py"),
                        "--transformer", tdir, "--scheduler", sdir,
                        "--out", os.path.join(out, "pixart_synth.gguf")],
                       check=True)
    with open(os.path.join(out, "pixart_synth_in.f32"), "wb") as f:
        f.write(x.numpy().astype("<f4").tobytes())
        f.write(y.numpy().astype("<f4").tobytes())
    o.numpy().astype("<f4").tofile(os.path.join(out, "pixart_synth_out.f32"))
    with open(os.path.join(out, "pixart_synth.json"), "w") as f:
        json.dump({"timesteps": t.tolist(), "masks": mask.tolist(),
                   "latent": [2, 4, 8, 8], "caption": [2, 6, 32],
                   "out": list(o.shape)}, f, indent=1)
        f.write("\n")


def read(d, name, idx):
    e = idx[name]
    dt = "<i8" if e["dtype"] == "int64" else "<f4"
    ext = ".i64" if e["dtype"] == "int64" else ".f32"
    return np.fromfile(os.path.join(d, name + ext), dtype=dt).reshape(
        e["shape"])


def real(gdir, out):
    with open(os.path.join(gdir, "index.json")) as f:
        idx = json.load(f)
    mask = read(gdir, "t5_mask", idx)
    pos = read(gdir, "t5_out", idx)
    neg = read(gdir, "t5_neg_out", idx)
    dit_in = read(gdir, "dit_in", idx)
    dit_t = read(gdir, "dit_t", idx)
    dit_out = read(gdir, "dit_out", idx)
    if not np.array_equal(dit_in[0], dit_in[1]):
        sys.exit("dit_in rows differ")
    if dit_t[0] != dit_t[1]:
        sys.exit("dit_t rows differ")
    n_pos = int(mask.sum())
    if not (mask[:n_pos] == 1).all():
        sys.exit("the mask is not a prefix of ones")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "dit_text.f32"), "wb") as f:
        f.write(neg[:1].astype("<f4").tobytes())  # "" is one token
        f.write(pos[:n_pos].astype("<f4").tobytes())
    dit_in[0].astype("<f4").tofile(os.path.join(out, "dit_in.f32"))
    dit_out.astype("<f4").tofile(os.path.join(out, "dit_out.f32"))
    with open(os.path.join(out, "dit.json"), "w") as f:
        json.dump({"revision": idx["revision"], "timestep": float(dit_t[0]),
                   "rows": [1, n_pos], "width": int(pos.shape[1])}, f,
                  indent=1)
        f.write("\n")


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
