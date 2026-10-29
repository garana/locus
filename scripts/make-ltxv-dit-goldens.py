#!/usr/bin/env python3
"""Writes the goldens of tests/test_ltxv_dit.cpp (i#357).

synth: a tiny LTX-Video Transformer3DModel (2 blocks, 2 heads of 8, so
width 16 and 4 RoPE padding entries; 8 latent channels; caption width
32) with seeded F32 weights, saved as a single-file checkpoint with its
config metadata, converted with scripts/convert-ltxv.py (DiT only),
and its torch output on two batch rows with different noise levels and
caption masks, on a 2 x 3 x 4 latent (24 tokens). Writes, into --out:

- ltxv_synth.gguf: the converted model;
- ltxv_synth.json: sigmas, masks, coordinates and shapes;
- ltxv_synth_in.f32: the latent [2, 8, 2, 3, 4], then the caption
  states [2, 6, 32];
- ltxv_synth_out.f32: the output, as [2, 8, 24] (channel-major).

real: the 2B checkpoint (--ckpt, ltxv-2b-0.9.8-distilled.safetensors)
upcast to float32, one call at sigma 0.9094 on a 2 x 4 x 6 latent (48
tokens) with 7 seeded caption rows (T5 width 4096, RMS about 0.2 like
real T5 states; no T5 needed). Writes, into --out (tests/golden/ltxv):

- dit.json: sigma, coordinates and shapes;
- dit_in.f32: the latent [128, 2, 4, 6], then the caption [7, 4096];
- dit_out.f32: the output as [128, 48].

The reference is the official package at a pinned commit, given as a
source checkout (--ltx-src, the directory holding ltx_video/):

    python3 -I scripts/make-ltxv-dit-goldens.py synth \\
        --ltx-src LTX-Video --out tests/golden
    python3 -I scripts/make-ltxv-dit-goldens.py real \\
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


def synth(ltx_src, out):
    sys.path.insert(0, str(pathlib.Path(ltx_src).resolve()))
    import torch
    from safetensors.torch import save_file
    from ltx_video.models.transformers.transformer3d import (
        Transformer3DModel)

    torch.manual_seed(357)
    model = Transformer3DModel(
        num_attention_heads=2, attention_head_dim=8, in_channels=8,
        out_channels=8, num_layers=2, cross_attention_dim=16,
        attention_bias=True, activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000, norm_elementwise_affine=False,
        norm_eps=1e-6, caption_channels=32, qk_norm="rms_norm",
        standardization_norm="rms_norm",
        positional_embedding_type="rope",
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        timestep_scale_multiplier=1000).eval()
    g = torch.Generator().manual_seed(357)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_((torch.rand(p.shape, generator=g) - 0.5) * 0.5)

    frames, rows, cols = 2, 3, 4
    n = frames * rows * cols
    x = torch.randn(2, 8, frames, rows, cols, generator=g)
    y = torch.randn(2, 6, 32, generator=g)
    sigma = torch.tensor([1.0, 0.725])
    mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    coords = pipeline_coords(frames, rows, cols)  # [3, n]
    grid = coords[None].expand(2, 3, n)
    tokens = x.flatten(2).transpose(1, 2)  # [2, n, 8]
    with torch.no_grad():
        o = model(tokens, grid, encoder_hidden_states=y,
                  encoder_attention_mask=mask, timestep=sigma,
                  return_dict=False)[0]  # [2, n, 8]

    cfg = {"transformer": dict(model.config),
           "scheduler": {"_class_name": "RectifiedFlowScheduler",
                         "num_train_timesteps": 1000,
                         "sampler": "LinearQuadratic"},
           "allowed_inference_steps": [1.0, 0.725]}
    sd = {"model.diffusion_model." + k: v.contiguous()
          for k, v in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "synth.safetensors")
        save_file(sd, ckpt, metadata={"config": json.dumps(cfg)})
        subprocess.run([sys.executable, "-I",
                        str(HERE / "convert-ltxv.py"), "--ckpt", ckpt,
                        "--dit-out", os.path.join(out, "ltxv_synth.gguf")],
                       check=True)
    with open(os.path.join(out, "ltxv_synth_in.f32"), "wb") as f:
        f.write(x.numpy().astype("<f4").tobytes())
        f.write(y.numpy().astype("<f4").tobytes())
    o.transpose(1, 2).contiguous().numpy().astype("<f4").tofile(
        os.path.join(out, "ltxv_synth_out.f32"))
    with open(os.path.join(out, "ltxv_synth.json"), "w") as f:
        json.dump({"sigmas": sigma.tolist(), "masks": mask.tolist(),
                   "coords": coords.tolist(),
                   "latent": list(x.shape), "caption": list(y.shape),
                   "out": [2, 8, n]}, f, indent=1)
        f.write("\n")


def pipeline_coords(frames, rows, cols, fps=25.0):
    """@returns [3, n] coordinates as the pipeline makes them: latent
    (f, h, w) times the VAE scale (8, 32, 32), time shifted for the
    causal first frame and divided by the frame rate."""
    import torch
    f_i, h_i, w_i = torch.meshgrid(torch.arange(frames),
                                   torch.arange(rows),
                                   torch.arange(cols), indexing="ij")
    t_px = (f_i.flatten() * 8 + 1 - 8).clamp(min=0).float() / fps
    return torch.stack([t_px, h_i.flatten().float() * 32,
                        w_i.flatten().float() * 32])


def real(ltx_src, ckpt, out):
    sys.path.insert(0, str(pathlib.Path(ltx_src).resolve()))
    import torch
    from ltx_video.models.transformers.transformer3d import (
        Transformer3DModel)

    model = Transformer3DModel.from_pretrained(ckpt).float().eval()
    g = torch.Generator().manual_seed(357)
    frames, rows, cols = 2, 4, 6
    n = frames * rows * cols
    c = model.config.in_channels
    x = torch.randn(1, c, frames, rows, cols, generator=g)
    y = torch.randn(1, 7, model.config.caption_channels, generator=g) * 0.2
    sigma = 0.9094
    coords = pipeline_coords(frames, rows, cols)
    with torch.no_grad():
        o = model(x.flatten(2).transpose(1, 2), coords[None],
                  encoder_hidden_states=y,
                  encoder_attention_mask=torch.ones(1, 7),
                  timestep=torch.tensor([sigma]), return_dict=False)[0]
    with open(os.path.join(out, "dit_in.f32"), "wb") as f:
        f.write(x.numpy().astype("<f4").tobytes())
        f.write(y.numpy().astype("<f4").tobytes())
    o[0].transpose(0, 1).contiguous().numpy().astype("<f4").tofile(
        os.path.join(out, "dit_out.f32"))
    with open(os.path.join(out, "dit.json"), "w") as f:
        json.dump({"sigma": sigma, "coords": coords.tolist(),
                   "latent": list(x.shape[1:]), "caption": [7, y.shape[2]],
                   "out": [c, n]}, f, indent=1)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "real"])
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
            sys.exit("real needs --ckpt")
        real(a.ltx_src, a.ckpt, a.out)


if __name__ == "__main__":
    main()
