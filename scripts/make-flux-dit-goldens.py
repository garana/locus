#!/usr/bin/env python3
"""Writes the goldens of tests/test_flux_dit.cpp (i#372): diffusers'
FluxTransformer2DModel in F32.

synth: a tiny FluxTransformer2DModel (2 double and 2 single blocks, 2
heads of 16, RoPE axes 4/6/6, 16 packed channels, T5 width 24, pooled
width 12, a guidance embedding) with seeded F32 weights, saved as a
diffusers FluxPipeline directory, converted with scripts/convert-flux.py
(--dit-out), and run on 12 image tokens (ids (0, r, c), a 3 x 4 grid), 4
Kontext-style reference tokens (ids (1, r, c), 2 x 2) and 5 text tokens
at sigma 0.7, guidance 3.5. Writes, into --out (tests/golden):

- flux_synth.gguf: the converted model;
- flux_synth.json: sigma, guidance, the ids and the shapes;
- flux_synth_in.f32: x [16, 16], text [5, 24], pooled [12];
- flux_synth_out.f32: the output [16, 16].

real: a FLUX.1 transformer directory (--model, BF16 upcast to F32), run
one block at a time (the 12B model does not fit in 31 GB as F32) with
diffusers' own block classes, on 64 image tokens (8 x 8), 16 reference
tokens (4 x 4, first id 1) and 16 seeded text rows (RMS about 0.2, like
real T5 states), at sigma 0.9, guidance 2.5. Writes, into --out
(tests/golden/flux):

- dit.json: sigma, guidance, the ids and the shapes;
- dit_in.f32: x [80, 64], text [16, 4096], pooled [768];
- dit_out.f32: the output [80, 64];
- dit_trace.f32: the image rows [80, 3072] and text rows [16, 3072]
  after double block 0, then the joined rows [96, 3072] after single
  block 0.

    python3 -I scripts/make-flux-dit-goldens.py synth \\
        --converter scripts/convert-flux.py --out tests/golden
    python3 -I scripts/make-flux-dit-goldens.py real \\
        --model DIR/transformer --out tests/golden/flux

Tested with Python 3.12, torch 2.5.1 (CPU), diffusers 0.31.0,
safetensors, gguf 0.19.0, on black-forest-labs/FLUX.1-Kontext-dev at
24e9ded.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from diffusers import FluxTransformer2DModel
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings, FluxPosEmbed)
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock, FluxTransformerBlock)


def grid_ids(first, rows, cols):
    """(first, r, c) for each cell of a rows x cols grid, row-major."""
    return [[float(first), float(r), float(c)]
            for r in range(rows) for c in range(cols)]


def write_f32(path, *arrays):
    with open(path, "wb") as f:
        for a in arrays:
            f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())


def synth(converter, out_dir):
    cfg = dict(patch_size=1, in_channels=16, num_layers=2,
               num_single_layers=2, attention_head_dim=16,
               num_attention_heads=2, joint_attention_dim=24,
               pooled_projection_dim=12, guidance_embeds=True,
               axes_dims_rope=[4, 6, 6])
    torch.manual_seed(372)
    model = FluxTransformer2DModel(**cfg).eval()
    for name, p in model.named_parameters():
        if name.endswith("norm_q.weight") or name.endswith("norm_k.weight") \
                or "norm_added_" in name:
            p.data = 1.0 + 0.1 * torch.randn_like(p)
        else:
            p.data = 0.08 * torch.randn_like(p)
    img_ids = grid_ids(0, 3, 4) + grid_ids(1, 2, 2)
    txt_ids = [[0.0, 0.0, 0.0]] * 5
    x = torch.randn(1, 16, 16)
    text = torch.randn(1, 5, 24)
    pooled = torch.randn(1, 12)
    sigma, guidance = 0.7, 3.5
    out = model(hidden_states=x, encoder_hidden_states=text,
                pooled_projections=pooled,
                timestep=torch.tensor([sigma]),
                img_ids=torch.tensor(img_ids),
                txt_ids=torch.tensor(txt_ids),
                guidance=torch.tensor([guidance]),
                return_dict=False)[0]
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "transformer"))
        os.makedirs(os.path.join(d, "scheduler"))
        with open(os.path.join(d, "model_index.json"), "w") as f:
            json.dump({"_class_name": "FluxPipeline"}, f)
        with open(os.path.join(d, "transformer", "config.json"), "w") as f:
            json.dump(dict(cfg, _class_name="FluxTransformer2DModel"), f)
        with open(os.path.join(d, "scheduler",
                               "scheduler_config.json"), "w") as f:
            json.dump({"_class_name": "FlowMatchEulerDiscreteScheduler",
                       "num_train_timesteps": 1000, "shift": 3.0,
                       "use_dynamic_shifting": True, "base_shift": 0.5,
                       "max_shift": 1.15, "base_image_seq_len": 256,
                       "max_image_seq_len": 4096}, f)
        sd = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(d, "transformer",
                                   "diffusion_pytorch_model.safetensors"))
        subprocess.run([sys.executable, "-I", converter, "--model", d,
                        "--dit-out",
                        os.path.join(out_dir, "flux_synth.gguf")],
                       check=True)
    write_f32(os.path.join(out_dir, "flux_synth_in.f32"),
              x[0].numpy(), text[0].numpy(), pooled[0].numpy())
    write_f32(os.path.join(out_dir, "flux_synth_out.f32"), out[0].numpy())
    with open(os.path.join(out_dir, "flux_synth.json"), "w") as f:
        json.dump({"sigma": sigma, "guidance": guidance, "n_img": 16,
                   "n_txt": 5, "img_ids": img_ids, "txt_ids": txt_ids}, f,
                  indent=1)
        f.write("\n")


class Weights:
    """The transformer's tensors across its shards, as F32 torch."""

    def __init__(self, model_dir):
        index = os.path.join(model_dir,
                             "diffusion_pytorch_model.safetensors.index.json")
        with open(index) as f:
            self.where = json.load(f)["weight_map"]
        self.dir = model_dir

    def load(self, module, prefix):
        sd = {}
        for name in module.state_dict():
            full = prefix + name
            path = os.path.join(self.dir, self.where[full])
            with safe_open(path, "pt") as f:
                sd[name] = f.get_tensor(full).float()
        module.load_state_dict(sd)
        return module.eval()


def real(model_dir, out_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    heads, hd = cfg["num_attention_heads"], cfg["attention_head_dim"]
    inner = heads * hd
    wts = Weights(model_dir)
    g = torch.Generator().manual_seed(3720)
    img_ids = grid_ids(0, 8, 8) + grid_ids(1, 4, 4)
    n_img, n_txt = len(img_ids), 16
    txt_ids = [[0.0, 0.0, 0.0]] * n_txt
    x = torch.randn(1, n_img, cfg["in_channels"], generator=g)
    text = 0.2 * torch.randn(1, n_txt, cfg["joint_attention_dim"],
                             generator=g)
    pooled = torch.randn(1, cfg["pooled_projection_dim"], generator=g)
    sigma, guidance = 0.9, 2.5

    # FluxTransformer2DModel.forward, one module at a time.
    lin = lambda name, i, o: wts.load(torch.nn.Linear(i, o), name + ".")
    h_img = lin("x_embedder", cfg["in_channels"], inner)(x)
    if cfg["guidance_embeds"]:
        tte = wts.load(CombinedTimestepGuidanceTextProjEmbeddings(
            inner, cfg["pooled_projection_dim"]), "time_text_embed.")
        temb = tte(torch.tensor([sigma * 1000.0]),
                   torch.tensor([guidance * 1000.0]), pooled)
    else:
        tte = wts.load(CombinedTimestepTextProjEmbeddings(
            inner, cfg["pooled_projection_dim"]), "time_text_embed.")
        temb = tte(torch.tensor([sigma * 1000.0]), pooled)
    h_txt = lin("context_embedder", cfg["joint_attention_dim"], inner)(text)
    ids = torch.cat((torch.tensor(txt_ids), torch.tensor(img_ids)), dim=0)
    rot = FluxPosEmbed(theta=10000, axes_dim=cfg["axes_dims_rope"])(ids)
    trace = []
    for i in range(cfg["num_layers"]):
        blk = wts.load(FluxTransformerBlock(inner, heads, hd),
                       "transformer_blocks.%d." % i)
        h_txt, h_img = blk(hidden_states=h_img, encoder_hidden_states=h_txt,
                           temb=temb, image_rotary_emb=rot)
        if i == 0:
            trace += [h_img[0].numpy().copy(), h_txt[0].numpy().copy()]
        print("double", i, flush=True)
    h = torch.cat([h_txt, h_img], dim=1)
    for i in range(cfg["num_single_layers"]):
        blk = wts.load(FluxSingleTransformerBlock(inner, heads, hd),
                       "single_transformer_blocks.%d." % i)
        h = blk(hidden_states=h, temb=temb, image_rotary_emb=rot)
        if i == 0:
            trace.append(h[0].numpy().copy())
        print("single", i, flush=True)
    h = h[:, n_txt:]
    norm_out = wts.load(AdaLayerNormContinuous(inner, inner,
                                               elementwise_affine=False,
                                               eps=1e-6), "norm_out.")
    out = lin("proj_out", inner, cfg["in_channels"])(norm_out(h, temb))

    write_f32(os.path.join(out_dir, "dit_in.f32"), x[0].numpy(),
              text[0].numpy(), pooled[0].numpy())
    write_f32(os.path.join(out_dir, "dit_out.f32"), out[0].numpy())
    write_f32(os.path.join(out_dir, "dit_trace.f32"), *trace)
    with open(os.path.join(out_dir, "dit.json"), "w") as f:
        json.dump({"revision": "24e9ded", "sigma": sigma,
                   "guidance": guidance, "n_img": n_img, "n_txt": n_txt,
                   "img_ids": img_ids, "txt_ids": txt_ids}, f, indent=1)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "real"])
    ap.add_argument("--converter", help="scripts/convert-flux.py")
    ap.add_argument("--model", help="the transformer directory")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    if a.mode == "synth":
        synth(a.converter, a.out)
    else:
        real(a.model, a.out)


if __name__ == "__main__":
    main()
