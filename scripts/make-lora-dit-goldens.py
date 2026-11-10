#!/usr/bin/env python3
"""Writes the DiT goldens of tests/test_flux_lora.cpp (i#372): the
output of diffusers' FluxTransformer2DModel with LoRAs merged into its
weights.

The model is the small DiT of tests/golden/flux_synth.gguf (its F32
weights loaded into a FluxTransformer2DModel of the same config), run
on the inputs of tests/golden/flux_synth_in.f32 and flux_synth.json,
as make-flux-dit-goldens.py ran it. A LoRA enters as its weight
changes, tests/golden/lora/NAME_delta.f32 (make-lora-goldens.py, from
diffusers and peft), added at a strength: W' = W + strength * delta.
First the model with no LoRA must reproduce flux_synth_out.f32, which
checks the weight loading.

Writes, into --out (tests/golden/lora):

- dit_CASE.f32: the output, n_img x 16, for each case below;
- dit_cases.json: each case's LoRAs, as [[NAME, strength], ...].

    python3 -I scripts/make-lora-dit-goldens.py \\
        --golden tests/golden --out tests/golden/lora

Tested with Python 3.12, torch 2.5.1 (CPU), diffusers 0.31.0, gguf
0.19.0, numpy 2.1.3.
"""
import argparse
import json
import os
import sys

import gguf
import numpy as np
import torch
from diffusers import FluxTransformer2DModel

# The config of tests/golden/flux_synth.gguf (make-flux-dit-goldens.py).
CFG = dict(patch_size=1, in_channels=16, num_layers=2, num_single_layers=2,
           attention_head_dim=16, num_attention_heads=2,
           joint_attention_dim=24, pooled_projection_dim=12,
           guidance_embeds=True, axes_dims_rope=[4, 6, 6])

# One LoRA at strength 1, one at another strength, and two at once on
# overlapping layers (both change transformer_blocks.0's q, k and v).
CASES = {
    "kohya": [["kohya", 1.0]],
    "diffusers_half": [["diffusers", 0.5]],
    "kohya_ranks": [["kohya", 1.0], ["ranks", 0.75]],
}


def base_model(path):
    """@returns The DiT with the GGUF's weights (all F32 here)."""
    r = gguf.GGUFReader(path)
    sd = {}
    for t in r.tensors:
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            sys.exit("%s: %s is not F32" % (path, t.name))
        # gguf-py gives the numpy shape, outermost first.
        sd[t.name] = torch.from_numpy(np.array(t.data, dtype=np.float32))
    model = FluxTransformer2DModel(**CFG).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        sys.exit("weights do not match the config: missing %s, unexpected %s"
                 % (missing, unexpected))
    return model


def deltas(lora_dir, name):
    """@returns {layer: delta (out x in)} from NAME.json / NAME_delta.f32."""
    with open(os.path.join(lora_dir, name + ".json")) as f:
        layers = json.load(f)
    flat = np.fromfile(os.path.join(lora_dir, name + "_delta.f32"), "<f4")
    out, at = {}, 0
    for layer, rows, cols in layers:
        out[layer] = flat[at:at + rows * cols].reshape(rows, cols)
        at += rows * cols
    if at != flat.size:
        sys.exit("%s: the delta file does not match its json" % name)
    return out


def run(model, golden):
    with open(os.path.join(golden, "flux_synth.json")) as f:
        j = json.load(f)
    n_img, n_txt = j["n_img"], j["n_txt"]
    x = np.fromfile(os.path.join(golden, "flux_synth_in.f32"), "<f4")
    nx, nt = n_img * 16, n_txt * 24
    with torch.no_grad():
        out = model(
            hidden_states=torch.from_numpy(x[:nx].reshape(1, n_img, 16)),
            encoder_hidden_states=torch.from_numpy(
                x[nx:nx + nt].reshape(1, n_txt, 24)),
            pooled_projections=torch.from_numpy(x[nx + nt:].reshape(1, 12)),
            timestep=torch.tensor([j["sigma"]]),
            img_ids=torch.tensor(j["img_ids"]),
            txt_ids=torch.tensor(j["txt_ids"]),
            guidance=torch.tensor([j["guidance"]]),
            return_dict=False)[0]
    return out[0].numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="tests/golden")
    ap.add_argument("--out", required=True, help="tests/golden/lora")
    a = ap.parse_args()
    dit = os.path.join(a.golden, "flux_synth.gguf")

    plain = run(base_model(dit), a.golden)
    ref = np.fromfile(os.path.join(a.golden, "flux_synth_out.f32"), "<f4")
    err = float(np.abs(plain.reshape(-1) - ref).max() / np.abs(ref).max())
    print("no LoRA against flux_synth_out.f32: max error %.2e" % err)
    if err > 1e-5:
        sys.exit("the base model does not reproduce flux_synth_out.f32")

    for case, loras in CASES.items():
        model = base_model(dit)
        params = dict(model.named_parameters())
        for name, strength in loras:
            for layer, d in deltas(a.out, name).items():
                w = params[layer + ".weight"]
                with torch.no_grad():
                    w += strength * torch.from_numpy(d)
        out = run(model, a.golden)
        out.astype("<f4").tofile(os.path.join(a.out, "dit_%s.f32" % case))
        change = float(np.abs(out - plain).max() / np.abs(plain).max())
        print("%s: output moves by %.3f of its peak" % (case, change))
    with open(os.path.join(a.out, "dit_cases.json"), "w") as f:
        json.dump(CASES, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
