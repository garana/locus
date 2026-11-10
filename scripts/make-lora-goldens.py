#!/usr/bin/env python3
"""Writes the goldens of tests/test_lora_convert.cpp (i#372): FLUX
LoRAs converted by scripts/convert-lora.py, and the change each one
makes to the DiT's weights, from diffusers and peft.

Three seeded LoRAs for the small DiT of tests/golden/flux_synth.gguf
(2 double and 2 single blocks, width 32):

- kohya.safetensors: kohya keys (lora_down, lora_up, alpha) on the
  double blocks (the fused img qkv, an MLP, a modulation, the txt
  projection) and the single blocks (linear2, modulation), F32, rank 4
  with a different alpha per module;
- diffusers.safetensors: diffusers keys (lora_A, lora_B), BF16, rank
  4, on single-block q and proj_mlp, a double-block output, x_embedder
  and norm_out, with no alpha (scale 1);
- ranks.safetensors: kohya modules of ranks 2, 3, 4 and 6, among them
  single_blocks_0_linear1, the fused q, k, v and MLP input of a single
  block.

For the first two, diffusers' FluxLoraLoaderMixin converts the keys,
loads them into a FluxTransformer2DModel of the same config, and peft
merges each LoRA layer; the reference is merged minus base weight. It
does not depend on the base weights, so the model is a fresh one. Each
of those files has one rank: diffusers 0.31 gives every module the
first module's rank as its alpha (get_peft_kwargs), which rescales a
module of another rank by first rank / r; the converter applies
alpha / r as kohya and PEFT define it. diffusers also cannot load the
ranks file at this width (it splits linear1 at FLUX.1's own sizes,
3072 x 3 + 12288), so its reference is alpha / r B A, split by rows,
computed here.

Writes, into --out (tests/golden/lora):

- NAME.gguf: the converted LoRA, for NAME in kohya, diffusers, ranks;
- NAME.json: the changed DiT layers in order, each [name, out, in];
- NAME_delta.f32: those layers' weight changes (out x in, row-major),
  concatenated.

    python3 -I scripts/make-lora-goldens.py \\
        --converter scripts/convert-lora.py \\
        --dit tests/golden/flux_synth.gguf --out tests/golden/lora

Tested with Python 3.12, torch 2.5.1 (CPU), diffusers 0.31.0, peft
0.13.2, safetensors 0.4.5, gguf 0.19.0, numpy 2.1.3.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file, save_file

# The config of tests/golden/flux_synth.gguf (make-flux-dit-goldens.py).
CFG = dict(patch_size=1, in_channels=16, num_layers=2, num_single_layers=2,
           attention_head_dim=16, num_attention_heads=2,
           joint_attention_dim=24, pooled_projection_dim=12,
           guidance_embeds=True, axes_dims_rope=[4, 6, 6])


def rand(g, *shape):
    return 0.1 * torch.randn(*shape, generator=g)


def kohya_lora(g):
    """@returns The kohya state dict: (module, in, out, alpha), rank 4."""
    mods = [
        ("double_blocks_0_img_attn_qkv", 32, 96, 8.0),
        ("double_blocks_0_img_mlp_0", 32, 128, 2.0),
        ("double_blocks_0_txt_mod_lin", 32, 192, 2.0),
        ("double_blocks_1_txt_attn_proj", 32, 32, 1.5),
        ("double_blocks_1_txt_mlp_2", 128, 32, 4.0),
        ("single_blocks_0_modulation_lin", 32, 96, 1.0),
        ("single_blocks_1_linear2", 160, 32, 6.0),
    ]
    sd = {}
    for name, n_in, n_out, alpha in mods:
        sd["lora_unet_%s.lora_down.weight" % name] = rand(g, 4, n_in)
        sd["lora_unet_%s.lora_up.weight" % name] = rand(g, n_out, 4)
        sd["lora_unet_%s.alpha" % name] = torch.tensor(alpha)
    return sd


def diffusers_lora(g):
    """@returns The diffusers state dict, BF16: (layer, in, out), rank 4."""
    mods = [
        ("single_transformer_blocks.0.attn.to_q", 32, 32),
        ("single_transformer_blocks.1.proj_mlp", 32, 128),
        ("transformer_blocks.1.attn.to_out.0", 32, 32),
        ("x_embedder", 16, 32),
        ("norm_out.linear", 32, 64),
    ]
    sd = {}
    r = 4
    for name, n_in, n_out in mods:
        sd["transformer.%s.lora_A.weight" % name] = \
            rand(g, r, n_in).to(torch.bfloat16)
        sd["transformer.%s.lora_B.weight" % name] = \
            rand(g, n_out, r).to(torch.bfloat16)
    return sd


def merged_deltas(path):
    """Loads one LoRA file into a fresh DiT the way FluxPipeline does and
    merges it.

    @returns [(layer, merged minus base weight)], sorted by layer.
    """
    # Seeded: merged minus base rounds by the base's values, so an
    # unseeded base changes the reference's last bits on every run.
    torch.manual_seed(372)
    model = FluxTransformer2DModel(**CFG).eval()
    # A dict, not the path: the loader then takes no Hub code path.
    sd, alphas = FluxPipeline.lora_state_dict(load_file(path),
                                              return_alphas=True)
    FluxPipeline.load_lora_into_transformer(sd, alphas, model)
    out = []
    for name, m in model.named_modules():
        if not isinstance(m, BaseTunerLayer):
            continue
        base = m.get_base_layer().weight.detach().clone()
        m.merge()
        delta = m.get_base_layer().weight.detach() - base
        out.append((name, delta.numpy().astype(np.float32)))
    if not out:
        sys.exit("%s: diffusers loaded no LoRA layer" % path)
    return sorted(out)


def ranks_lora(g):
    """A kohya LoRA whose modules differ in rank, with the fused
    single-block linear1 among them.

    @returns (the state dict, its reference [(layer, delta)]).
    """
    sb, db = "single_transformer_blocks.", "transformer_blocks."
    # (module, in, rank, alpha, [(DiT layer, out rows)]).
    mods = [
        ("single_blocks_0_linear1", 32, 4, 2.0,
         [(sb + "0.attn.to_q", 32), (sb + "0.attn.to_k", 32),
          (sb + "0.attn.to_v", 32), (sb + "0.proj_mlp", 128)]),
        ("double_blocks_0_img_attn_qkv", 32, 2, 3.0,
         [(db + "0.attn.to_q", 32), (db + "0.attn.to_k", 32),
          (db + "0.attn.to_v", 32)]),
        ("double_blocks_1_txt_attn_proj", 32, 3, 1.5,
         [(db + "1.attn.to_add_out", 32)]),
        ("single_blocks_1_linear2", 160, 6, 6.0,
         [(sb + "1.proj_out", 32)]),
    ]
    sd, ref = {}, []
    for name, n_in, r, alpha, parts in mods:
        n_out = sum(n for _, n in parts)
        a, b = rand(g, r, n_in), rand(g, n_out, r)
        sd["lora_unet_%s.lora_down.weight" % name] = a
        sd["lora_unet_%s.lora_up.weight" % name] = b
        sd["lora_unet_%s.alpha" % name] = torch.tensor(alpha)
        full = (alpha / r) * (b.double() @ a.double())
        row = 0
        for layer, n in parts:
            ref.append((layer, full[row:row + n].numpy().astype(np.float32)))
            row += n
    return sd, sorted(ref)


def write(out_dir, name, deltas):
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump([[n, int(d.shape[0]), int(d.shape[1])] for n, d in deltas],
                  f, indent=1)
        f.write("\n")
    with open(os.path.join(out_dir, name + "_delta.f32"), "wb") as f:
        for _, d in deltas:
            f.write(np.ascontiguousarray(d, dtype="<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--converter", required=True)
    ap.add_argument("--dit", required=True, help="the small DiT GGUF")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = torch.Generator().manual_seed(3721)
    ranks_sd, ranks_ref = ranks_lora(g)
    loras = [("kohya", kohya_lora(g), None),
             ("diffusers", diffusers_lora(g), None),
             ("ranks", ranks_sd, ranks_ref)]
    with tempfile.TemporaryDirectory() as d:
        for name, sd, ref in loras:
            path = os.path.join(d, name + ".safetensors")
            save_file({k: v.contiguous() for k, v in sd.items()}, path)
            deltas = ref if ref is not None else merged_deltas(path)
            write(a.out, name, deltas)
            subprocess.run([sys.executable, "-I", a.converter, "--dit", a.dit,
                            "--lora", path,
                            "--out", os.path.join(a.out, name + ".gguf")],
                           check=True)
            print(name, len(deltas), "layers", flush=True)


if __name__ == "__main__":
    main()
