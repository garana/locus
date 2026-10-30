#!/usr/bin/env python3
"""Writes the golden of tests/test_flux_pipeline.cpp (i#372): FluxPipeline
text to image in F32, from a given start latent.

The prompt below, max_sequence_length 64, 128 x 128 pixels (a 16 x 16
latent), 2 steps at guidance 3.5, on a FLUX.1 pipeline directory
(--model; Kontext dev runs as plain text to image when given no image).
Each part runs as FluxPipeline runs it:
- CLIPTokenizer + CLIPTextModel pooled;
- T5TokenizerFast padded to 64 + T5EncoderModel, without a mask;
- the scheduler setup, the transformer at t / 1000 per step and
  scheduler.step;
- the unpack, the VAE's scale and shift, and AutoencoderKL.decode.
The 12B transformer does not fit in 31 GB as F32, so it runs one block
at a time with diffusers' own block classes, as in
make-flux-dit-goldens.py. Writes, into --out (tests/golden/flux):

- pipeline.json: the prompt, the request, both id lists, the sigmas;
- pipeline_in.f32: the start latent [16, 16, 16];
- pipeline_out.f32: the image [3, 128, 128], before the postprocess.

With --kontext, the same as FluxKontextPipeline with a seeded 128 x 128
RGB reference image and auto_resize off (and max_area 128 x 128, so the
size stays): the reference as VaeImageProcessor preprocesses it, its
encode mode scaled and shifted, packed with first id 1 after the image
tokens on every step, the prediction cut to the image tokens; 2 steps
at guidance 2.5. Writes kontext.json, kontext_ref.rgb (the reference,
HWC bytes), kontext_in.f32 and kontext_out.f32.

    python3 -I scripts/make-flux-pipeline-goldens.py \\
        --model DIR/FLUX.1-Kontext-dev --out tests/golden/flux
    python3 -I scripts/make-flux-pipeline-goldens.py --kontext \\
        --model DIR/FLUX.1-Kontext-dev --out tests/golden/flux

Tested with Python 3.12, torch 2.5.1 (CPU), diffusers 0.31.0,
transformers 4.46.3, on black-forest-labs/FLUX.1-Kontext-dev at 24e9ded.
"""
import argparse
import gc
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings, FluxPosEmbed)
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock, FluxTransformerBlock)
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5TokenizerFast)

PROMPT = "a red fox sitting in fresh snow, morning light, photograph"
KONTEXT_PROMPT = "make the scene look like a winter night with stars"
SIDE, STEPS, GUIDANCE, MAX_SEQ = 128, 2, 3.5, 64
KONTEXT_GUIDANCE = 2.5


def reference_rgb():
    """@returns A seeded smooth 128 x 128 RGB pattern, HWC uint8."""
    g = torch.Generator().manual_seed(3724)
    yy, xx = torch.meshgrid(torch.arange(float(SIDE)),
                            torch.arange(float(SIDE)), indexing="ij")
    chans = []
    for c in range(3):
        f = 0.02 + 0.1 * torch.rand(4, generator=g)
        ph = 6.28 * torch.rand(4, generator=g)
        v = sum(torch.sin(f[i] * (xx if i % 2 else yy) + ph[i])
                for i in range(4)) / 4
        chans.append(v)
    img = (torch.stack(chans, dim=-1).clamp(-1, 1) + 1) * 127.5
    return img.round().to(torch.uint8).numpy()


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


def transformer(wts, cfg, x, text, pooled, timestep, guidance, img_ids,
                txt_ids):
    """FluxTransformer2DModel.forward, one module at a time."""
    heads, hd = cfg["num_attention_heads"], cfg["attention_head_dim"]
    inner = heads * hd
    lin = lambda name, i, o: wts.load(torch.nn.Linear(i, o), name + ".")
    h_img = lin("x_embedder", cfg["in_channels"], inner)(x)
    t = timestep * 1000
    if cfg["guidance_embeds"]:
        tte = wts.load(CombinedTimestepGuidanceTextProjEmbeddings(
            inner, cfg["pooled_projection_dim"]), "time_text_embed.")
        temb = tte(t, guidance * 1000, pooled)
    else:
        tte = wts.load(CombinedTimestepTextProjEmbeddings(
            inner, cfg["pooled_projection_dim"]), "time_text_embed.")
        temb = tte(t, pooled)
    h_txt = lin("context_embedder", cfg["joint_attention_dim"], inner)(text)
    rot = FluxPosEmbed(theta=10000, axes_dim=cfg["axes_dims_rope"])(
        torch.cat((txt_ids, img_ids), dim=0))
    for i in range(cfg["num_layers"]):
        blk = wts.load(FluxTransformerBlock(inner, heads, hd),
                       "transformer_blocks.%d." % i)
        h_txt, h_img = blk(hidden_states=h_img, encoder_hidden_states=h_txt,
                           temb=temb, image_rotary_emb=rot)
    h = torch.cat([h_txt, h_img], dim=1)
    for i in range(cfg["num_single_layers"]):
        blk = wts.load(FluxSingleTransformerBlock(inner, heads, hd),
                       "single_transformer_blocks.%d." % i)
        h = blk(hidden_states=h, temb=temb, image_rotary_emb=rot)
    h = h[:, h_txt.shape[1]:]
    norm_out = wts.load(AdaLayerNormContinuous(inner, inner,
                                               elementwise_affine=False,
                                               eps=1e-6), "norm_out.")
    return lin("proj_out", inner, cfg["in_channels"])(norm_out(h, temb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the pipeline directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kontext", action="store_true",
                    help="edit a seeded reference image")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    m = a.model
    prompt = KONTEXT_PROMPT if a.kontext else PROMPT
    guidance_scale = KONTEXT_GUIDANCE if a.kontext else GUIDANCE
    stem = "kontext" if a.kontext else "pipeline"

    clip_tok = CLIPTokenizer.from_pretrained(os.path.join(m, "tokenizer"))
    clip_ids = clip_tok(prompt, padding="max_length", max_length=77,
                        truncation=True).input_ids
    clip = CLIPTextModel.from_pretrained(os.path.join(m, "text_encoder"),
                                         torch_dtype=torch.float32).eval()
    pooled = clip(torch.tensor([clip_ids])).pooler_output
    del clip
    t5_tok = T5TokenizerFast.from_pretrained(os.path.join(m, "tokenizer_2"))
    t5_ids = t5_tok(prompt, padding="max_length", max_length=MAX_SEQ,
                    truncation=True).input_ids
    t5 = T5EncoderModel.from_pretrained(os.path.join(m, "text_encoder_2"),
                                        torch_dtype=torch.float32).eval()
    text = t5(torch.tensor([t5_ids]))[0]
    del t5
    gc.collect()
    print("encoded", flush=True)

    with open(os.path.join(m, "transformer", "config.json")) as f:
        cfg = json.load(f)
    wts = Weights(os.path.join(m, "transformer"))
    lh = SIDE // 8
    g = torch.Generator().manual_seed(3722)
    z = torch.randn(1, 16, lh, lh, generator=g)
    x = FluxPipeline._pack_latents(z, 1, 16, lh, lh)
    img_ids = FluxPipeline._prepare_latent_image_ids(1, lh, lh, "cpu",
                                                     torch.float32)
    txt_ids = torch.zeros(MAX_SEQ, 3)
    vae = AutoencoderKL.from_pretrained(os.path.join(m, "vae"),
                                        torch_dtype=torch.float32).eval()
    ref_tok = ref_ids = None
    if a.kontext:
        # VaeImageProcessor.preprocess: bytes / 255, then 2 x - 1, CHW.
        rgb = reference_rgb()
        ref = torch.from_numpy(rgb).float().div(255).permute(2, 0, 1)[None]
        ref = ref * 2 - 1
        mode = vae.encode(ref).latent_dist.mode()
        rl = (mode - vae.config.shift_factor) * vae.config.scaling_factor
        ref_tok = FluxPipeline._pack_latents(rl, 1, 16, lh, lh)
        # diffusers 0.31 halves the latent size itself.
        ref_ids = FluxPipeline._prepare_latent_image_ids(
            1, lh, lh, "cpu", torch.float32)
        ref_ids[..., 0] = 1
        img_ids_all = torch.cat([img_ids, ref_ids], dim=0)
    else:
        img_ids_all = img_ids
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(m, "scheduler"))
    c = sched.config
    mu = calculate_shift(x.shape[1], c.base_image_seq_len,
                         c.max_image_seq_len, c.base_shift, c.max_shift)
    sched.set_timesteps(sigmas=np.linspace(1.0, 1 / STEPS, STEPS), mu=mu)
    guidance = torch.full([1], guidance_scale, dtype=torch.float32)
    for t in sched.timesteps:
        inp = x if ref_tok is None else torch.cat([x, ref_tok], dim=1)
        v = transformer(wts, cfg, inp, text, pooled,
                        t.expand(1).to(x.dtype) / 1000, guidance,
                        img_ids_all, txt_ids)
        v = v[:, :x.shape[1]]
        x = sched.step(v, t, x, return_dict=False)[0]
        print("step", float(t), flush=True)

    lat = FluxPipeline._unpack_latents(x, SIDE, SIDE, 16)
    lat = lat / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.decode(lat, return_dict=False)[0]

    z[0].numpy().astype("<f4").tofile(os.path.join(a.out, stem + "_in.f32"))
    img[0].numpy().astype("<f4").tofile(
        os.path.join(a.out, stem + "_out.f32"))
    if a.kontext:
        with open(os.path.join(a.out, "kontext_ref.rgb"), "wb") as f:
            f.write(reference_rgb().tobytes())
    with open(os.path.join(a.out, stem + ".json"), "w") as f:
        json.dump({"revision": "24e9ded", "prompt": prompt, "width": SIDE,
                   "height": SIDE, "steps": STEPS,
                   "guidance": guidance_scale,
                   "max_sequence_length": MAX_SEQ, "clip_ids": clip_ids,
                   "t5_ids": t5_ids,
                   "sigmas": [float(s) for s in sched.sigmas]}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
