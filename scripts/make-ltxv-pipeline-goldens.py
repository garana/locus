#!/usr/bin/env python3
"""Writes the goldens of tests/test_ltxv_pipeline.cpp (i#357).

Runs the reference LTXVideoPipeline (Lightricks/LTX-Video at a pinned
commit) the way the 0.9.6 distilled config runs it: the base pipeline,
8 steps from the checkpoint's scheduler, guidance 1, deterministic
sampling, all in float32 on the CPU. The 0.9.8 2B distilled checkpoint
(--ckpt) and the T5 from a PixArt diffusers directory (--t5-dir, with
text_encoder/ in fp16 and tokenizer/; the same T5-XXL encoder weights
as PixArt-XL-2-1024-MS) are loaded like inference.py's
create_ltx_video_pipeline does, without its bf16 casts. The prompt is
encoded first, as encode_prompt does it, and passed as prompt_embeds,
so T5 and the DiT are never in memory together.

A 256 x 160 clip of 17 frames (a 3 x 5 x 8 latent) at 30 fps. Writes,
into --out (tests/golden/ltxv):

- pipeline.json: the prompt, its T5 token ids, sizes, seed, timesteps;
- pipeline_noise.f32: the start noise as [128, 3, 5, 8] (the
  reference draws it token-major, as [120, 128]);
- pipeline_latent.f32: the latent after the 8 steps, [128, 3, 5, 8];
- pipeline_decode_noise.f32: the decode noise, [128, 3, 5, 8];
- pipeline_frames.u8: frames 0, 8 and 16 of the decoded clip, each
  160 x 256 x 3 row-major RGB bytes.

    python3 -I scripts/make-ltxv-pipeline-goldens.py \\
        --ltx-src LTX-Video --ckpt ltxv-2b-0.9.8-distilled.safetensors \\
        --t5-dir pixart --out tests/golden/ltxv

Tested with Python 3.12, torch 2.5.1 (CPU), diffusers 0.31.0,
transformers 4.46.3, einops, on Lightricks/LTX-Video at 4b2d053.
"""
import argparse
import gc
import json
import os
import pathlib
import sys

PROMPT = "A red fox walks through fresh snow in a quiet pine forest."
WIDTH, HEIGHT, FRAMES, FPS, SEED = 256, 160, 17, 30, 357
DECODE_T, DECODE_S = 0.05, 0.025


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ltx-src", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--t5-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, str(pathlib.Path(a.ltx_src).resolve()))

    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from einops import rearrange
    from transformers import T5EncoderModel, T5Tokenizer
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder)
    from ltx_video.models.autoencoders.vae_encode import vae_decode
    from ltx_video.models.transformers.symmetric_patchifier import (
        SymmetricPatchifier)
    from ltx_video.models.transformers.transformer3d import (
        Transformer3DModel)
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
    from ltx_video.schedulers.rf import RectifiedFlowScheduler
    from safetensors import safe_open

    # T5-XXL in float32 (19 GB) and the DiT (8 GB) do not fit a 32 GB
    # box together: encode the prompt as encode_prompt does (pad to 256,
    # with the mask), free T5, then hand the pipeline prompt_embeds.
    tok = T5Tokenizer.from_pretrained(a.t5_dir, subfolder="tokenizer")
    t5 = T5EncoderModel.from_pretrained(
        a.t5_dir, subfolder="text_encoder", variant="fp16",
        torch_dtype=torch.float32).eval()
    text = tok([PROMPT.strip()], padding="max_length", max_length=256,
               truncation=True, add_special_tokens=True,
               return_tensors="pt")
    with torch.no_grad():
        embeds = t5(text.input_ids,
                    attention_mask=text.attention_mask)[0]
    mask = text.attention_mask
    del t5
    gc.collect()

    with safe_open(a.ckpt, framework="pt") as f:
        allowed = json.loads(f.metadata()["config"]).get(
            "allowed_inference_steps")
    vae = CausalVideoAutoencoder.from_pretrained(a.ckpt).float().eval()
    dit = Transformer3DModel.from_pretrained(a.ckpt).float().eval()
    sched = RectifiedFlowScheduler.from_pretrained(a.ckpt)
    pipe = LTXVideoPipeline(
        transformer=dit, patchifier=SymmetricPatchifier(patch_size=1),
        text_encoder=None, tokenizer=tok, scheduler=sched, vae=vae,
        prompt_enhancer_image_caption_model=None,
        prompt_enhancer_image_caption_processor=None,
        prompt_enhancer_llm_model=None, prompt_enhancer_llm_tokenizer=None,
        allowed_inference_steps=allowed)

    f, h, w = (FRAMES - 1) // 8 + 1, HEIGHT // 32, WIDTH // 32
    # The start noise as prepare_latents draws it, from the same seed.
    noise = randn_tensor((1, f * h * w, 128),
                         generator=torch.Generator().manual_seed(SEED),
                         device=torch.device("cpu"), dtype=torch.float32)
    noise = rearrange(noise, "b (f h w) c -> b c f h w", f=f, h=h, w=w)
    with torch.no_grad():
        latent = pipe(
            prompt=None, prompt_embeds=embeds, prompt_attention_mask=mask,
            # encode_prompt always encodes a negative prompt, which needs
            # T5; at guidance 1 the steps never read it (they take the
            # positive half), so zeros, as the pipeline itself fills in.
            negative_prompt=None,
            negative_prompt_embeds=torch.zeros_like(embeds),
            negative_prompt_attention_mask=torch.zeros_like(mask),
            height=HEIGHT, width=WIDTH,
            num_frames=FRAMES, frame_rate=FPS, num_inference_steps=8,
            guidance_scale=1, stg_scale=0, rescaling_scale=1,
            generator=torch.Generator().manual_seed(SEED),
            output_type="latent", is_video=True,
            vae_per_channel_normalize=True, enhance_prompt=False,
            stochastic_sampling=False, decode_timestep=DECODE_T,
            decode_noise_scale=DECODE_S, mixed_precision=False,
            offload_to_cpu=False, conditioning_items=None,
            media_items=None).images
    if latent.ndim == 3:  # token-major
        latent = rearrange(latent, "b (f h w) c -> b c f h w",
                           f=f, h=h, w=w)
    assert latent.shape == (1, 128, f, h, w), latent.shape

    # The decode, as the pipeline's non-latent branch does it.
    dnoise = torch.randn(latent.shape,
                         generator=torch.Generator().manual_seed(SEED + 1))
    mixed = latent * (1 - DECODE_S) + dnoise * DECODE_S
    with torch.no_grad():
        video = vae_decode(mixed, vae, True, vae_per_channel_normalize=True,
                           timestep=torch.tensor([DECODE_T]))
    video = pipe.image_processor.postprocess(video, output_type="pt")
    assert video.shape == (1, 3, FRAMES, HEIGHT, WIDTH), video.shape
    rgb = (video[0] * 255).round().clamp(0, 255).to(torch.uint8)

    ids = text.input_ids[0][mask[0].bool()].tolist()
    noise[0].numpy().astype("<f4").tofile(
        os.path.join(a.out, "pipeline_noise.f32"))
    latent[0].numpy().astype("<f4").tofile(
        os.path.join(a.out, "pipeline_latent.f32"))
    dnoise[0].numpy().astype("<f4").tofile(
        os.path.join(a.out, "pipeline_decode_noise.f32"))
    with open(os.path.join(a.out, "pipeline_frames.u8"), "wb") as out:
        for fr in (0, 8, 16):
            out.write(rgb[:, fr].permute(1, 2, 0).contiguous().numpy()
                      .tobytes())
    with open(os.path.join(a.out, "pipeline.json"), "w") as out:
        json.dump({"prompt": PROMPT, "ids": ids, "width": WIDTH,
                   "height": HEIGHT, "frames": FRAMES, "fps": FPS,
                   "seed": SEED, "latent": [128, f, h, w],
                   "timesteps": [float(x) for x in sched.timesteps],
                   "decode_timestep": DECODE_T,
                   "decode_noise_scale": DECODE_S,
                   "frame_indices": [0, 8, 16]}, out, indent=1)
        out.write("\n")


if __name__ == "__main__":
    main()
