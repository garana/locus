#!/usr/bin/env python3
"""Writes the goldens of tests/test_ltxv_pipeline.cpp (i#357).

Runs the reference LTXVideoPipeline (Lightricks/LTX-Video at a pinned
commit) the way the 0.9.6 distilled config runs it: the base pipeline,
8 steps from the checkpoint's scheduler, guidance 1, stochastic
sampling, all in float32 on the CPU; then once more with deterministic
sampling (the 0.9.8 multi-scale config's choice). The 0.9.8 2B
distilled checkpoint (--ckpt) and the T5 from a PixArt diffusers
directory (--t5-dir, with
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
- pipeline_latent.f32: the latent after the 8 stochastic steps,
  [128, 3, 5, 8];
- pipeline_step_noise.f32: the noise those 8 steps drew, [8, 128, 3, 5,
  8] (recorded from torch.randn_like during the run);
- pipeline_latent_det.f32: the latent after 8 deterministic steps;
- pipeline_decode_noise.f32: the decode noise, [128, 3, 5, 8];
- pipeline_frames.u8: frames 0, 8 and 16 of the decoded stochastic
  clip, each 160 x 256 x 3 row-major RGB bytes.

    python3 -I scripts/make-ltxv-pipeline-goldens.py \\
        --ltx-src LTX-Video --ckpt ltxv-2b-0.9.8-distilled.safetensors \\
        --t5-dir pixart --out tests/golden/ltxv

With --i2v, image to video instead: the same clip conditioned on a first
frame, as inference.py runs it (one ConditioningItem at frame 0,
strength 1, image_cond_noise_scale 0.15, deterministic steps). The frame
is a seeded 300 x 200 image, center-cropped to the clip's aspect,
resized by Pillow and blurred (3 x 3, sigma 1) as
load_image_to_tensor_with_resize_and_crop does, but without its CRF
step (an H.264 round trip). The VAE encode takes the latent mean (the
reference samples it; i2v.json records the distribution's std). Writes:

- i2v.json: sizes, seed, the noise scale, the encode std;
- i2v_src.rgb: the source image, 200 x 300 x 3 bytes;
- i2v_image.f32: the preprocessed frame, [3, 160, 256] in [-1, 1];
- i2v_noise.f32: the start noise, [128, 3, 5, 8];
- i2v_cond_noise.f32: the 8 steps' frame-0 noise, [8, 128, 5, 8];
- i2v_latent.f32: the latent after the 8 steps, [128, 3, 5, 8].

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
    ap.add_argument("--i2v", action="store_true",
                    help="image to video (see the module doc)")
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
    tokens_to_cfhw = "b (f h w) c -> b c f h w"
    if a.i2v:
        i2v(a, pipe, embeds, mask, noise, f, h, w)
        return

    def run(stochastic):
        """@returns the latent after the 8 steps and, when stochastic,
        the 8 step noises the scheduler drew (rf.py's step calls
        torch.randn_like(sample) on the token-major sample)."""
        drawn = []
        orig = torch.randn_like

        def record(x, *args, **kwargs):
            z = orig(x, *args, **kwargs)
            drawn.append(z.clone())
            return z

        torch.manual_seed(SEED + 2)
        torch.randn_like = record
        try:
            with torch.no_grad():
                out = pipe(
                    prompt=None, prompt_embeds=embeds,
                    prompt_attention_mask=mask,
                    # encode_prompt always encodes a negative prompt,
                    # which needs T5; at guidance 1 the steps never read
                    # it (they take the positive half), so zeros, as the
                    # pipeline itself fills in.
                    negative_prompt=None,
                    negative_prompt_embeds=torch.zeros_like(embeds),
                    negative_prompt_attention_mask=torch.zeros_like(mask),
                    height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                    frame_rate=FPS, num_inference_steps=8,
                    guidance_scale=1, stg_scale=0, rescaling_scale=1,
                    generator=torch.Generator().manual_seed(SEED),
                    output_type="latent", is_video=True,
                    vae_per_channel_normalize=True, enhance_prompt=False,
                    stochastic_sampling=stochastic,
                    decode_timestep=DECODE_T, decode_noise_scale=DECODE_S,
                    mixed_precision=False, offload_to_cpu=False,
                    conditioning_items=None, media_items=None).images
        finally:
            torch.randn_like = orig
        if out.ndim == 3:  # token-major
            out = rearrange(out, tokens_to_cfhw, f=f, h=h, w=w)
        assert out.shape == (1, 128, f, h, w), out.shape
        if not stochastic:
            assert not drawn, len(drawn)
            return out, None
        assert len(drawn) == 8, len(drawn)
        steps = torch.cat([rearrange(z, tokens_to_cfhw, f=f, h=h, w=w)
                           for z in drawn])
        return out, steps

    latent, step_noise = run(True)  # the 0.9.6 single-pass config
    latent_det, _ = run(False)

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
    step_noise.numpy().astype("<f4").tofile(
        os.path.join(a.out, "pipeline_step_noise.f32"))
    latent_det[0].numpy().astype("<f4").tofile(
        os.path.join(a.out, "pipeline_latent_det.f32"))
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


I2V_W, I2V_H, I2V_SEED, COND_NOISE = 300, 200, 3726, 0.15


def preprocess(rgb):
    """load_image_to_tensor_with_resize_and_crop (the default, not
    just_crop) without the CRF step: center crop to the clip's aspect,
    Pillow's default resize (bicubic), then torchvision's gaussian_blur
    (kernel 3, sigma 1), written out here since torchvision is not
    installed: the 1D kernel exp(-0.5 (x / sigma)^2) over x = -1, 0, 1,
    normalized, its outer product, reflect padding, one depthwise
    conv2d, all in float32. @returns [3, HEIGHT, WIDTH] in [-1, 1]."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    image = Image.fromarray(rgb)
    input_width, input_height = image.size
    aspect_ratio_target = WIDTH / HEIGHT
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2
    image = image.crop((x_start, y_start, x_start + new_width,
                        y_start + new_height))
    image = image.resize((WIDTH, HEIGHT))
    t = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    t = t.float().div(255)  # TVF.to_tensor
    x = torch.linspace(-1, 1, 3)
    pdf = torch.exp(-0.5 * (x / 1.0).pow(2))
    k1 = pdf / pdf.sum()
    k2 = torch.mm(k1[:, None], k1[None, :]).expand(3, 1, 3, 3)
    t = F.conv2d(F.pad(t[None], [1, 1, 1, 1], mode="reflect"), k2,
                 groups=3)[0]
    t = t * 255.0
    return t / 127.5 - 1.0


def i2v(a, pipe, embeds, mask, noise, f, h, w):
    """The --i2v goldens (see the module doc)."""
    import torch
    from diffusers.models.autoencoders.vae import (
        DiagonalGaussianDistribution)
    from einops import rearrange
    import ltx_video.pipelines.pipeline_ltx_video as plv

    g = torch.Generator().manual_seed(I2V_SEED)
    yy, xx = torch.meshgrid(torch.arange(float(I2V_H)),
                            torch.arange(float(I2V_W)), indexing="ij")
    chans = []
    for _ in range(3):
        fr = 0.02 + 0.08 * torch.rand(3, generator=g)
        ph = 6.28 * torch.rand(3, generator=g)
        chans.append((torch.sin(fr[0] * xx + ph[0]) +
                      torch.sin(fr[1] * yy + ph[1]) +
                      torch.sin(fr[2] * (xx + yy) + ph[2])) / 3)
    rgb = ((torch.stack(chans, -1) * 0.5 + 0.5) * 255).round().clamp(
        0, 255).to(torch.uint8).numpy()
    image = preprocess(rgb)
    assert image.shape == (3, HEIGHT, WIDTH), image.shape

    # vae_encode samples the latent distribution: take its mean, as
    # locus does, and record how wide the distribution is.
    stds = []
    orig_sample = DiagonalGaussianDistribution.sample

    def mean_sample(self, generator=None):
        stds.append((float(self.std.mean()), float(self.std.max())))
        return self.mean

    drawn = []
    orig_randn = plv.randn_tensor

    def record(*args, **kwargs):
        z = orig_randn(*args, **kwargs)
        drawn.append(z.clone())
        return z

    DiagonalGaussianDistribution.sample = mean_sample
    plv.randn_tensor = record
    try:
        with torch.no_grad():
            out = pipe(
                prompt=None, prompt_embeds=embeds,
                prompt_attention_mask=mask, negative_prompt=None,
                negative_prompt_embeds=torch.zeros_like(embeds),
                negative_prompt_attention_mask=torch.zeros_like(mask),
                height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                frame_rate=FPS, num_inference_steps=8, guidance_scale=1,
                stg_scale=0, rescaling_scale=1,
                generator=torch.Generator().manual_seed(SEED),
                output_type="latent", is_video=True,
                vae_per_channel_normalize=True, enhance_prompt=False,
                stochastic_sampling=False, decode_timestep=DECODE_T,
                decode_noise_scale=DECODE_S, mixed_precision=False,
                offload_to_cpu=False, media_items=None,
                conditioning_items=[plv.ConditioningItem(
                    image[None, :, None], 0, 1.0)],
                image_cond_noise_scale=COND_NOISE).images
    finally:
        DiagonalGaussianDistribution.sample = orig_sample
        plv.randn_tensor = orig_randn
    tok = "b (f h w) c -> b c f h w"
    if out.ndim == 3:
        out = rearrange(out, tok, f=f, h=h, w=w)
    assert out.shape == (1, 128, f, h, w), out.shape
    # The start noise, then one draw per step for the frame-0 tokens.
    assert len(drawn) == 9, len(drawn)
    assert torch.equal(rearrange(drawn[0], tok, f=f, h=h, w=w), noise)
    assert len(stds) == 1, len(stds)
    cond = torch.cat([rearrange(z, tok, f=f, h=h, w=w)[:, :, 0]
                      for z in drawn[1:]])  # [8, 128, h, w]

    with open(os.path.join(a.out, "i2v_src.rgb"), "wb") as fo:
        fo.write(rgb.tobytes())
    image.numpy().astype("<f4").tofile(os.path.join(a.out, "i2v_image.f32"))
    noise[0].numpy().astype("<f4").tofile(
        os.path.join(a.out, "i2v_noise.f32"))
    cond.numpy().astype("<f4").tofile(
        os.path.join(a.out, "i2v_cond_noise.f32"))
    out[0].numpy().astype("<f4").tofile(
        os.path.join(a.out, "i2v_latent.f32"))
    with open(os.path.join(a.out, "i2v.json"), "w") as fo:
        json.dump({"prompt": PROMPT, "width": WIDTH, "height": HEIGHT,
                   "frames": FRAMES, "fps": FPS, "seed": SEED,
                   "source": [I2V_H, I2V_W, 3], "strength": 1.0,
                   "image_cond_noise_scale": COND_NOISE,
                   "latent": [128, f, h, w],
                   "encode_std_mean": stds[0][0],
                   "encode_std_max": stds[0][1]}, fo, indent=1)
        fo.write("\n")


if __name__ == "__main__":
    main()
