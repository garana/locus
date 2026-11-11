#!/usr/bin/env python3
"""Makes the Qwen-Image-Edit denoising fixtures under tests/golden that
locus's image::QwenImageSampler and image::qwen_edit_denoise are
compared against (i#615). No model needed.

diffusers' own QwenImageEditPipeline.__call__ runs on the synthetic
MMDiT of scripts/make-qwen-image-dit-goldens.py and the synthetic VAE
of scripts/make-qwen-vae-goldens.py (both rebuilt here from the same
LCG), with no text encoder: the prompt and negative prompt are given
as embeddings. Two stand-ins keep it small: calculate_dimensions
returns the reference image's own size (the real one sizes it to about
1024 x 1024), and the noise latents are given. Writes:

  qwen_edit_sigmas.f32   the scheduler's sigmas (steps + 1 each) for
                         every (steps, image tokens) of SIGMA_CASES;
  qwen_edit_denoise.f32  for 2 steps with true CFG (scale 4): the
                         reference image's packed latents (as
                         prepare_latents makes them), the final packed
                         latents (output_type "latent"), then the
                         decoded image (output_type "np", [h][w][3]).

    run-offline.sh scripts/make-qwen-edit-denoise-golden.py

Tested with Python 3.12, torch 2.5.1, diffusers 0.35.1, numpy 2.1.3.
With those, the outputs' sha256 (on an M2 Pro, CPU):

  qwen_edit_sigmas.f32
    a6fabf6887707025d3b5349986d06d394dfeb4c18fd13d63276e699b5575a3fe
  qwen_edit_denoise.f32
    15acdd07269a5179016560a2ccbc61b561b179423a76b0d075def5dc99e977ae
"""
import importlib.util
import pathlib
import sys

import numpy as np
import torch
from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
                       QwenImageTransformer2DModel)
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as qie
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE.parent / "tests" / "golden"

# Keep in sync with tests/test_qwen_edit_denoise.cpp.
# Not 1 step: shift_terminal then divides 0 by 0 (every sigma NaN).
SIGMA_CASES = [(2, 12), (3, 256), (4, 1024), (20, 4096), (50, 8192),
               (28, 20000)]
REF_H, REF_W = 32, 24  # the reference image, in pixels
OUT_H, OUT_W = 16, 32  # the edit's size, in pixels
N_TXT, N_NEG = 5, 3
STEPS, CFG_SCALE = 2, 4.0


def load(name):
    """Imports scripts/<name>.py (its name has hyphens)."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"),
                                                  HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fill(model, mod, state):
    """The model's parameters from mod's LCG, in sorted name order."""
    sd = model.state_dict()
    for name in sorted(sd):
        vals, state = mod.lcg(state, sd[name].numel())
        sd[name].copy_(torch.from_numpy(vals).view(sd[name].shape))


def scheduler(dit_mod):
    cfg = {k: v for k, v in dit_mod.SCHED.items() if not k.startswith("_")}
    return FlowMatchEulerDiscreteScheduler(**cfg)


def sigmas(dit_mod):
    out = []
    for steps, seq in SIGMA_CASES:
        s = scheduler(dit_mod)
        c = s.config
        mu = qie.calculate_shift(seq, c.base_image_seq_len,
                                 c.max_image_seq_len, c.base_shift,
                                 c.max_shift)
        qie.retrieve_timesteps(s, steps, "cpu",
                               sigmas=np.linspace(1.0, 1 / steps, steps),
                               mu=mu)
        out.append(s.sigmas.numpy())
    return np.concatenate(out)


def main():
    dit_mod = load("make-qwen-image-dit-goldens")
    vae_mod = load("make-qwen-vae-goldens")
    torch.manual_seed(0)
    dit = QwenImageTransformer2DModel(**dit_mod.CFG).float().eval()
    fill(dit, dit_mod, 4242)
    vae = AutoencoderKLQwenImage(**vae_mod.CFG).float().eval()
    fill(vae, vae_mod, 12345)

    path = GOLDEN / "qwen_edit_sigmas.f32"
    path.write_bytes(sigmas(dit_mod).astype("<f4").tobytes())
    print(path)

    pipe = qie.QwenImageEditPipeline(scheduler=scheduler(dit_mod), vae=vae,
                                     text_encoder=None, tokenizer=None,
                                     processor=None, transformer=dit)
    pipe.set_progress_bar_config(disable=True)
    qie.calculate_dimensions = lambda area, ratio: (REF_W, REF_H, None)
    captured = {}
    prepare = pipe.prepare_latents

    def keep(*args, **kwargs):
        lat, img = prepare(*args, **kwargs)
        captured["ref"] = img.clone()
        return lat, img
    pipe.prepare_latents = keep

    rgb, _ = vae_mod.lcg(777, REF_H * REF_W * 3)
    rgb = np.clip((rgb + 0.25) * 510.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb.reshape(REF_H, REF_W, 3))
    s = vae_mod.CFG["z_dim"] * 4
    n_noise = (OUT_H // 8) * (OUT_W // 8)
    noise, _ = vae_mod.lcg(31337, n_noise * s)
    noise = torch.from_numpy(noise * np.float32(4.0)).view(1, n_noise, s)
    width = dit_mod.CFG["joint_attention_dim"]
    pos, _ = vae_mod.lcg(999, N_TXT * width)
    neg, _ = vae_mod.lcg(555, N_NEG * width)
    kw = dict(image=image, height=OUT_H, width=OUT_W,
              prompt_embeds=torch.from_numpy(pos * np.float32(4.0))
              .view(1, N_TXT, width),
              prompt_embeds_mask=torch.ones(1, N_TXT, dtype=torch.long),
              negative_prompt_embeds=torch.from_numpy(neg * np.float32(4.0))
              .view(1, N_NEG, width),
              negative_prompt_embeds_mask=torch.ones(1, N_NEG,
                                                     dtype=torch.long),
              true_cfg_scale=CFG_SCALE, num_inference_steps=STEPS)
    with torch.no_grad():
        lat = pipe(latents=noise.clone(), output_type="latent", **kw).images
        img = pipe(latents=noise.clone(), output_type="np", **kw).images
    assert lat.shape == (1, n_noise, s), lat.shape
    assert img.shape == (1, OUT_H, OUT_W, 3), img.shape
    ref = captured["ref"]
    assert ref.shape == (1, (REF_H // 8) * (REF_W // 8), s), ref.shape
    blob = np.concatenate([ref.numpy().ravel(), lat.numpy().ravel(),
                           np.asarray(img, dtype=np.float32).ravel()])
    path = GOLDEN / "qwen_edit_denoise.f32"
    path.write_bytes(blob.astype("<f4").tobytes())
    print(path, "rms ref %g latents %g image %g" % (
        float(ref.pow(2).mean().sqrt()), float(lat.pow(2).mean().sqrt()),
        float(np.sqrt(np.mean(np.square(img))))))


if __name__ == "__main__":
    main()
