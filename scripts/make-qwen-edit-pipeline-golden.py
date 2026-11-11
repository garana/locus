#!/usr/bin/env python3
"""Makes tests/golden/qwen_edit_pipeline.f32, the fixture that locus's
image::QwenEditPipeline is compared against end to end (i#616). It
needs the processor/ folder of Qwen/Qwen-Image-Edit at ac7f931 (its
tokenizer and preprocessor files only; no model code), and no model.

diffusers' own QwenImageEditPipeline.__call__ runs on tiny parts, each
filled from the tests' LCG as the tests build them:

- the text encoder: the tiny Qwen2.5-VL of make-qwen-edit-text-golden.py
  with the real tokenizer's vocabulary (151936 rows, so the template's
  real ids index it) and the real image and vision-start ids;
- the MMDiT of make-qwen-image-dit-goldens.py and the VAE of
  make-qwen-vae-goldens.py.

One stand-in keeps it small: calculate_dimensions aims at AREA pixels
instead of 1024 x 1024 (locus's Request::area does the same). The noise
latents are given. The reference is an 80 x 56 LCG image; the prompt
and the negative prompt are PROMPT and NEGATIVE; 2 steps of true CFG
(scale 4). Writes the decoded image (output_type "np"), [h][w][3]
float32, with h and w first as two float32 values.

    run-offline.sh scripts/make-qwen-edit-pipeline-golden.py \\
        --processor DIR

Tested with Python 3.12, torch 2.5.1, transformers 4.56.1, diffusers
0.35.1, numpy 2.1.3 and pillow 11.3.0. With those, the output's sha256
(on an M2 Pro, CPU):

  qwen_edit_pipeline.f32
    ecf5b0646cb28969cf16769c8ff10aabd036befcc7016527400b334deaa49eab
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import torch
from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
                       QwenImageTransformer2DModel)
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as qie
from PIL import Image
from transformers import (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration,
                          Qwen2VLProcessor)

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE.parent / "tests" / "golden"

# Keep in sync with tests/test_qwen_edit_pipeline.cpp.
VOCAB = 151936
IMAGE_ID, START_ID, VIDEO_ID = 151655, 151652, 151656
AREA = 96 * 96
REF_W, REF_H = 80, 56
PROMPT = "Make the sky purple."
NEGATIVE = " "
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


def text_encoder():
    """The tiny Qwen2.5-VL with the real vocabulary."""
    txt = load("make-qwen-edit-text-golden")
    mrope, vis = txt.mrope, txt.vis
    mrope.VOCAB = VOCAB  # gguf_shapes reads it
    lm = txt.filled(mrope.gguf_shapes())
    tower = txt.filled(vis.gguf_shapes(txt.VISION))
    sd = mrope.to_hf(lm)
    for k, v in vis.to_hf(tower, txt.VISION["depth"]).items():
        sd["model.visual." + k] = v
    cfg = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=VOCAB, hidden_size=mrope.E,
                         intermediate_size=mrope.FF,
                         num_hidden_layers=mrope.LAYERS,
                         num_attention_heads=mrope.HEADS,
                         num_key_value_heads=mrope.KV, rms_norm_eps=1e-6,
                         rope_theta=10000.0, max_position_embeddings=128,
                         hidden_act="silu", tie_word_embeddings=False,
                         rope_scaling={"type": "mrope",
                                       "mrope_section": mrope.SECTIONS}),
        vision_config=txt.VISION, image_token_id=IMAGE_ID,
        vision_start_token_id=START_ID, video_token_id=VIDEO_ID,
        vocab_size=VOCAB, tie_word_embeddings=False)
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    cfg._attn_implementation = "eager"
    m = Qwen2_5_VLForConditionalGeneration(cfg).float().eval()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all("inv_freq" in k for k in missing), missing
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processor", type=pathlib.Path, required=True)
    a = ap.parse_args()
    dit_mod = load("make-qwen-image-dit-goldens")
    vae_mod = load("make-qwen-vae-goldens")
    torch.manual_seed(0)
    dit = QwenImageTransformer2DModel(**dit_mod.CFG).float().eval()
    fill(dit, dit_mod, 4242)
    vae = AutoencoderKLQwenImage(**vae_mod.CFG).float().eval()
    fill(vae, vae_mod, 12345)
    proc = Qwen2VLProcessor.from_pretrained(a.processor)
    sched = {k: v for k, v in dit_mod.SCHED.items()
             if not k.startswith("_")}
    pipe = qie.QwenImageEditPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(**sched), vae=vae,
        text_encoder=text_encoder(), tokenizer=proc.tokenizer,
        processor=proc, transformer=dit)
    pipe.set_progress_bar_config(disable=True)
    dims = qie.calculate_dimensions
    qie.calculate_dimensions = lambda area, ratio: dims(AREA, ratio)

    rgb, _ = vae_mod.lcg(777, REF_H * REF_W * 3)
    rgb = np.clip((rgb + 0.25) * 510.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb.reshape(REF_H, REF_W, 3))
    w, h, _ = dims(AREA, REF_W / REF_H)
    s = vae_mod.CFG["z_dim"] * 4
    scale = 2 ** len(vae_mod.CFG["temperal_downsample"])
    n_noise = (h // scale // 2) * (w // scale // 2)
    noise, _ = vae_mod.lcg(31337, n_noise * s)
    noise = torch.from_numpy(noise * np.float32(4.0)).view(1, n_noise, s)
    with torch.no_grad():
        img = pipe(image=image, prompt=PROMPT, negative_prompt=NEGATIVE,
                   true_cfg_scale=CFG_SCALE, num_inference_steps=STEPS,
                   latents=noise, output_type="np").images
    assert img.shape == (1, h, w, 3), (img.shape, h, w)
    blob = np.concatenate([np.array([h, w], dtype=np.float32),
                           np.asarray(img, dtype=np.float32).ravel()])
    path = GOLDEN / "qwen_edit_pipeline.f32"
    path.write_bytes(blob.astype("<f4").tobytes())
    print(path, (h, w), float(np.sqrt(np.mean(np.square(img)))))


if __name__ == "__main__":
    main()
