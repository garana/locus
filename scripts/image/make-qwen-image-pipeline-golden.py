#!/usr/bin/env python3
"""Makes tests/golden/qwen_image_pipeline.f32, the fixture that locus's
image::QwenImagePipeline (Qwen-Image text to image) is compared against
end to end (i#1091). It needs the tokenizer/ folder of Qwen/Qwen-Image
(its tokenizer files only; no model code), and no model.

diffusers' own QwenImagePipeline.__call__ runs on the tiny parts of
make-qwen-edit-pipeline-golden.py, each filled from the tests' LCG as
the tests build them: the tiny Qwen2.5-VL with the real tokenizer's
vocabulary, the synthetic MMDiT and the synthetic VAE. The noise
latents are given. A WIDTH x HEIGHT image of PROMPT against NEGATIVE,
2 steps of true CFG (scale 4). Writes the decoded image (output_type
"np"), [h][w][3] float32, with h and w first as two float32 values.

    run-offline.sh scripts/image/make-qwen-image-pipeline-golden.py \\
        --tokenizer DIR

Tested with Python 3.12, torch 2.5.1, transformers 4.56.1, diffusers
0.35.1 and numpy 2.1.3.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import torch
from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
                       QwenImagePipeline, QwenImageTransformer2DModel)
from transformers import Qwen2Tokenizer

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE.parents[1] / "tests" / "golden"

# Keep in sync with tests/test_qwen_image_pipeline.cpp.
WIDTH, HEIGHT = 96, 80
PROMPT = "A red fox in fresh snow."
NEGATIVE = " "
STEPS, CFG_SCALE = 2, 4.0


def load(name):
    """Imports scripts/image/<name>.py (its name has hyphens)."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"),
                                                  HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=pathlib.Path, required=True)
    a = ap.parse_args()
    edit = load("make-qwen-edit-pipeline-golden")
    dit_mod = load("make-qwen-image-dit-goldens")
    vae_mod = load("make-qwen-vae-goldens")
    torch.manual_seed(0)
    dit = QwenImageTransformer2DModel(**dit_mod.CFG).float().eval()
    edit.fill(dit, dit_mod, 4242)
    vae = AutoencoderKLQwenImage(**vae_mod.CFG).float().eval()
    edit.fill(vae, vae_mod, 12345)
    sched = {k: v for k, v in dit_mod.SCHED.items()
             if not k.startswith("_")}
    pipe = QwenImagePipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(**sched), vae=vae,
        text_encoder=edit.text_encoder(),
        tokenizer=Qwen2Tokenizer.from_pretrained(a.tokenizer),
        transformer=dit)
    pipe.set_progress_bar_config(disable=True)

    s = vae_mod.CFG["z_dim"] * 4
    scale = 2 ** len(vae_mod.CFG["temperal_downsample"])
    n_noise = (HEIGHT // scale // 2) * (WIDTH // scale // 2)
    noise, _ = vae_mod.lcg(31337, n_noise * s)
    noise = torch.from_numpy(noise * np.float32(4.0)).view(1, n_noise, s)
    with torch.no_grad():
        img = pipe(prompt=PROMPT, negative_prompt=NEGATIVE,
                   true_cfg_scale=CFG_SCALE, num_inference_steps=STEPS,
                   width=WIDTH, height=HEIGHT, latents=noise,
                   output_type="np").images
    assert img.shape == (1, HEIGHT, WIDTH, 3), img.shape
    blob = np.concatenate([np.array([HEIGHT, WIDTH], dtype=np.float32),
                           np.asarray(img, dtype=np.float32).ravel()])
    path = GOLDEN / "qwen_image_pipeline.f32"
    path.write_bytes(blob.astype("<f4").tobytes())
    print(path, (HEIGHT, WIDTH), float(np.sqrt(np.mean(np.square(img)))))


if __name__ == "__main__":
    main()
