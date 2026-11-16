#!/usr/bin/env python3
"""Makes tests/golden/qwen_vl_chat.f32, the fixture that locus's
Qwen2.5-VL image chat (image/qwen_vl_chat) is compared against
(i#1092). It needs the processor/ folder of Qwen/Qwen-Image-Edit (its
tokenizer, chat template and preprocessor files only; no model code),
and no model.

transformers' Qwen2_5_VLForConditionalGeneration runs the tiny
Qwen2.5-VL of make-qwen-edit-pipeline-golden.py (the real vocabulary,
filled from the tests' LCG). The processor applies the chat template
(add_generation_prompt) to one user turn, an image then QUESTION, and
prepares an IMG_W x IMG_H LCG image; its sides are multiples of 28, so
smart_resize keeps them and no resize runs. Then greedy decoding:
the prompt's last logits, and STEPS more after each greedy token.

Writes float32: the prompt length, STEPS + 1, the STEPS + 1 greedy
ids, then STEPS + 1 rows of logits (vocabulary-wide).

    run-offline.sh scripts/image/make-qwen-vl-chat-golden.py \\
        --processor DIR

Tested with Python 3.12, torch 2.5.1, transformers 4.56.1, numpy 2.1.3
and pillow 11.3.0.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLProcessor

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE.parents[1] / "tests" / "golden"

# Keep in sync with tests/test_qwen_vl_chat.cpp.
IMG_W, IMG_H = 84, 56
QUESTION = "What is in the picture?"
STEPS = 3


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
    ap.add_argument("--processor", type=pathlib.Path, required=True)
    a = ap.parse_args()
    edit = load("make-qwen-edit-pipeline-golden")
    vae_mod = load("make-qwen-vae-goldens")
    torch.manual_seed(0)
    model = edit.text_encoder()
    proc = Qwen2VLProcessor.from_pretrained(a.processor)

    rgb, _ = vae_mod.lcg(777, IMG_H * IMG_W * 3)
    rgb = np.clip((rgb + 0.25) * 510.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb.reshape(IMG_H, IMG_W, 3))
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": QUESTION}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True,
                                    tokenize=False)
    inputs = proc(text=[text], images=[image], return_tensors="pt")
    grid = inputs["image_grid_thw"][0].tolist()
    assert grid == [1, IMG_H // 14, IMG_W // 14], grid
    n_prompt = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=STEPS + 1,
                             do_sample=False, output_logits=True,
                             return_dict_in_generate=True)
    ids = out.sequences[0, n_prompt:].tolist()
    assert len(ids) == STEPS + 1, ids
    logits = torch.stack([x[0] for x in out.logits]).float().numpy()
    blob = np.concatenate([
        np.array([n_prompt, STEPS + 1], dtype=np.float32),
        np.array(ids, dtype=np.float32), logits.ravel()])
    path = GOLDEN / "qwen_vl_chat.f32"
    path.write_bytes(blob.astype("<f4").tobytes())
    print(path, n_prompt, ids, text.encode())


if __name__ == "__main__":
    main()
