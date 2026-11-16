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
        --processor DIR [--image WxH] [--out NAME]

--image sets the image's size (both sides multiples of 28; default
84x56); --out the fixture's name under tests/golden (default
qwen_vl_chat.f32). At 252x252 (i#1096) the image has 81 merged tokens,
more than locus's 64-position prefill chunk, so a chunk boundary falls
inside it, and its 18 x 18 patches span several attention windows.

The processor is built from its image processor and tokenizer only
(ImageOnly): transformers 4.56.1's Qwen2VLProcessor also builds a video
processor, which needs torchvision, and the pinned venv has none. The
image path never uses the video processor, and at a multiple of 28 no
resize runs, so the slow image processor gives the fast one's pixels.

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
from transformers import (AutoTokenizer, Qwen2VLImageProcessor,
                          Qwen2VLProcessor)

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE.parents[1] / "tests" / "golden"

# Keep in sync with tests/test_qwen_vl_chat.cpp.
IMG_W, IMG_H = 84, 56
QUESTION = "What is in the picture?"
STEPS = 3


class ImageOnly(Qwen2VLProcessor):
    """Qwen2VLProcessor without its video processor (no torchvision):
    ProcessorMixin's __init__ on the image processor and tokenizer, then
    the token fields Qwen2VLProcessor's __init__ would set."""

    attributes = ["image_processor", "tokenizer"]

    def __init__(self, image_processor, tokenizer, chat_template=None):
        self.video_processor = None
        super(Qwen2VLProcessor, self).__init__(
            image_processor, tokenizer, chat_template=chat_template)
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)


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
    ap.add_argument("--image", default=f"{IMG_W}x{IMG_H}",
                    help="WxH, multiples of 28")
    ap.add_argument("--out", default="qwen_vl_chat.f32")
    a = ap.parse_args()
    img_w, img_h = (int(x) for x in a.image.split("x"))
    if img_w % 28 or img_h % 28 or not img_w or not img_h:
        sys.exit("--image sides must be positive multiples of 28")
    edit = load("make-qwen-edit-pipeline-golden")
    vae_mod = load("make-qwen-vae-goldens")
    torch.manual_seed(0)
    model = edit.text_encoder()
    proc = ImageOnly(Qwen2VLImageProcessor.from_pretrained(a.processor),
                     AutoTokenizer.from_pretrained(a.processor),
                     chat_template=(a.processor / "chat_template.jinja")
                     .read_text())

    rgb, _ = vae_mod.lcg(777, img_h * img_w * 3)
    rgb = np.clip((rgb + 0.25) * 510.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb.reshape(img_h, img_w, 3))
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": QUESTION}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True,
                                    tokenize=False)
    inputs = proc(text=[text], images=[image], return_tensors="pt")
    grid = inputs["image_grid_thw"][0].tolist()
    assert grid == [1, img_h // 14, img_w // 14], grid
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
    path = GOLDEN / a.out
    path.write_bytes(blob.astype("<f4").tobytes())
    print(path, n_prompt, ids, text.encode())


if __name__ == "__main__":
    main()
