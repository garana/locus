#!/usr/bin/env python3
"""Makes the fixtures locus's Qwen-Image-Edit reference preprocessing
(image::qwen_edit_reference, i#601) is compared against. No model
needed.

qwen_edit_sizes.i32: for each (w, h) of SIZES, diffusers'
calculate_dimensions(1024 * 1024, w / h) (width, height); then for each
(h, w) of SMART, transformers' Qwen2-VL smart_resize with the
Qwen-Image-Edit processor's limits (height, width). int32.

qwen_edit_image.f32: the 384 x 288 top-left crop of janus_apple.png
through the pipeline's steps: VaeImageProcessor.resize (LANCZOS) and
.preprocess, rows 0, h / 2 and h - 1 of each channel (3 x 3 x width
float32); then that image resized with Pillow BICUBIC to smart_resize's
size, as the slow Qwen2-VL processor does, rows 0, h / 2 and h - 1 of
each channel (as float32 bytes). It also prints how far the pipeline's
fast processor (torchvision) is from that.

    run-offline.sh scripts/make-qwen-edit-image-golden.py [--processor D]

D is Qwen/Qwen-Image-Edit's processor/ folder (at ac7f931), only for
the printed comparison. Tested with Python 3.12, transformers 4.56.1,
diffusers 0.35.1, numpy 2.1.3 and pillow 11.3.0.
"""
import argparse
import pathlib

import numpy as np
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    calculate_dimensions)
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    smart_resize)

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"

# Keep in sync with tests/test_qwen_edit_image.cpp.
SIZES = [(384, 288), (384, 384), (288, 384), (1920, 1080), (1080, 1920),
         (640, 480), (1, 1), (37, 1000), (5000, 3333), (1023, 1025)]
SMART = [(896, 1184), (1024, 1024), (20, 30), (4000, 4000), (14, 2700),
         (55, 57), (1000, 100), (896, 1194), (897, 1203)]
MIN_PIXELS, MAX_PIXELS = 3136, 12845056
CROP = (384, 288)  # width, height


def rows(a):
    """@returns rows 0, h / 2 and h - 1 of each channel of [3][h][w]."""
    h = a.shape[1]
    return np.stack([a[c, r] for c in range(3) for r in (0, h // 2, h - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processor", type=pathlib.Path)
    a = ap.parse_args()

    out = []
    for w, h in SIZES:
        cw, ch, _ = calculate_dimensions(1024 * 1024, w / h)
        out += [cw, ch]
    for h, w in SMART:
        sh, sw = smart_resize(h, w, 28, MIN_PIXELS, MAX_PIXELS)
        out += [sh, sw]
    (GOLDEN / "qwen_edit_sizes.i32").write_bytes(
        np.array(out, "<i4").tobytes())
    print("sizes", out)

    img = Image.open(GOLDEN / "janus_apple.png").convert("RGB")
    img = img.crop((0, 0) + CROP)
    cw, ch, _ = calculate_dimensions(1024 * 1024, CROP[0] / CROP[1])
    proc = VaeImageProcessor(vae_scale_factor=16)
    vae_img = proc.resize(img, ch, cw)
    x = proc.preprocess(vae_img, ch, cw)[0].numpy()  # [3][h][w]
    assert x.shape == (3, ch, cw), x.shape
    th, tw = smart_resize(ch, cw, 28, MIN_PIXELS, MAX_PIXELS)
    text = vae_img.resize((tw, th), Image.BICUBIC)
    t = np.asarray(text).transpose(2, 0, 1).astype(np.float32)
    blob = np.concatenate([rows(x).ravel(), rows(t).ravel()]).astype("<f4")
    (GOLDEN / "qwen_edit_image.f32").write_bytes(blob.tobytes())
    print("vae", (cw, ch), "text", (tw, th), "floats", blob.size)

    if a.processor:
        from transformers import (Qwen2VLImageProcessor,
                                  Qwen2VLImageProcessorFast)
        fast = Qwen2VLImageProcessorFast.from_pretrained(a.processor)
        slow = Qwen2VLImageProcessor.from_pretrained(a.processor)
        pf = fast(images=[vae_img], return_tensors="pt")["pixel_values"]
        ps = slow(images=[vae_img], return_tensors="np")["pixel_values"]
        d = np.abs(pf.numpy() - ps)
        print("fast vs slow processor: max", float(d.max()),
              "rms", float(np.sqrt((d ** 2).mean())),
              "differing", float((d > 1e-6).mean()))


if __name__ == "__main__":
    main()
