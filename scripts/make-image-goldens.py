#!/usr/bin/env python3
"""Makes the image-input fixtures under tests/golden/image_input that
locus's decoder and resize are compared against (i#241).

- src.png, src_rgba.png, src_gray.png: a 61 x 37 deterministic
  pattern (gradients plus seeded noise) saved by Pillow (compressed
  deflate, so the inflate path runs), as RGB, RGBA and L.
- src.jpg (quality 90) and src_jpg.rgb: Pillow's own decode of it,
  RGB, row-major. JPEG decoders may differ by a few levels.
- resize_WxH.rgb: Pillow's Image.resize((W, H), BICUBIC) of src.png
  for a downscale, an upscale and a one-axis change; locus must match
  them exactly (same fixed-point algorithm).
- fit32_wide.rgb, fit32_tall.rgb: the Janus reference
  VLMImageProcessor(image_size=32, min_size=4).resize() (resize the
  long side to 32, pad to a square with the mean's gray) of src.png
  and of its transpose, as HWC bytes.

    python3 scripts/make-image-goldens.py --janus SRC

SRC is a checkout of github.com/deepseek-ai/Janus. Tested with Python
3.10, Pillow 12.3.0, numpy 2.2.6, torchvision 0.29.1, Janus commit
1daa72f.
"""
import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

OUT = (pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden" /
       "image_input")
W, H = 61, 37
RESIZES = [(23, 17), (97, 59), (61, 20)]


def pattern():
    """@returns The H x W x 3 uint8 test pattern."""
    y, x = np.mgrid[0:H, 0:W]
    rng = np.random.RandomState(0)
    r = (x * 255 // (W - 1)).astype(np.int32)
    g = (y * 255 // (H - 1)).astype(np.int32)
    b = ((x * 7 + y * 13) % 256).astype(np.int32)
    img = np.stack([r, g, b], axis=-1) + rng.randint(-40, 41, (H, W, 3))
    # A hard edge, so the bicubic lobes overshoot and clip.
    img[:, W // 2:W // 2 + 3] = 255
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--janus", required=True, help="Janus checkout")
    args = ap.parse_args()
    sys.path.insert(0, str(pathlib.Path(args.janus).resolve()))
    from janus.models.image_processing_vlm import VLMImageProcessor  # noqa

    OUT.mkdir(parents=True, exist_ok=True)
    src = Image.fromarray(pattern(), "RGB")
    src.save(OUT / "src.png")
    alpha = Image.fromarray(
        (np.arange(W * H) % 251).astype(np.uint8).reshape(H, W), "L")
    rgba = src.copy()
    rgba.putalpha(alpha)
    rgba.save(OUT / "src_rgba.png")
    src.convert("L").save(OUT / "src_gray.png")
    src.save(OUT / "src.jpg", quality=90)
    jpg = Image.open(OUT / "src.jpg").convert("RGB")
    (OUT / "src_jpg.rgb").write_bytes(np.asarray(jpg).tobytes())
    for w, h in RESIZES:
        r = src.resize((w, h), Image.BICUBIC)
        (OUT / f"resize_{w}x{h}.rgb").write_bytes(np.asarray(r).tobytes())
    proc = VLMImageProcessor(image_size=32, min_size=4,
                             image_mean=(0.5, 0.5, 0.5),
                             image_std=(0.5, 0.5, 0.5))
    for name, im in (("wide", src),
                     ("tall", src.transpose(Image.Transpose.TRANSPOSE))):
        chw = proc.resize(im)
        (OUT / f"fit32_{name}.rgb").write_bytes(
            np.ascontiguousarray(chw.transpose(1, 2, 0)).tobytes())
    print(f"wrote {OUT}: background {proc.background_color}")


if __name__ == "__main__":
    main()
