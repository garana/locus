#!/usr/bin/env python3
"""Writes lanczos_WxH.rgb into tests/golden/image_input: Pillow's
Image.resize((W, H), LANCZOS) of src.png (from make-image-goldens.py),
for a downscale, an upscale and a one-axis change. locus's
resize_lanczos must match them exactly (i#372: VaeImageProcessor's
resample for the FLUX Kontext reference image).

    python3 -I scripts/make-lanczos-goldens.py --dir tests/golden/image_input

Tested with Python 3.12, Pillow 13.0.0, numpy.
"""
import argparse
import pathlib

import numpy as np
from PIL import Image

RESIZES = [(23, 17), (97, 59), (61, 20)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    src = Image.open(d / "src.png").convert("RGB")
    for w, h in RESIZES:
        r = src.resize((w, h), Image.LANCZOS)
        (d / f"lanczos_{w}x{h}.rgb").write_bytes(np.asarray(r).tobytes())


if __name__ == "__main__":
    main()
