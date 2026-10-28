#!/usr/bin/env python3
"""Writes the JPEG decoder test corpus (i#250) to tests/golden/jpeg/.

Synthetic photo-like sources (gradients, edges, noise) go through
cjpeg (libjpeg-turbo) with the encodings image input meets: every
common chroma subsampling and some unusual ones, baseline, optimized
and progressive scans, restart intervals, grayscale, RGB colorspace,
16-bit quantization tables (quality 1), odd sizes. The script writes
only the .jpg files. Each NAME.rgb next to them is stb_image v2.30's
3-channel output, written while stb_image was vendored (locus's
decoder matched it byte for byte); the tests compare with those. A
new file needs a reference made the same way. libjpeg-turbo's djpeg
(-dct int) is only a sanity check: it rounds differently, by up to 2
in about 3% of the bytes even at 4:4:4.

arith_*.jpg uses arithmetic coding, which the decoder refuses.

Needs cjpeg on PATH. Run from the repo root:

    python3 -I scripts/make-jpeg-corpus.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

OUT = os.path.join("tests", "golden", "jpeg")


class Rng:
    """A small LCG, so the corpus is the same on every run."""

    def __init__(self, seed):
        self.s = seed

    def next(self, n):
        self.s = (self.s * 1103515245 + 12345) & 0x7FFFFFFF
        return (self.s >> 8) % n


def pixels(w, h, seed):
    rng = Rng(seed)
    out = bytearray()
    for y in range(h):
        for x in range(w):
            r = (x * 255 // max(1, w - 1) + rng.next(24)) % 256
            g = (y * 255 // max(1, h - 1) + rng.next(24)) % 256
            edge = 220 if (x // 6 + y // 5) % 2 else 30
            b = (edge + rng.next(16)) % 256
            out += bytes([r, g, b])
    return bytes(out)


def write_ppm(path, w, h, rgb, gray=False):
    with open(path, "wb") as f:
        if gray:
            f.write(b"P5\n%d %d\n255\n" % (w, h))
            f.write(bytes(rgb[i] for i in range(0, len(rgb), 3)))
        else:
            f.write(b"P6\n%d %d\n255\n" % (w, h))
            f.write(rgb)


def main():
    if not os.path.isdir(os.path.join("tests", "golden")):
        sys.exit("run from the repo root")
    cjpeg = shutil.which("cjpeg")
    if cjpeg is None:
        sys.exit("cjpeg not found")
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp()
    cases = [
        # name, w, h, gray source, cjpeg options
        ("s444_q75", 37, 29, False, ["-quality", "75", "-sample", "1x1"]),
        ("s422_q75", 37, 29, False, ["-quality", "75", "-sample", "2x1"]),
        ("s420_q75", 37, 29, False, ["-quality", "75", "-sample", "2x2"]),
        ("s440_q75", 37, 29, False, ["-quality", "75", "-sample", "1x2"]),
        ("s411_q75", 37, 29, False, ["-quality", "75", "-sample", "4x1"]),
        ("s31_q75", 37, 29, False, ["-quality", "75", "-sample", "3x1"]),
        ("s22_11_21_q75", 37, 29, False,
         ["-quality", "75", "-sample", "2x2,1x1,2x1"]),
        ("s420_q100", 64, 48, False, ["-quality", "100", "-sample", "2x2"]),
        ("s420_q1_dqt16", 64, 48, False, ["-quality", "1", "-sample", "2x2"]),
        ("s420_q10", 103, 67, False, ["-quality", "10", "-sample", "2x2"]),
        ("s420_opt", 103, 67, False, ["-optimize", "-sample", "2x2"]),
        ("s420_prog", 103, 67, False, ["-progressive", "-sample", "2x2"]),
        ("s444_prog", 103, 67, False, ["-progressive", "-sample", "1x1"]),
        ("s422_prog_q95", 61, 33, False,
         ["-progressive", "-quality", "95", "-sample", "2x1"]),
        ("s420_restart1", 103, 67, False,
         ["-restart", "1B", "-sample", "2x2"]),
        ("s444_restart_rows", 61, 33, False,
         ["-restart", "1", "-sample", "1x1"]),
        ("s420_prog_restart", 61, 33, False,
         ["-progressive", "-restart", "3B", "-sample", "2x2"]),
        ("gray_q75", 37, 29, True, ["-quality", "75"]),
        ("gray_prog", 61, 33, True, ["-progressive"]),
        ("rgb_colorspace", 37, 29, False, ["-rgb", "-quality", "90"]),
        ("tiny_1x1", 1, 1, False, ["-sample", "2x2"]),
        ("tiny_1x1_gray", 1, 1, True, []),
        ("odd_7x5_420", 7, 5, False, ["-sample", "2x2"]),
        ("odd_17x9_422_prog", 17, 9, False, ["-progressive", "-sample", "2x1"]),
        ("wide_129x3_420", 129, 3, False, ["-sample", "2x2"]),
        ("tall_3x129_420", 3, 129, False, ["-sample", "2x2"]),
        ("arith_420", 37, 29, False, ["-arithmetic", "-sample", "2x2"]),
    ]
    for i, (name, w, h, gray, opts) in enumerate(cases):
        src = os.path.join(tmp, name + (".pgm" if gray else ".ppm"))
        write_ppm(src, w, h, pixels(w, h, 1000 + i), gray)
        dst = os.path.join(OUT, name + ".jpg")
        subprocess.run([cjpeg] + opts + ["-outfile", dst, src], check=True)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
