#!/usr/bin/env python3
"""Writes the PNG decoder test corpus (i#250) to tests/golden/png/.

Each case is a small PNG plus NAME.rgb, the RGB bytes stb_image gives
for it with 3 channels: 16-bit samples keep their high byte, gray below
8 bits scales to 0..255, gray fills R, G and B, alpha and tRNS are
dropped, palette indexes look up PLTE. The script knows the samples it
encodes, so the expected bytes come from them, not from a decoder.

It covers every color type and bit depth, both interlace modes, all
five row filters (row y uses filter y % 5), stored / fixed / dynamic /
Huffman-only deflate, IDAT split across chunks, and ancillary chunks.
Standard library only (zlib). Run from the repo root:

    python3 -I scripts/make-png-corpus.py
"""

import os
import struct
import sys
import zlib

OUT = os.path.join("tests", "golden", "png")

# Adam7: first column, first row, column step, row step.
ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
         (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def chunk(t, data):
    body = t + data
    return struct.pack(">I", len(data)) + body + struct.pack(
        ">I", zlib.crc32(body) & 0xFFFFFFFF)


class Rng:
    """A small LCG, so the corpus is the same on every run."""

    def __init__(self, seed):
        self.s = seed

    def next(self, n):
        self.s = (self.s * 1103515245 + 12345) & 0x7FFFFFFF
        return (self.s >> 8) % n


def samples(w, h, color, depth, rng, pal_len):
    """Per-pixel sample tuples: a gradient mixed with noise."""
    ch = CHANNELS[color]
    top = (1 << depth) - 1 if color != 3 else pal_len - 1
    px = []
    for y in range(h):
        row = []
        for x in range(w):
            row.append(tuple(
                ((x * 37 + y * 11 + c * 101) * (top + 1) // 97 + rng.next(3))
                % (top + 1) for c in range(ch)))
        px.append(row)
    return px


def pack_row(row, color, depth):
    out = bytearray()
    if depth >= 8:
        for p in row:
            for v in p:
                out += struct.pack(">H", v) if depth == 16 else bytes([v])
        return bytes(out)
    acc, n = 0, 0
    for p in row:
        acc = (acc << depth) | p[0]
        n += depth
        if n == 8:
            out.append(acc)
            acc, n = 0, 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_rows(rows, bpp, first_filter):
    out = bytearray()
    prev = bytes(len(rows[0])) if rows else b""
    for i, r in enumerate(rows):
        f = (first_filter + i) % 5
        enc = bytearray([f])
        for j, v in enumerate(r):
            a = r[j - bpp] if j >= bpp else 0
            b = prev[j]
            c = prev[j - bpp] if j >= bpp else 0
            pred = [0, a, b, (a + b) // 2, paeth(a, b, c)][f]
            enc.append((v - pred) & 0xFF)
        out += enc
        prev = r
    return bytes(out)


def raw_data(px, w, h, color, depth, interlace):
    bpp = max(1, CHANNELS[color] * depth // 8)
    if not interlace:
        return filter_rows([pack_row(r, color, depth) for r in px], bpp, 0)
    out = bytearray()
    for k, (x0, y0, dx, dy) in enumerate(ADAM7):
        sub = [[px[y][x] for x in range(x0, w, dx)]
               for y in range(y0, h, dy)]
        sub = [r for r in sub if r]
        if not sub:
            continue
        out += filter_rows([pack_row(r, color, depth) for r in sub], bpp,
                           k)
    return bytes(out)


def compress(raw, mode):
    if mode == "stored":
        return zlib.compress(raw, 0)
    strategy = {"fixed": zlib.Z_FIXED, "dynamic": zlib.Z_DEFAULT_STRATEGY,
                "huffman": zlib.Z_HUFFMAN_ONLY}[mode]
    c = zlib.compressobj(9, zlib.DEFLATED, 15, 9, strategy)
    return c.compress(raw) + c.flush()


def expected(px, color, depth, palette):
    out = bytearray()
    for row in px:
        for p in row:
            if color == 3:
                out += bytes(palette[p[0]])
                continue
            v = [s >> 8 if depth == 16 else s for s in p]
            if depth < 8:
                v = [s * (255 // ((1 << depth) - 1)) for s in v]
            out += bytes([v[0]] * 3) if color in (0, 4) else bytes(v[:3])
    return bytes(out)


def write(name, w, h, color, depth, interlace, mode, split=0, extras=()):
    # A fixed seed (Python's hash() of a str changes per run).
    rng = Rng(w * 7919 + h * 104729 + color * 31 + depth * 17 +
              int(interlace) * 3 + len(mode))
    pal_len = 1 << min(depth, 8) if color == 3 else 0
    if color == 3 and depth == 8:
        pal_len = 200  # a palette smaller than the index range
    palette = [((i * 53) % 256, (i * 97 + 30) % 256, (255 - i * 7) % 256)
               for i in range(pal_len)]
    px = samples(w, h, color, depth, rng, pal_len)
    z = compress(raw_data(px, w, h, color, depth, interlace), mode)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(
        ">IIBBBBB", w, h, depth, color, 0, 0, int(interlace)))
    for e in extras:
        if e == "gAMA":
            png += chunk(b"gAMA", struct.pack(">I", 45455))
        elif e == "tEXt":
            png += chunk(b"tEXt", b"Comment\x00locus corpus")
    if color == 3:
        png += chunk(b"PLTE", b"".join(bytes(c) for c in palette))
        if "tRNS" in extras:
            png += chunk(b"tRNS", bytes(range(0, 250, 50)))
    elif "tRNS" in extras and color == 0:
        png += chunk(b"tRNS", struct.pack(">H", 1))
    elif "tRNS" in extras and color == 2:
        png += chunk(b"tRNS", struct.pack(">HHH", 1, 2, 3))
    if split:
        for i in range(0, len(z), split):
            png += chunk(b"IDAT", z[i:i + split])
    else:
        png += chunk(b"IDAT", z)
    png += chunk(b"IEND", b"")
    with open(os.path.join(OUT, name + ".png"), "wb") as f:
        f.write(png)
    with open(os.path.join(OUT, name + ".rgb"), "wb") as f:
        f.write(expected(px, color, depth, palette))


def main():
    if not os.path.isdir(os.path.join("tests", "golden")):
        sys.exit("run from the repo root")
    os.makedirs(OUT, exist_ok=True)
    combos = [(0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (2, 8), (2, 16),
              (3, 1), (3, 2), (3, 4), (3, 8), (4, 8), (4, 16), (6, 8),
              (6, 16)]
    modes = ["dynamic", "fixed", "stored", "huffman"]
    for i, (color, depth) in enumerate(combos):
        for interlace in (False, True):
            mode = modes[(i + interlace) % len(modes)]
            name = "c%d_d%d_%s_%s" % (color, depth,
                                      "adam7" if interlace else "plain",
                                      mode)
            write(name, 13, 7, color, depth, interlace, mode)
    # Odd sizes, split IDAT, ancillary chunks.
    write("tiny_1x1_rgb", 1, 1, 2, 8, False, "dynamic")
    write("tiny_1x1_adam7_gray1", 1, 1, 0, 1, True, "stored")
    write("wide_61x3_rgba16_split", 61, 3, 6, 16, False, "dynamic",
          split=37)
    write("tall_3x41_pal4_adam7_split", 3, 41, 3, 4, True, "fixed",
          split=11, extras=("tRNS", "gAMA"))
    write("gray8_trns_text", 33, 17, 0, 8, False, "dynamic",
          extras=("tRNS", "tEXt"))
    write("rgb8_trns_adam7", 33, 17, 2, 8, True, "huffman",
          extras=("tRNS",))
    write("rgb8_big_dynamic", 129, 65, 2, 8, False, "dynamic")


if __name__ == "__main__":
    main()
