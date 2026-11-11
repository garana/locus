#!/usr/bin/env python3
"""Converts a timm ResNet checkpoint (safetensors only, never a pickle)
to the "resnet" GGUF that locus::image::ResNet reads (i#575).

Each batch norm is folded into the conv before it (eval mode: w' = w *
g / sqrt(v + eps), b' = beta - m * g / sqrt(v + eps)), so the GGUF holds
convs with a bias and the classifier. The preprocessing (from the
checkpoint's own config.json, pretrained_cfg) and the 1000 ImageNet
labels (timm's ImageNetInfo descriptions) go into the metadata.

    run-offline.sh scripts/convert-cnn.py --model DIR --out F.gguf

DIR holds model.safetensors and config.json of a timm checkpoint, for
example timm/resnet50.tv_in1k at revision
78f3ecfdb38e06d9b8397f662e7ab8fee96026fa (model.safetensors sha256
5d061a3c593d795bfe682d9b152bafbcf550579873492def3515b46db1189888).
Tested with Python 3.12, numpy 2.1.3, safetensors 0.6.2, gguf 0.17.1
and timm 1.0.19. With those, tests/models/resnet50-tv-f32.gguf (the
name tests/test_resnet.cpp looks for) has sha256
c6d5c8cb760f2a7298b4fc5ce33a45d9a6d58a5fa319b2ddd5cec1e4bf9cc486.
"""
import argparse
import json
import math
import pathlib
import sys

import gguf
import numpy as np
from safetensors.numpy import load_file

EPS = 1e-5  # torch.nn.BatchNorm2d's default, as timm's ResNet uses


READ = set()  # the checkpoint keys read, to catch one left out


def fold(t, conv, bn):
    """@returns The conv weight and bias with bn folded in."""
    READ.update([conv + ".weight"] + [bn + "." + x for x in (
        "weight", "bias", "running_mean", "running_var")])
    w = t[conv + ".weight"].astype(np.float64)
    g = t[bn + ".weight"].astype(np.float64)
    b = t[bn + ".bias"].astype(np.float64)
    m = t[bn + ".running_mean"].astype(np.float64)
    v = t[bn + ".running_var"].astype(np.float64)
    s = g / np.sqrt(v + EPS)
    return ((w * s[:, None, None, None]).astype(np.float32),
            (b - m * s).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pathlib.Path(a.model)
    cfg = json.loads((d / "config.json").read_text())
    arch = cfg["architecture"]
    if arch != "resnet50":
        sys.exit(f"unsupported architecture {arch}")
    pc = cfg["pretrained_cfg"]
    t = load_file(str(d / "model.safetensors"))

    out = {}
    out["conv1.weight"], out["conv1.bias"] = fold(t, "conv1", "bn1")
    layers = [3, 4, 6, 3]  # ResNet-50
    for s, n in enumerate(layers, start=1):
        for i in range(n):
            p = f"layer{s}.{i}."
            for c in range(1, 4):
                w, b = fold(t, p + f"conv{c}", p + f"bn{c}")
                out[p + f"conv{c}.weight"], out[p + f"conv{c}.bias"] = w, b
            if p + "downsample.0.weight" in t:
                w, b = fold(t, p + "downsample.0", p + "downsample.1")
                out[p + "downsample.weight"] = w
                out[p + "downsample.bias"] = b
    out["fc.weight"] = t["fc.weight"].astype(np.float32)
    out["fc.bias"] = t["fc.bias"].astype(np.float32)
    READ.update(["fc.weight", "fc.bias"])
    left = {k for k in t if not k.endswith("num_batches_tracked")} - READ
    if left:
        sys.exit(f"tensors not converted: {sorted(left)[:5]}")

    import timm.data  # the labels only
    info = timm.data.ImageNetInfo()
    labels = [info.index_to_description(i) for i in range(1000)]

    w = gguf.GGUFWriter(a.out, "resnet")
    w.add_name(f"{arch}.{pc.get('tag', '')}")
    k = "resnet."
    w.add_array(k + "layers", layers)
    w.add_uint32(k + "num_classes", 1000)
    size = pc["input_size"][1]
    w.add_uint32(k + "input_size", size)
    # timm's eval transform: shorter side to floor(size / crop_pct), then
    # a center crop of size.
    w.add_uint32(k + "resize", math.floor(size / pc["crop_pct"]))
    w.add_string(k + "interpolation", pc["interpolation"])
    w.add_array(k + "mean", [float(x) for x in pc["mean"]])
    w.add_array(k + "std", [float(x) for x in pc["std"]])
    w.add_array(k + "labels", labels)
    for name in sorted(out):
        w.add_tensor(name, np.ascontiguousarray(out[name]))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(a.out, len(out), "tensors")


if __name__ == "__main__":
    main()
