#!/usr/bin/env python3
"""Makes the CNN fixtures under tests/golden that locus::image::ResNet
is compared against (i#575): timm's model in float32 on the CPU, on
tests/golden/janus_apple.png (a 384 x 384 apple) through timm's own eval
transform for the checkpoint (its pretrained_cfg).

Writes resnet50_apple.f32: the transform's output at rows ROWS of each
channel (3 x len(ROWS) x 224 floats), then the 1000 logits, then the
2048-wide pooled feature (forward_head with pre_logits). Prints the
transform and the top 5.

    run-offline.sh scripts/make-cnn-goldens.py --model DIR

DIR as in scripts/convert-cnn.py (timm/resnet50.tv_in1k). Tested with
Python 3.12, torch 2.5.1, torchvision 0.20.1, timm 1.0.19, safetensors
0.6.2 and pillow 11.3.0. With those, on an M2 Pro (CPU), the output's
sha256 is c6b98258067f6da4e57e7d61eea7fb7357173d0e4e931168847e1bd7e148cd1e,
and timm's top 5 is Granny Smith 0.273, butternut squash 0.215,
spaghetti squash 0.207, bell pepper 0.062 and acorn squash 0.031.
"""
import argparse
import json
import pathlib

import numpy as np
import timm
import torch
from PIL import Image
from safetensors.torch import load_file

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden"
# Keep in sync with tests/test_resnet.cpp.
ROWS = (0, 111, 223)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    d = pathlib.Path(a.model)
    cfg = json.loads((d / "config.json").read_text())
    name = cfg["architecture"] + "." + cfg["pretrained_cfg"]["tag"]
    m = timm.create_model(name, pretrained=False).eval()
    m.load_state_dict(load_file(str(d / "model.safetensors")))
    dc = timm.data.resolve_data_config({}, model=m)
    tf = timm.data.create_transform(**dc)
    print(name, dc)
    print(tf)
    img = Image.open(GOLDEN / "janus_apple.png").convert("RGB")
    x = tf(img)
    assert x.shape == (3, 224, 224), x.shape
    with torch.no_grad():
        f = m.forward_features(x[None])
        emb = m.forward_head(f, pre_logits=True)[0]
        logits = m.forward_head(f)[0]
    assert emb.shape == (2048,) and logits.shape == (1000,)
    info = timm.data.ImageNetInfo()
    p = logits.softmax(0)
    for i in p.argsort(descending=True)[:5].tolist():
        print(i, info.index_to_description(i), float(p[i]))
    rows = x[:, list(ROWS), :]
    blob = np.concatenate([rows.numpy().ravel(), logits.numpy(),
                           emb.numpy()])
    path = GOLDEN / "resnet50_apple.f32"
    path.write_bytes(blob.astype("<f4").tobytes())
    print(path, blob.size)


if __name__ == "__main__":
    main()
