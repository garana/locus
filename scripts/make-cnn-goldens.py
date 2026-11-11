#!/usr/bin/env python3
"""Makes the CNN fixtures under tests/golden that locus::image::ResNet
(i#575) and locus::image::Inception (i#576) are compared against:
timm's model in float32 on the CPU, on tests/golden/janus_apple.png (a
384 x 384 apple) through timm's own eval transform for the checkpoint
(its pretrained_cfg).

Writes ARCHS' file for the checkpoint's architecture: the transform's
output at the arch's rows of each channel (3 x 3 x size floats), then
the 1000 logits, then the 2048-wide pooled feature (forward_head with
pre_logits). Prints the transform and the top 5.

    run-offline.sh scripts/make-cnn-goldens.py --model DIR

DIR as in scripts/convert-cnn.py. Tested with Python 3.12, torch 2.5.1,
torchvision 0.20.1, timm 1.0.19, safetensors 0.6.2 and pillow 11.3.0.
With those, on an M2 Pro (CPU):

  resnet50_apple.f32
    c6b98258067f6da4e57e7d61eea7fb7357173d0e4e931168847e1bd7e148cd1e
    top 5: Granny Smith 0.273, butternut squash 0.215, spaghetti
    squash 0.207, bell pepper 0.062, acorn squash 0.031
  inception_v3_apple.f32
    8e2ad8f0ff56122d8f23ea4eef80019556375a5b45b6093da61982db33ae0f6b
    top 5: bell pepper 0.360, Granny Smith 0.296, cucumber 0.014,
    banana 0.009, buckeye 0.009
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
# Per architecture: the sampled rows and the output file. Keep in sync
# with tests/test_resnet.cpp and tests/test_inception.cpp.
ARCHS = {
    "resnet50": ((0, 111, 223), "resnet50_apple.f32"),
    "inception_v3": ((0, 149, 298), "inception_v3_apple.f32"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    d = pathlib.Path(a.model)
    cfg = json.loads((d / "config.json").read_text())
    rows_at, fname = ARCHS[cfg["architecture"]]
    name = cfg["architecture"] + "." + cfg["pretrained_cfg"]["tag"]
    m = timm.create_model(name, pretrained=False).eval()
    m.load_state_dict(load_file(str(d / "model.safetensors")))
    dc = timm.data.resolve_data_config({}, model=m)
    tf = timm.data.create_transform(**dc)
    print(name, dc)
    print(tf)
    img = Image.open(GOLDEN / "janus_apple.png").convert("RGB")
    x = tf(img)
    size = dc["input_size"][1]
    assert x.shape == (3, size, size), x.shape
    with torch.no_grad():
        f = m.forward_features(x[None])
        emb = m.forward_head(f, pre_logits=True)[0]
        logits = m.forward_head(f)[0]
    assert emb.shape == (2048,) and logits.shape == (1000,)
    info = timm.data.ImageNetInfo()
    p = logits.softmax(0)
    for i in p.argsort(descending=True)[:5].tolist():
        print(i, info.index_to_description(i), float(p[i]))
    rows = x[:, list(rows_at), :]
    blob = np.concatenate([rows.numpy().ravel(), logits.numpy(),
                           emb.numpy()])
    path = GOLDEN / fname
    path.write_bytes(blob.astype("<f4").tobytes())
    print(path, blob.size)


if __name__ == "__main__":
    main()
