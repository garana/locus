#!/usr/bin/env python3
"""Makes the Qwen2.5-VL vision tower fixtures under tests/golden that
locus's image::QwenVision is compared against (i#530).

  synth  No model needed. A tiny tower (width 32, 2 heads, FFN 40, 4
         blocks with full attention in blocks 1 and 3, merger to 24)
         filled from the test's LCG in SORTED GGUF tensor-name order,
         on a 12 x 16 patch grid of LCG patches. The grid makes uneven
         windows and one window of padding only. tests/test_qwen_vision.cpp
         builds the same GGUF in memory, so the tower is checked on every
         box. Writes qwen_vision_synth.f32 (48 x 24).
  real   The Qwen2.5-VL 7B tower from llama.cpp's mmproj GGUF, mapped
         back to transformers' names (q, k and v joined into qkv, the two
         frame kernels stacked into the Conv3d), in float32 on the CPU,
         on the top-left 196 x 252 crop of janus_apple.png through the
         reference processor with do_resize off. Writes qwen_vision.f32:
         two processor patches (PATCHES), then ROWS sampled rows of the
         output (63 x 3584), then the RMS of the whole output.

    run-offline.sh scripts/make-qwen-vision-goldens.py synth
    run-offline.sh scripts/make-qwen-vision-goldens.py real --mmproj F

F is mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf (scripts/fetch-test-model.sh
qwen2.5-vl-mmproj). Its two tensors that llama.cpp's converter
transforms match the BF16 originals in Qwen/Qwen2.5-VL-7B-Instruct
(the Conv3d to 3e-8, the BF16 to F16 rounding; a qkv bias exactly).
Tested with Python 3.12, torch 2.5.1, transformers 4.56.1, numpy 2.1.3,
gguf 0.17.1 and pillow 11.3.0. With those, the outputs' sha256 (on an
M2 Pro, CPU):

  qwen_vision_synth.f32
    1ac9b3ff2c255f08a40ec9f907f3468b79f25bec56d25febf912130bbe44835a
  qwen_vision.f32
    f2d0256cb0350d7022bf3d6030689a528ea00b2adf514789cdfc055542e85ceb
"""
import argparse
import pathlib

import numpy as np
import torch
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qvl
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig)

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden"

# Keep in sync with tests/test_qwen_vision.cpp.
SYNTH = dict(depth=4, hidden_size=32, num_heads=2, intermediate_size=40,
             out_hidden_size=24, fullatt_block_indexes=[1, 3],
             window_size=112, patch_size=14, temporal_patch_size=2,
             spatial_merge_size=2, in_channels=3, hidden_act="silu")
SYNTH_GRID = (12, 16)
PATCHES = (0, 101)
ROWS, ROW_STEP = 16, 4
CROP = (196, 252)


def lcg(state, n):
    """
    n values from the test's LCG: state = state * 1664525 + 1013904223
    (mod 2^32), value = ((state >> 8) / 2^24 - 0.5) * 0.5. @returns
    (values, state).
    """
    vals = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        vals[i] = (np.float32(state >> 8) / np.float32(16777216.0) -
                   np.float32(0.5)) * np.float32(0.5)
    return vals, state


def gguf_shapes(c):
    """@returns {GGUF tensor name: torch-order shape} for config c."""
    w, f, m = c["hidden_size"], c["intermediate_size"], 4 * c["hidden_size"]
    p = c["patch_size"]
    s = {"v.patch_embd.weight": (w, 3, p, p),
         "v.patch_embd.weight.1": (w, 3, p, p),
         "v.post_ln.weight": (w,),
         "mm.0.weight": (m, m), "mm.0.bias": (m,),
         "mm.2.weight": (c["out_hidden_size"], m),
         "mm.2.bias": (c["out_hidden_size"],)}
    for i in range(c["depth"]):
        b = f"v.blk.{i}."
        s[b + "ln1.weight"] = (w,)
        s[b + "ln2.weight"] = (w,)
        for t in ("attn_q", "attn_k", "attn_v", "attn_out"):
            s[b + t + ".weight"] = (w, w)
            s[b + t + ".bias"] = (w,)
        for t in ("ffn_gate", "ffn_up"):
            s[b + t + ".weight"] = (f, w)
            s[b + t + ".bias"] = (f,)
        s[b + "ffn_down.weight"] = (w, f)
        s[b + "ffn_down.bias"] = (w,)
    return s


def to_hf(t, depth):
    """Maps GGUF-named arrays back to transformers' state dict, undoing
    llama.cpp's converter (Qwen2VLVisionModel.modify_tensors)."""
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))
    sd = {"patch_embed.proj.weight": T(np.stack(
              [t["v.patch_embd.weight"], t["v.patch_embd.weight.1"]], 2)),
          "merger.ln_q.weight": T(t["v.post_ln.weight"])}
    for j in (0, 2):
        sd[f"merger.mlp.{j}.weight"] = T(t[f"mm.{j}.weight"])
        sd[f"merger.mlp.{j}.bias"] = T(t[f"mm.{j}.bias"])
    for i in range(depth):
        b, h = f"v.blk.{i}.", f"blocks.{i}."
        sd[h + "norm1.weight"] = T(t[b + "ln1.weight"])
        sd[h + "norm2.weight"] = T(t[b + "ln2.weight"])
        for x in ("weight", "bias"):
            sd[h + "attn.qkv." + x] = T(np.concatenate(
                [t[b + f"attn_{q}.{x}"] for q in "qkv"], 0))
            sd[h + "attn.proj." + x] = T(t[b + "attn_out." + x])
            for g, l in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"),
                         ("down_proj", "ffn_down")):
                sd[h + f"mlp.{g}.{x}"] = T(t[b + f"{l}.{x}"])
    return sd


def tower(cfg, tensors):
    """@returns The reference tower in float32, eager attention."""
    c = Qwen2_5_VLVisionConfig(**cfg)
    c._attn_implementation = "eager"
    m = qvl.Qwen2_5_VisionTransformerPretrainedModel(c).float().eval()
    missing, unexpected = m.load_state_dict(to_hf(tensors, cfg["depth"]),
                                            strict=False)
    # inv_freq is a non-persistent buffer the constructor fills.
    assert not unexpected and all("inv_freq" in k for k in missing), (
        missing, unexpected)
    return m


def synth():
    shapes = gguf_shapes(SYNTH)
    state, t = 12345, {}
    for name in sorted(shapes):
        n = int(np.prod(shapes[name]))
        vals, state = lcg(state, n)
        t[name] = vals.reshape(shapes[name])
    gh, gw = SYNTH_GRID
    pd = 3 * 2 * 14 * 14
    px, _ = lcg(777, gh * gw * pd)
    px = torch.from_numpy(px * np.float32(4.0)).reshape(gh * gw, pd)
    with torch.no_grad():
        out = tower(SYNTH, t)(px, grid_thw=torch.tensor([[1, gh, gw]]))
    assert out.shape == (gh * gw // 4, SYNTH["out_hidden_size"])
    path = GOLDEN / "qwen_vision_synth.f32"
    path.write_bytes(out.numpy().astype("<f4").tobytes())
    print(path, out.shape, float(out.pow(2).mean().sqrt()))


def real(mmproj):
    import gguf
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor)
    r = gguf.GGUFReader(mmproj)
    kv = lambda k: r.fields[k].contents()
    cfg = dict(depth=kv("clip.vision.block_count"),
               hidden_size=kv("clip.vision.embedding_length"),
               num_heads=kv("clip.vision.attention.head_count"),
               intermediate_size=kv("clip.vision.feed_forward_length"),
               out_hidden_size=kv("clip.vision.projection_dim"),
               window_size=112, patch_size=kv("clip.vision.patch_size"),
               temporal_patch_size=2, spatial_merge_size=2, in_channels=3,
               hidden_act="silu")
    wa = kv("clip.vision.n_wa_pattern")
    cfg["fullatt_block_indexes"] = list(range(wa - 1, cfg["depth"], wa))
    shapes = gguf_shapes(cfg)
    t = {}
    for ti in r.tensors:
        a = np.array(ti.data, dtype=np.float32).reshape(shapes[ti.name])
        t[ti.name] = a
    assert sorted(t) == sorted(shapes)
    img = Image.open(GOLDEN / "janus_apple.png").convert("RGB")
    h, w = CROP
    img = img.crop((0, 0, w, h))
    proc = Qwen2VLImageProcessor(do_resize=False,
                                 image_mean=kv("clip.vision.image_mean"),
                                 image_std=kv("clip.vision.image_std"))
    b = proc(images=[img], return_tensors="pt")
    px, grid = b["pixel_values"].float(), b["image_grid_thw"]
    assert grid.tolist() == [[1, h // 14, w // 14]], grid
    with torch.no_grad():
        out = tower(cfg, t)(px, grid_thw=grid)
    assert out.shape == (h * w // 28 // 28, cfg["out_hidden_size"])
    rows = out[::ROW_STEP][:ROWS]
    assert rows.shape[0] == ROWS
    blob = np.concatenate([px[list(PATCHES)].numpy().ravel(),
                           rows.numpy().ravel(),
                           np.array([out.pow(2).mean().sqrt()])])
    path = GOLDEN / "qwen_vision.f32"
    path.write_bytes(blob.astype("<f4").tobytes())
    print(path, out.shape, float(blob[-1]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("synth")
    p = sub.add_parser("real")
    p.add_argument("--mmproj", required=True)
    a = ap.parse_args()
    if a.mode == "synth":
        synth()
    else:
        real(a.mmproj)


if __name__ == "__main__":
    main()
