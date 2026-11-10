#!/usr/bin/env python3
"""Converts a FLUX.1 LoRA (safetensors) to a "flux_lora" GGUF (i#372).

A LoRA changes a linear layer W (out x in) by a low-rank product:
W' = W + s B A, with A (r x in), B (out x r) and s = alpha / r. The
GGUF holds, for each changed layer of the DiT, two F32 tensors named
after the DiT's own weight (the diffusers name, as scripts/convert-
flux.py writes it):

- "<layer>.lora_a": A, r x in;
- "<layer>.lora_b": s B, out x r (alpha / r folded in, so a loader
  multiplies only by the user's strength).

Two key layouts are read, the ones diffusers 0.31 loads for FLUX:

- diffusers / PEFT: "[transformer.]<layer>.lora_A.weight" and
  ".lora_B.weight", an optional "<layer>.alpha" (none means s = 1),
  for any 2D weight of the DiT;
- kohya (sd-scripts): "lora_unet_<bfl_layer>.lora_down.weight",
  ".lora_up.weight" and ".alpha", on the original layer names of the
  double and single blocks, where one fused projection maps onto
  several DiT weights: double_blocks_N_img_attn_qkv onto to_q, to_k
  and to_v, and single_blocks_N_linear1 onto to_q, to_k, to_v and
  proj_mlp. A splits as is (every part keeps all r rows) and s B splits
  by rows, at the DiT's own output sizes.

s is alpha / r per module, as kohya and PEFT define it. diffusers 0.31
differs for a file whose modules differ in rank: it takes the first
module's rank as every module's alpha, so it rescales a module of
another rank by first rank / r. Files of one rank agree.

--dit names the DiT GGUF; only its header is read, to map the names
and to check every shape. Text-encoder layers (lora_te*) are skipped
with a note. Any other key is an error, not a silent drop.

The LoRA may be BF16, which safetensors.numpy cannot load, so this
reads the safetensors header and tensor bytes itself.

Tested with Python 3.12, gguf 0.19.0, numpy 2.1.3:

    python3 -I scripts/convert-lora.py --dit flux-kontext-dev.gguf \\
        --lora my-lora.safetensors --out my-lora.gguf
"""

import argparse
import json
import pathlib
import re
import struct
import sys

import gguf
import numpy as np


def read_safetensors(path):
    """@returns {name: array} of one safetensors file; BF16 widened to
    F32 exactly, F16 and F32 as floats."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, t in hdr.items():
        b0, b1 = t["data_offsets"]
        raw = mm[8 + n + b0:8 + n + b1]
        shape = t["shape"] or [1]
        if t["dtype"] == "BF16":
            a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif t["dtype"] in ("F16", "F32"):
            a = raw.view({"F16": np.float16, "F32": np.float32}[t["dtype"]])
        else:
            sys.exit("%s: unsupported dtype %s" % (name, t["dtype"]))
        out[name] = a.reshape(shape).astype(np.float32)
    return out


def dit_shapes(path):
    """@returns {layer: (out, in)} of the DiT GGUF's 2D weights, keyed by
    the name without ".weight"."""
    r = gguf.GGUFReader(str(path))
    arch = r.fields.get("general.architecture")
    if arch is None or bytes(arch.parts[-1]).decode() != "flux":
        sys.exit("%s: not a flux GGUF" % path)
    out = {}
    for t in r.tensors:
        if t.name.endswith(".weight") and len(t.shape) == 2:
            # GGUF lists dimensions innermost first: (in, out).
            out[t.name[:-len(".weight")]] = (int(t.shape[1]),
                                             int(t.shape[0]))
    return out


# kohya's double-block layers: (BFL suffix, the DiT layers it maps onto).
DOUBLE = [
    ("img_attn_qkv", ["attn.to_q", "attn.to_k", "attn.to_v"]),
    ("img_attn_proj", ["attn.to_out.0"]),
    ("img_mlp_0", ["ff.net.0.proj"]),
    ("img_mlp_2", ["ff.net.2"]),
    ("img_mod_lin", ["norm1.linear"]),
    ("txt_attn_qkv", ["attn.add_q_proj", "attn.add_k_proj",
                      "attn.add_v_proj"]),
    ("txt_attn_proj", ["attn.to_add_out"]),
    ("txt_mlp_0", ["ff_context.net.0.proj"]),
    ("txt_mlp_2", ["ff_context.net.2"]),
    ("txt_mod_lin", ["norm1_context.linear"]),
]
SINGLE = [
    ("linear1", ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"]),
    ("linear2", ["proj_out"]),
    ("modulation_lin", ["norm.linear"]),
]


def kohya_targets(module):
    """@returns The DiT layers a kohya module (without "lora_unet_")
    maps onto, or None if it is not a known block layer."""
    m = re.fullmatch(r"double_blocks_(\d+)_(.+)", module)
    if m:
        table, prefix = DOUBLE, "transformer_blocks.%s." % m[1]
    else:
        m = re.fullmatch(r"single_blocks_(\d+)_(.+)", module)
        if not m:
            return None
        table, prefix = SINGLE, "single_transformer_blocks.%s." % m[1]
    for suffix, layers in table:
        if m[2] == suffix:
            return [prefix + layer for layer in layers]
    return None


def group(sd):
    """Splits the state dict into modules.

    @returns (format, {module: {"a", "b", "alpha"}}, skipped count).
    """
    kohya = any(k.endswith(".lora_down.weight") for k in sd)
    peft = any(k.endswith(".lora_A.weight") for k in sd)
    if kohya == peft:
        sys.exit("expected either kohya (lora_down / lora_up) or diffusers "
                 "(lora_A / lora_B) keys")
    a_key, b_key = ((".lora_down.weight", ".lora_up.weight") if kohya
                    else (".lora_A.weight", ".lora_B.weight"))
    mods, skipped = {}, 0
    for k, v in sd.items():
        if k.startswith("lora_te"):
            skipped += 1
            continue
        for suffix, field in ((a_key, "a"), (b_key, "b"), (".alpha", "alpha")):
            if k.endswith(suffix):
                mods.setdefault(k[:-len(suffix)], {})[field] = v
                break
        else:
            sys.exit("%s: not a LoRA key this converter knows" % k)
    return ("kohya" if kohya else "diffusers"), mods, skipped


def convert(sd, shapes):
    """@returns (format, [(dit layer, A, s B)], skipped count)."""
    fmt, mods, skipped = group(sd)
    out = []
    for module, t in sorted(mods.items()):
        if "a" not in t or "b" not in t:
            sys.exit("%s: needs both its down (A) and up (B) weights" % module)
        a, b = t["a"], t["b"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            sys.exit("%s: A %s and B %s are not r x in and out x r"
                     % (module, a.shape, b.shape))
        rank = a.shape[0]
        alpha = float(t["alpha"].reshape(-1)[0]) if "alpha" in t else None
        if fmt == "kohya":
            layers = (kohya_targets(module[len("lora_unet_"):])
                      if module.startswith("lora_unet_") else None)
            if layers is None:
                sys.exit("%s: not a double- or single-block layer (diffusers "
                         "0.31 loads only those from kohya files)" % module)
            # kohya files always carry alpha; a missing one means s = 1.
            scale = alpha / rank if alpha is not None else 1.0
        else:
            layer = module[len("transformer."):] \
                if module.startswith("transformer.") else module
            layers = [layer]
            scale = alpha / rank if alpha is not None else 1.0
        for layer in layers:
            if layer not in shapes:
                sys.exit("%s: maps onto %s, which the DiT does not have"
                         % (module, layer))
        sizes = [shapes[layer][0] for layer in layers]
        if b.shape[0] != sum(sizes):
            sys.exit("%s: B has %d rows, the DiT layers %s have %d"
                     % (module, b.shape[0], layers, sum(sizes)))
        sb = (b * np.float32(scale)).astype(np.float32)
        row = 0
        for layer, size in zip(layers, sizes):
            if a.shape[1] != shapes[layer][1]:
                sys.exit("%s: A has %d columns, %s takes %d inputs"
                         % (module, a.shape[1], layer, shapes[layer][1]))
            out.append((layer, a, sb[row:row + size]))
            row += size
    seen = [layer for layer, _, _ in out]
    if len(set(seen)) != len(seen):
        sys.exit("two LoRA modules change the same DiT layer")
    return fmt, out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", required=True,
                    help="the FLUX DiT GGUF the LoRA is for (header only)")
    ap.add_argument("--lora", required=True, help="the LoRA safetensors")
    ap.add_argument("--out", required=True, help="the flux_lora GGUF")
    a = ap.parse_args()

    fmt, layers, skipped = convert(read_safetensors(a.lora),
                                   dit_shapes(a.dit))
    if not layers:
        sys.exit("no DiT layers in the LoRA")
    w = gguf.GGUFWriter(a.out, "flux_lora")
    w.add_name(pathlib.Path(a.lora).stem)
    w.add_string("flux_lora.source_format", fmt)
    w.add_uint32("flux_lora.layer_count", len(layers))
    for layer, la, lb in layers:
        w.add_tensor(layer + ".lora_a", np.ascontiguousarray(la))
        w.add_tensor(layer + ".lora_b", np.ascontiguousarray(lb))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("%s: %d DiT layers from a %s LoRA%s" % (
        a.out, len(layers), fmt,
        ", %d text-encoder tensors skipped" % skipped if skipped else ""))


if __name__ == "__main__":
    main()
