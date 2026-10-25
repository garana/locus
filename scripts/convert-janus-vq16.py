#!/usr/bin/env python3
"""Converts Janus-Pro's VQ-16 image decoder (LlamaGen's) to the "vq16"
GGUF that locus::image::Vq16Decoder loads (i#221 section 5).

Only the decode path is kept: the codebook (raw; the loader
l2-normalises it at lookup, as vq16.codebook_l2_norm = true records),
post_quant_conv and decoder.*, as F32 under their PyTorch module names
with the gen_vision_model. prefix stripped. The encoder and quant_conv
are dropped.

Before writing, it checks the mapping: it loads the written tensors
into a fresh Janus VQModel decoder and compares decode_code() with the
checkpoint's own on fixed random codes; it exits non-zero if they
differ by more than 1e-5.

Dev-time tool: needs a checkout of github.com/deepseek-ai/Janus (for
the reference classes) and the Janus-Pro snapshot (pytorch_model.bin,
loaded with torch's weights_only default). Tested with Python 3.10,
torch 2.14.1, gguf 0.19.0, numpy 2.2.6, Janus commit 1daa72f,
Janus-Pro-1B revision 960ab33191f61342a4c60ae74d8dc356a39fafcb:

    python3 scripts/convert-janus-vq16.py --janus SRC --model DIR \\
        --out tests/models/janus-vq16-decoder.gguf

With those versions the output (197 tensors, 170 MB) has sha256
3d243856e4f917b37d2731e9ae85543a193b96b6bce3148ec46d65904f20b672.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

PREFIX = "gen_vision_model."
KEEP = ("quantize.embedding.weight", "post_quant_conv.", "decoder.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--janus", required=True, help="Janus checkout")
    ap.add_argument("--model", required=True, help="Janus-Pro snapshot")
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(pathlib.Path(args.janus).resolve()))
    from janus.models import vq_model  # noqa: E402

    sd = torch.load(pathlib.Path(args.model) / "pytorch_model.bin",
                    map_location="cpu")
    tensors = {}
    for k, v in sd.items():
        if not k.startswith(PREFIX):
            continue
        name = k[len(PREFIX):]
        if name.startswith(KEEP):
            tensors[name] = v.detach().float().contiguous()
    if "quantize.embedding.weight" not in tensors:
        sys.exit("no gen_vision_model.quantize.embedding.weight: not a "
                 "Janus checkpoint")

    # The VQ_16 config the decoder graph assumes (i#221 section 4); refuse
    # a checkpoint whose shapes say otherwise.
    args16 = vq_model.ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4],
                                decoder_ch_mult=[1, 1, 2, 2, 4])
    ref = vq_model.VQModel(args16).eval()
    ref.load_state_dict(
        {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)},
        strict=True)
    cb = tensors["quantize.embedding.weight"]
    if tuple(cb.shape) != (args16.codebook_size, args16.codebook_embed_dim):
        sys.exit(f"codebook shape {tuple(cb.shape)} is not VQ_16's")
    if not args16.codebook_l2_norm:
        sys.exit("expected an l2-normalised codebook")

    # Self-check: a fresh decoder loaded from exactly the tensors written
    # must decode like the checkpoint's own model.
    fresh = vq_model.VQModel(args16).eval()
    missing = fresh.load_state_dict(tensors, strict=False)
    needed = [k for k in missing.missing_keys
              if k.startswith(KEEP)]
    if needed:
        sys.exit(f"decode tensors missing from the GGUF set: {needed[:5]}")
    g = torch.Generator().manual_seed(args.seed)
    codes = torch.randint(0, args16.codebook_size, (1, 576), generator=g)
    with torch.inference_mode():
        a = ref.decode_code(codes, shape=[1, 8, 24, 24])
        b = fresh.decode_code(codes, shape=[1, 8, 24, 24])
    diff = float((a - b).abs().max())
    print(f"self-check: max abs {diff:.3g} over {a.numel()} values")
    if diff > 1e-5:
        sys.exit("self-check failed")

    from gguf import GGUFWriter  # noqa: E402
    out = GGUFWriter(args.out, "vq16")
    out.add_uint32("vq16.codebook_size", args16.codebook_size)
    out.add_uint32("vq16.embed_dim", args16.codebook_embed_dim)
    out.add_bool("vq16.codebook_l2_norm", True)
    out.add_uint32("vq16.z_channels", args16.z_channels)
    out.add_uint32("vq16.ch", 128)
    out.add_array("vq16.ch_mult", list(args16.decoder_ch_mult))
    out.add_uint32("vq16.num_res_blocks", 2)
    out.add_array("vq16.attn_levels", [len(args16.decoder_ch_mult) - 1])
    out.add_uint32("vq16.norm_groups", 32)
    out.add_float32("vq16.norm_eps", 1e-6)
    for name in sorted(tensors):
        out.add_tensor(name, tensors[name].numpy().astype(np.float32))
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    print(f"wrote {args.out}: {len(tensors)} tensors")


if __name__ == "__main__":
    main()
