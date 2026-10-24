#!/usr/bin/env python3
"""Converts the SNAC 24 kHz neural audio codec to the "snac" GGUF that
locus-server's /v1/audio/speech loads with --audio-decoder.

SNAC turns the codec tokens an Orpheus-TTS model generates back into a
waveform. locus runs only its decoder, built from the conv1d /
conv1d_transpose / snake1d kernels (locus::audio::SnacDecoder). This
script folds each weight_norm convolution into a plain weight (along
dim 0: w = g * v / ||v||, the same for Conv1d and ConvTranspose1d) and
writes the decode-path tensors (codebooks, out_proj, folded conv
weights + biases, snake alphas) under their PyTorch module names, plus
the snac.* config the decoder reads.

Before writing, it checks the fold: it re-implements the decoder from
the folded tensors and compares it with the shipped model.decode() on
fixed random codes (a loud signal, not near-silence). It exits non-zero
if they differ by more than 1e-5.

Dev-time tool: needs network (HuggingFace) and torch, so it is not a
build dependency (DESIGN.md section 6). Tested with Python 3.10, torch
2.14.1, snac 1.2.1, gguf 0.19.0, numpy 2.2.6:

    pip install torch==2.14.1 snac==1.2.1 gguf==0.19.0 numpy==2.2.6
    python3 scripts/convert-snac.py --out tests/models/snac_24khz_decoder.gguf

For hubertsiuzdak/snac_24khz with those versions the output sha256 is
08acf9b1e8a00626ac82662ddc54b5b2ca7fd8fd99be236c87b94b6a6f60009e.
"""
import argparse
import sys

import numpy as np
import snac as snac_mod
import torch
import torch.nn.functional as F

# Max abs difference allowed between the folded re-implementation and
# model.decode(): float32 summation-order noise only.
TOLERANCE = 1e-5


def fold(g, v):
    """
    @param g weight_norm magnitude (original0), shape [dim0, 1, 1].
    @param v weight_norm direction (original1), shape [dim0, ...].
    @returns The plain weight g[j] * v[j] / ||v[j]||_2, norm over all
        dims except 0 (PyTorch weight_norm(dim=0)).
    """
    norm = v.reshape(v.shape[0], -1).norm(dim=1).reshape(-1, 1, 1)
    return g * v / norm


def folded_decode(model, sd, codes):
    """
    Decodes `codes` from the folded tensors with plain conv ops, in the
    same graph locus::audio::SnacDecoder runs (NoiseBlock removed).

    @param model The SNAC model (for its config).
    @param sd The model's state dict.
    @param codes One [1, T_i] long tensor per codebook.
    @returns The mono waveform, a 1-D tensor.
    """
    def w(name):
        return fold(sd[name + ".parametrizations.weight.original0"],
                    sd[name + ".parametrizations.weight.original1"])

    def b(name):
        return sd.get(name + ".bias")

    def snake(x, alpha):
        return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x) ** 2

    # from_codes: codebook lookup, 1x1 out_proj, stride upsample, sum.
    z_q = 0.0
    for i, stride in enumerate(model.vq_strides):
        q = f"quantizer.quantizers.{i}"
        z_p = F.embedding(codes[i], sd[q + ".codebook.weight"])
        z = F.conv1d(z_p.transpose(1, 2), w(q + ".out_proj"),
                     b(q + ".out_proj"))
        z_q = z_q + z.repeat_interleave(stride, dim=-1)

    latent = model.latent_dim
    x = F.conv1d(z_q, w("decoder.model.0"), b("decoder.model.0"),
                 padding=3, groups=latent)          # depthwise k7
    x = F.conv1d(x, w("decoder.model.1"), b("decoder.model.1"))  # 1x1

    for bi, rate in enumerate(model.decoder_rates):
        p = f"decoder.model.{2 + bi}.block"
        x = snake(x, sd[p + ".0.alpha"])
        x = F.conv_transpose1d(x, w(p + ".1"), b(p + ".1"), stride=rate,
                               padding=(rate + 1) // 2,
                               output_padding=rate % 2)
        dim = x.shape[1]
        # block.2 is the NoiseBlock (skipped); residual units are
        # block.3/4/5 with dilations 1/3/9.
        for ui, dil in ((3, 1), (4, 3), (5, 9)):
            rp = f"{p}.{ui}.block"
            y = snake(x, sd[rp + ".0.alpha"])
            y = F.conv1d(y, w(rp + ".1"), b(rp + ".1"), padding=3 * dil,
                         dilation=dil, groups=dim)
            y = snake(y, sd[rp + ".2.alpha"])
            y = F.conv1d(y, w(rp + ".3"), b(rp + ".3"))
            pad = (x.shape[-1] - y.shape[-1]) // 2
            x = (x[..., pad:-pad] if pad > 0 else x) + y

    # Tail: Snake1d (model.N), k7 conv to one channel (model.N+1), tanh.
    t = 2 + len(model.decoder_rates)
    x = snake(x, sd[f"decoder.model.{t}.alpha"])
    x = F.conv1d(x, w(f"decoder.model.{t + 1}"),
                 b(f"decoder.model.{t + 1}"), padding=3)
    return torch.tanh(x).squeeze()


def check_fold(model, sd, frames, seed):
    """
    @returns (max abs diff, reference RMS) of folded_decode() vs
        model.decode() on seeded random codes for `frames` frames.
    """
    gen = torch.Generator().manual_seed(seed)
    top = max(model.vq_strides)
    codes = [torch.randint(0, model.codebook_size,
                           (1, frames * top // s), generator=gen)
             for s in model.vq_strides]
    with torch.inference_mode():
        ref = model.decode(codes).squeeze()
        ours = folded_decode(model, sd, codes)
    if ours.shape != ref.shape:
        sys.exit(f"shape mismatch: ours {tuple(ours.shape)} "
                 f"vs reference {tuple(ref.shape)}")
    diff = float((ours - ref).abs().max())
    rms = float(ref.pow(2).mean().sqrt())
    return diff, rms


def emit_gguf(model, sd, path):
    """
    Writes the folded decoder as a GGUF (arch "snac"): every
    weight-normed conv on the decode path becomes `<module>.weight`;
    biases, snake alphas and codebooks keep their module names; the
    encoder and the quantizers' encode-side in_proj are left out.

    @param path Output file.
    """
    from gguf import GGUFWriter

    out = GGUFWriter(path, "snac")
    out.add_uint32("snac.sampling_rate", model.sampling_rate)
    out.add_uint32("snac.decoder_dim", model.decoder_dim)
    out.add_array("snac.decoder_rates", list(model.decoder_rates))
    out.add_array("snac.vq_strides", list(model.vq_strides))
    out.add_uint32("snac.latent_dim", model.latent_dim)
    out.add_uint32("snac.codebook_size", model.codebook_size)
    out.add_uint32("snac.codebook_dim", model.codebook_dim)
    out.add_uint32("snac.n_codebooks", model.n_codebooks)

    def put(name, t):
        out.add_tensor(name, t.detach().cpu().numpy().astype(np.float32))

    def wanted(k):
        if k.startswith("decoder."):
            return True
        return k.startswith("quantizer.") and \
            ("out_proj" in k or "codebook.weight" in k)

    suffix = ".parametrizations.weight.original0"
    for k in sd:
        if k.endswith(suffix) and wanted(k[:-len(suffix)]):
            prefix = k[:-len(suffix)]
            put(prefix + ".weight",
                fold(sd[prefix + suffix],
                     sd[prefix + ".parametrizations.weight.original1"]))
    for k, v in sd.items():
        if ".parametrizations." in k or not wanted(k):
            continue
        if k.endswith((".bias", ".alpha", "codebook.weight")):
            put(k, v)
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default="hubertsiuzdak/snac_24khz",
                    help="HuggingFace repo id or local SNAC directory")
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--check-frames", type=int, default=8,
                    help="frames of random codes for the fold check")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the fold-check codes")
    a = ap.parse_args()

    # The shipped decoder adds random noise (NoiseBlock); locus runs it
    # without, so compare against the same deterministic graph.
    snac_mod.layers.NoiseBlock.forward = lambda self, x: x
    model = snac_mod.SNAC.from_pretrained(a.repo).eval()
    if model.attn_window_size is not None:
        sys.exit("this SNAC has decoder attention; locus supports only "
                 "attn_window_size null (the 24 kHz model)")
    sd = model.state_dict()

    diff, rms = check_fold(model, sd, a.check_frames, a.seed)
    print(f"fold check: max abs diff {diff:.3g} on a signal of RMS "
          f"{rms:.3g} ({a.check_frames} frames, seed {a.seed})")
    if not diff <= TOLERANCE:
        sys.exit(f"fold check FAILED: {diff:.3g} > {TOLERANCE}")

    emit_gguf(model, sd, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
