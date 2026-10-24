#!/usr/bin/env python3
"""Makes the SNAC decoder parity fixtures under tests/golden that
tests/test_snac.cpp compares locus against.

  synth   No model needed. Builds a real torch snac.SNAC with the tiny
          synthetic config of the test's synth_snac() (latent 4,
          codebook 4 x 2, strides [4,2,1], width 4 * 2^rates), loads
          the SAME deterministic weights (the test's LCG, ported), and
          decodes the same codes. Writes the waveform per rate set, so
          the C++ test pins sample values, not just lengths: a wrong
          dilation, weight layout or snake formula changes the values
          even though every length stays the same.
  speech  Needs the Orpheus GGUF (llama-cpp-python) and the
          hubertsiuzdak/snac_24khz checkpoint. Samples Orpheus with the
          reference settings and a fixed seed, decodes the codes with
          torch SNAC, checks the result is audible speech, and writes
          the codes plus the waveform, so the decoder is checked on
          real speech and not only on the near-silent greedy golden.

The NoiseBlock is replaced by identity in both, as locus does. Tested
with Python 3.10, torch 2.14.1, snac 1.2.1, numpy 2.2.6 (speech also
llama-cpp-python 0.3.x):

    python3 scripts/make-snac-goldens.py synth
    python3 scripts/make-snac-goldens.py speech --gguf \\
        tests/models/orpheus-3b-0.1-ft-Q4_K_M.gguf
"""
import argparse
import pathlib
import sys

import numpy as np
import snac as snac_mod
import torch
from torch.nn.utils import parametrize

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden"
# The rate lists tests/test_snac.cpp decodes, and its frame count.
SYNTH_RATES = ([2, 2], [4, 2, 2, 2])
SYNTH_FRAMES = 2


def synth_name(rates):
    """@returns The fixture file name for one synthetic rate list."""
    return "snac_synth_r" + "-".join(map(str, rates)) + ".f32"


def lcg_weights(rates):
    """
    Ports synth_snac() in tests/test_snac.cpp: the same tensors, in the
    same order, filled from the same LCG (alphas are 1 and draw
    nothing). Keep the two in sync.

    @returns {module name: float32 array in PyTorch layout}.
    """
    latent, cb_size, cb_dim = 4, 4, 2
    dec = 4 << len(rates)
    state = [12345]
    out = {}

    def rnd():  # in [-0.25, 0.25): small enough that tanh stays linear
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return ((np.float32(state[0] >> 8) / np.float32(16777216.0)
                 - np.float32(0.5)) * np.float32(0.5))

    def add(name, shape, ones=False):
        n = int(np.prod(shape))
        v = (np.ones(n, np.float32) if ones
             else np.array([rnd() for _ in range(n)], np.float32))
        out[name] = v.reshape(shape)

    for i in range(3):
        q = f"quantizer.quantizers.{i}"
        add(q + ".codebook.weight", (cb_size, cb_dim))
        add(q + ".out_proj.weight", (latent, cb_dim, 1))
        add(q + ".out_proj.bias", (latent,))
    add("decoder.model.0.weight", (latent, 1, 7))
    add("decoder.model.0.bias", (latent,))
    add("decoder.model.1.weight", (dec, latent, 1))
    add("decoder.model.1.bias", (dec,))
    ch = dec
    for b, r in enumerate(rates):
        p = f"decoder.model.{2 + b}.block"
        o = ch // 2
        add(p + ".0.alpha", (1, ch, 1), True)
        add(p + ".1.weight", (ch, o, 2 * r))
        add(p + ".1.bias", (o,))
        for u in (3, 4, 5):
            rp = f"{p}.{u}.block"
            add(rp + ".0.alpha", (1, o, 1), True)
            add(rp + ".1.weight", (o, 1, 7))
            add(rp + ".1.bias", (o,))
            add(rp + ".2.alpha", (1, o, 1), True)
            add(rp + ".3.weight", (o, o, 1))
            add(rp + ".3.bias", (o,))
        ch = o
    t = 2 + len(rates)
    add(f"decoder.model.{t}.alpha", (1, ch, 1), True)
    add(f"decoder.model.{t + 1}.weight", (1, ch, 7))
    add(f"decoder.model.{t + 1}.bias", (1,))
    return out


def synth_codes(frames):
    """Ports synth_codes(frames, 0) in tests/test_snac.cpp."""
    return ([i % 4 for i in range(frames)],
            [(i + 1) % 4 for i in range(2 * frames)],
            [(i + 2) % 4 for i in range(4 * frames)])


def no_noise():
    """locus runs SNAC without its random NoiseBlock; match that."""
    snac_mod.layers.NoiseBlock.forward = lambda self, x: x


def decode(model, codes):
    """@returns model.decode() of three code lists as a 1-D float32."""
    with torch.inference_mode():
        t = [torch.tensor([c], dtype=torch.long) for c in codes]
        return model.decode(t).squeeze().numpy().astype("<f4")


def cmd_synth(_):
    no_noise()
    for rates in SYNTH_RATES:
        model = snac_mod.SNAC(
            sampling_rate=24000, encoder_dim=1, encoder_rates=[2, 2],
            latent_dim=4, decoder_dim=4 << len(rates),
            decoder_rates=list(rates), attn_window_size=None,
            codebook_size=4, codebook_dim=2, vq_strides=[4, 2, 1],
            noise=True, depthwise=True).eval()
        # Plain weights instead of weight_norm (g, v), so the loaded
        # values are exactly the LCG ones the C++ test writes.
        for m in model.modules():
            if parametrize.is_parametrized(m, "weight"):
                parametrize.remove_parametrizations(m, "weight")
        w = lcg_weights(rates)
        res = model.load_state_dict(
            {k: torch.from_numpy(v) for k, v in w.items()}, strict=False)
        if res.unexpected_keys:
            sys.exit(f"names torch does not have: {res.unexpected_keys}")
        unused = [k for k in res.missing_keys
                  if not k.startswith("encoder.") and "in_proj" not in k
                  and ".block.2.linear" not in k]
        if unused:
            sys.exit(f"decode-path tensors left unset: {unused}")
        wav = decode(model, synth_codes(SYNTH_FRAMES))
        path = GOLDEN / synth_name(rates)
        wav.tofile(path)
        print(f"{path.name}: {wav.size} samples, rms "
              f"{float(np.sqrt(np.mean(wav ** 2))):.4f}")


# Orpheus prompt wrapping and audio band (see i#153 and
# locus::model::codec): 7 codes per frame, each in its own 4096-wide
# sub-band starting at <custom_token_10>.
START_ID = 128259
END_IDS = [128009, 128260, 128261, 128257]
CODES_PER_FRAME = 7
CODEBOOK = 4096


def cmd_speech(a):
    from llama_cpp import Llama

    llm = Llama(model_path=str(a.gguf), n_ctx=2048, seed=a.seed,
                logits_all=False, verbose=False)
    base_ids = llm.tokenize(b"<custom_token_10>", add_bos=False,
                            special=True)
    if len(base_ids) != 1:
        sys.exit(f"<custom_token_10> is not one token: {base_ids}")
    base = base_ids[0]
    ids = ([START_ID]
           + llm.tokenize(f"{a.voice}: {a.text}".encode(), add_bos=True,
                          special=True)
           + END_IDS)
    # Orpheus reference sampling. top_k and min_p are off as in the
    # reference (llama-cpp-python enables both by default). This only
    # needs realistic speech codes for the DECODER check, so it does not
    # have to match locus's sampler token for token.
    gen, stop = [], None
    for t in llm.generate(ids, temp=0.6, top_p=0.8, top_k=0, min_p=0.0,
                          repeat_penalty=1.3):
        t = int(t)
        if not base <= t < base + CODES_PER_FRAME * CODEBOOK:
            stop = t
            break
        gen.append(t)
        if len(gen) >= a.max_tokens:
            break
    gen = gen[:len(gen) - len(gen) % CODES_PER_FRAME]
    layers = [[], [], []]
    for f in range(len(gen) // CODES_PER_FRAME):
        g = gen[f * CODES_PER_FRAME:(f + 1) * CODES_PER_FRAME]
        c = [(t - base) - i * CODEBOOK for i, t in enumerate(g)]
        if not all(0 <= x < CODEBOOK for x in c):
            sys.exit(f"frame {f} decodes out of range: {c}; try another "
                     "--seed")
        layers[0].append(c[0])
        layers[1] += [c[1], c[4]]
        layers[2] += [c[2], c[3], c[5], c[6]]

    no_noise()
    model = snac_mod.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    wav = decode(model, layers)
    rms = float(np.sqrt(np.mean(wav ** 2)))
    peak = float(np.max(np.abs(wav)))
    print(f"{len(layers[0])} frames, stop token {stop}, "
          f"{wav.size} samples, rms {rms:.4f}, peak {peak:.3f}")
    # Only real speech makes this a better check than the near-silent
    # greedy golden (RMS 0.0014).
    if stop is None or rms < 0.02 or peak < 0.2:
        sys.exit("not audible, terminated speech; try another --seed")
    (GOLDEN / "snac_speech_codes.txt").write_text(
        "".join(" ".join(map(str, l)) + "\n" for l in layers))
    wav.tofile(GOLDEN / "snac_speech_waveform.f32")
    print("wrote snac_speech_codes.txt, snac_speech_waveform.f32")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synth", help="synthetic-model value fixtures")
    sp = sub.add_parser("speech", help="sampled-speech decoder fixtures")
    sp.add_argument("--gguf", type=pathlib.Path, required=True,
                    help="Orpheus-TTS GGUF")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--voice", default="tara")
    sp.add_argument("--text",
                    default="The quick brown fox jumps over the lazy dog.")
    sp.add_argument("--max-tokens", type=int, default=1200)
    a = ap.parse_args()
    {"synth": cmd_synth, "speech": cmd_speech}[a.cmd](a)


if __name__ == "__main__":
    main()
