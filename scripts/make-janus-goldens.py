#!/usr/bin/env python3
"""Makes the Janus-Pro-1B / VQ-16 parity fixtures under tests/golden
that locus's image path is compared against (i#221 section 6).

  synth   No model needed. A tiny VQ-16 decoder (the Janus Decoder
          class with ch 32, ch_mult (1, 2), num_res_blocks 1, z 32,
          codebook 16 x 8, l2-normalised) filled from the test's LCG in
          SORTED tensor-name order, decoding a fixed 3 x 3 code grid.
          tests/test_vq16.cpp builds the same model in memory, so the
          decoder's values are checked on every box.
  tokens  Janus-Pro-1B in float32 on the CPU. The reference prompt
          wrapping and CFG (scale 5) with ARGMAX instead of
          torch.multinomial (which no other implementation reproduces
          from a seed), 576 image tokens fed to both sequences.
  decode  The real VQ-16 decoder in float32 on the tokens from
          `tokens`. Writes the (3, 384, 384) f32 output and the PNG,
          with the reference's truncating uint8 conversion.
  vision  Image understanding (i#241) on janus_apple.png (a copy of
          an earlier janus_decode.png, kept apart so the generation
          goldens can change), which is 384 x 384, so the
          preprocessing has no resize and no pad. The reference
          processor and SFT template with Janus-Pro's <|User|> /
          <|Assistant|> roles and one question, then greedy decoding.
          Writes janus_vision.f32 (VISION_ROWS sampled patches of the
          SigLIP output, 1024 wide, then the
          same rows of the aligner output, 2048 wide) and
          janus_understand.txt (the expanded prompt ids, the answer
          ids, then the RMS of the whole vision and aligner outputs).

    python3 scripts/make-janus-goldens.py synth
    python3 scripts/make-janus-goldens.py tokens --janus SRC --model DIR
    python3 scripts/make-janus-goldens.py decode --janus SRC --model DIR
    python3 scripts/make-janus-goldens.py vision --janus SRC --model DIR

SRC is a checkout of github.com/deepseek-ai/Janus and DIR the
deepseek-ai/Janus-Pro-1B snapshot (pytorch_model.bin, loaded with
torch's weights_only default). Tested with Python 3.10, torch 2.14.1,
transformers 4.47.1, numpy 2.2.6, Janus commit 1daa72f, Janus-Pro-1B
revision 960ab33191f61342a4c60ae74d8dc356a39fafcb. With those, the
outputs' sha256 (on an M2 Pro, CPU):

  janus_tokens.txt
    0ce2d1d59db0d0d99471cd275c0ce37be9eceb824eb9f9e1820bf1bf122df703
  janus_decode.f32
    f14949e191b8bc16924312e52d8d4e8d07a2816e2882d170ff0597bb743206a8
  janus_decode.png
    f30912bd5a7ae5a120960ea564d566eee5c984ea3c2414b83aef285476672412
  janus_vision.f32
    99379afcac1e6a73f5acbc6d41c195d9ed2c6a612fe8edb845bd862c69ee0225
  janus_understand.txt
    6f91da23cbd9ceade9a62e416fa5934bf7b3c8e7ce45fd99256a628501279105

The tokens are an f32 CPU argmax, so another machine can tip a near
tie; regenerate there and compare before trusting a mismatch.
"""
import argparse
import pathlib
import sys
import zlib
import struct

import numpy as np
import torch

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden"

PROMPT = "A red apple on a wooden table."
CFG = 5.0
IMAGE_TOKENS = 576
GRID = 24
PAD_ID = 100002

# Image understanding (keep in sync with tests/test_janus_vision.cpp).
QUESTION = "What is on the table?"
VISION_ROWS = list(range(0, 576, 36))  # 16 sampled patches
ANSWER_TOKENS = 24

# The synthetic decoder (keep in sync with tests/test_vq16.cpp).
SYNTH = dict(z_channels=32, ch=32, ch_mult=(1, 2), num_res_blocks=1)
SYNTH_CODEBOOK = (16, 8)
SYNTH_GRID = 3


def synth_codes():
    """@returns The fixed 3 x 3 code grid (row-major) the test decodes."""
    return [(i * 5 + 3) % SYNTH_CODEBOOK[0] for i in range(SYNTH_GRID ** 2)]


def lcg_fill(tensors):
    """
    Fills each tensor, in sorted name order and row-major element order,
    from the test's LCG: state = state * 1664525 + 1013904223 (mod 2^32),
    value = ((state >> 8) / 2^24 - 0.5) * 0.5. Keep in sync with
    tests/test_vq16.cpp.
    """
    state = 12345
    for name in sorted(tensors):
        t = tensors[name]
        flat = t.view(-1)
        vals = np.empty(flat.numel(), dtype=np.float32)
        for i in range(flat.numel()):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            vals[i] = (np.float32(state >> 8) / np.float32(16777216.0) -
                       np.float32(0.5)) * np.float32(0.5)
        flat.copy_(torch.from_numpy(vals))


def import_janus(src):
    sys.path.insert(0, str(pathlib.Path(src).resolve()))
    from janus.models import vq_model  # noqa: E402
    return vq_model


def cmd_synth(args):
    vq = import_janus(args.janus) if args.janus else None
    if vq is None:
        sys.exit("synth needs --janus SRC (for the reference Decoder class)")
    torch.manual_seed(0)
    dec = vq.Decoder(**SYNTH).eval()
    post = torch.nn.Conv2d(SYNTH_CODEBOOK[1], SYNTH["z_channels"], 1)
    emb = torch.nn.Embedding(*SYNTH_CODEBOOK)
    tensors = {"quantize.embedding.weight": emb.weight.data}
    for n, p in post.named_parameters():
        tensors["post_quant_conv." + n] = p.data
    for n, p in dec.named_parameters():
        tensors["decoder." + n] = p.data
    with torch.no_grad():
        lcg_fill(tensors)
        e = torch.nn.functional.normalize(emb.weight, p=2, dim=-1)
        z = e[torch.tensor(synth_codes())]
        z = z.reshape(1, SYNTH_GRID, SYNTH_GRID, -1).permute(0, 3, 1, 2)
        out = dec(post(z.contiguous()))
    a = out[0].numpy().astype("<f4")
    path = GOLDEN / "vq16_synth.f32"
    path.write_bytes(a.tobytes())
    print(f"wrote {path}: shape {tuple(a.shape)}, "
          f"rms {float(np.sqrt((a ** 2).mean())):.4f}, "
          f"tensors {len(tensors)}")


def load_janus(args):
    sys.path.insert(0, str(pathlib.Path(args.janus).resolve()))
    from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa
    proc = VLChatProcessor.from_pretrained(args.model)
    model = MultiModalityCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).eval()
    return proc, model


def prompt_ids(proc):
    # Janus-Pro's roles (its own special tokens, the README's Janus-Pro
    # text-to-image example); "User" / "Assistant" are Janus-1.3B's.
    conv = [{"role": "<|User|>", "content": PROMPT},
            {"role": "<|Assistant|>", "content": ""}]
    text = proc.apply_sft_template_for_multi_turn_prompts(
        conversations=conv, sft_format=proc.sft_format, system_prompt="")
    text += proc.image_start_tag
    return proc.tokenizer.encode(text)


@torch.inference_mode()
def cmd_tokens(args):
    proc, model = load_janus(args)
    ids = prompt_ids(proc)
    uncond = list(ids)
    for i in range(1, len(ids) - 1):
        uncond[i] = PAD_ID
    tokens = torch.tensor([ids, uncond], dtype=torch.long)
    embeds = model.language_model.get_input_embeddings()(tokens)
    out_ids = []
    past = None
    for i in range(IMAGE_TOKENS):
        o = model.language_model.model(inputs_embeds=embeds, use_cache=True,
                                       past_key_values=past)
        past = o.past_key_values
        logits = model.gen_head(o.last_hidden_state[:, -1, :])
        cond, unc = logits[0], logits[1]
        guided = unc + CFG * (cond - unc)
        nxt = int(torch.argmax(guided))
        out_ids.append(nxt)
        pair = torch.tensor([nxt, nxt], dtype=torch.long)
        embeds = model.prepare_gen_img_embeds(pair).unsqueeze(1)
        if i % 96 == 0:
            print(f"token {i}: {nxt}", flush=True)
    path = GOLDEN / "janus_tokens.txt"
    path.write_text(" ".join(map(str, ids)) + "\n" +
                    " ".join(map(str, uncond)) + "\n" +
                    " ".join(map(str, out_ids)) + "\n")
    print(f"wrote {path}: prompt {len(ids)} ids, {len(out_ids)} image ids")


def write_png(path, rgb):
    """8-bit RGB PNG, stored deflate (no compression), as locus writes."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 0))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


@torch.inference_mode()
def cmd_decode(args):
    _, model = load_janus(args)
    lines = (GOLDEN / "janus_tokens.txt").read_text().splitlines()
    codes = torch.tensor([[int(x) for x in lines[2].split()]],
                         dtype=torch.int)
    dec = model.gen_vision_model.decode_code(
        codes, shape=[1, 8, GRID, GRID])
    a = dec[0].float().numpy().astype("<f4")
    (GOLDEN / "janus_decode.f32").write_bytes(a.tobytes())
    hwc = np.clip((a.transpose(1, 2, 0) + 1) / 2 * 255, 0, 255)
    img = np.zeros(hwc.shape, dtype=np.uint8)
    img[:, :, :] = hwc  # truncating, as the reference
    write_png(GOLDEN / "janus_decode.png", img)
    print(f"wrote janus_decode.f32 {tuple(a.shape)} and janus_decode.png")


@torch.inference_mode()
def cmd_vision(args):
    from PIL import Image  # noqa: E402
    proc, model = load_janus(args)
    img = Image.open(GOLDEN / "janus_apple.png").convert("RGB")
    assert img.size == (384, 384), img.size
    # Janus-Pro's roles (its own special tokens, the README's Janus-Pro
    # examples); "User" / "Assistant" are Janus-1.3B's.
    conv = [{"role": "<|User|>",
             "content": "<image_placeholder>\n" + QUESTION},
            {"role": "<|Assistant|>", "content": ""}]
    inp = proc(conversations=conv, images=[img], force_batchify=True)
    # The processor's pixels must be exactly (x / 255 - 0.5) / 0.5 here.
    raw = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)
    want = (raw / 255.0 - 0.5) / 0.5
    px = inp.pixel_values[0, 0]
    print(f"pixel max |diff| vs (x/255-0.5)/0.5: "
          f"{float((px - want).abs().max()):.3g}")
    vis = model.vision_model(inp.pixel_values[0])[0]  # (576, 1024)
    ali = model.aligner(vis.unsqueeze(0))[0]          # (576, 2048)
    rows = torch.tensor(VISION_ROWS)
    out = np.concatenate([vis[rows].numpy().ravel(),
                          ali[rows].numpy().ravel()]).astype("<f4")
    (GOLDEN / "janus_vision.f32").write_bytes(out.tobytes())
    embeds = model.prepare_inputs_embeds(**inp)
    gen = model.language_model.generate(
        inputs_embeds=embeds, attention_mask=inp.attention_mask,
        pad_token_id=proc.tokenizer.eos_token_id,
        bos_token_id=proc.tokenizer.bos_token_id,
        eos_token_id=proc.tokenizer.eos_token_id,
        max_new_tokens=ANSWER_TOKENS, do_sample=False, use_cache=True)
    answer = gen[0].tolist()
    ids = inp.input_ids[0].tolist()

    def rms(t):
        return float(t.pow(2).mean().sqrt())
    (GOLDEN / "janus_understand.txt").write_text(
        " ".join(map(str, ids)) + "\n" + " ".join(map(str, answer)) + "\n" +
        f"{rms(vis):.9g} {rms(ali):.9g}\n")
    print(f"prompt {len(ids)} ids, answer {len(answer)} ids: "
          f"{proc.tokenizer.decode(answer)!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["synth", "tokens", "decode", "vision"])
    ap.add_argument("--janus", help="checkout of deepseek-ai/Janus")
    ap.add_argument("--model", help="Janus-Pro-1B snapshot directory")
    args = ap.parse_args()
    if args.stage != "synth" and not (args.janus and args.model):
        sys.exit("tokens / decode need --janus and --model")
    {"synth": cmd_synth, "tokens": cmd_tokens, "decode": cmd_decode,
     "vision": cmd_vision}[args.stage](args)


if __name__ == "__main__":
    main()
