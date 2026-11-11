"""A qwen2 GGUF's own weights, dequantized, through the Qwen2 decoder in
torch float64. Prints the top next-token log-probs after the given ids,
then the greedy continuation of N tokens. The reference of
tests/test_model_e2e.cpp's qwen2 golden (i#535): exact arithmetic on the
same quantized weights, with no rounding of the activations.

Needs gguf 0.19.0 and torch 2.5.1 (the golden venv):

    python3 -I scripts/make-qwen2-q8-ref.py \\
        tests/models/qwen2.5-0.5b-instruct-q8_0.gguf 12522,5193,264,882 32
"""
import sys

import gguf
import numpy as np
import torch

r = gguf.GGUFReader(sys.argv[1])
prompt = [int(x) for x in sys.argv[2].split(",")]
n_gen = int(sys.argv[3])
hp = {k[6:]: r.get_field(k).contents() for k in r.fields
      if k.startswith("qwen2.")}
w = {}
for t in r.tensors:
    a = gguf.quants.dequantize(t.data, t.tensor_type)
    w[t.name] = torch.from_numpy(np.asarray(a, dtype=np.float64).reshape(
        [int(x) for x in reversed(t.shape)]))
E = int(hp["embedding_length"])
H = int(hp["attention.head_count"])
KV = int(hp["attention.head_count_kv"])
hd = E // H
eps = float(hp["attention.layer_norm_rms_epsilon"])
base = float(hp["rope.freq_base"])
head = w["output.weight"] if "output.weight" in w else w["token_embd.weight"]


def rms(x, g):
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * g


def lin(x, p):
    y = x @ w[p + ".weight"].T
    return y + w[p + ".bias"] if p + ".bias" in w else y


def rope(x, n):  # x [heads, n, hd], NEOX pairs (i, i + hd/2)
    half = hd // 2
    inv = base ** (-torch.arange(half, dtype=torch.float64) * 2.0 / hd)
    ang = torch.arange(n, dtype=torch.float64)[:, None] * inv[None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * c - b * s, a * s + b * c], dim=-1)


def logprobs(ids):
    n = len(ids)
    x = w["token_embd.weight"][ids]
    mask = torch.full((n, n), float("-inf"), dtype=torch.float64).triu(1)
    for i in range(int(hp["block_count"])):
        p = f"blk.{i}."
        h = rms(x, w[p + "attn_norm.weight"])
        q = rope(lin(h, p + "attn_q").view(n, H, hd).transpose(0, 1), n)
        k = rope(lin(h, p + "attn_k").view(n, KV, hd).transpose(0, 1), n)
        v = lin(h, p + "attn_v").view(n, KV, hd).transpose(0, 1)
        k = k.repeat_interleave(H // KV, dim=0)
        v = v.repeat_interleave(H // KV, dim=0)
        s = torch.softmax(q @ k.transpose(1, 2) / hd ** 0.5 + mask, dim=-1)
        x = x + lin((s @ v).transpose(0, 1).reshape(n, E), p + "attn_output")
        h = rms(x, w[p + "ffn_norm.weight"])
        f = (torch.nn.functional.silu(lin(h, p + "ffn_gate")) *
             lin(h, p + "ffn_up"))
        x = x + lin(f, p + "ffn_down")
    x = rms(x[-1], w["output_norm.weight"])
    return torch.log_softmax(x @ head.T, dim=-1)


with torch.no_grad():
    lp = logprobs(prompt)
    top = torch.topk(lp, 4)
    print("top", [(int(i), round(float(v), 4))
                  for v, i in zip(top.values, top.indices)])
    ids = list(prompt)
    for _ in range(n_gen):
        ids.append(int(torch.argmax(logprobs(ids))))
    print("greedy", ids[len(prompt):])
