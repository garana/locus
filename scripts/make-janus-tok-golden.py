#!/usr/bin/env python3
"""Writes tests/golden/janus_tokenizer.jsonl: the HF tokenizer's ids
(no special tokens) for a corpus that covers the deepseek-llm
pre-tokenizer's splits, so tests/test_bpe_tokenizer.cpp can check
locus's split token for token (i#226).

Needs the Janus-Pro snapshot directory (tokenizer.json). Tested with
Python 3.10, transformers 4.47.1, Janus-Pro-1B revision
960ab33191f61342a4c60ae74d8dc356a39fafcb:

    python3 scripts/make-janus-tok-golden.py --model DIR
"""
import argparse
import json
import pathlib

from transformers import AutoTokenizer

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden"

CORPUS = [
    "The capital of France is",
    "User: A red apple on a wooden table.\n\nAssistant:",
    "12345 and 3.14 and 2026-10-25",
    "a\n\nb\r\nc\n",
    "tabs\there\tand  two  spaces",
    "trailing spaces   ",
    "   leading spaces",
    "你好，世界！这是一个测试。",
    "안녕하세요 세계",
    "こんにちは世界、カタカナ",
    "مرحبا بالعالم",
    "Привет, мир! Ёлка",
    "Γειά σου Κόσμε",
    "café naïve façade Straße",
    "I ❤ \U0001F34E and \U0001F600!",
    "ＡＢＣ１２３ｘｙｚ",
    "Hello, world! (test) [x] {y} <z> \"q\" 'r'",
    "def f(x):\n    return x**2  # square\n",
    "https://example.com/a?b=1&c=two#frag",
    "½ ² ① ٣ ४",
    "Bengali ৩৪ Thai ๕๖ Tamil ௧௨",
    "reversed Ↄ and ↄ are letters, ↂ is a number",
    "a　b　　c",
    "mixed中文English123混合",
    "‘quoted’ “double” — dash … ellipsis",
    "x" * 40,
    "",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Janus-Pro snapshot")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    path = GOLDEN / "janus_tokenizer.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for s in CORPUS:
            ids = tok.encode(s, add_special_tokens=False)
            f.write(json.dumps({"text": s, "ids": ids}, ensure_ascii=False) +
                    "\n")
    print(f"wrote {path}: {len(CORPUS)} cases")


if __name__ == "__main__":
    main()
