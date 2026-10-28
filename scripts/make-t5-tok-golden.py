#!/usr/bin/env python3
"""Writes tests/golden/t5_tokenizer.jsonl: the ids of transformers'
T5Tokenizer (the SentencePiece unigram model, add_special_tokens so
each line ends with </s>) for a corpus that covers the normalizer
(NFKC through the precompiled charsmap, whitespace collapse, the dummy
prefix), the unigram split and the <extra_id_N> tokens, so
tests/test_t5_tokenizer.cpp can check locus's tokenizer id for id
(i#308).

Needs the tokenizer directory of PixArt-alpha/PixArt-XL-2-512x512
(spiece.model, added_tokens.json, special_tokens_map.json,
tokenizer_config.json). Tested with Python 3.12, transformers 4.46.3,
sentencepiece 0.2.0, revision 50f7021:

    python3 -I scripts/make-t5-tok-golden.py --tokenizer DIR/tokenizer \\
        --out tests/golden/t5_tokenizer.jsonl
"""
import argparse
import json

from transformers import T5Tokenizer

CORPUS = [
    "a small red apple on a wooden table, soft morning light.",
    "A small red apple on a wooden table, soft morning light.",
    "The capital of France is Paris.",
    "12345 and 3.14 and 2026-10-28",
    "tabs\there\tand  two  spaces",
    "   leading and trailing spaces   ",
    "line one\nline two\r\nline three",
    "naïve café résumé",
    "ﬁne ﬂour",
    "Ⅻ ① ㎏ ½ ²",
    "ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３",
    "Straße, Ærøskøbing",
    "你好，世界！",
    "日本語のテキスト",
    "Привет, мир",
    "emoji 🍎 and ✨ sparkles",
    "x <extra_id_0> y <extra_id_99>",
    "don't stop; it's \"quoted\" (and [bracketed])",
    "a" * 40,
    "",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = T5Tokenizer.from_pretrained(a.tokenizer, legacy=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for text in CORPUS:
            ids = tok(text, add_special_tokens=True).input_ids
            f.write(json.dumps({"text": text, "ids": ids},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
