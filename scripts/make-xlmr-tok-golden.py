#!/usr/bin/env python3
"""Writes tests/golden/xlmr_tokenizer.jsonl: llama.cpp's ids for XLM-R's
SentencePiece unigram tokenizer (<s> first, </s> last, special-token
text not parsed) on a multilingual corpus, for
tests/test_t5_tokenizer.cpp (i#502). The vocab is bge-m3's, so the test
needs bge-m3-q8_0.gguf (ggml-org/bge-m3-Q8_0-GGUF, commit 9eba04c).

Needs llama.cpp's llama-tokenize (tested with build 12110, commit
cd959bae9). Python's standard library only:

    python3 -I scripts/make-xlmr-tok-golden.py \\
        --model tests/models/bge-m3-q8_0.gguf \\
        --llama-tokenize /opt/homebrew/bin/llama-tokenize \\
        --out tests/golden/xlmr_tokenizer.jsonl
"""
import argparse
import json
import os
import subprocess
import tempfile

CORPUS = [
    "Hello, World!",
    "The capital of France is Paris.",
    "   leading and trailing spaces   ",
    "tabs\there\tand  two  spaces",
    "line one\nline two\r\nline three",
    "Café naïve résumé Ærøskøbing Straße",
    "ﬁne ﬂour, Ⅻ ① ㎏ ½ ²",
    "ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３",
    "東京は日本の首都です。",
    "你好，世界！",
    "Привет, мир",
    "Γειά σου κόσμε",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "안녕하세요 세계",
    "emoji 🍎 and ✨ sparkles",
    "$5 + 3 = 8 | a^b ~ c `d` <e> {f}",
    "e-mail: someone@example.com, http://x.org/a?b=c",
    "query: how much protein should a female eat",
    "a <s> b </s> c <mask>",
    "",
    "a",
]


def ids_of(tokenize, model, text):
    """@returns llama-tokenize's ids for text (special tokens added,
    special-token text not parsed)."""
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(text.encode("utf-8"))
        path = f.name
    try:
        out = subprocess.run(
            [tokenize, "-m", model, "-f", path, "--ids", "--log-disable",
             "--no-parse-special"],
            check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    return json.loads(out.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-tokenize", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        for text in CORPUS:
            ids = ids_of(a.llama_tokenize, a.model, text)
            f.write(json.dumps({"text": text, "ids": ids},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
