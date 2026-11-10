#!/usr/bin/env python3
"""Writes the WordPiece tokenizer goldens of tests/test_wpm_tokenizer.cpp
(i#502):

- tests/golden/bert-vocab.gguf: the tokenizer keys of a BERT GGUF and
  no tensors, so the test runs without the model;
- tests/golden/bert_tokenizer.jsonl: llama.cpp's ids for a corpus that
  covers the normalizer (lowercase, accent stripping by NFD, control
  characters), the word split (whitespace, punctuation, ASCII symbols,
  CJK characters) and the greedy longest match (subwords, unknown
  words), with CLS and SEP added and special-token text not parsed.

Needs the gguf package (0.19.0) and llama.cpp's llama-tokenize (tested
with build 12110, commit cd959bae9) and bge-small-en-v1.5-q8_0.gguf
(ggml-org/bge-small-en-v1.5-Q8_0-GGUF, commit f2068ed):

    python3 -I scripts/make-bert-tok-golden.py \\
        --model tests/models/bge-small-en-v1.5-q8_0.gguf \\
        --llama-tokenize /opt/homebrew/bin/llama-tokenize \\
        --out-dir tests/golden
"""
import argparse
import json
import os
import subprocess
import tempfile

import gguf

CORPUS = [
    "Hello, World!",
    "The capital of France is Paris.",
    "what is the capital of china?",
    "   leading and trailing spaces   ",
    "tabs\there\tand  two  spaces",
    "line one\nline two\r\nline three",
    "Café naïve résumé Ærøskøbing Straße",
    "ÀÉÎÕÜ àéîõü ñ ç",
    "東京 is in 日本; 你好，世界！",
    "日本語のテキスト",
    "Привет, мир",
    "Γειά σου κόσμε",
    "emoji 🍎 and ✨ sparkles",
    "$5 + 3 = 8 | a^b ~ c `d` <e> {f}",
    "12345 and 3.14 and 2026-11-10",
    "e-mail: someone@example.com, http://x.org/a?b=c",
    "unbelievably antidisestablishmentarianism",
    "tokenization of subwords: walking, walked, walks",
    "a [SEP] b [CLS] c [MASK]",
    "zero​width and soft­hyphen",
    "control\u0007bell and nul-free",
    "",
    "a",
    "!!!???...",
]


def copy_vocab(model, out):
    """Copies the general.architecture and tokenizer.* keys of model
    into a tensorless GGUF at out."""
    r = gguf.GGUFReader(model)
    w = gguf.GGUFWriter(out, r.get_field("general.architecture").contents())
    for name, field in r.fields.items():
        if not name.startswith("tokenizer."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(name, field.contents(), vtype, sub_type=sub)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def ids_of(tokenize, model, text):
    """@returns llama-tokenize's ids for text (CLS and SEP added,
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
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    copy_vocab(a.model, os.path.join(a.out_dir, "bert-vocab.gguf"))
    with open(os.path.join(a.out_dir, "bert_tokenizer.jsonl"), "w",
              encoding="utf-8") as f:
        for text in CORPUS:
            ids = ids_of(a.llama_tokenize, a.model, text)
            f.write(json.dumps({"text": text, "ids": ids},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
