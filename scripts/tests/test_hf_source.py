#!/usr/bin/env python3
"""Unit tests for scripts/hf_source.py. Standard library only:

    python3 -I scripts/tests/test_hf_source.py
"""
import contextlib
import io
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import hf_source  # noqa: E402

COMMIT = "ac7f931d5c0e9a3f1b2c4d6e8f0a1b3c5d7e9f01"


class Cache:
    """A fake hub cache: models--Qwen--Qwen-Image with one snapshot whose
    file links into blobs, as huggingface_hub lays it out."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        repo = self.root / "models--Qwen--Qwen-Image"
        (repo / "blobs").mkdir(parents=True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(COMMIT)
        blob = repo / "blobs" / "0123abcd"
        blob.write_bytes(b"weights")
        self.snap = repo / "snapshots" / COMMIT
        (self.snap / "vae").mkdir(parents=True)
        self.file = self.snap / "vae" / "model.safetensors"
        self.file.symlink_to(blob)


class SnapshotOf(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.c = Cache(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_snapshot_folder(self):
        self.assertEqual(hf_source.snapshot_of(self.c.snap),
                         ("Qwen/Qwen-Image", COMMIT))

    def test_sub_folder_and_linked_file(self):
        want = ("Qwen/Qwen-Image", COMMIT)
        self.assertEqual(hf_source.snapshot_of(self.c.snap / "vae"), want)
        self.assertEqual(hf_source.snapshot_of(self.c.file), want)

    def test_link_to_the_snapshot(self):
        link = pathlib.Path(self.tmp.name) / "qwen-image"
        link.symlink_to(self.c.snap)
        want = ("Qwen/Qwen-Image", COMMIT)
        self.assertEqual(hf_source.snapshot_of(link), want)
        self.assertEqual(
            hf_source.snapshot_of(link / "vae" / "model.safetensors"), want)

    def test_relative_path(self):
        old = os.getcwd()
        os.chdir(self.c.snap)
        try:
            self.assertEqual(hf_source.snapshot_of("vae"),
                             ("Qwen/Qwen-Image", COMMIT))
        finally:
            os.chdir(old)

    def test_repo_without_org(self):
        snap = (pathlib.Path(self.tmp.name) / "models--gpt2" / "snapshots" /
                COMMIT)
        snap.mkdir(parents=True)
        self.assertEqual(hf_source.snapshot_of(snap), ("gpt2", COMMIT))

    def test_not_a_snapshot(self):
        plain = pathlib.Path(self.tmp.name) / "plain"
        plain.mkdir()
        self.assertIsNone(hf_source.snapshot_of(plain))
        short = (pathlib.Path(self.tmp.name) / "models--a--b" / "snapshots" /
                 "main")
        short.mkdir(parents=True)
        self.assertIsNone(hf_source.snapshot_of(short))
        other = pathlib.Path(self.tmp.name) / "x" / "snapshots" / COMMIT
        other.mkdir(parents=True)
        self.assertIsNone(hf_source.snapshot_of(other))


class SourceKeys(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.c = Cache(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_from_a_path(self):
        self.assertEqual(hf_source.source_keys(self.c.snap), {
            "general.source.url": "https://huggingface.co/Qwen/Qwen-Image",
            "general.source.revision": COMMIT})

    def test_from_a_repo_id(self):
        keys = hf_source.source_keys(repo="Qwen/Qwen-Image",
                                     cache=self.tmp.name)
        self.assertEqual(keys["general.source.revision"], COMMIT)

    def test_repo_id_not_cached_or_bad_ref(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(
                hf_source.source_keys(repo="a/b", cache=self.tmp.name), {})
        self.assertIn("not written", err.getvalue())
        ref = (pathlib.Path(self.tmp.name) / "models--Qwen--Qwen-Image" /
               "refs" / "main")
        ref.write_text("not-a-commit\n")
        self.assertIsNone(
            hf_source.cached_revision("Qwen/Qwen-Image", self.tmp.name))

    def test_plain_folder_writes_nothing(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(hf_source.source_keys(self.tmp.name), {})
        self.assertIn(self.tmp.name, err.getvalue())

    def test_add_source(self):
        class Writer:
            def __init__(self):
                self.kv = {}

            def add_string(self, k, v):
                self.kv[k] = v

        w = Writer()
        hf_source.add_source(w, self.c.file)
        self.assertEqual(w.kv["general.source.revision"], COMMIT)
        self.assertEqual(len(w.kv), 2)


class CacheDir(unittest.TestCase):
    def test_env_order(self):
        env = {k: os.environ.pop(k, None) for k in ("HF_HUB_CACHE",
                                                    "HF_HOME")}
        try:
            self.assertEqual(hf_source.cache_dir(),
                             pathlib.Path.home() / ".cache" / "huggingface" /
                             "hub")
            os.environ["HF_HOME"] = "/h"
            self.assertEqual(hf_source.cache_dir(), pathlib.Path("/h/hub"))
            os.environ["HF_HUB_CACHE"] = "/c"
            self.assertEqual(hf_source.cache_dir(), pathlib.Path("/c"))
        finally:
            for k, v in env.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
