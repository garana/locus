"""Records where a converter's input came from in the GGUF it writes.

When the input lies in a Hugging Face cache snapshot
(.../models--<org>--<name>/snapshots/<commit>/...), the GGUF gets:

- general.source.url: https://huggingface.co/<org>/<name> (gguf-py's
  standard key, the one llama.cpp's converter writes);
- general.source.revision: the 40 hex digit commit (a locus key; gguf
  has no standard key for it). locus's X-Model-Files header reports its
  first 12 digits as the version.

Any other input gets neither key, and a note on stderr says so. Standard
library only, so the unit test runs without the converters' venv.
"""
import os
import pathlib
import re
import sys

URL_KEY = "general.source.url"
REVISION_KEY = "general.source.revision"

_COMMIT = re.compile(r"[0-9a-f]{40}")


def _repo_id(cache_dir_name):
    """@returns "org/name" for "models--org--name", else None."""
    prefix = "models--"
    if not cache_dir_name.startswith(prefix):
        return None
    # The hub forbids "--" in names, so each "--" was a "/".
    return cache_dir_name[len(prefix):].replace("--", "/")


def _match(path):
    """@returns (repo id, commit) when path is in a snapshot, else None."""
    parts = path.parts
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] != "snapshots" or not _COMMIT.fullmatch(parts[i + 1]):
            continue
        repo = _repo_id(parts[i - 1])
        if repo:
            return repo, parts[i + 1]
    return None


def snapshot_of(path):
    """
    Finds the Hugging Face snapshot that holds a file or folder.

    A snapshot's files are links into the cache's blobs folder, so the
    path is tried as given, with only its folder resolved (a file
    reached through a linked folder), and fully resolved (a link to the
    snapshot folder itself).

    @param path A file or folder the converter reads.
    @returns (repo id, commit), or None outside a snapshot.
    """
    p = pathlib.Path(path).absolute()
    for c in (p, p.parent.resolve() / p.name, p.resolve()):
        m = _match(c)
        if m:
            return m
    return None


def cache_dir():
    """@returns The hub cache folder, as huggingface_hub picks it."""
    if os.environ.get("HF_HUB_CACHE"):
        return pathlib.Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME")
    if home:
        return pathlib.Path(home) / "hub"
    return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def cached_revision(repo, cache=None):
    """
    @param repo A repo id ("org/name") a library loaded from the cache.
    @param cache The hub cache folder; cache_dir() when None.
    @returns The commit the cache's refs/main names, or None.
    """
    root = pathlib.Path(cache) if cache else cache_dir()
    ref = root / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    try:
        commit = ref.read_text().strip()
    except OSError:
        return None
    return commit if _COMMIT.fullmatch(commit) else None


def source_keys(path=None, repo=None, cache=None):
    """
    @param path The input file or folder, looked up with snapshot_of.
    @param repo A repo id, for an input a library loaded by id; its
        commit comes from cached_revision.
    @param cache The hub cache folder for repo; cache_dir() when None.
    @returns {key: value} for the two keys, or {} when the source is
        unknown (and a note on stderr).
    """
    found = None
    if path is not None:
        found = snapshot_of(path)
    elif repo is not None:
        commit = cached_revision(repo, cache)
        found = (repo, commit) if commit else None
    if not found:
        print(f"note: {path or repo} is not in a Hugging Face snapshot; "
              f"{URL_KEY} and {REVISION_KEY} not written", file=sys.stderr)
        return {}
    return {URL_KEY: "https://huggingface.co/" + found[0],
            REVISION_KEY: found[1]}


def add_source(writer, path=None, repo=None):
    """
    Writes source_keys(path, repo) with a gguf.GGUFWriter.

    @param writer The GGUFWriter, before write_kv_data_to_file.
    @param path See source_keys.
    @param repo See source_keys.
    """
    for k, v in source_keys(path, repo).items():
        writer.add_string(k, v)
