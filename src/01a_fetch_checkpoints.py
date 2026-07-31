#!/usr/bin/env python
"""
01a_fetch_checkpoints.py

PURPOSE
    Check every model checkpoint that bacpipe has downloaded, and delete any
    file that will not open. bacpipe downloads a checkpoint again when its
    directory is absent, so a deleted file is replaced on the next run.

USAGE
    python src/01a_fetch_checkpoints.py

INPUT
    The bacpipe settings file, for the checkpoint directory.
    config.yaml, for the checksum of a file where one is recorded.

OUTPUT
    A report on the terminal. The exit code is 0 when every checkpoint opens.

WHY THIS SCRIPT EXISTS
    bacpipe downloads its own checkpoints. It decides whether to download by
    one test: does the directory of that model exist and hold at least one
    file. A download that stops part of the way through leaves a short file in
    that directory, so the test passes and bacpipe never downloads again. The
    model then fails on every run, with an error that names the file format
    rather than the cause.

    That happened here. A download stopped when the disk quota ran out and left
    113,246,208 bytes of a 363,145,291 byte BEATs checkpoint. Every later run
    read the short file and raised
    "PytorchStreamReader failed reading zip archive: failed finding central
    directory", and the model wrote no embeddings.

    The rule this script applies: an interrupted download must not become
    permanent. Open every checkpoint. Delete what does not open. bacpipe then
    fetches it again.

WHERE THE CHECKPOINTS ARE
    bacpipe reads model_base_path from settings.yaml inside its own installed
    package. The value is a relative path, so it resolves from the working
    directory, and run_all.sh runs every stage from the repository root.

    This script reads the same setting, so it checks the directory bacpipe
    reads whatever that value is.

HOW A FILE IS CHECKED
    torch checkpoint    It must load.
    keras or zip file   It must be a valid zip archive.
    tar archive         It must open and list its members.
    recorded checksum   Where config.yaml records a sha256 for a file, the file
                        must match it.

    A file of a format this script does not know is reported and left alone.
"""
import hashlib
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

# The default in the bacpipe settings file. It is used when that file cannot be
# read, so the check still runs.
DEFAULT_BASE = "bacpipe_model_checkpoints"


def checkpoint_root():
    """Return the directory bacpipe reads its checkpoints from."""
    try:
        import bacpipe
    except ImportError:
        return ROOT / DEFAULT_BASE

    settings = Path(bacpipe.__file__).parent / "settings.yaml"
    base = DEFAULT_BASE
    if settings.is_file():
        loaded = yaml.safe_load(settings.read_text()) or {}
        base = loaded.get("model_base_path", DEFAULT_BASE)

    path = Path(base)
    return path if path.is_absolute() else ROOT / path


def sha256_of(path):
    """Return the sha256 checksum of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_problem(path):
    """Open one file with the reader its format needs.

    Return an empty string when the file opens, or a message when it does not.
    A file of an unknown format returns an empty string, because this script
    cannot judge it.
    """
    suffix = path.suffix.lower()

    if suffix in (".pt", ".pth", ".ckpt"):
        try:
            import torch
        except ImportError:
            return ""
        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return ""

    if suffix in (".keras", ".zip"):
        if not zipfile.is_zipfile(path):
            return "not a valid zip archive"
        return ""

    if suffix in (".xz", ".gz", ".bz2", ".tar"):
        try:
            with tarfile.open(path) as archive:
                archive.getmembers()
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return ""

    return ""


def recorded_checksums():
    """Return the sha256 recorded in config.yaml, keyed by relative path."""
    recorded = {}
    for entry in (CONFIG.get("checkpoints") or {}).values():
        recorded.update(entry.get("sha256") or {})
    return recorded


def main():
    """Check every checkpoint. Delete what does not open. Return an exit code."""
    root = checkpoint_root()
    print(f"Checkpoint directory: {root}")

    if not root.is_dir():
        print("  No checkpoint has been downloaded yet. bacpipe downloads what")
        print("  it needs when it runs.")
        return 0

    checksums = recorded_checksums()
    checked = removed = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # The download cache holds lock and metadata files, not checkpoints.
        if ".cache" in path.parts:
            continue

        relative = path.relative_to(root).as_posix()
        problem = ""

        expected = checksums.get(relative)
        if expected:
            actual = sha256_of(path)
            if actual != expected:
                problem = (f"the checksum does not match config.yaml. "
                           f"expected {expected}, found {actual}")

        if not problem:
            problem = open_problem(path)

        checked += 1
        if not problem:
            continue

        print(f"  {relative}: {problem}")
        print(f"    {path.stat().st_size} bytes. Deleting the model directory,")
        print("    so bacpipe downloads this checkpoint again.")

        # Delete the whole model directory. bacpipe decides whether to download
        # by whether that directory exists and holds a file, so one file left
        # behind stops the download.
        model_dir = root / path.relative_to(root).parts[0]
        shutil.rmtree(model_dir, ignore_errors=True)
        removed += 1

    print(f"  {checked} file(s) checked, {removed} model directory removed.")
    if removed:
        print()
        print("bacpipe downloads the removed checkpoints when stage 1 runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
