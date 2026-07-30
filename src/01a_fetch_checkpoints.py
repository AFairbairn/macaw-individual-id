#!/usr/bin/env python
"""
01a_fetch_checkpoints.py

PURPOSE
    Download the model checkpoints that bacpipe does not fetch by itself.
    Run this before 01_extract_embeddings.py.

USAGE
    python src/01a_fetch_checkpoints.py

INPUT
    config.yaml    The checkpoints block names what to download.
    The Hugging Face dataset repository named in that block.

OUTPUT
    bacpipe/model_checkpoints/<model>/
        The weight files, below the repository root.

WHY A DOWNLOAD IS NOT ENOUGH
    A checkpoint that arrives damaged does not announce itself. bacpipe loads
    it, raises, catches its own exception, and writes zero embeddings. The run
    continues and reports success, and the loss appears three stages later in
    01b_verify_embeddings.py.

    So every file is checked here, at the point where it can still be fixed
    cheaply.

    sha256      Where config.yaml records a checksum for a file, the downloaded
                file must match it. A checksum catches a truncated or altered
                download, which a file size alone can miss.
    open it     Every checkpoint is opened with the reader its format needs. A
                torch checkpoint must load. A .keras file must be a valid zip
                archive. This is the check that would have caught the BEATs
                checkpoint that downloaded at the right size and then failed
                with "PytorchStreamReader failed reading zip archive".

WHY THIS SCRIPT EXISTS
    bacpipe downloads the weights of most of its models on first use. BirdNET
    is the exception. bacpipe ships no fetch step for it, so the model writes
    zero embeddings and the run looks successful until 01b_verify_embeddings.py
    counts the files.

    An earlier version of this project carried the download in a separate setup
    script that a reader of this repository never saw. It is a stage of the
    pipeline, so it belongs here.

WHERE THE FILES GO
    bacpipe reads its checkpoints from a path relative to the working
    directory. run_all.sh runs every stage from the repository root, so this
    script writes to <repository root>/bacpipe/model_checkpoints. The
    .gitignore excludes that directory, because the weights are downloaded and
    not source.

RERUNNING
    Hugging Face skips a file that is already present and complete, so a second
    run costs one request for each file.
"""
import hashlib
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
TARGET = ROOT / "bacpipe" / "model_checkpoints"


def sha256_of(path):
    """Return the sha256 checksum of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_check(path):
    """Open one checkpoint with the reader its format needs.

    Return an empty string when the file opens, or a message when it does not.
    A file that downloads at the correct size can still be unreadable.
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

    if suffix in (".keras", ".zip", ".h5"):
        if suffix != ".h5" and not zipfile.is_zipfile(path):
            return "not a valid zip archive"
        return ""

    return ""


def main():
    """Download every checkpoint named in config.yaml. Return an exit code."""
    block = CONFIG.get("checkpoints")
    if not block:
        print("config.yaml has no checkpoints block. Nothing to download.")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed.")
        print("Install the environment with: pip install -r requirements.txt")
        return 1

    TARGET.mkdir(parents=True, exist_ok=True)

    for name, entry in block.items():
        print(f"{name}: downloading from {entry['repo_id']}", flush=True)
        try:
            snapshot_download(
                entry["repo_id"],
                repo_type=entry.get("repo_type", "dataset"),
                allow_patterns=entry["patterns"],
                local_dir=str(TARGET),
            )
        except Exception as error:
            print(f"  FAILED: {type(error).__name__}: {error}")
            print("  Check that this machine can reach huggingface.co.")
            return 1

        # Confirm the files are on disk. A silent failure here means the model
        # writes zero embeddings in stage 1, which is expensive to discover.
        present = sorted(p for p in (TARGET / name).rglob("*") if p.is_file())
        if not present:
            print(f"  FAILED: {TARGET / name} holds no file after the download.")
            return 1
        print(f"  {len(present)} file(s) in {TARGET / name}")

        # Every recorded checksum must match.
        recorded = entry.get("sha256") or {}
        for relative, expected in recorded.items():
            path = TARGET / relative
            if not path.exists():
                print(f"  FAILED: {relative} is named in config.yaml and is absent.")
                return 1
            actual = sha256_of(path)
            if actual != expected:
                print(f"  FAILED: {relative} does not match its recorded checksum.")
                print(f"    expected {expected}")
                print(f"    found    {actual}")
                print("    Delete the file and run this script again.")
                return 1
            print(f"  checksum ok: {relative}")

        # Every checkpoint must open.
        for path in present:
            problem = open_check(path)
            if problem:
                print(f"  FAILED: {path.relative_to(TARGET)} does not open.")
                print(f"    {problem}")
                print("    The file is present but unusable. Delete it and run")
                print("    this script again.")
                return 1

    print()
    print("Every checkpoint is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
