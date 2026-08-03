#!/usr/bin/env python
"""
08_freeze_environment.py

PURPOSE
    Record the exact software and the exact model weights that produced the
    published results. Verify a later environment against that record.

WHY THIS SCRIPT EXISTS
    A pinned version number is not enough. Two risks remain.

    1. A package version can be a range, so a rerun installs a different build.
    2. A model weight file can change upstream while its version number stays
       the same. bacpipe downloads weights from external hosts. Those hosts can
       replace a file without notice.

    This script cannot stop an upstream change. It makes an upstream change
    visible. That is the difference between a result you can defend and a result
    you hope is still correct.

USAGE
    Record the environment of one profile. Run each with that profile's own
    interpreter, because this script reads the packages of the interpreter that
    runs it:

        .venv-extraction/bin/python src/08_freeze_environment.py freeze --profile extraction
        .venv-analysis/bin/python   src/08_freeze_environment.py freeze --profile analysis

    Check an environment against its record:
        python src/08_freeze_environment.py verify --profile analysis

INPUT
    The installed Python packages of the interpreter that runs this script.
    The downloaded model weight files of bacpipe, under the extraction profile.
    environment.lock/<profile>/   Verify mode reads the recorded values here.

OUTPUT
    environment.lock/<profile>/packages.txt       Every package and its version.
    environment.lock/<profile>/platform.json      Python, OS, CUDA, env_hash.
    environment.lock/extraction/model_weights.csv An md5 for every weight file.

WHY THERE ARE TWO PROFILES
    Stage 1 imports bacpipe, which pins the whole numerical and audio stack.
    Every other stage imports none of it. Holding both in one environment made
    every result table hostage to whatever bacpipe resolved, and on 2026-08-01
    that wrote one results tree from two different stacks.

WHY THE WEIGHTS ARE RECORDED UNDER THE EXTRACTION PROFILE ONLY
    Only stage 1 reads a model checkpoint. Recording the weights under the
    analysis profile as well would put a checksum in a record whose environment
    never opens the file.

    Record them only after every model has produced embeddings. On 2026-07-31 a
    checksum was recorded for a BEATs checkpoint that could not load, so the
    record agreed with a file that did not work.

WHEN TO RUN 'freeze'
    Run it once, on the machine that produced the published results. Commit the
    environment.lock directory. Do not regenerate it after that point.

WHEN TO RUN 'verify'
    Run it before every rerun. run_all.sh runs it automatically.

NOTE
    Run 'freeze' on the machine that holds the working environment. This script
    reads the installed packages and the downloaded weights. It cannot record an
    environment that is not present.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import common

ROOT = common.ROOT
LOCK_ROOT = ROOT / "environment.lock"

PROFILES = ["extraction", "analysis"]

# Only stage 1 opens a model checkpoint, and stage 1 runs in the extraction
# profile. See WHY THE WEIGHTS ARE RECORDED UNDER THE EXTRACTION PROFILE ONLY.
WEIGHTS_PROFILE = "extraction"


def lock_dir(profile):
    """Return the directory that holds the record of one profile."""
    return LOCK_ROOT / profile

CHUNK_BYTES = 1 << 20  # Read 1 MiB at a time, so large files do not fill memory.

# File types that hold model weights. bacpipe stores weights under several
# names, so the script matches on the suffix rather than on the file name.
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".tflite",
                   ".pb", ".h5", ".onnx", ".keras"}

# The smallest file that is treated as a weight file. A smaller file is usually
# a config file or a tokenizer, not a weight.
MIN_WEIGHT_BYTES = 100_000


def md5_of(path):
    """Return the md5 checksum of one file as a hexadecimal string."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The package list and the hash of it come from common, so that the record this
# script writes and the stamp every result row carries are the same thing
# computed the same way.
installed_packages = common.installed_packages
env_hash = common.env_hash


def weight_search_paths():
    """Return the directories that can hold downloaded model weights.

    The list covers the caches that bacpipe and its dependencies use. A missing
    directory is skipped, not an error.
    """
    home = Path.home()
    cache = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))

    # A cluster commonly redirects these caches to scratch. Read the variables
    # the libraries themselves read, so the record covers the weights that were
    # actually used rather than an empty default directory.
    candidates = [
        Path(os.environ.get("HUGGINGFACE_HUB_CACHE",
                            os.environ.get("HF_HOME", cache / "huggingface"))),
        Path(os.environ.get("TORCH_HOME", cache / "torch")),
        cache / "kagglehub",
        cache / "tfhub_modules",
        Path(os.environ.get("KERAS_HOME", home / ".keras")),
        # The checkpoints this pipeline downloads itself. These are the two the
        # project pins a checksum for, so they are the ones the record must
        # cover.
        ROOT / "bacpipe" / "model_checkpoints",
    ]

    # Add the directory of the installed bacpipe package, if it is present.
    try:
        import bacpipe  # noqa: F401
        candidates.append(Path(bacpipe.__file__).parent)
    except Exception as error:
        print(f"  bacpipe did not import, so its directory is not searched. "
              f"{type(error).__name__}: {error}")

    return [p for p in candidates if p.exists()]


def find_weight_files():
    """Return every model weight file that is present, with its checksum.

    The function reports the path relative to the home directory, so the record
    is the same on a different machine.
    """
    rows = []
    for base in weight_search_paths():
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            if path.stat().st_size < MIN_WEIGHT_BYTES:
                continue
            try:
                relative = path.relative_to(Path.home()).as_posix()
            except ValueError:
                relative = path.as_posix()
            rows.append(
                {"path": relative, "bytes": path.stat().st_size, "md5": md5_of(path)}
            )
    return sorted(rows, key=lambda r: r["path"])


def platform_record():
    """Return the platform details that affect a numerical result.

    env_hash is the md5 of the sorted package list. Every result row carries the
    same value, so a table can be traced to the record that describes it.
    """
    record = {
        "env_hash": env_hash(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch
        record["torch"] = torch.__version__
        record["cuda_available"] = bool(torch.cuda.is_available())
        record["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        record["torch"] = None
    try:
        import numpy
        record["numpy"] = numpy.__version__
    except Exception:
        record["numpy"] = None
    return record


def do_freeze(profile):
    """Write the environment of one profile to its lock directory."""
    directory = lock_dir(profile)
    directory.mkdir(parents=True, exist_ok=True)

    packages = installed_packages()
    (directory / "packages.txt").write_text("\n".join(packages) + "\n")

    record = platform_record()
    (directory / "platform.json").write_text(json.dumps(record, indent=2) + "\n")

    print(f"Wrote {directory}")
    print(f"  packages.txt        {len(packages)} packages")
    print(f"  platform.json       env_hash {record['env_hash'][:12]}, {record['platform']}")

    if profile != WEIGHTS_PROFILE:
        return

    weights = find_weight_files()
    with open(directory / "model_weights.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "md5"])
        writer.writeheader()
        writer.writerows(weights)
    print(f"  model_weights.csv   {len(weights)} weight files")

    if not weights:
        print()
        print("  Warning: no weight file was found.")
        print("  Run this after stage 1 has produced the embeddings, so that")
        print("  every recorded checksum describes a file that works.")


def do_verify(profile, strict):
    """Compare the running environment against the record of one profile.

    The function returns 0 when everything matches, and 1 when it does not.
    run_all.sh stops the run on a 1, unless --allow-env-drift was given.
    """
    directory = lock_dir(profile)
    PACKAGES = directory / "packages.txt"
    WEIGHTS = directory / "model_weights.csv"

    if not PACKAGES.exists():
        print(f"No record found at {directory}.")
        print(f"Create one with: python src/08_freeze_environment.py freeze --profile {profile}")
        return 1 if strict else 0

    problems = []

    def as_mapping(lines):
        """Turn 'name==version' lines into a name to version mapping."""
        out = {}
        for line in lines:
            if "==" in line:
                name, version = line.split("==", 1)
                out[name.strip()] = version.strip()
        return out

    recorded = as_mapping(PACKAGES.read_text(encoding="utf-8").splitlines())
    current = as_mapping(installed_packages())

    # Report the three kinds separately. A changed version matters more than a
    # package that one environment holds and the other does not.
    version_changed = sorted(
        (name, recorded[name], current[name])
        for name in recorded.keys() & current.keys()
        if recorded[name] != current[name]
    )
    only_recorded = sorted(recorded.keys() - current.keys())
    only_current = sorted(current.keys() - recorded.keys())

    if version_changed or only_recorded or only_current:
        total = len(version_changed) + len(only_recorded) + len(only_current)
        problems.append(f"{total} package difference(s)")
        print("PACKAGE DIFFERENCES")
        for name, was, now in version_changed[:20]:
            print(f"  [version ] {name}: recorded {was}, found {now}")
        for name in only_recorded[:10]:
            print(f"  [absent  ] {name}=={recorded[name]}")
        for name in only_current[:10]:
            print(f"  [added   ] {name}=={current[name]}")
        print()

    if WEIGHTS.exists():
        with open(WEIGHTS) as handle:
            recorded_weights = {r["path"]: r["md5"] for r in csv.DictReader(handle)}
        current_weights = {r["path"]: r["md5"] for r in find_weight_files()}

        missing = sorted(set(recorded_weights) - set(current_weights))
        altered = sorted(
            p for p in set(recorded_weights) & set(current_weights)
            if recorded_weights[p] != current_weights[p]
        )

        if missing:
            problems.append(f"{len(missing)} weight file(s) missing")
            print("MISSING WEIGHT FILES")
            for path in missing[:10]:
                print(f"  {path}")
            print()

        if altered:
            # This is the case the script exists to catch.
            problems.append(f"{len(altered)} weight file(s) changed")
            print("CHANGED WEIGHT FILES")
            print("  A model weight file differs from the published version.")
            print("  Results from this environment will not match the paper.")
            for path in altered[:10]:
                print(f"  {path}")
                print(f"    recorded {recorded_weights[path]}")
                print(f"    current  {current_weights[path]}")
            print()

    if not problems:
        return 0

    print(f"SUMMARY: {', '.join(problems)} in the {profile} environment.")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Record or verify the software and the model weights of one profile."
    )
    parser.add_argument("action", choices=["freeze", "verify"])
    parser.add_argument(
        "--profile",
        required=True,
        choices=PROFILES,
        help="Which environment. Run this with that profile's own interpreter.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat a missing record as a difference, rather than as nothing to compare.",
    )
    args = parser.parse_args()

    if args.action == "freeze":
        do_freeze(args.profile)
        return 0
    return do_verify(args.profile, args.strict)


if __name__ == "__main__":
    sys.exit(main())
