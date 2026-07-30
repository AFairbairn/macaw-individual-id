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
    Record the current environment:
        python src/08_freeze_environment.py freeze

    Check the current environment against the record:
        python src/08_freeze_environment.py verify

    Check the record, and stop the pipeline if anything differs:
        python src/08_freeze_environment.py verify --strict

INPUT
    The installed Python packages of the active environment.
    The downloaded model weight files of bacpipe.
    environment.lock/    Verify mode reads the recorded values from here.

OUTPUT
    environment.lock/packages.txt      Every installed package and its version.
    environment.lock/model_weights.csv An md5 checksum for every weight file.
    environment.lock/platform.json     Python, OS, CUDA, and driver versions.

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
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = ROOT / "environment.lock"
PACKAGES = LOCK_DIR / "packages.txt"
WEIGHTS = LOCK_DIR / "model_weights.csv"
PLATFORM = LOCK_DIR / "platform.json"

CHUNK_BYTES = 1 << 20  # Read 1 MiB at a time, so large weight files fit in memory.

# File types that hold model weights. bacpipe stores weights under several
# names, so the script matches on the suffix rather than on the file name.
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".tflite", ".pb", ".h5", ".onnx"}

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


def installed_packages():
    """Return the output of `pip freeze` as a list of lines.

    `pip freeze` reports the exact installed version of every package. It is
    more precise than requirements.txt, which can hold a version range.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line.strip())


def weight_search_paths():
    """Return the directories that can hold downloaded model weights.

    The list covers the caches that bacpipe and its dependencies use. A missing
    directory is skipped, not an error.
    """
    home = Path.home()
    candidates = [
        home / ".cache/huggingface",
        home / ".cache/torch",
        home / ".cache/kagglehub",
        home / ".cache/tfhub_modules",
        home / ".keras",
        ROOT / "model_weights",
    ]

    # Add the directory of the installed bacpipe package, if it is present.
    try:
        import bacpipe  # noqa: F401
        candidates.append(Path(bacpipe.__file__).parent)
    except Exception:
        pass

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
    """Return the platform details that affect a numerical result."""
    record = {
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


def do_freeze():
    """Write the current environment to the lock directory."""
    LOCK_DIR.mkdir(exist_ok=True)

    packages = installed_packages()
    PACKAGES.write_text("\n".join(packages) + "\n")

    weights = find_weight_files()
    with open(WEIGHTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "md5"])
        writer.writeheader()
        writer.writerows(weights)

    PLATFORM.write_text(json.dumps(platform_record(), indent=2) + "\n")

    print(f"Wrote {LOCK_DIR}")
    print(f"  packages.txt        {len(packages)} packages")
    print(f"  model_weights.csv   {len(weights)} weight files")
    print(f"  platform.json       {platform_record()['platform']}")

    if not weights:
        print()
        print("  Warning: no weight files were found.")
        print("  Run this script after stage 1 has downloaded the models.")


def do_verify(strict):
    """Compare the current environment against the lock directory.

    The function returns 0 when everything matches. It returns 1 when something
    differs and --strict is set.
    """
    if not PACKAGES.exists():
        print(f"No record found at {LOCK_DIR}.")
        print("Create one with: python src/08_freeze_environment.py freeze")
        return 1 if strict else 0

    problems = []

    recorded = set(PACKAGES.read_text().splitlines())
    current = set(installed_packages())

    changed = sorted(recorded ^ current)
    if changed:
        problems.append(f"{len(changed)} package difference(s)")
        print("PACKAGE DIFFERENCES")
        for line in changed[:20]:
            marker = "recorded" if line in recorded else "current "
            print(f"  [{marker}] {line}")
        if len(changed) > 20:
            print(f"  ... and {len(changed) - 20} more")
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
        print("The environment matches the record.")
        return 0

    print(f"SUMMARY: {', '.join(problems)}")
    if strict:
        print("Stopping, because --strict is set.")
        return 1
    print("Continuing. Results can differ from the published values.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Record or verify the software and the model weights."
    )
    parser.add_argument("action", choices=["freeze", "verify"])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if the environment differs from the record.",
    )
    args = parser.parse_args()

    if args.action == "freeze":
        do_freeze()
        return 0
    return do_verify(args.strict)


if __name__ == "__main__":
    sys.exit(main())
