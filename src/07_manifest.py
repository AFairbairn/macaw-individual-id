#!/usr/bin/env python
"""
07_manifest.py

PURPOSE
    Write an md5 checksum for every output file. The manifest lets you confirm
    that a rerun gives identical results.

USAGE
    python src/07_manifest.py

OUTPUT
    results/MANIFEST.csv
        One row for each output file, with these columns:
        path, bytes, md5, modified_utc

HOW TO USE THE MANIFEST
    Compare the manifest of two runs:

        diff <(sort results/MANIFEST.csv) <(sort results_previous/MANIFEST.csv)

    If the two manifests match, the results are identical. If a line differs,
    that file changed between the runs.

WHY THIS MATTERS
    Results are built over many sessions. A checksum catches a number that
    changes quietly. Reading tables by eye does not.

NOTE
    The manifest excludes itself, and it excludes log files. Log files hold
    timestamps, so they differ on every run.
"""
import csv
import datetime
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "MANIFEST.csv"

# Files that change on every run, or that are not results.
EXCLUDE_NAMES = {"MANIFEST.csv"}
EXCLUDE_SUFFIXES = {".log", ".tmp", ".pyc"}

CHUNK_BYTES = 1 << 20  # Read 1 MiB at a time, so large files do not fill memory.


def md5_of(path):
    """Return the md5 checksum of one file as a hexadecimal string."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_result(path):
    """Return True if the file belongs in the manifest."""
    if not path.is_file():
        return False
    if path.name in EXCLUDE_NAMES:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return True


def main():
    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} not found. Run the pipeline first.")

    files = sorted(p for p in RESULTS.rglob("*") if is_result(p))
    if not files:
        raise SystemExit(f"{RESULTS} holds no result files.")

    rows = []
    for path in files:
        stat = path.stat()
        modified = datetime.datetime.fromtimestamp(
            stat.st_mtime, datetime.timezone.utc
        ).isoformat(timespec="seconds")
        rows.append(
            {
                "path": path.relative_to(RESULTS).as_posix(),
                "bytes": stat.st_size,
                "md5": md5_of(path),
                "modified_utc": modified,
            }
        )

    with open(MANIFEST, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "bytes", "md5", "modified_utc"]
        )
        writer.writeheader()
        writer.writerows(rows)

    total_mb = sum(r["bytes"] for r in rows) / 1e6
    print(f"Wrote {MANIFEST}")
    print(f"  {len(rows)} files, {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
