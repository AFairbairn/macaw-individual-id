#!/usr/bin/env python
"""
05_finish_run.py

STAGE 5 of 5

PURPOSE
    Close a run. Three jobs, in this order.

    1. Score one model twice and compare, so the run proves its own numbers are
       stable.
    2. Write an md5 for every output, so a later run can be compared to this one.
    3. Write the two archives a user downloads.

USAGE
    python src/05_finish_run.py --log logs/run_20260802_143000.log

INPUT
    results/            Every table the earlier stages wrote.
    environment.lock/   The record of the environment, where it is present.
    The log named by --log.

OUTPUT
    results/determinism.json           Whether the two scorings matched.
    results/MANIFEST.csv               path, bytes, md5, md5_content.
    run_outputs/<run_id>_summary.zip   Every number that goes into the paper.
    run_outputs/<run_id>_full.zip      The whole results tree.

WHY THE MANIFEST HOLDS TWO CHECKSUMS
    Every result row carries run_id, env_hash and config_hash, so that a number
    can be traced to the run that produced it. run_id is the start time, so
    those three columns differ between two runs by design, and md5 of the file
    therefore differs as well.

    md5_content is the checksum of the same file with those three columns
    removed. Two runs of this code on this data give the same md5_content and a
    different md5. Compare md5_content to answer "are the numbers the same".

WHY DETERMINISM IS CHECKED HERE AND NOT ONLY IN A TEST
    A test that skips when the features are absent passes in silence on a fresh
    copy of the repository, which is the state a reader is in. This runs on the
    features the run just produced, so it cannot pass by skipping.

    Between 2026-07-31 and 2026-08-01 the frozen centroid accuracy of
    aves_especies moved from 0.331 to 0.308 and of birdmae from 0.406 to 0.387,
    for a head that trains nothing and seeds that are fixed. The cause was two
    environments, not chance, but nothing in the pipeline could tell those apart
    at the time. This check separates them: it runs twice inside one
    environment, so a difference here is a fault in the code.

WHY THERE ARE TWO ARCHIVES
    The summary archive is small enough to download over a slow connection and
    holds every number the manuscript reports, the record of the run, and the
    log. The full archive adds the per-fold prediction dumps and the head
    vectors, which are needed to draw a figure and are large.

    The log is included because the 2026-08-01 run recorded 154 package
    differences and continued. That fact was in the log and nowhere else, and
    the log stayed on the cluster.
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

import common

ROOT = common.ROOT
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "MANIFEST.csv"
OUT_DIR = ROOT / "run_outputs"

CHUNK_BYTES = 1 << 20  # Read 1 MiB at a time, so a large file does not fill memory.

# Files that change on every run, or that are not results.
EXCLUDE_NAMES = {"MANIFEST.csv"}
EXCLUDE_SUFFIXES = {".log", ".tmp", ".pyc"}

# The smallest representation on the smallest subset, so the check costs little.
# mfcc_lakdari holds 24 values for each clip, and aa/all holds 480 clips.
CHECK_MODEL = "mfcc_lakdari"
CHECK_SPECIES = "aa"
CHECK_SUBSET = "all"

# The files the summary archive holds. Each is a glob read from the repository
# root. A pattern that matches nothing is skipped without a message, because a
# run that skipped a stage has no output from it.
SUMMARY_PATTERNS = [
    "results/RUN.json",
    "results/determinism.json",
    "results/MANIFEST.csv",
    "results/*/rows.csv",
    "results/*/*/metric_learning/summary.csv",
    "results/*/*/metric_learning/eval_B_folds.csv",
    "results/diagnostics/*",
    "results/supplementary/*",
    "environment.lock/*/*",
]


# =============================================================================
# 1. Determinism
# =============================================================================

def score_once(destination):
    """Run stage 3 once, writing below destination. Return the summary table."""
    command = [
        sys.executable, str(ROOT / "src" / "03_metric_learning.py"),
        "--species", CHECK_SPECIES,
        "--subset", CHECK_SUBSET,
        "--models", CHECK_MODEL,
        "--device", "cpu",
        "--results-dir", str(destination),
    ]
    finished = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if finished.returncode != 0:
        raise RuntimeError(finished.stderr.strip().splitlines()[-1]
                           if finished.stderr.strip() else "stage 3 failed")

    summary = destination / CHECK_SPECIES / CHECK_SUBSET / "metric_learning" / "summary.csv"
    if not summary.exists():
        raise RuntimeError(f"no summary was written at {summary}")
    return pd.read_csv(summary)


def compare_scorings(first, second):
    """Return the differences between two scorings, ignoring the stamp columns."""
    if list(first.columns) != list(second.columns):
        return ["the two runs wrote different columns"]
    if len(first) != len(second):
        return [f"the two runs wrote {len(first)} and {len(second)} rows"]

    differences = []
    for column in first.columns:
        if column in common.PROVENANCE_COLUMNS:
            continue
        unequal = first[column].astype(str) != second[column].astype(str)
        if unequal.any():
            index = unequal.idxmax()
            differences.append(f"{column}: {first[column][index]!r} against "
                               f"{second[column][index]!r}")
    return differences


def check_determinism():
    """Score one model twice and compare. Return the record that is written."""
    print(f"  scoring {CHECK_MODEL} on {CHECK_SPECIES}/{CHECK_SUBSET} twice")
    workspace = Path(tempfile.mkdtemp(prefix="determinism_"))
    try:
        first = score_once(workspace / "first")
        second = score_once(workspace / "second")
    except RuntimeError as error:
        print(f"  could not run: {error}")
        return {"outcome": "not run", "reason": str(error),
                "model": CHECK_MODEL, "subset": f"{CHECK_SPECIES}/{CHECK_SUBSET}"}
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    differences = compare_scorings(first, second)
    record = {
        "outcome": "pass" if not differences else "fail",
        "model": CHECK_MODEL,
        "subset": f"{CHECK_SPECIES}/{CHECK_SUBSET}",
        "columns_compared": len(first.columns) - len(common.PROVENANCE_COLUMNS),
        "differences": differences,
    }
    if differences:
        print(f"  the two scorings differ in {len(differences)} column(s):")
        for line in differences[:10]:
            print(f"    {line}")
    else:
        print(f"  the two scorings match, over {record['columns_compared']} columns")
    return record


# =============================================================================
# 2. The manifest
# =============================================================================

def md5_of(path):
    """Return the md5 checksum of one file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_of_content(path):
    """Return the md5 of a table with the three stamp columns removed.

    A file that is not a table, or a table that carries no stamp, returns its
    own md5, so every row of the manifest holds a value in both columns.
    """
    if path.suffix.lower() != ".csv":
        return md5_of(path)
    try:
        table = pd.read_csv(path, dtype=str)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return md5_of(path)

    present = [c for c in common.PROVENANCE_COLUMNS if c in table.columns]
    if not present:
        return md5_of(path)

    buffer = io.StringIO()
    table.drop(columns=present).to_csv(buffer, index=False, lineterminator="\n")
    return hashlib.md5(buffer.getvalue().encode()).hexdigest()


def is_result(path):
    """Return True when a file belongs in the manifest."""
    return (path.is_file()
            and path.name not in EXCLUDE_NAMES
            and path.suffix not in EXCLUDE_SUFFIXES)


def write_manifest():
    """Write results/MANIFEST.csv. Return the number of files it covers."""
    files = sorted(p for p in RESULTS.rglob("*") if is_result(p))
    if not files:
        raise SystemExit(f"{RESULTS} holds no result file.")

    # The manifest holds the path, the size and the two checksums, and nothing
    # that changes by itself. A modification time changes on every run, so a
    # manifest that carried it would report that every file had changed.
    rows = [
        {
            "path": path.relative_to(RESULTS).as_posix(),
            "bytes": path.stat().st_size,
            "md5": md5_of(path),
            "md5_content": md5_of_content(path),
        }
        for path in files
    ]

    with open(MANIFEST, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=["path", "bytes", "md5", "md5_content"])
        writer.writeheader()
        writer.writerows(rows)

    total_mb = sum(r["bytes"] for r in rows) / 1e6
    print(f"  {MANIFEST.relative_to(ROOT)}: {len(rows)} files, {total_mb:.1f} MB")
    return len(rows)


# =============================================================================
# 3. The archives
# =============================================================================

def human(size):
    """Return a byte count as a string a reader can compare at a glance."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def write_archive(target, paths):
    """Write one zip holding the given paths, named relative to the repository."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    return target


def write_archives(log_path):
    """Write the summary and the full archive. Return both paths."""
    run_id = common.run_record()["run_id"]

    summary_files = []
    for pattern in SUMMARY_PATTERNS:
        summary_files += [p for p in sorted(ROOT.glob(pattern)) if p.is_file()]
    if log_path and log_path.is_file():
        summary_files.append(log_path)
    # A file can match two patterns. Keep the first occurrence of each.
    summary_files = list(dict.fromkeys(summary_files))

    full_files = [p for p in sorted(RESULTS.rglob("*")) if p.is_file()]
    if log_path and log_path.is_file():
        full_files.append(log_path)

    summary = write_archive(OUT_DIR / f"{run_id}_summary.zip", summary_files)
    full = write_archive(OUT_DIR / f"{run_id}_full.zip", full_files)

    print(f"  {summary.relative_to(ROOT)}   "
          f"{len(summary_files)} file(s), {human(summary.stat().st_size)}")
    print(f"  {full.relative_to(ROOT)}      "
          f"{len(full_files)} file(s), {human(full.stat().st_size)}")
    return summary, full


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Check, checksum and archive a run.")
    parser.add_argument("--log", default="", help="The log of this run, to include.")
    parser.add_argument("--skip-determinism", action="store_true",
                        help="Do not score a model twice. The manifest and the archives still run.")
    args = parser.parse_args()

    if not RESULTS.is_dir():
        raise SystemExit("results/ not found. Run the earlier stages first.")

    print("Determinism")
    if args.skip_determinism:
        record = {"outcome": "not run", "reason": "--skip-determinism was given"}
        print("  skipped, because --skip-determinism was given")
    else:
        record = check_determinism()
    (RESULTS / "determinism.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print("Manifest")
    write_manifest()

    print("Archives")
    write_archives((ROOT / args.log) if args.log else None)

    print()
    print("  Download the summary archive. It holds every number in the paper,")
    print("  the record of this run, and the log.")

    # A failed determinism check does not stop the archives, so the evidence
    # travels with the failure.
    return 1 if record["outcome"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
