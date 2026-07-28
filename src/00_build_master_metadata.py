#!/usr/bin/env python
"""
00_build_master_metadata.py

PURPOSE
    Provide the master metadata table for each species, and check that it is
    valid. The master table is the single source of truth for the bird identity
    and the source recording of every clip. Every later stage reads it.

USAGE
    python src/00_build_master_metadata.py            Validate the table.
    python src/00_build_master_metadata.py --rebuild  Rebuild it from source.

INPUT
    data/<species>/metadata/<species>_master.csv
        The published master table. Validate mode reads this file.
    The original field and colony records.
        Rebuild mode reads these instead. They are not published. See below.

OUTPUT
    data/<species>/metadata/<species>_master.csv

TWO MODES, AND WHY
    Validate (the default)
        The published master tables ship with this repository. This mode checks
        that they are present and internally consistent. This is the mode that
        anyone who downloads the repository will use, and it is the mode that
        run_all.sh calls.

    Rebuild
        This mode regenerates the tables from the original field and colony
        records. Those records are not published. They hold the studbook codes,
        the microchip numbers, and the parent names of individual animals.

        Rebuild therefore works only on the machine that holds those records. It
        exists so that the derivation is documented and repeatable by the
        authors, not so that a reader can run it.

WHY THE TABLE IS THE SINGLE SOURCE OF TRUTH
    An earlier version of this pipeline read the bird and the date from the file
    name. That was the source of a leak. The file name carries the date, but two
    recordings can share a date, and one recording can span two dates. Grouping
    on the date therefore put calls from one recording on both sides of the
    split.

    The master table carries a true recording_id instead. Every later stage
    reads the identity and the recording from this table, never from a file
    name.

THE RECORDING IDENTIFIER
    aa    The recording date and the TASCAM recorder file number.
    ag    The recording date and the recording start time.

    A clip whose source recording could not be resolved has session_known = 0.
    61 of the 1,061 ag clips are in that state. Those clips stay in the dataset,
    because they are valid calls. Every session-aware analysis excludes them.

PRIVACY
    The published table holds only fields that are safe to release: the housing
    aviary, the birth year, the sex, and an anonymised sibling group. The
    studbook codes, the microchip numbers, the parent names, and the rearing
    history stay in the internal records and are never written here.

    The sibling group is derived from the pedigree. Two or more birds in the
    dataset that share both parents receive the same anonymous label, such as
    family_1. The parent identifiers themselves are discarded.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = Path(os.environ.get("PARROT_DATA", ROOT / "data"))

# Every master table must hold these columns. A later stage reads each one.
REQUIRED_COLUMNS = [
    "standardized_filename", "original_stem", "species", "bird", "display_name",
    "call_type", "n_calls", "recording_id", "date", "location",
    "environment_class", "stimulus_only", "sex", "age_years", "social_group",
    "session_known", "kind", "rel_audio_path", "aviary", "birth_year",
    "sibling_group",
]

# The expected shape of each published table. A change here means the dataset
# changed, so the numbers in the manuscript no longer apply.
#
# The published dataset holds single calls only, because the paper reports the
# single-call set. The original aa collection also holds repeated call bouts,
# published as a separate supplementary set.
#
# Two files were removed as segmentation errors and are absent from the dataset:
# acorn_upstaris_240517_0777 (aa, 111 s, a bout) and john_241004_doublese_01
# (ag, 19.1 s). The pipeline filters no clip. Bad data was removed at source.
EXPECTED = {
    "aa": {"clips": 480, "single": 480, "birds": 8, "recordings": 211},
    "ag": {"clips": 1060, "single": 1060, "birds": 16, "recordings": 74},
}


def master_path(species):
    """Return the path of the master table of one species."""
    return DATA / f"{species}/metadata/{species}_master.csv"


def validate(species):
    """Check one master table. Return a list of problems.

    An empty list means the table passed every check.
    """
    problems = []
    path = master_path(species)

    if not path.exists():
        return [f"{path} not found."]

    table = pd.read_csv(path, dtype=str)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing_columns:
        problems.append(f"missing column(s): {', '.join(missing_columns)}")

    # A duplicated stem would make the join to the embeddings ambiguous.
    duplicated = table["original_stem"].duplicated().sum()
    if duplicated:
        problems.append(f"{duplicated} duplicated original_stem value(s)")

    expected = EXPECTED[species]
    if len(table) != expected["clips"]:
        problems.append(f"{len(table)} clips, but {expected['clips']} were expected")

    n_single = int((table["kind"] == "single").sum())
    if n_single != expected["single"]:
        problems.append(f"{n_single} single calls, but {expected['single']} were expected")

    n_birds = table["bird"].nunique()
    if n_birds != expected["birds"]:
        problems.append(f"{n_birds} birds, but {expected['birds']} were expected")

    # Every clip must have an audio file. A missing file means the dataset is
    # incomplete, which would silently reduce the sample size of a later stage.
    absent = 0
    for relative in table["rel_audio_path"]:
        # rel_audio_path starts with "data/", so it resolves from the parent
        # of the data directory. Both roots are tried, so the check works
        # whether PARROT_DATA points at the dataset or at its parent.
        if not (DATA.parent / relative).exists() and not (DATA / relative).exists():
            absent += 1
    if absent:
        problems.append(f"{absent} of {len(table)} audio file(s) not found under {DATA}")

    # A clip marked as resolved must carry a real recording_id.
    resolved = table[table["session_known"] == "1"]
    bad = resolved[resolved["recording_id"].isin(["NA", ""]) | resolved["recording_id"].isna()]
    if len(bad):
        problems.append(f"{len(bad)} clip(s) have session_known = 1 but no recording_id")

    return problems


def describe(species):
    """Print a summary of one master table."""
    table = pd.read_csv(master_path(species), dtype=str)
    resolved = table[table["session_known"] == "1"]
    counts = resolved.groupby("bird")["original_stem"].nunique()

    # Recordings that hold more than one bird. The encounter metric in
    # 03_classify.py depends on this number, so it is reported here.
    per_recording = resolved.groupby("recording_id")["bird"].nunique()
    shared = int((per_recording > 1).sum())

    print(f"  [{species}] {len(table)} clips, {table['bird'].nunique()} birds")
    print(f"  [{species}] {resolved['recording_id'].nunique()} recordings, "
          f"{len(resolved)} of {len(table)} clips resolved")
    print(f"  [{species}] calls for each bird: {counts.min()} to {counts.max()}")
    print(f"  [{species}] {shared} recording(s) hold more than one bird")


def rebuild():
    """Rebuild both master tables from the internal records.

    The function stops with a clear message when the internal records are not
    present. Those records are not published. See the module docstring.
    """
    required = [
        DATA / "colony_reference/birds_table.xlsx",
        DATA / "aa/metadata/aa_metadata.csv",
        DATA / "ag/metadata/ag_pipeline_rename_map.csv",
        DATA / "ag/metadata/ag_session_metadata.csv",
        DATA / "ag/metadata/ag_bird_info.csv",
    ]
    missing = [p for p in required if not p.exists()]

    if missing:
        print("Cannot rebuild. These internal records are not present:")
        for path in missing:
            print(f"  {path}")
        print()
        print("Those records hold identifying information about individual")
        print("animals, so they are not published. The published master tables")
        print("ship with this repository. Run this script without --rebuild to")
        print("validate them.")
        return 1

    # The build logic lives in a separate module, because it depends on the
    # internal record layout and is of no use to a reader without those files.
    from build_master_internal import build_all  # noqa: F401

    build_all(DATA)
    print("Rebuilt both master tables.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Provide and validate the master metadata.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the tables from the internal records. Authors only.",
    )
    args = parser.parse_args()

    if args.rebuild:
        return rebuild()

    print("Validating the master metadata.")
    failed = False

    for species in CONFIG["species"]:
        problems = validate(species)
        if problems:
            failed = True
            print(f"  [{species}] FAILED")
            for problem in problems:
                print(f"  [{species}]   {problem}")
        else:
            describe(species)

    if failed:
        print()
        print("The master metadata is not valid. The later stages will not run")
        print("correctly. Check that the dataset is complete and that")
        print("PARROT_DATA points at it.")
        return 1

    print("The master metadata is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
