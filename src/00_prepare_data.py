#!/usr/bin/env python
"""
00_prepare_data.py

STAGE 0 of 5

PURPOSE
    Check the master metadata table of each species, and check that every audio
    file it names is present and holds audio. The master table is the single
    source of truth for the bird identity and the source recording of every
    clip. Every later stage reads it.

USAGE
    python src/00_prepare_data.py            Check the tables and the audio.
    python src/00_prepare_data.py --rebuild  Rebuild the tables. Authors only.

INPUT
    data/<species>/metadata/<species>_master.csv    The clip list.
    The audio named in rel_audio_path, under the dataset root.

OUTPUT
    None. The stage stops the run when a table or an audio file fails a check.

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
    The file name carries the date, and the date is not the recording. Two
    recordings can share a date, and one recording can span two dates. Grouping
    on the date therefore puts calls from one recording on both sides of the
    split, which is a leak.

    The master table carries a true recording_id instead. Every later stage
    reads the identity and the recording from this table, never from a file
    name.

THE RECORDING IDENTIFIER
    Both species use the recording date and the TASCAM recorder file number, as
    YYMMDD-NNNN. Confirmed on 2026-08-02 against the Audacity projects of six ag
    birds: every recording_id already in the table is reproduced exactly by the
    source name the project records, 30 of 30.

    Every published clip carries a recording_id. A clip without one cannot enter
    a split that groups on the recording, so it cannot be scored, and a clip
    that cannot be scored is not published.

    60 ag clips were in that state. Nine were recovered on 2026-08-02 from the
    AV1 Audacity projects, by a duration match that is unique over every valid
    pairing of its call-type group. The other 51 match only project clips whose
    source name Audacity discarded during editing, so no method recovers them.
    Those 51 are archived, not deleted, with the reason beside them.

    ag therefore holds 1,009 clips, not the 1,060 that were published before.

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
import hashlib
import sys
from pathlib import Path

import pandas as pd
import soundfile

import common

ROOT = common.ROOT
CONFIG = common.CONFIG
DATA = common.resolve_data()

# Every master table must hold these columns. A later stage reads each one.
REQUIRED_COLUMNS = [
    "standardized_filename", "original_stem", "species", "bird", "display_name",
    "call_type", "n_calls", "recording_id", "date", "location",
    "environment_class", "stimulus_only", "sex", "age_years", "social_group",
    "kind", "rel_audio_path", "aviary", "birth_year",
    "sibling_group",
]

# The expected shape of each published table. A change here means the dataset
# changed, so the numbers in the manuscript no longer apply.
#
# The published dataset holds single calls only, because the paper reports the
# single-call set. The original aa collection also holds repeated call bouts,
# published as a separate supplementary set.
#
# Two segmentation errors were handled at source: acorn_upstaris_240517_0777
# (aa, 111 s, a bout) and john_241004_doublese_01 (ag, 19.1 s). The pipeline
# filters no clip, so the counts here are the counts that were analysed.
EXPECTED = {
    "aa": {"clips": 480, "single": 480, "birds": 8, "recordings": 211},
    "ag": {"clips": 1009, "single": 1009, "birds": 16, "recordings": 76},
}


def load_checksums():
    """Return the recorded md5 of every file, from the data package.

    The data package ships CHECKSUMS.csv beside its data directory. Where that
    file is absent, the function returns an empty mapping and the checksum
    comparison is skipped.
    """
    for candidate in (DATA.parent / "CHECKSUMS.csv", DATA / "CHECKSUMS.csv"):
        if candidate.exists():
            table = pd.read_csv(candidate, dtype=str, header=None,
                                names=["path", "bytes", "md5"])
            return dict(zip(table["path"], table["md5"]))
    return {}


def md5_of(path):
    """Return the md5 checksum of one file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_of_text(path):
    """Return the md5 of one text file, with every line ending read as LF.

    Two copies of the same table written on different platforms differ in every
    line and say the same thing. This reports a difference in what a table holds
    and never a difference in how its lines end.

    On 2026-08-03 that distinction stopped a run. The master tables in this
    repository and the copies in the published archive were identical cell for
    cell, and their checksums differed by the 481 carriage returns that Windows
    had written into one of them.

    The whole file is read at once. This reads master tables, the largest of
    which is 216 kB, and reading it in one piece is what makes the line ending
    substitution correct without holding a block boundary open.
    """
    return hashlib.md5(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def master_path(species):
    """Return the path of the master table of one species.

    The table is read from the repository, not from PARROT_DATA. It is curated
    metadata and it is versioned with this code. Only the audio comes from
    PARROT_DATA. One source for the table keeps the copy in the data package
    from drifting away from the repository copy. Stage 2 and the tests read
    the repository copy.
    """
    return ROOT / f"data/{species}/metadata/{species}_master.csv"


def check_no_rival_copy(species):
    """Return a problem when the dataset root carries a different master table.

    README section 1.2 states that the master tables live in this repository and
    that only the audio is read from the dataset root. This enforces it.

    Two copies of ag_master.csv exist today and they disagree. One holds 1060
    rows and the other 1061, and the second writes the string NA in 441 cells
    where the first leaves the cell empty. A user who points --data at a dataset
    root carrying its own tables would otherwise read a different table from the
    one this code ships, and every count would change without a message.

    A dataset root with no master table of its own is the normal case and passes.

    The comparison ignores line endings. A copy that Windows wrote and a copy
    that Linux wrote hold the same table, and the difference this check exists to
    report is a difference in the rows.
    """
    repository = master_path(species)
    if not repository.exists():
        return None

    recorded = md5_of_text(repository)
    for candidate in (DATA / f"{species}/metadata/{species}_master.csv",
                      DATA.parent / f"data/{species}/metadata/{species}_master.csv"):
        if not candidate.exists() or candidate.resolve() == repository.resolve():
            continue
        if md5_of_text(candidate) != recorded:
            return (
                "the dataset root carries a different master table.\n"
                f"    repository: {repository}\n"
                f"    dataset:    {candidate}\n"
                "    The repository copy is the one that counts. Delete or "
                "correct the dataset copy."
            )
    return None


def validate(species):
    """Check one master table. Return a list of problems.

    An empty list means the table passed every check.
    """
    problems = []
    path = master_path(species)

    if not path.exists():
        return [f"{path} not found."]

    rival = check_no_rival_copy(species)
    if rival:
        problems.append(rival)

    table = common.read_master(species)

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

    # Every clip must have an audio file, and that file must hold audio. A
    # missing file reduces the sample size of a later stage. A file that is
    # present but empty is worse, because it survives an existence check and
    # then stops the run in the middle of stage 0 or stage 1.
    #
    # A copy of the data package can truncate one of the 1489 files to zero
    # bytes. This check reports that file by name here, at the first stage,
    # instead of leaving it to a crash three stages later.
    absent, empty, corrupt, wrong_checksum = [], [], [], []
    checksums = load_checksums()

    for relative in table["rel_audio_path"]:
        # rel_audio_path starts with "data/", so it resolves from the parent
        # of the data directory. Both roots are tried, so the check works
        # whether PARROT_DATA points at the dataset or at its parent.
        path = None
        for candidate in (DATA.parent / relative, DATA / relative):
            if candidate.exists():
                path = candidate
                break

        if path is None:
            absent.append(relative)
            continue

        if path.stat().st_size == 0:
            empty.append(relative)
            continue

        try:
            frames = soundfile.info(path).frames
        except Exception as error:
            # Carry the message. A permission error, an unsupported codec
            # and a truncated file need different fixes.
            corrupt.append(f"{relative}  ({type(error).__name__}: {error})")
            continue
        if frames == 0:
            empty.append(relative)
            continue

        # The data package ships CHECKSUMS.csv. Where it is available, compare
        # against it. A checksum catches a file that was changed after the
        # package was built, which no other check here can see.
        recorded = checksums.get(relative)
        if recorded and md5_of(path) != recorded:
            wrong_checksum.append(relative)

    for label, found in (("not found", absent), ("empty", empty),
                         ("unreadable", corrupt), ("changed since CHECKSUMS.csv",
                                                   wrong_checksum)):
        if found:
            problems.append(
                f"{len(found)} of {len(table)} audio file(s) {label}, "
                f"first: {found[0]}"
            )

    # Every published clip must carry a recording_id. The split groups on the
    # recording, so a clip without one cannot be scored, and a clip that cannot
    # be scored is not published. common reads every missing token as NaN, so
    # one test covers the empty cell and the string NA together.
    bad = table[table["recording_id"].isna()]
    if len(bad):
        problems.append(
            f"{len(bad)} clip(s) have no recording_id. The published dataset "
            f"holds only clips the analysis can use. First: "
            f"{bad['original_stem'].iloc[0]}"
        )

    return problems


def describe(species):
    """Print a summary of one master table."""
    table = common.read_master(species)
    counts = table.groupby("bird")["original_stem"].nunique()

    # Recordings that hold more than one bird. The encounter metric in
    # 03_score_frozen.py depends on this number, so it is reported here.
    per_recording = table.groupby("recording_id")["bird"].nunique()
    shared = int((per_recording > 1).sum())

    print(f"  [{species}] {len(table)} clips, {table['bird'].nunique()} birds")
    print(f"  [{species}] {table['recording_id'].nunique()} recordings")
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


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Check the metadata and the audio.")
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
        print(f"correctly. Check that the dataset at {DATA} is complete.")
        return 1

    print("The master metadata is valid, and every audio file is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
