#!/usr/bin/env python
"""
common.py

PURPOSE
    Hold the four things every stage needs and no stage should define twice:
    where the code is, where the dataset is, how a missing value is read, and
    which run and which environment produced a result.

USAGE
    This module is imported, not run.

        import common

        master = common.read_master("aa")
        frame = common.stamp(frame)

    Every stage runs as `python src/NN_name.py`, so the src directory is on the
    import path and a plain `import common` finds this file.

INPUT
    config.yaml         Read once, at import.
    results/RUN.json    Read by run_record(). Written by run_all.sh.

OUTPUT
    This module writes nothing on import. write_provenance() writes one
    PROVENANCE.json where it is asked to.

WHY THIS MODULE EXISTS
    Eight scripts held their own copy of the same three lines that find the
    repository root, load config.yaml and resolve the dataset. The provenance
    record in section 4 of the specification needs one place to live, and these
    belong in the same place, so the count of copies goes from eight to one.

THE PROVENANCE RECORD
    On 2026-08-01 one results tree was written by two different environments.
    The classification rows came from one stack and the metric-learning rows,
    the diagnostics and the MFCC features came from another. Nothing in the
    output said so.

    Every table now carries run_id, env_hash and config_hash, and every stage
    checks those values before it reuses work from an earlier run. A mixed run
    then stops instead of writing a table nobody can defend.

    env_hash is computed from the packages installed in the interpreter that is
    running. A stage therefore records the environment it ran in, not the one it
    was meant to run in.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

RUN_JSON = ROOT / "results" / "RUN.json"

# The three columns that record which run and which environment wrote a row.
PROVENANCE_COLUMNS = ["run_id", "env_hash", "config_hash"]

# The audio that only the dataset root can supply. A candidate directory is the
# dataset when it holds both of these. See resolve_data().
AUDIO_MARKERS = ["aa/all_calls/single", "ag/audio/single"]


# =============================================================================
# Missing values
#
# Specification section 10.1. Two copies of ag_master.csv exist. One writes the
# string NA where the other leaves the cell empty, in 441 cells. Read the first
# copy with the wrong rule and 60 undated clips share one date that does not
# exist.
#
# One rule, in one place: a value is missing when it is empty or when it is one
# of the words below. Every stage that reads a master table reads it through
# read_master().
#
# CAUTION
#     This list holds only the tokens that these two tables use for a missing
#     value. Do not add 'none' or 'null' to it. sibling_group carries 'none' in
#     227 ag rows, where it means the bird has no sibling in the colony, and
#     that is a different fact from 'unknown', which the 432 rows for the birds
#     with no colony record carry. To read 'none' as missing would null a real
#     category and change results/diagnostics/kinship.csv.
#
#     Before a token is added here, count it in both master tables.
# =============================================================================

MISSING_TOKENS = ["", "NA", "na", "N/A", "n/a", "NaN", "nan"]


def is_missing(value):
    """Return True when a value stands for 'not recorded'.

    A float NaN is missing, because pandas produces one for an empty cell. A
    string is missing when it is blank or one of MISSING_TOKENS.
    """
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN is not equal to itself.
        return True
    return str(value).strip() in MISSING_TOKENS


def read_master(species):
    """Return the master table of one species, with missing values as NaN.

    The table is read from the repository, never from the dataset root. It is
    curated and versioned with this code, and 00_build_master_metadata.py stops
    the run when a copy under the dataset root differs from this one.

    Every column is read as text, because a bird name, a recording_id and a date
    are all identifiers here and none of them is arithmetic. Every token in
    MISSING_TOKENS becomes NaN, so a caller tests one thing rather than two.
    """
    path = ROOT / f"data/{species}/metadata/{species}_master.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run: python src/00_build_master_metadata.py"
        )
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       na_values=MISSING_TOKENS)


# NOTE
#     There was a has_recording() here, and a session_known column behind it.
#     Both are gone. Every published clip now carries a recording_id, because a
#     clip that cannot enter a split which groups on the recording cannot be
#     scored, and a clip that cannot be scored is not published.
#
#     51 ag clips were removed on 2026-08-02 for that reason, and archived with
#     the reason beside them. A stage that needs the test now writes
#     `if is_missing(record.get("recording_id"))`, which is a check against a
#     fault rather than a routine filter.


# =============================================================================
# The dataset
# =============================================================================

def resolve_data(argument=None):
    """Return the dataset root, or stop and print every path that was tried.

    The order is the --data argument, then PARROT_DATA, then dataset_path in
    config.yaml. There is no default and no search of the filesystem.

    A candidate is accepted only when it holds the audio. The master tables are
    tracked in this repository on purpose, so a test that looks for them accepts
    the repository's own data directory, which holds no audio at all. That is
    what happened at 15:11 on 2026-08-01: the path resolved, and stage 0 then
    reported 1060 of 1060 files missing.

    A candidate may point at the archive root or at the data directory inside
    it. Both are tried, so either works.
    """
    tried = []

    def accept(candidate, source):
        """Return the directory that holds the audio, or None.

        A relative path is resolved against the repository, not against the
        working directory, so that dataset_path in config.yaml means the same
        thing wherever a stage is started from.
        """
        if not candidate:
            return None
        base = Path(candidate).expanduser()
        if not base.is_absolute():
            base = (ROOT / base).resolve()
        for path in (base, base / "data"):
            tried.append(f"{path}   (from {source})")
            if all((path / marker).is_dir() for marker in AUDIO_MARKERS):
                return path
        return None

    for candidate, source in (
        (argument, "--data"),
        (os.environ.get("PARROT_DATA"), "PARROT_DATA"),
        (CONFIG.get("dataset_path"), "dataset_path in config.yaml"),
    ):
        found = accept(candidate, source)
        if found:
            return found

    message = ["The dataset was not found. These paths were tried:"]
    message += [f"  {line}" for line in tried] or ["  (nothing was given)"]
    message += [
        "",
        "A directory is the dataset when it holds both of these:",
    ]
    message += [f"  {marker}/" for marker in AUDIO_MARKERS]
    message += [
        "",
        "Give the path in one of these ways:",
        "  ./run_all.sh --data /path/to/the/dataset",
        "  export PARROT_DATA=/path/to/the/dataset",
        "  dataset_path: /path/to/the/dataset      in config.yaml",
    ]

    # Naming where the data is published turns "your path is wrong" into "here
    # is the data". The entry is empty until the deposit has a DOI.
    doi = CONFIG.get("dataset_doi")
    if doi:
        message += ["", f"The dataset is published at {doi}"]

    raise SystemExit("\n".join(message))


# =============================================================================
# The environment
# =============================================================================

def installed_packages():
    """Return every installed package as one 'name==version' line.

    The list comes from importlib.metadata, which is part of the standard
    library and reads the same installed metadata that pip reads. A virtual
    environment made by uv holds no pip, so `pip freeze` fails there, and uv is
    how this project gets Python 3.11 on a machine that ships 3.10.

    importlib.metadata is also the more useful record. `pip freeze` writes
    'package @ file:///path/to/wheel' for a locally built wheel and '-e git+...'
    for an editable install, so its output carries paths from the machine that
    produced it and can never match on another one.

    The name is normalised the way the packaging standard defines, so two
    machines that spell a name differently still compare equal.
    """
    from importlib import metadata

    seen = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if not name:
            continue
        key = re.sub(r"[-_.]+", "-", name).lower()
        seen[key] = f"{key}=={distribution.version}"
    return sorted(seen.values())


def env_hash():
    """Return the md5 of the packages installed in the running interpreter.

    This is computed from the live interpreter and not from a profile name, so a
    stage records the environment it ran in rather than the one it was meant to
    run in. That is the difference the 2026-08-01 run turned on.
    """
    return hashlib.md5("\n".join(installed_packages()).encode()).hexdigest()


def config_hash():
    """Return the md5 of config.yaml, which holds every analysis choice."""
    return hashlib.md5(CONFIG_PATH.read_bytes()).hexdigest()


def git_commit():
    """Return the current commit, or 'unknown' outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# =============================================================================
# The run record
# =============================================================================

def run_record():
    """Return results/RUN.json, or a record for a stage that is run by hand.

    run_all.sh writes RUN.json before the first stage. A stage run on its own
    has no RUN.json, so it stamps its rows with run_id 'manual'.
    """
    if RUN_JSON.exists():
        return json.loads(RUN_JSON.read_text(encoding="utf-8"))
    return {
        "run_id": "manual",
        "env_hash": {},
        "config_hash": config_hash(),
        "git_commit": git_commit(),
        "allow_env_drift": False,
    }


def results_dir():
    """Return the directory a stage writes its results to.

    A pipeline run writes to results/. A stage run by hand writes to
    results_manual/.

    WHY A MANUAL RUN IS SENT ELSEWHERE
        A manual run stamps a real env_hash, because env_hash is read from the
        interpreter that is running. check_table would therefore accept those
        rows, and the already-scored set in 03_classify.py would then skip that
        model in the next pipeline run. The published table would hold a row no
        pipeline run produced, and nothing in the file would say so.

        That is the shape of the 2026-08-01 fault with a different trigger.
        Debugging one model stays as easy as it was. Its output cannot reach
        the published tree.
    """
    return ROOT / ("results" if RUN_JSON.exists() else "results_manual")


def stamp(frame):
    """Return the frame with run_id, env_hash and config_hash added.

    The frame may be a DataFrame, one row as a dict, or a list of rows. The
    return type matches the argument, so a caller passes its rows through
    without changing how it writes them.
    """
    record = run_record()
    values = {
        "run_id": record["run_id"],
        "env_hash": env_hash(),
        "config_hash": config_hash(),
    }

    if isinstance(frame, dict):
        return {**frame, **values}
    if isinstance(frame, list):
        return [{**row, **values} for row in frame]

    stamped = frame.copy()
    for column, value in values.items():
        stamped[column] = value
    return stamped


def write_provenance(directory):
    """Write PROVENANCE.json beside a set of embeddings or features.

    Stage 1 and stage 2 write this. Stage 3 and later read it, so features made
    in one environment cannot be scored in another without the run stopping.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = run_record()
    (directory / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "run_id": record["run_id"],
                "env_hash": env_hash(),
                "config_hash": config_hash(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _stop(what, cached_env, cached_config, expected_env, expected_config=None):
    """Print the section 4 message and stop the run."""
    raise SystemExit("\n".join([
        "",
        f"{what} was written by a different environment or a different config.",
        f"  written  env_hash:    {cached_env}",
        f"  expected env_hash:    {expected_env}",
        f"  written  config_hash: {cached_config}",
        f"  expected config_hash: {expected_config or config_hash()}",
        "",
        "Delete the file and score it again, or run with --fresh.",
    ]))


def check_table(path, expected_env=None):
    """Stop the run when a table to be extended came from another environment.

    Stage 3 appends one row for each model and skips the models it has already
    scored. If the environment changed since those rows were written, the file
    would hold rows from two stacks, which is what happened on 2026-08-01.

    A file without the provenance columns stops the run as well. It cannot be
    judged, and a file that cannot be judged is the hole 2026-08-01 went
    through. Use --fresh, which moves the old tree to previous_runs/.
    """
    path = Path(path)
    if not path.exists():
        return
    expected_env = expected_env or env_hash()

    try:
        table = pd.read_csv(path, dtype=str)
    except (OSError, pd.errors.EmptyDataError):
        return

    if not set(PROVENANCE_COLUMNS) <= set(table.columns):
        raise SystemExit(
            f"\n{path} carries no provenance record, so the environment that "
            f"wrote it cannot be checked.\nIt was written before this release. "
            f"Run with --fresh, which moves it to previous_runs/."
        )

    for cached_env, cached_config in zip(table["env_hash"], table["config_hash"]):
        if cached_env != expected_env or cached_config != config_hash():
            _stop(f"The rows in {path}", cached_env, cached_config, expected_env)


def check_features(directories):
    """Stop the run when the feature sets to be scored disagree on their source.

    Stage 1 and stage 2 write PROVENANCE.json beside every embedding and feature
    set. This reads all of them and requires that they agree.

    THE RULE
        Every directory must carry a PROVENANCE.json, all of them must record
        the same env_hash and config_hash, and where RUN.json holds an extraction
        env_hash they must match that too.

    WHY AGREEMENT IS THE TEST
        On 2026-08-01 the embeddings came from one environment and the MFCC
        features from another, and the same results tree scored both. Requiring
        the sets to agree catches that on a machine that never built the
        extraction environment, which is the case for anyone who downloads the
        published embeddings.

    Returns the env_hash they agree on, for the caller to record.
    """
    directories = [Path(d) for d in directories]
    records = {}

    for directory in directories:
        path = directory / "PROVENANCE.json"
        if not path.exists():
            raise SystemExit(
                f"\n{directory} carries no provenance record, so the environment "
                f"that produced it cannot be checked.\nRun stage 1 and stage 2 "
                f"again, or run with --fresh."
            )
        records[directory] = json.loads(path.read_text(encoding="utf-8"))

    reference_dir, reference = next(iter(records.items()))
    for directory, record in records.items():
        if (record.get("env_hash") != reference.get("env_hash")
                or record.get("config_hash") != reference.get("config_hash")):
            _stop(f"The features in {directory} and in {reference_dir} disagree.\n"
                  f"  {directory}",
                  record.get("env_hash"), record.get("config_hash"),
                  reference.get("env_hash"), reference.get("config_hash"))

    recorded = (run_record().get("env_hash") or {}).get("extraction")
    if recorded and recorded != reference.get("env_hash"):
        _stop(f"The features in {reference_dir}", reference.get("env_hash"),
              reference.get("config_hash"), recorded)

    return reference.get("env_hash")


def start_run(run_id, data_argument, stages, extraction_env_hash, allow_env_drift):
    """Write results/RUN.json at the start of a run. Return the record.

    run_all.sh calls this before stage 0. Every stage then stamps its rows with
    the run_id recorded here, and compares the environment it is running in
    against these values before it reuses any earlier work.

    WHY THE EXTRACTION HASH IS PASSED IN
        Stage 1 runs in a different environment from this one, so this
        interpreter cannot read the packages of that one. run_all.sh reads it
        from the extraction interpreter and passes it here. Where the extraction
        environment is not built, which is the case for anyone who scores the
        published embeddings, the field is absent and the feature sets are
        checked against each other instead. See check_features().

    WHY THE DATASET IS RESOLVED HERE
        One implementation of the search order, and one error message. run_all.sh
        reads the answer back out of RUN.json and exports it for every stage.
    """
    dataset = resolve_data(data_argument or None)

    environments = {"analysis": env_hash()}
    if extraction_env_hash:
        environments["extraction"] = extraction_env_hash

    record = {
        "run_id": run_id,
        "env_hash": environments,
        "config_hash": config_hash(),
        "git_commit": git_commit(),
        "dataset_path": str(dataset),
        "stages": stages,
        "allow_env_drift": bool(allow_env_drift),
    }

    RUN_JSON.parent.mkdir(parents=True, exist_ok=True)
    RUN_JSON.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _main():
    """Print the environment, or start a run.

    With no argument this prints the environment of the interpreter that runs
    it. run_all.sh reads the extraction env_hash that way, because only that
    interpreter knows its own packages.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Report the environment, or start a run.")
    parser.add_argument("--start-run", action="store_true",
                        help="Write results/RUN.json.")
    parser.add_argument("--run-id", default="", help="The start time, as YYYYMMDD_HHMMSS.")
    parser.add_argument("--data", default="", help="The dataset root. Resolved when not given.")
    parser.add_argument("--stages", default="", help="The stage range this part covers.")
    parser.add_argument("--extraction-env-hash", default="",
                        help="The env_hash of the extraction environment, where it is built.")
    parser.add_argument("--allow-env-drift", action="store_true",
                        help="Record that the run continued past an environment difference.")
    args = parser.parse_args()

    if not args.start_run:
        print(json.dumps({
            "env_hash": env_hash(),
            "config_hash": config_hash(),
            "git_commit": git_commit(),
            "packages": len(installed_packages()),
            "python": sys.version.split()[0],
        }, indent=2))
        return 0

    if not args.run_id:
        raise SystemExit("--start-run needs --run-id.")

    record = start_run(args.run_id, args.data, args.stages,
                       args.extraction_env_hash, args.allow_env_drift)

    print(f"  dataset      {record['dataset_path']}")
    print(f"  run_id       {record['run_id']}")
    print(f"  git_commit   {record['git_commit'][:12]}")
    print(f"  config_hash  {record['config_hash'][:12]}")
    for profile, value in sorted(record["env_hash"].items()):
        print(f"  env_hash     {profile}: {value[:12]}")
    if record["allow_env_drift"]:
        print("  allow_env_drift  true. This run continued past an environment difference.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
