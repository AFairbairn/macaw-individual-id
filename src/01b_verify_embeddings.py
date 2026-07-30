#!/usr/bin/env python
"""
01b_verify_embeddings.py

PURPOSE
    Confirm that stage 1 wrote one embedding for every call, for every model.

USAGE
    python src/01b_verify_embeddings.py

INPUT
    data/<species>/metadata/<species>_master.csv    The expected clip list.
    bacpipe_results/<species>/embeddings/           The extracted embeddings.

OUTPUT
    A report on the terminal. The exit code is 0 when every model is complete,
    and 1 when a model is incomplete or missing.

WHY THIS SCRIPT EXISTS
    bacpipe prints many tracebacks after it writes the embeddings. Those
    messages come from a dashboard step that this pipeline does not use. The
    embeddings are already on disk when those messages appear.

    A traceback therefore does not mean the extraction failed, and a clean log
    does not mean it succeeded. One model can write zero files while the log
    looks normal. avesecho_passt does exactly that when it runs on CUDA.

    Count the files. Do not read the log.

WHAT COUNTS AS COMPLETE
    A model is complete when it wrote one .npy file for every clip in the master
    table. A model that wrote fewer files is reported as incomplete, with the
    number of missing clips.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = Path(os.environ.get("PARROT_DATA", ROOT / "data"))

EXPECTED_MODEL_COUNT = 15


def expected_stems(species):
    """Return the clip stems that the master table lists for one species."""
    path = ROOT / f"data/{species}/metadata/{species}_master.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run src/00_build_master_metadata.py first.")
    table = pd.read_csv(path, dtype=str)
    return set(table["original_stem"])


def found_stems(directory, model):
    """Return the clip stems that one model wrote."""
    suffix = f"_{model}.npy"
    return {path.name[: -len(suffix)] for path in directory.rglob(f"*{suffix}")}


def check_species(species):
    """Report every model of one species. Return True when all are complete."""
    root = ROOT / f"bacpipe_results/{species}/embeddings"
    print(f"\n=== {species} ===")

    if not root.exists():
        print(f"  {root} not found. Run stage 1 first.")
        return False

    expected = expected_stems(species)
    print(f"  The master table lists {len(expected)} clips.")

    directories = sorted(d for d in root.iterdir() if d.is_dir() and "___" in d.name)
    if not directories:
        print("  No model directories found.")
        return False

    all_complete = True

    for directory in directories:
        model = directory.name.split("___")[1].rsplit("-", 1)[0]
        found = found_stems(directory, model)
        missing = expected - found

        if not found:
            # This is the avesecho_passt on CUDA case.
            print(f"  {model:22} FAILED. The model wrote no files.")
            all_complete = False
        elif missing:
            print(f"  {model:22} INCOMPLETE. {len(found)}/{len(expected)} clips.")
            for stem in sorted(missing)[:3]:
                print(f"  {'':22}   missing: {stem}")
            if len(missing) > 3:
                print(f"  {'':22}   and {len(missing) - 3} more")
            all_complete = False
        else:
            print(f"  {model:22} complete. {len(found)} clips.")

    if len(directories) < EXPECTED_MODEL_COUNT:
        print(f"  Only {len(directories)} of {EXPECTED_MODEL_COUNT} models are present.")
        all_complete = False

    return all_complete


def main():
    print("Verifying the extracted embeddings.")
    print("A traceback in the stage 1 log does not mean the extraction failed.")
    print("This script counts the output files instead.")

    results = [check_species(species) for species in CONFIG["species"]]

    print()
    if all(results):
        print("Every model is complete.")
        return 0

    print("One or more models are incomplete. See the report above.")
    print("To recompute one model, delete its directory and run stage 1 again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
