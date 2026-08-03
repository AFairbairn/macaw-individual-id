#!/usr/bin/env python
"""
09_supplementary_bouts.py

PURPOSE
    Measure whether more acoustic material for each detection improves
    individual identification. This is a supplementary analysis. It does not
    enter any main result.

USAGE
    python src/09_supplementary_bouts.py

INPUT
    config.yaml
    supplementary/aa_repeated_bouts/metadata/aa_bouts_master.csv
    bacpipe_results/aa/embeddings/
    mfcc_results/aa/embeddings/

OUTPUT
    results/supplementary/bout_comparison.csv

THE QUESTION
    A single call is one vocalisation. A bout is a run of several calls. A bout
    therefore gives a model more acoustic material.

    This script scores the same models twice for Ara ambiguus: on single calls
    alone, and on single calls together with 255 repeated bouts. The difference
    estimates what extra material buys.

    The finding is useful when you design a field deployment. If a bout is much
    easier than a call, a recorder should capture whole bouts.

WHY THIS IS NOT IN THE MAIN ANALYSIS
    Ara glaucogularis has no bouts. If Ara ambiguus used bouts and Ara
    glaucogularis did not, part of the reported difference between the two
    species would come from the acoustic unit rather than from the biology.

    The main result of the paper is that Ara ambiguus is easier than Ara
    glaucogularis. That result must not depend on the call set. The main
    analysis therefore uses single calls for both species, and this comparison
    stays in the supplement.

NOTE
    The bouts are published as a separate supplementary set, so a machine can
    hold the main dataset alone. Where the supplementary audio is absent, the
    script stops with a clear message.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = Path(os.environ.get("PARROT_DATA", ROOT / "data"))

SPECIES = "aa"
BOUTS_MASTER = ROOT / "supplementary/aa_repeated_bouts/metadata/aa_bouts_master.csv"

# The comparison uses the leading models only. The purpose is to size one
# effect, not to rebuild the whole leaderboard.
MODELS = ["birdnet", "birdmae", "perch_bird", "perch_v2", "avesecho_passt", "mfcc_full"]


def load_combined_master():
    """Return the single calls and the bouts in one table.

    The function returns None when the supplementary metadata is absent.
    """
    single_path = ROOT / f"data/{SPECIES}/metadata/{SPECIES}_master.csv"
    if not single_path.exists():
        raise SystemExit(f"{single_path} not found. Run stage 0 first.")

    if not BOUTS_MASTER.exists():
        return None

    single = pd.read_csv(single_path, dtype=str)
    bouts = pd.read_csv(BOUTS_MASTER, dtype=str)
    return pd.concat([single, bouts], ignore_index=True)


def main():
    # Import the shared helpers from the main scoring script. The module name
    # starts with a digit, so it needs importlib rather than a plain import.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "classify", Path(__file__).resolve().parent / "03_classify.py"
    )
    classify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classify)

    combined = load_combined_master()
    if combined is None:
        print("The supplementary bout set is not present.")
        print(f"Expected: {BOUTS_MASTER}")
        print()
        print("The bouts are published as a separate supplementary set, because")
        print("the main analysis uses single calls only. Download that set to")
        print("run this comparison. Skipping.")
        return 0

    master = {row["original_stem"]: row for _, row in combined.iterrows()}

    model_dirs = classify.find_model_dirs(
        ROOT / f"bacpipe_results/{SPECIES}/embeddings",
        ROOT / f"mfcc_results/{SPECIES}/embeddings",
    )
    if not model_dirs:
        raise SystemExit(f"No embeddings found for {SPECIES}. Run stage 1 and stage 2 first.")

    single_stems = classify.collect_single_stems(model_dirs)

    rows = []
    for model in MODELS:
        if model not in model_dirs:
            print(f"  {model}: no embeddings, skipped")
            continue

        frame = classify.load_embeddings(model_dirs[model], model, master, single_stems)
        if frame.empty:
            continue

        singles_only = frame[frame["kind"] == "single"]

        result = {"species": SPECIES, "model": model}

        for label, subset in (("single", singles_only), ("with_bouts", frame)):
            if len(subset) < 20 or subset["bird"].nunique() < 2:
                continue
            features = np.stack(subset["embedding"].to_list())
            labels = subset["bird"].to_numpy()
            groups = subset["session"].to_numpy()
            probe = classify.run_probe(features, labels, groups, "by_recording")
            result[f"{label}_n_clips"] = int(subset["clip_id"].nunique())
            result[f"{label}_probe"] = probe["accuracy"]
            result[f"{label}_sd"] = probe["sd"]

        if "single_probe" in result and "with_bouts_probe" in result:
            result["gain_from_bouts"] = result["with_bouts_probe"] - result["single_probe"]
            rows.append(result)
            print(
                f"  {model}: single {result['single_probe']:.3f} "
                f"-> with bouts {result['with_bouts_probe']:.3f} "
                f"(gain {result['gain_from_bouts']:+.3f})",
                flush=True,
            )

    if not rows:
        # Return 0. This comparison never enters a main result, and the
        # supplementary audio is published separately from the main dataset, so
        # a machine without it must still finish the run. The message names the
        # file that is needed.
        print("No model produced a usable comparison.")
        print(f"The bout metadata is {BOUTS_MASTER}.")
        print("Download the supplementary set to run this comparison.")
        return 0

    out_dir = ROOT / "results/supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bout_comparison.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    mean_gain = float(np.mean([r["gain_from_bouts"] for r in rows]))
    print()
    print(f"Wrote {out_path}")
    print(f"  Mean gain from bouts: {mean_gain:+.3f} accuracy across {len(rows)} models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
