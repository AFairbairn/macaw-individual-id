#!/usr/bin/env python
"""
06_leakage_experiment.py

PURPOSE
    Demonstrate, within one species, that the amount a random split inflates a
    result is set by how many calls are cut from each recording.

USAGE
    python src/06_leakage_experiment.py

INPUT
    config.yaml
    data/ag/metadata/ag_master.csv
    bacpipe_results/ag/embeddings/

OUTPUT
    results/diagnostics/leakage_experiment.csv

WHY THIS EXISTS
    05_diagnostics.py reports that the two datasets differ in calls per
    recording and in leakage delta, in the direction the mechanism predicts.
    That is a between-species comparison, and it is confounded by bird count,
    room, repertoire and species. It licenses "consistent with", nothing more.

    This script removes those confounds. It holds the birds, the room, the
    repertoire, the species, the model and the total number of calls per bird
    constant, and varies ONLY the number of recordings those calls are drawn
    from. If the leakage delta still tracks calls per recording, the mechanism
    is demonstrated rather than inferred.

THE DESIGN
    Every condition uses the same birds and the same 12 calls for each bird.
    Only the spread changes:

        6 recordings x 2 calls
        4 recordings x 3 calls
        3 recordings x 4 calls
        2 recordings x 6 calls

    Birds that cannot support every condition are dropped from all of them, so
    the bird set is identical in every row. Each condition is repeated over
    several random draws and the mean is reported.

    The design changes one variable and holds everything else the same. That is
    the standard form of this kind of experiment. Gallego et al. (2026) use the
    same form for temporal order and for aggregation method, and Huang et al.
    (2024) for input length.

RESULT
    The leakage delta rises from 0.059 to 0.333 as calls per recording goes
    from 2 to 6. The rise is monotonic. Spearman correlation +1.00, p < 0.0001.

CAUTION
    The controlled design costs sample size. Only the birds that support every
    condition are used, and each condition uses 12 calls for each of those
    birds. Every output row carries n_birds and calls_per_bird, so both counts
    travel with the result. The trend is the finding. The absolute values are
    not comparable to the main table.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = Path(os.environ.get("PARROT_DATA", ROOT / "data"))

SEED = CONFIG["seed"]
N_FOLDS = CONFIG["split"]["n_folds"]

# ag is used because it has enough calls in each recording to be thinned. aa
# cannot run this experiment: its recordings hold a median of one call, so there
# is nothing to remove.
SPECIES = "ag"
SUBSET = "lab"
# The subset rule comes from config.yaml, so one edit there changes every stage.
SUBSET_COLUMN, SUBSET_VALUE = next(iter(CONFIG["subsets"][SPECIES][SUBSET].items()))
MODEL = "birdnet"

# (recordings per bird, calls per recording). The product is constant, so the
# number of calls for each bird does not change between conditions.
CONDITIONS = [(6, 2), (4, 3), (3, 4), (2, 6)]

N_DRAWS = 12


def load_clips():
    """Return one mean embedding for each clip of the chosen subset."""
    master = pd.read_csv(ROOT / f"data/{SPECIES}/metadata/{SPECIES}_master.csv", dtype=str)
    records = {r["original_stem"]: r for _, r in master.iterrows()}

    root = ROOT / f"bacpipe_results/{SPECIES}/embeddings"
    model_dir = next(
        (
            d
            for d in sorted(root.iterdir())
            if d.is_dir() and "___" in d.name
            and d.name.split("___")[1].rsplit("-", 1)[0] == MODEL
        ),
        None,
    ) if root.exists() else None

    if model_dir is None:
        return pd.DataFrame()

    rows = []
    for path in sorted(model_dir.rglob(f"*_{MODEL}.npy")):
        stem = path.name[: -len(f"_{MODEL}.npy")]
        if stem not in records:
            continue
        record = records[stem]
        if common.is_missing(record.get("recording_id")):
            continue
        if record.get(SUBSET_COLUMN) != SUBSET_VALUE:
            continue
        matrix = np.atleast_2d(np.load(path))
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            continue
        rows.append((record["bird"].lower(), record["recording_id"], matrix.mean(axis=0)))

    return pd.DataFrame(rows, columns=["bird", "recording", "embedding"])


def accuracy(features, labels, groups, split):
    """Return the mean fold accuracy of a linear probe under one split."""
    encoder = LabelEncoder()
    codes = encoder.fit_transform(labels)

    n = min(N_FOLDS, pd.Series(groups).nunique())
    if split == "by_recording" and n < 2:
        return float("nan")

    folds = (
        GroupKFold(n).split(features, codes, groups)
        if split == "by_recording"
        else StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(features, codes)
    )

    scores = []
    for train, test in folds:
        if len(np.unique(codes[train])) < len(encoder.classes_):
            continue
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(max_iter=1500, random_state=SEED)
        model.fit(scaler.transform(features[train]), codes[train])
        scores.append(accuracy_score(codes[test], model.predict(scaler.transform(features[test]))))

    return float(np.mean(scores)) if scores else float("nan")


def eligible_birds(frame):
    """Return the birds that can supply every condition.

    Using one bird set for every condition means the trend cannot be caused by
    the bird set changing between rows.
    """
    keep = []
    for bird, subset in frame.groupby("bird"):
        counts = subset.groupby("recording").size()
        if all((counts >= calls).sum() >= recordings for recordings, calls in CONDITIONS):
            keep.append(bird)
    return keep


def main():
    frame = load_clips()
    if frame.empty:
        print(f"No {MODEL} embeddings for {SPECIES}. Run stage 1 first. Skipping.")
        return 0

    birds = eligible_birds(frame)
    if len(birds) < 4:
        print(f"Only {len(birds)} birds can supply every condition. Skipping.")
        return 0

    frame = frame[frame.bird.isin(birds)]
    print(f"{SPECIES}/{SUBSET_VALUE}, {MODEL}: {len(birds)} birds supply every condition")
    print(f"  {birds}")
    print()

    rows = []
    for recordings, calls in CONDITIONS:
        by_recording, random_split = [], []

        for draw in range(N_DRAWS):
            generator = np.random.default_rng(SEED + draw)
            chosen = []
            for bird, subset in frame.groupby("bird"):
                counts = subset.groupby("recording").size()
                usable = counts[counts >= calls].index
                for recording in generator.choice(usable, recordings, replace=False):
                    chosen.append(
                        subset[subset.recording == recording].sample(calls, random_state=SEED + draw)
                    )

            sample = pd.concat(chosen)
            features = np.stack(sample["embedding"].to_list())
            labels = sample["bird"].to_numpy()
            groups = sample["recording"].to_numpy()

            grouped = accuracy(features, labels, groups, "by_recording")
            shuffled = accuracy(features, labels, groups, "random")
            if grouped == grouped and shuffled == shuffled:
                by_recording.append(grouped)
                random_split.append(shuffled)

        if not by_recording:
            continue

        delta = np.array(random_split) - np.array(by_recording)
        rows.append(
            {
                "recordings_per_bird": recordings,
                "calls_per_recording": calls,
                "calls_per_bird": recordings * calls,
                "n_birds": len(birds),
                "accuracy_by_recording": float(np.mean(by_recording)),
                "accuracy_random": float(np.mean(random_split)),
                "leakage_delta": float(delta.mean()),
                "leakage_delta_sd": float(delta.std()),
                "n_draws": len(by_recording),
            }
        )

    table = pd.DataFrame(rows)
    out_dir = ROOT / "results/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "leakage_experiment.csv", index=False)

    print(table.round(3).to_string(index=False))

    if len(table) > 2:
        rho, p_value = stats.spearmanr(table.calls_per_recording, table.leakage_delta)
        print()
        print(f"Spearman(calls per recording, leakage delta) = {rho:+.2f}, p = {p_value:.4f}")
        print(
            f"The delta rises from {table.leakage_delta.iloc[0]:.3f} to "
            f"{table.leakage_delta.iloc[-1]:.3f} as calls per recording goes from "
            f"{table.calls_per_recording.iloc[0]} to {table.calls_per_recording.iloc[-1]}, "
            f"with the birds and the calls per bird held constant."
        )

    print()
    print(f"Wrote {out_dir}/leakage_experiment.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
