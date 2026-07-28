#!/usr/bin/env python
"""
05_diagnostics.py

PURPOSE
    Run three diagnostics that test what the embeddings actually encode. Each
    one answers a question that a referee will ask about the main results.

USAGE
    python src/05_diagnostics.py

INPUT
    config.yaml
    data/<species>/metadata/<species>_master.csv
    bacpipe_results/<species>/embeddings/
    mfcc_results/<species>/embeddings/

OUTPUT
    results/diagnostics/domain_shift.csv        One row for each bird and model.
    results/diagnostics/within_call_type.csv    One row for each call type.
    results/diagnostics/kinship.csv             One row for each model.

RUNTIME
    About 15 minutes on 8 CPU cores. No GPU is needed.

NOTE
    This script writes CSV files only. It draws no figures. Figures are made
    separately from these files.

THE THREE DIAGNOSTICS

    1. Domain shift
       Question: do recordings carry an acoustic signature that a model could
       use instead of the voice?

       Method: within ONE bird, predict which recording each call came from.
       Running the test one bird at a time removes identity as an explanation,
       so any result above chance is a per-recording signature. That signature
       can come from the microphone, the position of the bird, the state of the
       room, or the time of day.

       This is the measurable form of the question that Jakob Abesser raised:
       classify the background and see whether it beats chance. This version
       needs no source separation and no new feature extraction.

       Result: recordings are 2.7 to 3.7 times more identifiable than chance.
       The signature is real. It is why the by-recording split is the only
       honest one, and it explains the size of the leakage delta.

    2. Within call type
       Question: which call type separates individuals best?

       Method: score individual identification separately for each ag call
       type, using the session-aware split.

       Caution: the number of birds differs between call types, so the chance
       level differs too. Compare each call type against its own chance level,
       never against another call type.

    3. Kinship
       Question: are related birds harder to tell apart? If kin sound alike,
       part of what looks like weak individual identification would really be
       family resemblance.

       Method: build one mean template for each bird, then compare the cosine
       similarity of same-family pairs against different-family pairs. Test the
       difference with a permutation test on the family labels.

       Result: no effect in any model. This is a clean negative that closes the
       question and rules out kin-aware negative sampling as a necessary step.

       Caution: there are only 2 or 3 families in each species. The test rules
       out a large kinship effect, not a small one. State that limit.

    4. Pretraining exposure
       Question: is Ara ambiguus easier than Ara glaucogularis only because the
       models saw more of it during pretraining?

       This matters. Xeno-canto holds 48 recordings of A. ambiguus and 17 of
       A. glaucogularis (checked 2026-07-28), and Xeno-canto is the training
       corpus for the bird-supervised models. So exposure differs in the same
       direction as the result. Huang et al. (2024) raise the same concern for
       their own species.

       Method: split the model panel by whether its training corpus includes
       Xeno-canto, then compare the aa-to-ag performance ratio between the two
       groups. Accuracy is divided by chance first, so the different bird counts
       (8 against 12) do not confound the comparison.

       Result: no difference. The ratio is 1.31 for models that saw Xeno-canto
       and 1.36 for models that did not (Mann-Whitney p = 0.59). The clearest
       single case is wavlm_base_plus_sv, trained only on human speech with zero
       exposure to either bird, which shows the LARGEST gap of any model (1.68).
       Pretraining exposure does not explain the species difference.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = Path(os.environ.get("PARROT_DATA", ROOT / "data"))

SEED = CONFIG["seed"]
N_FOLDS = CONFIG["split"]["n_folds"]

# The diagnostics use the leading models plus the MFCC baseline. The purpose is
# to size an effect, not to rebuild the whole leaderboard.
MODELS = ["birdnet", "birdmae", "perch_bird", "mfcc_lakdari", "mfcc_full"]

# A recording needs this many calls before it can be a class in diagnostic 1.
MIN_CALLS_FOR_RECORDING = 3

# The number of label shuffles in the kinship permutation test.
N_PERMUTATIONS = 2000


def load_clips(species, model):
    """Return one mean embedding for each clip of one model.

    The function averages the windows of a clip, so every model contributes one
    vector for each clip. Without that step, a model that returns several
    windows would carry more weight than a model that returns one.
    """
    master_path = DATA / f"{species}/metadata/{species}_master.csv"
    master = pd.read_csv(master_path, dtype=str)
    records = {row["original_stem"]: row for _, row in master.iterrows()}

    roots = [
        ROOT / f"bacpipe_results/{species}/embeddings",
        ROOT / f"mfcc_results/{species}/embeddings",
    ]

    model_dir = None
    for root in roots:
        if not root.exists():
            continue
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or "___" not in directory.name:
                continue
            if directory.name.split("___")[1].rsplit("-", 1)[0] == model:
                model_dir = directory
                break

    if model_dir is None:
        return pd.DataFrame()

    rows = []
    for path in sorted(model_dir.rglob(f"*_{model}.npy")):
        stem = path.name[: -len(f"_{model}.npy")]

        if stem not in records:
            continue

        record = records[stem]
        if str(record.get("session_known")) != "1" or record.get("recording_id") in (None, "NA"):
            continue

        matrix = np.atleast_2d(np.load(path))
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            continue

        rows.append(
            (
                stem,
                record["bird"].lower(),
                record["recording_id"],
                record.get("call_type", "NA"),
                record.get("sibling_group", "unknown"),
                record.get("environment_class", "NA"),
                matrix.mean(axis=0).astype(np.float32),
            )
        )

    return pd.DataFrame(
        rows,
        columns=["clip", "bird", "recording", "call_type", "family", "environment_class", "embedding"],
    )


# =============================================================================
# Diagnostic 1: domain shift
# =============================================================================

def domain_shift(frame, species, model):
    """Test whether recordings are identifiable within one bird.

    The test runs separately for each bird, so the bird identity cannot explain
    the result. A score above chance means the recording carries its own
    acoustic signature.

    The split here is random on purpose. The question is whether the signature
    exists at all, so the calls of one recording must appear on both sides.
    """
    results = []

    for bird, subset in frame.groupby("bird"):
        counts = subset["recording"].value_counts()
        usable = counts[counts >= MIN_CALLS_FOR_RECORDING].index
        subset = subset[subset["recording"].isin(usable)]

        if subset["recording"].nunique() < 2 or len(subset) < 12:
            continue

        features = np.stack(subset["embedding"].to_list())
        encoder = LabelEncoder()
        codes = encoder.fit_transform(subset["recording"].to_numpy())
        n_recordings = len(encoder.classes_)

        n_folds = min(N_FOLDS, int(counts.loc[usable].min()))
        splitter = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)

        accuracies = []
        for train, test in splitter.split(features, codes):
            scaler = StandardScaler().fit(features[train])
            model_fit = LogisticRegression(max_iter=1000, random_state=SEED)
            model_fit.fit(scaler.transform(features[train]), codes[train])
            accuracies.append(
                accuracy_score(codes[test], model_fit.predict(scaler.transform(features[test])))
            )

        accuracy = float(np.mean(accuracies))
        chance = 1.0 / n_recordings

        results.append(
            {
                "species": species,
                "model": model,
                "bird": bird,
                "n_recordings": n_recordings,
                "n_calls": len(subset),
                "chance": chance,
                "accuracy": accuracy,
                "lift_over_chance": accuracy / chance,
            }
        )

    return results


# =============================================================================
# Diagnostic 2: within call type
# =============================================================================

def session_aware_centroid(features, labels, groups):
    """Return the session-aware nearest-centroid accuracy.

    The function returns nan when the data cannot support a grouped split.
    """
    encoder = LabelEncoder()
    codes = encoder.fit_transform(labels)

    n_folds = min(N_FOLDS, pd.Series(groups).nunique())
    if n_folds < 2:
        return float("nan")

    accuracies = []
    for train, test in GroupKFold(n_folds).split(features, codes, groups):
        if len(np.unique(codes[train])) < len(encoder.classes_):
            continue
        train_vectors = normalize(features[train])
        templates = normalize(
            np.stack(
                [train_vectors[codes[train] == c].mean(axis=0) for c in range(len(encoder.classes_))]
            )
        )
        predicted = (normalize(features[test]) @ templates.T).argmax(axis=1)
        accuracies.append(accuracy_score(codes[test], predicted))

    return float(np.mean(accuracies)) if accuracies else float("nan")


def within_call_type(frame, species, model):
    """Score individual identification separately for each call type.

    A bird enters a call type only when it has that call type in two or more
    recordings. Without that rule the split cannot be session aware.

    Caution: the number of birds differs between call types, so the chance level
    differs too. Read each row against its own chance column.
    """
    results = []

    def score(name, subset):
        subset = subset.groupby("bird").filter(lambda g: g["recording"].nunique() >= 2)
        n_birds = subset["bird"].nunique()
        if n_birds < 3 or len(subset) < 20:
            return None
        accuracy = session_aware_centroid(
            np.stack(subset["embedding"].to_list()),
            subset["bird"].to_numpy(),
            subset["recording"].to_numpy(),
        )
        chance = 1.0 / n_birds
        return {
            "species": species,
            "model": model,
            "call_type": name,
            "n_birds": n_birds,
            "n_clips": len(subset),
            "chance": chance,
            "accuracy": accuracy,
            "lift_over_chance": accuracy / chance if accuracy == accuracy else float("nan"),
        }

    pooled = score("__pooled__", frame)
    if pooled:
        results.append(pooled)

    for call_type, subset in frame.groupby("call_type"):
        row = score(call_type, subset)
        if row:
            results.append(row)

    return results


# =============================================================================
# Diagnostic 3: kinship
# =============================================================================

def kinship(frame, species, model):
    """Test whether related birds sit closer together than unrelated birds.

    The function builds one mean template for each bird, then compares the mean
    cosine similarity of same-family pairs against different-family pairs.

    The permutation test shuffles the family labels across birds. It reports how
    often a shuffle produces a gap at least as large as the observed gap. A
    small p value means the families are closer than chance.

    The function returns None when there are too few families or too few birds
    with a known family.
    """
    known = frame[~frame["family"].isin(["unknown", "none", "NA"])]
    if known["family"].nunique() < 2 or known["bird"].nunique() < 3:
        return None

    templates = known.groupby("bird").agg(
        family=("family", "first"),
        embedding=("embedding", lambda e: np.mean(np.stack(e.to_list()), axis=0)),
    )

    vectors = normalize(np.stack(templates["embedding"].to_list()))
    families = templates["family"].to_numpy()
    n_birds = len(templates)

    pairs = [(i, j) for i in range(n_birds) for j in range(i + 1, n_birds)]
    similarities = np.array([float(vectors[i] @ vectors[j]) for i, j in pairs])
    same_family = np.array([families[i] == families[j] for i, j in pairs])

    if same_family.sum() < 2 or (~same_family).sum() < 2:
        return None

    observed_gap = similarities[same_family].mean() - similarities[~same_family].mean()

    generator = np.random.default_rng(SEED)
    null_gaps = []
    for _ in range(N_PERMUTATIONS):
        shuffled = generator.permutation(families)
        mask = np.array([shuffled[i] == shuffled[j] for i, j in pairs])
        if mask.sum() and (~mask).sum():
            null_gaps.append(similarities[mask].mean() - similarities[~mask].mean())

    null_gaps = np.array(null_gaps)
    p_value = float((np.sum(null_gaps >= observed_gap) + 1) / (len(null_gaps) + 1))

    return {
        "species": species,
        "model": model,
        "n_birds": n_birds,
        "n_families": int(pd.Series(families).nunique()),
        "same_family_similarity": float(similarities[same_family].mean()),
        "different_family_similarity": float(similarities[~same_family].mean()),
        "gap": float(observed_gap),
        "p_permutation": p_value,
        "n_same_family_pairs": int(same_family.sum()),
        "n_different_family_pairs": int((~same_family).sum()),
    }


# =============================================================================
# Diagnostic 4: pretraining exposure
# =============================================================================

def pretraining_exposure(all_results):
    """Test whether the species gap tracks Xeno-canto exposure.

    all_results is the per-model, per-species accuracy collected by the caller.

    The function reports accuracy divided by chance, so the different bird
    counts of the two species do not confound the comparison. It then compares
    the aa-to-ag ratio between models whose training corpus includes Xeno-canto
    and models whose corpus does not.

    A difference would mean the species gap is partly an artefact of how much of
    each macaw the models saw during pretraining. No difference means the gap is
    a property of the calls.
    """
    from scipy import stats

    panel = CONFIG["models"]
    rows = []
    for model, scores in all_results.items():
        if "aa" not in scores or "ag" not in scores or model not in panel:
            continue
        rows.append(
            {
                "model": model,
                "family": panel[model]["family"],
                "saw_xeno_canto": bool(panel[model]["xeno_canto"]),
                "aa_lift": scores["aa"][0] / scores["aa"][1],
                "ag_lift": scores["ag"][0] / scores["ag"][1],
            }
        )

    if len(rows) < 4:
        return rows, None

    frame = pd.DataFrame(rows)
    frame["species_gap"] = frame["aa_lift"] / frame["ag_lift"]

    exposed = frame[frame.saw_xeno_canto]["species_gap"]
    unexposed = frame[~frame.saw_xeno_canto]["species_gap"]

    summary = None
    if len(exposed) >= 2 and len(unexposed) >= 2:
        statistic, p_value = stats.mannwhitneyu(exposed, unexposed)
        summary = {
            "gap_with_xeno_canto": float(exposed.mean()),
            "gap_without_xeno_canto": float(unexposed.mean()),
            "n_with": int(len(exposed)),
            "n_without": int(len(unexposed)),
            "mannwhitney_u": float(statistic),
            "p_value": float(p_value),
        }

    return frame.to_dict("records"), summary


# =============================================================================
# Entry point
# =============================================================================

def species_scores():
    """Return {model: {species: (accuracy, chance)}} from the stage 3 results.

    Diagnostic 4 needs every model, not just the five the other diagnostics use,
    so it reads the results table rather than recomputing.
    """
    scores = {}
    for species in CONFIG["species"]:
        path = ROOT / "results" / species / "rows.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table = table[table.call_set == "single"]
        # For ag use the lab subset, which is the confound-free one.
        subset = "lab" if species == "ag" and "lab" in set(table.subset) else "all"
        table = table[table.subset == subset]
        for _, row in table.iterrows():
            scores.setdefault(row["model"], {})[species] = (
                float(row["accuracy_byrec"]),
                float(row["chance_inverse_n_birds"]),
            )
    return scores


def main():
    shift_rows, call_type_rows, kinship_rows = [], [], []

    for species in CONFIG["species"]:
        for model in MODELS:
            frame = load_clips(species, model)
            if frame.empty:
                print(f"  {species}/{model}: no embeddings, skipped")
                continue

            print(
                f"{species}/{model}: {len(frame)} clips, "
                f"{frame['bird'].nunique()} birds, "
                f"{frame['recording'].nunique()} recordings",
                flush=True,
            )

            shift_rows += domain_shift(frame, species, model)

            # The within-call-type diagnostic needs more than one call type.
            if frame["call_type"].nunique() > 1:
                call_type_rows += within_call_type(frame, species, model)

            row = kinship(frame, species, model)
            if row:
                kinship_rows.append(row)

    out_dir = ROOT / "results/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(shift_rows).to_csv(out_dir / "domain_shift.csv", index=False)
    pd.DataFrame(call_type_rows).to_csv(out_dir / "within_call_type.csv", index=False)
    pd.DataFrame(kinship_rows).to_csv(out_dir / "kinship.csv", index=False)

    # Diagnostic 4. Needs the stage 3 results, so it is skipped when they are
    # not present.
    exposure_rows, exposure_summary = pretraining_exposure(species_scores())
    if exposure_rows:
        pd.DataFrame(exposure_rows).to_csv(out_dir / "pretraining_exposure.csv", index=False)

    print()
    print(f"Wrote {out_dir}")
    print(f"  domain_shift.csv     {len(shift_rows)} rows")
    print(f"  within_call_type.csv {len(call_type_rows)} rows")
    print(f"  kinship.csv          {len(kinship_rows)} rows")
    if exposure_rows:
        print(f"  pretraining_exposure.csv {len(exposure_rows)} rows")
    if exposure_summary:
        print()
        print("Pretraining exposure control (aa/ag performance ratio):")
        print(f"  models that saw Xeno-canto     {exposure_summary['gap_with_xeno_canto']:.2f} "
              f"(n={exposure_summary['n_with']})")
        print(f"  models that did not            {exposure_summary['gap_without_xeno_canto']:.2f} "
              f"(n={exposure_summary['n_without']})")
        print(f"  Mann-Whitney p = {exposure_summary['p_value']:.3f}")

    if shift_rows:
        summary = pd.DataFrame(shift_rows).groupby(["species", "model"])["lift_over_chance"].mean()
        print()
        print("Recordings are identifiable within one bird at this multiple of chance:")
        for (species, model), lift in summary.items():
            print(f"  {species}/{model:14} {lift:.2f}x")


if __name__ == "__main__":
    main()
