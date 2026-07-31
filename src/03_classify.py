#!/usr/bin/env python
"""
03_classify.py

PURPOSE
    Score every embedding model at individual identification. This script
    produces the main results table of the paper.

USAGE
    python src/03_classify.py --species aa
    python src/03_classify.py --species ag

INPUT
    config.yaml                                     Every analysis choice.
    data/<species>/metadata/<species>_master.csv    The bird and the recording
                                                    of every clip.
    bacpipe_results/<species>/embeddings/           The 15 pre-trained models.
    mfcc_results/<species>/embeddings/              The 3 MFCC variants.

OUTPUT
    results/<species>/rows.csv    One row for each model, subset, and call set.
    results/<species>/preds.csv   One row for each scored window.

RESUMING
    The script appends one row for each model and skips completed work. To
    rescore one model, delete its row from results/<species>/rows.csv and run
    the script again.

THE SPLIT
    The split groups on recording_id. Calls from one recording never appear in
    both the train set and the test set.

    This is the most important choice in the analysis. A recording carries an
    acoustic signature. Within one bird, a model identifies the recording at 2.7
    to 3.7 times chance, from the audio alone. If one recording appears on both
    sides of the split, the model can use that signature instead of the voice.

    The script also reports a random split. That result is too high. The
    difference between the two splits is the leakage delta.

THE METRICS
    probe        A logistic regression classifier on the embeddings.
    centroid     Cosine distance to one mean template for each bird.
    encounter    See the note below.
    verification EER and AUROC over pairs of calls.
    clustering   KMeans, given the true number of birds.

NOTE ON THE ENCOUNTER METRIC
    The encounter metric pools the calls of one bird within one recording into
    one mean query. It answers this question: given a set of calls that are
    known to come from one bird in one recording, which bird is it?

    The metric assumes that the calls are already grouped by individual. The
    assumption is necessary, because 14 of 74 ag recordings and 2 of 211 aa
    recordings hold calls from two birds. To pool every call in such a
    recording would average two birds into one query.

CAUTION ON THE BASELINE
    ag is not balanced. The number of calls for each bird runs from 41 to 108.
    The script reports both `chance_inverse_n_birds` and `chance_majority_class`.
    Report the majority-class value. If you report 1/n_birds alone, the result
    looks better than it is.

CAUTION ON THE CLUSTERING METRICS
    Do not report NMI on its own. NMI increases as the number of clusters
    increases, so it favours any method that produces many clusters. The script
    reports NMI, AMI and ARI together. ARI is the primary value, because it is
    corrected for chance.
"""
import argparse
import collections
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

SEED = CONFIG["seed"]
N_FOLDS = CONFIG["split"]["n_folds"]

# The dataset can live outside the repository. run_all.sh sets PARROT_DATA.
DATA = Path(os.environ.get("PARROT_DATA", ROOT / "data"))

# =============================================================================
# Loading
# =============================================================================

def load_master(species):
    """Return the master table of one species, keyed by the clip stem.

    The master table is the single source of truth for the bird identity and the
    recording of every clip. 00_build_master_metadata.py writes it.
    """
    path = ROOT / f"data/{species}/metadata/{species}_master.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run src/00_build_master_metadata.py first.")
    table = pd.read_csv(path, dtype=str)
    return {row["original_stem"]: row for _, row in table.iterrows()}


def find_model_dirs(*roots):
    """Return a dictionary that maps a model name to its embedding directory.

    A model directory holds '___' in its name. The model name is the part
    between '___' and the trailing species tag.
    """
    found = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or "___" not in directory.name:
                continue
            if not any(directory.rglob("*.npy")):
                continue
            name = directory.name.split("___")[1].rsplit("-", 1)[0]
            found[name] = directory
    return found


def collect_single_stems(model_dirs):
    """Return the stems of every clip that holds one call.

    The set is built across all models, so that the definition of a single call
    does not depend on which model is being scored.
    """
    return {
        path.name[: -len(f"_{model}.npy")]
        for model, directory in model_dirs.items()
        for path in directory.rglob(f"*_{model}.npy")
        if "/single/" in path.as_posix()
    }


def load_embeddings(model_dir, model, master, single_stems):
    """Return one row for each embedding window of one model.

    The bird and the recording come from the master table, never from the file
    name. Parsing the file name was the source of the leakage that this
    protocol fixes.

    The function skips a clip when any of these is true:
      - The clip is not in the master table.
      - The recording of the clip could not be resolved.
      - The embedding file does not load, or it holds no matrix.
    """
    rows = []
    for path in sorted(model_dir.rglob(f"*_{model}.npy")):
        stem = path.name[: -len(f"_{model}.npy")]

        if stem not in master:
            continue

        record = master[stem]
        if str(record.get("session_known")) != "1" or record.get("recording_id") in (None, "NA"):
            continue

        try:
            matrix = np.atleast_2d(np.load(path))
        except (OSError, ValueError) as error:
            # A clip that is dropped here reduces the sample size of this model
            # against every other. Say which clip, and why.
            print(f"  {model}: {path.name} did not load, so this clip is not "
                  f"scored. {type(error).__name__}: {error}", flush=True)
            continue
        if matrix.ndim != 2:
            continue

        kind = "single" if stem in single_stems else record.get("kind", "repeated")

        for window in matrix:
            if np.isfinite(window).all():
                rows.append(
                    (
                        stem,
                        record["bird"].lower(),
                        record["recording_id"],
                        kind,
                        record["environment_class"],
                        window.astype(np.float32),
                    )
                )

    return pd.DataFrame(
        rows, columns=["clip_id", "bird", "session", "kind", "environment_class", "embedding"]
    )


# =============================================================================
# Metrics
# =============================================================================

def make_splits(features, labels, groups, mode):
    """Return the train and test folds for one split mode.

    mode is 'by_recording' or 'random'.

    'by_recording' groups on the recording, so no recording appears on both
    sides. This is the primary split.

    'random' ignores the recording. Its result is too high. Report it only as
    the leaky upper bound.
    """
    if mode == "by_recording":
        return GroupKFold(n_splits=N_FOLDS).split(features, labels, groups)
    return StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(features, labels)


def run_probe(features, labels, groups, mode, clips=None):
    """Fit a logistic regression classifier and return its accuracy.

    The classifier is refitted inside every fold. The scaler is fitted on the
    train fold only, so no information from the test fold reaches the model.
    """
    encoder = LabelEncoder()
    codes = encoder.fit_transform(labels)

    accuracies, f1_scores, truths, predictions, clip_ids = [], [], [], [], []

    for train, test in make_splits(features, codes, groups, mode):
        # Skip a fold that cannot enrol every bird. tests/test_splits.py asserts
        # that this never happens for the published subsets.
        if len(np.unique(codes[train])) < len(encoder.classes_):
            continue

        scaler = StandardScaler().fit(features[train])
        # The fit runs on one thread. run_all.sh sets OMP_NUM_THREADS,
        # MKL_NUM_THREADS and OPENBLAS_NUM_THREADS to 1, which holds the
        # underlying linear algebra to one thread and makes the result the same
        # on every machine.
        model = LogisticRegression(max_iter=1500, random_state=SEED)
        model.fit(scaler.transform(features[train]), codes[train])
        predicted = model.predict(scaler.transform(features[test]))

        accuracies.append(accuracy_score(codes[test], predicted))
        f1_scores.append(f1_score(codes[test], predicted, average="macro"))
        truths += list(encoder.classes_[codes[test]])
        predictions += list(encoder.classes_[predicted])
        if clips is not None:
            clip_ids += list(clips[test])

    return {
        "accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "sd": float(np.std(accuracies)) if accuracies else float("nan"),
        "f1": float(np.mean(f1_scores)) if f1_scores else float("nan"),
        "y_true": truths,
        "y_pred": predictions,
        "clip": clip_ids,
    }


def run_centroid(features, labels, groups, mode, clips=None):
    """Classify each test call by cosine distance to one template per bird.

    The template is the mean of that bird's training calls. This metric matches
    the enrolment case, where one reference template represents each individual.

    The vectors are L2 normalised, so a dot product equals a cosine similarity.
    """
    encoder = LabelEncoder()
    codes = encoder.fit_transform(labels)

    accuracies, f1_scores, truths, predictions, clip_ids = [], [], [], [], []

    for train, test in make_splits(features, codes, groups, mode):
        if len(np.unique(codes[train])) < len(encoder.classes_):
            continue

        train_vectors = normalize(features[train])
        test_vectors = normalize(features[test])
        templates = normalize(
            np.stack(
                [train_vectors[codes[train] == c].mean(axis=0) for c in range(len(encoder.classes_))]
            )
        )
        predicted = (test_vectors @ templates.T).argmax(axis=1)

        accuracies.append(accuracy_score(codes[test], predicted))
        f1_scores.append(f1_score(codes[test], predicted, average="macro"))
        truths += list(encoder.classes_[codes[test]])
        predictions += list(encoder.classes_[predicted])
        if clips is not None:
            clip_ids += list(clips[test])

    return {
        "accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "sd": float(np.std(accuracies)) if accuracies else float("nan"),
        "f1": float(np.mean(f1_scores)) if f1_scores else float("nan"),
        "y_true": truths,
        "y_pred": predictions,
        "clip": clip_ids,
    }


def fuse_windows_to_clips(y_true, y_pred, clips):
    """Combine the window predictions of one clip into one prediction.

    Most models return one window for each clip. Two models return more. This
    function makes the accuracy comparable across all models, by reducing every
    model to one prediction per clip. The rule is a majority vote.
    """
    votes, truth = collections.defaultdict(list), {}
    for clip, true_label, predicted in zip(clips, y_true, y_pred):
        votes[clip].append(predicted)
        truth[clip] = true_label

    clip_ids = list(votes)
    truths = [truth[c] for c in clip_ids]
    predictions = [collections.Counter(votes[c]).most_common(1)[0][0] for c in clip_ids]

    return (
        float(accuracy_score(truths, predictions)),
        float(f1_score(truths, predictions, average="macro")),
    )


def run_encounter(features, labels, groups):
    """Pool the calls of one bird in one recording, then classify the pool.

    The pooled query is the mean of the L2 normalised calls. The query is
    matched to the nearest bird template.

    This is the deployment unit. In the field you rarely classify one call. You
    classify an encounter.

    NOTE: The function groups on (bird, recording), not on recording alone. Some
    recordings hold two birds. To pool every call of such a recording would
    average two birds into one query. The metric therefore assumes that the
    calls are already grouped by individual.
    """
    encoder = LabelEncoder()
    codes = encoder.fit_transform(labels)
    accuracies = []

    for train, test in GroupKFold(n_splits=N_FOLDS).split(features, codes, groups):
        if len(np.unique(codes[train])) < len(encoder.classes_):
            continue

        train_vectors = normalize(features[train])
        templates = normalize(
            np.stack(
                [train_vectors[codes[train] == c].mean(axis=0) for c in range(len(encoder.classes_))]
            )
        )

        test_vectors = normalize(features[test])
        buckets = collections.defaultdict(list)
        for index in range(len(test)):
            buckets[(codes[test][index], groups[test][index])].append(test_vectors[index])

        if not buckets:
            continue

        query_labels = np.array([key[0] for key in buckets])
        queries = normalize(np.stack([np.mean(v, axis=0) for v in buckets.values()]))
        accuracies.append(accuracy_score(query_labels, (queries @ templates.T).argmax(axis=1)))

    return float(np.mean(accuracies)) if accuracies else float("nan")


def run_verification(features, labels, groups):
    """Return the ROC AUC and the equal error rate over pairs of calls.

    A positive pair holds two calls from the same bird in different recordings.
    Pairs from the same recording are excluded, because they would measure the
    recording rather than the voice.

    A negative pair holds two calls from different birds.

    The equal error rate is the point where the false positive rate equals the
    false negative rate. Lower is better. Chance is 0.5.

    This metric matters most, because it generalises to the open set. A
    verification score does not need the bird to be enrolled in advance.
    """
    vectors = normalize(features)
    similarity = vectors @ vectors.T

    upper = np.triu_indices(len(labels), k=1)
    scores = similarity[upper]
    same_bird = (labels[:, None] == labels[None, :])[upper]
    same_recording = (groups[:, None] == groups[None, :])[upper]

    positive = same_bird & ~same_recording
    negative = ~same_bird
    keep = positive | negative

    truth = positive[keep].astype(int)
    kept_scores = scores[keep]

    if truth.sum() == 0 or truth.sum() == len(truth):
        return float("nan"), float("nan")

    auc = float(roc_auc_score(truth, kept_scores))
    false_positive, true_positive, _ = roc_curve(truth, kept_scores)
    false_negative = 1 - true_positive
    eer = float(false_positive[np.nanargmin(np.abs(false_negative - false_positive))])
    return auc, eer


def run_clustering(features, labels, groups):
    """Cluster the embeddings and score the clusters against the bird identity.

    One algorithm runs: KMeans, given the true number of birds. Three metrics
    are reported. ARI is the primary one, because it is corrected for chance.
    NMI is reported for comparison with Lakdari et al. (2024), who use it.

    CAUTION: Do not report NMI on its own. NMI increases as the number of
    clusters increases, so it favours any method that produces many clusters.
    Read it next to ARI.

    NOTE ON THE SPLIT: This function clusters every call at once. There is no
    train and test split, because clustering needs no training. That matches how
    the field reports clustering (Lakdari et al. 2024, Clink et al. 2021).

    Clustering has no training step, so there is nothing for a split to
    protect. A session-aware variant would cluster fewer calls, which lowers
    the scores for a reason that has nothing to do with the question.

    The real concern is different, and it is measured directly: a cluster can
    form around a RECORDING instead of around a BIRD. So the function scores the
    same clustering against both labellings. Compare `cluster_ari` (against the
    bird) with `cluster_ari_recording` (against the recording). If the second is
    higher, the structure the model finds is the recording, not the voice.

    Affinity propagation and HDBSCAN were tested and removed. Affinity
    propagation inferred 37 to 85 clusters for 8 to 16 birds, and HDBSCAN
    labelled 36 to 100 percent of calls as noise. Neither recovered individual
    identity, and reporting them added length without adding information.
    """
    vectors = normalize(features)
    n_birds = len(np.unique(labels))
    assigned = KMeans(n_birds, random_state=SEED, n_init=10).fit_predict(vectors)

    return {
        # Against the bird. This is the result.
        "cluster_ari": float(adjusted_rand_score(labels, assigned)),
        "cluster_ami": float(adjusted_mutual_info_score(labels, assigned)),
        "cluster_nmi": float(normalized_mutual_info_score(labels, assigned)),
        # Against the recording. This is the control. A higher value here than
        # above means the clusters track the recording, not the bird.
        "cluster_ari_recording": float(adjusted_rand_score(groups, assigned)),
    }


# =============================================================================
# One model
# =============================================================================

def score_model(frame, species, subset, call_set, model):
    """Score one model on one subset and return its result row.

    The function returns (None, None) when the data is too small to score.
    """
    if len(frame) < 20 or frame["bird"].nunique() < 2:
        return None, None

    features = np.stack(frame["embedding"].to_list())
    labels = frame["bird"].to_numpy()
    groups = frame["session"].to_numpy()
    clips = frame["clip_id"].to_numpy()

    n_birds = int(frame["bird"].nunique())

    # Both splits are scored with clips, so the random split also yields a
    # clip-level macro-F1. The primary metric is macro-F1, so its leakage delta
    # needs both halves.
    probe_grouped = run_probe(features, labels, groups, "by_recording", clips=clips)
    probe_random = run_probe(features, labels, groups, "random", clips=clips)
    centroid_grouped = run_centroid(features, labels, groups, "by_recording", clips=clips)
    centroid_random = run_centroid(features, labels, groups, "random", clips=clips)

    # The reported unit is the CLIP, that is one call. Most models return one
    # window for each clip, so this changes nothing for them. Two models
    # (aves_especies, birdaves_especies) return more than one window for a few
    # clips. Fusing to the clip keeps every model on the same unit. The measured
    # difference is at most 0.004 accuracy.
    probe_clip_accuracy, probe_clip_f1 = fuse_windows_to_clips(
        probe_grouped["y_true"], probe_grouped["y_pred"], probe_grouped["clip"]
    )
    probe_random_clip_accuracy, probe_random_clip_f1 = fuse_windows_to_clips(
        probe_random["y_true"], probe_random["y_pred"], probe_random["clip"]
    )
    centroid_clip_accuracy, _ = fuse_windows_to_clips(
        centroid_grouped["y_true"], centroid_grouped["y_pred"], centroid_grouped["clip"]
    )

    encounter_accuracy = run_encounter(features, labels, groups)
    auc, eer = run_verification(features, labels, groups)

    # Two baselines. For an unbalanced set, the majority-class rate is the
    # honest one. See the caution in the module docstring.
    call_counts = frame.groupby("bird")["clip_id"].nunique()
    chance_inverse = 1.0 / n_birds
    chance_majority = float(call_counts.max() / call_counts.sum())

    row = {
        # --- identifiers -----------------------------------------------------
        "species": species,
        "subset": subset,
        "call_set": call_set,
        "model": model,
        "n_clips": int(frame["clip_id"].nunique()),
        "n_windows": len(frame),
        "n_birds": n_birds,
        "n_recordings": int(frame["session"].nunique()),
        "dim": int(features.shape[1]),

        # --- baselines -------------------------------------------------------
        # ag is unbalanced, so the majority-class rate is the honest baseline.
        # aa is balanced, so the two values agree there.
        "chance_inverse_n_birds": chance_inverse,
        "chance_majority_class": chance_majority,

        # --- MAIN TEXT metric 1 of 4: accuracy, both splits -------------------
        # The unit is the clip, that is one call.
        "accuracy_byrec": probe_clip_accuracy,
        "accuracy_random": probe_random_clip_accuracy,
        "accuracy_leakage_delta": probe_random_clip_accuracy - probe_clip_accuracy,
        "accuracy_byrec_sd": probe_grouped["sd"],

        # --- MAIN TEXT metric 2 of 4: macro-F1, both splits -------------------
        # Macro-F1 is the PRIMARY metric, because it is not distorted by the ag
        # class imbalance. Gallego et al. (2026) use it as primary for the same
        # reason.
        "macro_f1_byrec": probe_clip_f1,
        "macro_f1_random": probe_random_clip_f1,
        "macro_f1_leakage_delta": probe_random_clip_f1 - probe_clip_f1,

        # --- MAIN TEXT metric 3 of 4: verification AUC ------------------------
        # Comparable to Stowell et al. (2019) and Huang et al. (2025), who both
        # report AU-ROC. EER is kept below for the supplement. The two rank
        # the models almost identically (Spearman 0.98).
        "verification_auc": auc,

        # --- MAIN TEXT metric 4 of 4: encounter accuracy ----------------------
        # The deployment unit. See the note in the module docstring.
        "encounter_accuracy": encounter_accuracy,

        # --- SUPPLEMENT ------------------------------------------------------
        "verification_eer": eer,
        "centroid_accuracy_byrec": centroid_grouped["accuracy"],
        "centroid_accuracy_random": centroid_random["accuracy"],
        "centroid_macro_f1_byrec": centroid_grouped["f1"],
        # Window-level accuracy, kept so the clip-level fusion can be checked.
        "accuracy_byrec_window_level": probe_grouped["accuracy"],
    }
    row.update(run_clustering(features, labels, groups))

    predictions = [
        {
            "species": species,
            "subset": subset,
            "call_set": call_set,
            "model": model,
            "clip_id": clip,
            "true": true_label,
            "pred": predicted,
        }
        for clip, true_label, predicted in zip(
            probe_grouped["clip"], probe_grouped["y_true"], probe_grouped["y_pred"]
        )
    ]
    return row, predictions


# =============================================================================
# Entry point
# =============================================================================

def completed_keys(path):
    """Return the (subset, call_set, model) keys that are already scored."""
    if not Path(path).exists():
        return set()
    try:
        done = pd.read_csv(path)
        return set(zip(done["subset"], done["call_set"], done["model"]))
    except (OSError, KeyError, pd.errors.EmptyDataError):
        return set()


def append_rows(path, rows):
    """Append one or more rows to a CSV file, writing the header once."""
    frame = pd.DataFrame([rows]) if isinstance(rows, dict) else pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not Path(path).exists(), index=False)


def subset_filter(species, name):
    """Return a function that selects the rows of one subset."""
    rule = CONFIG["subsets"][species][name]
    if rule is None:
        return lambda record: True
    column, value = next(iter(rule.items()))
    return lambda record: record[column] == value


def main():
    parser = argparse.ArgumentParser(description="Score every model at individual identification.")
    parser.add_argument("--species", required=True, choices=CONFIG["species"])
    args = parser.parse_args()
    species = args.species

    master = load_master(species)
    model_dirs = find_model_dirs(
        ROOT / f"bacpipe_results/{species}/embeddings",
        ROOT / f"mfcc_results/{species}/embeddings",
    )
    if not model_dirs:
        raise SystemExit(
            f"No embeddings found for {species}. Run stage 1 and stage 2 first."
        )

    single_stems = collect_single_stems(model_dirs)

    out_dir = ROOT / "results" / species
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.csv"
    preds_path = out_dir / "preds.csv"

    done = completed_keys(rows_path)
    print(
        f"species={species} | {len(model_dirs)} models | "
        f"subsets={list(CONFIG['subsets'][species])} | already scored={len(done)}",
        flush=True,
    )

    for subset in CONFIG["subsets"][species]:
        keep = subset_filter(species, subset)
        keep_stems = {stem for stem, record in master.items() if keep(record)}

        # The main analysis uses single calls only, for both species. This is
        # the matched acoustic unit across the two species.
        #
        # aa also holds repeated call bouts. A bout carries more acoustic
        # material, so including it raises aa accuracy by 0.03 to 0.07. ag has
        # no bouts. To include the bouts for aa alone would put part of the
        # reported species difference down to the call set rather than the
        # biology, so the main analysis excludes them.
        #
        # 09_supplementary_bouts.py scores the bouts separately.
        call_sets = [(CONFIG["call_set"], True)]

        for call_set, single_only in call_sets:
            for model, directory in sorted(model_dirs.items()):
                if (subset, call_set, model) in done:
                    continue

                frame = load_embeddings(directory, model, master, single_stems)
                frame = frame[frame["clip_id"].isin(keep_stems)]
                if single_only:
                    frame = frame[frame["kind"] == "single"]

                row, predictions = score_model(frame, species, subset, call_set, model)
                if row is None:
                    print(f"  [{subset}/{call_set}] {model}: too few calls, skipped", flush=True)
                    continue

                append_rows(rows_path, row)
                append_rows(preds_path, predictions)
                done.add((subset, call_set, model))

                print(
                    f"  [{subset}/{call_set}] {model}: "
                    f"F1 {row['macro_f1_byrec']:.3f} "
                    f"acc {row['accuracy_byrec']:.3f} "
                    f"(random {row['accuracy_random']:.3f}, "
                    f"delta {row['accuracy_leakage_delta']:+.3f}, "
                    f"majority {row['chance_majority_class']:.3f}) "
                    f"auc {row['verification_auc']:.3f} "
                    f"encounter {row['encounter_accuracy']:.3f}",
                    flush=True,
                )

    print(f"species={species} complete: {len(done)} model combinations", flush=True)


if __name__ == "__main__":
    main()
