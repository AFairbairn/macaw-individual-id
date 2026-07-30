#!/usr/bin/env python
"""
04_metric_learning.py

PURPOSE
    Train a small metric-learning head on the frozen embeddings, and test
    whether it makes the distance between two calls encode identity.

USAGE
    python src/04_metric_learning.py --species aa --set single --device cpu
    python src/04_metric_learning.py --species ag --set single --device cpu

    run_all.sh adds --dump-embeddings to both of these commands.

    --set               The call set. 'single' is the only value.
    --device            'cuda' or 'cpu'. The device is detected when this
                        argument is empty. run_all.sh passes 'cpu', because
                        CUDA kernels are not deterministic and the head is
                        small enough to train on a CPU.
    --dump-embeddings   Also write predictions.csv and head_embeddings.csv.

INPUT
    config.yaml
    data/<species>/metadata/<species>_master.csv
    bacpipe_results/<species>/embeddings/
    mfcc_results/<species>/embeddings/

OUTPUT
    results/<species>/<subset>/metric_learning/summary.csv
        One row for each model, method, and evaluation.
    results/<species>/<subset>/metric_learning/eval_B_folds.csv
        One row for each leave-individuals-out repeat, for the variance.
    results/<species>/<subset>/metric_learning/predictions.csv
        One row for each scored window. Written with --dump-embeddings.
    results/<species>/<subset>/metric_learning/head_embeddings.csv
        The vectors in the learned space. Written with --dump-embeddings.

CAUTION
    This script writes its results when a subset finishes. If the process stops
    in the middle of a subset, the work for that subset is lost. Run the script
    inside tmux or screen, or submit it as a batch job. 03_classify.py is
    different, because it appends its results for each model.

THE PROBLEM THIS ADDRESSES
    The frozen embeddings show that identity is DECODABLE. A linear probe
    reaches 0.62 for aa. But the raw cosine geometry does not EXPOSE identity.
    Verification sits at an equal error rate of 0.43 to 0.45, against a chance
    of 0.5.

    In other words, a classifier can find the identity, but distance alone
    cannot. Distance is what an open-set system needs, because an open-set
    system must handle a bird that was never enrolled.

    A metric-learning head is trained to reorganise the space, so that distance
    itself carries the identity.

THE DESIGN
    The head is one linear layer, followed by L2 normalisation.

        StandardScaler -> optional PCA -> Linear(in_dim, projection_dim) -> L2

    The head is deliberately small. A deep network learns the training calls
    instead of the general pattern when the training set is this small.

    Two losses are compared.
      proto   Prototypical networks (Snell et al. 2017).
      supcon  Supervised contrastive (Khosla et al. 2020).

    Knight et al. (2024) recommend triplet loss for this task. The two losses
    used here are more recent and score higher on this data.

THE TWO EVALUATIONS
    A. Closed set, session aware.
       Every bird is enrolled. This tests whether the head helps for birds that
       the system already knows.

       The baseline is the FROZEN cosine centroid, not the linear probe. A
       fair comparison sets the head against the same kind of measurement,
       that is one distance to one template for each bird.

    B. Leave individuals out.
       The head is trained on some birds. It is then scored on birds it has
       never seen. This is the real goal, because a field system meets birds
       that were never enrolled.

       This evaluation has high variance with only 8 aa birds. Report the
       standard deviation across repeats, not the mean alone.

NOTE ON THE ENCOUNTER METRIC
    Evaluation A pools the calls of one bird within one recording into one mean
    query, in the same way as 03_classify.py. The definition is identical, so
    the frozen result and the head result can be compared directly.
"""
import argparse
import collections
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler, normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = Path(os.environ.get("PARROT_DATA", ROOT / "data"))

SEED = CONFIG["seed"]
N_FOLDS = CONFIG["split"]["n_folds"]

SETTINGS = CONFIG["metric_learning"]
PROJECTION_DIM = SETTINGS["projection_dim"]
PCA_DIM = SETTINGS["pca_dim"]
TEMPERATURE = SETTINGS["temperature"]
LEARNING_RATE = SETTINGS["learning_rate"]
WEIGHT_DECAY = SETTINGS["weight_decay"]
MAX_EPOCHS = SETTINGS["max_epochs"]
PATIENCE = SETTINGS["patience"]
LOSSES = SETTINGS["losses"]
HOLDOUT_BIRDS = SETTINGS["holdout_birds"]
HOLDOUT_REPEATS = SETTINGS["holdout_repeats"]
KNN_K = SETTINGS["knn_k"]
FEW_SHOT_SHOTS = SETTINGS["few_shot_shots"]

# The share of each known bird's calls that is held back to score the
# known side of the known-against-novel comparison.
KNOWN_TEST_FRACTION = 0.2

# The number of support-set resamples in the few-shot test.
FEW_SHOT_DRAWS = 5

# The share of each class that forms the support set in the prototypical loss.
PROTO_SUPPORT_FRACTION = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Loading
# =============================================================================

def load_master(species):
    """Return the master table of one species, keyed by the clip stem."""
    path = ROOT / f"data/{species}/metadata/{species}_master.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run src/00_build_master_metadata.py first.")
    table = pd.read_csv(path, dtype=str)
    return {row["original_stem"]: row for _, row in table.iterrows()}


def find_model_dirs(*roots):
    """Return a dictionary that maps a model name to its embedding directory."""
    found = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for directory in sorted(root.iterdir()):
            if directory.is_dir() and "___" in directory.name and any(directory.rglob("*.npy")):
                found[directory.name.split("___")[1].rsplit("-", 1)[0]] = directory
    return found


def load_embeddings(model_dir, model, master, keep_stems):
    """Return one row for each embedding window of one model.

    The bird and the recording come from the master table, never from the file
    name. keep_stems limits the result to the active subset.
    """
    rows = []
    for path in sorted(model_dir.rglob(f"*_{model}.npy")):
        stem = path.name[: -len(f"_{model}.npy")]

        if stem not in master or stem not in keep_stems:
            continue

        record = master[stem]
        if str(record.get("session_known")) != "1" or record.get("recording_id") in (None, "NA"):
            continue

        matrix = np.atleast_2d(np.load(path))
        if matrix.ndim != 2:
            continue

        for window in matrix:
            if np.isfinite(window).all():
                rows.append(
                    (stem, record["bird"].lower(), record["recording_id"], window.astype(np.float32))
                )

    return pd.DataFrame(rows, columns=["clip", "bird", "session", "embedding"])


# =============================================================================
# The head and the losses
# =============================================================================

class Head(nn.Module):
    """One linear projection into the metric space, with an L2 normalised output.

    The output is normalised, so a dot product between two outputs equals their
    cosine similarity.
    """

    def __init__(self, input_dim, projection_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, projection_dim)

    def forward(self, x):
        return F.normalize(self.layer(x), dim=1)


def supcon_loss(z, y, temperature=TEMPERATURE):
    """Return the supervised contrastive loss (Khosla et al. 2020).

    The loss pulls together every pair of calls from the same bird, and pushes
    apart every pair from different birds. The input z must be L2 normalised.
    """
    n = z.shape[0]
    similarity = (z @ z.T) / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

    identity = torch.eye(n, device=z.device)
    exponent = torch.exp(similarity) * (1 - identity)
    log_probability = similarity - torch.log(exponent.sum(1, keepdim=True) + 1e-9)

    positive = (y[:, None] == y[None, :]).float() * (1 - identity)
    positive_count = positive.sum(1)

    per_sample = -(positive * log_probability).sum(1) / (positive_count + 1e-9)
    return (per_sample * (positive_count > 0)).sum() / ((positive_count > 0).sum() + 1e-9)


def proto_loss(z, y, classes, temperature=TEMPERATURE, generator=None):
    """Return the prototypical loss (Snell et al. 2017).

    The function splits each bird's calls into a support set and a query set. It
    builds one prototype for each bird from the support set. It then classifies
    the query calls by cosine similarity to those prototypes.
    """
    support = torch.zeros(len(y), dtype=torch.bool, device=z.device)
    for bird in classes:
        index = (y == bird).nonzero(as_tuple=True)[0]
        k = max(1, int(round(len(index) * PROTO_SUPPORT_FRACTION)))
        chosen = index[torch.randperm(len(index), generator=generator, device=z.device)[:k]]
        support[chosen] = True

    prototypes = F.normalize(
        torch.stack([z[(y == bird) & support].mean(0) for bird in classes]), dim=1
    )

    query = ~support
    # The raw score of each query call against each bird prototype. The
    # temperature sets how sharply the loss separates the birds.
    scores = (z[query] @ prototypes.T) / temperature
    position = {int(bird): i for i, bird in enumerate(classes)}
    target = torch.tensor([position[int(v)] for v in y[query]], device=z.device)
    return F.cross_entropy(scores, target)


def train_head(features, labels, loss_kind, classes):
    """Train one head and return it.

    Training stops when the loss on a held-out slice stops improving. The
    slice never takes a whole class, so every bird stays in the training set.
    """
    generator_numpy = np.random.default_rng(SEED)

    validation = np.zeros(len(labels), bool)
    for bird in classes:
        index = np.where(labels == bird)[0]
        if len(index) < 2:
            continue
        k = min(max(1, int(round(len(index) * 0.15))), len(index) - 1)
        validation[generator_numpy.choice(index, size=k, replace=False)] = True

    position = {bird: i for i, bird in enumerate(classes)}
    codes = np.array([position[v] for v in labels], dtype=np.int64)

    x_train = torch.tensor(features[~validation], dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(codes[~validation], device=DEVICE)
    x_validate = torch.tensor(features[validation], dtype=torch.float32, device=DEVICE)
    y_validate = torch.tensor(codes[validation], device=DEVICE)

    class_index = torch.arange(len(classes), device=DEVICE)
    generator_torch = torch.Generator(device=DEVICE)
    generator_torch.manual_seed(SEED)

    head = Head(features.shape[1], PROJECTION_DIM).to(DEVICE)
    optimiser = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_loss, best_state, since_improvement = 1e9, None, 0

    for _ in range(MAX_EPOCHS):
        head.train()
        optimiser.zero_grad()
        z = head(x_train)
        loss = (
            supcon_loss(z, y_train)
            if loss_kind == "supcon"
            else proto_loss(z, y_train, class_index, generator=generator_torch)
        )
        loss.backward()
        optimiser.step()

        head.eval()
        with torch.no_grad():
            z_validate = head(x_validate)
            validation_loss = (
                supcon_loss(z_validate, y_validate)
                if loss_kind == "supcon"
                else proto_loss(z_validate, y_validate, class_index, generator=generator_torch)
            ).item()

        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= PATIENCE:
                break

    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    return head


def project(head, features):
    """Return the features in the learned metric space."""
    with torch.no_grad():
        return head(torch.tensor(features, dtype=torch.float32, device=DEVICE)).cpu().numpy()


def make_preprocessor(train_features):
    """Return a function that scales, and optionally reduces, the features.

    The scaler is fitted on the training data only, so no information from the
    test data reaches the head.
    """
    scaler = StandardScaler().fit(train_features)
    if PCA_DIM:
        components = min(PCA_DIM, train_features.shape[0] - 1, train_features.shape[1])
        pca = PCA(n_components=components, random_state=SEED).fit(scaler.transform(train_features))
        return lambda x: pca.transform(scaler.transform(x)).astype(np.float32)
    return lambda x: scaler.transform(x).astype(np.float32)


# =============================================================================
# Scoring
# =============================================================================
# Every scorer takes L2 normalised vectors, so a dot product equals a cosine
# similarity.

def centroid_predict(train_z, train_y, test_z, classes):
    """Classify by cosine distance to one mean template for each bird."""
    templates = normalize(np.stack([train_z[train_y == c].mean(0) for c in classes]))
    return classes[(test_z @ templates.T).argmax(1)]


def knn_predict(train_z, train_y, test_z, k=KNN_K):
    """Classify by a vote among the k nearest reference calls.

    A vote handles a bird with several call types better than a single mean
    template, because it does not average across call types.
    """
    classifier = KNeighborsClassifier(n_neighbors=min(k, len(train_z)), metric="cosine")
    classifier.fit(train_z, train_y)
    return classifier.predict(test_z)


def verification_scores(z, y, groups):
    """Return the ROC AUC and the equal error rate over pairs of calls.

    A positive pair holds two calls from the same bird in different recordings.
    Pairs from the same recording are excluded, because they would measure the
    recording rather than the voice.
    """
    similarity = z @ z.T
    upper = np.triu_indices(len(y), 1)
    scores = similarity[upper]

    same_bird = (y[:, None] == y[None, :])[upper]
    same_recording = (groups[:, None] == groups[None, :])[upper]

    positive = same_bird & ~same_recording
    negative = ~same_bird
    keep = positive | negative

    truth = positive[keep].astype(int)
    if truth.sum() == 0 or truth.sum() == len(truth):
        return float("nan"), float("nan")

    auc = roc_auc_score(truth, scores[keep])
    false_positive, true_positive, _ = roc_curve(truth, scores[keep])
    false_negative = 1 - true_positive
    eer = float(false_positive[np.nanargmin(np.abs(false_negative - false_positive))])
    return auc, eer


def cluster_scores(z, y):
    """Return the ARI and the AMI of a KMeans clustering.

    Both metrics are corrected for chance, so they can be compared across
    different numbers of birds.
    """
    k = len(np.unique(y))
    assigned = KMeans(k, random_state=SEED, n_init=10).fit_predict(z)
    return adjusted_rand_score(y, assigned), adjusted_mutual_info_score(y, assigned)


def few_shot_novel_accuracy(known_z, novel_z, novel_y, n_shot, generator):
    """Score the enrolment of a novel bird from a few reference calls.

    The function gives n_shot reference calls for each novel bird. It then
    classifies that bird's remaining calls by the nearest reference. Every call
    of every known bird is added as a distractor, so the test also penalises a
    novel call that is mistaken for a known bird.

    This mirrors the field task: enrol a new individual from a few clips, then
    re-identify it without confusing it for a bird already on file.
    """
    novel_birds = np.unique(novel_y)
    accuracies = []

    for _ in range(FEW_SHOT_DRAWS):
        support = np.zeros(len(novel_y), bool)
        usable = True
        for bird in novel_birds:
            index = np.where(novel_y == bird)[0]
            if len(index) <= n_shot:
                usable = False
                break
            support[generator.choice(index, size=n_shot, replace=False)] = True

        if not usable:
            return float("nan")

        reference_z = np.vstack([novel_z[support], known_z])
        reference_y = np.concatenate(
            [novel_y[support], np.array(["__known__"] * len(known_z))]
        )
        query_z, query_y = novel_z[~support], novel_y[~support]
        predicted = reference_y[(query_z @ reference_z.T).argmax(1)]
        accuracies.append(balanced_accuracy_score(query_y, predicted))

    return float(np.mean(accuracies)) if accuracies else float("nan")


def open_set_auroc(known_reference_z, known_reference_y, known_test_z, novel_z):
    """Score the detection of a novel bird against a known bird.

    The novelty score is the highest cosine similarity to any known bird
    template. A known call should score high. A novel call should score low.
    """
    templates = normalize(
        np.stack(
            [known_reference_z[known_reference_y == c].mean(0) for c in np.unique(known_reference_y)]
        )
    )
    known_scores = (known_test_z @ templates.T).max(1)
    novel_scores = (novel_z @ templates.T).max(1)

    scores = np.concatenate([known_scores, novel_scores])
    labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(novel_scores))])

    if labels.sum() in (0, len(labels)):
        return float("nan")
    return float(roc_auc_score(labels, scores))


# =============================================================================
# One model
# =============================================================================

def evaluate_model(model, frame, dump=False):
    """Score one model under both evaluations.

    Returns (summary rows, per-repeat rows, dumped vectors, predictions).
    """
    features = np.stack(frame["embedding"].to_list())
    labels = frame["bird"].to_numpy()
    groups = frame["session"].to_numpy()
    clips = frame["clip"].to_numpy()
    classes = np.unique(labels)

    summary_rows, repeat_rows, predictions = [], [], []
    dumped = {}

    # ---- Evaluation A: closed set, session aware ----------------------------
    def evaluate_a(head_kind):
        centroid_acc, knn_acc, encounter_acc = [], [], []
        eers, aucs, aris, amis = [], [], [], []

        for train, test in GroupKFold(N_FOLDS).split(features, labels, groups):
            enrollable = np.unique(labels[train])
            keep = np.isin(labels[test], enrollable)

            preprocess = make_preprocessor(features[train])
            x_train, x_test = preprocess(features[train]), preprocess(features[test])

            if head_kind == "frozen":
                z_train, z_test = normalize(x_train), normalize(x_test)
            else:
                head = train_head(x_train, labels[train], head_kind, enrollable)
                z_train, z_test = project(head, x_train), project(head, x_test)

            if keep.any():
                predicted = centroid_predict(z_train, labels[train], z_test[keep], enrollable)
                centroid_acc.append(accuracy_score(labels[test][keep], predicted))
                knn_acc.append(
                    accuracy_score(
                        labels[test][keep], knn_predict(z_train, labels[train], z_test[keep])
                    )
                )

                # Encounter fusion. The definition matches 03_classify.py, so
                # the frozen result and the head result are comparable.
                buckets = collections.defaultdict(list)
                for bird, recording, vector in zip(
                    labels[test][keep], groups[test][keep], z_test[keep]
                ):
                    buckets[(bird, recording)].append(vector)
                query_y = np.array([key[0] for key in buckets])
                queries = normalize(np.stack([np.mean(v, 0) for v in buckets.values()]))
                encounter_acc.append(
                    accuracy_score(
                        query_y, centroid_predict(z_train, labels[train], queries, enrollable)
                    )
                )

                if dump:
                    for global_index, prediction in zip(test[keep], predicted):
                        predictions.append(
                            {
                                "model": model,
                                "method": head_kind,
                                "eval": "A_closed",
                                "clip_id": clips[global_index],
                                "bird": labels[global_index],
                                "session": groups[global_index],
                                "pred": prediction,
                            }
                        )

            auc, eer = verification_scores(z_test, labels[test], groups[test])
            eers.append(eer)
            aucs.append(auc)
            ari, ami = cluster_scores(z_test, labels[test])
            aris.append(ari)
            amis.append(ami)

            if dump:
                dumped.setdefault(head_kind, []).extend(
                    (clips[i], labels[i], groups[i], z_test[j]) for j, i in enumerate(test)
                )

        return {
            "centroid_acc": np.mean(centroid_acc),
            "centroid_acc_sd": np.std(centroid_acc),
            "knn_acc": np.mean(knn_acc),
            "knn_acc_sd": np.std(knn_acc),
            "encounter_acc": np.mean(encounter_acc) if encounter_acc else float("nan"),
            "encounter_acc_sd": np.std(encounter_acc) if encounter_acc else float("nan"),
            "eer": np.nanmean(eers),
            "auc": np.nanmean(aucs),
            "ari": np.mean(aris),
            "ami": np.mean(amis),
        }

    # The linear probe is the decodability ceiling, for context only. It is not
    # the baseline for the head. The baseline is the frozen centroid.
    probe_scores = []
    for train, test in GroupKFold(N_FOLDS).split(features, labels, groups):
        scaler = StandardScaler().fit(features[train])
        classifier = LogisticRegression(max_iter=2000, random_state=SEED)
        classifier.fit(scaler.transform(features[train]), labels[train])
        probe_scores.append(
            accuracy_score(labels[test], classifier.predict(scaler.transform(features[test])))
        )
    probe_ceiling = float(np.mean(probe_scores))

    for kind in ["frozen"] + LOSSES:
        summary_rows.append(
            dict(
                model=model,
                method=kind,
                eval="A_closed",
                **evaluate_a(kind),
                probe_ceiling=probe_ceiling if kind == "frozen" else float("nan"),
                n_windows=len(labels),
                n_birds=len(classes),
                n_recordings=len(np.unique(groups)),
            )
        )

    # ---- Evaluation B: leave individuals out --------------------------------
    generator = np.random.default_rng(SEED)
    held_out_sets = [
        generator.choice(classes, HOLDOUT_BIRDS, replace=False) for _ in range(HOLDOUT_REPEATS)
    ]
    few_shot_generator = np.random.default_rng(SEED + 1)

    for kind in ["frozen"] + LOSSES:
        collected = collections.defaultdict(list)

        for repeat, held_out in enumerate(held_out_sets):
            is_novel = np.isin(labels, held_out)
            known_classes = np.unique(labels[~is_novel])

            # Split the known calls into a part that trains the head and a part
            # that is held back. The head never sees the held-back part, so the
            # known side of the open-set comparison is scored on unseen calls.
            known_index = np.where(~is_novel)[0]
            train_parts, test_parts = [], []
            for bird in known_classes:
                bird_index = known_index[labels[known_index] == bird].copy()
                few_shot_generator.shuffle(bird_index)
                cut = max(1, int(round(len(bird_index) * (1 - KNOWN_TEST_FRACTION))))
                train_parts.append(bird_index[:cut])
                test_parts.append(bird_index[cut:])

            known_train = np.concatenate(train_parts)
            known_test = np.concatenate(test_parts)

            preprocess = make_preprocessor(features[known_train])
            x_known_train = preprocess(features[known_train])
            x_known_test = preprocess(features[known_test])
            x_novel = preprocess(features[is_novel])

            y_known_train = labels[known_train]
            y_novel = labels[is_novel]
            groups_novel = groups[is_novel]

            if kind == "frozen":
                z_known_train = normalize(x_known_train)
                z_known_test = normalize(x_known_test)
                z_novel = normalize(x_novel)
            else:
                head = train_head(x_known_train, y_known_train, kind, known_classes)
                z_known_train = project(head, x_known_train)
                z_known_test = project(head, x_known_test)
                z_novel = project(head, x_novel)

            _, eer = verification_scores(z_novel, y_novel, groups_novel)
            ari, ami = cluster_scores(z_novel, y_novel)
            auroc = (
                open_set_auroc(z_known_train, y_known_train, z_known_test, z_novel)
                if len(z_known_test)
                else float("nan")
            )

            values = {
                "eer": eer,
                "ari": ari,
                "ami": ami,
                "open_set_auroc": auroc,
                f"few_shot_{FEW_SHOT_SHOTS[0]}": few_shot_novel_accuracy(
                    z_known_train, z_novel, y_novel, FEW_SHOT_SHOTS[0], few_shot_generator
                ),
                f"few_shot_{FEW_SHOT_SHOTS[1]}": few_shot_novel_accuracy(
                    z_known_train, z_novel, y_novel, FEW_SHOT_SHOTS[1], few_shot_generator
                ),
            }

            for key, value in values.items():
                collected[key].append(value)

            repeat_rows.append(
                dict(
                    model=model,
                    method=kind,
                    repeat=repeat,
                    held_out="|".join(map(str, held_out)),
                    **values,
                )
            )

        summary_rows.append(
            dict(
                model=model,
                method=kind,
                eval="B_leave_individuals_out",
                eer=np.nanmean(collected["eer"]),
                eer_sd=np.nanstd(collected["eer"]),
                ari=np.nanmean(collected["ari"]),
                ari_sd=np.nanstd(collected["ari"]),
                ami=np.nanmean(collected["ami"]),
                open_set_auroc=np.nanmean(collected["open_set_auroc"]),
                **{
                    f"few_shot_{n}": np.nanmean(collected[f"few_shot_{n}"])
                    for n in FEW_SHOT_SHOTS
                },
            )
        )

    return summary_rows, repeat_rows, dumped, predictions


# =============================================================================
# Entry point
# =============================================================================

def subset_filter(species, name):
    """Return a function that selects the rows of one subset."""
    rule = CONFIG["subsets"][species][name]
    if rule is None:
        return lambda record: True
    column, value = next(iter(rule.items()))
    return lambda record: record[column] == value


def main():
    parser = argparse.ArgumentParser(description="Train and score the metric-learning head.")
    parser.add_argument("--species", required=True, choices=CONFIG["species"])
    parser.add_argument("--set", default="single", choices=["single"])
    parser.add_argument("--device", default="", help="'cuda' or 'cpu'. Detected when not set.")
    parser.add_argument(
        "--dump-embeddings",
        action="store_true",
        help="Also write the head vectors and the per-window predictions.",
    )
    parser.add_argument("--subset", default="", help="Score one subset only.")
    args = parser.parse_args()

    global DEVICE
    if args.device:
        DEVICE = args.device

    species = args.species
    master = load_master(species)
    model_dirs = find_model_dirs(
        ROOT / f"bacpipe_results/{species}/embeddings",
        ROOT / f"mfcc_results/{species}/embeddings",
    )
    if not model_dirs:
        raise SystemExit(f"No embeddings found for {species}. Run stage 1 and stage 2 first.")

    subsets = [args.subset] if args.subset else list(CONFIG["subsets"][species])
    print(
        f"species={species} | {len(model_dirs)} models | subsets={subsets} | "
        f"device={DEVICE} | projection_dim={PROJECTION_DIM}",
        flush=True,
    )

    for subset in subsets:
        keep = subset_filter(species, subset)
        keep_stems = {stem for stem, record in master.items() if keep(record)}

        out_dir = ROOT / "results" / species / subset / "metric_learning"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- subset={subset} ({len(keep_stems)} clips) -> {out_dir}", flush=True)

        all_summary, all_repeats, all_predictions, all_dumps = [], [], [], []

        for model, directory in sorted(model_dirs.items()):
            frame = load_embeddings(directory, model, master, keep_stems)
            if len(frame) < 20 or frame["bird"].nunique() < 3:
                print(f"  {model}: too few calls, skipped", flush=True)
                continue

            summary, repeats, dumped, predictions = evaluate_model(
                model, frame, dump=args.dump_embeddings
            )
            all_summary += summary
            all_repeats += repeats
            all_predictions += predictions

            if dumped:
                for method, records in dumped.items():
                    if method == "frozen":
                        # The frozen vectors are already on disk as .npy files.
                        continue
                    for clip, bird, session, vector in records:
                        all_dumps.append(
                            {
                                "model": model,
                                "method": method,
                                "clip_id": clip,
                                "bird": bird,
                                "session": session,
                                **{f"e{i}": float(v) for i, v in enumerate(vector)},
                            }
                        )

            frozen = next(r for r in summary if r["method"] == "frozen" and r["eval"] == "A_closed")
            best = next(r for r in summary if r["method"] == LOSSES[-1] and r["eval"] == "A_closed")
            print(
                f"  {model}: frozen centroid {frozen['centroid_acc']:.3f} "
                f"encounter {frozen['encounter_acc']:.3f} eer {frozen['eer']:.3f} "
                f"-> {LOSSES[-1]} centroid {best['centroid_acc']:.3f} "
                f"encounter {best['encounter_acc']:.3f} eer {best['eer']:.3f}",
                flush=True,
            )

        # The results are written once, when the subset finishes. See the
        # caution in the module docstring.
        pd.DataFrame(all_summary).to_csv(out_dir / "summary.csv", index=False)
        pd.DataFrame(all_repeats).to_csv(out_dir / "eval_B_folds.csv", index=False)
        print(f"  wrote summary.csv and eval_B_folds.csv", flush=True)

        if all_predictions:
            pd.DataFrame(all_predictions).to_csv(out_dir / "predictions.csv", index=False)
            print(f"  wrote predictions.csv ({len(all_predictions)} rows)", flush=True)
        if all_dumps:
            pd.DataFrame(all_dumps).to_csv(out_dir / "head_embeddings.csv", index=False)
            print(f"  wrote head_embeddings.csv ({len(all_dumps)} rows)", flush=True)


if __name__ == "__main__":
    main()
