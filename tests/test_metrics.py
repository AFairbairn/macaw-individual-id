#!/usr/bin/env python
"""
Tests for the scoring functions that stage 3 and stage 4 add.

PURPOSE
    Check the new metrics on data whose right answer is known, so a failure
    points at the metric and not at the embeddings.

USAGE
    pytest tests/

WHAT EACH TEST CHECKS
    test_balanced_accuracy_ignores_class_size
        On an unbalanced set, balanced accuracy falls when a small class is
        missed, and plain accuracy hides it.
    test_fused_clip_scores_are_three_values
        The window-to-clip fusion returns accuracy, macro-F1 and balanced
        accuracy, in that order.
    test_pair_scores_separate_a_clean_set
        Verification reaches AUC 1.0 when the birds are far apart, and sits at
        chance when the features hold no identity.
    test_verification_carries_a_fold_spread
        The folded verification returns six values and the spreads are real
        numbers.
    test_label_shuffle_collapses_to_chance
        The control returns about the chance level when the labels are
        permuted, on features that do carry identity.
    test_call_type_task_is_empty_for_one_call_type
        The call-type columns are absent when a species has one call type.
    test_oscr_rewards_rejection_and_identification
        A model that identifies known birds and rejects novel ones scores above
        a model that accepts everything.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name, filename):
    """Import a stage module whose file name starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


score = load("score_frozen", "03_score_frozen.py")


def separable_features(n_birds=4, per_bird=24, recordings=6, dim=8, seed=0):
    """Return features where each bird sits on its own axis, plus noise.

    Every bird appears in every recording, so the recording-grouped split can
    enrol all of them in every fold.
    """
    generator = np.random.default_rng(seed)
    features, labels, groups, clips = [], [], [], []
    for bird in range(n_birds):
        centre = np.zeros(dim)
        centre[bird] = 6.0
        for call in range(per_bird):
            features.append(centre + generator.normal(0, 0.3, dim))
            labels.append(f"bird{bird}")
            groups.append(f"rec{call % recordings}")
            clips.append(f"bird{bird}_call{call}")
    return (
        np.array(features),
        np.array(labels),
        np.array(groups),
        np.array(clips),
    )


def test_balanced_accuracy_ignores_class_size():
    truth = ["a"] * 90 + ["b"] * 10
    always_a = ["a"] * 100

    accuracy, _, balanced = score.fuse_windows_to_clips(
        truth, always_a, [f"clip{i}" for i in range(100)]
    )
    assert accuracy == pytest.approx(0.90)
    assert balanced == pytest.approx(0.50)


def test_fused_clip_scores_are_three_values():
    truth = ["a", "a", "b", "b"]
    predicted = ["a", "b", "b", "b"]
    result = score.fuse_windows_to_clips(truth, predicted, ["c1", "c2", "c3", "c4"])
    assert len(result) == 3
    assert all(isinstance(value, float) for value in result)


def test_pair_scores_separate_a_clean_set():
    from sklearn.preprocessing import normalize

    features, labels, groups, _ = separable_features()
    auc, eer = score.score_pairs(normalize(features), labels, groups)
    assert auc > 0.99
    assert eer < 0.05

    generator = np.random.default_rng(1)
    noise = generator.normal(0, 1, features.shape)
    auc, _ = score.score_pairs(normalize(noise), labels, groups)
    assert 0.4 < auc < 0.6


def test_verification_carries_a_fold_spread():
    features, labels, groups, _ = separable_features()
    result = score.run_verification(features, labels, groups)
    assert len(result) == 6

    auc, eer, auc_sd, eer_sd, auc_scaled, eer_scaled = result
    assert auc > 0.95
    assert auc_scaled > 0.95
    assert auc_sd == auc_sd
    assert eer_sd == eer_sd
    assert eer < 0.1 and eer_scaled < 0.1


def test_label_shuffle_collapses_to_chance():
    features, labels, groups, clips = separable_features()
    real = score.run_probe(features, labels, groups, "by_recording", clips=clips)
    _, _, real_balanced = score.fuse_windows_to_clips(
        real["y_true"], real["y_pred"], real["clip"]
    )
    assert real_balanced > 0.9

    control = score.run_label_shuffle(features, labels, groups, clips)
    chance = 1.0 / len(np.unique(labels))
    assert control["shuffle_balanced_byrec"] < chance + 0.15


def test_call_type_task_is_empty_for_one_call_type():
    features, labels, groups, clips = separable_features()
    one_type = np.array(["contact"] * len(labels))
    assert score.run_call_type_task(features, one_type, groups, clips) == {}


def test_call_type_task_scores_several_call_types():
    features, labels, groups, clips = separable_features()
    # The call type is the axis the feature sits on, so it is recoverable.
    call_types = np.array(["low" if f[:4].argmax() < 2 else "high" for f in features])
    result = score.run_call_type_task(features, call_types, groups, clips)
    assert result["n_call_types"] == 2
    assert result["call_type_chance"] == pytest.approx(0.5)
    assert result["call_type_balanced_byrec"] > 0.9


def test_oscr_rewards_rejection_and_identification():
    pytest.importorskip("torch")
    metric = load("metric_learning_oscr", "04_metric_learning.py")
    from sklearn.preprocessing import normalize

    generator = np.random.default_rng(3)
    known_y = np.array(["a"] * 20 + ["b"] * 20)
    known_train = np.zeros((40, 4))
    known_train[:20, 0] = 5.0
    known_train[20:, 1] = 5.0
    known_train += generator.normal(0, 0.2, known_train.shape)

    known_test = known_train + generator.normal(0, 0.2, known_train.shape)
    novel = np.zeros((20, 4))
    novel[:, 3] = 5.0
    novel += generator.normal(0, 0.2, novel.shape)

    separable = metric.open_set_classification_rate(
        normalize(known_train), known_y, normalize(known_test), known_y, normalize(novel)
    )

    # A set where the novel birds sit on top of the known ones cannot be
    # rejected, so it must not score higher.
    overlapping = metric.open_set_classification_rate(
        normalize(known_train),
        known_y,
        normalize(known_test),
        known_y,
        normalize(known_train + generator.normal(0, 0.2, known_train.shape)),
    )
    assert separable > overlapping
