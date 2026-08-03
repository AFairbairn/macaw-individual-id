#!/usr/bin/env python
"""
Tests for the train and test split.

PURPOSE
    Assert that the split does not leak. Leakage is the failure mode that this
    paper is about, so it is tested directly.

USAGE
    pytest tests/

WHAT EACH TEST CHECKS
    test_no_recording_spans_the_split
        No recording_id appears in both the train set and the test set.
    test_every_fold_is_scored
        No fold is dropped because the train set is missing a bird.
    test_chance_levels_are_correct
        The reported baselines match the data.
    test_encounter_grouping_is_documented
        Recordings that hold two birds are counted, because the encounter
        metric depends on that count.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())


def load_master(species):
    """Return the scored rows of the master table for one species.

    The function keeps only single calls with a known recording. Those are the
    rows that the session-aware analysis uses.
    """
    path = ROOT / f"data/{species}/metadata/{species}_master.csv"
    if not path.exists():
        pytest.skip(f"{path} not found. Run src/00_prepare_data.py first.")
    table = pd.read_csv(path, dtype=str)
    return table[table["kind"] == "single"]


def subsets_for(species):
    """Yield each (name, table) subset that the pipeline scores."""
    master = load_master(species)
    for name, rule in CONFIG["subsets"][species].items():
        if rule is None:
            yield name, master
        else:
            column, value = next(iter(rule.items()))
            yield name, master[master[column] == value]


ALL_SUBSETS = [
    (species, name)
    for species in CONFIG["species"]
    for name, _ in subsets_for(species)
]


@pytest.mark.parametrize("species,subset", ALL_SUBSETS)
def test_no_recording_spans_the_split(species, subset):
    """Assert that no recording appears on both sides of the split.

    This is the central guarantee of the evaluation protocol. If it fails, every
    accuracy in the paper is too high.
    """
    table = dict(subsets_for(species))[subset]
    labels = table["bird"].to_numpy()
    groups = table["recording_id"].to_numpy()

    splitter = GroupKFold(n_splits=CONFIG["split"]["n_folds"])
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        shared = set(groups[train]) & set(groups[test])
        assert not shared, (
            f"{species}/{subset} fold {fold}: "
            f"{len(shared)} recording(s) appear in both the train set and the "
            f"test set. Example: {sorted(shared)[:3]}"
        )


@pytest.mark.parametrize("species,subset", ALL_SUBSETS)
def test_every_fold_is_scored(species, subset):
    """Assert that no fold is dropped.

    03_score_frozen.py skips a fold when the train set does not hold every bird. A
    dropped fold makes the reported standard deviation misleading, because the
    mean then covers fewer folds than the paper states.
    """
    table = dict(subsets_for(species))[subset]
    labels = table["bird"].to_numpy()
    groups = table["recording_id"].to_numpy()
    n_birds = len(np.unique(labels))
    n_folds = CONFIG["split"]["n_folds"]

    scored = sum(
        len(np.unique(labels[train])) == n_birds
        for train, _ in GroupKFold(n_splits=n_folds).split(np.zeros(len(labels)), labels, groups)
    )
    assert scored == n_folds, (
        f"{species}/{subset}: only {scored} of {n_folds} folds are scored. "
        f"A fold was dropped because its train set is missing a bird."
    )


@pytest.mark.parametrize("species,subset", ALL_SUBSETS)
def test_chance_levels_are_correct(species, subset):
    """Assert that both baselines are available and ordered correctly.

    The majority-class rate is never below 1/n_birds. For a balanced set the two
    values are equal. For an unbalanced set the majority-class rate is higher,
    and it is the honest baseline to report.
    """
    table = dict(subsets_for(species))[subset]
    counts = table["bird"].value_counts()

    inverse_n_birds = 1.0 / len(counts)
    majority_class = counts.max() / len(table)

    assert majority_class >= inverse_n_birds - 1e-9, (
        f"{species}/{subset}: the majority-class rate ({majority_class:.3f}) is "
        f"below 1/n_birds ({inverse_n_birds:.3f}). This is impossible."
    )

    if species == "aa":
        assert abs(majority_class - inverse_n_birds) < 0.01, (
            "aa is expected to be balanced at 60 calls for each bird."
        )


@pytest.mark.parametrize("species", CONFIG["species"])
def test_encounter_grouping_is_documented(species):
    """Count the recordings that hold more than one bird.

    The encounter metric pools the calls of one bird within one recording. It
    therefore assumes that the calls are already grouped by individual.

    That assumption is necessary, because some recordings hold two birds. To
    pool every call in such a recording would average two birds into one query.

    This test does not fail on a multi-bird recording. It asserts that the count
    matches the value stated in the README, so that a change in the data cannot
    silently invalidate the documented assumption.
    """
    expected = {"aa": 2, "ag": 15}
    master = load_master(species)
    per_recording = master.groupby("recording_id")["bird"].nunique()
    found = int((per_recording > 1).sum())

    assert found == expected[species], (
        f"{species}: {found} recordings hold more than one bird, but the README "
        f"states {expected[species]}. Update the README and Section 6.4, then "
        f"update this test."
    )
