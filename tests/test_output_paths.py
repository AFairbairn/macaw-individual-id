#!/usr/bin/env python
"""
Tests for the embedding output paths.

PURPOSE
    Assert that every model writes where the later stages read. bacpipe takes
    its output path from the name of the audio directory it is given, so a
    change to that directory silently moves the output of 13 of the 15 models.
    That happened once. The 13 bacpipe models were extracted and then never
    found, and stage 1 reported success.

USAGE
    pytest tests/

WHAT EACH TEST CHECKS
    test_audio_dir_is_named_after_the_species
        The directory handed to bacpipe is named 'aa' or 'ag'. bacpipe writes
        to <main_results_dir>/<audio_dir.stem>/, so this name is the output
        path.
    test_audio_dir_holds_every_clip
        The directory is high enough in the tree to hold every clip of the
        species. Naming it correctly is worthless if it holds half the audio.
    test_readers_and_writer_agree
        03_score_frozen.py reads the directory that 01_extract_embeddings.py
        writes, and stage 1 counts its own output in the same place.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
SPECIES = CONFIG["species"]


def load_stage_one():
    """Import 01_extract_embeddings.py, whose name is not a valid identifier."""
    path = ROOT / "src/01_extract_embeddings.py"
    spec = importlib.util.spec_from_file_location("stage_one", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_padded_tree(root, species):
    """Write one empty file for every clip of one species, at its real path."""
    master = ROOT / f"data/{species}/metadata/{species}_master.csv"
    if not master.exists():
        pytest.skip(f"{master} not found.")
    table = pd.read_csv(master, dtype=str)
    for relative in table["rel_audio_path"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    return len(table)


@pytest.mark.parametrize("species", SPECIES)
def test_audio_dir_is_named_after_the_species(tmp_path, monkeypatch, species):
    """bacpipe names its output directory after the directory it is given."""
    stage_one = load_stage_one()
    build_padded_tree(tmp_path, species)
    monkeypatch.setattr(stage_one, "PADDED", tmp_path)

    audio_dir = stage_one.audio_dir_for(species)

    assert audio_dir.name == species, (
        f"bacpipe writes to bacpipe_results/{audio_dir.name}/, but the count "
        f"check in stage 1 and 03_score_frozen.py read "
        f"bacpipe_results/{species}/."
    )


@pytest.mark.parametrize("species", SPECIES)
def test_audio_dir_holds_every_clip(tmp_path, monkeypatch, species):
    """The directory handed to bacpipe holds every clip of the species."""
    stage_one = load_stage_one()
    expected = build_padded_tree(tmp_path, species)
    monkeypatch.setattr(stage_one, "PADDED", tmp_path)

    audio_dir = stage_one.audio_dir_for(species)
    found = len(list(audio_dir.rglob("*.wav")))

    assert found == expected, (
        f"{audio_dir} holds {found} clips. The master table lists {expected}."
    )


@pytest.mark.parametrize("species", SPECIES)
def test_readers_and_writer_agree(species):
    """The stage that writes and the stages that read use the same path."""
    written = f"bacpipe_results/{species}/embeddings"
    for name in ("src/01_extract_embeddings.py", "src/03_score_frozen.py",
                 "src/04_metric_learning.py"):
        text = (ROOT / name).read_text()
        assert 'bacpipe_results/{species}/embeddings' in text, (
            f"{name} no longer reads {written}. Check it against "
            "audio_dir_for in src/01_extract_embeddings.py."
        )
