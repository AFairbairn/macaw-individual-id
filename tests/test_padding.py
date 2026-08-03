#!/usr/bin/env python
"""
Tests for the padding stage.

PURPOSE
    Assert that the padded copy differs from the source in length only. A clip
    below the minimum is written by soundfile and a clip at or above it is
    copied, so the two branches must agree on everything except the added
    silence. Clip length decides the branch, and mean clip length differs
    between the two species, so any other difference would sit inside the
    species comparison.

USAGE
    pytest tests/

WHAT EACH TEST CHECKS
    test_short_clip_reaches_the_minimum
        A clip below the minimum comes out at the minimum.
    test_long_clip_is_unchanged
        A clip at or above the minimum comes out byte for byte identical.
    test_sample_format_survives_the_pad
        The written file keeps the sample rate, the channel count and the
        sample format of the source. soundfile writes 16 bit by default, which
        would requantise every padded clip and leave every copied clip alone.
    test_no_partial_file_is_left_behind
        The temporary file the writer uses does not survive.
    test_the_added_samples_are_silence
        The padding adds no acoustic content.
"""
import filecmp
import importlib.util
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

ROOT = Path(__file__).resolve().parents[1]
MINIMUM_SECONDS = 1.0
RATE = 48000


def load_stage():
    """Import 00_prepare_data.py, whose name is not a valid identifier."""
    path = ROOT / "src/00_prepare_data.py"
    spec = importlib.util.spec_from_file_location("pad_stage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_source(path, seconds, subtype="PCM_24", channels=1):
    """Write one test clip and return its path."""
    frames = int(round(seconds * RATE))
    generator = np.random.default_rng(0)
    audio = generator.standard_normal((frames, channels)).astype("float32") * 0.1
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, audio, RATE, subtype=subtype)
    return path


def pad(module, source, target):
    """Run the padding function with the test minimum."""
    return module.pad_one(source, target, lambda rate: int(round(MINIMUM_SECONDS * rate)))


def test_short_clip_reaches_the_minimum(tmp_path):
    """A clip below the minimum comes out at the minimum."""
    module = load_stage()
    source = write_source(tmp_path / "in/short.wav", 0.3)
    target = tmp_path / "out/short.wav"

    added = pad(module, source, target)

    assert added > 0
    info = soundfile.info(target)
    assert info.frames == int(round(MINIMUM_SECONDS * RATE))


def test_long_clip_is_unchanged(tmp_path):
    """A clip at or above the minimum comes out byte for byte identical."""
    module = load_stage()
    source = write_source(tmp_path / "in/long.wav", 1.5)
    target = tmp_path / "out/long.wav"

    added = pad(module, source, target)

    assert added == 0
    assert filecmp.cmp(source, target, shallow=False), (
        "A clip that needs no padding must be copied without change."
    )


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
@pytest.mark.parametrize("channels", [1, 2])
def test_sample_format_survives_the_pad(tmp_path, subtype, channels):
    """The padded file keeps the sample rate, channels and sample format."""
    module = load_stage()
    source = write_source(tmp_path / f"in/{subtype}_{channels}.wav", 0.3,
                          subtype=subtype, channels=channels)
    target = tmp_path / f"out/{subtype}_{channels}.wav"

    pad(module, source, target)

    before, after = soundfile.info(source), soundfile.info(target)
    assert after.subtype == before.subtype, (
        f"The source is {before.subtype} and the padded copy is {after.subtype}. "
        "soundfile writes 16 bit by default. Pass the subtype of the source."
    )
    assert after.samplerate == before.samplerate
    assert after.channels == before.channels


def test_no_partial_file_is_left_behind(tmp_path):
    """The writer's temporary file does not survive a successful write."""
    module = load_stage()
    source = write_source(tmp_path / "in/short.wav", 0.3)
    target = tmp_path / "out/short.wav"

    pad(module, source, target)

    leftovers = sorted(p.name for p in target.parent.iterdir() if ".part" in p.name)
    assert not leftovers, f"Left behind: {leftovers}"


def test_the_added_samples_are_silence(tmp_path):
    """The padding adds no acoustic content, and the call is not moved."""
    module = load_stage()
    source = write_source(tmp_path / "in/short.wav", 0.3)
    target = tmp_path / "out/short.wav"

    pad(module, source, target)

    original, _ = soundfile.read(source, always_2d=True)
    padded, _ = soundfile.read(target, always_2d=True)

    assert np.allclose(padded[: len(original)], original, atol=1e-6), (
        "The call must stay at the start of the padded clip."
    )
    assert np.all(padded[len(original):] == 0), "The padding must be silence."
