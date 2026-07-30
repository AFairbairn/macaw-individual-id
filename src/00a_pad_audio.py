#!/usr/bin/env python
"""
00a_pad_audio.py

PURPOSE
    Write a padded copy of every clip. A clip shorter than the minimum length
    gets trailing silence. A clip at or above the minimum is copied unchanged.
    Stage 1 and stage 2 read this copy, so every representation sees the same
    audio.

USAGE
    python src/00a_pad_audio.py

INPUT
    config.yaml                                     The padding block.
    data/<species>/metadata/<species>_master.csv    The clip list.
    The audio named in rel_audio_path, below PARROT_DATA.

OUTPUT
    audio_padded/<rel_audio_path>
        One file for each row of each master table, below the repository root.

WHY THIS STAGE EXISTS
    The calls are short. Every Ara ambiguus clip is between 0.192 and 1.093
    seconds. Two models fail on clips at the low end of that range.

    BirdNET raises "index 1 is out of bounds for axis 0 with size 1" and the
    clip is skipped, so the model writes no embedding for it. The run continues
    and the log gives no total, so the loss is only visible when
    01b_verify_embeddings.py counts the files.

    WavLMForXVector fails the same way. Its convolution and TDNN stack needs a
    minimum number of frames. An earlier version of this project lost about 75
    WavLM embeddings to this before the input was padded.

    The fix is the same for both. Give every model an input at or above the
    minimum length.

WHY BOTH SPECIES GET THE SAME TREATMENT
    The main result of the paper compares the two species. To pad one and not
    the other would put a preprocessing difference inside that comparison. The
    rule is applied to every clip of both species, and a clip that is already
    long enough is copied without change, so the rule costs nothing where it is
    not needed.

WHAT THE PADDING DOES TO THE SIGNAL
    The padding is silence at the end of the clip. It adds no acoustic content.
    It moves the call to the start of a longer window, which is where a model
    that pads internally would put it anyway. Report the minimum length in the
    methods, because it is a preprocessing choice and not a property of the
    recordings.

RERUNNING
    A file that is already present in audio_padded is left alone. To rebuild
    the whole tree, delete the directory and run the script again.
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = Path(os.environ.get("PARROT_DATA", ROOT / "data"))
PADDED = ROOT / "audio_padded"


def source_path(relative):
    """Return the source file for one rel_audio_path value.

    rel_audio_path starts with "data/", so it resolves either from the parent
    of the data directory or from the data directory itself. Both roots are
    tried, so the script works whether PARROT_DATA points at the dataset or at
    its parent.
    """
    for candidate in (DATA.parent / relative, DATA / relative):
        if candidate.exists():
            return candidate
    return None


def pad_one(source, target, minimum_samples):
    """Write one padded file. Return the number of samples that were added."""
    audio, rate = sf.read(source, always_2d=True)
    needed = minimum_samples(rate) - len(audio)
    target.parent.mkdir(parents=True, exist_ok=True)

    if needed <= 0:
        shutil.copyfile(source, target)
        return 0

    silence = np.zeros((needed, audio.shape[1]), dtype=audio.dtype)
    sf.write(target, np.concatenate([audio, silence]), rate)
    return needed


def main():
    """Pad every clip of every species. Return an exit code."""
    block = CONFIG.get("padding")
    if not block:
        print("config.yaml has no padding block. Nothing to do.")
        return 0

    minimum_seconds = float(block["min_seconds"])
    print(f"Minimum clip length: {minimum_seconds} s")
    print(f"Writing to {PADDED}")
    print()

    failed = False
    for species in CONFIG["species"]:
        master = ROOT / f"data/{species}/metadata/{species}_master.csv"
        if not master.exists():
            print(f"[{species}] {master} not found. Run 00_build_master_metadata.py first.")
            return 1

        table = pd.read_csv(master, dtype=str)
        padded = copied = skipped = missing = 0

        for relative in table["rel_audio_path"]:
            target = PADDED / relative
            if target.exists():
                skipped += 1
                continue

            source = source_path(relative)
            if source is None:
                missing += 1
                if missing <= 5:
                    print(f"[{species}] not found: {relative}")
                continue

            added = pad_one(source, target, lambda rate: int(round(minimum_seconds * rate)))
            if added:
                padded += 1
            else:
                copied += 1

        print(f"[{species}] {len(table)} clips: "
              f"{padded} padded, {copied} copied, {skipped} already present")
        if missing:
            print(f"[{species}] FAILED: {missing} source file(s) not found under {DATA}")
            failed = True

    if failed:
        print()
        print("Check that the audio is complete and that PARROT_DATA points at it.")
        return 1

    print()
    print("Every clip is present in audio_padded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
