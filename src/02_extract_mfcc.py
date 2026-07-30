#!/usr/bin/env python
"""
02_extract_mfcc.py

PURPOSE
    Compute MFCC features for every clip and write them in the bacpipe layout.
    Stage 3 then scores them with the same code that scores the 15 pre-trained
    models. No change to stage 3 is needed.

WHY THIS STAGE EXISTS
    The benchmark needs a classical floor. Without one, a reader cannot tell how
    much the pre-trained models add.

    Lakdari et al. (2024, Ecological Informatics 80:102457) compared MFCCs
    against pre-trained embeddings for individual gibbon identification. MFCCs
    matched the embeddings at close range, and beat them for unsupervised
    clustering. That paper appears in the target journal of this manuscript, so
    the comparison is expected.

USAGE
    python src/02_extract_mfcc.py

INPUT
    config.yaml                              The MFCC settings.
    data/<species>/metadata/<species>_master.csv   The clip list.
    audio_padded/    The padded audio that 00a_pad_audio.py writes.

OUTPUT
    mfcc_results/<species>/embeddings/<stamp>___<variant>-<species>/single/
        One .npy file for each clip. The file holds one row.

THE THREE VARIANTS
    mfcc_lakdari
        The formula of Lakdari et al. (2024). 12 coefficients. The mean and the
        standard deviation of each coefficient across frames. 24 values.
    mfcc_full
        20 coefficients and their deltas. The mean and the standard deviation of
        each. 80 values.
    mfcc_cmvn
        mfcc_full, with cepstral mean and variance normalisation applied for
        each recording.

CAUTION
    Do not report mfcc_cmvn as a test of channel effects. CMVN removes the mean
    and the variance of each recording. In this data, most recordings hold one
    bird, so CMVN removes the bird as well as the channel. The variant shows
    only that the MFCC identity signal sits in the static spectral statistics of
    a recording. To compare channel effects across feature types, use
    src/05_diagnostics.py.

DEPARTURES FROM THE GIBBON PAPER
    1. The band is 500 to 8000 Hz. Their band is 400 to 1600 Hz. Our band comes
       from our own audio. Call energy sits between 1.3 and 5 kHz in both
       species, measured as the 5th to 95th percentile of spectral energy.
    2. The sample rate is 16 kHz. The Nyquist frequency is then 8 kHz, which
       matches the top of the band.
    3. Their gibbon calls last 9 to 27 seconds. Our calls last 0.19 to 1.28
       seconds. One call gives 17 to 125 frames. The standard deviation of each
       coefficient is therefore noisier than in their study. That noise is a
       limit of the MFCC baseline on calls this short.
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PADDED = ROOT / "audio_padded"
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
MFCC = CONFIG["mfcc"]

# A fixed stamp keeps the output directory name stable across runs. A changing
# name would break the manifest comparison in 07_manifest.py.
STAMP = "mfcc"


def frame_features(path, n_mfcc, deltas):
    """Return the frame-level MFCC matrix for one audio file.

    The returned array has the shape (n_coefficients, n_frames).

    If the clip is shorter than two analysis windows, the function pads it with
    zeros. Without the pad, librosa returns too few frames to summarise.
    """
    audio, _ = librosa.load(path, sr=MFCC["sample_rate"], mono=True)

    window_samples = int(MFCC["sample_rate"] * MFCC["window_seconds"])
    if audio.size < window_samples * 2:
        audio = np.pad(audio, (0, window_samples * 2 - audio.size))

    features = librosa.feature.mfcc(
        y=audio,
        sr=MFCC["sample_rate"],
        n_mfcc=n_mfcc,
        n_mels=MFCC["n_mels"],
        n_fft=window_samples,
        hop_length=int(MFCC["sample_rate"] * MFCC["hop_seconds"]),
        fmin=MFCC["fmin_hz"],
        fmax=MFCC["fmax_hz"],
    )

    if deltas:
        # The delta width must be odd and must not exceed the frame count.
        width = min(9, max(3, (features.shape[1] // 2) * 2 - 1))
        features = np.vstack([features, librosa.feature.delta(features, width=width)])

    return features


def summarise(features):
    """Reduce a frame-level matrix to one vector for the clip.

    The summary is the mean and the standard deviation of each coefficient
    across frames. This is the summary that Lakdari et al. (2024) use.
    """
    return np.concatenate([features.mean(axis=1), features.std(axis=1)]).astype(np.float32)


def apply_cmvn(per_clip_features, clip_to_recording):
    """Normalise the features of each recording to zero mean and unit variance.

    The function computes one mean and one standard deviation for each
    recording, using every frame of every clip in that recording. It then
    applies those values to each clip of the recording.

    CAUTION: In this data, most recordings hold one bird. This normalisation
    therefore removes bird-level offsets as well as channel-level offsets. See
    the caution in the module docstring.
    """
    by_recording = {}
    for clip, recording in clip_to_recording.items():
        by_recording.setdefault(recording, []).append(clip)

    normalised = {}
    for recording, clips in by_recording.items():
        pooled = np.hstack([per_clip_features[c] for c in clips])
        mean = pooled.mean(axis=1, keepdims=True)
        sd = pooled.std(axis=1, keepdims=True) + 1e-8
        for clip in clips:
            normalised[clip] = (per_clip_features[clip] - mean) / sd
    return normalised


def clips_for(species):
    """Return the list of (stem, recording_id, audio_path) for one species.

    The function reads the master table and keeps only single calls. It skips a
    clip when the audio file is missing, and it reports the count at the end.
    """
    master_path = ROOT / f"data/{species}/metadata/{species}_master.csv"
    master = pd.read_csv(master_path, dtype=str)
    master = master[master["kind"] == "single"]

    rows, missing = [], 0
    for _, record in master.iterrows():
        audio_path = PADDED / record["rel_audio_path"]
        if not audio_path.exists():
            missing += 1
            continue
        rows.append((record["original_stem"], record["recording_id"], audio_path))

    # A missing file means this baseline is scored on fewer clips than the
    # pre-trained models, which would make the comparison wrong. Stop instead.
    if missing:
        raise SystemExit(
            f"{missing} of {len(master)} clip(s) are absent from {PADDED}. "
            "Run: python src/00a_pad_audio.py"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compute the MFCC baseline features.")
    parser.add_argument(
        "--species", default="", help="Process one species only. Default: every species."
    )
    args = parser.parse_args()

    species_list = [args.species] if args.species else CONFIG["species"]

    for species in species_list:
        rows = clips_for(species)
        print(f"{species}: {len(rows)} clips")

        clip_to_recording = {stem: recording for stem, recording, _ in rows}

        for variant, settings in MFCC["variants"].items():
            # Compute the frame-level features once for this variant.
            frames = {
                stem: frame_features(path, settings["n_mfcc"], settings["deltas"])
                for stem, _, path in rows
            }

            if settings["cmvn"]:
                frames = apply_cmvn(frames, clip_to_recording)

            out_dir = (
                ROOT
                / "mfcc_results"
                / species
                / "embeddings"
                / f"{STAMP}___{variant}-{species}"
                / "single"
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            for stem, _, _ in rows:
                vector = summarise(frames[stem])
                # Save one row, so stage 3 reads the clip as a single window.
                np.save(out_dir / f"{stem}_{variant}.npy", vector[None, :])

            dimension = summarise(next(iter(frames.values()))).shape[0]
            print(f"  {variant}: {len(rows)} files, {dimension} values for each clip")


if __name__ == "__main__":
    main()
