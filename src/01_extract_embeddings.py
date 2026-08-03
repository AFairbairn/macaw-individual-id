#!/usr/bin/env python
"""
01_extract_embeddings.py

PURPOSE
    Compute the embedding of every call, for all 15 pre-trained models.

USAGE
    python src/01_extract_embeddings.py --species aa --device cuda
    python src/01_extract_embeddings.py --species ag --device cpu

    To recompute one model, or a few, without touching the rest:
    python src/01_extract_embeddings.py --species ag --models beats,audiomae

    Delete the output directory of a model before you recompute it. bacpipe
    tries to continue a partly filled directory, and that path is unreliable.

INPUT
    audio_padded/    The padded audio that 00a_pad_audio.py writes.

OUTPUT
    bacpipe_results/<species>/embeddings/<stamp>___<model>-<species>/.../
        One .npy file for each call and each model.

RESUMING
    A model that already wrote one .npy file for every call is skipped. To
    recompute one model, delete its output directory and run the script again.

THE MODELS
    13 models come from bacpipe. 2 speech models come from their own libraries.
    The speech models are included because human speaker recognition is the
    closest analogue to individual identification in animals.

HARDWARE
    A small GPU is sufficient. On the LRZ AI Systems cluster, select the P100.
    Do not queue for the A100. The extra memory gives no benefit here.

    Three groups of models always run on CPU. This is expected behaviour.
      - perch_bird, perch_v2, and surfperch use JAX.
      - birdnet uses TensorFlow.
      - avesecho_passt is pinned to CPU. See the caution below.

CAUTION
    bacpipe reads its device from settings.yaml inside the installed package. It
    does not detect CUDA. The shipped value is 'cpu'. If you do not change that
    value, the torch models run on CPU on a GPU node. The run is then much
    slower and gives no error message. This script rewrites the value
    before each model, so a reinstall of bacpipe cannot strand the run on CPU.

CAUTION
    avesecho_passt moves the model to the device but not the input audio. On
    CUDA, every file fails and the model writes zero embeddings. The model is
    listed in CPU_ONLY_MODELS, so it always runs on CPU.

WARNING
    Do not run `pip install -U jax[cuda12]` to make the JAX models use the GPU.
    The -U flag upgrades numpy to version 2, which breaks the pinned torch and
    bacpipe environment. To repair a broken environment, delete it and build it
    again:
        rm -rf .venv-extraction && ./setup.sh extraction

NOTE
    bacpipe prints many tracebacks after it writes the embeddings. Those
    messages come from a dashboard step that this pipeline does not use. Do not
    use the absence of tracebacks to confirm success. Run
    src/01b_verify_embeddings.py instead, which counts the output files.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

# BirdNET ships a Keras 3 checkpoint. Set this before TensorFlow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import common  # noqa: E402

ROOT = common.ROOT
CONFIG = common.CONFIG
PADDED = ROOT / "audio_padded"

# The 13 models that bacpipe provides.
BACPIPE_MODELS = [
    "birdnet", "perch_bird", "perch_v2", "surfperch", "convnext_birdset",
    "audioprotopnet", "avesecho_passt", "birdmae", "birdaves_especies",
    "aves_especies", "beats", "biolingual", "audiomae",
]

# The 2 speech models. These come from speechbrain and transformers.
SPEECH_MODELS = ["ecapa_tdnn", "wavlm_base_plus_sv"]

# Models that must run on CPU. See the caution in the module docstring.
CPU_ONLY_MODELS = set(CONFIG["cpu_only_models"])

# The working sample rate of the speech models.
SPEECH_SAMPLE_RATE = 16000


def audio_dir_for(species):
    """Return the audio directory of one species.

    The directory is the padded copy that 00a_pad_audio.py writes, not the
    published audio. Two models fail on the shortest clips, so every model
    reads the same padded input. See the docstring of 00a_pad_audio.py.

    THE DIRECTORY NAME IS THE OUTPUT PATH
        bacpipe writes to <main_results_dir>/<audio_dir.stem>/embeddings/. It
        takes the name from the directory it is given and offers no way to set
        the output path directly.

        So this function returns the species directory, whose name is the
        species code, and bacpipe writes to bacpipe_results/<species>/. That
        is the path 01b_verify_embeddings.py and 03_classify.py read.
    """
    root = PADDED / "data" / species
    if root.exists() and any(root.rglob("*.wav")):
        return root
    raise SystemExit(
        f"No padded audio found for {species} under {PADDED}. "
        "Run: python src/00a_pad_audio.py"
    )


# The settings this pipeline requires from bacpipe, written into settings.yaml
# inside the installed package before every model.
#
# device
#     bacpipe does not detect CUDA. The shipped value is 'cpu', so without this
#     the torch models run on CPU on a GPU node and give no error message.
#
# run_pretrained_classifier
#     BirdNET ships a species classifier. bacpipe runs it after it has saved the
#     embedding, writes a species prediction file, and builds an annotation
#     table from it. On a single-time-bin file that code raises
#     "index 1 is out of bounds for axis 0 with size 1", which bacpipe reports
#     as "Error generating embeddings for <file>, skipping file" even though the
#     embedding was already written.
#
#     Every clip here is shorter than the 3 second BirdNET segment, so every
#     clip is a single time bin and the whole species is exposed to it.
#
#     This pipeline uses the embeddings and never the species predictions, so
#     the classifier is turned off. The embedding model is a separate Keras
#     model from the classifier head, so the embeddings do not change.
#
# save_raven_tables
#     Raven tables are built from the classifier output. With no classifier
#     there is nothing to write.
BACPIPE_SETTINGS = {
    "run_pretrained_classifier": "False",
    "save_raven_tables": "False",
}


def set_bacpipe_device(device):
    """Write the settings this pipeline requires into the bacpipe package.

    The function runs before every model, so a reinstall of bacpipe between
    models cannot leave the run on CPU or turn the classifier back on.
    """
    import re
    import bacpipe

    path = Path(bacpipe.__file__).parent / "settings.yaml"
    if not path.is_file():
        return

    text = path.read_text()
    updated = re.sub(r"(?m)^device:.*$", f"device: '{device}'", text)
    for key, value in BACPIPE_SETTINGS.items():
        updated = re.sub(rf"(?m)^{key}:.*$", f"{key}: {value}", updated)
    if updated != text:
        path.write_text(updated)


def clear_model_output(audio_dir, model):
    """Delete the existing output of one model, before it is recomputed.

    bacpipe tries to continue a partly filled output directory. That path
    carries an off-by-one that desynchronises its file list from its per-file
    bin counts, and it fails with a length mismatch. So a model that is asked
    for by name starts from an empty directory.

    Only the named model is touched. Every other model is left alone, which is
    the point of asking for one by name.
    """
    import shutil

    parent = ROOT / "bacpipe_results" / audio_dir.name / "embeddings"
    if not parent.is_dir():
        return

    for directory in sorted(parent.iterdir()):
        if not directory.is_dir() or "___" not in directory.name:
            continue
        if directory.name.split("___")[1].rsplit("-", 1)[0] != model:
            continue
        shutil.rmtree(directory)
        print(f"  removed the previous output: {directory.name}", flush=True)


def run_bacpipe_models(audio_dir, device, wanted=None, clear=False):
    """Compute the embeddings of the bacpipe models.

    When wanted is set, only those models run. Everything else is left alone,
    so a model that failed can be recomputed without touching the rest.
    """
    import bacpipe

    for model in [m for m in BACPIPE_MODELS if wanted is None or m in wanted]:
        if clear:
            clear_model_output(audio_dir, model)
        model_device = "cpu" if model in CPU_ONLY_MODELS else device
        set_bacpipe_device(model_device)
        print(f"=== bacpipe: {model} (device={model_device}) ===", flush=True)

        bacpipe.config.audio_dir = str(audio_dir)
        bacpipe.config.models = [model]
        bacpipe.config.dim_reduction_model = "None"

        # Turn off the dashboard step. This pipeline reads only the .npy files.
        # An older bacpipe can lack these settings, so each one is set
        # separately and a failure is ignored.
        for attribute, value in (("dashboard", False), ("evaluation_task", None)):
            try:
                setattr(bacpipe.config, attribute, value)
            except AttributeError:
                pass

        try:
            bacpipe.play()
        except Exception as error:
            # bacpipe runs an evaluation step after it writes the embeddings.
            # That step fails when dim_reduction_model is "None", and the
            # embeddings are already on disk when it does. So one model is not
            # allowed to end the run.
            #
            # The message and the traceback are both printed. One exception
            # type covers a disk that filled up, a checkpoint that would not
            # load and a harmless evaluation step. The type alone does not say
            # which one happened.
            import traceback

            print(f"  {model}: bacpipe raised {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
            print(
                f"  {model}: continuing. 01b_verify_embeddings.py counts the "
                "files and fails the run if this model wrote nothing.",
                flush=True,
            )


def load_audio_16k(path, device):
    """Load one audio file, convert it to mono, and resample it to 16 kHz.

    The function pads a clip that is shorter than one second.

    WavLMForXVector needs a minimum number of frames. Without the pad, it fails
    on short trimmed calls. ECAPA-TDNN accepts short clips, and trailing silence
    does not change its output.
    """
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    if sample_rate != SPEECH_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sample_rate, SPEECH_SAMPLE_RATE)(waveform)

    if waveform.shape[-1] < SPEECH_SAMPLE_RATE:
        pad = SPEECH_SAMPLE_RATE - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    return waveform.to(device)


def build_speech_encoder(model, device):
    """Return a function that turns one waveform into one embedding vector."""
    if model == "ecapa_tdnn":
        from speechbrain.inference.speaker import EncoderClassifier

        network = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=".cache/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        network.eval()
        return lambda w: network.encode_batch(w).squeeze().detach().cpu().numpy()

    from transformers import AutoFeatureExtractor, WavLMForXVector

    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    network = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
    network.eval()

    def encode(waveform):
        inputs = extractor(
            waveform.squeeze().cpu().numpy(),
            sampling_rate=SPEECH_SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return network(**inputs).embeddings.squeeze().detach().cpu().numpy()

    return encode


def run_speech_models(audio_dir, device, wanted=None, clear=False):
    """Compute the embeddings of the 2 speech models.

    The output path matches the bacpipe layout, so 03_classify.py reads all 15
    models with the same code.
    """
    files = sorted(audio_dir.rglob("*.wav"))

    for model in [m for m in SPEECH_MODELS if wanted is None or m in wanted]:
        if clear:
            clear_model_output(audio_dir, model)
        print(f"=== speech: {model} ({len(files)} files, device={device}) ===", flush=True)

        out_dir = (
            ROOT / "bacpipe_results" / audio_dir.name / "embeddings"
            / f"speech___{model}-{audio_dir.name}"
        )

        if out_dir.is_dir() and len(list(out_dir.rglob("*.npy"))) >= len(files):
            print(f"  {model}: already complete, skipped", flush=True)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            encode = build_speech_encoder(model, device)
        except Exception as error:
            print(f"  {model}: the model did not load, skipped. "
                  f"{type(error).__name__}: {error}", flush=True)
            continue

        for index, path in enumerate(files, start=1):
            try:
                vector = np.asarray(encode(load_audio_16k(path, device))).reshape(-1)
                destination = out_dir / path.relative_to(audio_dir).parent / f"{path.stem}_{model}.npy"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, vector)
            except Exception as error:
                print(f"  {model} failed on {path.name}: {type(error).__name__}", flush=True)

            if index % 100 == 0:
                print(f"  {index}/{len(files)}", flush=True)


# =============================================================================
# The checkpoint check, before extraction
#
# bacpipe decides whether to download a checkpoint by one test: does the
# directory of that model exist and hold at least one file. A download that
# stopped part of the way through leaves a short file, so the test passes and
# bacpipe never downloads again. The model then fails on every run, with an
# error that names the file format rather than the cause.
#
# That happened here. A download stopped when the disk quota ran out and left
# 113,246,208 bytes of a 363,145,291 byte BEATs checkpoint, and the run of
# 2026-07-31 13:23 died on it.
#
# NOTHING IS DELETED UNLESS IT IS ASKED FOR
#     An earlier version deleted the directory of any model whose checkpoint
#     would not open. That is the wrong default. A model directory holds
#     hundreds of megabytes, and deleting it commits the user to downloading it
#     again on a machine that may have no network, or the same full disk that
#     truncated the file in the first place. A download that failed once fails
#     again for the same reason.
#
#     So this reports, and stops the run. --repair-checkpoints deletes the
#     broken copies, and the user decides when to use it.
# =============================================================================

# The default in the bacpipe settings file, used when that file cannot be read.
DEFAULT_CHECKPOINT_BASE = "bacpipe_model_checkpoints"

# The two speech models come from their own libraries and cache elsewhere, and
# the MFCC variants have no checkpoint at all.
NOT_FROM_BACPIPE = set(SPEECH_MODELS) | {"mfcc_lakdari", "mfcc_full", "mfcc_cmvn"}


def panel_models():
    """Return the models in config.yaml whose checkpoint bacpipe downloads."""
    return [name for name in CONFIG["models"] if name not in NOT_FROM_BACPIPE]


def checkpoint_root():
    """Return the directory bacpipe reads its checkpoints from.

    bacpipe reads model_base_path from settings.yaml inside its own installed
    package. This reads the same setting, so it checks the directory bacpipe
    reads, whatever that value is. The resolved path is printed, because a check
    of the wrong directory reports success and means nothing.
    """
    try:
        import bacpipe
    except ImportError:
        return ROOT / DEFAULT_CHECKPOINT_BASE

    settings = Path(bacpipe.__file__).parent / "settings.yaml"
    base = DEFAULT_CHECKPOINT_BASE
    if settings.is_file():
        loaded = yaml.safe_load(settings.read_text()) or {}
        base = loaded.get("model_base_path", DEFAULT_CHECKPOINT_BASE)

    path = Path(base)
    return path if path.is_absolute() else ROOT / path


def sha256_of(path):
    """Return the sha256 checksum of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_problem(path):
    """Open one file with the reader its format needs.

    Return an empty string when the file opens, the word 'unchecked' when this
    does not know the format, or a message when the file will not open.
    """
    suffix = path.suffix.lower()

    if suffix in (".pt", ".pth", ".ckpt"):
        try:
            import torch
        except ImportError:
            return "unchecked"
        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return ""

    if suffix in (".keras", ".zip"):
        return "" if zipfile.is_zipfile(path) else "not a valid zip archive"

    if suffix in (".xz", ".gz", ".bz2", ".tar"):
        try:
            with tarfile.open(path) as archive:
                archive.getmembers()
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return ""

    return "unchecked"


def recorded_checksums():
    """Return the sha256 recorded in config.yaml, keyed by relative path."""
    recorded = {}
    for entry in (CONFIG.get("checkpoints") or {}).values():
        recorded.update(entry.get("sha256") or {})
    return recorded


def check_one_checkpoint(root, model, checksums):
    """Check one model. Return (state, [messages]).

    The state is 'ok', 'absent', 'broken' or 'unchecked'.
    """
    directory = root / model
    if not directory.is_dir():
        return "absent", ["no directory. bacpipe downloads it when this stage runs."]

    files = [p for p in sorted(directory.rglob("*"))
             if p.is_file() and ".cache" not in p.parts]
    if not files:
        return "absent", ["the directory is empty."]

    messages, opened, unchecked = [], 0, 0
    for path in files:
        relative = path.relative_to(root).as_posix()

        expected = checksums.get(relative)
        if expected and sha256_of(path) != expected:
            messages.append(f"{relative}: the checksum does not match config.yaml.")
            continue

        problem = open_problem(path)
        if problem == "unchecked":
            unchecked += 1
        elif problem:
            messages.append(f"{relative}: {problem} ({path.stat().st_size} bytes)")
        else:
            opened += 1

    if messages:
        return "broken", messages
    if opened == 0:
        return "unchecked", [f"{unchecked} file(s), none of a format this reads."]
    return "ok", [f"{opened} file(s) opened" + (f", {unchecked} unchecked" if unchecked else "")]


def check_checkpoints(repair=False):
    """Check every checkpoint. Return True when the run may continue."""
    root = checkpoint_root()
    print(f"Checkpoint directory: {root}")
    if not root.is_dir():
        print("  No checkpoint has been downloaded yet.")
        print("  bacpipe downloads what it needs when it runs.")
        return True

    checksums = recorded_checksums()
    results = {model: check_one_checkpoint(root, model, checksums)
               for model in panel_models()}

    for model, (state, messages) in results.items():
        label = {"ok": "ok       ", "absent": "absent   ",
                 "broken": "BROKEN   ", "unchecked": "unchecked"}[state]
        print(f"  {label} {model:<20} {messages[0]}")
        for line in messages[1:]:
            print(f"           {' ' * 20} {line}")

    broken = [m for m, (state, _) in results.items() if state == "broken"]
    if not broken:
        print(f"  {len(results)} model(s) checked. Every checkpoint present opens.")
        return True

    print()
    print(f"{len(broken)} checkpoint(s) will not open: {', '.join(broken)}")
    print()

    if not repair:
        print("This stage would fail on those models and write no embedding for them.")
        print("Nothing has been deleted. Choose one:")
        print()
        print("  1. Free the disk space or restore the network, then delete the")
        print("     broken copies so bacpipe fetches them again:")
        print("       python src/01_extract_embeddings.py --species aa --repair-checkpoints")
        print()
        print("  2. Copy a working checkpoint into the directory by hand.")
        print()
        print("A download that failed once fails again for the same reason. On")
        print("2026-07-31 the disk quota ran out part of the way through the BEATs")
        print("checkpoint. Check the free space before you repair.")
        return False

    for model in broken:
        # bacpipe decides whether to download by whether the directory exists
        # and holds a file, so one file left behind stops the download.
        target = root / model
        print(f"  removing {target}")
        shutil.rmtree(target, ignore_errors=True)

    print()
    print(f"{len(broken)} model directory removed. bacpipe downloads them when")
    print("this stage runs. Run the pipeline again.")
    return False


# =============================================================================
# The count check, after extraction
#
# bacpipe prints many tracebacks after it writes the embeddings. Those come from
# a dashboard step this pipeline does not use, and the embeddings are already on
# disk when they appear. A traceback therefore does not mean the extraction
# failed, and a clean log does not mean it succeeded. One model can write zero
# files while the log looks normal, which is what avesecho_passt does on CUDA.
#
# Count the files. Do not read the log.
# =============================================================================

EXPECTED_MODEL_COUNT = 15


def expected_stems(species):
    """Return the clip stems that the master table lists for one species."""
    return set(common.read_master(species)["original_stem"])


def found_stems(directory, model):
    """Return the clip stems that one model wrote."""
    suffix = f"_{model}.npy"
    return {path.name[: -len(suffix)] for path in directory.rglob(f"*{suffix}")}


def check_species_counts(species):
    """Report every model of one species. Return True when all are complete."""
    root = ROOT / f"bacpipe_results/{species}/embeddings"
    print(f"\n=== {species} ===")

    if not root.exists():
        print(f"  {root} not found.")
        return False

    expected = expected_stems(species)
    print(f"  The master table lists {len(expected)} clips.")

    directories = sorted(d for d in root.iterdir() if d.is_dir() and "___" in d.name)
    if not directories:
        print("  No model directory found.")
        return False

    all_complete = True
    for directory in directories:
        model = directory.name.split("___")[1].rsplit("-", 1)[0]
        found = found_stems(directory, model)
        missing = expected - found

        if not found:
            # This is the avesecho_passt on CUDA case.
            print(f"  {model:22} FAILED. The model wrote no file.")
            all_complete = False
        elif missing:
            print(f"  {model:22} INCOMPLETE. {len(found)}/{len(expected)} clips.")
            for stem in sorted(missing)[:3]:
                print(f"  {'':22}   missing: {stem}")
            if len(missing) > 3:
                print(f"  {'':22}   and {len(missing) - 3} more")
            all_complete = False
        else:
            print(f"  {model:22} complete. {len(found)} clips.")

    if len(directories) < EXPECTED_MODEL_COUNT:
        print(f"  Only {len(directories)} of {EXPECTED_MODEL_COUNT} models are present.")
        all_complete = False

    return all_complete


def verify_counts(species_list):
    """Count the output of every model. Return True when every one is complete."""
    print("\nCounting the extracted embeddings.")
    print("A traceback above does not mean the extraction failed. This counts files.")
    results = [check_species_counts(species) for species in species_list]

    print()
    if all(results):
        print("Every model is complete.")
        return True
    print("One or more models are incomplete. See the report above.")
    print("To recompute one model, run this stage with --models <name>.")
    return False


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute the embeddings of all 15 models.")
    parser.add_argument("--species", required=True, choices=CONFIG["species"])
    parser.add_argument(
        "--device",
        default="",
        help="'cuda' or 'cpu'. The script detects the device when this is not set.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help=(
            "Delete the existing output of every model this run computes, "
            "before it computes it. run_all.sh passes this on the full path. "
            "bacpipe tries to continue a partly filled directory, and that "
            "path is unreliable."
        ),
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "A comma separated list of model names. Only these models run. "
            "Use this to recompute one model without touching the others. "
            "The default is every model."
        ),
    )
    parser.add_argument(
        "--repair-checkpoints",
        action="store_true",
        help="Delete the directory of every model whose checkpoint will not open.",
    )
    args = parser.parse_args()

    # Check the checkpoints before anything is extracted. A model whose
    # checkpoint will not open writes no embedding, and bacpipe reports that as
    # a traceback that looks like every other traceback it prints.
    if not check_checkpoints(repair=args.repair_checkpoints):
        return 1

    wanted = None
    if args.models:
        wanted = [name.strip() for name in args.models.split(",") if name.strip()]
        known = set(BACPIPE_MODELS) | set(SPEECH_MODELS)
        unknown = [name for name in wanted if name not in known]
        if unknown:
            raise SystemExit(
                f"Unknown model name(s): {', '.join(unknown)}. "
                f"Known models: {', '.join(sorted(known))}"
            )

    device = args.device
    if not device:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    audio_dir = audio_dir_for(args.species)
    print(f"Species: {args.species}", flush=True)
    print(f"Audio:   {audio_dir}", flush=True)
    print(f"Device:  {device}", flush=True)

    if wanted:
        print(f"Models:  {', '.join(wanted)}", flush=True)

    # A model asked for by name is always recomputed from nothing. Otherwise
    # --clear decides.
    clear = args.clear or wanted is not None

    run_bacpipe_models(audio_dir, device, wanted, clear)
    run_speech_models(audio_dir, device, wanted, clear)

    # The embeddings carry the environment that produced them. Stage 3 and later
    # read this and stop if the feature sets disagree, which is the fault that
    # went unnoticed on 2026-08-01.
    common.write_provenance(ROOT / f"bacpipe_results/{args.species}/embeddings")

    # Count the files. See the comment above check_species_counts.
    if not verify_counts([args.species]):
        return 1

    print("Extraction finished, and every model wrote a file for every clip.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
