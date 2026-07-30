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
    bacpipe environment. To repair a broken environment, delete it and rebuild
    it from requirements.txt.

NOTE
    bacpipe prints many tracebacks after it writes the embeddings. Those
    messages come from a dashboard step that this pipeline does not use. Do not
    use the absence of tracebacks to confirm success. Run
    src/01b_verify_embeddings.py instead, which counts the output files.
"""
import argparse
import os
from pathlib import Path

# BirdNET ships a Keras 3 checkpoint. Set this before TensorFlow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = Path(os.environ.get("PARROT_DATA", ROOT / "data"))
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
        species code, and bacpipe writes to bacpipe_results/<species>/.

        An earlier version returned the subdirectory below it, which is named
        all_calls for aa and audio for ag. bacpipe then wrote to
        bacpipe_results/all_calls/ and bacpipe_results/audio/, while
        01b_verify_embeddings.py and 03_classify.py read
        bacpipe_results/<species>/. The 13 bacpipe models were extracted and
        then never found. Only the 2 speech models, whose output path this
        file controls, were in the right place.
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


def run_bacpipe_models(audio_dir, device, wanted=None):
    """Compute the embeddings of the bacpipe models.

    When wanted is set, only those models run. Everything else is left alone,
    so a model that failed can be recomputed without touching the rest.
    """
    import bacpipe

    for model in [m for m in BACPIPE_MODELS if wanted is None or m in wanted]:
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
            # That step fails when dim_reduction_model is "None". The embeddings
            # are already on disk at this point. Catching the error here also
            # stops one broken model from ending the whole run.
            print(f"  ({model}: post-embedding step skipped: {type(error).__name__})", flush=True)


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


def run_speech_models(audio_dir, device, wanted=None):
    """Compute the embeddings of the 2 speech models.

    The output path matches the bacpipe layout, so 03_classify.py reads all 15
    models with the same code.
    """
    files = sorted(audio_dir.rglob("*.wav"))

    for model in [m for m in SPEECH_MODELS if wanted is None or m in wanted]:
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
            print(f"  {model}: the model did not load, skipped ({type(error).__name__})", flush=True)
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


def main():
    parser = argparse.ArgumentParser(description="Compute the embeddings of all 15 models.")
    parser.add_argument("--species", required=True, choices=CONFIG["species"])
    parser.add_argument(
        "--device",
        default="",
        help="'cuda' or 'cpu'. The script detects the device when this is not set.",
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
    args = parser.parse_args()

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

    run_bacpipe_models(audio_dir, device, wanted)
    run_speech_models(audio_dir, device, wanted)

    print("Extraction finished. Now run src/01b_verify_embeddings.py.", flush=True)


if __name__ == "__main__":
    main()
