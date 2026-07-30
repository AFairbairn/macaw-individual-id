#!/usr/bin/env bash
# =============================================================================
# run_all.sh
#
# PURPOSE
#   Produce every result table in the macaw individual-identification paper.
#
#   This pipeline produces DATA, not figures. Every output is a CSV file.
#   Figures are made separately from those CSV files.
#
# USAGE
#   ./run_all.sh [STAGE]
#
# STAGE
#   all       Run every stage. Extracts embeddings first. Needs a GPU.
#   embed     Run stage 1 only. Extracts embeddings. Needs a GPU.
#   analyse   Run stages 2 to 5. Uses existing embeddings. Does not need a GPU.
#   auto      Choose the stage automatically. This is the default.
#
#   In 'auto' mode, the script runs 'analyse' if embeddings exist. If no
#   embeddings exist, the script runs 'all'.
#
# ENVIRONMENT VARIABLES
#   FORCE_DEVICE   Set to 'cpu' or 'cuda' to override the detected device.
#
# EXAMPLES
#   ./run_all.sh analyse                 Reproduce all numbers on a laptop.
#   ./run_all.sh embed                   Extract embeddings on a GPU node.
#   FORCE_DEVICE=cpu ./run_all.sh all    Run everything on CPU.
#
# EXIT CODES
#   0   Success.
#   1   A stage failed, or the environment is incomplete.
#
# NOTES
#   Every stage is idempotent. To rerun a stage, delete its output and run the
#   script again.
#   Stage 1 is the only stage that uses a GPU. Stages 2 to 5 are CPU bound,
#   because they run scikit-learn on fewer than 1,100 vectors.
# =============================================================================
set -euo pipefail

STAGE="${1:-auto}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

mkdir -p logs results
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

# -----------------------------------------------------------------------------
# Helper functions.
# -----------------------------------------------------------------------------

# say: print a stage banner with a timestamp.
say() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }

# die: print an error message and stop the script.
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Stage timing.
#
# begin_stage prints the banner and starts the clock. end_stage prints how long
# the stage took and adds it to TIMINGS, which the summary at the end prints as
# a table. Use those durations to plan the next run.
# -----------------------------------------------------------------------------
STAGE_START=0
STAGE_NAME=""
TIMINGS=()

# hms: turn a number of seconds into hh:mm:ss.
hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

begin_stage() { STAGE_NAME="$1"; STAGE_START=$SECONDS; say "$1"; }

end_stage() {
  local seconds=$((SECONDS - STAGE_START))
  TIMINGS+=("$(printf '%-46s %s' "$STAGE_NAME" "$(hms "$seconds")")")
  printf '\033[1m         %s took %s\033[0m\n' "$STAGE_NAME" "$(hms "$seconds")"
}

# -----------------------------------------------------------------------------
# detect_device
#
# Return 'cuda' if a CUDA device is available. If no CUDA device is available,
# return 'cpu'. If FORCE_DEVICE is set, return its value unchanged.
# -----------------------------------------------------------------------------
detect_device() {
  if [[ -n "${FORCE_DEVICE:-}" ]]; then
    echo "${FORCE_DEVICE}"
    return
  fi
  if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "cuda"
  else
    echo "cpu"
  fi
}

# -----------------------------------------------------------------------------
# set_bacpipe_device DEVICE
#
# Write the settings this pipeline requires into the settings.yaml file of the
# installed bacpipe package.
#
# CAUTION: bacpipe reads its device from this file. It does not detect CUDA.
# The shipped value is 'cpu'. If you do not change this value, the torch models
# run on CPU on a GPU node. The run is then much slower and gives no error
# message.
#
# The function also turns off the pre-trained species classifier. See
# BACPIPE_SETTINGS in src/01_extract_embeddings.py for why.
#
# The function keeps a backup of the original file.
# -----------------------------------------------------------------------------
set_bacpipe_device() {
  local dev="$1"
  local yml

  yml="$(python - <<'PY' 2>/dev/null || true
import importlib.util, pathlib
spec = importlib.util.find_spec("bacpipe")
print(pathlib.Path(spec.origin).parent / "settings.yaml" if spec and spec.origin else "")
PY
)"

  if [[ -z "$yml" || ! -f "$yml" ]]; then
    echo "  Note: bacpipe settings.yaml not found."
    echo "  01_extract_embeddings.py sets the device for each model instead."
    return
  fi

  cp "$yml" "${yml}.bak_$(date +%Y%m%d)" 2>/dev/null || true

  python - "$yml" "$dev" <<'PY'
import re, sys
path, device = sys.argv[1], sys.argv[2]
text = open(path).read()
new = re.sub(r"^\s*device\s*:.*$", f"device: '{device}'", text, flags=re.M)
for key in ("run_pretrained_classifier", "save_raven_tables"):
    new = re.sub(rf"^\s*{key}\s*:.*$", f"{key}: False", new, flags=re.M)
if new == text:
    new = text.rstrip() + f"\ndevice: '{device}'\n"
open(path, "w").write(new)
print(f"  bacpipe settings.yaml: device = '{device}'")
PY
}

# -----------------------------------------------------------------------------
# Step 1. Select the device and check the environment.
# -----------------------------------------------------------------------------
DEVICE="$(detect_device)"

if [[ -n "${FORCE_DEVICE:-}" ]]; then
  say "Device: ${DEVICE} (forced by FORCE_DEVICE)"
else
  say "Device: ${DEVICE} (detected)"
fi

if [[ "$DEVICE" == "cpu" ]]; then
  cat <<'EOF'
  Stages 2 to 5 are CPU bound. They run normally on CPU.
  Stage 1 on CPU is very slow.
  To reproduce the analysis only, run: ./run_all.sh analyse
EOF
fi

# The pipeline needs Python 3.11 or newer, because bacpipe publishes no build
# for 3.10. Check this before the imports, so the message names the real cause.
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" \
  || die "Python 3.11 or newer is required. This is $(python -V 2>&1). See README section 1.1."

python -c "import numpy, pandas, sklearn, torch, librosa, yaml" \
  || die "The environment is incomplete. Create it with: pip install -r requirements.txt"

# -----------------------------------------------------------------------------
# Step 1b. Check the software and the model weights against the record.
#
# A pinned version number is not enough. A model weight file can change upstream
# while its version number stays the same. This check compares an md5 checksum
# of every weight file against environment.lock/model_weights.csv.
#
# The check reports a difference and continues. To stop the run on a difference,
# set STRICT_ENV=1.
# -----------------------------------------------------------------------------
if [[ -d environment.lock ]]; then
  if [[ "${STRICT_ENV:-0}" == "1" ]]; then
    python src/08_freeze_environment.py verify --strict \
      || die "The environment does not match environment.lock. See the output above."
  else
    python src/08_freeze_environment.py verify || true
  fi
else
  echo "  Note: no environment.lock found."
  echo "  To record this environment, run: python src/08_freeze_environment.py freeze"
fi

# -----------------------------------------------------------------------------
# Step 1c. Locate the dataset.
#
# The audio is published separately from the code. Set PARROT_DATA to the
# dataset directory. If PARROT_DATA is not set, the script uses ./data.
# -----------------------------------------------------------------------------
export PARROT_DATA="${PARROT_DATA:-${HERE}/data}"
if [[ ! -d "$PARROT_DATA" ]]; then
  die "Dataset not found at ${PARROT_DATA}. Download it and set PARROT_DATA."
fi
say "Dataset: ${PARROT_DATA}"

# -----------------------------------------------------------------------------
# Step 2. Choose the stage.
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "auto" ]]; then
  if compgen -G "bacpipe_results/*/embeddings/*___*" > /dev/null; then
    STAGE="analyse"
    say "Embeddings found. Stage = analyse. To force re-extraction, run: ./run_all.sh all"
  else
    STAGE="all"
    say "No embeddings found. Stage = all."
  fi
fi

# =============================================================================
begin_stage "STAGE 0 of 5   Master metadata"
# =============================================================================
# The master table is the single source of truth for the bird identity and the
# recording_id of every clip. Every later stage reads it. Build it first.
python src/00_build_master_metadata.py

for sp in aa ag; do
  [[ -f "data/${sp}/metadata/${sp}_master.csv" ]] \
    || die "Missing data/${sp}/metadata/${sp}_master.csv"
done

# Two models fail on the shortest clips. Every model reads the padded copy, so
# the input is the same for all of them. See src/00a_pad_audio.py.
python src/00a_pad_audio.py \
  || die "The audio could not be padded. See the output above."
end_stage

# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "embed" ]]; then
  begin_stage "STAGE 1 of 5   Embedding extraction   [GPU, resumable]"
# =============================================================================
  set_bacpipe_device "$DEVICE"

  # bacpipe does not fetch the BirdNET checkpoint. Without it that model writes
  # zero embeddings and the log looks normal.
  python src/01a_fetch_checkpoints.py \
    || die "A model checkpoint could not be downloaded. See the output above."

  for sp in aa ag; do
    python src/01_extract_embeddings.py --species "$sp" --device "$DEVICE"
  done

  # CAUTION: bacpipe prints many tracebacks after it writes the embeddings.
  # Those messages come from a dashboard step that this pipeline does not use.
  # Do not use the absence of tracebacks to confirm success. Count the files.
  python src/01b_verify_embeddings.py || die "The embedding count check failed."
  end_stage
else
  say "STAGE 1 of 5   Skipped. Using existing embeddings."
fi

if [[ "$STAGE" == "embed" ]]; then
  say "COMPLETE (stage 1 only)"
  exit 0
fi

# =============================================================================
begin_stage "STAGE 2 of 5   MFCC baseline   [CPU]"
# =============================================================================
# The MFCC baseline is the classical floor for the benchmark. It writes its
# features in the bacpipe layout, so stage 3 treats it as three more models.
python src/02_extract_mfcc.py
end_stage

# =============================================================================
begin_stage "STAGE 3 of 5   Classification, verification, clustering   [CPU]"
# =============================================================================
# This stage appends one row for each model and skips completed work. To rerun
# one model, delete its row from results/<species>/rows.csv.
for sp in aa ag; do
  python src/03_classify.py --species "$sp"
done
end_stage

# =============================================================================
begin_stage "STAGE 4 of 5   Metric learning   [CPU bound]"
# =============================================================================
# CAUTION: This stage writes its results when a subset finishes. If the process
# stops in the middle of a subset, the work for that subset is lost. Run this
# script inside tmux or screen, or submit it as a batch job.
for sp in aa ag; do
  python src/04_metric_learning.py \
    --species "$sp" --set single --device "$DEVICE" --dump-embeddings
done
end_stage

# =============================================================================
begin_stage "STAGE 5 of 5   Diagnostics and manifest   [CPU]"
# =============================================================================
# Domain shift, within call type, kinship, and split structure. Figures are made
# separately from these tables.
python src/05_diagnostics.py

# The leakage experiment. Demonstrates the mechanism within one species, with
# birds, room, repertoire and calls per bird all held constant.
python src/06_leakage_experiment.py

# The supplementary bout comparison. It runs only when the supplementary audio
# is present, and it never enters a main result.
python src/09_supplementary_bouts.py

# An md5 checksum for every output, so a rerun can be diffed.
python src/07_manifest.py
end_stage

say "COMPLETE"
echo "  results/                Every result table."
echo "  results/MANIFEST.csv    An md5 checksum for every output."
echo "  ${LOG}                  The log of this run."
echo
echo "Duration of each stage:"
for line in "${TIMINGS[@]}"; do echo "  ${line}"; done
echo "  $(printf '%-46s %s' 'TOTAL' "$(hms "$SECONDS")")"
echo
echo "To confirm that a rerun gives the same results, compare the manifests:"
echo "  diff <(sort results/MANIFEST.csv) <(sort results_previous/MANIFEST.csv)"
