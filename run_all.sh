#!/usr/bin/env bash
# =============================================================================
# run_all.sh
#
# PURPOSE
#   Reproduce every result in the parrot individual-identification paper.
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
# Write DEVICE into the settings.yaml file of the installed bacpipe package.
#
# CAUTION: bacpipe reads its device from this file. It does not detect CUDA.
# The shipped value is 'cpu'. If you do not change this value, the torch models
# run on CPU on a GPU node. The run takes about 10 times longer and gives no
# error message.
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
  Stage 1 on CPU is very slow. Expect about one day for 15 models and 1,541 clips.
  To reproduce the analysis only, run: ./run_all.sh analyse
EOF
fi

python -c "import numpy, pandas, sklearn, torch, librosa, yaml" \
  || die "The environment is incomplete. Create it with: conda env create -f environment.yml"

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
say "STAGE 0 of 5   Master metadata"
# =============================================================================
# The master table is the single source of truth for the bird identity and the
# recording_id of every clip. Every later stage reads it. Build it first.
python src/00_build_master_metadata.py

for sp in aa ag; do
  [[ -f "data/${sp}/metadata/${sp}_master.csv" ]] \
    || die "Missing data/${sp}/metadata/${sp}_master.csv"
done

# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "embed" ]]; then
  say "STAGE 1 of 5   Embedding extraction   [GPU, 4 to 8 hours, resumable]"
# =============================================================================
  set_bacpipe_device "$DEVICE"

  for sp in aa ag; do
    python src/01_extract_embeddings.py --species "$sp" --device "$DEVICE"
  done

  # CAUTION: bacpipe prints many tracebacks after it writes the embeddings.
  # Those messages come from a dashboard step that this pipeline does not use.
  # Do not use the absence of tracebacks to confirm success. Count the files.
  python src/01b_verify_embeddings.py || die "The embedding count check failed."
else
  say "STAGE 1 of 5   Skipped. Using existing embeddings."
fi

if [[ "$STAGE" == "embed" ]]; then
  say "COMPLETE (stage 1 only)"
  exit 0
fi

# =============================================================================
say "STAGE 2 of 5   MFCC baseline   [CPU, 1 minute]"
# =============================================================================
# The MFCC baseline is the classical floor for the benchmark. It writes its
# features in the bacpipe layout, so stage 3 treats it as three more models.
python src/02_extract_mfcc.py

# =============================================================================
say "STAGE 3 of 5   Classification, verification, clustering   [CPU, 1 to 2 hours]"
# =============================================================================
# This stage appends one row for each model and skips completed work. To rerun
# one model, delete its row from results/<species>/rows.csv.
for sp in aa ag; do
  python src/03_classify.py --species "$sp"
done

# =============================================================================
say "STAGE 4 of 5   Metric learning   [CPU bound, 3 to 6 hours]"
# =============================================================================
# CAUTION: This stage writes its results when a subset finishes. If the process
# stops in the middle of a subset, the work for that subset is lost. Run this
# script inside tmux or screen, or submit it as a batch job.
for sp in aa ag; do
  python src/04_metric_learning.py \
    --species "$sp" --set single --device "$DEVICE" --dump-embeddings
done

# =============================================================================
say "STAGE 5 of 5   Diagnostics, tables, figures, manifest   [CPU, 10 minutes]"
# =============================================================================
python src/05_diagnostics.py     # Domain shift, within call type, kinship.
python src/06_tables_figures.py  # Every table and figure in the paper.
python src/07_manifest.py        # An md5 checksum for every output.

say "COMPLETE"
echo "  results/                All tables and figures."
echo "  results/MANIFEST.csv    An md5 checksum for every output."
echo "  ${LOG}                  The log of this run."
echo
echo "To confirm that a rerun gives the same results, compare the manifests:"
echo "  diff <(sort results/MANIFEST.csv) <(sort results_previous/MANIFEST.csv)"
