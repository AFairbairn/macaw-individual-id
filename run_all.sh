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
#   ./run_all.sh [OPTIONS]
#
#   With no options it builds its own environments, finds the dataset, runs
#   stages 0 to 6, and writes two archives to run_outputs/. Nothing else is
#   needed, and no variable has to be set.
#
# OPTIONS
#   --data PATH          The dataset root. Resolved from config.yaml when it is
#                        not given. See THE DATASET below.
#   --stages N-M         Run stages N to M only. Use this to split a run across
#                        short allocations. Every part keeps one run_id.
#   --fresh              Move results/, mfcc_results/ and bacpipe_results/ to
#                        previous_runs/ and start with nothing to reuse.
#   --allow-env-drift    Continue when the installed packages differ from
#                        environment.lock. The difference is recorded in
#                        results/RUN.json and in every result row.
#   --device cpu|cuda    Override the device stage 1 uses. Detected when this is
#                        not given.
#   -h, --help           Print this usage and stop.
#
# EXAMPLES
#   ./run_all.sh                          Everything.
#   ./run_all.sh --stages 0-1             Stage 0 and stage 1. Needs a GPU.
#   ./run_all.sh --stages 2-6             Score existing embeddings. No GPU.
#   ./run_all.sh --fresh                  Everything, reusing nothing.
#   ./run_all.sh --data /scratch/parrot   Read the audio from there.
#
# THE STAGES
#   0   Master metadata, and padding the short clips.        analysis
#   1   Embedding extraction, 15 models.                     extraction, GPU
#   2   The MFCC baseline, 3 variants.                       analysis
#   3   Classification, verification, clustering.            analysis
#   4   Metric learning.                                     analysis
#   5   Diagnostics, the leakage experiment, the manifest.   analysis
#   6   The output archives.                                 analysis
#
# THE TWO ENVIRONMENTS
#   Stage 1 imports bacpipe, which pins the whole numerical and audio stack.
#   Every other stage imports none of it. One environment made every result
#   table hostage to whatever bacpipe resolved, and on 2026-08-01 that wrote one
#   results tree from two different stacks.
#
#   setup.sh builds .venv-extraction/ and .venv-analysis/. This script calls
#   each stage with the absolute path of its interpreter. Nothing is activated,
#   here or by the user.
#
# THE DATASET
#   Resolved in this order, first hit wins: --data, then PARROT_DATA, then
#   dataset_path in config.yaml. A directory is the dataset when it holds the
#   audio. If none of them does, the run stops and prints every path it tried.
#
# THE MACHINE
#   No job scheduler. Use --stages to split a run across short allocations.
#   No required GPU. Stage 1 uses one when it is present. Every other stage is
#   CPU only.
#   No network at run time, once the environments and the checkpoints are on
#   disk.
#   Python 3.11 or newer, on Linux or macOS.
#
# EXIT CODES
#   0   Every stage finished and every check passed.
#   1   A stage failed, an environment differs from its record, or a check in
#       stage 5 failed. A failed check does not stop the run. The archives are
#       written first, so the evidence travels with the failure.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# -----------------------------------------------------------------------------
# Defaults.
# -----------------------------------------------------------------------------
DATA_ARG=""
DEVICE_ARG=""
FRESH=0
ALLOW_DRIFT=0
FIRST_STAGE=0
LAST_STAGE=6

VENV_ANALYSIS="${HERE}/.venv-analysis"
VENV_EXTRACTION="${HERE}/.venv-extraction"
PY_ANALYSIS="${VENV_ANALYSIS}/bin/python"
PY_EXTRACTION="${VENV_EXTRACTION}/bin/python"

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
say() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }

usage() { sed -n '2,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# -----------------------------------------------------------------------------
# Read the options.
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)   [[ $# -ge 2 ]] || die "--data needs a path."; DATA_ARG="$2"; shift 2 ;;
    --device) [[ $# -ge 2 ]] || die "--device needs 'cpu' or 'cuda'."; DEVICE_ARG="$2"; shift 2 ;;
    --fresh)  FRESH=1; shift ;;
    --allow-env-drift) ALLOW_DRIFT=1; shift ;;
    --stages)
      [[ $# -ge 2 ]] || die "--stages needs a range, for example 2-6."
      [[ "$2" =~ ^([0-6])-([0-6])$ ]] || die "--stages takes a range from 0 to 6, for example 2-6. This was '$2'."
      FIRST_STAGE="${BASH_REMATCH[1]}"
      LAST_STAGE="${BASH_REMATCH[2]}"
      [[ "$FIRST_STAGE" -le "$LAST_STAGE" ]] || die "--stages ${2}: the first stage is after the last."
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    # The old positional form. Name the replacement rather than fail silently.
    all|embed|analyse|auto)
      die "'$1' is no longer an argument. Use --stages instead.
  ./run_all.sh              was 'all'
  ./run_all.sh --stages 0-1 was 'embed'
  ./run_all.sh --stages 2-6 was 'analyse'" ;;
    *) die "Unknown option '$1'. Run ./run_all.sh --help" ;;
  esac
done

wanted() { [[ "$1" -ge "$FIRST_STAGE" && "$1" -le "$LAST_STAGE" ]]; }

# One thread for the numerical libraries. A reduction runs in a different order
# on a machine with a different core count, which changes the last digits of
# every result and therefore every checksum in results/MANIFEST.csv. Section 7
# of the README asks the reader to compare those checksums, so they have to be
# reproducible across machines.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Read and write UTF-8, whatever the locale of the machine.
export PYTHONUTF8=1

mkdir -p logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

# -----------------------------------------------------------------------------
# Stage timing. Use these durations to plan the next run.
# -----------------------------------------------------------------------------
STAGE_START=0
STAGE_NAME=""
TIMINGS=()

hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
begin_stage() { STAGE_NAME="$1"; STAGE_START=$SECONDS; say "$1"; }
end_stage() {
  local seconds=$((SECONDS - STAGE_START))
  TIMINGS+=("$(printf '%-46s %s' "$STAGE_NAME" "$(hms "$seconds")")")
  printf '\033[1m         %s took %s\033[0m\n' "$STAGE_NAME" "$(hms "$seconds")"
}

# =============================================================================
# Step 1. Start fresh, if that was asked for.
#
# bacpipe_results/ is moved with the rest. A new extraction environment resolves
# its own torch, tensorflow and jax, so embeddings kept from an older one would
# reintroduce the fault this script exists to stop.
# =============================================================================
if [[ "$FRESH" -eq 1 ]]; then
  ARCHIVE="previous_runs/$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$ARCHIVE"
  MOVED=0
  for directory in results mfcc_results bacpipe_results results_manual; do
    if [[ -d "$directory" ]]; then
      mv "$directory" "${ARCHIVE}/"
      echo "  moved ${directory}/ to ${ARCHIVE}/"
      MOVED=1
    fi
  done
  [[ "$MOVED" -eq 1 ]] || echo "  nothing to move."
  say "Fresh run. Earlier output is in ${ARCHIVE}/"
fi

# =============================================================================
# Step 2. Build the environments this run needs.
#
# The extraction environment is built only when stage 1 is in the range, so a
# machine that scores the published embeddings never installs bacpipe.
# =============================================================================
say "Environments"

PROFILES=(analysis)
wanted 1 && PROFILES=(extraction analysis)

./setup.sh "${PROFILES[@]}" || die "The environments could not be built. See the output above."

[[ -x "$PY_ANALYSIS" ]] || die "${PY_ANALYSIS} not found after setup.sh ran."

# =============================================================================
# Step 3. Compare each environment against its record.
#
# The lock is written by the authors, once, with a separate command. This script
# never writes it. See the README section for the authors.
# =============================================================================
check_environment() {
  local profile="$1" interpreter="$2"
  local lock="environment.lock/${profile}/packages.txt"

  if [[ ! -f "$lock" ]]; then
    echo "  ${profile}: no environment record. Continuing."
    return 0
  fi

  if "$interpreter" src/freeze_environment.py verify --profile "$profile"; then
    echo "  ${profile}: matches ${lock}"
    return 0
  fi

  if [[ "$ALLOW_DRIFT" -eq 1 ]]; then
    echo "  ${profile}: differs from ${lock}. Continuing, because --allow-env-drift is set."
    return 0
  fi

  die "The ${profile} environment differs from ${lock}. The differences are above.

This is the check that would have caught the run of 2026-08-01, where one
results tree was written by two different stacks.

To reproduce the published numbers, rebuild the environment:
  rm -rf .venv-${profile} && ./setup.sh ${profile}

To continue anyway, and record that you did:
  ./run_all.sh --allow-env-drift"
}

say "Environment record"
wanted 1 && check_environment extraction "$PY_EXTRACTION"
check_environment analysis "$PY_ANALYSIS"

# =============================================================================
# Step 4. Identify this run, and find the dataset.
#
# A run split across allocations with --stages keeps the run_id of its first
# part, so the parts of one run carry one identifier. That only holds while the
# environment holds, so the check above has to have passed first.
# =============================================================================
RUN_ID=""
if [[ -f results/RUN.json && "$FRESH" -eq 0 ]]; then
  RUN_ID="$("$PY_ANALYSIS" -c 'import json;print(json.load(open("results/RUN.json"))["run_id"])')"
  say "Continuing run ${RUN_ID}, stages ${FIRST_STAGE} to ${LAST_STAGE}"
else
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  say "Starting run ${RUN_ID}, stages ${FIRST_STAGE} to ${LAST_STAGE}"
fi

EXTRACTION_HASH=""
if wanted 1 && [[ -x "$PY_EXTRACTION" ]]; then
  EXTRACTION_HASH="$("$PY_EXTRACTION" src/common.py | "$PY_ANALYSIS" -c 'import json,sys;print(json.load(sys.stdin)["env_hash"])')"
fi

DRIFT_FLAG=()
[[ "$ALLOW_DRIFT" -eq 1 ]] && DRIFT_FLAG=(--allow-env-drift)

"$PY_ANALYSIS" src/common.py --start-run \
  --run-id "$RUN_ID" \
  --data "$DATA_ARG" \
  --stages "${FIRST_STAGE}-${LAST_STAGE}" \
  --extraction-env-hash "$EXTRACTION_HASH" \
  "${DRIFT_FLAG[@]}" \
  || die "The run record could not be written. See the output above."

# Every stage reads the audio through this variable. The user never sets it.
PARROT_DATA="$("$PY_ANALYSIS" -c 'import json;print(json.load(open("results/RUN.json"))["dataset_path"])')"
export PARROT_DATA

# =============================================================================
# Step 5. The device for stage 1.
# =============================================================================
DEVICE="cpu"
if wanted 1; then
  if [[ -n "$DEVICE_ARG" ]]; then
    case "$DEVICE_ARG" in
      cpu|cuda) DEVICE="$DEVICE_ARG" ;;
      *) die "--device is '${DEVICE_ARG}'. Use 'cpu' or 'cuda'." ;;
    esac
    say "Device: ${DEVICE} (given by --device)"
  else
    # A broken torch install must not read as 'no GPU'. Import first, and stop
    # with the real message when the import fails.
    "$PY_EXTRACTION" -c "import torch" \
      || die "torch does not import in the extraction environment. Rebuild it:
  rm -rf .venv-extraction && ./setup.sh extraction"
    if "$PY_EXTRACTION" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
      DEVICE="cuda"
    fi
    say "Device: ${DEVICE} (detected)"
  fi
  [[ "$DEVICE" == "cpu" ]] && echo "  Stage 1 on CPU is slow. Every other stage is CPU bound anyway."
fi

# =============================================================================
if wanted 0; then
  begin_stage "STAGE 0 of 6   Metadata, audio and padding      [CPU]"
# =============================================================================
  # The master table is the single source of truth for the bird identity and the
  # recording_id of every clip. Every later stage reads it. This checks it, then
  # checks that every audio file is present and holds audio, then pads the clips
  # below the minimum length, because two models fail on the shortest ones.
  "$PY_ANALYSIS" src/00_prepare_data.py \
    || die "The data could not be prepared. See the output above."
  end_stage
fi

# =============================================================================
if wanted 1; then
  begin_stage "STAGE 1 of 6   Embedding extraction             [GPU, resumable]"
# =============================================================================
  # This stage checks every checkpoint before it starts, runs the 15 models, and
  # counts the output files afterwards. It deletes no checkpoint unless
  # --repair-checkpoints is given.
  #
  # CAUTION: bacpipe prints many tracebacks after it writes the embeddings.
  # Those come from a dashboard step this pipeline does not use. Do not read the
  # absence of tracebacks as success. The count at the end of the stage is what
  # says whether it worked.
  for sp in aa ag; do
    "$PY_EXTRACTION" src/01_extract_embeddings.py \
      --species "$sp" --device "$DEVICE" --clear \
      || die "Extraction failed for ${sp}. See the output above."
  done
  end_stage
fi

# =============================================================================
if wanted 2; then
  begin_stage "STAGE 2 of 6   MFCC baseline                    [CPU]"
# =============================================================================
  # The MFCC baseline is the classical floor for the benchmark. It writes its
  # features in the bacpipe layout, so stage 3 treats it as three more models.
  "$PY_ANALYSIS" src/02_extract_mfcc.py
  end_stage
fi

# =============================================================================
if wanted 3; then
  begin_stage "STAGE 3 of 6   Scoring the frozen embeddings    [CPU]"
# =============================================================================
  # Probe, nearest centroid, verification and clustering, on every
  # representation exactly as its model produces it. Nothing is trained here.
  # This stage appends one row for each model and skips completed work. To
  # rescore one model, delete its row from results/<species>/rows.csv.
  for sp in aa ag; do
    "$PY_ANALYSIS" src/03_score_frozen.py --species "$sp"
  done
  end_stage
fi

# =============================================================================
if wanted 4; then
  begin_stage "STAGE 4 of 6   Metric learning                  [CPU]"
# =============================================================================
  for sp in aa ag; do
    # CPU, not "$DEVICE". The head is one small network on fewer than 1,100
    # vectors, and CUDA kernels are not deterministic, so a GPU run gives
    # different numbers on every machine and between two runs on one machine.
    "$PY_ANALYSIS" src/04_metric_learning.py \
      --species "$sp" --set single --device cpu --dump-embeddings
  done
  end_stage
fi

# =============================================================================
if wanted 5; then
  begin_stage "STAGE 5 of 6   Diagnostics                      [CPU]"
# =============================================================================
  # Domain shift, within call type, kinship, split structure, the by-date
  # comparison for aa, and the checks that the reported claims still hold.
  "$PY_ANALYSIS" src/05_diagnostics.py

  # The leakage experiment. It demonstrates the mechanism within one species,
  # with birds, room, repertoire and calls for each bird all held constant.
  "$PY_ANALYSIS" src/05a_leakage_experiment.py

  # The supplementary bout comparison. It runs only when the supplementary audio
  # is present, and it never enters a main result, so a machine without that
  # audio must still finish the run.
  "$PY_ANALYSIS" src/05b_supplementary_bouts.py \
    || say "The supplementary bout comparison did not run. No main result uses it."
  end_stage
fi

# =============================================================================
DETERMINISM="not run"
if wanted 6; then
  begin_stage "STAGE 6 of 6   Determinism, manifest, archives  [CPU]"
# =============================================================================
  # Score one model twice and compare, write an md5 for every output, then write
  # the two archives. A failed determinism check does not stop the archives, so
  # the evidence travels with the failure.
  if "$PY_ANALYSIS" src/06_finish_run.py --log "$LOG"; then
    DETERMINISM="passed"
  else
    DETERMINISM="FAILED"
  fi
  end_stage
fi

# =============================================================================
# The result.
# =============================================================================
CHECKS="not run"
CHECKS_FILE="results/diagnostics/checks.csv"
if [[ -f "$CHECKS_FILE" ]]; then
  if "$PY_ANALYSIS" -c "
import sys, pandas as pd
table = pd.read_csv('${CHECKS_FILE}')
failed = table[table['outcome'] != 'pass']
for _, row in failed.iterrows():
    print(f\"  FAILED  {row['check']}: expected {row['expected']}, observed {row['observed']}\")
sys.exit(1 if len(failed) else 0)
"; then
    CHECKS="passed"
  else
    CHECKS="FAILED"
  fi
fi

say "COMPLETE"
echo "  run_id                  ${RUN_ID}"
echo "  results/                Every result table."
echo "  results/RUN.json        The record of this run."
echo "  results/MANIFEST.csv    An md5 for every output."
echo "  run_outputs/            The archives to download."
echo "  ${LOG}"
echo
echo "Duration of each stage:"
for line in "${TIMINGS[@]}"; do echo "  ${line}"; done
echo "  $(printf '%-46s %s' 'TOTAL' "$(hms "$SECONDS")")"
echo
echo "  Determinism check:  ${DETERMINISM}"
echo "  Reported claims:    ${CHECKS}"

if [[ "$DETERMINISM" == "FAILED" || "$CHECKS" == "FAILED" ]]; then
  echo
  echo "A check failed. The archives above still hold every number and the log,"
  echo "so the evidence is in run_outputs/ with the failure."
  exit 1
fi
