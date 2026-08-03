#!/usr/bin/env bash
# =============================================================================
# setup.sh
#
# PURPOSE
#   Build the two virtual environments the pipeline runs in. run_all.sh calls
#   this script when an environment is absent, so the user never runs it.
#
# USAGE
#   ./setup.sh              Build both environments.
#   ./setup.sh extraction   Build one of them.
#   ./setup.sh analysis
#
# INPUT
#   requirements-extraction.txt
#   requirements-analysis.txt
#
# OUTPUT
#   .venv-extraction/       The environment stage 1 runs in.
#   .venv-analysis/         The environment stages 0 and 2 to 6 run in.
#   .venv-<profile>/.requirements.md5
#                           The md5 of the requirements file that built the
#                           environment. See WHY THERE IS A STAMP FILE.
#
# WHY THERE ARE TWO ENVIRONMENTS
#   Stage 1 imports bacpipe, which pins the whole numerical and audio stack.
#   Stages 0 and 2 to 6 import none of it. One environment made every result
#   table hostage to whatever bacpipe resolved, and on 2026-08-01 that wrote one
#   results tree from two different stacks.
#
# WHY THERE IS A STAMP FILE
#   A rerun must cost nothing. The stamp holds the md5 of the requirements file
#   that built the environment. If the stamp matches, the profile is skipped. If
#   the requirements file changed, the packages are installed again.
#
#   The stamp is not a substitute for environment.lock. The lock records what
#   was installed. The stamp records what was asked for.
#
# CAUTION
#   This script needs network access, because pip downloads packages. If pip
#   cannot reach the package index, the script stops and prints the pip error
#   and the profile name. It does not continue into a stage. Once the
#   environments exist, no stage needs the network.
#
# EXIT CODES
#   0   Every requested profile is ready.
#   1   An interpreter is too old, or pip failed.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# find_python
#
# Return the path of a Python 3.11 or newer interpreter. Set PYTHON to choose a
# different one. Many Linux distributions ship no 'python', only 'python3', and
# a cluster module can leave an older one first on the path, so the named
# versions are tried before the bare names.
# -----------------------------------------------------------------------------
find_python() {
  local candidate
  for candidate in "${PYTHON:-}" python3.13 python3.12 python3.11 python3 python; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

# -----------------------------------------------------------------------------
# md5_of: return the md5 of one file.
#
# Linux ships md5sum and macOS ships md5. Python is present either way, because
# this script cannot run without it, so Python does it and the result is the
# same on both.
# -----------------------------------------------------------------------------
md5_of() {
  "$PY_BIN" -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"
}

# -----------------------------------------------------------------------------
# build_profile
#
# Build one environment. The steps are:
#   1. If the stamp matches the requirements file, do nothing.
#   2. Create the environment if the directory is absent.
#   3. Install the requirements file.
#   4. Write the stamp.
# -----------------------------------------------------------------------------
build_profile() {
  local profile="$1"
  local venv=".venv-${profile}"
  local requirements="requirements-${profile}.txt"
  local stamp="${venv}/.requirements.md5"

  [[ -f "$requirements" ]] || die "${requirements} not found."

  local wanted
  wanted="$(md5_of "$requirements")"

  if [[ -f "$stamp" ]] && [[ "$(cat "$stamp")" == "$wanted" ]] && [[ -x "${venv}/bin/python" ]]; then
    printf '  %-11s ready. %s is unchanged.\n' "${profile}:" "$requirements"
    return 0
  fi

  if [[ ! -x "${venv}/bin/python" ]]; then
    printf '  %-11s creating %s with %s\n' "${profile}:" "$venv" "$("$PY_BIN" -V 2>&1)"
    "$PY_BIN" -m venv "$venv" || die "Could not create ${venv}."
  else
    printf '  %-11s %s changed. Installing again.\n' "${profile}:" "$requirements"
  fi

  # A stale stamp must not survive a failed install. Remove it first, so an
  # interrupted install is repeated rather than reported as ready.
  rm -f "$stamp"

  printf '  %-11s installing %s\n' "${profile}:" "$requirements"
  "${venv}/bin/python" -m pip install -q -U pip \
    || die "pip could not be upgraded in the ${profile} environment. Check the network."
  "${venv}/bin/python" -m pip install -q -r "$requirements" \
    || die "pip failed for the ${profile} profile. The error is above. Check that the package index is reachable."

  printf '%s' "$wanted" > "$stamp"
  printf '  %-11s ready.\n' "${profile}:"
}

# -----------------------------------------------------------------------------
# Run.
# -----------------------------------------------------------------------------
PY_BIN="$(find_python)" || die "Python 3.11 or newer is required, and none was found.
Set PYTHON to the interpreter you want, for example:
  PYTHON=/usr/bin/python3.11 ./setup.sh
On a machine with no Python 3.11, uv installs one without administrator rights:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv python install 3.11"

echo "Interpreter: ${PY_BIN} ($("$PY_BIN" -V 2>&1))"

if [[ $# -gt 0 ]]; then
  for name in "$@"; do
    case "$name" in
      extraction|analysis) build_profile "$name" ;;
      *) die "Unknown profile '${name}'. Use 'extraction' or 'analysis'." ;;
    esac
  done
else
  build_profile extraction
  build_profile analysis
fi
