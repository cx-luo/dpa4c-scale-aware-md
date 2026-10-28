#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
build=/workspace/dpa4c-production-build
stage=/workspace/dpa4c-production-stage
config=Release
jobs=8
incremental=OFF
while (($#)); do
  case "$1" in
    --build-dir) build=$2; shift 2;;
    --stage-dir) stage=$2; shift 2;;
    --config) config=$2; shift 2;;
    --jobs) jobs=$2; shift 2;;
    --incremental) incremental=ON; shift;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2;;
  esac
done
[[ $config == Release ]] || { echo 'Only audited Release builds are supported' >&2; exit 2; }
[[ $build == /workspace/* && $stage == /workspace/* && $build != "$stage" ]] || exit 2
export TMPDIR="$build/tmp"
mkdir -p "$TMPDIR"
set +u
source "${PPU_SDK:-/usr/local/PPU_SDK}/envsetup.sh" cuda >/dev/null
source "${DPA_PYTHON_ENV:-/opt/dpa-production-venv}/bin/activate"
set -u
cmake -S "$root" -B "$build/control" \
  -DCMAKE_BUILD_TYPE="$config" -DDPA_WORK_DIR="$build" -DDPA_STAGE_DIR="$stage" \
  -DDPA_JOBS="$jobs" -DDPA_INCREMENTAL="$incremental"
cmake --build "$build/control" --target runtime_all
