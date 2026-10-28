#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
python3 "$root/tools/build/build_runtime.py" --root "$root" "$@" --stage-only
