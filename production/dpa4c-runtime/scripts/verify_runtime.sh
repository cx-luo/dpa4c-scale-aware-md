#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
python3 "$root/tools/manifest/validate_runtime.py" --root "$root" "$@"
