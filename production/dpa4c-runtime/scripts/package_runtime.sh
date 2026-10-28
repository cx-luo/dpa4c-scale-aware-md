#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
python3 "$root/tools/packaging/package_runtime.py" --root "$root" "$@"
