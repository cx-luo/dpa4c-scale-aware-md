#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
exec /opt/dpa-production-venv/bin/python "$here/host_cxx.py" "$@"
