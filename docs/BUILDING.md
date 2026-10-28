# Building and verification

## CPU-only checks

Python 3.12+ is sufficient for `scripts/check_source.py`,
`scripts/verify_results.py`, and `scripts/run_cpu_checks.py`.
They verify packaging, archived arithmetic, Python API edge cases and overlay
integrity. They do not validate a newly compiled MD executable.

## Frozen native dependencies

The canonical lock is
`production/dpa4c-runtime/third_party/sources.lock.json`.
The pinned archives are `deepmd.tar.gz` and `lammps.tar.gz`.
Obtain matching archives from a maintainer-provided release or an authorized
local copy, then stage them without extraction:

```bash
python scripts/stage_dependencies.py --from-dir /path/to/dependency-assets
python scripts/stage_dependencies.py --check
```

The script checks both SHA256 values before copying anything and refuses to
overwrite a different file. No dependency is downloaded automatically. Archived
upstream URLs are provenance references, not an availability guarantee; the
saved LAMMPS retrieval reference was not accessible through the public raw-file
endpoint during packaging. Preserve the exact matching source archive.

Required native environment:

- Linux and a compatible PPU SDK/CUDA-compatible compiler.
- GCC 13, CMake >=3.25, Ninja, patchelf, and the compatible PyTorch/Python environment.
- `PPU_SDK`, `CUDA_HOME`, `DPA_PYTHON_ENV`, and `DPA_MODEL` as described in the
  historical build instructions. Defaults point into `/usr/local/PPU_SDK`
  and `/opt/dpa-production-venv`.
- The model SHA256 must match `data/frozen-version.json`. Weights are external.
- The host compiler and post-link adapters verify exact compiler/kernel
  fingerprints. A source archive alone does not supply that toolchain.

In a dedicated Linux environment, use fresh, disjoint build and stage paths:

```bash
bash production/dpa4c-runtime/scripts/build_all.sh \
  --build-dir /workspace/dpa4c-paper-build \
  --stage-dir /workspace/dpa4c-paper-stage \
  --config Release
```

The original script owns and may replace files in its build/stage directories;
never point it at an active deployment. See the preserved
[historical instructions](../production/dpa4c-runtime/docs/BUILDING.md).

## What a rebuilt executable would establish

A newly built runtime must receive its own identity manifest, actual loaded
library checks, physical correctness gates and timing measurements. Its build
success does not inherit the historical score or binary hashes.

The original bounded `tests/benchmark_abba.py` and `tests/validate_behavior.py`
require external evaluator assets. The former explicitly limits itself to five
cases and is not the nine-case scoring driver. This repository does not recreate
or redistribute the competition evaluator. Follow `docs/RESULTS.md` when reporting
new results, and keep them separate from the retained measurements.
