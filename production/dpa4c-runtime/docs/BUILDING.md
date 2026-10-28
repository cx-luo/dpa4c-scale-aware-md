# Building the complete runtime

Run inside the audited Linux PPU image with its PPU SDK, GCC, Python/PyTorch
environment, CMake, Ninja and patchelf installed. The exact identities of the
compiler, SDK, driver, model and shared external libraries are recorded by each
build in `manifest/runtime-manifest.json`.

The upstream DeePMD commit is `46fdc3e8b0c92524e85dad48ee00a8555a0a7595`.
The upstream LAMMPS commit is `a33449868448baf3c73d8eacdb2d329b13361696`;
its bundled Kokkos is 4.6.2. `third_party/sources.lock.json` pins commit,
retrieval reference, URL, saved archive and SHA256. Copy the verified archives
from the necessary-files package into `third_party/archives/` before building
an exported Git checkout. The normal build performs no network fetch.
The necessary-files package also preserves `source/source-provenance.json`,
which supplies the source Git commit when building an archive without `.git`.

```bash
cd /workspace/dpa4c-production-source/production/dpa4c-runtime
./scripts/build_all.sh \
  --build-dir /workspace/dpa4c-production-build \
  --stage-dir /workspace/dpa4c-production-stage \
  --config Release
```

The default rebuild removes only this build's known source, object, install and
output directories. It validates the two upstream archives, extracts them under
`/workspace`, copies canonical overlays, builds LAMMPS/Kokkos, builds the entire
DeePMD runtime, builds chain, compiles all pair/plugin sources and builds output,
observer and restart libraries. It then renders the one engine template and
generates the runtime manifest. All compiler temporaries use a `/workspace`
TMPDIR. Shared objects are staged under one directory with origin-relative
loader paths. SDK and PyTorch paths refer only to declared external image
dependencies.

The LAMMPS Python API is also an owned build product. The complete package in
`components/lammps-python/package` is checked against its SHA256 lock before
building and copied to `stage/python`. The engine template selects exactly
that directory through `PYTHONPATH`; it never relies on a mutable installed
Python package. The runtime manifest records the package lock, all Python
members and the rendered import path. Verification checks the actual imported
core path, and packaging checks both source and runtime Python members.
Missing, changed or redirected packages fail closed. Render the engine for the
final in-image stage path when relocating a runtime.

The selected Python change preserves the native API: built-in string `mass`
with inferred type directly requests the current native double pointer, and
built-in integer image flags use the native image-width bit layout. Other
arguments retain the existing calls. Atom pointers are never cached. Native
libraries, simulation commands, evaluation workers and outputs are unchanged.
Run `python tools/build/test_python_overlay.py` for small staging and rejection
tests; these tests do not establish a new performance score.

The image decoder returns a fresh zero list for the exact native zero-image
encoding and skips integer truncation only within the signed imageint range.
Oversized integers still wrap exactly, and non-exact integers or unknown ABI
widths keep the native fallback. The mass getter uses PYFUNCTYPE only for its
short native current-pointer lookup, retaining the GIL without caching data.
Run `python tools/build/test_python_api.py` for 32/64-bit, unknown-ABI,
fresh-list and oversized-input regressions. The real-library smoke also checks
mass pointer freshness, generic getter isolation and signed decode behavior.

For an incremental rebuild of this same project:

```bash
./scripts/build_all.sh --incremental \
  --build-dir /workspace/dpa4c-production-build \
  --stage-dir /workspace/dpa4c-production-stage --config Release
```

`--jobs` defaults to eight; chain/plugin compile concurrency is capped at four.
The build never takes objects from a different build tree. Release is the only
audited configuration. Optional environment overrides are `PPU_SDK`,
`CUDA_HOME`, `DPA_PYTHON_ENV` and `DPA_MODEL`; model SHA verification remains
mandatory. A nonempty stage must have this project's ownership marker before
it may be replaced. Build and stage directories must be disjoint descendants
of `/workspace`.

The PPU compatibility overlay retains fail-closed device abort via a trap where
the SDK lacks CUDA device assert. A compiler adapter pins the exact SDK driver
and GCC hashes and fixes the SDK's generated host parameter declaration when a
name appears inside a nested array/member-pointer type. It never rewrites
device code or installed SDK files. Each transformation records the generated
source's input/output SHA256 and affected signature hashes. The original
sorting and minimizer implementations are retained. All adapter code lives in
tools/build; unsupported compiler identities fail closed.
The fitting build also performs an audited, fail-closed post-link traversal rewrite.
The source-built low-register device kernel is identified independently for PPU
1.0 and PPU 1.5 by a stable fingerprint over every device text section, device
constant/data section and kernel resource record. Compiler command provenance,
build-id, debug, symbol and string tables are excluded so source-path changes do
not alter the identity. Exactly 69 instruction sites must match their
before-images: 36 implement independent backward-task reverse traversal and 33
remove the layer-0 MMA tile
whose descriptor-gradient outputs are never consumed. The receipt
is staged as manifest/lowreg-reverse-patch.json and included in the runtime
manifest. No runtime code mutation is used.

The pinned PyTorch package loader creates a system temporary directory without
reading TMPDIR, as shown in its [upstream implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/csrc/inductor/aoti_package/model_package_loader.cpp).
The build stages the model's verified runtime members once and the backend
calls the same AOTI container runner directly. The immutable model wrapper
is recorded separately from source-built project libraries.


## Python GC policy validation

The source-owned Python package lock includes the GC ownership policy, and
`runtime-defaults.json` supplies `LAMMPS_PYTHON_GC_THRESHOLD=524288` to the normal
engine-template merge. The existing overlay/manifest/package checks retain their
closed file inventory and hash requirements. A missing core file or wrong lock
still fails; do not use an old package manifest or old engine receipt.

Run `tools/build/test_python_gc_policy.py` for isolated ownership regressions and
`tools/validation/python_gc_policy_smoke.py` in the selected candidate environment
for real-library lifecycle checks. The latter verifies initialization failure,
repeated/exceptional close, context exceptions, two owners, disabled/zero-threshold
entry, external serial policy changes, destruction and existing NumPy cycles.
Also retain `test_python_api.py`, `test_python_overlay.py` and the three-material
`python_api_smoke.py` gates. All smoke commands perform zero MD steps.

An overlay-only check must call `stage_python_overlay`, not `stage_runtime` or
the full build path: the latter can rewrite native ELF metadata. Verify actual
import path/core SHA, rendered engine opt-in, before/during/after GC tuple,
native library hashes and immutable organizer hashes. See `docs/RUNTIME.md`
for the dedicated-worker, serial-GC-control and no-active-fork scope. Setting
the engine key explicitly to `0` is the runtime opt-out; a source rollback must
restore both core/package lock and runtime-defaults from the rollback tag.
