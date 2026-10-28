# Runtime files and semantic configuration

The sole formal configuration is `config/engine.production.json.in`; the build
renders it to `STAGE/config/engine.json`. `runtime-defaults.json` holds adopted
switches and launch parameters. The engine points at staged libraries, plugins,
model and Python launcher. The launcher enters the declared external Python
environment. The manifest records all stage files, symlink targets, SHA256,
SONAME, loader paths, build target and external dependency identities.

Built components are DeePMD core/CUDA/operator/C/C++/PT-expt interfaces,
LAMMPS/Kokkos, the DPA chain, `dpplugin.so`, NVT integration, the guarded small-NPT coroutine
integration, native state output, device observer, device restart, zerocopy
restart and the fitting launch adapter. The direct priming and combined node
capabilities are compiled into their semantic owners. No external precompiled
project library replaces a source build.

The PPU SDK, PyTorch distribution, Python, MPI, GNU runtime and system libraries
are supplied by the audited runtime image. Their real path, SONAME, version,
SHA256, origin and image-provided status are recorded separately from built
products. The fixed compressed model is copied with verified SHA256
`6ab078d8b5f2a3e6ed2adbfefefa5e4be2244bc2c83f7db58bef126657d342e7`.
Cutoff remains 6 Angstrom and skin 1 Angstrom.

The build also extracts the immutable model's runtime members into
`models/aoti-model/` and records every archive member and SHA256 in
`manifest/model-package.json`. Its wrapper library is a model-supplied
artifact, not a project library imported from a previous build. The original
archive and model bytes stay unchanged. The backend uses the image's existing
AOTI container runner directly, so this production configuration performs no
runtime model extraction or compilation. The two `DPA_AOTI_MODEL_*` paths
must resolve inside the staged model directory. The model wrapper keeps its
original ELF metadata; an absent SONAME is intentional for this absolute-path
model artifact.
The provider exposes the driver version and user-space driver libraries, whose
identities are recorded. Its host kernel module file is not mounted in the
container, so its binary hash is explicitly unavailable in the manifest.

The primary policies are:

- `DPA_LARGE_FORCE_ALIAS=1`: permit identity-map, contiguous, single-rank output
  aliasing when the full capability guard succeeds; maximum node count remains
  the audited value in runtime-defaults.
- `DPA_CONDITIONAL_VIRIAL_WRITEBACK=1`: skip virial widening when the stream's
  explicitly requested virial state permits it.
- `DPA_NPT_GLOBAL_VIRIAL_COMPONENTS=6`: the ordinary pressure path computes the
  six physical tensor components; full-centroid consumers select nine.
- `DPA_FITTING_PIPELINE=1`: independent forward/backward tasks share a block
  weight image and reverse the independent backward task traversal.
- `DPA_NEIGHBOR_CACHE_POLICY=1`: retain stable candidate storage while honoring
  native rebuild decisions and the exact current-step cutoff filter.
- `DPA_SMALL_NVT_PACKED_FORCE=1`: consume packed force directly only after the
  explicit NVT capability is armed; observers and unsupported consumers
  materialize the FP64 representation.
- `DPA_NVT_DEVICE_MAX_ATOMS=1000000`: admit the existing strictly guarded
  cooperative NVT device-control path for the audited single-rank large NVT
  shapes. Unsupported capacities, observers, output phases and capability
  failures retain the native path. The analogous large NPT finite-epoch route
  remains capped at 65536 atoms because its large-shape continuous-trajectory
  and native-restart gates failed.
- `DPA_SMALL_NPT_COROUTINE=1`: enable only for its fail-closed, single-rank,
  at-most-4096-atom, fixed-target diagonal ISO-NPT capability. Every
  unsupported state uses native `FixNPTKokkos` behavior.
- `DPA_DEVICE_RESTART_ASYNC_HANDOFF=1`: for the existing strictly gated, single-rank
  atomic restart path, produce a private device snapshot on the default stream,
  copy it into the zero-copy writer's pinned slot on a non-blocking stream and
  let the writer wait on the completion event. Final and unsupported restarts
  keep the synchronous path and flush every outstanding periodic job.

Other adopted knobs in runtime-defaults control compact graph, adaptive launch
shape, dense/sparse execution windows, readout, small fitting, neighbor-check
integration, observer scheduling and native restart. They are read using the
same semantic names throughout the owned source. Unadopted alternative routes
are absent from the production dispatch. The guarded small-NPT coroutine is
the selected NPT implementation; its prior cross-step source remains in-tree
as migration evidence but is not registered concurrently.

Changing a stage's library or engine invalidates its validation identities.
Building or verifying this directory never replaces the active production
engine or libraries. Promotion is a separate user-authorized action.

## Complete source-built shared library inventory

- `lib/libdeepmd.so`: locked DeePMD core; SONAME `libdeepmd.so`.
- `lib/libdeepmd_backend_ptexpt.so`: PT-expt backend with direct staged AOTI model loading; SONAME `libdeepmd_backend_ptexpt.so`.
- `lib/libdeepmd_c.so`: locked DeePMD C API; SONAME `libdeepmd_c.so`.
- `lib/libdeepmd_cc.so`: locked DeePMD C++ API; SONAME `libdeepmd_cc.so`.
- `lib/libdeepmd_cc_low.so`: byte-identical C++ API alias; SONAME `libdeepmd_cc.so`.
- `lib/libdeepmd_dyn_cudart.so`: locked DeePMD dynamic CUDA adapter; SONAME `libdeepmd_dyn_cudart.so`.
- `lib/libdeepmd_op_cuda.so`: locked DeePMD CUDA operator library; SONAME `libdeepmd_op_cuda.so`.
- `lib/libdeepmd_op_pt.so`: canonical operator overlays and model registration/priming; SONAME `libdeepmd_op_pt.so`.
- `lib/libdeviceobserver.so`: device observer and output scheduling; SONAME `libdeviceobserver.so`.
- `lib/libdevicerestart.so`: device restart packing; SONAME `libdevicerestart.so`.
- `lib/libdpa-chain.so`: canonical semantic forward/fitting/node/force chain; SONAME `libdpa-chain.so`.
- `lib/libgrid.so`: fitting launch adapter; SONAME `libgrid.so`.
- `lib/liblammps.so`: LAMMPS link alias; SONAME `liblammps.so.0`.
- `lib/liblammps.so.0`: locked LAMMPS/Kokkos with canonical control overlays; SONAME `liblammps.so.0`.
- `lib/libzerocopyrestart.so`: borrowed restart writer; SONAME `libzerocopyrestart.so`.
- `plugin/dpplugin.so`: deepmd/kk pair plugin and required support sources; SONAME `dpplugin.so`.
- `plugin/native-state-plugin.so`: native state/scalar/restart registration; SONAME `native-state-plugin.so`.
- `plugin/npt-integration-plugin.so`: guarded small ISO-NPT atomic-ticket mapped host/device coroutine; unsupported states fall back to native `FixNPTKokkos`; SONAME `npt-integration-plugin.so`.
- `plugin/nvt-integration-plugin.so`: selected NVT integration; SONAME `nvt-integration-plugin.so`.

The final runtime manifest records complete SHA256 for these nineteen file paths (seventeen distinct ELF images), their real targets and all observed external libraries. Actual candidate workers load only staged project libraries and the staged immutable model wrapper.


## Bulk restart ABI

The locked LAMMPS `WriteRestart::write` checks the optional
`dpa_device_restart_pack_batch` entry before its per-atom virtual loop.
The runtime accepts only a pinned destination owned by the periodic asynchronous
writer, the strict single-rank 11-double atomic layout, and an exact match to
`AtomVec::size_restart()`. A rejected batch returns -1 before submitting work,
so the original loop remains the fallback. A non-negative size mismatch aborts
instead of racing a partial asynchronous payload with the fallback.

The first periodic file, unsupported shapes and final restart retain their
previous behavior. The batch reuses the existing event-owned transfer pool;
it changes neither restart bytes nor the model/physics. Required output cadence
and final regular-file completion are preserved.


## Shared end-step output snapshot

`DPA_SHARED_OUTPUT_SNAPSHOT=1` lets the supported periodic binary state dump
produce one sorted 14-double state image plus original atom order, mask and image
metadata. One asynchronous D2H transfers 124 bytes per atom. The native restart
writer converts that immutable host snapshot to its exact 11-double format in
its independently-owned pinned destination. Medium uses one conversion thread;
large defaults to four through `DPA_SHARED_OUTPUT_THREADS`.

The state writer owns the copy event and pinned snapshot until the restart
conversion finishes. The restart writer owns the converted destination until
its final file is closed. All file payloads and final completion remain exact.
Both snapshot slots are allocated during initialization. The GPU ownership flag
is restored after any later allocation growth.

Sharing requires a single matching periodic restart on the same end-step,
strict atomic restart eligibility, one supported state dump and optional scalar
dump, no restart PBC remap and no build-once reset. The first periodic checkpoint
and final dump in each run use independent routes. Unsupported layouts fall
back. Internal-force capture remains a separate phase and buffer.

`DPA_SHARED_OUTPUT_VALIDATE=1` is an optional diagnostic: state bytes are compared
with stock pack/sort and restart bytes with an independent device restart pack
on the same snapshot. It is absent from the production engine. These checks and
bounded uninterrupted trajectories/native replay passed without a full
nine-case evaluation.

## Low-register large fitting path

DPA_LOWREG_FITTING=1 selects the low-register implementation only after the
existing fitting pipeline accepts the call. The production threshold remains
DPA_FITTING_PIPELINE_MIN_NODES=100000, so small and medium routing is unchanged.
The build-time reverse traversal changes the order of independent grid-stride
tasks while preserving per-task arithmetic and accumulation order. The same
fail-closed build step removes only layer-0 descriptor-gradient dimensions 62..69,
which the coordinate VJP never consumes.


## Opt-in Python cyclic-GC policy

`LAMMPS_PYTHON_GC_THRESHOLD=524288` raises only the process gen0 collection
threshold while opted-in LAMMPS instances exist. The first successfully created
instance saves the actual threshold tuple and enabled state. A module lock and
owner count share the policy; the last close/destructor restores the previous
tuple in a finally block. The policy neither disables GC nor changes gen1/gen2.
An already-zero threshold remains zero. Invalid settings fail explicitly.

This is a process-wide policy for dedicated workers whose GC settings are
controlled serially. Do not fork while the policy owns an active instance.
The lock coordinates participating LAMMPS owners; it is not an atomic
compare-and-swap with arbitrary concurrent `gc.set_threshold` callers.
Sequential external threshold changes are preserved by compare-and-restore,
and external enable/disable changes are never overwritten. Set the engine key
explicitly to `0` to opt out, including rollback/control engines, since runner
environment merging inherits unspecified parent environment variables.

The measured image decoder returns fresh acyclic lists of integers, which
remain subject to ordinary reference counting. The higher finite threshold
avoids repeatedly scanning these transient lists. It does not change decoder,
mass, model, MD, output, organizer, NumPy or native-library code. Owner tracking
stores no LAMMPS instance references and introduces no self-cycle.

On the audited Python `3.12.7+gc`, the actual baseline is `(2000,10,0)` and the
third setter argument is ignored; tests compare observable values. All 8 retained
endpoint tests and 14 real-library lifecycle checks passed, with no candidate
collections in 48 legs. Peak large RSS was 408 MiB versus 406.36 MiB for the 2M
reference. The four-case conditional score budget was +0.123212%, including only
the image-expression savings; this is not a measured new nine-case score.

This source change has no new image or formal ready package. Rebuilding an engine
changes its validation identity; prior nine-case reports cannot qualify it.
