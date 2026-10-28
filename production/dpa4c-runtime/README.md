# Unified DPA4C production runtime

This directory is the sole canonical production source entry. Its complete
clean Release build, static loader/ABI checks, five-case 1/32/native-restart
gates and finite A-B-B-A screen passed on 2026-10-13.

`scripts/build_all.sh` builds every owned runtime component from the locked
offline upstream archives and canonical source. `scripts/verify_runtime.sh`
checks the staged runtime and saved validation identities; `--run-gates`
performs fresh bounded gates. `scripts/package_runtime.sh` creates the
necessary-files archive with verified upstream archives and lightweight
validation receipts.

The build also delivers the source-owned LAMMPS Python package and selects it
with the engine's in-stage `PYTHONPATH`. Its locked, validated API fastpaths are
included in source and runtime archives; manifest and package checks reject a
missing or different Python overlay. See `docs/BUILDING.md` for the staging
and import checks. This delivery integration does not change native libraries
or claim a new nine-case score.

Read `docs/ARCHITECTURE.md`, `BUILDING.md`, `RUNTIME.md`, `VALIDATION.md` and
`MIGRATION.md`. Older production directories remain only as rollback evidence;
the canonical build has no dependency on them.

The selected small ISO-NPT implementation is the guarded atomic-ticket
mapped host/device coroutine. Unsupported ensembles and shapes use native
Kokkos behavior. The live runtime was switched separately to the validated
adopted small-NPT coroutine engine. The source-built unified stage remains
the
reproducible production artifact because its bounded five-case timing showed a
small positive geometric mean but slightly lower MgO-small gain than the live
artifact; it was not substituted for the faster live binary composition.


The periodic device restart path includes an optional guarded bulk ABI in the
locked LAMMPS source overlay. This removes redundant per-atom virtual calls after
the asynchronous transfer has already been submitted. Bounded checks cover a
small fallback, a medium batch route, and large NVT/NPT native restart replay.


The supported medium/large periodic state and native restart writers share one
immutable end-step snapshot. They retain separate exact output formats and final
completion, while unsupported and first/final output cases keep the independent
paths. The output optimization has bounded same-snapshot byte checks, trajectory
and native replay evidence; its output pressure timing is not a nine-case score.
