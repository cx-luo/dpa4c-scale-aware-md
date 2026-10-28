# Validation procedure

The independent stage passed a complete clean Release build, static loader/ABI
checks, all fifteen behavior gates and the five-case finite A-B-B-A screen on
2026-10-13. The live runtime was switched separately to the validated adopted
engine. The source-built stage below is an independent reproducible artifact;
it is not substituted for the slightly faster live binary composition.

```bash
./scripts/verify_runtime.sh --stage-dir <stage-dir>
```

For a fresh build, one command runs the static checks, all five behavior gates,
the bounded A-B-B-A screen, and the final hash-bound verification:

```bash
./scripts/verify_runtime.sh --run-gates \
  --stage-dir <stage-dir> \
  --assets /workspace/dpa-clean/test-assets \
  --control-engine <control-engine> \
  --evidence-dir /workspace/dpa4c-production-validation/final
```

The control engine and organizer assets are explicit validation inputs. They
are not inputs to the production build or runtime.

Static checks verify the source naming boundary, absence of executable-memory
mutation, complete owned libraries, engine path resolution, every manifest
hash, external library identity, ldd resolution, SONAME/RPATH/RUNPATH and the
public ABI. Raw force and forward symbols have exactly one owner. The verifier
permits only the exact owner sets of nine audited interposition or plugin-entry
symbols. Other repeated strong exports fail the gate.

```bash
/opt/dpa-production-venv/bin/python tests/validate_behavior.py \
  --assets /workspace/dpa-clean/test-assets \
  --control-engine <control-engine> \
  --stage-dir <stage-dir> \
  --output-dir /workspace/dpa4c-production-validation/behavior
/opt/dpa-production-venv/bin/python tests/benchmark_abba.py \
  --assets /workspace/dpa-clean/test-assets \
  --control-engine <control-engine> \
  --stage-dir <stage-dir> \
  --output-dir /workspace/dpa4c-production-validation/performance
```

The five default cases are Si-small, MgO-small NPT, Si-medium, CuNi-large
NVT and MgO-large NPT. The gate runs the first step, one uninterrupted
32-step trajectory and the candidate's native checkpoint at step 16 followed
by 16 resumed steps. Native
state/scalar/restart files, input and restart hashes, identities, energy,
forces, stress, unwrapped positions, velocity, cell, integrator state, neighbor
build counts and path markers are checked. Actual worker library maps bind the
run to its runtime rather than relying on configured paths alone.

Limits are inherited from the audited production gate: energy per atom
2e-5 eV, force RMSE 4e-5 eV/Angstrom, maximum force component 5e-4 eV/Angstrom,
stress 4e-4 GPa, positions 1e-4 Angstrom, velocities 1.2e-4 and integrator
state 2.05e-5. They are finite correctness checks, not a substitute for a
long NVE drift audit.

Timing uses finite A-B-B-A on those five cases with identical native inputs,
ensemble, timestep, skin, thermo and production dump cadence. Warmup/measurement
steps are 512/8192 for small, 128/1024 for medium and 64/256 for large. Timing
comes from the existing parent-controlled worker runner. A five-percent time
regression is an explicit engineering stop threshold; raw repeats and equal
neighbor counts remain visible in evidence. No nine-case score is computed,
no 27-interval audit is run and no official evaluation is submitted.

Only lightweight JSON, log identities, concise path markers, ELF/ABI evidence
and the runtime/source manifests enter the package. Full trajectories, tensor
dumps, profiler traces, object files and build caches remain outside it.

## Recorded small-NPT coroutine integration results

The clean Release build rebuilt LAMMPS/Kokkos, DeePMD, the DPA chain and every
runtime plugin from the canonical source. The generated runtime contained 39
manifested files after validation. Static source, loader, ABI, SONAME/RPATH,
owned-dependency and hash checks all passed. The unvalidated pair-to-NPT
early-arm ABI is absent.

- Stage: `<stage-dir>`
- Candidate engine SHA256: `ad01177b4a52804e5a1169c580ebb245f56c0a95a0a314b0d8261ff74c53812f`
- Control engine SHA256: `82bb8f3c615f9f2e5a032b06be05a890f4db8f90b00d5fdc65ca1a292d58f5bf`
- NPT plugin SHA256: `f5c1407215159497e02b35305739c33f30889648aa4e82501edcc075468ef0b2`
- Model SHA256: `6ab078d8b5f2a3e6ed2adbfefefa5e4be2244bc2c83f7db58bef126657d342e7`
- Build-log SHA256: `14d3d93d7aba7b4d75c7c28c5267df3d1c262a4f8efd7a5d948b877f1cf34bdc`

All fifteen behavior comparisons passed. Across the five cases and the first
step, uninterrupted 32 steps, and native 16+16 restart replay, the worst
observed values were:

| Metric | Maximum |
|---|---:|
| energy error per atom | `1.7030e-08 eV` |
| force RMSE | `1.2404e-06 eV/Angstrom` |
| maximum force component error | `7.8679e-06 eV/Angstrom` |
| maximum stress error | `8.7594e-07 GPa` |
| maximum unwrapped-position error | `1.3330e-07 Angstrom` |
| maximum velocity error | `1.4452e-05` |
| maximum integrator-state error | `3.4709e-07` |

MgO-small route evidence recorded `active=1` and `fused=1`, with the legacy
cross-step route inactive. Every A/B pair had equal neighbor-build counts.

## Finite timing comparison

| Case | control ms/step | candidate ms/step | throughput change | time change |
|---|---:|---:|---:|---:|
| Si-small | 0.088745 | 0.089255 | -0.571% | +0.574% |
| MgO-small | 0.123893 | 0.120751 | **+2.602%** | **-2.536%** |
| Si-medium | 0.532831 | 0.536009 | -0.593% | +0.596% |
| CuNi-large | 6.392041 | 6.420130 | -0.438% | +0.439% |
| MgO-large | 7.296556 | 7.332184 | -0.486% | +0.488% |

The five-case geometric-mean throughput change is **+0.095%**. All cases stay
well inside the five-percent engineering stop threshold. The adopted live
artifact has a stronger independent eight-repeat MgO-small result of
**+3.253% throughput** with identical neighbor counts, so the live engine keeps
that validated binary composition while this stage supplies the clean,
reproducible source build.

These bounded results are not a nine-case score, a 27-interval audit or an
official submission. They establish correctness, route selection, loader/ABI
integrity and absence of an obvious production-consolidation regression.
