# Third-party notices

| Component | Pinned identity | Included material |
| --- | --- | --- |
| DeePMD-kit / DPA4C | `46fdc3e8b0c92524e85dad48ee00a8555a0a7595` | Native API/operator and LAMMPS plugin overlays; see original SPDX headers |
| LAMMPS | Source identity `a33449868448baf3c73d8eacdb2d329b13361696`; saved retrieval reference `9621ce9c6136874188a2f29c6659821a11d3bc47` | LAMMPS/Kokkos source overlays and Python package |
| Kokkos | Version 4.6.2 in the pinned LAMMPS archive | CUDA abort compatibility overlay and referenced interfaces |

Original license texts were recovered from the SHA256-matching pinned source
archives, not inferred from current upstream main branches. Their digests are
in `metadata/upstream-notices.json`; the full archive locks are in
`production/dpa4c-runtime/third_party/sources.lock.json`.

Upstream projects: https://github.com/deepmodeling/deepmd-kit and
https://github.com/lammps/lammps. The closest model paper is
*Universal Machine-learning Molecular Dynamics at the Speed of Empirical
Potentials*, https://arxiv.org/abs/2608.19041.

The DPA4C model, canonical graph and upstream fused/tiled operators are not
claimed as original contributions of this repository. See
`data/contribution-provenance.json` for documented project increments.
