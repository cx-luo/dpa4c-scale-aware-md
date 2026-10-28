# Publishing this prepared snapshot

Suggested repository name: **dpa4c-scale-aware-md**.
Suggested description: **Scale-aware DPA4C molecular dynamics on PPU: runtime
source, reproducible analysis, and paper artifacts.**
Suggested topics: `molecular-dynamics`, `dpa4c`, `deepmd`, `lammps`, `kokkos`,
`scientific-machine-learning`, `performance`, `reproducibility`.

1. Review the root README, source attribution and `LICENSE.md`; finalize the
   license for project-specific work. Do not replace upstream licenses with a
   GitHub-generated MIT license.
2. Upload only the contents of this repository folder, including dotfiles.
   Do not upload the surrounding preparation directory or private audit files.
3. Use Git or GitHub Desktop for the initial import so the whole tree and
   dotfiles are included. Set executable bits for the Bash scripts as needed;
   the documented `bash ...` entry point also works without them.
4. Run the CPU checks. Native build/PPU validation remain separate operations.
5. Publish the two large, hash-locked upstream source archives as release assets
   if redistributing them, together with their original notices. Keep them out
   of ordinary Git commits. Users can stage downloaded assets with
   `scripts/stage_dependencies.py`.
6. A suggested first tag is `v0.1.0-preprint`. Record the GitHub commit/tag in
   the paper's artifact statement after publication. Add the actual repository
   URL and preprint identifier only when they exist.

This preparation did not create a GitHub repository, Git commit, tag or release.
The archived development score must not be described as a result of a fresh
build or independent third-party validation.
