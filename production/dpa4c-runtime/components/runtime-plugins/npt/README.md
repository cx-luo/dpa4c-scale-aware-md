# NPT integration component

`npt_coroutine.cpp` is the selected implementation. It registers `npt/kk`
and `npt/kk/device`, enables only under strict small single-rank fixed-target
diagonal ISO-NPT guards, and otherwise calls native `FixNPTKokkos` methods.
The implementation is the adopted atomic-ticket mapped host/device coroutine;
it launches from the NPT fix after force completion.

`npt_cross_step.cpp` is retained as the earlier cross-step semantic source and
migration reference. It is deliberately not built beside the selected source
because both files register the same LAMMPS fix styles.
