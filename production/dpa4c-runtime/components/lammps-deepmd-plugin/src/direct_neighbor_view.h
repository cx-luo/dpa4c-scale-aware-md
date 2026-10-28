#pragma once
#include <cstdint>
// Experiment-local ABI. Coordinates and indices remain owned by LAMMPS.
// A fresh view is bound on every build() call; it never stores forces or answers.
struct DpaDirectNeighborView {
  const double* x;
  const int* model_to_atom;
  const int* candidate_atom;
  const std::uint32_t* source;
  std::int64_t stride_atom,stride_xyz,node_count;
  double cutoff_squared,inverse_distance_unit;
};
extern "C" void dpa_bind_direct_neighbor_view(const DpaDirectNeighborView*);
extern "C" DpaDirectNeighborView dpa_get_direct_neighbor_view();
