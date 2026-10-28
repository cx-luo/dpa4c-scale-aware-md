#include "direct_neighbor_view.h"
// SPDX-License-Identifier: LGPL-3.0-or-later
// The compact canonical graph builder is available when the LAMMPS Kokkos
// package is enabled.
#ifdef LMP_KOKKOS

#ifndef LMP_COMPACT_CANONICAL_GRAPH_KOKKOS_H
#define LMP_COMPACT_CANONICAL_GRAPH_KOKKOS_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <type_traits>
#include <vector>

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "transpose_helper_kokkos.h"
#include "neighbor.h"
#include "pointers.h"

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace LAMMPS_NS {

// Device-resident model node set and compact canonical neighbor graph.
//
// A ``.pt2`` artifact frozen with the compact canonical lower consumes a
// dual-CSR topology over a contiguous range of model nodes: uint32 source
// indices, float32 edge vectors, an int64 row pointer per node in each
// direction, and a permutation that reorders the destination-major edge list
// into source-major order. This component derives all of it from the Kokkos
// device neighbor list and leaves it on the device, so a pair style hands the
// model raw device pointers and never stages the graph through the host.
//
// The node set is the model's view of the atoms. Atom types that map to no
// model element (NULL in ``pair_coeff``) are virtual: they carry no node, so
// the surviving atoms are compacted into a contiguous node index range, which
// degenerates to the identity when no type maps to NULL. A single rank folds
// every ghost onto the local atom that owns it and works on the minimum-image
// node set, which requires the box to be thicker than twice the cutoff along
// every periodic direction. Domain decomposition instead gives every real
// ghost a node of its own -- the extended node set -- and leaves the pair
// style to fold the ghost outputs onto their owners by reverse communication.
//
// build() is one function, and public, because CUDA forbids extended device
// lambdas inside non-public members: its stages cannot be split into private
// helpers.
template <class DeviceType>
class CompactCanonicalGraphKokkos : protected Pointers {
 public:
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_neighbors_2d_lr d_transposed_candidates;
  const void* cached_neighbor_data = nullptr;
  const NeighList* cached_neighbor_list = nullptr;
  // Experimental ABI: UINT32_MAX marks an inactive candidate; consumers
  // skip it before using it as a node index. All real indices remain intact.
  Kokkos::View<int*, DeviceType> d_candidate_atom;
  bool candidate_topology_valid = false;
  // Epoch of the LAMMPS candidate layout, independent of coordinate updates.
  std::uint64_t narrow_topology_epoch = 0;
  bool candidate_abi_checked = false;
  const void* candidate_neighbor_data = nullptr;
  const NeighList* candidate_neighbor_list = nullptr;
  int candidate_nodes = -1;
  int candidate_nall = -1;
  bool candidate_map_complete = false;
  // The fixed direct DPA4C consumer only walks destination rows.  When its
  // operator-side readiness query succeeds, the source-major CSR is a dead
  // materialization and the pair passes typed aliases for the unused inputs.
  // If the direct consumer is not ready, build() keeps the full canonical
  // graph so the exported-model fallback remains valid.
  bool destination_only_graph = false;

  CompactCanonicalGraphKokkos(class LAMMPS* lmp)
      : Pointers(lmp),
        nloc_model(0),
        nnode_model(0),
        has_null_types(false),
        storage_count(0),
        execution_space(ExecutionSpaceFromDevice<DeviceType>::space),
        extended_nodes(false),
        edge_capacity(0)
#ifdef KOKKOS_ENABLE_CUDA
        ,
        h_edge_count_pinned(nullptr),
        edge_count_ready(nullptr)
#endif
  {}

  ~CompactCanonicalGraphKokkos() {
#ifdef KOKKOS_ENABLE_CUDA
    if (h_edge_count_pinned != nullptr) {
      cudaFreeHost(h_edge_count_pinned);
      h_edge_count_pinned = nullptr;
    }
    if (edge_count_ready != nullptr) {
      cudaEventDestroy(edge_count_ready);
      edge_count_ready = nullptr;
    }
#endif
  }

  // Cache the LAMMPS type (1-based) -> model type map on the device and select
  // the node-set mode; ``extended`` gives every real ghost a node of its own.
  // The map comes from the pair style's coeff(), so this runs once the styles
  // know their model, before the first graph is built.
  void setup(const std::vector<int>& type_idx_map, bool extended) {
    host_type_map = type_idx_map;
    extended_nodes = extended;
    has_null_types = false;
    const int ntypes = static_cast<int>(host_type_map.size());
    d_type_map =
        Kokkos::View<int*, DeviceType>("compact_canonical:type_map", ntypes);
    auto h_type_map = Kokkos::create_mirror_view(d_type_map);
    for (int t = 0; t < ntypes; ++t) {
      if (host_type_map[t] < 0) {
        has_null_types = true;  // some LAMMPS type is a virtual (NULL) atom
      }
      h_type_map(t) = host_type_map[t];
    }
    Kokkos::deep_copy(d_type_map, h_type_map);
  }

  // Rebuild the atom -> node compaction, which only moves when the neighbor
  // list is rebuilt or the atom count outgrows the maps. A pair style that
  // builds its own graph over the same node set calls this directly; build()
  // calls it for the compact canonical graph.
  void refresh_nodes() {
    const int nlocal = atom->nlocal;
    const int nall = atom->nlocal + atom->nghost;

    if (neighbor->ago != 0 && (int)k_loc2model.extent(0) >= nall) {
      return;
    }

    // The model CUDA graph keys its direct-neighbor view by the backing
    // pointers below.  Allocating to the current ghost count makes those
    // pointers move whenever a later neighbor rebuild grows nall, forcing a
    // graph destroy/capture/instantiate cycle.  LAMMPS already reserves all
    // atom arrays to atom->nmax, so use that stable capacity for the maps as
    // well.  The active ranges and topology remain nall/nnode_model.
    const char* fast_max_setting = std::getenv("DPA_LMP_FAST_MAX_ATOMS");
    const long fast_max_atoms =
        fast_max_setting ? std::atol(fast_max_setting) : 0L;
    const bool stable_small_buffers =
        fast_max_atoms > 0 && nlocal <= fast_max_atoms;
    const int map_capacity = stable_small_buffers
                                 ? std::max(nall, atom->nmax)
                                 : nall;
    if ((int)k_candidate_to_model.extent(0) < nall) {
      k_candidate_to_model = DAT::tdual_int_1d(
          "compact_canonical:candidate_to_model", map_capacity);
    }
    if ((int)k_loc2model.extent(0) < nall) {
      k_loc2model =
          DAT::tdual_int_1d("compact_canonical:loc2model", map_capacity);
      k_model2loc =
          DAT::tdual_int_1d("compact_canonical:model2loc", map_capacity);
    }
    atomKK->sync(Host, TAG_MASK | TYPE_MASK);
    auto h_loc2model = k_loc2model.view_host();
    auto h_model2loc = k_model2loc.view_host();
    const int* lmp_type = atom->type;
    int m = 0;
    for (int i = 0; i < nlocal; ++i) {
      if (host_type_map[lmp_type[i] - 1] >= 0) {
        h_loc2model(i) = m;
        h_model2loc(m) = i;
        ++m;
      } else {
        h_loc2model(i) = -1;
      }
    }
    nloc_model = m;
    for (int j = nlocal; j < nall; ++j) {
      if (extended_nodes && host_type_map[lmp_type[j] - 1] >= 0) {
        h_loc2model(j) = m;
        h_model2loc(m) = j;
        ++m;
      } else {
        h_loc2model(j) = -1;
      }
    }
    nnode_model = m;

    // Resolve each candidate atom to its model node once, on the host. In the
    // folded representation a ghost contributes to the node of the local atom
    // that owns it, so the resolution is a composition of the ownership map
    // with the model map; the extended representation gives ghosts their own
    // nodes and the composition degenerates to the model map. Collapsing it
    // here leaves the device traversal, which visits every candidate of every
    // center, with a single gather.
    auto h_candidate_to_model = k_candidate_to_model.view_host();
    candidate_map_complete = true;
    if (extended_nodes) {
      for (int j = 0; j < nall; ++j) {
        h_candidate_to_model(j) = h_loc2model(j);
        candidate_map_complete =
            candidate_map_complete && h_candidate_to_model(j) >= 0;
      }
    } else {
      for (int j = 0; j < nall; ++j) {
        const int owner = (j < nlocal) ? j : atom->map(atom->tag[j]);
        h_candidate_to_model(j) = owner < 0 ? -1 : h_loc2model(owner);
        candidate_map_complete =
            candidate_map_complete && h_candidate_to_model(j) >= 0;
      }
    }
    k_candidate_to_model.template modify<LMPHostType>();
    k_candidate_to_model.template sync<DeviceType>();
    d_candidate_to_model = k_candidate_to_model.template view<DeviceType>();
    k_loc2model.template modify<LMPHostType>();
    k_loc2model.template sync<DeviceType>();
    d_loc2model = k_loc2model.template view<DeviceType>();
    k_model2loc.template modify<LMPHostType>();
    k_model2loc.template sync<DeviceType>();
    d_model2loc = k_model2loc.template view<DeviceType>();
  }

  // Build the compact canonical graph of the current configuration from the
  // Kokkos full neighbor list. Bond vectors are the center-to-neighbor
  // displacements divided by ``dist_unit_cvt_factor``, the same conversion the
  // pair style applies to coordinates it hands the model.
  void build(class NeighList* list,
             double cutoff,
             double dist_unit_cvt_factor,
             bool allow_destination_only_consumer = false) {
    refresh_nodes();

    auto* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
    const int inum = k_list->inum;
    auto d_numneigh = k_list->d_numneigh;
    auto d_neighbors = k_list->d_neighbors;
    const char* candidate_setting = std::getenv("DPA_NEIGHBOR_CACHE_POLICY");
    const bool stable_candidates = candidate_setting && candidate_setting[0] == '1'
        && (nnode_model <= 4096 || std::getenv("DPA_DIRECT_STABLE_ALL"))
        && std::is_same<DeviceType, LMPDeviceType>::value && !extended_nodes;
    static const bool direct_raw_neighbors = [] {
      const char* value = std::getenv("DPA_DIRECT_RAW_NEIGHBORS");
      return value && value[0] == '1';
    }();
    static const long direct_raw_min_nodes = [] {
      const char* value = std::getenv("DPA_RAW_MIN_NODES");
      return value ? std::atol(value) : 4096L;
    }();

    // Select destination-only storage only when the complete consumer-ready
    // plan is available. Every other case enters the unchanged full-canonical
    // kernels below; in particular, small keeps its branch-free hot loops.
    destination_only_graph = false;
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same<DeviceType, LMPDeviceType>::value) {
      static const bool enabled = [] {
        const char* value = std::getenv("DPA_DESTINATION_ONLY_GRAPH");
        return value && value[0] == '1';
      }();
      static const long minimum = [] {
        const char* value = std::getenv("DPA_DESTINATION_ONLY_MIN_NODES");
        return value ? std::atol(value) : 16384L;
      }();
      if (allow_destination_only_consumer && enabled &&
          nnode_model >= minimum && stable_candidates &&
          candidate_map_complete && direct_raw_neighbors &&
          nnode_model > direct_raw_min_nodes &&
          !extended_nodes && !has_null_types) {
        using ready_function = int (*)(std::int64_t, std::int64_t);
        static const ready_function ready = [] {
          void* symbol =
              dlsym(RTLD_DEFAULT, "dpa_direct_destination_only_ready");
          if (!symbol) {
            if (void* handle =
                    dlopen("libdeepmd_op_pt.so", RTLD_NOW | RTLD_NOLOAD)) {
              symbol = dlsym(handle, "dpa_direct_destination_only_ready");
              dlclose(handle);
            }
          }
          return reinterpret_cast<ready_function>(symbol);
        }();
        if (!ready) {
          error->one(FLERR,
                     "Destination-only graph requires the matching direct DPA4C operator ABI");
        }
        if (ready)
          destination_only_graph = ready(nnode_model, nloc_model) == 1;
      }
    }
#endif
    const bool materialize_source_major = !destination_only_graph;
    // The direct raw-neighbor consumer recomputes exact geometry at the point
    // of use. On a topology rebuild it only needs row boundaries, the model
    // source, and the original candidate atom. Do not scan coordinates or
    // materialize edge vectors that this plan will never read.
    const bool consumer_ready_candidates = destination_only_graph;
    if (!stable_candidates) candidate_topology_valid = false;
    if (stable_candidates && !candidate_abi_checked) {
      void* handle = dlopen("libdeepmd_op_pt.so", RTLD_NOW | RTLD_NOLOAD);
      using abi_function = int (*)();
      auto abi = handle ? reinterpret_cast<abi_function>(
          dlsym(handle, "deepmd_dpa4c_candidate_mask_abi")) : nullptr;
      const bool ok = abi && abi() == 1;
      if (handle) dlclose(handle);
      if (!ok) error->one(FLERR, "Candidate graph requires the matching masked DPA4C operator ABI");
      candidate_abi_checked = true;
    }
    const bool reuse_candidates = stable_candidates && candidate_topology_valid
        && neighbor->ago != 0 && candidate_neighbor_data == d_neighbors.data()
        && candidate_neighbor_list == list && candidate_nodes == nnode_model
        && candidate_nall == atom->nlocal + atom->nghost;
    narrow_topology_epoch = !reuse_candidates ? ((narrow_topology_epoch + 2) | 1) : (narrow_topology_epoch & ~std::uint64_t{1});
    // Only cache a layout transform of LAMMPS's skin candidate IDs. Distances
    // and the physical cutoff membership are still recomputed every step.
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same<DeviceType, LMPDeviceType>::value) {
      const bool shape_changed =
          d_transposed_candidates.extent(0) != d_neighbors.extent(0) ||
          d_transposed_candidates.extent(1) != d_neighbors.extent(1);
      if (shape_changed) {
        d_transposed_candidates = typename AT::t_neighbors_2d_lr(
            Kokkos::NoInit("compact_canonical:transposed_candidates"),
            d_neighbors.extent(0), d_neighbors.extent(1));
      }
      if (shape_changed || neighbor->ago == 0 ||
          cached_neighbor_data != d_neighbors.data() ||
          cached_neighbor_list != list) {
        TransposeHelperKokkos<DeviceType, typename AT::t_neighbors_2d_lr,
                              typename AT::t_neighbors_2d>(
            d_transposed_candidates, d_neighbors);
        cached_neighbor_data = d_neighbors.data();
        cached_neighbor_list = list;
      }
    }
#endif
    // Membership is recomputed from current coordinates every build. This
    // bitset only shares the result between count and fill in the same step.
    bool use_keep_bits = false;
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same<DeviceType, LMPDeviceType>::value) {
      if (!consumer_ready_candidates) {
        const std::size_t words = (d_neighbors.extent(1) + 31) / 32;
        if (d_keep_bits.extent(0) != d_neighbors.extent(0) ||
            d_keep_bits.extent(1) != words) {
          d_keep_bits = decltype(d_keep_bits)(
              Kokkos::NoInit("compact_canonical:keep_bits"),
              d_neighbors.extent(0), words);
        }
        use_keep_bits = true;
      }
    }
#endif
    auto keep_bits = d_keep_bits;
    auto transposed_candidates = d_transposed_candidates;
    const bool use_transposed_candidates =
        transposed_candidates.data() != nullptr &&
        transposed_candidates.extent(0) == d_neighbors.extent(0) &&
        transposed_candidates.extent(1) == d_neighbors.extent(1);
    auto d_ilist = k_list->d_ilist;
    auto loc2model = d_loc2model;
    auto candidate_to_model = d_candidate_to_model;
    auto model2loc = d_model2loc;
    const double cutsq = cutoff * cutoff;
    const double inv_dist = 1.0 / dist_unit_cvt_factor;
    const int node_count_int = nnode_model;
    const std::size_t node_count = static_cast<std::size_t>(node_count_int);
    const int nall = atom->nlocal + atom->nghost;

    // Types only change when the neighbor list is rebuilt (atom order / ghosts).
    if ((int)d_model_type.extent(0) < node_count_int) {
      d_model_type = Kokkos::View<std::int64_t*, DeviceType>(
          "compact_canonical:model_type", nall);
    }
    if (neighbor->ago == 0) {
      atomKK->sync(execution_space, TYPE_MASK);
      auto type = atomKK->k_type.template view<DeviceType>();
      auto type_map = d_type_map;
      auto model_type = d_model_type;
      Kokkos::parallel_for(
          "compact_canonical:node_type",
          Kokkos::RangePolicy<DeviceType>(0, node_count_int),
          KOKKOS_LAMBDA(const int m) {
            model_type(m) = type_map(type(model2loc(m)) - 1);
          });
    }

    atomKK->sync(execution_space, X_MASK);
    auto x = atomKK->k_x.template view<DeviceType>();

    if (d_destination_row_ptr.extent(0) < node_count + 1) {
      d_destination_row_ptr = Kokkos::View<std::int64_t*, DeviceType>(
          "compact_canonical:destination_row_ptr", node_count + 1);
    }
    if (materialize_source_major &&
        d_source_row_ptr.extent(0) < node_count + 1) {
      d_source_counts = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source_counts", node_count);
      d_source_row_ptr = Kokkos::View<std::int64_t*, DeviceType>(
          "compact_canonical:source_row_ptr", node_count + 1);
      d_source_cursor = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source_cursor", node_count);
    }
    if (node_count_int == 0) {
      storage_count = 0;
      return;
    }
    if (reuse_candidates) {
      update_candidate_geometry(cutoff, dist_unit_cvt_factor);
      return;
    }
    auto destination_row_ptr = d_destination_row_ptr;
    auto source_counts = d_source_counts;
    if (materialize_source_major) {
      Kokkos::parallel_for(
          "compact_canonical:zero_counts",
          Kokkos::RangePolicy<DeviceType>(0, node_count_int + 1),
          KOKKOS_LAMBDA(const int i) {
            destination_row_ptr(i) = 0;
            if (i < node_count_int) source_counts(i) = 0;
          });
    } else {
      Kokkos::parallel_for(
          "compact_canonical:zero_destination_rows",
          Kokkos::RangePolicy<DeviceType>(0, node_count_int + 1),
          KOKKOS_LAMBDA(const int i) { destination_row_ptr(i) = 0; });
    }

    // === Destination-major CSR: per-node edge count, then its prefix sum ===
    if (consumer_ready_candidates) {
      Kokkos::parallel_for(
          "consumer_ready_graph:row_counts",
          Kokkos::RangePolicy<DeviceType>(0, inum),
          KOKKOS_LAMBDA(const int ii) {
            const int i = d_ilist(ii);
            const int mi = loc2model(i);
            if (mi >= 0) destination_row_ptr(mi) = d_numneigh(i);
          });
    } else {
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same<DeviceType, LMPDeviceType>::value) {
      using count_policy = Kokkos::TeamPolicy<DeviceType>;
      using count_member = typename count_policy::member_type;
      Kokkos::parallel_for(
          "compact_canonical:warp_count", count_policy(inum, 32),
          KOKKOS_LAMBDA(const count_member& team) {
            const int i = d_ilist(team.league_rank());
            const int mi = loc2model(i);
            if (mi < 0) return;
            const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
            const int lane = team.team_rank();
            const int jnum = d_numneigh(i);
            unsigned count = 0;
            for (int base = 0; base < jnum; base += 32) {
              const int jj = base + lane;
              bool accepted = false;
              if (jj < jnum) {
                const int j = (use_transposed_candidates ? transposed_candidates(i, jj)
                                                          : d_neighbors(i, jj)) & NEIGHMASK;
                const int mj = candidate_to_model(j);
                if (mj >= 0) {
                  const double dx = x(j, 0) - xi;
                  const double dy = x(j, 1) - yi;
                  const double dz = x(j, 2) - zi;
                  accepted = stable_candidates || dx * dx + dy * dy + dz * dz < cutsq;
                  if (accepted) {
                    ++count;
                    Kokkos::atomic_fetch_add(&source_counts(mj), std::uint32_t{1});
                  }
                }
              }
              unsigned selected = accepted ? (1u << lane) : 0u;
#if defined(__CUDA_ARCH__)
              selected = __ballot_sync(0xffffffffu, accepted);
#endif
              // Keep captures identical in host/device compiler passes.
              if (lane == 0) keep_bits(i, base / 32) = selected;
            }
#if defined(__CUDA_ARCH__)
            for (int offset = 16; offset > 0; offset /= 2)
              count += __shfl_down_sync(0xffffffffu, count, offset);
#endif
            if (lane == 0) destination_row_ptr(mi) = count;
          });
    } else
#endif
    {
    Kokkos::parallel_for(
        "compact_canonical:count", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii) {
          const int i = d_ilist(ii);
          const int mi = loc2model(i);
          if (mi < 0) {
            return;
          }
          const double xi = x(i, 0);
          const double yi = x(i, 1);
          const double zi = x(i, 2);
          const int jnum = d_numneigh(i);
          std::int64_t count = 0;
          for (int jj = 0; jj < jnum; ++jj) {
            const int j = d_neighbors(i, jj) & NEIGHMASK;
            const int mj = candidate_to_model(j);
            if (mj < 0) {
              continue;
            }
            const double dx = x(j, 0) - xi;
            const double dy = x(j, 1) - yi;
            const double dz = x(j, 2) - zi;
            if (stable_candidates || dx * dx + dy * dy + dz * dz < cutsq) {
              ++count;
              Kokkos::atomic_fetch_add(&source_counts(mj), std::uint32_t{1});
            }
          }
          destination_row_ptr(mi) = count;
        });

    }
    }

    Kokkos::parallel_scan(
        "compact_canonical:destination_scan",
        Kokkos::RangePolicy<DeviceType>(0, node_count_int),
        KOKKOS_LAMBDA(const int node, std::int64_t& update, const bool final) {
          const std::int64_t count = destination_row_ptr(node);
          if (final) {
            destination_row_ptr(node) = update;
          }
          update += count;
          if (final && node == node_count_int - 1) {
            destination_row_ptr(node_count_int) = update;
          }
        });
    // Upper bound: every skin neighbor can become a cutoff edge. Size the
    // buffers from that so the host does not wait on the prefix-sum D2H
    // before fill can start.
    const std::size_t neigh_bound =
        static_cast<std::size_t>(inum) *
            static_cast<std::size_t>(d_neighbors.extent(1)) +
        static_cast<std::size_t>(min_storage_edges);
    if (edge_capacity < neigh_bound) {
      edge_capacity = neigh_bound;
      d_source = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source", edge_capacity);
      d_edge_vec = Kokkos::View<float*, DeviceType>(
          "compact_canonical:edge_vec", edge_capacity * 3);
    }
    if (materialize_source_major &&
        d_source_order.extent(0) < edge_capacity) {
      d_source_order = Kokkos::View<std::uint32_t*, DeviceType>(
          "compact_canonical:source_order", edge_capacity);
    }
    if (stable_candidates && d_candidate_atom.extent(0) < edge_capacity) {
      d_candidate_atom = Kokkos::View<int*, DeviceType>(
          Kokkos::NoInit("candidate_canonical:atom_index"), edge_capacity);
    }
    std::int64_t edge_count = 0;
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same<DeviceType, LMPDeviceType>::value) {
      if (h_edge_count_pinned == nullptr) {
        cudaMallocHost(&h_edge_count_pinned, sizeof(std::int64_t));
        cudaEventCreateWithFlags(&edge_count_ready, cudaEventDisableTiming);
      }
      cudaMemcpyAsync(h_edge_count_pinned,
                      d_destination_row_ptr.data() + node_count_int,
                      sizeof(std::int64_t), cudaMemcpyDeviceToHost,
                      DeviceType().cuda_stream());
      cudaEventRecord(edge_count_ready, DeviceType().cuda_stream());
    } else {
      Kokkos::deep_copy(edge_count,
                        Kokkos::subview(d_destination_row_ptr, node_count_int));
    }
#else
    Kokkos::deep_copy(edge_count,
                      Kokkos::subview(d_destination_row_ptr, node_count_int));
#endif

    // Source counts are already in from the destination walk, so the
    // source-major prefix can run before fill. Fill then writes source_order
    // in the same launch and the separate scatter kernel goes away.
    auto source_row_ptr = d_source_row_ptr;
    auto source_cursor = d_source_cursor;
    if (materialize_source_major) {
      Kokkos::parallel_scan(
          "compact_canonical:source_scan",
          Kokkos::RangePolicy<DeviceType>(0, node_count_int),
          KOKKOS_LAMBDA(const int node, std::int64_t& update, const bool final) {
            const std::int64_t count =
                static_cast<std::int64_t>(source_counts(node));
            if (final) {
              source_row_ptr(node) = update;
              source_cursor(node) = static_cast<std::uint32_t>(update);
            }
            update += count;
            if (final && node == node_count_int - 1) {
              source_row_ptr(node_count_int) = update;
            }
          });
    }

    auto source = d_source;
    auto edge_vec = d_edge_vec;
    auto source_order = d_source_order;
    auto candidate_atom = d_candidate_atom;
    // === Destination-major fill, one warp per center ===
    // Compact survivors with a warp ballot instead of a Kokkos team scan plus
    // shared scratch. Edge order stays the candidate order (low lane first).
    constexpr int neighbor_lanes = 32;
    using team_policy = Kokkos::TeamPolicy<DeviceType>;
    using member_type = typename team_policy::member_type;
    if (consumer_ready_candidates) {
      Kokkos::parallel_for(
          "consumer_ready_graph:fill_topology",
          Kokkos::Experimental::require(
              team_policy(inum, neighbor_lanes),
              Kokkos::Experimental::WorkItemProperty::HintLightWeight),
          KOKKOS_LAMBDA(const member_type& team) {
            const int i = d_ilist(team.league_rank());
            const int mi = loc2model(i);
            if (mi < 0) return;
            const int jnum = d_numneigh(i);
            const std::int64_t begin = destination_row_ptr(mi);
            for (int jj = team.team_rank(); jj < jnum;
                 jj += team.team_size()) {
              const int j =
                  (use_transposed_candidates ? transposed_candidates(i, jj)
                                               : d_neighbors(i, jj)) &
                  NEIGHMASK;
              const int mj = candidate_to_model(j);
              const std::int64_t position = begin + jj;
              source(position) = static_cast<std::uint32_t>(mj);
              candidate_atom(position) = j;
            }
          });
    } else {
      Kokkos::parallel_for(
          "compact_canonical:fill",
          Kokkos::Experimental::require(
              team_policy(inum, neighbor_lanes),
              Kokkos::Experimental::WorkItemProperty::HintLightWeight),
          KOKKOS_LAMBDA(const member_type& team) {
          const int i = d_ilist(team.league_rank());
          const int mi = loc2model(i);
          if (mi < 0) {
            return;
          }
          const double xi = x(i, 0);
          const double yi = x(i, 1);
          const double zi = x(i, 2);
          const int jnum = d_numneigh(i);
          const int lane = team.team_rank();
          std::int64_t edge = destination_row_ptr(mi);
          for (int base = 0; base < jnum; base += neighbor_lanes) {
            const int jj = base + lane;
            int mj = -1;
            float vx = 0.0f;
            float vy = 0.0f;
            float vz = 0.0f;
            const unsigned selected = use_keep_bits ? keep_bits(i, base / 32) : 0u;
            if (jj < jnum && (!use_keep_bits || ((selected >> lane) & 1u))) {
              const int j = (use_transposed_candidates ? transposed_candidates(i, jj)
                                                        : d_neighbors(i, jj)) & NEIGHMASK;
              mj = candidate_to_model(j);
              if (mj >= 0) {
                const double dx = x(j, 0) - xi;
                const double dy = x(j, 1) - yi;
                const double dz = x(j, 2) - zi;
                if (stable_candidates || use_keep_bits || dx * dx + dy * dy + dz * dz < cutsq) {
                  vx = static_cast<float>(dx * inv_dist);
                  vy = static_cast<float>(dy * inv_dist);
                  vz = static_cast<float>(dz * inv_dist);
                } else {
                  mj = -1;
                }
              }
            }
#if defined(__CUDA_ARCH__)
            const unsigned keep =
                __ballot_sync(0xffffffffu, mj >= 0);
            const int prefix =
                __popc(keep & ((1u << lane) - 1u));
            const int kept = __popc(keep);
            if (mj >= 0) {
              const std::int64_t position = edge + prefix;
              source(position) = static_cast<std::uint32_t>(mj);
              if (stable_candidates) candidate_atom(position) =
                  (use_transposed_candidates ? transposed_candidates(i, jj)
                                             : d_neighbors(i, jj)) & NEIGHMASK;
              edge_vec(3 * position + 0) = vx;
              edge_vec(3 * position + 1) = vy;
              edge_vec(3 * position + 2) = vz;
              const auto src_pos = Kokkos::atomic_fetch_add(
                  &source_cursor(mj), std::uint32_t{1});
              source_order(src_pos) = static_cast<std::uint32_t>(position);
            }
            edge += kept;
#else
            // Host fallback: all lanes recount, lane 0 writes. Contest GPU
            // never takes this path.
            int extra = 0;
            for (int slot = 0; slot < neighbor_lanes; ++slot) {
              const int sj = base + slot;
              if (sj >= jnum) {
                break;
              }
              const int j = d_neighbors(i, sj) & NEIGHMASK;
              const int tj = candidate_to_model(j);
              if (tj < 0) {
                continue;
              }
              const double dx = x(j, 0) - xi;
              const double dy = x(j, 1) - yi;
              const double dz = x(j, 2) - zi;
              if (stable_candidates || dx * dx + dy * dy + dz * dz < cutsq) {
                if (lane == 0) {
                  const std::int64_t position = edge + extra;
                  source(position) = static_cast<std::uint32_t>(tj);
                  if (stable_candidates) candidate_atom(position) = j;
                  edge_vec(3 * position + 0) =
                      static_cast<float>(dx * inv_dist);
                  edge_vec(3 * position + 1) =
                      static_cast<float>(dy * inv_dist);
                  edge_vec(3 * position + 2) =
                      static_cast<float>(dz * inv_dist);
                  const auto src_pos = Kokkos::atomic_fetch_add(
                      &source_cursor(tj), std::uint32_t{1});
                  source_order(src_pos) =
                      static_cast<std::uint32_t>(position);
                }
                ++extra;
              }
            }
            team.team_barrier();
            edge += extra;
#endif
          }
          });
    }
    // The 8-byte D2H was issued after the destination scan and has been
    // overlapped with source CSR + fill. Waiting on that event does not
    // stall for the graph kernels; the model still waits via the stream
    // handoff.
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same<DeviceType, LMPDeviceType>::value) {
      cudaEventSynchronize(edge_count_ready);
      edge_count = *h_edge_count_pinned;
    }
#endif
    storage_count = std::max<std::int64_t>(edge_count, min_storage_edges);
    // Fixed storage shapes permit graph replay across neighbor rebuilds.
    // The device CSR still bounds physical candidate edges; padding is zeroed.
    const char* fixed_storage=std::getenv("DPA_FIXED_STORAGE");
    if(stable_candidates && fixed_storage && fixed_storage[0]=='1')
      storage_count=static_cast<std::int64_t>(edge_capacity);
    if (static_cast<std::uint64_t>(storage_count) >
        std::numeric_limits<std::uint32_t>::max()) {
      error->one(FLERR,
                 "Compact canonical graph exceeds the uint32 edge-index range");
    }
    if (static_cast<std::size_t>(storage_count) > edge_capacity) {
      error->one(FLERR,
                 "Compact canonical graph exceeded the neighbor-list edge bound");
    }

    // === Guard edges past the physical end of the graph ===
    if (storage_count > edge_count && !consumer_ready_candidates) {
      Kokkos::parallel_for(
          "compact_canonical:guards",
          Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(
              edge_count, storage_count),
          KOKKOS_LAMBDA(const std::int64_t edge) {
            source(edge) = std::uint32_t{0};
            edge_vec(3 * edge + 0) = 0.0f;
            edge_vec(3 * edge + 1) = 0.0f;
            edge_vec(3 * edge + 2) = 0.0f;
            source_order(edge) = static_cast<std::uint32_t>(edge);
          });
    }
    if (destination_only_graph) {
      static bool announced = false;
      if (!announced) {
        announced = true;
        std::fprintf(stderr,
                     "DESTINATION_ONLY_GRAPH active nodes=%d edges=%lld "
                     "storage=%lld elided_source_atomics=%lld "
                     "elided_edge_vec_bytes=%lld\n",
                     node_count_int, static_cast<long long>(edge_count),
                     static_cast<long long>(storage_count),
                     static_cast<long long>(2 * edge_count),
                     static_cast<long long>(12 * storage_count));
      }
    }
    if (stable_candidates) {
      candidate_neighbor_data = d_neighbors.data();
      candidate_neighbor_list = list;
      candidate_nodes = nnode_model;
      candidate_nall = nall;
      candidate_topology_valid = true;
      update_candidate_geometry(cutoff, dist_unit_cvt_factor);
    }

  }

  // === Node set, valid after refresh_nodes() ===
  int nloc_model;   // real local model nodes; the energy is summed over these
  int nnode_model;  // total model nodes (== nloc_model folded; + ghost
                    // extended)
  bool has_null_types;  // some LAMMPS type maps to no model element
  // LAMMPS type (1-based) -> model type, resident on the device.
  Kokkos::View<int*, DeviceType> d_type_map;
  DAT::tdual_int_1d k_model2loc;      // (nall) model node index -> atom index
  typename AT::t_int_1d d_loc2model;  // (nall) atom -> model node index, or -1
  typename AT::t_int_1d d_model2loc;
  // Geometry and the physical cutoff are refreshed on every call. Reuse is
  // limited to LAMMPS's valid skin topology, never to coordinates or answers.
  void update_candidate_geometry(double cutoff, double dist_unit_cvt_factor) {
    auto x = atomKK->k_x.template view<DeviceType>();
    if constexpr (std::is_same<DeviceType,LMPDeviceType>::value) {
      // Match the operator's configured raw-neighbor size policy, including
      // the production all-size setting DPA_RAW_MIN_NODES=0.
      const char* raw_min_setting = std::getenv("DPA_RAW_MIN_NODES");
      const long raw_min_nodes = raw_min_setting ? std::atol(raw_min_setting) : 4096L;
      if(std::getenv("DPA_DIRECT_RAW_NEIGHBORS") && nnode_model>raw_min_nodes) {
        using bind_type=void(*)(const DpaDirectNeighborView*);
        void* handle=dlopen("libdeepmd_op_pt.so",RTLD_NOW|RTLD_NOLOAD);
        auto bind=handle?reinterpret_cast<bind_type>(dlsym(handle,"dpa_bind_direct_neighbor_view")):nullptr;
        if(!bind)error->one(FLERR,"Raw-neighbor experiment requires matching operator");
        DpaDirectNeighborView view{x.data(),d_model2loc.data(),d_candidate_atom.data(),d_source.data(),
            static_cast<std::int64_t>(x.stride_0()),static_cast<std::int64_t>(x.stride_1()),nnode_model,
            cutoff*cutoff,1.0/dist_unit_cvt_factor};
        // Optional narrow-candidate consumer. No operator ABI is changed.
        using narrow_notify = void(*)(const void*,std::uint64_t,int,int,void*);
        static auto notify = reinterpret_cast<narrow_notify>(
            dlsym(RTLD_DEFAULT,"dpa_narrow_geometry_epoch"));
        if (notify) notify(this,narrow_topology_epoch,
                           atom->nlocal+atom->nghost,atom->nmax,
                           reinterpret_cast<void*>(DeviceType().cuda_stream()));
        bind(&view);if(handle)dlclose(handle);
        return;
      }
    }

    auto rows = d_destination_row_ptr;
    auto atom_index = d_candidate_atom;
    auto source = d_source;
    auto edge_vec = d_edge_vec;
    auto model2loc = d_model2loc;
    auto candidate_to_model = d_candidate_to_model;
    const double cutsq = cutoff * cutoff;
    const double inv_dist = 1.0 / dist_unit_cvt_factor;
    using policy = Kokkos::TeamPolicy<DeviceType>;
    using member = typename policy::member_type;
    Kokkos::parallel_for("candidate_canonical:geometry_mask", policy(nnode_model,32),
      KOKKOS_LAMBDA(const member& team) {
        const int node = team.league_rank();
        const int i = model2loc(node);
        const double xi=x(i,0), yi=x(i,1), zi=x(i,2);
        for (std::int64_t edge=rows(node)+team.team_rank(); edge<rows(node+1); edge+=32) {
          const int j=atom_index(edge);
          const double dx=x(j,0)-xi, dy=x(j,1)-yi, dz=x(j,2)-zi;
          const bool active=dx*dx+dy*dy+dz*dz<cutsq;
          source(edge)=active ? static_cast<std::uint32_t>(candidate_to_model(j))
                              : std::uint32_t{0xffffffffu};
          edge_vec(3*edge)=active ? static_cast<float>(dx*inv_dist) : 0.f;
          edge_vec(3*edge+1)=active ? static_cast<float>(dy*inv_dist) : 0.f;
          edge_vec(3*edge+2)=active ? static_cast<float>(dz*inv_dist) : 0.f;
        }
      });
  }

  // (nall) candidate atom -> model node index, or -1, with a ghost already
  // folded onto its owner when the node set is not extended.
  typename AT::t_int_1d d_candidate_to_model;

  // === Compact canonical artifact, valid after build() ===
  // The edge arrays are padded to ``storage_count`` rows: the traced program
  // declares its edge axis with a lower bound of two, so a graph with fewer
  // physical edges is completed with guard rows that carry a zero bond vector
  // on node zero.
  std::int64_t storage_count;
  Kokkos::View<std::int64_t*, DeviceType> d_model_type;  // (nnode_model)
  Kokkos::View<std::uint32_t*, DeviceType> d_source;     // (storage_count)
  Kokkos::View<float*, DeviceType> d_edge_vec;           // (3 * storage_count)
  Kokkos::View<std::int64_t*, DeviceType> d_destination_row_ptr;
  Kokkos::View<std::int64_t*, DeviceType> d_source_row_ptr;
  Kokkos::View<std::uint32_t*, DeviceType> d_source_order;

 private:
  // Shortest edge axis the traced program accepts.
  static constexpr std::int64_t min_storage_edges = 2;

  ExecutionSpace execution_space;
  std::vector<int> host_type_map;
  bool extended_nodes;
  std::size_t edge_capacity;

  DAT::tdual_int_1d k_loc2model;
  DAT::tdual_int_1d k_candidate_to_model;
  // Per-node counts are bounded by the LAMMPS neighbor limit. CSR offsets
  // remain int64 so the compact graph retains its full global edge range.
  Kokkos::View<std::uint32_t*, DeviceType> d_source_counts;
  Kokkos::View<std::uint32_t*, DeviceType> d_source_cursor;
  Kokkos::View<std::uint32_t**, Kokkos::LayoutRight, DeviceType> d_keep_bits;
#ifdef KOKKOS_ENABLE_CUDA
  std::int64_t* h_edge_count_pinned;
  cudaEvent_t edge_count_ready;
#endif
};

}  // namespace LAMMPS_NS

#endif

#endif  // LMP_KOKKOS
