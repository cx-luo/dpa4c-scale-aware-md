#include "direct_neighbor_view.h"
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Torch bindings of the compressed degree-wise DPA4C descriptor.
//
// This translation unit owns argument validation, the scalar-width dispatch,
// and the operator registration. The kernels themselves are instantiated in
// one translation unit per scalar width.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
extern "C" __attribute__((weak)) bool dpa_chain_try_widen(long,long,const float*,const float*,double*,double*,int,int,cudaStream_t);

#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>
#include <dlfcn.h>

#include "../graph_ops.h"
#include "graph_compress_launch.h"
#include "../dpa_l2_persist.h"

bool fitting_tiled_energy_gradient_range(cudaStream_t stream,
    const FittingLayerPlan& plan, float* x, long input_width, const long* atype,
    const std::vector<torch::Tensor>& ws, const std::vector<torch::Tensor>& bs,
    const std::vector<int64_t>& resnets, const torch::Tensor& w_head,
    const torch::Tensor& b_head, const torch::Tensor& bias_atom_e, int64_t act,
    const double* seed, long n, double* e);


namespace {

using deepmd_dpa4c::Arguments;
using deepmd_dpa4c::IndexKind;
using deepmd_dpa4c::Profile;

#define DPA4C_CHECK_LAUNCH(name)                                              \
  do {                                                                        \
    const cudaError_t error = cudaGetLastError();                             \
    TORCH_CHECK(error == cudaSuccess, name, ": ", cudaGetErrorString(error)); \
  } while (0)

/// Every width the host must reproduce to validate one operator invocation.
struct Dimensions {
  int moment_width;
  int output_width;
  int degree_one;
  int coupling_records;
  int spin_channels;
};

template <int Channels, int Lmax, bool HasSpin>
Dimensions dimensions_of() {
  using P = Profile<Channels, Lmax, HasSpin>;
  return {P::MomentWidth, P::OutputWidth, P::C1,
          deepmd_dpa4c::coupling_record_count(Lmax), P::Cs};
}

template <int Channels, bool HasSpin>
Dimensions dimensions_for(int lmax) {
  switch (lmax) {
    case 2:
      return dimensions_of<Channels, 2, HasSpin>();
    case 3:
      return dimensions_of<Channels, 3, HasSpin>();
    case 4:
      return dimensions_of<Channels, 4, HasSpin>();
    default:
      TORCH_CHECK(false, "dpa4c_graph_compress: unsupported lmax ", lmax);
  }
}

Dimensions profile_dimensions(int channels, int lmax, bool has_spin) {
#define DPA4C_DIMENSIONS(width)                           \
  if (channels == width) {                                \
    return has_spin ? dimensions_for<width, true>(lmax)   \
                    : dimensions_for<width, false>(lmax); \
  }
  DPA4C_FOR_EACH_CHANNEL(DPA4C_DIMENSIONS)
#undef DPA4C_DIMENSIONS
  TORCH_CHECK(false, "dpa4c_graph_compress: unsupported channels ", channels);
}

void dispatch(int channels,
              bool backward,
              const Arguments& arguments,
              cudaStream_t stream) {
#define DPA4C_DISPATCH(width)                                    \
  if (channels == width) {                                       \
    if (backward) {                                              \
      deepmd_dpa4c::launch_backward_c##width(arguments, stream); \
    } else {                                                     \
      deepmd_dpa4c::launch_forward_c##width(arguments, stream);  \
    }                                                            \
    DPA4C_CHECK_LAUNCH("dpa4c_graph_compress");                  \
    return;                                                      \
  }
  DPA4C_FOR_EACH_CHANNEL(DPA4C_DISPATCH)
#undef DPA4C_DISPATCH
  TORCH_CHECK(false, "dpa4c_graph_compress: unsupported channels ", channels);
}

/// Immutable tensors and scalars shared by both directions of the operator.
struct Payload {
  torch::Tensor edge_index;
  torch::Tensor edge_mask;
  torch::Tensor destination_order;
  torch::Tensor destination_row_ptr;
  torch::Tensor atype;
  torch::Tensor table;
  torch::Tensor pair_film;
  torch::Tensor pair_mixing;
  torch::Tensor type_embedding;
  torch::Tensor readout_matrices;
  torch::Tensor coupling_meta;
  torch::Tensor coupling_entry;
  torch::Tensor coupling_value;
  torch::Tensor output_mean;
  torch::Tensor output_inv_std;
  // Native spin inputs. Empty together when the descriptor is spin free.
  torch::Tensor spin;
  torch::Tensor spin_pair;
  torch::Tensor spin_type;
  bool canonical;
  int64_t lmax;
  double table_stride;
  double table_max;
  double rcut;
  double eps;
  double degree_floor;
  int64_t node_begin = 0;
};

IndexKind index_kind_of(const torch::Tensor& edge_index) {
  switch (edge_index.scalar_type()) {
    case torch::kInt32:
    case torch::kUInt32:
      return IndexKind::Bits32;
    case torch::kInt64:
      return IndexKind::Bits64;
    default:
      TORCH_CHECK(false,
                  "dpa4c_graph_compress: edge indices must be int32, uint32, "
                  "or int64");
  }
}

Arguments build_arguments(const Payload& payload,
                          const torch::Tensor& edge_vec,
                          int channels,
                          const Dimensions& widths) {
  // A tiled caller passes a slice of the destination row pointer, so the run
  // length follows the slice while the type table stays system-wide.
  const long node_count = payload.destination_row_ptr.numel() - 1;
  const int type_count = static_cast<int>(payload.type_embedding.size(0));
  const int radial_modes = payload.pair_mixing.numel() == 0
                               ? 0
                               : static_cast<int>(payload.pair_mixing.size(2));

  const torch::Device device = edge_vec.device();
  for (const torch::Tensor* tensor :
       {&edge_vec, &payload.edge_index, &payload.edge_mask,
        &payload.destination_order, &payload.destination_row_ptr,
        &payload.atype, &payload.table, &payload.pair_film,
        &payload.pair_mixing, &payload.type_embedding,
        &payload.readout_matrices, &payload.coupling_meta,
        &payload.coupling_entry, &payload.coupling_value, &payload.output_mean,
        &payload.output_inv_std}) {
    TORCH_CHECK(tensor->is_cuda() && tensor->device() == device,
                "dpa4c_graph_compress: every tensor input must be a CUDA "
                "tensor on the device of edge_vec");
    TORCH_CHECK(tensor->is_contiguous(),
                "dpa4c_graph_compress: all tensor inputs must be contiguous");
  }
  TORCH_CHECK(payload.destination_order.scalar_type() ==
                  payload.edge_index.scalar_type(),
              "dpa4c_graph_compress: destination_order dtype must match "
              "edge_index");
  TORCH_CHECK(payload.canonical ||
                  payload.destination_order.numel() == edge_vec.size(0),
              "dpa4c_graph_compress: non-canonical input requires one "
              "destination_order entry per edge");
  TORCH_CHECK(payload.edge_mask.scalar_type() == torch::kBool,
              "dpa4c_graph_compress: edge_mask must be bool");
  TORCH_CHECK(
      payload.destination_row_ptr.scalar_type() == torch::kInt64 &&
          payload.atype.scalar_type() == torch::kInt64,
      "dpa4c_graph_compress: row pointers and atom types must be int64");
  TORCH_CHECK(payload.coupling_meta.scalar_type() == torch::kInt32 &&
                  payload.coupling_entry.scalar_type() == torch::kInt32,
              "dpa4c_graph_compress: the coupling layout must be int32");
  for (const torch::Tensor* tensor :
       {&payload.table, &payload.pair_film, &payload.pair_mixing,
        &payload.type_embedding, &payload.readout_matrices,
        &payload.coupling_value, &payload.output_mean,
        &payload.output_inv_std}) {
    TORCH_CHECK(tensor->scalar_type() == torch::kFloat32,
                "dpa4c_graph_compress: tables and weights must be fp32");
  }
  TORCH_CHECK(payload.lmax >= 2 && payload.lmax <= 4,
              "dpa4c_graph_compress: lmax must be 2, 3, or 4");
  TORCH_CHECK(payload.table_max >= payload.rcut,
              "dpa4c_graph_compress: table_max must cover rcut");
  // The shared mode cache is sized from a compile-time maximum and the split
  // spline row assumes an even table width, so the rank set is closed.
  TORCH_CHECK(radial_modes == 0 || radial_modes == 2 || radial_modes == 4 ||
                  radial_modes == 8,
              "dpa4c_graph_compress: radial_modes must be 0, 2, 4, or 8, got ",
              radial_modes);
  TORCH_CHECK(edge_vec.dim() == 2 && edge_vec.size(1) == 3,
              "dpa4c_graph_compress: edge_vec must have shape (E, 3)");
  TORCH_CHECK(payload.table.dim() == 2 && payload.table.size(0) > 0 &&
                  payload.table.size(1) == 6 * (channels + radial_modes),
              "dpa4c_graph_compress: invalid radial table shape");
  TORCH_CHECK(payload.type_embedding.dim() == 2 && type_count > 1,
              "dpa4c_graph_compress: invalid type embedding shape");
  TORCH_CHECK(
      payload.pair_film.sizes() ==
          torch::IntArrayRef(
              {static_cast<long>(type_count) * type_count, channels, 2}),
      "dpa4c_graph_compress: invalid PairFiLM cache shape");
  TORCH_CHECK(
      radial_modes == 0 ||
          payload.pair_mixing.sizes() ==
              torch::IntArrayRef({static_cast<long>(type_count) * type_count,
                                  channels, radial_modes}),
      "dpa4c_graph_compress: invalid mode-mixing cache shape");
  // The kernel addresses the packed projections with a fixed degree-one
  // stride, so the padded block must have exactly that extent.
  TORCH_CHECK(payload.readout_matrices.sizes() ==
                  torch::IntArrayRef({8, widths.degree_one, widths.degree_one}),
              "dpa4c_graph_compress: invalid readout matrix shape");
  TORCH_CHECK(payload.coupling_meta.dim() == 2 &&
                  payload.coupling_meta.size(1) == 8 &&
                  payload.coupling_meta.size(0) == widths.coupling_records,
              "dpa4c_graph_compress: the coupling layout must describe ",
              widths.coupling_records, " degree triples");
  TORCH_CHECK(payload.coupling_entry.numel() == payload.coupling_value.numel(),
              "dpa4c_graph_compress: coupling coordinates and values must "
              "have equal length");
  TORCH_CHECK(payload.output_mean.numel() == widths.output_width &&
                  payload.output_inv_std.numel() == widths.output_width,
              "dpa4c_graph_compress: invalid output calibration shape");
  // A run covers ``node_count`` destination rows starting at ``node_begin``;
  // the type table is system-wide because neighbor lookups index it with
  // absolute source indices. Whole-system entry points additionally require
  // the two to describe the same node axis.
  TORCH_CHECK(payload.atype.size(0) >= payload.node_begin + node_count,
              "dpa4c_graph_compress: atype does not cover the destination "
              "node window");

  // === Native spin ===
  // The three inputs are present together or not at all. ``spin`` spans the
  // absolute node axis because neighbour lookups address it with source
  // indices, and ``spin_type`` packs the four per-type scalars a node reads.
  const bool has_spin = payload.spin.dim() == 2;
  if (has_spin) {
    for (const torch::Tensor* tensor :
         {&payload.spin, &payload.spin_pair, &payload.spin_type}) {
      TORCH_CHECK(tensor->is_cuda() && tensor->device() == device &&
                      tensor->is_contiguous() &&
                      tensor->scalar_type() == torch::kFloat32,
                  "dpa4c_graph_compress: spin inputs must be contiguous fp32 "
                  "CUDA tensors on the device of edge_vec");
    }
    TORCH_CHECK(payload.spin.dim() == 2 && payload.spin.size(1) == 3 &&
                    payload.spin.size(0) == payload.atype.size(0),
                "dpa4c_graph_compress: spin must have shape (N_all, 3)");
    TORCH_CHECK(
        payload.spin_pair.sizes() ==
            torch::IntArrayRef({static_cast<long>(type_count) * type_count,
                                widths.spin_channels, 2}),
        "dpa4c_graph_compress: invalid ordered spin cache shape");
    TORCH_CHECK(payload.spin_type.sizes() ==
                    torch::IntArrayRef({static_cast<long>(type_count), 4}),
                "dpa4c_graph_compress: invalid per-type spin table shape");
  } else {
    TORCH_CHECK(
        payload.spin.dim() == 1 && payload.spin.numel() == 0 &&
            payload.spin_pair.numel() == 0 && payload.spin_type.numel() == 0,
        "dpa4c_graph_compress: absent spin must be a rank-one empty tensor, "
        "and spin tables require a spin input");
  }

  Arguments arguments;
  arguments.has_spin = has_spin;
  arguments.spin = has_spin ? payload.spin.data_ptr<float>() : nullptr;
  arguments.spin_pair =
      has_spin ? payload.spin_pair.data_ptr<float>() : nullptr;
  arguments.spin_type =
      has_spin ? payload.spin_type.data_ptr<float>() : nullptr;
  arguments.node_count = node_count;
  arguments.edge_count = edge_vec.size(0);
  arguments.lmax = static_cast<int>(payload.lmax);
  arguments.interval_count = static_cast<int>(payload.table.size(0));
  arguments.type_count = type_count;
  arguments.table_width = channels + radial_modes;
  arguments.radial_modes = radial_modes;
  arguments.coupling_count = static_cast<int>(payload.coupling_meta.size(0));
  arguments.table_stride = static_cast<float>(payload.table_stride);
  arguments.table_max = static_cast<float>(payload.table_max);
  arguments.rcut = static_cast<float>(payload.rcut);
  arguments.eps = static_cast<float>(payload.eps);
  arguments.degree_floor = static_cast<float>(payload.degree_floor);
  // The production model uses its exact cutoff without experimental radial caps.
  arguments.edge_rmax_sq = 0.0f;
  arguments.tier_inner_r2 = 0.0f;
  arguments.canonical = payload.canonical;
  arguments.index_kind = index_kind_of(payload.edge_index);
  arguments.edge_index = payload.edge_index.data_ptr();
  arguments.destination_order = payload.destination_order.numel() != 0
                                    ? payload.destination_order.data_ptr()
                                    : nullptr;
  arguments.edge_vec = edge_vec.data_ptr<float>();
  arguments.edge_mask = payload.edge_mask.numel() != 0
                            ? payload.edge_mask.data_ptr<bool>()
                            : nullptr;
  arguments.destination_row_ptr = payload.destination_row_ptr.data_ptr<long>();
  arguments.node_begin = payload.node_begin;
  arguments.atype = payload.atype.data_ptr<long>();
  arguments.table = payload.table.data_ptr<float>();
  arguments.pair_film = payload.pair_film.data_ptr<float>();
  arguments.pair_mixing =
      radial_modes != 0 ? payload.pair_mixing.data_ptr<float>() : nullptr;
  arguments.type_embedding = payload.type_embedding.data_ptr<float>();
  arguments.readout_matrices = payload.readout_matrices.data_ptr<float>();
  arguments.coupling_meta = payload.coupling_meta.data_ptr<int>();
  arguments.coupling_entry = payload.coupling_entry.data_ptr<int>();
  arguments.coupling_value = payload.coupling_value.data_ptr<float>();
  arguments.output_mean = payload.output_mean.data_ptr<float>();
  arguments.output_inv_std = payload.output_inv_std.data_ptr<float>();
  return arguments;
}

}  // namespace

__global__ void stage_graph_inputs(long n,const long* types,const double* seed,
    long ntypes,const double* bias,const float* head_bias,
    long* out_types,double* out_seed,double* out_bias,float* out_head_bias) {
  const long i=blockIdx.x*static_cast<long>(blockDim.x)+threadIdx.x;
  if(i<n) {out_types[i]=types[i];out_seed[i]=seed[i];}
  if(i<ntypes) out_bias[i]=bias[i];
  if(i==0 && head_bias && out_head_bias) out_head_bias[0]=head_bias[0];
}

std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
replay_stable_candidate_graph(
    cudaStream_t caller,
    const torch::Tensor& edge_vec,
    const torch::Tensor& edge_vec_float,
    const torch::Tensor& source,
    const torch::Tensor& destination_row_ptr,
    const torch::Tensor& atype,
    const torch::Tensor& table,
    const torch::Tensor& pair_film,
    const torch::Tensor& pair_mixing,
    const torch::Tensor& type_embedding,
    const torch::Tensor& readout_matrices,
    const torch::Tensor& coupling_meta,
    const torch::Tensor& coupling_entry,
    const torch::Tensor& coupling_value,
    const torch::Tensor& output_mean,
    const torch::Tensor& output_inv_std,
    const torch::Tensor& spin,
    const torch::Tensor& spin_pair,
    const torch::Tensor& spin_type,
    const torch::Tensor& seed_c,
    const std::vector<torch::Tensor>& ws,
    const std::vector<torch::Tensor>& bs,
    const std::vector<int64_t>& resnets,
    const torch::Tensor& w_head,
    const torch::Tensor& b_head,
    const torch::Tensor& bias_atom_e,
    int64_t act,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor,
    int channels,
    const Dimensions& widths,
    const FittingLayerPlan& plan,
    long node_count) {
  struct EdgeSlot {
    torch::Tensor edge_in;
    torch::Tensor source_in;
    torch::Tensor edge_gradient;
    cudaGraphExec_t exec = nullptr;
    bool warmed = false;
    bool failed = false;
  };
  struct Holder {
    cudaStream_t side = nullptr;
    cudaStream_t last_caller = nullptr;
    cudaEvent_t to_side = nullptr;
    cudaEvent_t to_caller = nullptr;
    long node_count = -1;
    torch::Tensor descriptor;
    torch::Tensor state;
    torch::Tensor saved;
    torch::Tensor scratch;
    torch::Tensor energy;
    torch::Tensor output_blob;
    torch::Tensor head_bias_in;
    torch::Tensor atom_bias_in;
    long gradient_offset = 0;
    std::vector<std::uintptr_t> identity;
    std::vector<double> scalars;
    std::vector<torch::Tensor> constants;
    torch::Tensor row_in;
    torch::Tensor atype_in;
    torch::Tensor seed_in;
    torch::Tensor empty_index;
    torch::Tensor empty_mask;
    torch::Tensor spin_gradient;
    torch::Tensor edge_spin_gradient;
    std::unordered_map<long, EdgeSlot> edges;
  };
  thread_local Holder* holder = nullptr;
  if (holder == nullptr) {
    holder = new Holder();
  }
  if (holder->side == nullptr) {
    TORCH_CHECK(cudaStreamCreateWithFlags(&holder->side, cudaStreamNonBlocking) ==
                    cudaSuccess,
                "canon graph: side stream");
    TORCH_CHECK(cudaEventCreateWithFlags(&holder->to_side, cudaEventDisableTiming) ==
                    cudaSuccess,
                "canon graph: event");
    TORCH_CHECK(cudaEventCreateWithFlags(&holder->to_caller,
                                         cudaEventDisableTiming) == cudaSuccess,
                "canon graph: event");
  }
  // Read-only model constants are owned by the cache and keyed by mutation
  // version. Dynamic types, seed and biases are refreshed by one device kernel.
  std::vector<torch::Tensor> constants{table,pair_film,pair_mixing,type_embedding,
    readout_matrices,coupling_meta,coupling_entry,coupling_value,
    output_mean,output_inv_std,w_head};
  constants.insert(constants.end(),ws.begin(),ws.end());
  constants.insert(constants.end(),bs.begin(),bs.end());
  std::vector<std::uintptr_t> identity{
    reinterpret_cast<std::uintptr_t>(edge_vec_float.data_ptr()),
    reinterpret_cast<std::uintptr_t>(source.data_ptr()),
    reinterpret_cast<std::uintptr_t>(destination_row_ptr.data_ptr()),
    static_cast<std::uintptr_t>(edge_vec_float.numel()),
    static_cast<std::uintptr_t>(node_count),
    static_cast<std::uintptr_t>(bias_atom_e.numel()),
    static_cast<std::uintptr_t>(b_head.numel()),
    static_cast<std::uintptr_t>(edge_vec.get_device())};
  for (const auto& t:constants) {
    if (!t.numel()) continue;
    const auto& version=t.unsafeGetTensorImpl()->version_counter();
    if (!version.enabled()) return std::nullopt;
    identity.push_back(reinterpret_cast<std::uintptr_t>(t.data_ptr()));
    identity.push_back(static_cast<std::uintptr_t>(version.current_version()));
    identity.push_back(static_cast<std::uintptr_t>(t.numel()));
  }
  for(auto v:resnets) identity.push_back(static_cast<std::uintptr_t>(v));
  std::vector<double> scalar_key{double(act),double(lmax),table_stride,table_max,
    rcut,eps,degree_floor,double(channels)};
  if (holder->identity != identity || holder->scalars != scalar_key) {
    if (holder->node_count >= 0) {
      cudaStreamSynchronize(holder->side);
      if(holder->last_caller) cudaStreamSynchronize(holder->last_caller);
    }
    holder->node_count=-1;
    holder->identity=std::move(identity);
    holder->scalars=std::move(scalar_key);
  }
  if (holder->node_count != node_count) {
    for (auto& item : holder->edges) {
      if (item.second.exec != nullptr) {
        cudaGraphExecDestroy(item.second.exec);
      }
    }
    holder->edges.clear();
    holder->node_count = node_count;
    holder->constants = constants;
    auto f32 = edge_vec_float.options();
    const int nslot = plan.n_layer > 1 ? 2 : 1;
    holder->descriptor = torch::empty({node_count, widths.output_width}, f32);
    holder->state = torch::empty({node_count, widths.moment_width + 2}, f32);
    holder->saved = torch::empty({node_count * plan.saved_width()}, f32);
    holder->scratch = torch::empty({nslot, node_count, plan.width_max}, f32);
    holder->gradient_offset = ((node_count * 8 + 255) / 256) * 256;
    const long bytes=holder->gradient_offset+edge_vec_float.numel()*4;
    holder->output_blob=torch::empty({bytes},edge_vec.options().dtype(torch::kUInt8));
    holder->energy=holder->output_blob.narrow(0,0,node_count*8)
        .view(torch::kFloat64).view({node_count,1});
    holder->row_in = destination_row_ptr;
    holder->head_bias_in = torch::empty_like(b_head);
    holder->atom_bias_in = torch::empty_like(bias_atom_e);
    holder->atype_in = torch::empty_like(atype);
    holder->seed_in = torch::empty_like(seed_c);
    holder->empty_index = torch::empty({0}, source.options());
    holder->empty_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
    holder->spin_gradient = torch::empty({0}, f32);
    holder->edge_spin_gradient = torch::empty({0}, f32);
  }
  if (!holder->atype_in.defined() ||
      holder->atype_in.numel() != atype.numel() ||
      holder->row_in.numel() != destination_row_ptr.numel()) {
    for (auto& item : holder->edges) {
      if (item.second.exec != nullptr) {
        cudaGraphExecDestroy(item.second.exec);
        item.second.exec = nullptr;
      }
      item.second.warmed = false;
      item.second.failed = false;
    }
    holder->row_in = destination_row_ptr;
    holder->atype_in = torch::empty_like(atype);
    holder->seed_in = torch::empty_like(seed_c);
  }
  const long edge_count = edge_vec_float.size(0);
  EdgeSlot& slot = holder->edges[edge_count];
  if (slot.failed) {
    return std::nullopt;
  }
  if (!slot.edge_in.defined()) {
    slot.edge_in = edge_vec_float;
    slot.source_in = source;
    slot.edge_gradient = holder->output_blob.narrow(0,holder->gradient_offset,
        edge_vec_float.numel()*4).view(torch::kFloat32).view(edge_vec_float.sizes());
  }

  // Replays run on the consumer stream. Only graph creation uses the
  // isolated stream required by the PPU capture runtime.
  if(holder->last_caller && holder->last_caller!=caller) {
    cudaEventRecord(holder->to_caller,holder->last_caller);
    cudaStreamWaitEvent(caller,holder->to_caller,0);
  }
  holder->last_caller=caller;
  if(slot.exec) {
    const long count=std::max<long>(node_count,bias_atom_e.numel());
    stage_graph_inputs<<<(count+255)/256,256,0,caller>>>(
      node_count,atype.data_ptr<long>(),seed_c.data_ptr<double>(),
      bias_atom_e.numel(),bias_atom_e.data_ptr<double>(),
      b_head.numel()?b_head.data_ptr<float>():nullptr,
      holder->atype_in.data_ptr<long>(),holder->seed_in.data_ptr<double>(),
      holder->atom_bias_in.data_ptr<double>(),
      holder->head_bias_in.numel()?holder->head_bias_in.data_ptr<float>():nullptr);
    DPA4C_CHECK_LAUNCH("direct stream graph input staging");
    TORCH_CHECK(cudaGraphLaunch(slot.exec,caller)==cudaSuccess,"stable graph replay failed");
    auto e=holder->energy.clone();
    auto grad=slot.edge_gradient.clone();
    return std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>{
      e,grad,holder->spin_gradient,holder->edge_spin_gradient};
  }
  cudaEventRecord(holder->to_side, caller);
  cudaStreamWaitEvent(holder->side, holder->to_side, 0);
  c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(
      holder->side, edge_vec.get_device()));
  const long stage_count=std::max<long>(node_count,bias_atom_e.numel());
  stage_graph_inputs<<<(stage_count+255)/256,256,0,holder->side>>>(
    node_count,atype.data_ptr<long>(),seed_c.data_ptr<double>(),
    bias_atom_e.numel(),bias_atom_e.data_ptr<double>(),
    b_head.numel()?b_head.data_ptr<float>():nullptr,
    holder->atype_in.data_ptr<long>(),holder->seed_in.data_ptr<double>(),
    holder->atom_bias_in.data_ptr<double>(),
    holder->head_bias_in.numel()?holder->head_bias_in.data_ptr<float>():nullptr);
  DPA4C_CHECK_LAUNCH("stable graph input staging");

  auto finish = [&](std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                               torch::Tensor>
                        value) {
    cudaEventRecord(holder->to_caller, holder->side);
    cudaStreamWaitEvent(caller, holder->to_caller, 0);
    return std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                                    torch::Tensor>>{std::move(value)};
  };
  auto result = [&] {
    // Both public outputs own independent allocations, as required by
    // the unaliased operator schema. Cached graph buffers stay internal.
    auto result_energy=holder->energy.clone();
    auto result_gradient=slot.edge_gradient.clone();
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>{
        result_energy, result_gradient, holder->spin_gradient, holder->edge_spin_gradient};
  };
  auto run_body = [&] {
    float* activation[2] = {
        holder->scratch[0].data_ptr<float>(),
        holder->scratch.size(0) > 1 ? holder->scratch[1].data_ptr<float>()
                                    : holder->scratch[0].data_ptr<float>()};
    const Payload payload{
        slot.source_in,     holder->empty_mask,
        holder->empty_index, holder->row_in,
        holder->atype_in,   table,
        pair_film,          pair_mixing,
        type_embedding,     readout_matrices,
        coupling_meta,      coupling_entry,
        coupling_value,     output_mean,
        output_inv_std,     spin,
        spin_pair,          spin_type,
        true,               lmax,
        table_stride,       table_max,
        rcut,               eps,
        degree_floor,       0};
    Arguments arguments =
        build_arguments(payload, slot.edge_in, channels, widths);
    arguments.descriptor = holder->descriptor.data_ptr<float>();
    arguments.state_out = holder->state.data_ptr<float>();
    dispatch(channels, false, arguments, holder->side);
    if (!fitting_tiled_energy_gradient_range(
            holder->side, plan, holder->descriptor.data_ptr<float>(),
            widths.output_width, holder->atype_in.data_ptr<long>(), ws, bs,
            resnets, w_head, holder->head_bias_in, holder->atom_bias_in, act,
            holder->seed_in.data_ptr<double>(), node_count,
            holder->energy.data_ptr<double>())) {
    fitting_forward_range(
        holder->side, plan, holder->descriptor.data_ptr<float>(),
        widths.output_width, holder->atype_in.data_ptr<long>(), ws, bs, resnets,
        w_head, holder->head_bias_in, holder->atom_bias_in, act, node_count,
        holder->saved.data_ptr<float>(), activation,
        holder->energy.data_ptr<double>());
    fitting_backward_range(
        holder->side, plan, holder->seed_in.data_ptr<double>(),
        holder->saved.data_ptr<float>(), ws, bs, resnets, w_head, act,
        node_count, activation[0], plan.n_layer > 1 ? activation[1] : nullptr,
        holder->descriptor.data_ptr<float>());
    }
    arguments.descriptor_gradient = holder->descriptor.data_ptr<float>();
    arguments.state = holder->state.data_ptr<float>();
    arguments.moment_gradient = holder->state.data_ptr<float>();
    arguments.edge_gradient = slot.edge_gradient.data_ptr<float>();
    arguments.clear_padding = true;
    dispatch(channels, true, arguments, holder->side);
  };

  if (slot.exec != nullptr) {
    if (cudaGraphLaunch(slot.exec, holder->side) == cudaSuccess) {
      static bool announced = false;
      if (!announced) {
        std::fprintf(stderr, "STABLE_GRAPH_REPLAY edges=%ld\n", edge_count);
        announced = true;
      }
      return finish(result());
    }
    cudaGraphExecDestroy(slot.exec);
    slot.exec = nullptr;
    slot.failed = true;
  }
  if (slot.failed || !slot.warmed) {
    run_body();
    if (!slot.failed) {
      slot.warmed = true;
      std::fprintf(stderr, "STABLE_GRAPH_WARMUP edges=%ld\n", edge_count);
    }
    return finish(result());
  }
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  const cudaError_t query = cudaStreamIsCapturing(holder->side, &capture_status);
  if (query != cudaSuccess || capture_status != cudaStreamCaptureStatusNone ||
      cudaStreamBeginCapture(holder->side, cudaStreamCaptureModeThreadLocal) !=
          cudaSuccess) {
    const cudaError_t err = cudaGetLastError();
    std::fprintf(stderr, "STABLE_GRAPH_FAILED begin %s\n", cudaGetErrorString(err));
    slot.failed = true;
    run_body();
    return finish(result());
  }
  bool threw = false;
  try {
    run_body();
  } catch (...) {
    threw = true;
  }
  cudaGraph_t graph = nullptr;
  const cudaError_t ended = cudaStreamEndCapture(holder->side, &graph);
  cudaGraphExec_t exec = nullptr;
  const bool ok = !threw && ended == cudaSuccess && graph != nullptr &&
                  cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }
  if (!ok || cudaGraphLaunch(exec, holder->side) != cudaSuccess) {
    if (exec != nullptr) {
      cudaGraphExecDestroy(exec);
    }
    const cudaError_t err = cudaGetLastError();
    std::fprintf(stderr, "STABLE_GRAPH_FAILED instantiate %s\n",
                 cudaGetErrorString(err));
    slot.failed = true;
    run_body();
    return finish(result());
  }
  slot.exec = exec;
  std::fprintf(stderr, "STABLE_GRAPH_CAPTURED edges=%ld\n", edge_count);
  return finish(result());
}

std::tuple<torch::Tensor, torch::Tensor> dpa4c_graph_compress(
    torch::Tensor edge_vec,
    torch::Tensor edge_index,
    torch::Tensor edge_mask,
    torch::Tensor destination_order,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor pair_film,
    torch::Tensor pair_mixing,
    torch::Tensor type_embedding,
    torch::Tensor readout_matrices,
    torch::Tensor coupling_meta,
    torch::Tensor coupling_entry,
    torch::Tensor coupling_value,
    torch::Tensor output_mean,
    torch::Tensor output_inv_std,
    torch::Tensor spin,
    torch::Tensor spin_pair,
    torch::Tensor spin_type,
    bool canonical,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor) {
  const Payload payload{edge_index,
                        edge_mask,
                        destination_order,
                        destination_row_ptr,
                        atype,
                        table,
                        pair_film,
                        pair_mixing,
                        type_embedding,
                        readout_matrices,
                        coupling_meta,
                        coupling_entry,
                        coupling_value,
                        output_mean,
                        output_inv_std,
                        spin,
                        spin_pair,
                        spin_type,
                        canonical,
                        lmax,
                        table_stride,
                        table_max,
                        rcut,
                        eps,
                        degree_floor};
  const long node_count = destination_row_ptr.numel() - 1;
  // The destination row pointer defines the node axis; the type table must
  // describe exactly that axis, or the two disagree on how many nodes exist.
  TORCH_CHECK(atype.size(0) == destination_row_ptr.numel() - 1,
              "dpa4c_graph_compress: atype and destination_row_ptr describe "
              "different node counts");
  const int channels = static_cast<int>(type_embedding.size(1));
  const Dimensions widths =
      profile_dimensions(channels, static_cast<int>(lmax), spin.dim() == 2);
  TORCH_CHECK(edge_vec.is_cuda(),
              "dpa4c_graph_compress: edge_vec must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(edge_vec.device());
  auto options = edge_vec.options().dtype(torch::kFloat32);
  auto descriptor = torch::empty({node_count, widths.output_width}, options);
  auto state = torch::empty({node_count, widths.moment_width + 2}, options);
  if (node_count == 0) {
    return {descriptor, state};
  }
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  Arguments arguments =
      build_arguments(payload, edge_vec_float, channels, widths);
  arguments.descriptor = descriptor.data_ptr<float>();
  arguments.state_out = state.data_ptr<float>();
  dispatch(channels, false, arguments, at::cuda::getCurrentCUDAStream());
  return {descriptor, state};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_graph_compress_backward_impl(torch::Tensor descriptor_gradient,
                                   torch::Tensor state,
                                   torch::Tensor edge_vec,
                                   torch::Tensor edge_index,
                                   torch::Tensor edge_mask,
                                   torch::Tensor destination_order,
                                   torch::Tensor destination_row_ptr,
                                   torch::Tensor atype,
                                   torch::Tensor table,
                                   torch::Tensor pair_film,
                                   torch::Tensor pair_mixing,
                                   torch::Tensor type_embedding,
                                   torch::Tensor readout_matrices,
                                   torch::Tensor coupling_meta,
                                   torch::Tensor coupling_entry,
                                   torch::Tensor coupling_value,
                                   torch::Tensor output_mean,
                                   torch::Tensor output_inv_std,
                                   torch::Tensor spin,
                                   torch::Tensor spin_pair,
                                   torch::Tensor spin_type,
                                   bool canonical,
                                   int64_t lmax,
                                   double table_stride,
                                   double table_max,
                                   double rcut,
                                   double eps,
                                   double degree_floor,
                                   bool reuse_state) {
  const Payload payload{edge_index,
                        edge_mask,
                        destination_order,
                        destination_row_ptr,
                        atype,
                        table,
                        pair_film,
                        pair_mixing,
                        type_embedding,
                        readout_matrices,
                        coupling_meta,
                        coupling_entry,
                        coupling_value,
                        output_mean,
                        output_inv_std,
                        spin,
                        spin_pair,
                        spin_type,
                        canonical,
                        lmax,
                        table_stride,
                        table_max,
                        rcut,
                        eps,
                        degree_floor};
  const bool has_spin = spin.dim() == 2;
  const long node_count = destination_row_ptr.numel() - 1;
  // The destination row pointer defines the node axis; the type table must
  // describe exactly that axis, or the two disagree on how many nodes exist.
  TORCH_CHECK(
      atype.size(0) == destination_row_ptr.numel() - 1,
      "dpa4c_graph_compress_backward: atype and destination_row_ptr describe "
      "different node counts");
  const int channels = static_cast<int>(type_embedding.size(1));
  const Dimensions widths =
      profile_dimensions(channels, static_cast<int>(lmax), has_spin);
  TORCH_CHECK(edge_vec.is_cuda(),
              "dpa4c_graph_compress_backward: edge_vec must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(edge_vec.device());
  TORCH_CHECK(state.is_cuda() && state.device() == edge_vec.device() &&
                  state.is_contiguous() &&
                  state.scalar_type() == torch::kFloat32 &&
                  state.sizes() ==
                      torch::IntArrayRef({node_count, widths.moment_width + 2}),
              "dpa4c_graph_compress_backward: invalid saved state");
  TORCH_CHECK(
      descriptor_gradient.is_cuda() &&
          descriptor_gradient.device() == edge_vec.device() &&
          descriptor_gradient.numel() == node_count * widths.output_width,
      "dpa4c_graph_compress_backward: invalid descriptor gradient");
  auto float_options = edge_vec.options().dtype(torch::kFloat32);
  // Every absent output receives its own allocation. The schema declares three
  // unannotated results, so returning one empty tensor in two slots would
  // introduce an alias the schema does not describe, which is undefined under
  // functionalization for all three inputs rather than only for spin.
  const auto absent = [&float_options] {
    return torch::empty({0}, float_options);
  };
  if (node_count == 0) {
    if (has_spin) {
      return {torch::zeros_like(edge_vec),
              torch::empty({node_count, 3}, float_options),
              torch::empty(edge_vec.sizes(), float_options)};
    }
    return {torch::zeros_like(edge_vec), absent(), absent()};
  }
  auto descriptor_gradient_float =
      descriptor_gradient.to(torch::kFloat32).contiguous();
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  auto edge_gradient = torch::empty_like(edge_vec_float);
  // The moment cotangent has exactly the layout of the saved state, so an
  // inference caller that no longer needs the state can reuse its storage.
  auto moment_gradient = reuse_state ? state : torch::empty_like(state);
  // The on-site magnetic gradient closes in the node kernel; the neighbour
  // part is emitted per edge and reduced onto source nodes by the shared edge
  // assembly, which already walks the source CSR for the conservative force.
  auto spin_gradient =
      has_spin ? torch::empty({node_count, 3}, float_options) : absent();
  auto edge_spin_gradient =
      has_spin ? torch::empty_like(edge_vec_float) : absent();
  Arguments arguments =
      build_arguments(payload, edge_vec_float, channels, widths);
  arguments.descriptor_gradient = descriptor_gradient_float.data_ptr<float>();
  arguments.state = state.data_ptr<float>();
  arguments.moment_gradient = moment_gradient.data_ptr<float>();
  arguments.edge_gradient = edge_gradient.data_ptr<float>();
  if (has_spin) {
    arguments.spin_gradient = spin_gradient.data_ptr<float>();
    arguments.edge_spin_gradient = edge_spin_gradient.data_ptr<float>();
  }
  dispatch(channels, true, arguments, at::cuda::getCurrentCUDAStream());
  return {edge_gradient.to(edge_vec.scalar_type()), spin_gradient,
          edge_spin_gradient};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_graph_compress_backward(torch::Tensor descriptor_gradient,
                              torch::Tensor state,
                              torch::Tensor edge_vec,
                              torch::Tensor edge_index,
                              torch::Tensor edge_mask,
                              torch::Tensor destination_order,
                              torch::Tensor destination_row_ptr,
                              torch::Tensor atype,
                              torch::Tensor table,
                              torch::Tensor pair_film,
                              torch::Tensor pair_mixing,
                              torch::Tensor type_embedding,
                              torch::Tensor readout_matrices,
                              torch::Tensor coupling_meta,
                              torch::Tensor coupling_entry,
                              torch::Tensor coupling_value,
                              torch::Tensor output_mean,
                              torch::Tensor output_inv_std,
                              torch::Tensor spin,
                              torch::Tensor spin_pair,
                              torch::Tensor spin_type,
                              bool canonical,
                              int64_t lmax,
                              double table_stride,
                              double table_max,
                              double rcut,
                              double eps,
                              double degree_floor) {
  return dpa4c_graph_compress_backward_impl(
      descriptor_gradient, state, edge_vec, edge_index, edge_mask,
      destination_order, destination_row_ptr, atype, table, pair_film,
      pair_mixing, type_embedding, readout_matrices, coupling_meta,
      coupling_entry, coupling_value, output_mean, output_inv_std, spin,
      spin_pair, spin_type, canonical, lmax, table_stride, table_max, rcut, eps,
      degree_floor, false);
}

std::tuple<torch::Tensor, torch::Tensor> dpa4c_canonical_compress(
    torch::Tensor edge_vec,
    torch::Tensor source,
    torch::Tensor destination_row_ptr,
    torch::Tensor atype,
    torch::Tensor table,
    torch::Tensor pair_film,
    torch::Tensor pair_mixing,
    torch::Tensor type_embedding,
    torch::Tensor readout_matrices,
    torch::Tensor coupling_meta,
    torch::Tensor coupling_entry,
    torch::Tensor coupling_value,
    torch::Tensor output_mean,
    torch::Tensor output_inv_std,
    torch::Tensor spin,
    torch::Tensor spin_pair,
    torch::Tensor spin_type,
    int64_t lmax,
    double table_stride,
    double table_max,
    double rcut,
    double eps,
    double degree_floor) {
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress: source and edge_vec must share the "
              "edge axis");
  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source.options());
  return dpa4c_graph_compress(
      edge_vec, source, edge_mask, destination_order, destination_row_ptr,
      atype, table, pair_film, pair_mixing, type_embedding, readout_matrices,
      coupling_meta, coupling_entry, coupling_value, output_mean,
      output_inv_std, spin, spin_pair, spin_type, true, lmax, table_stride,
      table_max, rcut, eps, degree_floor);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_backward_common(torch::Tensor descriptor_gradient,
                                         torch::Tensor state,
                                         torch::Tensor edge_vec,
                                         torch::Tensor source,
                                         torch::Tensor destination_row_ptr,
                                         torch::Tensor atype,
                                         torch::Tensor table,
                                         torch::Tensor pair_film,
                                         torch::Tensor pair_mixing,
                                         torch::Tensor type_embedding,
                                         torch::Tensor readout_matrices,
                                         torch::Tensor coupling_meta,
                                         torch::Tensor coupling_entry,
                                         torch::Tensor coupling_value,
                                         torch::Tensor output_mean,
                                         torch::Tensor output_inv_std,
                                         torch::Tensor spin,
                                         torch::Tensor spin_pair,
                                         torch::Tensor spin_type,
                                         int64_t lmax,
                                         double table_stride,
                                         double table_max,
                                         double rcut,
                                         double eps,
                                         double degree_floor,
                                         bool reuse_state) {
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress_backward: source and edge_vec must "
              "share the edge axis");
  auto edge_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto destination_order = torch::empty({0}, source.options());
  return dpa4c_graph_compress_backward_impl(
      descriptor_gradient, state, edge_vec, source, edge_mask,
      destination_order, destination_row_ptr, atype, table, pair_film,
      pair_mixing, type_embedding, readout_matrices, coupling_meta,
      coupling_entry, coupling_value, output_mean, output_inv_std, spin,
      spin_pair, spin_type, true, lmax, table_stride, table_max, rcut, eps,
      degree_floor, reuse_state);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_backward(torch::Tensor descriptor_gradient,
                                  torch::Tensor state,
                                  torch::Tensor edge_vec,
                                  torch::Tensor source,
                                  torch::Tensor destination_row_ptr,
                                  torch::Tensor atype,
                                  torch::Tensor table,
                                  torch::Tensor pair_film,
                                  torch::Tensor pair_mixing,
                                  torch::Tensor type_embedding,
                                  torch::Tensor readout_matrices,
                                  torch::Tensor coupling_meta,
                                  torch::Tensor coupling_entry,
                                  torch::Tensor coupling_value,
                                  torch::Tensor output_mean,
                                  torch::Tensor output_inv_std,
                                  torch::Tensor spin,
                                  torch::Tensor spin_pair,
                                  torch::Tensor spin_type,
                                  int64_t lmax,
                                  double table_stride,
                                  double table_max,
                                  double rcut,
                                  double eps,
                                  double degree_floor) {
  return dpa4c_canonical_compress_backward_common(
      descriptor_gradient, state, edge_vec, source, destination_row_ptr, atype,
      table, pair_film, pair_mixing, type_embedding, readout_matrices,
      coupling_meta, coupling_entry, coupling_value, output_mean,
      output_inv_std, spin, spin_pair, spin_type, lmax, table_stride, table_max,
      rcut, eps, degree_floor, false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_backward_inplace(torch::Tensor descriptor_gradient,
                                          torch::Tensor state,
                                          torch::Tensor edge_vec,
                                          torch::Tensor source,
                                          torch::Tensor destination_row_ptr,
                                          torch::Tensor atype,
                                          torch::Tensor table,
                                          torch::Tensor pair_film,
                                          torch::Tensor pair_mixing,
                                          torch::Tensor type_embedding,
                                          torch::Tensor readout_matrices,
                                          torch::Tensor coupling_meta,
                                          torch::Tensor coupling_entry,
                                          torch::Tensor coupling_value,
                                          torch::Tensor output_mean,
                                          torch::Tensor output_inv_std,
                                          torch::Tensor spin,
                                          torch::Tensor spin_pair,
                                          torch::Tensor spin_type,
                                          int64_t lmax,
                                          double table_stride,
                                          double table_max,
                                          double rcut,
                                          double eps,
                                          double degree_floor) {
  return dpa4c_canonical_compress_backward_common(
      descriptor_gradient, state, edge_vec, source, destination_row_ptr, atype,
      table, pair_film, pair_mixing, type_embedding, readout_matrices,
      coupling_meta, coupling_entry, coupling_value, output_mean,
      output_inv_std, spin, spin_pair, spin_type, lmax, table_stride, table_max,
      rcut, eps, degree_floor, true);
}

// Energy and edge cotangent of one compressed inference step, evaluated over
// runs of consecutive destination nodes.
//
// Destination-sorted CSR gives a run a contiguous span of the edge axis and
// assigns every edge to exactly one destination, so the runs partition the
// work rather than splitting any reduction. Folding the descriptor, the
// fitting and the descriptor backward into one operator lets a run retire its
// descriptor, its cotangent and its moment state before the next run starts,
// which leaves only the graph and the edge cotangent at system scale. Nothing
// is recomputed: the fitting seed is the ownership mask, known up front.
//
// The loop lives here rather than in Python because its trip count follows a
// dynamic node count, which export cannot trace.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_energy_force(torch::Tensor edge_vec,
                                         torch::Tensor source,
                                         torch::Tensor destination_row_ptr,
                                         torch::Tensor atype,
                                         torch::Tensor table,
                                         torch::Tensor pair_film,
                                         torch::Tensor pair_mixing,
                                         torch::Tensor type_embedding,
                                         torch::Tensor readout_matrices,
                                         torch::Tensor coupling_meta,
                                         torch::Tensor coupling_entry,
                                         torch::Tensor coupling_value,
                                         torch::Tensor output_mean,
                                         torch::Tensor output_inv_std,
                                         torch::Tensor spin,
                                         torch::Tensor spin_pair,
                                         torch::Tensor spin_type,
                                         int64_t lmax,
                                         double table_stride,
                                         double table_max,
                                         double rcut,
                                         double eps,
                                         double degree_floor,
                                         std::vector<torch::Tensor> ws,
                                         std::vector<torch::Tensor> bs,
                                         std::vector<int64_t> resnets,
                                         torch::Tensor w_head,
                                         torch::Tensor b_head,
                                         torch::Tensor bias_atom_e,
                                         int64_t act,
                                         torch::Tensor seed,
                                         int64_t tile, torch::Tensor n_node_per_frame);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_energy_gradient(torch::Tensor edge_vec,
                                         torch::Tensor source,
                                         torch::Tensor destination_row_ptr,
                                         torch::Tensor atype,
                                         torch::Tensor table,
                                         torch::Tensor pair_film,
                                         torch::Tensor pair_mixing,
                                         torch::Tensor type_embedding,
                                         torch::Tensor readout_matrices,
                                         torch::Tensor coupling_meta,
                                         torch::Tensor coupling_entry,
                                         torch::Tensor coupling_value,
                                         torch::Tensor output_mean,
                                         torch::Tensor output_inv_std,
                                         torch::Tensor spin,
                                         torch::Tensor spin_pair,
                                         torch::Tensor spin_type,
                                         int64_t lmax,
                                         double table_stride,
                                         double table_max,
                                         double rcut,
                                         double eps,
                                         double degree_floor,
                                         std::vector<torch::Tensor> ws,
                                         std::vector<torch::Tensor> bs,
                                         std::vector<int64_t> resnets,
                                         torch::Tensor w_head,
                                         torch::Tensor b_head,
                                         torch::Tensor bias_atom_e,
                                         int64_t act,
                                         torch::Tensor seed,
                                         int64_t tile) {
  static thread_local bool direct_primed=false;
  if (!direct_primed) {
    direct_primed=true;
    auto frame=torch::zeros({1},atype.options().dtype(torch::kLong));
    dpa4c_canonical_compress_energy_force(edge_vec,source,destination_row_ptr,atype,
      table,pair_film,pair_mixing,type_embedding,readout_matrices,coupling_meta,
      coupling_entry,coupling_value,output_mean,output_inv_std,spin,spin_pair,spin_type,
      lmax,table_stride,table_max,rcut,eps,degree_floor,ws,bs,resnets,w_head,b_head,
      bias_atom_e,act,seed,tile,frame);
    std::fprintf(stderr,"DPA_DIRECT_PRIME source initialization complete\n");
  }

  TORCH_CHECK(edge_vec.is_cuda(),
              "dpa4c_canonical_compress_energy_gradient: edge_vec must be a "
              "CUDA tensor");
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress_energy_gradient: source and edge_vec "
              "must share the edge axis");
  const c10::cuda::CUDAGuard device_guard(edge_vec.device());
  const long node_count = atype.size(0);
  TORCH_CHECK(destination_row_ptr.numel() == node_count + 1,
              "dpa4c_canonical_compress_energy_gradient: atype and "
              "destination_row_ptr describe different node counts");
  const int channels = static_cast<int>(type_embedding.size(1));
  const bool has_spin = spin.dim() == 2;
  const Dimensions widths =
      profile_dimensions(channels, static_cast<int>(lmax), has_spin);
  auto f32 = edge_vec.options().dtype(torch::kFloat32);
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  const char* early_graph_mode=std::getenv("DPA_STABLE_GRAPH");
  if(node_count>0 && node_count<=4096 && early_graph_mode && early_graph_mode[0]=='1') {
  auto seed_c = seed.contiguous();
  TORCH_CHECK(
      seed_c.numel() == node_count && seed_c.scalar_type() == torch::kFloat64,
      "dpa4c_canonical_compress_energy_gradient: seed must be fp64 "
      "with one entry per node");

  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(plan.n_layer > 0 && ws[0].size(0) == widths.output_width,
              "dpa4c_canonical_compress_energy_gradient: first fitting weight "
              "does not match the descriptor width");
  // FULL_TILE source integration: the public positive tile is a framework
  // scheduling hint. The fixed Nano pipeline is faster as one exact run.
  long run = node_count;
#if defined(__HGGCCC__)
  // Bound the working set of the Nano descriptor/fitting/backward pipeline.
  // Consecutive destination rows partition the work; all six FP32 GEMMs and
  // all per-node reductions are retained. Dispatch depends only on shape.
  if (channels == 8 && lmax == 2 && !has_spin && plan.n_layer == 3 &&
      widths.output_width == 70 && plan.width_max == 96) {
    run = std::min<long>(run, 524288);
  }
#endif
  const char* graph_mode=std::getenv("DPA_STABLE_GRAPH");
  if (graph_mode && graph_mode[0]=='1' && node_count<=4096 && run==node_count &&
      channels==8 && lmax==2 && !has_spin && pair_mixing.numel()==0 &&
      plan.n_layer==3 && widths.output_width==70 && plan.width_max==96 &&
      edge_vec.scalar_type()==torch::kFloat32) {
    if (auto replay=replay_stable_candidate_graph(at::cuda::getCurrentCUDAStream(),
        edge_vec,edge_vec_float,source,destination_row_ptr,atype,table,pair_film,
        pair_mixing,type_embedding,readout_matrices,coupling_meta,coupling_entry,
        coupling_value,output_mean,output_inv_std,spin,spin_pair,spin_type,
        seed_c,ws,bs,resnets,w_head,b_head,bias_atom_e,act,lmax,table_stride,
        table_max,rcut,eps,degree_floor,channels,widths,plan,node_count)) return *replay;
  }
  }
  auto energy =
      torch::empty({node_count, 1}, edge_vec.options().dtype(torch::kFloat64));
  // With no destination rows every edge slot is padding, and the loop that
  // would clear it never runs.
  auto edge_gradient = node_count == 0 ? torch::zeros_like(edge_vec_float)
                                       : torch::empty_like(edge_vec_float);
  // Every absent output receives its own allocation, so that no two of the
  // four unannotated results share storage.
  const auto absent = [&f32] { return torch::empty({0}, f32); };
  // The on-site magnetic gradient is node local, so a run writes only its own
  // rows. The neighbour part belongs to source nodes that other runs own, so
  // it is materialized over the whole edge axis and reduced once afterwards,
  // exactly like the conservative edge cotangent.
  auto spin_gradient = has_spin ? torch::empty({node_count, 3}, f32) : absent();
  auto edge_spin_gradient =
      has_spin ? (node_count == 0 ? torch::zeros_like(edge_vec_float)
                                  : torch::empty_like(edge_vec_float))
               : absent();
  if (node_count == 0) {
    return {energy, edge_gradient.to(edge_vec.scalar_type()), spin_gradient,
            edge_spin_gradient};
  }
  auto seed_c = seed.contiguous();
  TORCH_CHECK(
      seed_c.numel() == node_count && seed_c.scalar_type() == torch::kFloat64,
      "dpa4c_canonical_compress_energy_gradient: seed must be fp64 "
      "with one entry per node");

  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(plan.n_layer > 0 && ws[0].size(0) == widths.output_width,
              "dpa4c_canonical_compress_energy_gradient: first fitting weight "
              "does not match the descriptor width");
  // FULL_TILE source integration: the public positive tile is a framework
  // scheduling hint. The fixed Nano pipeline is faster as one exact run.
  long run = node_count;
#if defined(__HGGCCC__)
  // Bound the working set of the Nano descriptor/fitting/backward pipeline.
  // Consecutive destination rows partition the work; all six FP32 GEMMs and
  // all per-node reductions are retained. Dispatch depends only on shape.
  if (channels == 8 && lmax == 2 && !has_spin && plan.n_layer == 3 &&
      widths.output_width == 70 && plan.width_max == 96) {
    run = std::min<long>(run, 524288);
  }
#endif
  const int slots = plan.n_layer > 1 ? 2 : 1;
  auto descriptor = torch::empty({run, widths.output_width}, f32);
  auto state = torch::empty({run, widths.moment_width + 2}, f32);
  auto saved = torch::empty({run * plan.saved_width()}, f32);
  auto scratch = torch::empty({slots, run, plan.width_max}, f32);
  float* slot[2] = {
      scratch[0].data_ptr<float>(),
      slots > 1 ? scratch[1].data_ptr<float>() : scratch[0].data_ptr<float>()};

  auto empty_index = torch::empty({0}, source.options());
  auto empty_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto stream = at::cuda::getCurrentCUDAStream();
  for (long begin = 0; begin < node_count; begin += run) {
    const long count = std::min(run, node_count - begin);
    const Payload payload{
        source,         empty_mask,
        empty_index,    destination_row_ptr.slice(0, begin, begin + count + 1),
        atype,          table,
        pair_film,      pair_mixing,
        type_embedding, readout_matrices,
        coupling_meta,  coupling_entry,
        coupling_value, output_mean,
        output_inv_std, spin,
        spin_pair,      spin_type,
        true,           lmax,
        table_stride,   table_max,
        rcut,           eps,
        degree_floor,   begin};
    Arguments arguments =
        build_arguments(payload, edge_vec_float, channels, widths);
    arguments.descriptor = descriptor.data_ptr<float>();
    arguments.state_out = state.data_ptr<float>();
    dispatch(channels, false, arguments, stream);

    if(!fitting_tiled_energy_gradient_range(stream,plan,
        descriptor.data_ptr<float>(),widths.output_width,atype.data_ptr<long>()+begin,
        ws,bs,resnets,w_head,b_head,bias_atom_e,act,seed_c.data_ptr<double>()+begin,
        count,energy.data_ptr<double>()+begin)) {
    fitting_forward_range(stream, plan, descriptor.data_ptr<float>(),
                          widths.output_width, atype.data_ptr<long>() + begin,
                          ws, bs, resnets, w_head, b_head, bias_atom_e, act,
                          count, saved.data_ptr<float>(), slot,
                          energy.data_ptr<double>() + begin);
    // The cotangent replaces the descriptor, which the run no longer needs.
    fitting_backward_range(
        stream, plan, seed_c.data_ptr<double>() + begin,
        saved.data_ptr<float>(), ws, bs, resnets, w_head, act, count, slot[0],
        plan.n_layer > 1 ? slot[1] : nullptr, descriptor.data_ptr<float>());

    }

    arguments.descriptor_gradient = descriptor.data_ptr<float>();
    arguments.state = state.data_ptr<float>();
    arguments.moment_gradient = state.data_ptr<float>();
    arguments.edge_gradient = edge_gradient.data_ptr<float>();
    if (has_spin) {
      // Node-indexed like the energy, so the run addresses its own slice.
      arguments.spin_gradient = spin_gradient.data_ptr<float>() + begin * 3;
      arguments.edge_spin_gradient = edge_spin_gradient.data_ptr<float>();
    }
    // Only the final run reaches the reserved edge slots; its row pointer ends
    // at the last physical edge, which is exactly where the padding begins.
    arguments.clear_padding = begin + count == node_count;
    dispatch(channels, true, arguments, stream);
  }
  return {energy, edge_gradient.to(edge_vec.scalar_type()), spin_gradient,
          edge_spin_gradient};
}

namespace { thread_local DpaDirectNeighborView current_direct_view{}; }
namespace {
void dump_frame_once(const char* dir, const torch::Tensor& source,
                     const torch::Tensor& destination_row_ptr, const torch::Tensor& atype,
                     const torch::Tensor& state, const torch::Tensor& force,
                     const torch::Tensor& node_virial, long node_count) {
  static bool done = false;
  if (done || !dir || !dir[0]) return;
  done = true;
  auto view = dpa_get_direct_neighbor_view();
  std::string base(dir);
  auto save = [&](const std::string& name, const torch::Tensor& t) {
    torch::save(t.cpu(), base + "/" + name + ".pt");
  };
  save("source", source);
  save("destination_row_ptr", destination_row_ptr);
  save("atype", atype);
  save("state", state);
  save("force", force);
  save("node_virial", node_virial);
  std::vector<int64_t> meta = {view.stride_atom, view.stride_xyz, view.node_count,
                               static_cast<int64_t>(view.cutoff_squared * 1e12),
                               static_cast<int64_t>(view.inverse_distance_unit * 1e9)};
  torch::save(torch::tensor(meta), base + "/view_meta.pt");
  std::fprintf(stderr, "DPA_DUMP_FRAME wrote %s nodes=%ld\n", dir, node_count);
  std::fflush(stderr);
}
}  // namespace

extern "C" void dpa_bind_direct_neighbor_view(const DpaDirectNeighborView* v) { current_direct_view=v?*v:DpaDirectNeighborView{}; }
extern "C" DpaDirectNeighborView dpa_get_direct_neighbor_view() { return current_direct_view; }
namespace deepmd_dpa4c {
void launch_raw_forward_c8(const Arguments&,const DpaDirectNeighborView&,cudaStream_t);
void launch_prepare_geometry_c8(const long*,const DpaDirectNeighborView&,long,float4*,int*,cudaStream_t);
float* g_half_peer_delta = nullptr;
int g_force_fused = 0;
long* g_half_peer_link = nullptr;
void launch_raw_force_c8(const Arguments&,const DpaDirectNeighborView&,cudaStream_t,float*,float*);
void launch_direct_node_c8(const Arguments&,cudaStream_t);
void launch_direct_force_c8(const Arguments&,cudaStream_t,float*,float*);
}
namespace {
struct StageTimer {
  bool on = false;
  int seen = 0;
  int counted = 0;
  cudaEvent_t ev[5];
  double acc[4] = {0, 0, 0, 0};
  StageTimer() {
    const char* v = std::getenv("DPA_KTIME");
    on = v && v[0] == '1';
    if (on)
      for (auto& e : ev) cudaEventCreate(&e);
  }
  void finish(long nodes) {
    cudaEventSynchronize(ev[4]);
    if (++seen <= 5) return;
    for (int i = 0; i < 4; ++i) {
      float ms = 0;
      cudaEventElapsedTime(&ms, ev[i], ev[i + 1]);
      acc[i] += ms;
    }
    if (++counted % 20 == 0) {
      std::fprintf(stderr,
                   "DPA_KTIME nodes=%ld calls=%d forward=%.2f fitting=%.2f node=%.2f force=%.2f us\n",
                   nodes, counted, acc[0] / counted * 1e3, acc[1] / counted * 1e3,
                   acc[2] / counted * 1e3, acc[3] / counted * 1e3);
      std::fflush(stderr);
    }
  }
};
StageTimer& stage_timer() {
  static StageTimer timer;
  return timer;
}
}  // namespace
torch::Tensor fused_frame_virial(torch::Tensor,torch::Tensor);
namespace {
// Constant operands of the exported energy_force call, captured from the first
// AOTI run. Per-run temporaries of the wrapper (bias, head bias) are cloned.
using UnitSeedSetter = void (*)(const double*, long);
UnitSeedSetter unit_seed_setter() {
  static std::atomic<UnitSeedSetter> cached{nullptr};
  UnitSeedSetter setter = cached.load(std::memory_order_acquire);
  if (!setter) {
    setter = reinterpret_cast<UnitSeedSetter>(
        dlsym(RTLD_DEFAULT, "dpa_chain_set_unit_seed_range"));
    if (setter) cached.store(setter, std::memory_order_release);
  }
  return setter;
}
void register_unit_seed(const torch::Tensor& seed) {
  const UnitSeedSetter setter = unit_seed_setter();
  if (setter && seed.defined() && seed.scalar_type() == torch::kFloat64 &&
      seed.numel() > 0) {
    setter(seed.data_ptr<double>(), seed.numel());
  }
}

struct DirectEngineCache {
  bool ready = false;
  torch::Tensor table, pair_film, pair_mixing, type_embedding, readout_matrices;
  torch::Tensor coupling_meta, coupling_entry, coupling_value, output_mean, output_inv_std;
  torch::Tensor spin, spin_pair, spin_type, w_head, b_head, bias_atom_e, n_node_per_frame;
  std::vector<torch::Tensor> ws, bs;
  std::vector<int64_t> resnets;
  int64_t lmax = 0, act = 0, tile = 0;
  double table_stride = 0, table_max = 0, rcut = 0, eps = 0, degree_floor = 0;
  torch::Tensor seed;
};
DirectEngineCache& direct_engine_cache() {
  static DirectEngineCache cache;
  return cache;
}
bool direct_engine_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("DPA_DIRECT_ENGINE");
    return e && e[0] == '1';
  }();
  return on;
}
thread_local bool g_direct_call = false;
thread_local double* g_direct_energy_out = nullptr;
thread_local torch::Tensor g_direct_fp32_global_virial_owner;
thread_local const float* g_direct_fp32_global_virial = nullptr;
__device__ const int* g_uniform_widen_requested_virial = nullptr;

namespace {
using CONDITIONAL_WRITEBACKVirialGetter = const int* (*)();
using DIRECT_FP32_GLOBAL_VIRIALUseGetter = int (*)();
CONDITIONAL_WRITEBACKVirialGetter conditional_writeback_resolve_virial_getter() {
  return reinterpret_cast<CONDITIONAL_WRITEBACKVirialGetter>(
      dlsym(RTLD_DEFAULT, "dpa_chain_requested_virial_device_pointer"));
}
DIRECT_FP32_GLOBAL_VIRIALUseGetter direct_fp32_global_virial_resolve_use_getter() {
  return reinterpret_cast<DIRECT_FP32_GLOBAL_VIRIALUseGetter>(
      dlsym(RTLD_DEFAULT, "dpa_chain_use_direct_fp32_global_virial"));
}

}  // namespace

extern "C" __global__ void dpa_direct_widen_kernel(long n_force, long n_total, const float* __restrict__ force,
                                    const float* __restrict__ virial, double* __restrict__ force_out,
                                    double* __restrict__ virial_out) {
  const int* requested = g_uniform_widen_requested_virial;
  const bool need_virial = !requested || __ldg(requested) != 0;
  const long limit = need_virial ? n_total : n_force;
  for (long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; i < limit;
       i += static_cast<long>(gridDim.x) * blockDim.x) {
    if (i < n_force) force_out[i] = static_cast<double>(force[i]);
    else virial_out[i - n_force] = static_cast<double>(virial[i - n_force]);
  }
}
}  // namespace

// CUDA graph replay does not re-enter dpa_direct_energy_force on the host.  Keep
// returning the captured output address while its tensor owner remains current;
// clearing this pointer after one lookup made every replay fall back to the
// unwritten FP64 atom-virial buffer.  A key change executes the eager/capture
// body again and replaces or clears the owner before the new graph is replayed.
extern "C" __attribute__((visibility("default")))
const float* dpa_direct_fp32_global_virial_pointer() {
  return g_direct_fp32_global_virial_owner.defined()
             ? g_direct_fp32_global_virial_owner.data_ptr<float>()
             : nullptr;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dpa4c_canonical_compress_energy_force(torch::Tensor edge_vec,
                                         torch::Tensor source,
                                         torch::Tensor destination_row_ptr,
                                         torch::Tensor atype,
                                         torch::Tensor table,
                                         torch::Tensor pair_film,
                                         torch::Tensor pair_mixing,
                                         torch::Tensor type_embedding,
                                         torch::Tensor readout_matrices,
                                         torch::Tensor coupling_meta,
                                         torch::Tensor coupling_entry,
                                         torch::Tensor coupling_value,
                                         torch::Tensor output_mean,
                                         torch::Tensor output_inv_std,
                                         torch::Tensor spin,
                                         torch::Tensor spin_pair,
                                         torch::Tensor spin_type,
                                         int64_t lmax,
                                         double table_stride,
                                         double table_max,
                                         double rcut,
                                         double eps,
                                         double degree_floor,
                                         std::vector<torch::Tensor> ws,
                                         std::vector<torch::Tensor> bs,
                                         std::vector<int64_t> resnets,
                                         torch::Tensor w_head,
                                         torch::Tensor b_head,
                                         torch::Tensor bias_atom_e,
                                         int64_t act,
                                         torch::Tensor seed,
                                         int64_t tile, torch::Tensor n_node_per_frame) {
  TORCH_CHECK(edge_vec.is_cuda(),
              "dpa4c_canonical_compress_energy_force: edge_vec must be a "
              "CUDA tensor");
  TORCH_CHECK(source.dim() == 1 && source.numel() == edge_vec.size(0),
              "dpa4c_canonical_compress_energy_force: source and edge_vec "
              "must share the edge axis");
  const c10::cuda::CUDAGuard device_guard(edge_vec.device());
  const long node_count = atype.size(0);
  static const long raw_min_nodes=[]{const char* e=std::getenv("DPA_RAW_MIN_NODES");return e&&e[0]?std::atol(e):4096L;}();
  const bool raw_enabled=std::getenv("DPA_DIRECT_RAW_NEIGHBORS") && node_count>raw_min_nodes;
  const auto raw_view=dpa_get_direct_neighbor_view();
  if(raw_enabled) TORCH_CHECK(raw_view.x && raw_view.source==source.data_ptr<std::uint32_t>() && raw_view.node_count==node_count,
                             "Missing or stale raw-neighbor binding");

  TORCH_CHECK(destination_row_ptr.numel() == node_count + 1,
              "dpa4c_canonical_compress_energy_force: atype and "
              "destination_row_ptr describe different node counts");
  const int channels = static_cast<int>(type_embedding.size(1));
  const bool has_spin = spin.dim() == 2;
  const Dimensions widths =
      profile_dimensions(channels, static_cast<int>(lmax), has_spin);
  auto f32 = edge_vec.options().dtype(torch::kFloat32);
  auto edge_vec_float = edge_vec.to(torch::kFloat32).contiguous();
  auto energy = g_direct_energy_out
      ? torch::from_blob(g_direct_energy_out, {node_count, 1}, edge_vec.options().dtype(torch::kFloat64))
      : torch::empty({node_count, 1}, edge_vec.options().dtype(torch::kFloat64));
  if (raw_enabled && !g_direct_call && direct_engine_enabled()) {
    auto& c = direct_engine_cache();
    c.table = table; c.pair_film = pair_film; c.pair_mixing = pair_mixing;
    c.type_embedding = type_embedding; c.readout_matrices = readout_matrices;
    c.coupling_meta = coupling_meta; c.coupling_entry = coupling_entry;
    c.coupling_value = coupling_value; c.output_mean = output_mean;
    c.output_inv_std = output_inv_std; c.spin = spin; c.spin_pair = spin_pair;
    c.spin_type = spin_type; c.ws = ws; c.bs = bs; c.resnets = resnets;
    c.w_head = w_head; c.b_head = b_head.clone(); c.bias_atom_e = bias_atom_e.clone();
    c.n_node_per_frame = n_node_per_frame.clone();
    c.lmax = lmax; c.act = act; c.tile = tile; c.table_stride = table_stride;
    c.table_max = table_max; c.rcut = rcut; c.eps = eps; c.degree_floor = degree_floor;
    const int64_t seed_capacity = node_count + node_count / 4 + 64;
    const int64_t seed_shape[] = {seed_capacity};
    const auto seed_options = edge_vec.options().dtype(torch::kFloat64);
    c.seed = at::_ops::ones::call(
        c10::fromIntArrayRefSlow(at::IntArrayRef(seed_shape, 1)),
        c10::optTypeMetaToScalarType(seed_options.dtype_opt()),
        seed_options.layout_opt(), seed_options.device_opt(),
        seed_options.pinned_memory_opt());
    if (node_count >= 100000) register_unit_seed(c.seed);
    c.ready = true;
  }
  TORCH_CHECK(channels==8 && lmax==2 && !has_spin && pair_mixing.numel()==0 &&
      source.scalar_type()==torch::kUInt32 && edge_vec.scalar_type()==torch::kFloat32,
      "direct prototype requires canonical C8 L2 FP32 uint32 without spin/modes");
  auto force=torch::empty({node_count,3},f32);
  auto node_virial=torch::empty({node_count,3,3},f32);
  if(node_count==0) return {energy,force,node_virial,fused_frame_virial(node_virial,n_node_per_frame)};
  auto seed_c = seed.contiguous();
  TORCH_CHECK(
      seed_c.numel() == node_count && seed_c.scalar_type() == torch::kFloat64,
      "dpa4c_canonical_compress_energy_force: seed must be fp64 "
      "with one entry per node");

  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(plan.n_layer > 0 && ws[0].size(0) == widths.output_width,
              "dpa4c_canonical_compress_energy_force: first fitting weight "
              "does not match the descriptor width");
  // FULL_TILE source integration: the public positive tile is a framework
  // scheduling hint. The fixed Nano pipeline is faster as one exact run.
  long run = node_count;
#if defined(__HGGCCC__)
  // Bound the working set of the Nano descriptor/fitting/backward pipeline.
  // Consecutive destination rows partition the work; all six FP32 GEMMs and
  // all per-node reductions are retained. Dispatch depends only on shape.
  if (channels == 8 && lmax == 2 && !has_spin && plan.n_layer == 3 &&
      widths.output_width == 70 && plan.width_max == 96) {
    run = std::min<long>(run, 524288);
  }
#endif
  // The fused K3n fitting keeps no node-scale intermediates, so one run over
  // all nodes replaces a shared-memory image load and a tail per tile.
  static const bool k3n_untiled = [] {
    const char* fit = std::getenv("DPA_GENERIC_K3N_FITTING");
    const char* v = std::getenv("DPA_UNTILED_FITTING");
    return fit && fit[0] && fit[0] != '0' && v && v[0] && v[0] != '0';
  }();
  if (raw_enabled && k3n_untiled) run = node_count;
  constexpr int fused_env = 0;
  const int fused = raw_enabled && run == node_count ? fused_env : 0;
  const int slots = plan.n_layer > 1 ? 2 : 1;
  auto descriptor = torch::empty({run, widths.output_width}, f32);
  auto state = torch::empty({node_count, widths.moment_width + 2}, f32);
  Arguments force_arguments{};
  // Only the unfused fitting path reads these, so they are sized on demand.
  torch::Tensor saved;
  torch::Tensor scratch;
  float* slot[2] = {nullptr, nullptr};

  auto empty_index = torch::empty({0}, source.options());
  auto empty_mask = torch::empty({0}, edge_vec.options().dtype(torch::kBool));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto prepared_records=torch::empty({raw_enabled?source.numel():0,4},f32);
  auto prepared_counts=torch::empty({raw_enabled?node_count:0},f32.dtype(torch::kInt32));
  auto& ktimer = stage_timer();
  cudaStreamCaptureStatus kcapture = cudaStreamCaptureStatusNone;
  if (ktimer.on) cudaStreamIsCapturing(stream, &kcapture);
  const bool ktime = ktimer.on && kcapture == cudaStreamCaptureStatusNone && run == node_count;
  if (ktime) cudaEventRecord(ktimer.ev[0], stream);


  for (long begin = 0; begin < node_count; begin += run) {
    const long count = std::min(run, node_count - begin);
    const Payload payload{
        source,         empty_mask,
        empty_index,    destination_row_ptr.slice(0, begin, begin + count + 1),
        atype,          table,
        pair_film,      pair_mixing,
        type_embedding, readout_matrices,
        coupling_meta,  coupling_entry,
        coupling_value, output_mean,
        output_inv_std, spin,
        spin_pair,      spin_type,
        true,           lmax,
        table_stride,   table_max,
        rcut,           eps,
        degree_floor,   begin};
    Arguments arguments =
        build_arguments(payload, edge_vec_float, channels, widths);
    arguments.descriptor = descriptor.data_ptr<float>();
    if(raw_enabled){arguments.edge_vec=prepared_records.data_ptr<float>();arguments.edge_mask=reinterpret_cast<const bool*>(prepared_counts.data_ptr<int>());}

    arguments.state_out = state.data_ptr<float>() + begin * (widths.moment_width+2);
    if(raw_enabled){
      const DpaDirectNeighborView vw=dpa_get_direct_neighbor_view();
      if(vw.x && vw.node_count>0 && vw.stride_atom>0)
        dpa_l2_pin(stream, vw.x, size_t(vw.node_count)*size_t(vw.stride_atom)*sizeof(double), "coords");
      dpa_l2_pin(stream, arguments.table,
                 size_t(std::max(1, arguments.interval_count))*size_t(std::max(1, arguments.table_width))*sizeof(float),
                 "table");
    }
    if(raw_enabled)deepmd_dpa4c::launch_raw_forward_c8(arguments,raw_view,stream);
    else dispatch(channels, false, arguments, stream);
    if (ktime) cudaEventRecord(ktimer.ev[1], stream);

    if(!fitting_tiled_energy_gradient_range(stream,plan,
        descriptor.data_ptr<float>(),widths.output_width,atype.data_ptr<long>()+begin,
        ws,bs,resnets,w_head,b_head,bias_atom_e,act,seed_c.data_ptr<double>()+begin,
        count,energy.data_ptr<double>()+begin)) {
    if (!saved.defined()) {
      saved = torch::empty({run * plan.saved_width()}, f32);
      scratch = torch::empty({slots, run, plan.width_max}, f32);
      slot[0] = scratch[0].data_ptr<float>();
      slot[1] = slots > 1 ? scratch[1].data_ptr<float>() : slot[0];
    }
    fitting_forward_range(stream, plan, descriptor.data_ptr<float>(),
                          widths.output_width, atype.data_ptr<long>() + begin,
                          ws, bs, resnets, w_head, b_head, bias_atom_e, act,
                          count, saved.data_ptr<float>(), slot,
                          energy.data_ptr<double>() + begin);
    // The cotangent replaces the descriptor, which the run no longer needs.
    fitting_backward_range(
        stream, plan, seed_c.data_ptr<double>() + begin,
        saved.data_ptr<float>(), ws, bs, resnets, w_head, act, count, slot[0],
        plan.n_layer > 1 ? slot[1] : nullptr, descriptor.data_ptr<float>());

    }

    if (ktime) cudaEventRecord(ktimer.ev[2], stream);
    arguments.descriptor_gradient = descriptor.data_ptr<float>();
    arguments.state = arguments.state_out;
    arguments.moment_gradient = arguments.state_out;
    arguments.edge_gradient = nullptr;
    arguments.clear_padding = false;
    if (!fused) deepmd_dpa4c::launch_direct_node_c8(arguments,stream);
    if (ktime) cudaEventRecord(ktimer.ev[3], stream);
    force_arguments=arguments;
  }
  // Same-stream ordering guarantees every q_j is ready before force gathering.
  force_arguments.node_begin=0;
  force_arguments.node_count=node_count;
  force_arguments.destination_row_ptr=destination_row_ptr.data_ptr<long>();
  force_arguments.moment_gradient=state.data_ptr<float>();
  force_arguments.edge_index=source.data_ptr();
  torch::Tensor peer_delta;
  torch::Tensor peer_link;
  constexpr bool half_link = false;
  if (half_link) {
    const long tot = destination_row_ptr[node_count].item<long>();
    peer_delta = torch::empty({tot, 3}, f32);
    peer_link = torch::full({tot}, -1L, destination_row_ptr.options());
    deepmd_dpa4c::g_half_peer_delta = peer_delta.data_ptr<float>();
    deepmd_dpa4c::g_half_peer_link = peer_link.data_ptr<long>();
  } else {
    deepmd_dpa4c::g_half_peer_delta = nullptr;
    deepmd_dpa4c::g_half_peer_link = nullptr;
  }
  if(raw_enabled) {
    deepmd_dpa4c::g_force_fused = fused;
    dpa_l2_pin(stream, force_arguments.table,
               size_t(std::max(1, force_arguments.interval_count))*size_t(std::max(1, force_arguments.table_width))*sizeof(float),
               "table-force");
    deepmd_dpa4c::launch_raw_force_c8(force_arguments,raw_view,stream,force.data_ptr<float>(),node_virial.data_ptr<float>());
    deepmd_dpa4c::g_force_fused = 0;
  } else {
    deepmd_dpa4c::launch_direct_force_c8(force_arguments,stream,force.data_ptr<float>(),node_virial.data_ptr<float>());
  }
  deepmd_dpa4c::g_half_peer_delta = nullptr;
  deepmd_dpa4c::g_half_peer_link = nullptr;
  static bool block_ledger_reported = false;
  const char* block_ledger = std::getenv("DPA_BLOCK_LEDGER_DIAG");
  if (!block_ledger_reported && block_ledger && std::atoi(block_ledger) != 0 &&
      raw_enabled) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &capture);
    if (capture == cudaStreamCaptureStatusNone) {
      std::vector<int> host_counts(static_cast<std::size_t>(node_count));
      long candidate_edges = 0;
      cudaMemcpyAsync(host_counts.data(), prepared_counts.data_ptr<int>(),
                      host_counts.size() * sizeof(int), cudaMemcpyDeviceToHost,
                      stream);
      cudaMemcpyAsync(&candidate_edges,
                      destination_row_ptr.data_ptr<long>() + node_count,
                      sizeof(long), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      long active_edges = 0;
      int min_degree = node_count ? host_counts[0] : 0;
      int max_degree = min_degree;
      for (int count : host_counts) {
        active_edges += count;
        min_degree = std::min(min_degree, count);
        max_degree = std::max(max_degree, count);
      }
      std::fprintf(stderr,
                   "B_BLOCK_LEDGER nodes=%ld candidate_edges=%ld active_edges=%ld "
                   "storage_slots=%ld degree_min=%d degree_max=%d\n",
                   node_count, candidate_edges, active_edges,
                   static_cast<long>(source.numel()), min_degree, max_degree);
      block_ledger_reported = true;
    }
  }
  if (const char* dump = std::getenv("DPA_DUMP_FRAME")) {
    dump_frame_once(dump, source, destination_row_ptr, atype, state, force, node_virial, node_count);
  }
  if (ktime) {
    cudaEventRecord(ktimer.ev[4], stream);
    ktimer.finish(node_count);
  }
  if (g_direct_call) return {energy,force,node_virial,torch::Tensor()};
  return {energy,force,node_virial,fused_frame_virial(node_virial,n_node_per_frame)};
}

// The LAMMPS graph builder uses this query before omitting the source-major
// CSR.  Readiness is monotonic after the fixed model has primed the direct
// engine cache; the direct entry below has the same eligibility conditions.
extern "C" int dpa_direct_destination_only_ready(int64_t node_count,
                                                  int64_t nloc) {
  static const bool enabled = [] {
    const char* value = std::getenv("DPA_DIRECT_ENGINE");
    return value && value[0] == '1';
  }();
  auto& cache = direct_engine_cache();
  return enabled && cache.ready && node_count > 0 && nloc == node_count
             ? 1
             : 0;
}

// Returns 0 when the caller must run the exported model instead.
extern "C" int dpa_direct_energy_force(const int64_t* atype, const int64_t* row_ptr,
                                       const std::uint32_t* source, const float* edge_vec,
                                       int64_t node_count, int64_t nloc, int64_t edge_storage,
                                       double* energy_out, double* force_out, double* virial_out) {
  g_direct_fp32_global_virial = nullptr;
  auto& c = direct_engine_cache();
  if (!c.ready || nloc != node_count || node_count <= 0) return 0;
  static const bool uniform_widen_enabled = [] {
    const char* value = std::getenv("DPA_UNIFORM_WIDEN_UNIFORM_WIDEN_LIMIT");
    return value && std::atoi(value) != 0;
  }();
  static const long uniform_widen_min_nodes = [] {
    const char* value = std::getenv("DPA_UNIFORM_WIDEN_MIN_NODES");
    return value && value[0] ? std::atol(value) : 65536L;
  }();
  const int* uniform_widen_requested = nullptr;
  if (uniform_widen_enabled && node_count >= uniform_widen_min_nodes) {
    using Getter = const int* (*)();
    static const Getter getter = conditional_writeback_resolve_virial_getter();
    static const int* installed = nullptr;
    uniform_widen_requested = getter ? getter() : nullptr;
    if (uniform_widen_requested && uniform_widen_requested != installed) {
      DPA4C_CHECK_LAUNCH("uniform_widen pre-symbol state");
      const cudaError_t status = cudaMemcpyToSymbol(
          g_uniform_widen_requested_virial, &uniform_widen_requested, sizeof(uniform_widen_requested));
      TORCH_CHECK(status == cudaSuccess, "uniform_widen requested-virial binding: ",
                  cudaGetErrorString(status));
      installed = uniform_widen_requested;
      std::fprintf(stderr, "UNIFORM_WIDEN_UNIFORM_WIDEN_LIMIT active flag=%p\n",
                   static_cast<const void*>(uniform_widen_requested));
    }
  }
  const auto device = c.table.device();
  const auto opt = torch::TensorOptions().device(device);
  auto atype_t = torch::from_blob(const_cast<int64_t*>(atype), {node_count}, opt.dtype(torch::kInt64));
  auto row_t = torch::from_blob(const_cast<int64_t*>(row_ptr), {node_count + 1}, opt.dtype(torch::kInt64));
  auto source_t = torch::from_blob(const_cast<std::uint32_t*>(source), {edge_storage}, opt.dtype(torch::kUInt32));
  auto edge_t = torch::from_blob(const_cast<float*>(edge_vec), {edge_storage, 3}, opt.dtype(torch::kFloat32));
  if (!c.seed.defined() || c.seed.numel() < node_count) {
    c.seed = torch::ones({node_count + node_count / 4 + 64}, opt.dtype(torch::kFloat64));
    if (node_count >= 100000) register_unit_seed(c.seed);
  }
  g_direct_call = true;
  g_direct_energy_out = energy_out;
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> out;
  try {
    out = dpa4c_canonical_compress_energy_force(
        edge_t, source_t, row_t, atype_t, c.table, c.pair_film, c.pair_mixing, c.type_embedding,
        c.readout_matrices, c.coupling_meta, c.coupling_entry, c.coupling_value, c.output_mean,
        c.output_inv_std, c.spin, c.spin_pair, c.spin_type, c.lmax, c.table_stride, c.table_max,
        c.rcut, c.eps, c.degree_floor, c.ws, c.bs, c.resnets, c.w_head, c.b_head, c.bias_atom_e,
        c.act, c.seed.slice(0, 0, node_count), c.tile, c.n_node_per_frame);
  } catch (...) {
    g_direct_call = false;
    g_direct_energy_out = nullptr;
    throw;
  }
  g_direct_call = false;
  g_direct_energy_out = nullptr;
  const auto& force = std::get<1>(out);
  const auto& virial = std::get<2>(out);
  static const DIRECT_FP32_GLOBAL_VIRIALUseGetter direct_fp32_use_getter =
      direct_fp32_global_virial_resolve_use_getter();
  const bool direct_fp32_global_virial =
      direct_fp32_use_getter && direct_fp32_use_getter() != 0;
  if (direct_fp32_global_virial) {
    g_direct_fp32_global_virial_owner = virial;
    g_direct_fp32_global_virial = virial.data_ptr<float>();
  } else {
    g_direct_fp32_global_virial_owner = torch::Tensor();
  }
  const long n_force = node_count * 3;
  const long n_total = direct_fp32_global_virial ? n_force : node_count * 12;
  const int threads = 256;
  int blocks = static_cast<int>(std::min<long>((n_total + threads - 1) / threads, 4096));
  if (uniform_widen_requested) {
    const char* setting = std::getenv("DPA_UNIFORM_WIDEN_WIDEN_BLOCKS");
    blocks = setting && setting[0] ? std::max(1, std::atoi(setting)) : 256;
  }
  if (!dpa_chain_try_widen || !dpa_chain_try_widen(
      n_force, n_total, force.data_ptr<float>(), virial.data_ptr<float>(), force_out, virial_out, blocks, threads, at::cuda::getCurrentCUDAStream())) {
  dpa_direct_widen_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      n_force, n_total, force.data_ptr<float>(), virial.data_ptr<float>(), force_out, virial_out);
  }
  return 1;
}

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.def("dpa4c_canonical_compress_energy_force(Tensor edge_vec, Tensor source, Tensor destination_row_ptr, Tensor atype, Tensor table, Tensor pair_film, Tensor pair_mixing, Tensor type_embedding, Tensor readout_matrices, Tensor coupling_meta, Tensor coupling_entry, Tensor coupling_value, Tensor output_mean, Tensor output_inv_std, Tensor spin, Tensor spin_pair, Tensor spin_type, int lmax, float table_stride, float table_max, float rcut, float eps, float degree_floor, Tensor[] ws, Tensor[] bs, int[] resnets, Tensor w_head, Tensor b_head, Tensor bias_atom_e, int act, Tensor seed, int tile, Tensor n_node_per_frame) -> (Tensor, Tensor, Tensor, Tensor)");
  library.impl("dpa4c_canonical_compress_energy_force",torch::kCUDA,&dpa4c_canonical_compress_energy_force);
  library.impl("dpa4c_graph_compress", torch::kCUDA, &dpa4c_graph_compress);
  library.impl("dpa4c_graph_compress_backward", torch::kCUDA,
               &dpa4c_graph_compress_backward);
  library.impl("dpa4c_canonical_compress", torch::kCUDA,
               &dpa4c_canonical_compress);
  library.impl("dpa4c_canonical_compress_backward", torch::kCUDA,
               &dpa4c_canonical_compress_backward);
  library.impl("dpa4c_canonical_compress_backward_inplace", torch::kCUDA,
               &dpa4c_canonical_compress_backward_inplace);
  library.impl("dpa4c_canonical_compress_energy_gradient", torch::kCUDA,
               &dpa4c_canonical_compress_energy_gradient);
}
