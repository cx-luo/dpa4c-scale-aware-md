#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "graph_compress.cuh"
#include "raw_geometry.cuh"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include "fit_node_bridge.h"
namespace deepmd_dpa4c {

namespace chain_compact_graph {
void launch_raw_forward_c8(const Arguments&, const DpaDirectNeighborView&,
                           cudaStream_t);
}
namespace chain_hermite_forward {

constexpr int kModeStride=kMaxRadialModes+4;
constexpr int kROW_LAYOUTPackedRowFloat4 = 8;
constexpr int kROW_LAYOUTPaddedRowFloat4 = 9;
constexpr long kROW_LAYOUTPaddedMinNodes = 8192;
constexpr long kROW_LAYOUTPaddedMaxNodes = 40000;
template <int RowFloat4>
constexpr int row_layout_forward_shared_bytes() {
  return 600 * RowFloat4 * int(sizeof(float4));
}
// Pipelined row compaction: identical output to compact_emit_row (same kept
// order, same float conversion), but each chunk of D*32 candidates issues all
// index loads, then all coordinate gathers, before the first ballot.
// With ktype, the kept edges' source types are written alongside (packed rows only).
template <int D>
__device__ __forceinline__ int compact_emit_row_pf(
    const DpaDirectNeighborView& v,long center,long begin,long end,float4* kept,float4* records,
    unsigned char* ktype,const Arguments& a) {
  const int lane=threadIdx.x&31;
  const long atom=v.model_to_atom[center];
  const double xi=__ldg(v.x+atom*v.stride_atom);
  const double yi=__ldg(v.x+atom*v.stride_atom+v.stride_xyz);
  const double zi=__ldg(v.x+atom*v.stride_atom+2*v.stride_xyz);
  int count=0;
  for(long chunk=begin;chunk<end;chunk+=D*32) {
    long j[D];std::uint32_t src[D];
#pragma unroll
    for(int k=0;k<D;++k) {
      const long e=chunk+k*32+lane;
      j[k]=-1;src[k]=0;
      if(e<end){j[k]=v.candidate_atom[e];src[k]=v.source[e];}
    }
    double px[D],py[D],pz[D];int ty[D];
#pragma unroll
    for(int k=0;k<D;++k) {
      px[k]=py[k]=pz[k]=0.0;ty[k]=0;
      if(j[k]>=0){
        const double* p=v.x+j[k]*v.stride_atom;
        px[k]=__ldg(p);py[k]=__ldg(p+v.stride_xyz);pz[k]=__ldg(p+2*v.stride_xyz);
        if(ktype)ty[k]=int(a.atype[src[k]]);
      }
    }
#pragma unroll
    for(int k=0;k<D;++k) {
      if(chunk+k*32>=end)break;
      bool active=false;float x=0,y=0,z=0;
      if(j[k]>=0) {
        const double dx=px[k]-xi,dy=py[k]-yi,dz=pz[k]-zi;
        active=filtered_exact_cut(dx,dy,dz,v.cutoff_squared);
        if(active){x=float(dx*v.inverse_distance_unit);y=float(dy*v.inverse_distance_unit);z=float(dz*v.inverse_distance_unit);}
      }
      const unsigned mask=__ballot_sync(0xffffffffu,active);
      if(active){
        const int at=count+__popc(mask&((1u<<lane)-1u));
        float4 record=make_float4(x,y,z,__uint_as_float(src[k]));
        if(kept)kept[at]=record;
        if(ktype)ktype[at]=static_cast<unsigned char>(ty[k]);
        if(records)records[begin+at]=record;
      }
      count+=__popc(mask);
    }
  }
  __syncwarp();
  return count;
}

__device__ __forceinline__ EdgeGeometry geometry_from_record_type(
    const Arguments& a,const float4& record,float (&xyz)[3],int source_type) {
  EdgeGeometry g;
  xyz[0]=record.x;xyz[1]=record.y;xyz[2]=record.z;
  const float square=record.x*record.x+record.y*record.y+record.z*record.z+a.eps*a.eps;
  g.inverse_radius=rsqrtf(square);g.radius=square*g.inverse_radius;
  g.ux=record.x*g.inverse_radius;g.uy=record.y*g.inverse_radius;g.uz=record.z*g.inverse_radius;
  g.envelope=c3_envelope(g.radius,a.rcut);g.source=__float_as_uint(record.w);
  g.source_type=source_type;return g;
}
template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Canonical,
          typename index_t, int CM>
__device__ __forceinline__ void raw_forward_body(Arguments args,DpaDirectNeighborView view,const long node) {
  using P = Profile<Channels, Lmax, HasSpin>;
  constexpr int EdgeWidth = P::ForwardEdgeWidth;
  constexpr int Groups = kWarpSize / EdgeWidth;
  constexpr int ChannelTiles = Channels / EdgeWidth;
  constexpr int AngularTiles = (P::C1 + EdgeWidth - 1) / EdgeWidth;
  constexpr int TensorTiles = (P::C2 + EdgeWidth - 1) / EdgeWidth;
  constexpr int HighTiles = (P::HighCount + EdgeWidth - 1) / EdgeWidth;
  // The spin families share the neighbour width with degree two, so they
  // inherit its tiling. The single neighbour quadrupole distributes its five
  // components across lanes, exactly as the single-channel high degrees do.
  constexpr int SpinTiles = HasSpin ? TensorTiles : 0;
  constexpr int SpinTensorTiles = HasSpin ? (5 + EdgeWidth - 1) / EdgeWidth : 0;

  constexpr int PW = ((CM >> 8) & 15) ? ((CM >> 8) & 15) : 1;
  const int persist_warp = PW == 1 ? 0 : int(threadIdx.x >> 5);
  const int thread = threadIdx.x & 31;
  const int base_channel = thread & (EdgeWidth - 1);
  const int group = thread / EdgeWidth;
  const int leader = group * EdgeWidth;
  const unsigned mask = subwarp_mask<EdgeWidth>(leader);
  const auto* edge_index = static_cast<const index_t*>(args.edge_index);
  const auto* destination_order =
      static_cast<const index_t*>(args.destination_order);
  const int center_type = static_cast<int>(args.atype[args.node_begin + node]);
  const long begin = args.destination_row_ptr[node];
  const long end = (CM&8192) ? begin+static_cast<const int*>(args.destination_order)[node] : args.destination_row_ptr[node + 1];
  const int radial_modes = HasModes ? args.radial_modes : 0;

  float scalar[ChannelTiles] = {};
  float vector[AngularTiles][3] = {};
  float tensor[TensorTiles][5] = {};
  float high[HighTiles > 0 ? HighTiles : 1] = {};
  float scalar_mass = 0.0f;
  float angular_mass = 0.0f;
  float spin_magnitude[SpinTiles > 0 ? SpinTiles : 1] = {};
  float spin_coordination[SpinTiles > 0 ? SpinTiles : 1] = {};
  float spin_vector[SpinTiles > 0 ? SpinTiles : 1][3] = {};
  float spin_bond[SpinTiles > 0 ? SpinTiles : 1][3] = {};
  float spin_tensor[SpinTensorTiles > 0 ? SpinTensorTiles : 1] = {};

  // Explicitly aligned: the mode residual reads these rows as float4, and
  // a preceding shared array of arbitrary length would otherwise leave the
  // block on a four-byte boundary.
  __shared__ __align__(
      16) float mode_cache[HasModes ? Groups * kModeStride : 1];
  float* modes = mode_cache + (HasModes ? group * kModeStride : 0);

  __shared__ float4 compact_bank[PW][256];
  float4* compact_indices = compact_bank[persist_warp];
  const bool packed=end-begin<=256;
  auto* records=reinterpret_cast<float4*>(const_cast<float*>(args.edge_vec));
  auto* counts=reinterpret_cast<int*>(const_cast<bool*>(args.edge_mask));
  constexpr int kDepth=CM&15;
  constexpr bool kType=(CM&16)!=0&&kDepth>0;
  __shared__ unsigned char compact_types[kType?256:1];
  const bool use_ctype=kType&&packed;
  long live;
  DpaDirectNeighborView geometry = view;
  if (geometry.cutoff_squared < 0.0) geometry.cutoff_squared = -geometry.cutoff_squared;
  if constexpr(kDepth==0) {
    live=compact_emit_row(geometry,args.node_begin+node,begin,end,packed?compact_indices:nullptr,records);
  } else {
    live=compact_emit_row_pf<(kDepth>0?kDepth:1)>(geometry,args.node_begin+node,begin,end,packed?compact_indices:nullptr,(args.radial_modes&256)&&packed?nullptr:records,use_ctype?compact_types:nullptr,args);
  }
  if(thread==0)counts[args.node_begin+node]=live;
  if(args.radial_modes & 8)return;
  // === Step 1. Reduce the destination row into degree-wise moments ===
  for (long position = group; position < live; position += Groups) {
    EdgeGeometry geometry;float xyz[3];
    if(use_ctype)geometry=geometry_from_record_type(args,compact_indices[position],xyz,compact_types[position]);
    else geometry=geometry_from_cache(args,packed?compact_indices[position]:records[begin+position],xyz);
    const TableLocation location =
        locate_table<Canonical>(geometry.radius, args.table_stride,
                                args.table_max, args.interval_count);
    if constexpr (!Canonical) {
      if (center_type >= args.type_count - 1 ||
          geometry.source_type >= args.type_count - 1) {
        continue;
      }
    }
    const TableRow row = table_row(args.table, location, args.table_width);
    const float coordinate = location.coordinate;
    if constexpr (HasModes) {
      __syncwarp(mask);
      for (int mode = base_channel; mode < radial_modes; mode += EdgeWidth) {
        modes[mode] = evaluate_table(row, Channels + mode, coordinate);
      }
      __syncwarp(mask);
    }

    float basis[9];
    fill_angular_basis(geometry, basis);
    const float envelope = geometry.envelope;
    const long pair =
        static_cast<long>(center_type) * args.type_count + geometry.source_type;
    const float2* film_row =
        reinterpret_cast<const float2*>(args.pair_film + pair * Channels * 2) +
        base_channel;
    const float* mixing_row =
        HasModes
            ? args.pair_mixing + (pair * Channels + base_channel) * radial_modes
            : nullptr;

    // The spin payload is hoisted out of the channel loop: every channel
    // multiplies the same conditioned neighbour moment and the same projection
    // of that moment onto the bond, and only the amplitude varies with the
    // channel.
    float neighbor_spin[3] = {0.0f, 0.0f, 0.0f};
    float neighbor_magnitude = 0.0f;
    float neighbor_gate = 0.0f;
    float bond_alignment = 0.0f;
    const float2* spin_row = nullptr;
    if constexpr (HasSpin) {
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, geometry.source_type);
      load_conditioned_spin(args.spin, weights.scale, geometry.source,
                            neighbor_spin);
      neighbor_magnitude = neighbor_spin[0] * neighbor_spin[0] +
                           neighbor_spin[1] * neighbor_spin[1] +
                           neighbor_spin[2] * neighbor_spin[2];
      neighbor_gate = weights.gate;
      // Component of the neighbour moment along the bond. The bond-projected
      // family is this scalar times the unit direction, which is `basis[1..3]`.
      bond_alignment = neighbor_spin[0] * basis[1] +
                       neighbor_spin[1] * basis[2] +
                       neighbor_spin[2] * basis[3];
      spin_row =
          reinterpret_cast<const float2*>(args.spin_pair + pair * P::Cs * 2) +
          base_channel;
    }

    float angular_zero = 0.0f;
    float spin_zero = 0.0f;
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      const float radial = (args.radial_modes & 1) ? 0.5f : evaluate_table(row, channel, coordinate);
      const float2 film = __ldg(film_row + tile * EdgeWidth);
      float film_value = fmaf(film.x, radial, film.y);
      if constexpr (HasModes) {
        accumulate_modes(mixing_row + tile * (EdgeWidth * radial_modes), modes,
                         radial_modes, film_value);
      }
      // Degree zero carries one envelope factor; every non-scalar degree
      // carries a second one.
      const float amplitude = film_value * envelope;
      const float angular = amplitude * envelope;
      scalar[tile] += amplitude;
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          vector[tile][component] =
              fmaf(angular, basis[1 + component], vector[tile][component]);
        }
      }
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          tensor[tile][component] =
              fmaf(angular, basis[4 + component], tensor[tile][component]);
        }
      }
      if constexpr (HasSpin) {
        if (channel < P::Cs) {
          const float2 spin_film = __ldg(spin_row + tile * EdgeWidth);
          const float weight =
              fmaf(spin_film.x, radial, spin_film.y) * envelope * envelope;
          spin_magnitude[tile] =
              fmaf(weight, neighbor_magnitude, spin_magnitude[tile]);
          spin_coordination[tile] =
              fmaf(weight, neighbor_gate, spin_coordination[tile]);
          const float projected = weight * bond_alignment;
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            spin_vector[tile][component] = fmaf(
                weight, neighbor_spin[component], spin_vector[tile][component]);
            spin_bond[tile][component] = fmaf(projected, basis[1 + component],
                                              spin_bond[tile][component]);
          }
          if (channel == 0) {
            spin_zero = weight;
          }
        }
      }
      if constexpr (Lmax >= 3) {
        if (channel == 0) {
          angular_zero = angular;
        }
      }
    }
    if constexpr (HasSpin) {
      // The single neighbour quadrupole reads the leading spin channel only,
      // so its five components are distributed across the lanes of the edge
      // group and each is evaluated on the one lane that owns it.
      const float weight = __shfl_sync(mask, spin_zero, leader);
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        if (component % EdgeWidth == base_channel) {
          spin_tensor[component / EdgeWidth] =
              fmaf(weight,
                   spin_quadrupole_component(neighbor_spin[0], neighbor_spin[1],
                                             neighbor_spin[2], component),
                   spin_tensor[component / EdgeWidth]);
        }
      }
    }
    if constexpr (Lmax >= 3) {
      // Degrees three and above read only the leading channel, so their
      // components are distributed across the lanes of the edge group. The
      // component is iterated at compile time and selected with a lane
      // predicate, which folds the harmonic selector and keeps the tile index
      // constant.
      const float amplitude = __shfl_sync(mask, angular_zero, leader);
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          high[component / EdgeWidth] =
              fmaf(amplitude, high_basis_value(geometry, component),
                   high[component / EdgeWidth]);
        }
      }
    }
    if (base_channel == 0) {
      const float squared = envelope * envelope;
      scalar_mass += squared;
      angular_mass = fmaf(squared, squared, angular_mass);
    }
  }

  // === Step 2. Merge the concurrent edge groups and normalize ===
  if constexpr (Groups > 1) {
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      scalar[tile] = reduce_channel_groups<EdgeWidth>(scalar[tile]);
    }
#pragma unroll
    for (int tile = 0; tile < AngularTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        vector[tile][component] =
            reduce_channel_groups<EdgeWidth>(vector[tile][component]);
      }
    }
#pragma unroll
    for (int tile = 0; tile < TensorTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        tensor[tile][component] =
            reduce_channel_groups<EdgeWidth>(tensor[tile][component]);
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int tile = 0; tile < HighTiles; ++tile) {
        high[tile] = reduce_channel_groups<EdgeWidth>(high[tile]);
      }
    }
    if constexpr (HasSpin) {
#pragma unroll
      for (int tile = 0; tile < SpinTiles; ++tile) {
        spin_magnitude[tile] =
            reduce_channel_groups<EdgeWidth>(spin_magnitude[tile]);
        spin_coordination[tile] =
            reduce_channel_groups<EdgeWidth>(spin_coordination[tile]);
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          spin_vector[tile][component] =
              reduce_channel_groups<EdgeWidth>(spin_vector[tile][component]);
          spin_bond[tile][component] =
              reduce_channel_groups<EdgeWidth>(spin_bond[tile][component]);
        }
      }
#pragma unroll
      for (int tile = 0; tile < SpinTensorTiles; ++tile) {
        spin_tensor[tile] = reduce_channel_groups<EdgeWidth>(spin_tensor[tile]);
      }
    }
  }
  __shared__ float normalizer_bank[PW][4];
  float* normalizer_shared = normalizer_bank[persist_warp];
  {
    const float total_scalar = warp_sum(scalar_mass);
    const float total_angular = warp_sum(angular_mass);
    if (thread == 0) {
      normalizer_shared[0] = rsqrtf(total_scalar + args.degree_floor);
      normalizer_shared[1] = rsqrtf(total_angular + args.degree_floor);
      normalizer_shared[2] = sqrtf(total_scalar + args.degree_floor);
      normalizer_shared[3] = sqrtf(total_angular + args.degree_floor);
    }
  }
  if constexpr (PW == 1) __syncthreads(); else __syncwarp();
  const float scalar_norm = normalizer_shared[0];
  const float angular_norm = normalizer_shared[1];
  if (thread < 2) {
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputDivisor + thread,
                     normalizer_shared[2 + thread]);
  }

  // === Step 3. Publish the normalized moments and the saved state ===
  __shared__ float moments_bank[PW][P::MomentWidth];
  float* moments = moments_bank[persist_warp];
  if (thread < EdgeWidth) {
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      moments[P::ScalarOffset + base_channel + tile * EdgeWidth] =
          scalar[tile] * scalar_norm;
    }
#pragma unroll
    for (int tile = 0; tile < AngularTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          moments[P::VectorOffset + component * P::C1 + channel] =
              vector[tile][component] * angular_norm;
        }
      }
    }
#pragma unroll
    for (int tile = 0; tile < TensorTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          moments[P::TensorOffset + component * P::C2 + channel] =
              tensor[tile][component] * angular_norm;
        }
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          moments[P::HighOffset + component] =
              high[component / EdgeWidth] * angular_norm;
        }
      }
    }
    if constexpr (HasSpin) {
      // The reduced spin families carry the same squared envelope as every
      // non-scalar geometric moment and therefore share its normalizer.
#pragma unroll
      for (int tile = 0; tile < SpinTiles; ++tile) {
        const int channel = base_channel + tile * EdgeWidth;
        if (channel < P::Cs) {
          moments[P::SpinMagnitude + channel] =
              spin_magnitude[tile] * angular_norm;
          moments[P::SpinCoordination + channel] =
              spin_coordination[tile] * angular_norm;
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            moments[P::SpinVector + component * P::Cs + channel] =
                spin_vector[tile][component] * angular_norm;
            moments[P::SpinBond + component * P::Cs + channel] =
                spin_bond[tile][component] * angular_norm;
          }
        }
      }
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        if (component % EdgeWidth == base_channel) {
          moments[P::SpinTensor + component] =
              spin_tensor[component / EdgeWidth] * angular_norm;
        }
      }
    }
  }
  if constexpr (HasSpin) {
    // The on-site families are node local. They are written after the
    // division so that an invariant pairing an on-site channel with a
    // neighbour channel carries exactly one neighborhood normalizer.
    if (thread == 0) {
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, center_type);
      float center_spin[3];
      load_conditioned_spin(args.spin, weights.scale, args.node_begin + node,
                            center_spin);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        moments[P::SpinOnsiteVector + component] =
            weights.vector * center_spin[component];
      }
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        moments[P::SpinOnsiteTensor + component] =
            weights.quadrupole *
            spin_quadrupole_component(center_spin[0], center_spin[1],
                                      center_spin[2], component);
      }
    }
  }
  if constexpr (PW == 1) __syncthreads(); else __syncwarp();

  const long state_offset = node * P::StateWidth;
  for (int coordinate = thread; coordinate < P::MomentWidth;
       coordinate += P::Threads) {
    args.state_out[state_offset + coordinate] = moments[coordinate];
  }
  if (thread == 0) {
    args.state_out[state_offset + P::MomentWidth] = scalar_norm;
    args.state_out[state_offset + P::MomentWidth + 1] = angular_norm;
  }

}



template<bool Shared, int RowFloat4 = kROW_LAYOUTPackedRowFloat4>
__device__ __forceinline__ float adaptive_forward_radial(const TableRow& row,const float* cache,int index,int c,float x) {
  if constexpr(!Shared)return evaluate_table(row,c,x);
  else {
    const float4 coefficients = reinterpret_cast<const float4*>(cache)[
        index * RowFloat4 + c];
    return coefficients.x+(coefficients.y+
        (coefficients.z+coefficients.w*x)*x)*x;
  }
}
template <int Channels,
          int Lmax,
          bool HasModes,
          bool HasSpin,
          bool Canonical,
          typename index_t, int CM, int PW, int Capacity, bool Shared>
__device__ __forceinline__ void adaptive_forward_forward_body(Arguments args,DpaDirectNeighborView view,const long node,const float* shared_table) {
  using P = Profile<Channels, Lmax, HasSpin>;
  constexpr int EdgeWidth = P::ForwardEdgeWidth;
  constexpr int Groups = kWarpSize / EdgeWidth;
  constexpr int ChannelTiles = Channels / EdgeWidth;
  constexpr int AngularTiles = (P::C1 + EdgeWidth - 1) / EdgeWidth;
  constexpr int TensorTiles = (P::C2 + EdgeWidth - 1) / EdgeWidth;
  constexpr int HighTiles = (P::HighCount + EdgeWidth - 1) / EdgeWidth;
  // The spin families share the neighbour width with degree two, so they
  // inherit its tiling. The single neighbour quadrupole distributes its five
  // components across lanes, exactly as the single-channel high degrees do.
  constexpr int SpinTiles = HasSpin ? TensorTiles : 0;
  constexpr int SpinTensorTiles = HasSpin ? (5 + EdgeWidth - 1) / EdgeWidth : 0;

  const int persist_warp = PW == 1 ? 0 : int(threadIdx.x >> 5);
  const int thread = threadIdx.x & 31;
  const int base_channel = thread & (EdgeWidth - 1);
  const int group = thread / EdgeWidth;
  const int leader = group * EdgeWidth;
  const unsigned mask = subwarp_mask<EdgeWidth>(leader);
  const auto* edge_index = static_cast<const index_t*>(args.edge_index);
  const auto* destination_order =
      static_cast<const index_t*>(args.destination_order);
  const int center_type = static_cast<int>(args.atype[args.node_begin + node]);
  const long begin = args.destination_row_ptr[node];
  const long end = (CM&8192) ? begin+static_cast<const int*>(args.destination_order)[node] : args.destination_row_ptr[node + 1];
  const int radial_modes = HasModes ? args.radial_modes : 0;

  float scalar[ChannelTiles] = {};
  float vector[AngularTiles][3] = {};
  float tensor[TensorTiles][5] = {};
  float high[HighTiles > 0 ? HighTiles : 1] = {};
  float scalar_mass = 0.0f;
  float angular_mass = 0.0f;
  float spin_magnitude[SpinTiles > 0 ? SpinTiles : 1] = {};
  float spin_coordination[SpinTiles > 0 ? SpinTiles : 1] = {};
  float spin_vector[SpinTiles > 0 ? SpinTiles : 1][3] = {};
  float spin_bond[SpinTiles > 0 ? SpinTiles : 1][3] = {};
  float spin_tensor[SpinTensorTiles > 0 ? SpinTensorTiles : 1] = {};

  // Explicitly aligned: the mode residual reads these rows as float4, and
  // a preceding shared array of arbitrary length would otherwise leave the
  // block on a four-byte boundary.
  __shared__ __align__(
      16) float mode_cache[HasModes ? Groups * kModeStride : 1];
  float* modes = mode_cache + (HasModes ? group * kModeStride : 0);

  __shared__ float4 compact_bank[PW][Capacity ? Capacity : 1];
  float4* compact_indices = compact_bank[persist_warp];
  const bool packed=Capacity>0 && end-begin<=Capacity;
  auto* records=reinterpret_cast<float4*>(const_cast<float*>(args.edge_vec));
  auto* counts=reinterpret_cast<int*>(const_cast<bool*>(args.edge_mask));
  constexpr int kDepth=CM&15;
  constexpr bool kType=(CM&16)!=0&&kDepth>0;
  __shared__ unsigned char compact_types[kType?256:1];
  const bool use_ctype=kType&&packed;
  long live;
  if constexpr(kDepth==0) live=compact_emit_row(view,args.node_begin+node,begin,end,packed?compact_indices:nullptr,records);
  else live=compact_emit_row_pf<(kDepth>0?kDepth:1)>(view,args.node_begin+node,begin,end,packed?compact_indices:nullptr,(args.radial_modes&256)&&packed?nullptr:records,use_ctype?compact_types:nullptr,args);
  if(thread==0)counts[args.node_begin+node]=live;
  if(args.radial_modes & 8)return;
  // === Step 1. Reduce the destination row into degree-wise moments ===
  for (long position = group; position < live; position += Groups) {
    EdgeGeometry geometry;float xyz[3];
    if(use_ctype)geometry=geometry_from_record_type(args,compact_indices[position],xyz,compact_types[position]);
    else geometry=geometry_from_cache(args,packed?compact_indices[position]:records[begin+position],xyz);
    const TableLocation location =
        locate_table<Canonical>(geometry.radius, args.table_stride,
                                args.table_max, args.interval_count);
    if constexpr (!Canonical) {
      if (center_type >= args.type_count - 1 ||
          geometry.source_type >= args.type_count - 1) {
        continue;
      }
    }
    const TableRow row = table_row(args.table, location, args.table_width);
    const float coordinate = location.coordinate;
    if constexpr (HasModes) {
      __syncwarp(mask);
      for (int mode = base_channel; mode < radial_modes; mode += EdgeWidth) {
        modes[mode] = evaluate_table(row, Channels + mode, coordinate);
      }
      __syncwarp(mask);
    }

    float basis[9];
    fill_angular_basis(geometry, basis);
    const float envelope = geometry.envelope;
    const long pair =
        static_cast<long>(center_type) * args.type_count + geometry.source_type;
    const float2* film_row =
        reinterpret_cast<const float2*>(args.pair_film + pair * Channels * 2) +
        base_channel;
    const float* mixing_row =
        HasModes
            ? args.pair_mixing + (pair * Channels + base_channel) * radial_modes
            : nullptr;

    // The spin payload is hoisted out of the channel loop: every channel
    // multiplies the same conditioned neighbour moment and the same projection
    // of that moment onto the bond, and only the amplitude varies with the
    // channel.
    float neighbor_spin[3] = {0.0f, 0.0f, 0.0f};
    float neighbor_magnitude = 0.0f;
    float neighbor_gate = 0.0f;
    float bond_alignment = 0.0f;
    const float2* spin_row = nullptr;
    if constexpr (HasSpin) {
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, geometry.source_type);
      load_conditioned_spin(args.spin, weights.scale, geometry.source,
                            neighbor_spin);
      neighbor_magnitude = neighbor_spin[0] * neighbor_spin[0] +
                           neighbor_spin[1] * neighbor_spin[1] +
                           neighbor_spin[2] * neighbor_spin[2];
      neighbor_gate = weights.gate;
      // Component of the neighbour moment along the bond. The bond-projected
      // family is this scalar times the unit direction, which is `basis[1..3]`.
      bond_alignment = neighbor_spin[0] * basis[1] +
                       neighbor_spin[1] * basis[2] +
                       neighbor_spin[2] * basis[3];
      spin_row =
          reinterpret_cast<const float2*>(args.spin_pair + pair * P::Cs * 2) +
          base_channel;
    }

    float angular_zero = 0.0f;
    float spin_zero = 0.0f;
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      const float radial = (args.radial_modes & 1) ? 0.5f : adaptive_forward_radial<Shared>(row,shared_table,location.index,channel,coordinate);
      const float2 film = __ldg(film_row + tile * EdgeWidth);
      float film_value = fmaf(film.x, radial, film.y);
      if constexpr (HasModes) {
        accumulate_modes(mixing_row + tile * (EdgeWidth * radial_modes), modes,
                         radial_modes, film_value);
      }
      // Degree zero carries one envelope factor; every non-scalar degree
      // carries a second one.
      const float amplitude = film_value * envelope;
      const float angular = amplitude * envelope;
      scalar[tile] += amplitude;
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          vector[tile][component] =
              fmaf(angular, basis[1 + component], vector[tile][component]);
        }
      }
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          tensor[tile][component] =
              fmaf(angular, basis[4 + component], tensor[tile][component]);
        }
      }
      if constexpr (HasSpin) {
        if (channel < P::Cs) {
          const float2 spin_film = __ldg(spin_row + tile * EdgeWidth);
          const float weight =
              fmaf(spin_film.x, radial, spin_film.y) * envelope * envelope;
          spin_magnitude[tile] =
              fmaf(weight, neighbor_magnitude, spin_magnitude[tile]);
          spin_coordination[tile] =
              fmaf(weight, neighbor_gate, spin_coordination[tile]);
          const float projected = weight * bond_alignment;
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            spin_vector[tile][component] = fmaf(
                weight, neighbor_spin[component], spin_vector[tile][component]);
            spin_bond[tile][component] = fmaf(projected, basis[1 + component],
                                              spin_bond[tile][component]);
          }
          if (channel == 0) {
            spin_zero = weight;
          }
        }
      }
      if constexpr (Lmax >= 3) {
        if (channel == 0) {
          angular_zero = angular;
        }
      }
    }
    if constexpr (HasSpin) {
      // The single neighbour quadrupole reads the leading spin channel only,
      // so its five components are distributed across the lanes of the edge
      // group and each is evaluated on the one lane that owns it.
      const float weight = __shfl_sync(mask, spin_zero, leader);
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        if (component % EdgeWidth == base_channel) {
          spin_tensor[component / EdgeWidth] =
              fmaf(weight,
                   spin_quadrupole_component(neighbor_spin[0], neighbor_spin[1],
                                             neighbor_spin[2], component),
                   spin_tensor[component / EdgeWidth]);
        }
      }
    }
    if constexpr (Lmax >= 3) {
      // Degrees three and above read only the leading channel, so their
      // components are distributed across the lanes of the edge group. The
      // component is iterated at compile time and selected with a lane
      // predicate, which folds the harmonic selector and keeps the tile index
      // constant.
      const float amplitude = __shfl_sync(mask, angular_zero, leader);
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          high[component / EdgeWidth] =
              fmaf(amplitude, high_basis_value(geometry, component),
                   high[component / EdgeWidth]);
        }
      }
    }
    if (base_channel == 0) {
      const float squared = envelope * envelope;
      scalar_mass += squared;
      angular_mass = fmaf(squared, squared, angular_mass);
    }
  }

  // === Step 2. Merge the concurrent edge groups and normalize ===
  if constexpr (Groups > 1) {
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      scalar[tile] = reduce_channel_groups<EdgeWidth>(scalar[tile]);
    }
#pragma unroll
    for (int tile = 0; tile < AngularTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        vector[tile][component] =
            reduce_channel_groups<EdgeWidth>(vector[tile][component]);
      }
    }
#pragma unroll
    for (int tile = 0; tile < TensorTiles; ++tile) {
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        tensor[tile][component] =
            reduce_channel_groups<EdgeWidth>(tensor[tile][component]);
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int tile = 0; tile < HighTiles; ++tile) {
        high[tile] = reduce_channel_groups<EdgeWidth>(high[tile]);
      }
    }
    if constexpr (HasSpin) {
#pragma unroll
      for (int tile = 0; tile < SpinTiles; ++tile) {
        spin_magnitude[tile] =
            reduce_channel_groups<EdgeWidth>(spin_magnitude[tile]);
        spin_coordination[tile] =
            reduce_channel_groups<EdgeWidth>(spin_coordination[tile]);
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          spin_vector[tile][component] =
              reduce_channel_groups<EdgeWidth>(spin_vector[tile][component]);
          spin_bond[tile][component] =
              reduce_channel_groups<EdgeWidth>(spin_bond[tile][component]);
        }
      }
#pragma unroll
      for (int tile = 0; tile < SpinTensorTiles; ++tile) {
        spin_tensor[tile] = reduce_channel_groups<EdgeWidth>(spin_tensor[tile]);
      }
    }
  }
  __shared__ float normalizer_bank[PW][4];
  float* normalizer_shared = normalizer_bank[persist_warp];
  {
    const float total_scalar = warp_sum(scalar_mass);
    const float total_angular = warp_sum(angular_mass);
    if (thread == 0) {
      normalizer_shared[0] = rsqrtf(total_scalar + args.degree_floor);
      normalizer_shared[1] = rsqrtf(total_angular + args.degree_floor);
      normalizer_shared[2] = sqrtf(total_scalar + args.degree_floor);
      normalizer_shared[3] = sqrtf(total_angular + args.degree_floor);
    }
  }
  if constexpr (PW == 1) __syncthreads(); else __syncwarp();
  const float scalar_norm = normalizer_shared[0];
  const float angular_norm = normalizer_shared[1];
  if (thread < 2) {
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputDivisor + thread,
                     normalizer_shared[2 + thread]);
  }

  // === Step 3. Publish the normalized moments and the saved state ===
  __shared__ float moments_bank[PW][P::MomentWidth];
  float* moments = moments_bank[persist_warp];
  if (thread < EdgeWidth) {
#pragma unroll
    for (int tile = 0; tile < ChannelTiles; ++tile) {
      moments[P::ScalarOffset + base_channel + tile * EdgeWidth] =
          scalar[tile] * scalar_norm;
    }
#pragma unroll
    for (int tile = 0; tile < AngularTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      if (channel < P::C1) {
#pragma unroll
        for (int component = 0; component < 3; ++component) {
          moments[P::VectorOffset + component * P::C1 + channel] =
              vector[tile][component] * angular_norm;
        }
      }
    }
#pragma unroll
    for (int tile = 0; tile < TensorTiles; ++tile) {
      const int channel = base_channel + tile * EdgeWidth;
      if (channel < P::C2) {
#pragma unroll
        for (int component = 0; component < 5; ++component) {
          moments[P::TensorOffset + component * P::C2 + channel] =
              tensor[tile][component] * angular_norm;
        }
      }
    }
    if constexpr (Lmax >= 3) {
#pragma unroll
      for (int component = 0; component < P::HighCount; ++component) {
        if (component % EdgeWidth == base_channel) {
          moments[P::HighOffset + component] =
              high[component / EdgeWidth] * angular_norm;
        }
      }
    }
    if constexpr (HasSpin) {
      // The reduced spin families carry the same squared envelope as every
      // non-scalar geometric moment and therefore share its normalizer.
#pragma unroll
      for (int tile = 0; tile < SpinTiles; ++tile) {
        const int channel = base_channel + tile * EdgeWidth;
        if (channel < P::Cs) {
          moments[P::SpinMagnitude + channel] =
              spin_magnitude[tile] * angular_norm;
          moments[P::SpinCoordination + channel] =
              spin_coordination[tile] * angular_norm;
#pragma unroll
          for (int component = 0; component < 3; ++component) {
            moments[P::SpinVector + component * P::Cs + channel] =
                spin_vector[tile][component] * angular_norm;
            moments[P::SpinBond + component * P::Cs + channel] =
                spin_bond[tile][component] * angular_norm;
          }
        }
      }
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        if (component % EdgeWidth == base_channel) {
          moments[P::SpinTensor + component] =
              spin_tensor[component / EdgeWidth] * angular_norm;
        }
      }
    }
  }
  if constexpr (HasSpin) {
    // The on-site families are node local. They are written after the
    // division so that an invariant pairing an on-site channel with a
    // neighbour channel carries exactly one neighborhood normalizer.
    if (thread == 0) {
      const SpinTypeWeights weights =
          load_spin_type(args.spin_type, center_type);
      float center_spin[3];
      load_conditioned_spin(args.spin, weights.scale, args.node_begin + node,
                            center_spin);
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        moments[P::SpinOnsiteVector + component] =
            weights.vector * center_spin[component];
      }
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        moments[P::SpinOnsiteTensor + component] =
            weights.quadrupole *
            spin_quadrupole_component(center_spin[0], center_spin[1],
                                      center_spin[2], component);
      }
    }
  }
  if constexpr (PW == 1) __syncthreads(); else __syncwarp();

  const long state_offset = node * P::StateWidth;
  for (int coordinate = thread; coordinate < P::MomentWidth;
       coordinate += P::Threads) {
    args.state_out[state_offset + coordinate] = moments[coordinate];
  }
  if (thread == 0) {
    args.state_out[state_offset + P::MomentWidth] = scalar_norm;
    args.state_out[state_offset + P::MomentWidth + 1] = angular_norm;
  }

}

#include "readout_kernels.cuh"

struct NarrowState {
  int rebuild, disabled;
  long nall, reference_capacity;
  unsigned long long calls, rebuilds;
};
struct NarrowCache {
  const void* owner = nullptr;
  std::uint64_t epoch = 0;
  long nall = 0, nmax = 0, nodes = 0, edges = 0;
  int *atoms = nullptr, *counts = nullptr;
  std::uint32_t* sources = nullptr;
  double* reference = nullptr;
  NarrowState* state = nullptr;
  cudaStream_t stream = nullptr;
  int adaptive_forward_policy = -1;
};
static std::vector<NarrowCache*> caches;
static NarrowCache* notified_cache = nullptr;
static void narrow_check(cudaError_t e) {
  if(e!=cudaSuccess){fprintf(stderr,"COMPACT_FORWARD CUDA: %s\n",cudaGetErrorString(e));abort();}
}
__global__ void compact_forward_notify(NarrowState* s,long nall,int rebuild) {
  s->nall=nall;
  // Keep the graph and all captured addresses alive. Growth beyond the
  // reserved reference array switches this graph to the original algorithm.
  if(nall>s->reference_capacity)s->disabled=1;
  if(rebuild)s->rebuild=1;
}
static void narrow_report() {
  for(auto* c:caches) if(c->state) {
    NarrowState s{};
    if(cudaMemcpy(&s,c->state,sizeof(s),cudaMemcpyDeviceToHost)==cudaSuccess)
      fprintf(stderr,"COMPACT_FORWARD_CACHE nodes=%ld calls=%llu rebuilds=%llu nall=%ld disabled=%d\n",
          c->nodes,s.calls,s.rebuilds,s.nall,s.disabled);
  }
}
extern "C" void dpa_narrow_geometry_epoch(const void* owner,std::uint64_t packed_epoch,int nall,int nmax,void* producer_stream) {
  auto it=std::find_if(caches.begin(),caches.end(),[&](auto* c){return c->owner==owner;});
  NarrowCache* c;
  if(it==caches.end()) {c=new NarrowCache();c->owner=owner;caches.push_back(c);}
  else c=*it;
  // Low bit explicitly marks a rebuilt topology, including the first call
  // of a new LAMMPS instance at a previously used owner address.
  const auto epoch=packed_epoch>>1;
  const bool invalid=(packed_epoch&1) || epoch!=c->epoch || nall!=c->nall;
  // Existing wait_producer_cuda_stream orders this update before inference,
  // including graph replays issued on a different stream from capture.
  if(c->state && invalid)compact_forward_notify<<<1,1,0,(cudaStream_t)producer_stream>>>(c->state,nall,1);
  c->epoch=epoch;c->nall=nall;c->nmax=nmax;notified_cache=c;
}

__global__ void compact_forward_displacement(DpaDirectNeighborView v,const double* ref,
                               NarrowState* s,double limit2,int always_refresh) {
  if(s->disabled)return;
  if(always_refresh) {if(blockIdx.x==0&&threadIdx.x==0)s->rebuild=1;return;}
  if(s->rebuild)return;
  for(long i=blockIdx.x*blockDim.x+threadIdx.x;i<s->nall;i+=gridDim.x*blockDim.x) {
    auto p=v.x+i*v.stride_atom;
    const double dx=__dsub_rn(p[0],ref[3*i]);
    const double dy=__dsub_rn(p[v.stride_xyz],ref[3*i+1]);
    const double dz=__dsub_rn(p[2*v.stride_xyz],ref[3*i+2]);
    // Double precision, conservative threshold; NaN also invalidates.
    const double sq=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
    if(!(sq<limit2))atomicExch(&s->rebuild,1);
  }
}
__device__ __forceinline__ void compact_forward_build_row(Arguments a,DpaDirectNeighborView v,
       double cut2,int* atoms,std::uint32_t* sources,int* counts,long i) {
  int lane=threadIdx.x&31;long atom=v.model_to_atom[i];
  double xi=__ldg(v.x+atom*v.stride_atom),yi=__ldg(v.x+atom*v.stride_atom+v.stride_xyz),zi=__ldg(v.x+atom*v.stride_atom+2*v.stride_xyz);
  long begin=a.destination_row_ptr[i],end=a.destination_row_ptr[i+1];int count=0;
  for(long start=begin;start<end;start+=128){
    int ja[4];std::uint32_t sj[4];double dx[4],dy[4],dz[4];
    #pragma unroll
    for(int k=0;k<4;++k){long e=start+32*k+lane;ja[k]=e<end?v.candidate_atom[e]:-1;sj[k]=e<end?v.source[e]:0;}
    #pragma unroll
    for(int k=0;k<4;++k){dx[k]=dy[k]=dz[k]=0;if(ja[k]>=0){auto p=v.x+long(ja[k])*v.stride_atom;dx[k]=__ldg(p)-xi;dy[k]=__ldg(p+v.stride_xyz)-yi;dz[k]=__ldg(p+2*v.stride_xyz)-zi;}}
    #pragma unroll
    for(int k=0;k<4;++k){
      if(start+32*k>=end)break;
      bool keep=ja[k]>=0 && filtered_exact_cut(dx[k],dy[k],dz[k],cut2);
      unsigned mask=__ballot_sync(0xffffffffu,keep);
      if(keep){int pos=count+__popc(mask&((1u<<lane)-1u));atoms[begin+pos]=ja[k];sources[begin+pos]=sj[k];}
      count+=__popc(mask);
    }
  }
  if(lane==0)counts[i]=count;
  __syncwarp();
}
__global__ void compact_forward_build(Arguments a,DpaDirectNeighborView v,double cut2,
       int* atoms,std::uint32_t* sources,int* counts,NarrowState* s) {
  if(s->disabled || !s->rebuild)return;
  for(long i=blockIdx.x;i<a.node_count;i+=gridDim.x)
    compact_forward_build_row(a,v,cut2,atoms,sources,counts,i);
}
__global__ void compact_forward_fast_forward(Arguments a,DpaDirectNeighborView v,NarrowState* s) {
  if(s->disabled)return;
  raw_forward_body<8,2,false,false,true,std::uint32_t,8196>(a,v,blockIdx.x);
}
__global__ void compact_forward_fallback_forward(Arguments a,DpaDirectNeighborView v,NarrowState* s) {
  if(!s->disabled)return;
  for(long i=blockIdx.x;i<a.node_count;i+=gridDim.x)
    raw_forward_body<8,2,false,false,true,std::uint32_t,4>(a,v,i);
}
__global__ void compact_forward_save_reference(DpaDirectNeighborView v,double* ref,NarrowState* s) {
  if(s->disabled || !s->rebuild)return;
  for(long i=blockIdx.x*blockDim.x+threadIdx.x;i<s->nall;i+=gridDim.x*blockDim.x) {
    auto p=v.x+i*v.stride_atom;
    ref[3*i]=p[0];ref[3*i+1]=p[v.stride_xyz];ref[3*i+2]=p[2*v.stride_xyz];
  }
}
__global__ void compact_forward_finish(NarrowState* s) {
  ++s->calls;if(!s->disabled&&s->rebuild)++s->rebuilds;s->rebuild=0;
}

// DENSE_FORWARD: exact fixed-profile implementation for the production C8/Lmax=2,
// no-mode/no-spin canonical path.  The generic template expresses moments as
// nested arrays and keeps the full nine-value angular basis live across both
// channel tiles; on PPU this spills 80 bytes per thread.  Here every moment is
// a named scalar and each harmonic is consumed immediately.  The graph filter,
// edge-record format, reduction order and published state layout are unchanged.
template <int PW, int RowFloat4 = kROW_LAYOUTPackedRowFloat4,
          bool RegisterNormalizer = false>
__device__ __forceinline__ void dense_forward_forward_body_c8_l2(
    Arguments args, DpaDirectNeighborView view, const long node,
    const float* shared_table) {
  using P = Profile<8, 2, false>;
  constexpr int EdgeWidth = 4;
  constexpr int Groups = 8;
  const int persist_warp = int(threadIdx.x >> 5);
  const int thread = threadIdx.x & 31;
  const int base_channel = thread & 3;
  const int group = thread >> 2;

  const int center_type = int(args.atype[args.node_begin + node]);
  const long begin = args.destination_row_ptr[node];
  const long end = begin + static_cast<const int*>(args.destination_order)[node];
  auto* records = reinterpret_cast<float4*>(const_cast<float*>(args.edge_vec));
  auto* counts = reinterpret_cast<int*>(const_cast<bool*>(args.edge_mask));
  const long live = compact_emit_row_pf<1>(
      view, args.node_begin + node, begin, end, nullptr, records, nullptr, args);
  if (thread == 0) counts[args.node_begin + node] = int(live);
  if (args.radial_modes & 8) return;

  float scalar0 = 0.0f;
  float scalar1 = 0.0f;
  float vector0 = 0.0f;
  float vector1 = 0.0f;
  float vector2 = 0.0f;
  float tensor0 = 0.0f;
  float tensor1 = 0.0f;
  float tensor2 = 0.0f;
  float tensor3 = 0.0f;
  float tensor4 = 0.0f;
  float scalar_mass = 0.0f;
  float angular_mass = 0.0f;

  for (long position = group; position < live; position += Groups) {
    float xyz[3];
    const EdgeGeometry geometry =
        geometry_from_cache(args, records[begin + position], xyz);
    const TableLocation location =
        locate_table<true>(geometry.radius, args.table_stride,
                           args.table_max, args.interval_count);
    const TableRow row = table_row(args.table, location, args.table_width);
    const long pair = long(center_type) * args.type_count + geometry.source_type;
    const float2* film_row =
        reinterpret_cast<const float2*>(args.pair_film + pair * 16) +
        base_channel;
    const float envelope = geometry.envelope;

    const float radial0 = adaptive_forward_radial<true, RowFloat4>(
        row, shared_table, location.index, base_channel, location.coordinate);
    const float2 film0 = __ldg(film_row);
    const float film_value0 = fmaf(film0.x, radial0, film0.y);
    const float amplitude0 = film_value0 * envelope;
    const float angular0 = amplitude0 * envelope;
    scalar0 += amplitude0;

    const float ux = geometry.ux;
    const float uy = geometry.uy;
    const float uz = geometry.uz;
    vector0 = fmaf(angular0, ux, vector0);
    vector1 = fmaf(angular0, uy, vector1);
    vector2 = fmaf(angular0, uz, vector2);
    tensor0 = fmaf(angular0, kSqrtThree * ux * uy, tensor0);
    tensor1 = fmaf(angular0, kSqrtThree * uy * uz, tensor1);
    tensor2 = fmaf(angular0, 0.5f * (3.0f * uz * uz - 1.0f), tensor2);
    tensor3 = fmaf(angular0, kSqrtThree * ux * uz, tensor3);
    tensor4 = fmaf(angular0,
                   0.5f * kSqrtThree * (ux * ux - uy * uy), tensor4);

    const int channel1 = base_channel + 4;
    const float radial1 = adaptive_forward_radial<true, RowFloat4>(
        row, shared_table, location.index, channel1, location.coordinate);
    const float2 film1 = __ldg(film_row + 4);
    const float film_value1 = fmaf(film1.x, radial1, film1.y);
    scalar1 += film_value1 * envelope;

    if (base_channel == 0) {
      const float squared = envelope * envelope;
      scalar_mass += squared;
      angular_mass = fmaf(squared, squared, angular_mass);
    }
  }

  scalar0 = reduce_channel_groups<EdgeWidth>(scalar0);
  scalar1 = reduce_channel_groups<EdgeWidth>(scalar1);
  vector0 = reduce_channel_groups<EdgeWidth>(vector0);
  vector1 = reduce_channel_groups<EdgeWidth>(vector1);
  vector2 = reduce_channel_groups<EdgeWidth>(vector2);
  tensor0 = reduce_channel_groups<EdgeWidth>(tensor0);
  tensor1 = reduce_channel_groups<EdgeWidth>(tensor1);
  tensor2 = reduce_channel_groups<EdgeWidth>(tensor2);
  tensor3 = reduce_channel_groups<EdgeWidth>(tensor3);
  tensor4 = reduce_channel_groups<EdgeWidth>(tensor4);

  const float total_scalar = warp_sum(scalar_mass);
  const float total_angular = warp_sum(angular_mass);
  float scalar_norm, angular_norm;
  if constexpr (RegisterNormalizer) {
    float owned_norm = 0.0f;
    float owned_divisor = 0.0f;
    if (thread < 2) {
      const float total = thread == 0 ? total_scalar : total_angular;
      const float denominator = total + args.degree_floor;
      owned_norm = rsqrtf(denominator);
      owned_divisor = sqrtf(denominator);
    }
    // All warp lanes participate; lane 0 owns scalar and lane 1 angular.
    scalar_norm = __shfl_sync(0xffffffffu, owned_norm, 0);
    angular_norm = __shfl_sync(0xffffffffu, owned_norm, 1);
    if (thread < 2) {
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputDivisor + thread,
                       owned_divisor);
    }
  } else {
    __shared__ float normalizer_bank[PW][4];
    float* normalizer = normalizer_bank[persist_warp];
    if (thread == 0) {
      normalizer[0] = rsqrtf(total_scalar + args.degree_floor);
      normalizer[1] = rsqrtf(total_angular + args.degree_floor);
      normalizer[2] = sqrtf(total_scalar + args.degree_floor);
      normalizer[3] = sqrtf(total_angular + args.degree_floor);
    }
    __syncwarp();
    scalar_norm = normalizer[0];
    angular_norm = normalizer[1];
    if (thread < 2) {
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputDivisor + thread,
                       normalizer[2 + thread]);
    }
  }

  // DIRECT_FORWARD_STATE: the four channel lanes already own all 40 reduced moments.
  // Write each four-value component group directly to the final state row,
  // avoiding the warp-local shared round trip and its synchronization.
  const long state_offset = node * P::StateWidth;
  float* state_row = args.state_out + state_offset;
  if (thread < EdgeWidth) {
    state_row[P::ScalarOffset + base_channel] = scalar0 * scalar_norm;
    state_row[P::ScalarOffset + base_channel + 4] = scalar1 * scalar_norm;
    state_row[P::VectorOffset + base_channel] = vector0 * angular_norm;
    state_row[P::VectorOffset + 4 + base_channel] = vector1 * angular_norm;
    state_row[P::VectorOffset + 8 + base_channel] = vector2 * angular_norm;
    state_row[P::TensorOffset + base_channel] = tensor0 * angular_norm;
    state_row[P::TensorOffset + 4 + base_channel] = tensor1 * angular_norm;
    state_row[P::TensorOffset + 8 + base_channel] = tensor2 * angular_norm;
    state_row[P::TensorOffset + 12 + base_channel] = tensor3 * angular_norm;
    state_row[P::TensorOffset + 16 + base_channel] = tensor4 * angular_norm;
  }
  if (thread == 0) {
    state_row[P::MomentWidth] = scalar_norm;
    state_row[P::MomentWidth + 1] = angular_norm;
  }
}

// SMALL_MODEL transports the fixed forward radial table directly from global to
// shared memory on SM80.  The destination bytes and the synchronization point
// are identical to the original scalar/float4 copy; no model arithmetic or
// table representation changes.
__device__ __forceinline__ void small_model_copy_word_async(
    float4* destination, const float4* source) {
  const unsigned shared_address =
      static_cast<unsigned>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                   "r"(shared_address), "l"(source));
}

template <int RowFloat4>
__device__ __forceinline__ void small_model_copy_forward_table_async(
    float* destination_float, const float* source_float) {
  auto* destination = reinterpret_cast<float4*>(destination_float);
  const auto* source = reinterpret_cast<const float4*>(source_float);
  constexpr int words = 600 * 8;
  constexpr int words_per_group_per_thread = 8;
  for (int base = threadIdx.x; base < words;
       base += blockDim.x * words_per_group_per_thread) {
#pragma unroll
    for (int lane_word = 0; lane_word < words_per_group_per_thread;
         ++lane_word) {
      const int item = base + lane_word * blockDim.x;
      if (item < words) {
        const int destination_item =
            RowFloat4 == kROW_LAYOUTPackedRowFloat4
                ? item
                : (item >> 3) * RowFloat4 + (item & 7);
        small_model_copy_word_async(destination + destination_item, source + item);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
}

template<int W,int Capacity,bool Shared,int Traversal,int Depth,
         int RowFloat4 = kROW_LAYOUTPackedRowFloat4>
__global__ __launch_bounds__(W*32,(Traversal==2?1:2)) void adaptive_forward_forward(Arguments a,DpaDirectNeighborView v,NarrowState* state,const float* cubic) {
  if(state->disabled)return;
  extern __shared__ __align__(16) float table_cache[];
  if constexpr(Shared)
    small_model_copy_forward_table_async<RowFloat4>(table_cache, cubic);
  const int warp=threadIdx.x/32;
  if constexpr(Traversal==2) {
    const long n=long(blockIdx.x)*W+warp;
    if(n<a.node_count)dense_forward_forward_body_c8_l2<W, RowFloat4,
          (W == 28 || W == 32) && RowFloat4 == kROW_LAYOUTPaddedRowFloat4>(
              a,v,n,table_cache);
  } else if constexpr(Traversal==1) {
    const long tiles=(a.node_count+W-1)/W;
    const long lo=(tiles*blockIdx.x/gridDim.x)*W,hi=(tiles*(blockIdx.x+1)/gridDim.x)*W;
    for(long n=lo+warp;n<hi&&n<a.node_count;n+=W)
      dense_forward_forward_body_c8_l2<W, RowFloat4,
          (W == 28 || W == 32) && RowFloat4 == kROW_LAYOUTPaddedRowFloat4>(
              a,v,n,table_cache);
  } else {
    for(long n=long(blockIdx.x)*W+warp;n<a.node_count;n+=long(gridDim.x)*W)
      dense_forward_forward_body_c8_l2<W, RowFloat4,
          (W == 28 || W == 32) && RowFloat4 == kROW_LAYOUTPaddedRowFloat4>(
              a,v,n,table_cache);
  }
}

template<int W,int RowFloat4>
__global__ __launch_bounds__(W*32,2) void forward_pipeline_forward_window(
    Arguments a,DpaDirectNeighborView v,NarrowState* state,const float* cubic,
    long begin,long count) {
  if(state->disabled)return;
  extern __shared__ __align__(16) float table_cache[];
  small_model_copy_forward_table_async<RowFloat4>(table_cache,cubic);
  const int warp=threadIdx.x/32;
  const long tiles=(count+W-1)/W;
  const long lo=begin+(tiles*blockIdx.x/gridDim.x)*W;
  const long hi=begin+(tiles*(blockIdx.x+1)/gridDim.x)*W;
  for(long n=lo+warp;n<hi&&n<begin+count;n+=W)
    dense_forward_forward_body_c8_l2<
        W, RowFloat4, W == 32 && RowFloat4 == kROW_LAYOUTPackedRowFloat4>(
            a,v,n,table_cache);
}

__global__ void forward_pipeline_fallback_forward_window(
    Arguments a,DpaDirectNeighborView v,NarrowState* state,
    long begin,long count) {
  if(!state->disabled)return;
  const long end=begin+count;
  for(long i=begin+blockIdx.x;i<end;i+=gridDim.x)
    raw_forward_body<8,2,false,false,true,std::uint32_t,4>(a,v,i);
}

template<int RowFloat4>
static void forward_pipeline_launch_forward_window_layout(
    int warps,int blocks,Arguments a,DpaDirectNeighborView v,
    NarrowState* state,const float* cubic,long begin,long count,
    cudaStream_t stream) {
  constexpr int shared=row_layout_forward_shared_bytes<RowFloat4>();
  if(warps==16)
    forward_pipeline_forward_window<16,RowFloat4><<<blocks,512,shared,stream>>>(
        a,v,state,cubic,begin,count);
  else if(warps==21)
    forward_pipeline_forward_window<21,RowFloat4><<<blocks,672,shared,stream>>>(
        a,v,state,cubic,begin,count);
  else if(warps==28)
    forward_pipeline_forward_window<28,RowFloat4><<<blocks,896,shared,stream>>>(
        a,v,state,cubic,begin,count);
  else
    forward_pipeline_forward_window<32,RowFloat4><<<blocks,1024,shared,stream>>>(
        a,v,state,cubic,begin,count);
}

static const void* forward_pipeline_forward_function(int warps,bool padded) {
  if(padded) {
    if(warps==16)return reinterpret_cast<const void*>(
        forward_pipeline_forward_window<16,kROW_LAYOUTPaddedRowFloat4>);
    if(warps==21)return reinterpret_cast<const void*>(
        forward_pipeline_forward_window<21,kROW_LAYOUTPaddedRowFloat4>);
    if(warps==28)return reinterpret_cast<const void*>(
        forward_pipeline_forward_window<28,kROW_LAYOUTPaddedRowFloat4>);
    return reinterpret_cast<const void*>(
        forward_pipeline_forward_window<32,kROW_LAYOUTPaddedRowFloat4>);
  }
  if(warps==16)return reinterpret_cast<const void*>(
      forward_pipeline_forward_window<16,kROW_LAYOUTPackedRowFloat4>);
  if(warps==21)return reinterpret_cast<const void*>(
      forward_pipeline_forward_window<21,kROW_LAYOUTPackedRowFloat4>);
  if(warps==28)return reinterpret_cast<const void*>(
      forward_pipeline_forward_window<28,kROW_LAYOUTPackedRowFloat4>);
  return reinterpret_cast<const void*>(
      forward_pipeline_forward_window<32,kROW_LAYOUTPackedRowFloat4>);
}

__global__ void hermite_build_forward_cubic(const float* source, float h,
                                        float4* destination) {
  const int item = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= 600 * 8) return;
  const int interval = item >> 3;
  const int channel = item & 7;
  const float* row = source + interval * 48;
  const float4 low = reinterpret_cast<const float4*>(row)[channel];
  const float2 high = reinterpret_cast<const float2*>(row + 32)[channel];
  const double hd = double(h), h2 = hd * hd, h3 = h2 * hd;
  const float c2 = float(double(low.z) - h2 * double(high.x) -
                         2.0 * h3 * double(high.y));
  const float c3 = float(double(low.w) + 2.0 * hd * double(high.x) +
                         3.0 * h2 * double(high.y));
  destination[item] = make_float4(low.x, low.y, c2, c3);
}

static const float* hermite_forward_cubic(const Arguments& a,
                                      cudaStream_t stream) {
  static float4* table = nullptr;
  static const float* source = nullptr;
  static float source_stride = 0.0f;
  if (!table || source != a.table || source_stride != a.table_stride) {
    cudaStreamCaptureStatus capture;
    narrow_check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) return nullptr;
    float4* next = nullptr;
    narrow_check(cudaMalloc(&next, 600 * 8 * sizeof(float4)));
    table = next;
    source = a.table;
    source_stride = a.table_stride;
    hermite_build_forward_cubic<<<19, 256, 0, stream>>>(a.table, a.table_stride,
                                                    table);
    narrow_check(cudaGetLastError());
  }
  return reinterpret_cast<const float*>(table);
}

static const void* hermite_forward_function(int warps, bool padded) {
  if (padded) {
    if (warps == 16) return reinterpret_cast<const void*>(
        adaptive_forward_forward<16, 0, true, 1, 1, kROW_LAYOUTPaddedRowFloat4>);
    if (warps == 21) return reinterpret_cast<const void*>(
        adaptive_forward_forward<21, 0, true, 1, 1, kROW_LAYOUTPaddedRowFloat4>);
    if (warps == 28) return reinterpret_cast<const void*>(
        adaptive_forward_forward<28, 0, true, 1, 1, kROW_LAYOUTPaddedRowFloat4>);
    return reinterpret_cast<const void*>(
        adaptive_forward_forward<32, 0, true, 1, 1, kROW_LAYOUTPaddedRowFloat4>);
  }
  if (warps == 16) return reinterpret_cast<const void*>(
      adaptive_forward_forward<16, 0, true, 1, 1, kROW_LAYOUTPackedRowFloat4>);
  if (warps == 21) return reinterpret_cast<const void*>(
      adaptive_forward_forward<21, 0, true, 1, 1, kROW_LAYOUTPackedRowFloat4>);
  if (warps == 28) return reinterpret_cast<const void*>(
      adaptive_forward_forward<28, 0, true, 1, 1, kROW_LAYOUTPackedRowFloat4>);
  return reinterpret_cast<const void*>(
      adaptive_forward_forward<32, 0, true, 1, 1, kROW_LAYOUTPackedRowFloat4>);
}

template <int RowFloat4>
static void hermite_launch_forward_layout(int warps, int blocks,
                                      const Arguments& a,
                                      const DpaDirectNeighborView& view,
                                      NarrowState* state, const float* cubic,
                                      cudaStream_t stream) {
  constexpr int SharedBytes = row_layout_forward_shared_bytes<RowFloat4>();
  if (warps == 16)
    adaptive_forward_forward<16, 0, true, 1, 1, RowFloat4>
        <<<blocks, 512, SharedBytes, stream>>>(a, view, state, cubic);
  else if (warps == 21)
    adaptive_forward_forward<21, 0, true, 1, 1, RowFloat4>
        <<<blocks, 672, SharedBytes, stream>>>(a, view, state, cubic);
  else if (warps == 28)
    adaptive_forward_forward<28, 0, true, 1, 1, RowFloat4>
        <<<blocks, 896, SharedBytes, stream>>>(a, view, state, cubic);
  else
    adaptive_forward_forward<32, 0, true, 1, 1, RowFloat4>
        <<<blocks, 1024, SharedBytes, stream>>>(a, view, state, cubic);
}

static void hermite_launch_forward(int warps, int blocks, const Arguments& a,
                               const DpaDirectNeighborView& view,
                               NarrowState* state, const float* cubic,
                               bool padded, cudaStream_t stream) {
  if (padded)
    hermite_launch_forward_layout<kROW_LAYOUTPaddedRowFloat4>(
        warps, blocks, a, view, state, cubic, stream);
  else
    hermite_launch_forward_layout<kROW_LAYOUTPackedRowFloat4>(
        warps, blocks, a, view, state, cubic, stream);
}

static inline void forward_pipeline_launch_readout_window(
    const Arguments& input,long begin,long count,cudaStream_t stream) {
  Arguments a=input;
  a.node_count=count;
  a.node_begin=0;
  a.atype+=begin;
  a.state+=begin*42;
  a.descriptor+=begin*70;
  if (std::getenv("DPA_VECTOR_READOUT_VECTOR_READOUT") &&
      std::atoi(std::getenv("DPA_VECTOR_READOUT_VECTOR_READOUT")) != 0 &&
      count >= (std::getenv("DPA_VECTOR_READOUT_MIN_NODES")
                   ? std::max(1L,std::atol(std::getenv("DPA_VECTOR_READOUT_MIN_NODES")))
                   : 100000L))
    generated_readout_float2<<<(count+127)/128,128,0,stream>>>(a);
  else
    generated_readout<<<(count+127)/128,128,0,stream>>>(a);
}

static inline void vector_readout_launch_readout(const Arguments& a,
                                           cudaStream_t stream) {
  static const bool requested = getenv("DPA_VECTOR_READOUT_VECTOR_READOUT") &&
      atoi(getenv("DPA_VECTOR_READOUT_VECTOR_READOUT")) != 0;
  static const long min_nodes = getenv("DPA_VECTOR_READOUT_MIN_NODES")
      ? std::max(1L, atol(getenv("DPA_VECTOR_READOUT_MIN_NODES"))) : 100000L;
  const bool selected = requested && a.node_count >= min_nodes;
  if (selected)
    generated_readout_float2<<<(a.node_count + 127) / 128, 128, 0, stream>>>(a);
  else
    generated_readout<<<(a.node_count + 127) / 128, 128, 0, stream>>>(a);
  static bool reported = false;
  if (!reported && selected) {
    reported = true;
    cudaFuncAttributes attr{}; int resident = 0;
    narrow_check(cudaFuncGetAttributes(&attr, generated_readout_float2));
    narrow_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, generated_readout_float2, 128, 0));
    fprintf(stderr,
        "VECTOR_READOUT_VECTOR_READOUT active nodes=%ld regs=%d local=%zu "
        "shared=%zu resident=%d vector_state_load=2 vector_descriptor_store=2\n",
        a.node_count, attr.numRegs, attr.localSizeBytes,
        attr.sharedSizeBytes, resident);
  }
}

static int adaptive_forward_tune(const Arguments& na,const DpaDirectNeighborView& nv,
                    NarrowState* state,int& grid,const float* cubic,
                    int& warps,bool adaptive,bool padded,
                    cudaStream_t stream) {
  cudaStreamCaptureStatus capture;narrow_check(cudaStreamIsCapturing(stream,&capture));
  if(capture!=cudaStreamCaptureStatusNone)return -1;
  NarrowState hs{};narrow_check(cudaMemcpyAsync(&hs,state,sizeof(hs),cudaMemcpyDeviceToHost,stream));narrow_check(cudaStreamSynchronize(stream));
  if(hs.disabled)return 0;
  Arguments b=na;b.state=na.state_out;
  const int default_warps=warps,default_grid=grid;
  constexpr int alternate_warps=28,alternate_grid=128;
  auto launch=[&](int which){
    if(which==0)compact_forward_fast_forward<<<na.node_count,32,0,stream>>>(na,nv,state);
    else if(which==2)hermite_launch_forward(alternate_warps,alternate_grid,na,nv,state,cubic,padded,stream);
    else hermite_launch_forward(default_warps,default_grid,na,nv,state,cubic,padded,stream);
    vector_readout_launch_readout(b,stream);
  };
  const int variants=adaptive?3:2;
  for(int j=0;j<2;++j)for(int which=0;which<variants;++which)launch(which);
  cudaEvent_t begin,end_event;narrow_check(cudaEventCreate(&begin));narrow_check(cudaEventCreate(&end_event));
  std::vector<float> times[3];
  for(int round=0;round<3;++round) {
    if(adaptive) {
      for(int which:{0,1,2,2,1,0}) {
        narrow_check(cudaEventRecord(begin,stream));for(int j=0;j<4;++j)launch(which);
        narrow_check(cudaEventRecord(end_event,stream));narrow_check(cudaEventSynchronize(end_event));
        float ms=0;narrow_check(cudaEventElapsedTime(&ms,begin,end_event));times[which].push_back(ms/4);
      }
    } else {
      for(int which:{0,1,1,0}) {
        narrow_check(cudaEventRecord(begin,stream));for(int j=0;j<4;++j)launch(which);
        narrow_check(cudaEventRecord(end_event,stream));narrow_check(cudaEventSynchronize(end_event));
        float ms=0;narrow_check(cudaEventElapsedTime(&ms,begin,end_event));times[which].push_back(ms/4);
      }
    }
  }
  for(int which=0;which<variants;++which)std::sort(times[which].begin(),times[which].end());
  auto center=[&](int which){return (times[which][2]+times[which][3])*.5f;};
  const float baseline=center(0),default_ms=center(1);
  float selected_ms=default_ms,alternate_ms=0.0f;
  if(adaptive) {
    alternate_ms=center(2);
    if(alternate_ms*1.02f<default_ms) {
      warps=alternate_warps;grid=alternate_grid;selected_ms=alternate_ms;
    }
  }
  const int policy=selected_ms*1.03f<baseline;
  fprintf(stderr,
      "ADAPTIVE_FORWARD_TUNE nodes=%ld candidate_slots=%ld baseline_ms=%.9g "
      "default_ms=%.9g alternate_ms=%.9g selected_ms=%.9g "
      "selected_warps=%d selected_blocks=%d adaptive=%d speed_pct=%.6g "
      "policy=%d threshold_speed_pct=3 select_margin_pct=2 "
      "samples=6x4 per_variant initialization_only\n",
      na.node_count,na.edge_count,baseline,default_ms,alternate_ms,selected_ms,
      warps,grid,int(adaptive),100*(baseline/selected_ms-1),policy);
  narrow_check(cudaEventDestroy(begin));narrow_check(cudaEventDestroy(end_event));
  return policy;
}
void launch_raw_forward_c8(const Arguments& a,const DpaDirectNeighborView& v,cudaStream_t stream) {
  using Fn=void(*)(const Arguments&,const DpaDirectNeighborView&,cudaStream_t);
    static const long min_nodes=getenv("COMPACT_FORWARD_MIN_NODES")?std::max(4096L,atol(getenv("COMPACT_FORWARD_MIN_NODES"))):65536L;
  auto* c=notified_cache;
  if(!c || a.node_count<min_nodes || a.node_begin || a.has_spin || a.lmax!=2 ||
      a.table_width!=8 || a.interval_count!=600 || a.type_count!=119 || a.radial_modes || std::abs(v.cutoff_squared-36.)>1e-12 ||
      (c->state&&(a.node_count!=c->nodes || a.edge_count>c->edges))) {
    chain_compact_graph::launch_raw_forward_c8(a,v,stream);return;
  }
  const bool row_layout_padded = a.node_count >= kROW_LAYOUTPaddedMinNodes &&
      a.node_count <= kROW_LAYOUTPaddedMaxNodes;
  const int row_layout_shared_bytes = row_layout_padded
      ? row_layout_forward_shared_bytes<kROW_LAYOUTPaddedRowFloat4>()
      : row_layout_forward_shared_bytes<kROW_LAYOUTPackedRowFloat4>();
  static int adaptive_forward_ready=0,adaptive_forward_grid=0,adaptive_forward_warps=0;
  static bool adaptive_forward_adaptive=false;
  if(!adaptive_forward_ready) {
    cudaStreamCaptureStatus capture;narrow_check(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone){chain_compact_graph::launch_raw_forward_c8(a,v,stream);return;}
    int device=0;cudaDeviceProp prop{};narrow_check(cudaGetDevice(&device));narrow_check(cudaGetDeviceProperties(&prop,device));
    adaptive_forward_warps=getenv("HERMITE_FWD_WARPS")?atoi(getenv("HERMITE_FWD_WARPS")):21;
    if(adaptive_forward_warps!=16&&adaptive_forward_warps!=21&&adaptive_forward_warps!=28&&adaptive_forward_warps!=32)adaptive_forward_warps=21;
    adaptive_forward_grid=getenv("HERMITE_FWD_BLOCKS")?atoi(getenv("HERMITE_FWD_BLOCKS")):(adaptive_forward_warps==21?192:256);
    adaptive_forward_adaptive=getenv("DPA_ADAPTIVE_FORWARD_CONFIG")&&atoi(getenv("DPA_ADAPTIVE_FORWARD_CONFIG"))!=0;
    const void* fn=hermite_forward_function(adaptive_forward_warps,row_layout_padded);
    cudaError_t error = cudaSuccess;
    auto configure_layout = [&](const void* function, int bytes) {
      const cudaError_t status = cudaFuncSetAttribute(
          function, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
      if (error == cudaSuccess && status != cudaSuccess) error = status;
    };
    configure_layout(
        hermite_forward_function(adaptive_forward_warps, false),
        row_layout_forward_shared_bytes<kROW_LAYOUTPackedRowFloat4>());
    configure_layout(
        hermite_forward_function(adaptive_forward_warps, true),
        row_layout_forward_shared_bytes<kROW_LAYOUTPaddedRowFloat4>());
    configure_layout(
        forward_pipeline_forward_function(adaptive_forward_warps, false),
        row_layout_forward_shared_bytes<kROW_LAYOUTPackedRowFloat4>());
    configure_layout(
        forward_pipeline_forward_function(adaptive_forward_warps, true),
        row_layout_forward_shared_bytes<kROW_LAYOUTPaddedRowFloat4>());
    if (adaptive_forward_adaptive) {
      configure_layout(
          hermite_forward_function(28, false),
          row_layout_forward_shared_bytes<kROW_LAYOUTPackedRowFloat4>());
      configure_layout(
          hermite_forward_function(28, true),
          row_layout_forward_shared_bytes<kROW_LAYOUTPaddedRowFloat4>());
      configure_layout(
          forward_pipeline_forward_function(28, false),
          row_layout_forward_shared_bytes<kROW_LAYOUTPackedRowFloat4>());
      configure_layout(
          forward_pipeline_forward_function(28, true),
          row_layout_forward_shared_bytes<kROW_LAYOUTPaddedRowFloat4>());
    }
    adaptive_forward_ready=error==cudaSuccess?1:-1;if(error!=cudaSuccess)cudaGetLastError();
    cudaFuncAttributes attr{};int resident=0;
    if(adaptive_forward_ready>0){narrow_check(cudaFuncGetAttributes(&attr,fn));narrow_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident,fn,adaptive_forward_warps*32,row_layout_shared_bytes));}
    fprintf(stderr,"ROW_LAYOUT_FWD_SETUP active=%d grid=%d warps=%d adaptive=%d padded=%d regs=%d local=%zu static_shared=%zu dynamic_shared=%d theoretical_resident_blocks=%d small_model_cp_async=1\n",adaptive_forward_ready,adaptive_forward_grid,adaptive_forward_warps,int(adaptive_forward_adaptive),int(row_layout_padded),attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,row_layout_shared_bytes,resident);
  }
  const float* hermite_cubic=hermite_forward_cubic(a,stream);
  if(!hermite_cubic){chain_compact_graph::launch_raw_forward_c8(a,v,stream);return;}
  // ADAPTIVE_DELTA selects cache skin from workload; every consumer still exact-filters at 6 A.
  static const char* configured_delta = getenv("COMPACT_FORWARD_DELTA");
  static const bool adaptive_delta = getenv("DPA_ADAPTIVE_DELTA_ADAPTIVE_DELTA") &&
      atoi(getenv("DPA_ADAPTIVE_DELTA_ADAPTIVE_DELTA")) != 0;
  const long candidate_slots_per_node = a.node_count > 0
      ? (a.edge_count + a.node_count - 1) / a.node_count : 0;
  // ADAPTIVE_VACUUM treats COMPACT_FORWARD's materialized narrow view like a database vacuum:
  // sparse candidate tables tolerate a wider delta and rebuild roughly half
  // as often.  Candidate density, rather than problem-size class, predicts
  // that tradeoff; small systems remain outside COMPACT_FORWARD_MIN_NODES.
  static const bool adaptive_vacuum = getenv("DPA_ADAPTIVE_VACUUM_ADAPTIVE_VACUUM") &&
      atoi(getenv("DPA_ADAPTIVE_VACUUM_ADAPTIVE_VACUUM")) != 0;
  const bool wide_delta = candidate_slots_per_node < 180 &&
      (adaptive_vacuum || a.node_count >= 65536);
  const double delta = configured_delta ? atof(configured_delta)
      : (adaptive_delta && wide_delta ? 0.4 : 0.2);
  static const bool profiling=getenv("COMPACT_FORWARD_PROFILE") && atoi(getenv("COMPACT_FORWARD_PROFILE"))!=0;
  static const int refresh=getenv("COMPACT_FORWARD_FORCE_REFRESH")?atoi(getenv("COMPACT_FORWARD_FORCE_REFRESH")):0;
  if(!(delta>1e-5&&delta<1.0))abort();
  if(!c->state) {
    cudaStreamCaptureStatus capture;
    narrow_check(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone) {chain_compact_graph::launch_raw_forward_c8(a,v,stream);return;}
    c->nodes=a.node_count;c->edges=a.edge_count;c->stream=stream;
    const long capacity=std::max(c->nmax*2L,a.node_count*4L);
    narrow_check(cudaMalloc(&c->atoms,c->edges*sizeof(int)));
    narrow_check(cudaMalloc(&c->sources,c->edges*sizeof(std::uint32_t)));
    narrow_check(cudaMalloc(&c->counts,c->nodes*sizeof(int)));
    narrow_check(cudaMalloc(&c->reference,capacity*3*sizeof(double)));
    narrow_check(cudaMalloc(&c->state,sizeof(NarrowState)));
    NarrowState init{1,0,c->nall,capacity,0,0};
    narrow_check(cudaMemcpy(c->state,&init,sizeof(init),cudaMemcpyHostToDevice));
    static bool registered=false;if(!registered){atexit(narrow_report);registered=true;}
    fprintf(stderr,"COMPACT_FORWARD_INIT nodes=%ld capacity=%ld nall=%ld epoch=%llu delta=%.8g version=%d candidate_slots_per_node=%ld adaptive_delta=%d adaptive_vacuum=%d\n",
        c->nodes,capacity,c->nall,(unsigned long long)c->epoch,delta,6,
        candidate_slots_per_node,int(adaptive_delta),int(adaptive_vacuum));
    const void* fn=(const void*)compact_forward_fast_forward;
    cudaFuncAttributes attr{};int resident=0;
    narrow_check(cudaFuncGetAttributes(&attr,fn));
    narrow_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,32,0));
    fprintf(stderr,"COMPACT_FORWARD_RESOURCE regs=%d local_bytes=%zu shared_bytes=%zu max_resident_blocks=%d\n",
        attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,resident);
  }
  const double limit=delta*0.5-1e-7,cut=6.+delta;
  // FORWARD_PIPELINE forks exact forward windows onto a producer stream.  EXECUTION_WINDOWS waits on
  // each window before fitting/node and hands it to the force stream, yielding
  // forward -> fitting/node -> force overlap without changing arithmetic.
  static const bool forward_pipeline_requested=[] {
    const char* p=std::getenv("DPA_FORWARD_PIPELINE_FORWARD_PIPE");return p&&p[0]=='1';
  }();
  static cudaStream_t forward_pipeline_producer=nullptr;
  static cudaEvent_t forward_pipeline_start=nullptr,forward_pipeline_ready[8]{},forward_pipeline_complete=nullptr;
  cudaStreamCaptureStatus forward_pipeline_capture=cudaStreamCaptureStatusNone;
  narrow_check(cudaStreamIsCapturing(stream,&forward_pipeline_capture));
  if(forward_pipeline_requested&&!forward_pipeline_producer&&forward_pipeline_capture==cudaStreamCaptureStatusNone){
    narrow_check(cudaStreamCreateWithFlags(&forward_pipeline_producer,cudaStreamNonBlocking));
    narrow_check(cudaEventCreateWithFlags(&forward_pipeline_start,cudaEventDisableTiming));
    for(auto& event:forward_pipeline_ready)
      narrow_check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
    narrow_check(cudaEventCreateWithFlags(&forward_pipeline_complete,cudaEventDisableTiming));
    const void* forward_pipeline_fn=forward_pipeline_forward_function(adaptive_forward_warps,row_layout_padded);
    cudaFuncAttributes forward_pipeline_attr{};int forward_pipeline_resident=0;
    narrow_check(cudaFuncGetAttributes(&forward_pipeline_attr,forward_pipeline_fn));
    narrow_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &forward_pipeline_resident,forward_pipeline_fn,adaptive_forward_warps*32,row_layout_shared_bytes));
    std::fprintf(stderr,
        "FORWARD_PIPELINE_RESOURCE warps=%d regs=%d local=%zu static_shared=%zu "
        "dynamic_shared=%d resident_blocks_per_sm=%d\n",
        adaptive_forward_warps,forward_pipeline_attr.numRegs,forward_pipeline_attr.localSizeBytes,
        forward_pipeline_attr.sharedSizeBytes,row_layout_shared_bytes,forward_pipeline_resident);
  }
  const long forward_pipeline_density=a.node_count>0?
      (a.edge_count+a.node_count-1)/a.node_count:0;
  const int forward_pipeline_density_threshold=std::getenv("EXECUTION_WINDOWS_DENSITY_THRESHOLD")?
      std::max(1,std::atoi(std::getenv("EXECUTION_WINDOWS_DENSITY_THRESHOLD"))):128;
  const bool forward_pipeline_medium=a.node_count<100000;
  const int forward_pipeline_configured_windows=std::getenv("EXECUTION_WINDOWS_WINDOWS")?
      std::atoi(std::getenv("EXECUTION_WINDOWS_WINDOWS")):0;
  const int forward_pipeline_medium_windows=std::getenv("FIT_NODE_PIPELINE_WINDOWS")?
      std::atoi(std::getenv("FIT_NODE_PIPELINE_WINDOWS")):2;
  const int dense_windows_dense_windows=std::getenv("DPA_DENSE_WINDOWS_DENSE_WINDOWS")?
      std::atoi(std::getenv("DPA_DENSE_WINDOWS_DENSE_WINDOWS")):0;
  const bool dense_windows_dense_override=!forward_pipeline_medium&&
      forward_pipeline_density>=forward_pipeline_density_threshold&&dense_windows_dense_windows>0;
  const int forward_pipeline_windows=forward_pipeline_medium?
      std::max(2,std::min(8,forward_pipeline_medium_windows)):
      (dense_windows_dense_override?
       std::max(2,std::min(8,dense_windows_dense_windows)):
       (forward_pipeline_configured_windows>0?
        std::max(2,std::min(8,forward_pipeline_configured_windows)):
        (forward_pipeline_density<forward_pipeline_density_threshold?2:6)));
  const int macro_forward_requested_windows=std::getenv("DPA_MACRO_FORWARD_FORWARD_MACRO_WINDOWS")?
      std::atoi(std::getenv("DPA_MACRO_FORWARD_FORWARD_MACRO_WINDOWS")):0;
  const bool macro_forward_macro_active=!forward_pipeline_medium&&macro_forward_requested_windows>=2&&
      macro_forward_requested_windows<forward_pipeline_windows&&
      forward_pipeline_windows%macro_forward_requested_windows==0;
  const int macro_forward_producer_windows=macro_forward_macro_active?
      macro_forward_requested_windows:forward_pipeline_windows;
  if(forward_pipeline_requested&&forward_pipeline_producer&&c->adaptive_forward_policy==1&&adaptive_forward_ready>0&&
      chain_fit_node::wants_forward_pipeline(a,stream)){
    narrow_check(cudaEventRecord(forward_pipeline_start,stream));
    narrow_check(cudaStreamWaitEvent(forward_pipeline_producer,forward_pipeline_start,0));
    const int blocks=std::min<long>(2048,(c->nmax*2L+255)/256);
    const int guard_blocks=std::min<long>(4096,a.node_count);
    compact_forward_displacement<<<blocks,256,0,forward_pipeline_producer>>>(
        v,c->reference,c->state,limit*limit,refresh);
    compact_forward_build<<<guard_blocks,32,0,forward_pipeline_producer>>>(
        a,v,cut*cut,c->atoms,c->sources,c->counts,c->state);
    Arguments na=a;na.destination_order=c->counts;
    DpaDirectNeighborView nv=v;nv.candidate_atom=c->atoms;nv.source=c->sources;
    const int forward_pipeline_default_block_target=std::getenv("FORWARD_PIPELINE_FWD_BLOCKS")?
        std::max(1,std::atoi(std::getenv("FORWARD_PIPELINE_FWD_BLOCKS"))):adaptive_forward_grid;
    const int forward_block_target=macro_forward_macro_active&&
        std::getenv("DPA_MACRO_FORWARD_FORWARD_BLOCKS")?
        std::max(1,std::atoi(std::getenv("DPA_MACRO_FORWARD_FORWARD_BLOCKS"))):
        forward_pipeline_default_block_target;
    long previous=0;
    for(int w=0;w<macro_forward_producer_windows;++w){
      const long end=(w+1==macro_forward_producer_windows)?a.node_count:
          ((a.node_count*long(w+1)/macro_forward_producer_windows)/16)*16;
      const long count=end-previous;
      if(count<=0)std::abort();
      // Keep enough CTAs to preserve the original forward throughput.  The
      // window reduces work per CTA; dividing the grid by the window count
      // would underfill the 64-SM device and serialize away the pipeline gain.
      const int window_blocks=std::min<long>(
          forward_block_target,(count+adaptive_forward_warps-1)/adaptive_forward_warps);
      if(row_layout_padded)
        forward_pipeline_launch_forward_window_layout<kROW_LAYOUTPaddedRowFloat4>(
            adaptive_forward_warps,window_blocks,na,nv,c->state,hermite_cubic,previous,count,
            forward_pipeline_producer);
      else
        forward_pipeline_launch_forward_window_layout<kROW_LAYOUTPackedRowFloat4>(
            adaptive_forward_warps,window_blocks,na,nv,c->state,hermite_cubic,previous,count,
            forward_pipeline_producer);
      forward_pipeline_fallback_forward_window<<<std::min<long>(window_blocks,count),32,0,
                                    forward_pipeline_producer>>>(
          a,v,c->state,previous,count);
      Arguments readout=a;readout.state=a.state_out;
      forward_pipeline_launch_readout_window(readout,previous,count,forward_pipeline_producer);
      narrow_check(cudaEventRecord(forward_pipeline_ready[w],forward_pipeline_producer));
      previous=end;
    }
    compact_forward_save_reference<<<blocks,256,0,forward_pipeline_producer>>>(
        v,c->reference,c->state);
    compact_forward_finish<<<1,1,0,forward_pipeline_producer>>>(c->state);
    narrow_check(cudaEventRecord(forward_pipeline_complete,forward_pipeline_producer));
    cudaEvent_t macro_forward_exposed_ready[8]{};
    const cudaEvent_t* macro_forward_ready=forward_pipeline_ready;
    if(macro_forward_macro_active){
      for(int w=0;w<forward_pipeline_windows;++w){
        const int producer=std::min(
            macro_forward_producer_windows-1,
            ((w+1)*macro_forward_producer_windows+forward_pipeline_windows-1)/forward_pipeline_windows-1);
        macro_forward_exposed_ready[w]=forward_pipeline_ready[producer];
      }
      macro_forward_ready=macro_forward_exposed_ready;
    }
    chain_fit_node::set_forward_pipeline(
        forward_pipeline_windows,macro_forward_ready,forward_pipeline_complete,stream);
    static bool forward_pipeline_announced=false;
    if(!forward_pipeline_announced){
      std::fprintf(stderr,
          "FORWARD_PIPELINE_THREE_STAGE_FORWARD nodes=%ld density=%ld windows=%d "
          "producer_windows=%d forward_block_target=%d "
          "exact_multistream_graph_ready\\n",
          a.node_count,forward_pipeline_density,forward_pipeline_windows,macro_forward_producer_windows,
          forward_block_target);
      forward_pipeline_announced=true;
    }
    return;
  }
  // Capture and replay streams may differ. Producer notifications are ordered
  // through the integration's existing Kokkos-to-model event handoff.
  const int blocks=std::min<long>(2048,(c->nmax*2L+255)/256);
  static cudaEvent_t events[6];
  static bool events_ready=false;
  static long profile_calls=0,profile_samples=0;
  static double profile_sums[5]={};
  if(profiling&&!events_ready){for(auto& e:events)narrow_check(cudaEventCreate(&e));events_ready=true;}
  auto stamp=[&](int i){if(profiling)narrow_check(cudaEventRecord(events[i],stream));};
  stamp(0);
  compact_forward_displacement<<<blocks,256,0,stream>>>(v,c->reference,c->state,limit*limit,refresh);
  stamp(1);
  const int guard_blocks=std::min<long>(4096,a.node_count);
  compact_forward_build<<<guard_blocks,32,0,stream>>>(a,v,cut*cut,c->atoms,c->sources,c->counts,c->state);
  stamp(2);
  Arguments na=a;na.destination_order=c->counts;
  DpaDirectNeighborView nv=v;nv.candidate_atom=c->atoms;nv.source=c->sources;
  if(c->adaptive_forward_policy<0 && adaptive_forward_ready>0)c->adaptive_forward_policy=adaptive_forward_tune(na,nv,c->state,adaptive_forward_grid,hermite_cubic,adaptive_forward_warps,adaptive_forward_adaptive,row_layout_padded,stream);
  if(adaptive_forward_ready>0 && c->adaptive_forward_policy==1)hermite_launch_forward(adaptive_forward_warps,adaptive_forward_grid,na,nv,c->state,hermite_cubic,row_layout_padded,stream);
  else compact_forward_fast_forward<<<a.node_count,32,0,stream>>>(na,nv,c->state);
  compact_forward_fallback_forward<<<guard_blocks,32,0,stream>>>(a,v,c->state);
  stamp(3);
  compact_forward_save_reference<<<blocks,256,0,stream>>>(v,c->reference,c->state);
  compact_forward_finish<<<1,1,0,stream>>>(c->state);
  stamp(4);
  Arguments b=a;b.state=a.state_out;
  vector_readout_launch_readout(b,stream);
  stamp(5);
  if(profiling){
    narrow_check(cudaEventSynchronize(events[5]));
    if(++profile_calls>10){
      ++profile_samples;
      for(int i=0;i<5;++i){float ms;narrow_check(cudaEventElapsedTime(&ms,events[i],events[i+1]));profile_sums[i]+=ms;}
      if(profile_samples%40==0)fprintf(stderr,"COMPACT_FORWARD_TIMES samples=%ld check_us=%.5f build_us=%.5f forward_fallback_us=%.5f reference_finish_us=%.5f readout_us=%.5f\n",
         profile_samples,1000*profile_sums[0]/profile_samples,1000*profile_sums[1]/profile_samples,1000*profile_sums[2]/profile_samples,1000*profile_sums[3]/profile_samples,1000*profile_sums[4]/profile_samples);
    }
  }
}
}
}  // namespace chain_hermite_forward
