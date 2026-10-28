#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "graph_compress.cuh"
#include "raw_geometry.cuh"
#include "operator_fallback.h"

#include <cstdio>
#include <cstdlib>
namespace deepmd_dpa4c {
namespace chain_compact_graph {

constexpr int kModeStride=kMaxRadialModes+4;
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

template<bool Shared>
__device__ __forceinline__ float adaptive_forward_radial(const TableRow& row,const float* cache,int index,int c,float x) {
  if constexpr(!Shared)return evaluate_table(row,c,x);
  else {
    const float* p=cache+index*52;
    const float4 low=reinterpret_cast<const float4*>(p)[c];
    const float2 high=reinterpret_cast<const float2*>(p+32)[c];
    return low.x+(low.y+(low.z+(low.w+(high.x+high.y*x)*x)*x)*x)*x;
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
  int compact_graph_reported = 0;
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
static void compact_graph_narrow_geometry_epoch_unused(const void* owner,std::uint64_t packed_epoch,int nall,int nmax,void* producer_stream) {
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
template<int W,int Capacity,bool Shared,int Traversal,int Depth>
__global__ __launch_bounds__(W*32,(Traversal==2?1:2)) void adaptive_forward_forward(Arguments a,DpaDirectNeighborView v,NarrowState* state) {
  if(state->disabled)return;
  extern __shared__ float table_cache[];
  if constexpr(Shared) {
    for(int k=threadIdx.x;k<600*48;k+=blockDim.x)table_cache[(k/48)*52+k%48]=a.table[k];
    __syncthreads();
  }
  const int warp=threadIdx.x/32;
  if constexpr(Traversal==2) {
    const long n=long(blockIdx.x)*W+warp;
    if(n<a.node_count)adaptive_forward_forward_body<8,2,false,false,true,std::uint32_t,8192+Depth,W,Capacity,Shared>(a,v,n,table_cache);
  } else if constexpr(Traversal==1) {
    const long tiles=(a.node_count+W-1)/W;
    const long lo=(tiles*blockIdx.x/gridDim.x)*W,hi=(tiles*(blockIdx.x+1)/gridDim.x)*W;
    for(long n=lo+warp;n<hi&&n<a.node_count;n+=W)
      adaptive_forward_forward_body<8,2,false,false,true,std::uint32_t,8192+Depth,W,Capacity,Shared>(a,v,n,table_cache);
  } else {
    for(long n=long(blockIdx.x)*W+warp;n<a.node_count;n+=long(gridDim.x)*W)
      adaptive_forward_forward_body<8,2,false,false,true,std::uint32_t,8192+Depth,W,Capacity,Shared>(a,v,n,table_cache);
  }
}
static int adaptive_forward_tune(const Arguments& na,const DpaDirectNeighborView& nv,NarrowState* state,int grid,cudaStream_t stream) {
  cudaStreamCaptureStatus capture;narrow_check(cudaStreamIsCapturing(stream,&capture));
  if(capture!=cudaStreamCaptureStatusNone)return -1;
  NarrowState hs{};narrow_check(cudaMemcpyAsync(&hs,state,sizeof(hs),cudaMemcpyDeviceToHost,stream));narrow_check(cudaStreamSynchronize(stream));
  if(hs.disabled)return 0;
  Arguments b=na;b.state=na.state_out;
  auto launch=[&](int which){
    if(which)adaptive_forward_forward<32,0,true,1,1><<<grid,1024,124800,stream>>>(na,nv,state);
    else compact_forward_fast_forward<<<na.node_count,32,0,stream>>>(na,nv,state);
    generated_readout<<<(na.node_count+127)/128,128,0,stream>>>(b);
  };
  for(int j=0;j<2;++j){launch(0);launch(1);}
  cudaEvent_t start,end;narrow_check(cudaEventCreate(&start));narrow_check(cudaEventCreate(&end));
  std::vector<float> times[2];
  for(int round=0;round<3;++round)for(int which:{0,1,1,0}) {
    narrow_check(cudaEventRecord(start,stream));for(int j=0;j<4;++j)launch(which);
    narrow_check(cudaEventRecord(end,stream));narrow_check(cudaEventSynchronize(end));
    float ms=0;narrow_check(cudaEventElapsedTime(&ms,start,end));times[which].push_back(ms/4);
  }
  for(auto& values:times)std::sort(values.begin(),values.end());
  const float baseline=(times[0][2]+times[0][3])*.5f,candidate=(times[1][2]+times[1][3])*.5f;
  // A conservative event margin, rather than a material/atom-count whitelist.
  const int policy=candidate*1.03f<baseline;
  fprintf(stderr,"ADAPTIVE_FORWARD_TUNE nodes=%ld candidate_slots=%ld baseline_ms=%.9g batch_ms=%.9g speed_pct=%.6g policy=%d threshold_speed_pct=3 samples=6x4 per_variant initialization_only\n",na.node_count,na.edge_count,baseline,candidate,100*(baseline/candidate-1),policy);
  narrow_check(cudaEventDestroy(start));narrow_check(cudaEventDestroy(end));
  return policy;
}
void launch_raw_forward_c8(const Arguments& a,const DpaDirectNeighborView& v,cudaStream_t stream) {

  static const long min_nodes=getenv("COMPACT_FORWARD_MIN_NODES")?std::max(4096L,atol(getenv("COMPACT_FORWARD_MIN_NODES"))):65536L;
  // Cache maintenance only pays for itself once the actual padded graph has
  // enough work. This workload gate is independent of material and atom count,
  // and keeps sparse medium graphs on the original exact path.
  static const long min_candidate_slots=getenv("COMPACT_GRAPH_MIN_CANDIDATE_SLOTS")?
      std::max(0L,atol(getenv("COMPACT_GRAPH_MIN_CANDIDATE_SLOTS"))):4000000L;
  auto* c=notified_cache;
  const bool workload_ok=a.edge_count>=min_candidate_slots;
  if(c && !c->compact_graph_reported) {
    fprintf(stderr,"COMPACT_GRAPH_GATE nodes=%ld candidate_slots=%ld min_nodes=%ld min_candidate_slots=%ld active=%d workload_metric=padded_candidate_slots\n",
        a.node_count,a.edge_count,min_nodes,min_candidate_slots,
        int(a.node_count>=min_nodes && workload_ok));
    c->compact_graph_reported=1;
  }
  if(!c || a.node_count<min_nodes || !workload_ok || a.node_begin || a.has_spin || a.lmax!=2 ||
      a.table_width!=8 || a.interval_count!=600 || a.type_count!=119 || a.radial_modes || std::abs(v.cutoff_squared-36.)>1e-12 ||
      (c->state&&(a.node_count!=c->nodes || a.edge_count>c->edges))) {
    deepmd_dpa4c::operator_fallback_forward(a,v,stream);return;
  }
  static int adaptive_forward_ready=0,adaptive_forward_grid=0;
  if(!adaptive_forward_ready) {
    cudaStreamCaptureStatus capture;narrow_check(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone){deepmd_dpa4c::operator_fallback_forward(a,v,stream);return;}
    int device=0;cudaDeviceProp prop{};narrow_check(cudaGetDevice(&device));narrow_check(cudaGetDeviceProperties(&prop,device));
    adaptive_forward_grid=4*prop.multiProcessorCount;
    auto error=cudaFuncSetAttribute(adaptive_forward_forward<32,0,true,1,1>,cudaFuncAttributeMaxDynamicSharedMemorySize,124800);
    adaptive_forward_ready=error==cudaSuccess?1:-1;if(error!=cudaSuccess)cudaGetLastError();
    cudaFuncAttributes attr{};int resident=0;
    if(adaptive_forward_ready>0){narrow_check(cudaFuncGetAttributes(&attr,adaptive_forward_forward<32,0,true,1,1>));narrow_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,adaptive_forward_forward<32,0,true,1,1>,1024,124800));}
    fprintf(stderr,"ADAPTIVE_FORWARD_SETUP active=%d grid=%d warps=32 depth=1 compact_capacity=0 regs=%d local=%zu static_shared=%zu dynamic_shared=124800 theoretical_resident_blocks=%d\n",adaptive_forward_ready,adaptive_forward_grid,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,resident);
  }
  static const double delta=getenv("COMPACT_FORWARD_DELTA")?atof(getenv("COMPACT_FORWARD_DELTA")):0.2;
  static const bool profiling=getenv("COMPACT_FORWARD_PROFILE") && atoi(getenv("COMPACT_FORWARD_PROFILE"))!=0;
  static const int refresh=getenv("COMPACT_FORWARD_FORCE_REFRESH")?atoi(getenv("COMPACT_FORWARD_FORCE_REFRESH")):0;
  if(!(delta>1e-5&&delta<1.0))abort();
  if(!c->state) {
    cudaStreamCaptureStatus capture;
    narrow_check(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone) {deepmd_dpa4c::operator_fallback_forward(a,v,stream);return;}
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
    fprintf(stderr,"COMPACT_FORWARD_INIT nodes=%ld capacity=%ld nall=%ld epoch=%llu delta=%.8g version=%d\n",
        c->nodes,capacity,c->nall,(unsigned long long)c->epoch,delta,4);
    const void* fn=(const void*)compact_forward_fast_forward;
    cudaFuncAttributes attr{};int resident=0;
    narrow_check(cudaFuncGetAttributes(&attr,fn));
    narrow_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,32,0));
    fprintf(stderr,"COMPACT_FORWARD_RESOURCE regs=%d local_bytes=%zu shared_bytes=%zu max_resident_blocks=%d\n",
        attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,resident);
  }
  // Capture and replay streams may differ. Producer notifications are ordered
  // through the integration's existing Kokkos-to-model event handoff.
  const double limit=delta*0.5-1e-7,cut=6.+delta;
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
  if(c->adaptive_forward_policy<0 && adaptive_forward_ready>0)c->adaptive_forward_policy=adaptive_forward_tune(na,nv,c->state,adaptive_forward_grid,stream);
  if(adaptive_forward_ready>0 && c->adaptive_forward_policy==1)adaptive_forward_forward<32,0,true,1,1><<<adaptive_forward_grid,1024,124800,stream>>>(na,nv,c->state);
  else compact_forward_fast_forward<<<a.node_count,32,0,stream>>>(na,nv,c->state);
  compact_forward_fallback_forward<<<guard_blocks,32,0,stream>>>(a,v,c->state);
  stamp(3);
  compact_forward_save_reference<<<blocks,256,0,stream>>>(v,c->reference,c->state);
  compact_forward_finish<<<1,1,0,stream>>>(c->state);
  stamp(4);
  Arguments b=a;b.state=a.state_out;
  generated_readout<<<(a.node_count+127)/128,128,0,stream>>>(b);
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
}  // namespace chain_compact_graph
