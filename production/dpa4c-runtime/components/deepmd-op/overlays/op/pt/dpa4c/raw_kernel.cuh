#include "raw_geometry.cuh"
#include "streamed_vjp.cuh"
namespace deepmd_dpa4c {
#include "raw_half_force.cuh"
#include "compact_pf.cuh"
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

  const int thread = threadIdx.x;
  const int base_channel = thread & (EdgeWidth - 1);
  const int group = thread / EdgeWidth;
  const int leader = group * EdgeWidth;
  const unsigned mask = subwarp_mask<EdgeWidth>(leader);
  const auto* edge_index = static_cast<const index_t*>(args.edge_index);
  const auto* destination_order =
      static_cast<const index_t*>(args.destination_order);
  const int center_type = static_cast<int>(args.atype[args.node_begin + node]);
  const long begin = args.destination_row_ptr[node];
  const long end = args.destination_row_ptr[node + 1];
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

  __shared__ float4 compact_indices[256];
  const bool packed=end-begin<=256;
  auto* records=reinterpret_cast<float4*>(const_cast<float*>(args.edge_vec));
  auto* counts=reinterpret_cast<int*>(const_cast<bool*>(args.edge_mask));
  constexpr int kDepth=CM&15;
  constexpr bool kType=(CM&16)!=0&&kDepth>0;
  __shared__ unsigned char compact_types[kType?256:1];
  const bool use_ctype=kType&&packed;
  long live;
  if constexpr(kDepth==0) live=compact_emit_row(view,args.node_begin+node,begin,end,packed?compact_indices:nullptr,records);
  else live=compact_emit_row_pf<(kDepth>0?kDepth:1)>(view,args.node_begin+node,begin,end,packed?compact_indices:nullptr,records,use_ctype?compact_types:nullptr,args);
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
  __shared__ float normalizer_shared[4];
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
  __syncthreads();
  const float scalar_norm = normalizer_shared[0];
  const float angular_norm = normalizer_shared[1];
  if (thread < 2) {
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputDivisor + thread,
                     normalizer_shared[2 + thread]);
  }

  // === Step 3. Publish the normalized moments and the saved state ===
  __shared__ float moments[P::MomentWidth];
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
  __syncthreads();

  const long state_offset = node * P::StateWidth;
  for (int coordinate = thread; coordinate < P::MomentWidth;
       coordinate += P::Threads) {
    args.state_out[state_offset + coordinate] = moments[coordinate];
  }
  if (thread == 0) {
    args.state_out[state_offset + P::MomentWidth] = scalar_norm;
    args.state_out[state_offset + P::MomentWidth + 1] = angular_norm;
  }

  if(args.radial_modes & 2)return;
  // === Step 4. Align and project the wide degrees ===
  __shared__ float aligned[P::AlignedWidth];
  if (thread < P::C1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input) {
        value = fmaf(moments[P::VectorOffset + component * P::C1 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 0,
                                                    input, thread),
                     value);
      }
      aligned[component * P::C1 + thread] = value;
    }
  }
  if (thread < P::C2) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input) {
        value = fmaf(moments[P::TensorOffset + component * P::C2 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 2,
                                                    input, thread),
                     value);
      }
      aligned[3 * P::C1 + component * P::C2 + thread] = value;
    }
  }
  __syncthreads();

  __shared__ float probes[P::ProbeWidth];
  if (thread < P::K1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input) {
        value = fmaf(aligned[component * P::C1 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 4,
                                                    input, thread),
                     value);
      }
      probes[component * P::K1 + thread] = value;
    }
  }
  if (thread < P::K2) {
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input) {
        value = fmaf(aligned[3 * P::C1 + component * P::C2 + input],
                     readout_weight<Channels, Lmax>(args.readout_matrices, 6,
                                                    input, thread),
                     value);
      }
      probes[3 * P::K1 + component * P::K2 + thread] = value;
    }
  }
  __syncthreads();

  // === Step 5. Emit the invariant blocks ===
  for (int channel = thread; channel < Channels; channel += P::Threads) {
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputScalar + channel,
                     moments[P::ScalarOffset + channel]);
    store_descriptor(
        args.descriptor, args.output_mean, args.output_inv_std, node,
        P::OutputWidth, P::OutputType + channel,
        __ldg(args.type_embedding + static_cast<long>(center_type) * Channels +
              channel));
  }

  for (int pair = thread; pair < P::Gram1; pair += P::Threads) {
    int row, column;
    decode_upper_pair(pair, P::C1, row, column);
    float value = 0.0f;
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      value = fmaf(aligned[component * P::C1 + row],
                   aligned[component * P::C1 + column], value);
    }
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputGram1 + pair,
                     (row == column ? 1.0f : kSqrtTwo) * value);
  }
  for (int pair = thread; pair < P::Gram2; pair += P::Threads) {
    int row, column;
    decode_upper_pair(pair, P::C2, row, column);
    float value = 0.0f;
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      value = fmaf(aligned[3 * P::C1 + component * P::C2 + row],
                   aligned[3 * P::C1 + component * P::C2 + column], value);
    }
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputGram2 + pair,
                     (row == column ? 1.0f : kSqrtTwo) * value);
  }
  if constexpr (Lmax >= 3) {
    if (thread < Lmax - 2) {
      const int offset = P::HighOffset + (thread == 0 ? 0 : P::High3);
      const int count = 2 * (3 + thread) + 1;
      float value = 0.0f;
      for (int component = 0; component < count; ++component) {
        const float moment = moments[offset + component];
        value = fmaf(moment, moment, value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputGram3 + thread, value);
    }
  }

  for (int output = thread; output < P::Bis112; output += P::Threads) {
    const int tensor_index = output % P::K2;
    int first, second;
    decode_upper_pair(output / P::K2, P::K1, first, second);
    float packed[5];
    float left[3];
    float right[3];
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      packed[component] = probes[3 * P::K1 + component * P::K2 + tensor_index];
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      left[component] = probes[component * P::K1 + first];
      right[component] = probes[component * P::K1 + second];
    }
    float product[3];
    matrix_vector(packed_to_stf(packed), right, product);
    const float value =
        -kInvSqrtFive *
        (left[0] * product[0] + left[1] * product[1] + left[2] * product[2]);
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputBis112 + output,
                     (first == second ? 1.0f : kSqrtTwo) * value);
  }

  if (thread < P::Bis222) {
    constexpr int entries[4][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    constexpr float scales[4] = {1.0f, kSqrtThree, kSqrtThree, 1.0f};
    Matrix3 matrices[3];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      float packed[5];
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        packed[component] =
            probes[3 * P::K1 + component * P::K2 + entries[thread][axis]];
      }
      matrices[axis] = packed_to_stf(packed);
    }
    const float value =
        kBis222Scale *
        matrix_trace(matrix_product(matrix_product(matrices[0], matrices[1]),
                                    matrices[2]));
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputBis222 + thread,
                     scales[thread] * value);
  }

  if constexpr (Lmax >= 3) {
    for (int record = 0; record < args.coupling_count; ++record) {
      const int* meta = args.coupling_meta + record * 8;
      const int degree_1 = meta[0];
      const int degree_2 = meta[1];
      const int degree_3 = meta[2];
      const int nonzero_begin = meta[3];
      const int nonzero_count = meta[4];
      const int probe_begin = meta[5];
      const int probe_count = meta[6];
      const int coordinate = meta[7];
      for (int output = thread; output < probe_count; output += P::Threads) {
        const int selection = __ldg(args.coupling_entry + probe_begin + output);
        const int index_1 = selection & 0xFF;
        const int index_2 = (selection >> 8) & 0xFF;
        const int index_3 = (selection >> 16) & 0xFF;
        float value = 0.0f;
        for (int term = 0; term < nonzero_count; ++term) {
          const int components =
              __ldg(args.coupling_entry + nonzero_begin + term);
          const float weight =
              __ldg(args.coupling_value + nonzero_begin + term);
          const float first = probe_value<Channels, Lmax>(
              probes, moments, degree_1, components & 0xFF, index_1);
          const float second = probe_value<Channels, Lmax>(
              probes, moments, degree_2, (components >> 8) & 0xFF, index_2);
          const float third = probe_value<Channels, Lmax>(
              probes, moments, degree_3, (components >> 16) & 0xFF, index_3);
          value = fmaf(weight * first * second, third, value);
        }
        store_descriptor(
            args.descriptor, args.output_mean, args.output_inv_std, node,
            P::OutputWidth, coordinate + output,
            value * __ldg(args.coupling_value + probe_begin + output));
      }
    }
  }

  for (int output = thread; output < P::Quartic; output += P::Threads) {
    const int vector_index = output % P::K1;
    const int tensor_index = output / P::K1;
    float packed[5];
    float value[3];
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      packed[component] = probes[3 * P::K1 + component * P::K2 + tensor_index];
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      value[component] = probes[component * P::K1 + vector_index];
    }
    float product[3];
    matrix_vector(packed_to_stf(packed), value, product);
    store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                     node, P::OutputWidth, P::OutputQuartic + output,
                     product[0] * product[0] + product[1] * product[1] +
                         product[2] * product[2]);
  }

  // === Step 6. Emit the spin invariants of even spin order ===
  if constexpr (HasSpin) {
    for (int pair = thread; pair < P::SpinGramVector; pair += P::Threads) {
      int row, column;
      decode_upper_pair(pair, P::SpinVectorWidth, row, column);
      float value = 0.0f;
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        value =
            fmaf(spin_vector_value<Channels, Lmax>(moments, component, row),
                 spin_vector_value<Channels, Lmax>(moments, component, column),
                 value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinGramVector + pair,
                       (row == column ? 1.0f : kSqrtTwo) * value);
    }
    if (thread < P::SpinGramTensor) {
      int row, column;
      decode_spin_tensor_pair(thread, row, column);
      float value = 0.0f;
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        value =
            fmaf(spin_tensor_value<Channels, Lmax>(moments, component, row),
                 spin_tensor_value<Channels, Lmax>(moments, component, column),
                 value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinGramTensor + thread,
                       (row == column ? 1.0f : kSqrtTwo) * value);
    }
    // Cross Gram against the unaligned geometric degree-two moments. Both
    // factors have even spin order, and with a unit direction the on-site row
    // evaluates to the single-ion anisotropy sum over neighbours.
    for (int output = thread; output < P::SpinCross; output += P::Threads) {
      const int probe = output / P::C2;
      const int channel = output % P::C2;
      float value = 0.0f;
#pragma unroll
      for (int component = 0; component < 5; ++component) {
        value =
            fmaf(spin_tensor_value<Channels, Lmax>(moments, component, probe),
                 moments[P::TensorOffset + component * P::C2 + channel], value);
      }
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinCross + output,
                       value);
    }
    for (int channel = thread; channel < P::Cs; channel += P::Threads) {
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth, P::OutputSpinMagnitude + channel,
                       moments[P::SpinMagnitude + channel]);
      store_descriptor(args.descriptor, args.output_mean, args.output_inv_std,
                       node, P::OutputWidth,
                       P::OutputSpinCoordination + channel,
                       moments[P::SpinCoordination + channel]);
    }
  }
}
template <int Channels, int Lmax, bool HasModes, bool HasSpin, bool Canonical, typename index_t, int CM = 0>
__global__ __launch_bounds__(
    Profile<Channels, Lmax, HasSpin>::Threads,
    Profile<Channels, Lmax, HasSpin>::
        MinBlocksPerSm) void raw_forward_kernel(Arguments args,DpaDirectNeighborView view) {
  for (long node = blockIdx.x; node < args.node_count; node += gridDim.x) {
    raw_forward_body<Channels, Lmax, HasModes, HasSpin, Canonical, index_t, CM>(args, view, node);
    __syncthreads();
  }
}
template<int Warps,bool Streamed> __global__ void raw_force_kernel(Arguments a,DpaDirectNeighborView view,float* f,float* v) {
  using P=Profile<8,2,false>;
  const int lane=threadIdx.x&31,warp=threadIdx.x/32;
  const long i=long(blockIdx.x)*Warps+warp;
  __shared__ float own[Warps][P::StateWidth];
  if(i<a.node_count) for(int k=lane;k<P::StateWidth;k+=32) own[warp][k]=a.moment_gradient[i*P::StateWidth+k];
  __syncthreads();
  if(i>=a.node_count)return;
  const long begin=a.destination_row_ptr[i];
  const long live=reinterpret_cast<const int*>(a.edge_mask)[i];
  float force[3]={},virial[9]={};
  const auto* sources=static_cast<const std::uint32_t*>(a.edge_index);
  const int ti=int(a.atype[i]);
  for(long position=lane;position<live;position+=32) {
    EdgeGeometry g;float xyz[3];
    g=geometry_from_cache(a,reinterpret_cast<const float4*>(a.edge_vec)[begin+position],xyz);
    const std::uint32_t j=static_cast<std::uint32_t>(g.source);
    float gij[3],gji[3];
    if constexpr(Streamed)pair_vjp_streamed(a,g,own[warp],ti,a.moment_gradient+long(j)*P::StateWidth,g.source_type,gij,gji);
    else {
    const auto loc=locate_table<true>(g.radius,a.table_stride,a.table_max,a.interval_count);
    const auto row=table_row(a.table,loc,a.table_width);
    float2 radial[8];
    #pragma unroll
    for(int c=0;c<8;++c)radial[c]=evaluate_table_with_derivative(row,c,loc.coordinate,loc.clamped);

    direct_vjp(a,g,own[warp],1,ti,radial,gij);
    const int tj=g.source_type;
    g.ux=-g.ux;g.uy=-g.uy;g.uz=-g.uz;g.source_type=ti;
    direct_vjp(a,g,a.moment_gradient+long(j)*P::StateWidth,1,tj,radial,gji);
    }
    #pragma unroll
    for(int c=0;c<3;++c) {
      force[c]+=gij[c]-gji[c];
      #pragma unroll
      for(int d=0;d<3;++d)virial[c*3+d]+=gji[c]*xyz[d];
    }
  }
  #pragma unroll
  for(int offset=16;offset;offset>>=1) {
    #pragma unroll
    for(int c=0;c<3;++c)force[c]+=__shfl_down_sync(0xffffffffu,force[c],offset);
    #pragma unroll
    for(int c=0;c<9;++c)virial[c]+=__shfl_down_sync(0xffffffffu,virial[c],offset);
  }
  if(lane==0) {
    #pragma unroll
    for(int c=0;c<3;++c)f[i*3+c]=force[c];
    #pragma unroll
    for(int c=0;c<9;++c)v[i*9+c]=virial[c];
  }
}
void operator_fallback_forward(const Arguments& a,const DpaDirectNeighborView& view,cudaStream_t s) {
 constexpr int ab=0;
 Arguments b=a;b.radial_modes=ab;
 static const long fwd_npb=[]{const char* e=std::getenv("DPA_FWD_NPB");const long v=e?std::atol(e):1;return v>0?v:1;}();
 static const int fwd_cm=[]{const char* e=std::getenv("DPA_FWD_CM");return e?std::atoi(e):0;}();
 const long fwd_grid=(a.node_count+fwd_npb-1)/fwd_npb;constexpr int fwd_threads=Profile<8,2,false>::Threads;
 switch(fwd_cm){
   case 4: raw_forward_kernel<8,2,false,false,true,std::uint32_t,4><<<fwd_grid,fwd_threads,0,s>>>(b,view);break;
   case 8: raw_forward_kernel<8,2,false,false,true,std::uint32_t,8><<<fwd_grid,fwd_threads,0,s>>>(b,view);break;
   case 20: raw_forward_kernel<8,2,false,false,true,std::uint32_t,20><<<fwd_grid,fwd_threads,0,s>>>(b,view);break;
   case 24: raw_forward_kernel<8,2,false,false,true,std::uint32_t,24><<<fwd_grid,fwd_threads,0,s>>>(b,view);break;
   default: raw_forward_kernel<8,2,false,false,true,std::uint32_t><<<fwd_grid,fwd_threads,0,s>>>(b,view);
 }

}
#include "raw_force_onesided.cuh"
#include "raw_force_split.cuh"
extern int g_force_fused;
#include "raw_force_fused.cuh"
void operator_fallback_force(const Arguments& a,const DpaDirectNeighborView& view,cudaStream_t s,float* f,float* v) {
 const char* one=std::getenv("DPA_FORCE_ONESIDED");
 constexpr int ab=0;
 if(g_force_fused){cudaMemsetAsync(f,0,a.node_count*3*sizeof(float),s); const int gb=(a.node_count+1)/2;
   if(g_force_fused==2)raw_force_fused_kernel<2,2><<<gb,64,0,s>>>(a,view,f,v);
   else if(g_force_fused==8)raw_force_fused_kernel<2,8><<<gb,64,0,s>>>(a,view,f,v);
   else raw_force_fused_kernel<2,4><<<gb,64,0,s>>>(a,view,f,v);
   return;}
 if(one && one[0]=='1'){Arguments b=a;b.radial_modes=ab;launch_raw_force_onesided_c8(b,view,s,f,v);return;}
 static const bool pack=[]{const char* e=std::getenv("DPA_FORCE_PACK");return e&&e[0]=='1';}();
 if(pack && one && one[0]=='2' && !(ab&16)){Arguments b=a;b.radial_modes=ab;unsigned long long* fp=force_pack_buffer(a.node_count);
   cudaMemsetAsync(fp,0,a.node_count*2*sizeof(unsigned long long),s);
   raw_force_split_pack_kernel<2,2><<<(a.node_count+1)/2,64,0,s>>>(b,view,fp,v);
   unpack_force_kernel<<<(int)std::min<long>((a.node_count+255)/256,4096),256,0,s>>>(a.node_count,fp,f);
   return;}
 if(one && (one[0]=='2'||one[0]=='4')){Arguments b=a;b.radial_modes=ab;cudaMemsetAsync(f,0,a.node_count*3*sizeof(float),s);
   if(one[0]=='2')raw_force_split_kernel<2,2><<<(a.node_count+1)/2,64,0,s>>>(b,view,f,v);
   else raw_force_split_kernel<2,4><<<(a.node_count+1)/2,64,0,s>>>(b,view,f,v);
   return;}
 const char* w=std::getenv("DPA_STREAM_WARPS");const char* c=std::getenv("DPA_EMIT_STREAMED");
 if(c && c[0]=='1'){

   if(w && w[0]=='1')raw_force_kernel<1,true><<<a.node_count,32,0,s>>>(a,view,f,v);
   else if(w && w[0]=='2')raw_force_kernel<2,true><<<(a.node_count+1)/2,64,0,s>>>(a,view,f,v);
   else raw_force_kernel<4,true><<<(a.node_count+3)/4,128,0,s>>>(a,view,f,v);
 } else raw_force_kernel<4,false><<<(a.node_count+3)/4,128,0,s>>>(a,view,f,v);
}
}
