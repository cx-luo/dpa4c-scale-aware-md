// Node VJP fused into the one-sided force: a warp owns node i, derives q_i from
// dE/dD and the moments in shared memory, then scans the live edges of i.
// Valid only when descriptor_gradient covers every node (untiled fitting).

__device__ __forceinline__ void node_vjp_warp_c8(const Arguments& args, long node, int lane, float* moments,
                                                 float* aligned, float* probes, float* d_probes, float* d_aligned,
                                                 float* q) {
  using P = Profile<8, 2, false>;
  constexpr int W = 32;
  const long state_offset = node * P::StateWidth;
  for (int k = lane; k < P::MomentWidth; k += W) moments[k] = __ldg(args.state + state_offset + k);
  const float scalar_norm = __ldg(args.state + state_offset + P::MomentWidth);
  const float angular_norm = __ldg(args.state + state_offset + P::MomentWidth + 1);
  __syncwarp();
  if (lane < P::C1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input)
        value = fmaf(moments[P::VectorOffset + component * P::C1 + input],
                     readout_weight<8, 2>(args.readout_matrices, 0, input, lane), value);
      aligned[component * P::C1 + lane] = value;
    }
  } else if (lane >= 8 && lane < 8 + P::C2) {
    const int l = lane - 8;
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input)
        value = fmaf(moments[P::TensorOffset + component * P::C2 + input],
                     readout_weight<8, 2>(args.readout_matrices, 2, input, l), value);
      aligned[3 * P::C1 + component * P::C2 + l] = value;
    }
  }
  __syncwarp();
  if (lane < P::K1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C1; ++input)
        value = fmaf(aligned[component * P::C1 + input],
                     readout_weight<8, 2>(args.readout_matrices, 4, input, lane), value);
      probes[component * P::K1 + lane] = value;
    }
  } else if (lane >= 8 && lane < 8 + P::K2) {
    const int l = lane - 8;
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int input = 0; input < P::C2; ++input)
        value = fmaf(aligned[3 * P::C1 + component * P::C2 + input],
                     readout_weight<8, 2>(args.readout_matrices, 6, input, l), value);
      probes[3 * P::K1 + component * P::K2 + l] = value;
    }
  }
  __syncwarp();
  if (lane < P::K1) {
    float d_vector[3] = {};
    for (int output = 0; output < P::Bis112; ++output) {
      const int tensor_index = output % P::K2;
      int first, second;
      decode_upper_pair(output / P::K2, P::K1, first, second);
      if (lane != first && lane != second) continue;
      float packed[5], other[3];
#pragma unroll
      for (int component = 0; component < 5; ++component)
        packed[component] = probes[3 * P::K1 + component * P::K2 + tensor_index];
      const int other_index = lane == first ? second : first;
#pragma unroll
      for (int component = 0; component < 3; ++component) other[component] = probes[component * P::K1 + other_index];
      const float gradient = -kInvSqrtFive * (first == second ? 2.0f : kSqrtTwo) *
                             load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                                                  P::OutputWidth, P::OutputBis112 + output);
      float product[3];
      matrix_vector(packed_to_stf(packed), other, product);
#pragma unroll
      for (int component = 0; component < 3; ++component)
        d_vector[component] = fmaf(gradient, product[component], d_vector[component]);
    }
    for (int tensor_index = 0; tensor_index < P::K2; ++tensor_index) {
      const float gradient = load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                                                  P::OutputWidth, P::OutputQuartic + tensor_index * P::K1 + lane);
      float packed[5], value[3];
#pragma unroll
      for (int component = 0; component < 5; ++component)
        packed[component] = probes[3 * P::K1 + component * P::K2 + tensor_index];
#pragma unroll
      for (int component = 0; component < 3; ++component) value[component] = probes[component * P::K1 + lane];
      const Matrix3 matrix = packed_to_stf(packed);
      float first[3], second[3];
      matrix_vector(matrix, value, first);
      matrix_vector(matrix, first, second);
#pragma unroll
      for (int component = 0; component < 3; ++component)
        d_vector[component] = fmaf(2.0f * gradient, second[component], d_vector[component]);
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) d_probes[component * P::K1 + lane] = d_vector[component];
  } else if (lane >= 8 && lane < 8 + P::K2) {
    const int l = lane - 8;
    float d_tensor[5] = {};
    add_bis222_probe_gradient<8, 2, false>(l, node, args, probes + 3 * P::K1, d_tensor);
    for (int output = l; output < P::Bis112; output += P::K2) {
      int first, second;
      decode_upper_pair(output / P::K2, P::K1, first, second);
      float left[3], right[3];
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        left[component] = probes[component * P::K1 + first];
        right[component] = probes[component * P::K1 + second];
      }
      const float gradient = -kInvSqrtFive * (first == second ? 1.0f : kSqrtTwo) *
                             load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                                                  P::OutputWidth, P::OutputBis112 + output);
      Matrix3 matrix{};
#pragma unroll
      for (int row = 0; row < 3; ++row)
#pragma unroll
        for (int column = 0; column < 3; ++column) matrix.value[row][column] = gradient * left[row] * right[column];
      float packed_gradient[5];
      matrix_gradient_to_packed(matrix, packed_gradient);
#pragma unroll
      for (int component = 0; component < 5; ++component) d_tensor[component] += packed_gradient[component];
    }
    for (int vector_index = 0; vector_index < P::K1; ++vector_index) {
      const float gradient = load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                                                  P::OutputWidth, P::OutputQuartic + l * P::K1 + vector_index);
      float packed[5], value[3];
#pragma unroll
      for (int component = 0; component < 5; ++component) packed[component] = probes[3 * P::K1 + component * P::K2 + l];
#pragma unroll
      for (int component = 0; component < 3; ++component) value[component] = probes[component * P::K1 + vector_index];
      float product[3];
      matrix_vector(packed_to_stf(packed), value, product);
      Matrix3 matrix{};
#pragma unroll
      for (int row = 0; row < 3; ++row)
#pragma unroll
        for (int column = 0; column < 3; ++column)
          matrix.value[row][column] = 2.0f * gradient * product[row] * value[column];
      float packed_gradient[5];
      matrix_gradient_to_packed(matrix, packed_gradient);
#pragma unroll
      for (int component = 0; component < 5; ++component) d_tensor[component] += packed_gradient[component];
    }
#pragma unroll
    for (int component = 0; component < 5; ++component) d_probes[3 * P::K1 + component * P::K2 + l] = d_tensor[component];
  }
  __syncwarp();
  if (lane < P::C1) {
    float gradient[3] = {};
    for (int other = 0; other < P::C1; ++other) {
      const float upstream = (lane == other ? 2.0f : kSqrtTwo) *
                             load_output_gradient(args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
                                                  P::OutputGram1 + gram_pair_position(lane, other, P::C1));
#pragma unroll
      for (int component = 0; component < 3; ++component)
        gradient[component] = fmaf(upstream, aligned[component * P::C1 + other], gradient[component]);
    }
    for (int probe = 0; probe < P::K1; ++probe) {
      const float weight = readout_weight<8, 2>(args.readout_matrices, 5, probe, lane);
#pragma unroll
      for (int component = 0; component < 3; ++component)
        gradient[component] = fmaf(weight, d_probes[component * P::K1 + probe], gradient[component]);
    }
#pragma unroll
    for (int component = 0; component < 3; ++component) d_aligned[component * P::C1 + lane] = gradient[component];
  } else if (lane >= 8 && lane < 8 + P::C2) {
    const int l = lane - 8;
    float gradient[5] = {};
    for (int other = 0; other < P::C2; ++other) {
      const float upstream = (l == other ? 2.0f : kSqrtTwo) *
                             load_output_gradient(args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth,
                                                  P::OutputGram2 + gram_pair_position(l, other, P::C2));
#pragma unroll
      for (int component = 0; component < 5; ++component)
        gradient[component] = fmaf(upstream, aligned[3 * P::C1 + component * P::C2 + other], gradient[component]);
    }
    for (int probe = 0; probe < P::K2; ++probe) {
      const float weight = readout_weight<8, 2>(args.readout_matrices, 7, probe, l);
#pragma unroll
      for (int component = 0; component < 5; ++component)
        gradient[component] = fmaf(weight, d_probes[3 * P::K1 + component * P::K2 + probe], gradient[component]);
    }
#pragma unroll
    for (int component = 0; component < 5; ++component)
      d_aligned[3 * P::C1 + component * P::C2 + l] = gradient[component];
  }
  __syncwarp();
  float* d_moments = q;
  if (lane < 8) {
    d_moments[P::ScalarOffset + lane] = load_output_gradient(args.descriptor_gradient, args.output_inv_std, node,
                                                             P::OutputWidth, P::OutputScalar + lane);
  } else if (lane < 8 + P::C1) {
    const int l = lane - 8;
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      float value = 0.0f;
      for (int output = 0; output < P::C1; ++output)
        value = fmaf(d_aligned[component * P::C1 + output], readout_weight<8, 2>(args.readout_matrices, 1, output, l),
                     value);
      d_moments[P::VectorOffset + component * P::C1 + l] = value;
    }
  } else if (lane < 8 + P::C1 + P::C2) {
    const int l = lane - 8 - P::C1;
#pragma unroll
    for (int component = 0; component < 5; ++component) {
      float value = 0.0f;
      for (int output = 0; output < P::C2; ++output)
        value = fmaf(d_aligned[3 * P::C1 + component * P::C2 + output],
                     readout_weight<8, 2>(args.readout_matrices, 3, output, l), value);
      d_moments[P::TensorOffset + component * P::C2 + l] = value;
    }
  }
  __syncwarp();
  float scalar_dot = 0.0f, angular_dot = 0.0f;
  for (int k = lane; k < P::MomentWidth; k += W) {
    const float product = d_moments[k] * moments[k];
    if (k < 8) scalar_dot += product;
    else angular_dot += product;
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    scalar_dot += __shfl_xor_sync(0xffffffffu, scalar_dot, offset);
    angular_dot += __shfl_xor_sync(0xffffffffu, angular_dot, offset);
  }
  const float scalar_mass_gradient =
      0.5f * scalar_norm *
      (load_output_gradient(args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth, P::OutputDivisor + 0) -
       scalar_dot * scalar_norm);
  const float angular_mass_gradient =
      0.5f * angular_norm *
      (load_output_gradient(args.descriptor_gradient, args.output_inv_std, node, P::OutputWidth, P::OutputDivisor + 1) -
       angular_dot * angular_norm);
  __syncwarp();
  for (int k = lane; k < P::MomentWidth; k += W) d_moments[k] *= k < 8 ? scalar_norm : angular_norm;
  if (lane == 0) {
    q[P::MomentWidth] = scalar_mass_gradient;
    q[P::MomentWidth + 1] = angular_mass_gradient;
  }
  __syncwarp();
}

template <int Warps, int E>
__global__ void raw_force_fused_kernel(Arguments a, DpaDirectNeighborView view, float* f, float* v) {
  using P = Profile<8, 2, false>;
  constexpr int Groups = 32 / E;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int base = lane % E, group = lane / E;
  const long i = long(blockIdx.x) * Warps + warp;
  __shared__ float own[Warps][P::StateWidth];
  __shared__ float moments[Warps][P::MomentWidth];
  __shared__ float aligned[Warps][P::AlignedWidth];
  __shared__ float probes[Warps][P::ProbeWidth];
  __shared__ float d_probes[Warps][P::ProbeWidth];
  __shared__ float d_aligned[Warps][P::AlignedWidth];
  if (i >= a.node_count) return;
  node_vjp_warp_c8(a, i, lane, moments[warp], aligned[warp], probes[warp], d_probes[warp], d_aligned[warp], own[warp]);
  const long begin = a.destination_row_ptr[i];
  const long live = reinterpret_cast<const int*>(a.edge_mask)[i];
  float force[3] = {}, virial[9] = {};
  const int ti = int(a.atype[i]);
  const float4* records = reinterpret_cast<const float4*>(a.edge_vec) + begin;
  for (long start = 0; start < live; start += Groups) {
    const long position = start + group;
    const bool valid = position < live;
    float xyz[3];
    EdgeGeometry g = geometry_from_cache(a, records[valid ? position : 0], xyz);
    float gij[3];
    pair_vjp_split<E>(a, g, own[warp], ti, g.source_type, base, valid, gij);
    if (valid && base == 0) {
      const std::uint32_t j = static_cast<std::uint32_t>(g.source);
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        force[c] += gij[c];
        atomicAdd(f + long(j) * 3 + c, -gij[c]);
#pragma unroll
        for (int d = 0; d < 3; ++d) virial[c * 3 + d] -= gij[c] * xyz[d];
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int c = 0; c < 3; ++c) force[c] += __shfl_down_sync(0xffffffffu, force[c], offset);
#pragma unroll
    for (int c = 0; c < 9; ++c) virial[c] += __shfl_down_sync(0xffffffffu, virial[c], offset);
  }
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < 3; ++c) atomicAdd(f + i * 3 + c, force[c]);
#pragma unroll
    for (int c = 0; c < 9; ++c) v[i * 9 + c] = virial[c];
  }
}
