#pragma once
namespace deepmd_dpa4c {
void operator_fallback_forward(const Arguments&,const DpaDirectNeighborView&,cudaStream_t);
void operator_fallback_force(const Arguments&,const DpaDirectNeighborView&,cudaStream_t,float*,float*);
void operator_fallback_node(const Arguments&,cudaStream_t);
}
