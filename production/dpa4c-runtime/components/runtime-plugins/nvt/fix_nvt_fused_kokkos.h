#ifndef LMP_FIX_NVT_FUSED_KOKKOS_H
#define LMP_FIX_NVT_FUSED_KOKKOS_H

#include "fix_nvt_kokkos.h"

namespace LAMMPS_NS {

class FixNVTFusedKokkos : public FixNVTKokkos<LMPDeviceType> {
 public:
  FixNVTFusedKokkos(class LAMMPS *, int, char **);
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void fused_integrate(int) override;
  void post_run() override;
  void restart(char *) override;
  double compute_scalar() override;
  double compute_vector(int) override;
  int pack_restart_data(double *) override;

 private:
  bool dpa_fuse_enabled = false;
  bool cooperative_nhc_coop_enabled = false;
  bool packed_nvt_packed_current = false;
  bool packed_nvt_armed = false;
  bool routegraph_output_initial_pending = false;
  bool cooperative_nhc_integrate();
  bool cooperative_nhc_output_final();
  bool cooperative_nhc_output_initial();
  // Invalidated after every device integration phase, rather than just by
  // timestep: POST_FORCE and END_OF_STEP may read at different phases.
  bool dpa_nhc_host_current = true;
  void sync_nhc_host_read();
};

}  // namespace LAMMPS_NS

#endif
