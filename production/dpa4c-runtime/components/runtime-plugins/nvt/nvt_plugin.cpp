#include "fix_nvt_fused_kokkos.h"
#include "lammpsplugin.h"
#include "version.h"

using namespace LAMMPS_NS;

static Fix *fix_nvt_packed_nvt(LAMMPS *lmp, int narg, char **arg)
{
  return new FixNVTFusedKokkos(lmp, narg, arg);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  auto register_plugin = (lammpsplugin_regfunc) regfunc;
  lammpsplugin_t plugin{};
  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "nvt/kk";
  plugin.info = "PACKED_NVT packed-force consuming small-NVT integration";
  plugin.author = "DPA optimization";
  plugin.creator.v2 = (lammpsplugin_factory2 *) &fix_nvt_packed_nvt;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
  plugin.name = "nvt/kk/device";
  (*register_plugin)(&plugin, lmp);
}
