
#include "atom.h"
#include "atom_masks.h"
#include "compute.h"
#include "domain.h"
#include "dump_custom.h"
#include "error.h"
#include "fix.h"
#include "input.h"
#include "lammps.h"
#include "lammpsplugin.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "thermo.h"
#include "update.h"
#include "variable.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <charconv>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <memory>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace LAMMPS_NS;

namespace {

using SyncFn = void (*)(void *, int, unsigned int);
using RestartAvailableFn = int (*)(void *);
using SharedSnapshotAvailableFn = int (*)(const char *);
using SharedSnapshotRegisterFn = int (*)(void *, int64_t, int, const double *,
    const int32_t *, const int *, cudaEvent_t);
using SharedSnapshotReleaseFn = int (*)(const double *);

constexpr unsigned int ALL_ATOM_MASK = ALL_MASK;

struct DeviceViews {
  const int32_t *tag = nullptr;
  const int32_t *type = nullptr;
  const int32_t *image = nullptr;
  const double *x = nullptr;
  const double *v = nullptr;
  const double *f = nullptr;
  size_t capacity = 0;
};

template <class Type>
Type *device_pointer(void *object, size_t view_offset)
{
  auto *base = static_cast<unsigned char *>(object) + view_offset;
  return reinterpret_cast<Type *>(*reinterpret_cast<void **>(base + 8));
}

size_t view_extent(void *object, size_t view_offset, int axis)
{
  auto *base = static_cast<unsigned char *>(object) + view_offset;
  return *reinterpret_cast<size_t *>(base + 16 + axis * sizeof(size_t));
}

bool get_device_views(void *object, int count, DeviceViews &views)
{
  views.tag = device_pointer<int32_t>(object, 0x820);
  views.type = device_pointer<int32_t>(object, 0x868);
  views.image = device_pointer<int32_t>(object, 0x8f8);
  views.x = device_pointer<double>(object, 0x940);
  views.v = device_pointer<double>(object, 0x998);
  views.f = device_pointer<double>(object, 0x9f0);
  views.capacity = view_extent(object, 0x940, 0);
  if (count <= 0 || static_cast<size_t>(count) > views.capacity ||
      view_extent(object, 0x940, 1) != 3 || view_extent(object, 0x998, 1) != 3 ||
      view_extent(object, 0x9f0, 1) != 3)
    return false;
  const void *pointers[] = {views.tag, views.type, views.image, views.x, views.v, views.f};
  for (const void *pointer : pointers) {
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess ||
        attributes.type != cudaMemoryTypeDevice) {
      cudaGetLastError();
      return false;
    }
  }
  return true;
}

__global__ void pack_device_state(DeviceViews views, int count, double *values, int *invalid)
{
  const int source = blockIdx.x * blockDim.x + threadIdx.x;
  if (source >= count) return;
  const int32_t id = views.tag[source];
  if (id < 1 || id > count) {
    atomicExch(invalid, 1);
    return;
  }
  double *row = values + static_cast<size_t>(id - 1) * 14;
  row[0] = static_cast<double>(id);
  row[1] = static_cast<double>(views.type[source]);
  const size_t offset = static_cast<size_t>(source) * 3;
  row[2] = views.x[offset];
  row[3] = views.x[offset + 1];
  row[4] = views.x[offset + 2];
  row[5] = views.v[offset];
  row[6] = views.v[offset + 1];
  row[7] = views.v[offset + 2];
  row[8] = views.f[offset];
  row[9] = views.f[offset + 1];
  row[10] = views.f[offset + 2];
  const int32_t image = views.image[source];
  row[11] = static_cast<double>((image & 1023) - 512);
  row[12] = static_cast<double>(((image >> 10) & 1023) - 512);
  row[13] = static_cast<double>((image >> 20) - 512);
}

__global__ void pack_shared_device_state(DeviceViews views, const int32_t *mask,
                                          int count, double *values, int *invalid)
{
  const int source = blockIdx.x * blockDim.x + threadIdx.x;
  if (source >= count) return;
  const int32_t id = views.tag[source];
  const int32_t type = views.type[source];
  if (id < 1 || id > count || type <= 0) {
    atomicExch(invalid, 1);
    return;
  }
  double *row = values + static_cast<size_t>(id - 1) * 14;
  row[0] = static_cast<double>(id);
  row[1] = static_cast<double>(type);
  const size_t offset = static_cast<size_t>(source) * 3;
  row[2] = views.x[offset];
  row[3] = views.x[offset + 1];
  row[4] = views.x[offset + 2];
  row[5] = views.v[offset];
  row[6] = views.v[offset + 1];
  row[7] = views.v[offset + 2];
  row[8] = views.f[offset];
  row[9] = views.f[offset + 1];
  row[10] = views.f[offset + 2];
  const int32_t image = views.image[source];
  row[11] = static_cast<double>((image & 1023) - 512);
  row[12] = static_cast<double>(((image >> 10) & 1023) - 512);
  row[13] = static_cast<double>((image >> 20) - 512);
  int32_t *metadata = reinterpret_cast<int32_t *>(values + static_cast<size_t>(count) * 14);
  metadata[static_cast<size_t>(source) * 3] = id;
  metadata[static_cast<size_t>(source) * 3 + 1] = mask[source];
  metadata[static_cast<size_t>(source) * 3 + 2] = image;
}

class DumpCustomFastDEVICE_DUMP;
DumpCustomFastDEVICE_DUMP *active_device_dump = nullptr;

constexpr const char *EXPECTED_COLUMNS =
    "id type x y z vx vy vz fx fy fz ix iy iz";

struct Field {
  int kind;
  int precision;
  bool space;
};

bool parse(const char *format, bool is_int, bool is_double, Field &field)
{
  std::string f(format);
  field.space = !f.empty() && f.back() == ' ';
  if (field.space) f.pop_back();
  if (is_int && f == "%d") {
    field.kind = 0;
    return true;
  }
  if (!is_double) return false;
  field.kind = 1;
  if (f == "%g") {
    field.precision = 6;
    return true;
  }
  if (f.size() < 4 || f.compare(0, 2, "%.") != 0 || f.back() != 'g') return false;
  const std::string digits = f.substr(2, f.size() - 3);
  if (digits.empty() || digits.size() > 2 ||
      digits.find_first_not_of("0123456789") != std::string::npos)
    return false;
  field.precision = std::atoi(digits.c_str());
  return field.precision >= 1 && field.precision <= 17;
}

char *put_double(char *p, double value, int precision)
{
  if (std::isfinite(value))
    return std::to_chars(p, p + 48, value, std::chars_format::general, precision).ptr;
  return p + std::snprintf(p, 48, "%.*g", precision, value);
}

class DumpCustomFastDEVICE_DUMP : public DumpCustom {
 public:
  DumpCustomFastDEVICE_DUMP(LAMMPS *lmp, int narg, char **arg) : DumpCustom(lmp, narg, arg)
  {
    const char *enabled = std::getenv("DPA_FAST_BINARY_DUMP");
    fast_binary = enabled && std::strcmp(enabled, "0") != 0;
    const char *shared = std::getenv("DPA_SHARED_OUTPUT_SNAPSHOT");
    shared_output_snapshot = shared && std::strcmp(shared, "0") != 0;
    timing = std::getenv("DPA_FAST_DUMP_TIMING") != nullptr;
    validate = std::getenv("DPA_FAST_BINARY_DUMP_VALIDATE") != nullptr;
    async_binary = std::getenv("DPA_ASYNC_DUMP_ASYNC_BINARY_DUMP") != nullptr;
    device_binary = std::getenv("DPA_DEVICE_DUMP_DEVICE_BINARY_DUMP") != nullptr;
    sync_downgrade = std::getenv("DPA_DEVICE_DUMP_OUTPUT_SYNC_DOWNGRADE") != nullptr;
    fast_scalar = std::getenv("DPA_SCALAR_DUMP_FAST_SCALAR_DUMP") != nullptr;
    validate_scalar = std::getenv("DPA_SCALAR_DUMP_FAST_SCALAR_VALIDATE") != nullptr;
    if (const char *value = std::getenv("DPA_FAST_BINARY_DUMP_MIN_NODES"))
      minimum_nodes = std::max(0, std::atoi(value));
    if (const char *value = std::getenv("DPA_FAST_BINARY_DUMP_THREADS"))
      threads = std::clamp(std::atoi(value), 1, 64);
    if (const char *value = std::getenv("DPA_ASYNC_DUMP_ASYNC_BINARY_DUMP_MIN_NODES"))
      async_minimum_nodes = std::max(0, std::atoi(value));
    if (const char *value = std::getenv("DPA_DEVICE_DUMP_DEVICE_BINARY_DUMP_MIN_NODES"))
      device_minimum_nodes = std::max(0, std::atoi(value));
    if (const char *value = std::getenv("DPA_SCALAR_DUMP_FAST_SCALAR_MIN_NODES"))
      scalar_minimum_nodes = std::max(0, std::atoi(value));
    if (device_binary && sync_downgrade && binary && igroup == 0 && size_one == 14 &&
        nfield == 14 && columns && std::strcmp(columns, EXPECTED_COLUMNS) == 0)
      active_device_dump = this;
  }

  ~DumpCustomFastDEVICE_DUMP() override
  {
    if (active_device_dump == this) active_device_dump = nullptr;
    stop_writer();
  }

  void write() override
  {
    if (fast_scalar && atom->nlocal >= scalar_minimum_nodes && scalar_eligible()) {
      write_fast_scalar();
      return;
    }
    if (!eligible()) {
      if (!timing) return DumpCustom::write();
      t_convert = t_data = 0.0;
      const auto begin = Clock::now();
      DumpCustom::write();
      std::fprintf(stderr,
                   "BINARY_STATE_FAST_DUMP fallback total_ms=%.3f convert_ms=%.3f data_ms=%.3f\n",
                   milliseconds(Clock::now() - begin), t_convert, t_data);
      return;
    }
    if (device_binary && async_binary && flush_flag &&
        atom->nlocal >= device_minimum_nodes)
      write_device_binary();
    else if (async_binary && flush_flag && atom->nlocal >= async_minimum_nodes)
      write_async_binary();
    else write_fast_binary();
  }

  bool replace_output_sync(void *object, int space, unsigned int mask,
                           unsigned int &replacement)
  {
    if (!device_binary || !sync_downgrade || validate || space != static_cast<int>(Host) ||
        mask != ALL_ATOM_MASK || object != static_cast<void *>(atom) ||
        atom->nlocal < device_minimum_nodes || !eligible())
      return false;
    const bigint step = update->ntimestep;
    if (output->next != step || output->ndump < 1 || output->any_time_dumps ||
        output->var_thermo || modify->n_energy_global || !ensure_device_ready())
      return false;
    bool owns_scheduled_device_dump = false;
    for (int index = 0; index < output->ndump; ++index) {
      if (output->next_dump[index] != step) continue;
      auto *candidate = dynamic_cast<DumpCustomFastDEVICE_DUMP *>(output->dump[index]);
      if (candidate == this) {
        owns_scheduled_device_dump = true;
        continue;
      }
      if (!candidate || !candidate->fast_scalar ||
          atom->nlocal < candidate->scalar_minimum_nodes ||
          !candidate->scalar_eligible())
        return false;
    }
    if (!owns_scheduled_device_dump) return false;
    const bool restart_due = output->restart_flag && step >= output->next_restart;
    if (restart_due) {
      static auto restart_available = reinterpret_cast<RestartAvailableFn>(
          dlsym(RTLD_DEFAULT, "dpa_device_restart_device_restart_available"));
      if (!restart_available || !restart_available(static_cast<void *>(atom))) return false;
    }

    replacement = 0;
    if (output->next_thermo == step) {
      if (!output->thermo) return false;
      replacement |= output->thermo->host_atom_read_mask();
      for (auto *compute : modify->get_compute_list())
        if (compute->execution_space == Host) replacement |= compute->datamask_read;
      for (auto *fix : modify->get_fix_list())
        if (fix->execution_space == Host) replacement |= fix->datamask_read;
    }
    ++downgraded_syncs;
    if (timing)
      std::fprintf(stderr,
                   "DEVICE_DUMP_OUTPUT_SYNC step=%lld original=0x%x replacement=0x%x count=%lld\n",
                   static_cast<long long>(step), mask, replacement,
                   static_cast<long long>(downgraded_syncs));
    return true;
  }

  bool is_output_sync(void *object, int space, unsigned int mask) const
  {
    const bigint step = update->ntimestep;
    return object == static_cast<void *>(atom) && space == static_cast<int>(Host) &&
        mask == ALL_ATOM_MASK && output->next == step && output->ndump > 0 &&
        output->next_dump_any == step;
  }

  void report_original_output_sync(double elapsed_ms) const
  {
    if (!timing) return;
    std::fprintf(stderr,
                 "DEVICE_DUMP_ORIGINAL_OUTPUT_SYNC step=%lld elapsed_ms=%.3f dumps=%d "
                 "thermo=%d restart=%d energy_global=%d\n",
                 static_cast<long long>(update->ntimestep), elapsed_ms, output->ndump,
                 output->next_thermo == update->ntimestep ? 1 : 0,
                 output->restart_flag && update->ntimestep >= output->next_restart ? 1 : 0,
                 modify->n_energy_global);
  }

 protected:
  using Clock = std::chrono::steady_clock;
  bool fast_binary = false;
  bool shared_output_snapshot = false;
  bool timing = false;
  bool validate = false;
  bool async_binary = false;
  bool device_binary = false;
  bool sync_downgrade = false;
  bool device_ready = false;
  bool fast_scalar = false;
  bool validate_scalar = false;
  bool scalar_layout_checked = false;
  bool scalar_validated = false;
  bool validated_once = false;
  bool eligibility_reported = false;
  int minimum_nodes = 16384;
  int async_minimum_nodes = 100000;
  int device_minimum_nodes = 32768;
  int scalar_minimum_nodes = 16384;
  int threads = 8;
  long long downgraded_syncs = 0;
  double t_convert = 0.0;
  double t_data = 0.0;
  std::vector<std::unique_ptr<char[]>> chunks;
  std::vector<size_t> capacity;

  struct AsyncFrame {
    std::vector<double> values;
    int value_count = 0;
    double *device_values = nullptr;
    double *pinned_values = nullptr;
    int *device_invalid = nullptr;
    int *pinned_invalid = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t packed_event = nullptr;
    cudaEvent_t copied_event = nullptr;
    size_t device_capacity = 0;
    bool gpu = false;
    bool shared_snapshot = false;
  };
  std::mutex writer_mutex;
  std::condition_variable writer_cv;
  std::deque<std::unique_ptr<AsyncFrame>> writer_queue;
  std::deque<std::unique_ptr<AsyncFrame>> free_frames;
  std::thread writer_thread;
  bool writer_started = false;
  bool writer_stopping = false;
  bool writer_busy = false;
  bool writer_failed = false;
  int allocated_frames = 0;
  DeviceViews device_views;
  void *cached_atom_kokkos = nullptr;
  std::vector<int> scalar_equal_indices;

  static double milliseconds(Clock::duration duration)
  {
    return std::chrono::duration<double, std::milli>(duration).count();
  }

  template <class Function>
  void parallel_for(int count, Function function)
  {
    const int use_threads = count < 32768 ? 1 : std::min(threads, count);
    if (use_threads == 1) return function(0, count);
    std::vector<std::thread> workers;
    workers.reserve(use_threads);
    for (int thread = 0; thread < use_threads; ++thread) {
      const int begin = static_cast<int>(static_cast<long long>(count) * thread / use_threads);
      const int end = static_cast<int>(static_cast<long long>(count) * (thread + 1) / use_threads);
      workers.emplace_back(function, begin, end);
    }
    for (auto &worker : workers) worker.join();
  }

  bool scalar_eligible()
  {
    if (scalar_layout_checked) return !scalar_equal_indices.empty();
    scalar_layout_checked = true;
    if (binary || compressed || multifile || multiproc || nprocs != 1 || me != 0 ||
        !filewriter || !fp || !id || std::strcmp(id, "organizer_scalars") != 0 ||
        idregion || nthresh || ncompute || nfix || ncustom || triclinic_general ||
        nfield < 12 || nfield > 64 || size_one != nfield || nvariable != nfield - 1 ||
        !columns || !atom->tag || !atom->mask)
      return false;

    std::string expected = "id";
    scalar_equal_indices.reserve(static_cast<size_t>(nvariable));
    for (int index = 0; index < nvariable; ++index) {
      const std::string scalar = "organizer_scalar_" + std::to_string(index);
      const std::string global = "organizer_global_" + std::to_string(index);
      expected += " v_" + scalar;
      if (!id_variable[index] || scalar != id_variable[index] ||
          field2index[index + 1] != index || vtype[index + 1] != Dump::DOUBLE) {
        scalar_equal_indices.clear();
        return false;
      }
      const int variable_index = input->variable->find(global.c_str());
      if (variable_index < 0 || !input->variable->equalstyle(variable_index)) {
        scalar_equal_indices.clear();
        return false;
      }
      scalar_equal_indices.push_back(variable_index);
    }
    if (vtype[0] != Dump::INT || expected != columns) {
      scalar_equal_indices.clear();
      return false;
    }
    int selected = 0;
    int selected_tag_one = 0;
    for (int index = 0; index < atom->nlocal; ++index) {
      if (!(atom->mask[index] & groupbit)) continue;
      ++selected;
      if (atom->tag[index] == 1) ++selected_tag_one;
    }
    if (selected != 1 || selected_tag_one != 1) scalar_equal_indices.clear();
    return !scalar_equal_indices.empty();
  }

  void fill_fast_scalar(double *target)
  {
    target[0] = 1.0;
    for (size_t index = 0; index < scalar_equal_indices.size(); ++index)
      target[index + 1] = input->variable->compute_equal(scalar_equal_indices[index]);
  }

  void validate_fast_scalar(const double *candidate)
  {
    nme = count();
    ntotal = nme;
    if (nme != 1)
      error->all(FLERR, Error::NOLASTLINE, "SCALAR_DUMP scalar dump stock count is not one");
    allocate_fast_buffer(nme);
    if (nme > maxids) {
      maxids = nme;
      memory->destroy(ids);
      memory->create(ids, maxids, "dump:ids");
    }
    pack(ids);
    if (nme > 1) sort();
    double maximum_scaled_error = 0.0;
    for (int field = 0; field < nfield; ++field) {
      const double scale = std::max(1.0, std::abs(buf[field]));
      const double scaled = std::abs(candidate[field] - buf[field]) / scale;
      maximum_scaled_error = std::max(maximum_scaled_error, scaled);
      if (!std::isfinite(candidate[field]) ||
          scaled > 64.0 * std::numeric_limits<double>::epsilon())
        error->all(FLERR, Error::NOLASTLINE,
                   "SCALAR_DUMP direct scalar differs from stock atom-variable evaluation");
    }
    scalar_validated = true;
    if (timing)
      std::fprintf(stderr, "SCALAR_DUMP_SCALAR_VALIDATION max_scaled_error=%.17g\n",
                   maximum_scaled_error);
  }

  void write_fast_scalar()
  {
    const auto total_begin = Clock::now();
    set_box_bounds();
    if (fp) clearerr(fp);
    nme = ntotal = 1;
    allocate_fast_buffer(1);
    std::vector<double> candidate(static_cast<size_t>(nfield));
    const auto eval_begin = Clock::now();
    fill_fast_scalar(candidate.data());
    const auto evaluated = Clock::now();
    if (validate_scalar && !scalar_validated)
      validate_fast_scalar(candidate.data());
    std::memcpy(buf, candidate.data(), static_cast<size_t>(nfield) * sizeof(double));
    if (filewriter && write_header_flag) write_header(ntotal);
    const int text_bytes = DumpCustom::convert_string(1, buf);
    if (text_bytes < 0)
      error->one(FLERR, Error::NOLASTLINE, "SCALAR_DUMP scalar output line is too long");
    DumpCustom::write_data(text_bytes, reinterpret_cast<double *>(sbuf));
    if (flush_flag && fp) std::fflush(fp);
    if (fp && ferror(fp))
      error->one(FLERR, Error::NOLASTLINE, "Error writing dump {}: {}", id,
                 utils::getsyserror());
    if (timing)
      std::fprintf(stderr,
                   "SCALAR_DUMP_FAST_SCALAR total_ms=%.3f eval_ms=%.3f write_ms=%.3f "
                   "values=%d validated=%d\n",
                   milliseconds(Clock::now() - total_begin),
                   milliseconds(evaluated - eval_begin),
                   milliseconds(Clock::now() - evaluated), nfield,
                   scalar_validated ? 1 : 0);
  }

  void release_gpu_frame(AsyncFrame &frame)
  {
    if (frame.copied_event) cudaEventDestroy(frame.copied_event);
    if (frame.packed_event) cudaEventDestroy(frame.packed_event);
    if (frame.copy_stream) cudaStreamDestroy(frame.copy_stream);
    if (frame.device_invalid) cudaFree(frame.device_invalid);
    if (frame.pinned_invalid) cudaFreeHost(frame.pinned_invalid);
    if (frame.device_values) cudaFree(frame.device_values);
    if (frame.pinned_values) cudaFreeHost(frame.pinned_values);
    frame.copied_event = nullptr;
    frame.packed_event = nullptr;
    frame.copy_stream = nullptr;
    frame.device_invalid = nullptr;
    frame.pinned_invalid = nullptr;
    frame.device_values = nullptr;
    frame.pinned_values = nullptr;
    frame.device_capacity = 0;
    frame.gpu = false;
  }

  bool prepare_gpu_frame(AsyncFrame &frame, size_t value_count)
  {
    if (frame.device_capacity >= value_count && frame.device_values &&
        frame.pinned_values && frame.device_invalid && frame.pinned_invalid &&
        frame.copy_stream && frame.packed_event && frame.copied_event)
      return true;
    release_gpu_frame(frame);
    const size_t bytes = value_count * sizeof(double);
    if (cudaMalloc(reinterpret_cast<void **>(&frame.device_values), bytes) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&frame.pinned_values), bytes,
                      cudaHostAllocPortable) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&frame.device_invalid), sizeof(int)) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&frame.pinned_invalid), sizeof(int),
                      cudaHostAllocPortable) != cudaSuccess ||
        cudaStreamCreateWithFlags(&frame.copy_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&frame.packed_event, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&frame.copied_event, cudaEventDisableTiming) != cudaSuccess) {
      cudaGetLastError();
      release_gpu_frame(frame);
      return false;
    }
    frame.device_capacity = value_count;
    return true;
  }

  bool cached_views_match() const
  {
    return device_ready && cached_atom_kokkos == static_cast<void *>(atom) &&
        static_cast<size_t>(atom->nlocal) <= device_views.capacity &&
        device_pointer<int32_t>(cached_atom_kokkos, 0x820) == device_views.tag &&
        device_pointer<int32_t>(cached_atom_kokkos, 0x868) == device_views.type &&
        device_pointer<int32_t>(cached_atom_kokkos, 0x8f8) == device_views.image &&
        device_pointer<double>(cached_atom_kokkos, 0x940) == device_views.x &&
        device_pointer<double>(cached_atom_kokkos, 0x998) == device_views.v &&
        device_pointer<double>(cached_atom_kokkos, 0x9f0) == device_views.f;
  }

  bool ensure_device_ready()
  {
    if (cached_views_match()) return true;
    DeviceViews views;
    if (!get_device_views(static_cast<void *>(atom), atom->nlocal, views)) return false;
    ensure_writer();
    const size_t state_bytes = static_cast<size_t>(atom->nlocal) * 14 * sizeof(double);
    const size_t snapshot_bytes = state_bytes + (shared_output_snapshot ?
        static_cast<size_t>(atom->nlocal) * 3 * sizeof(int32_t) : 0);
    const size_t value_count = (snapshot_bytes + sizeof(double) - 1) / sizeof(double);
    std::lock_guard<std::mutex> lock(writer_mutex);
    while (allocated_frames < 2) {
      auto frame = std::make_unique<AsyncFrame>();
      if (!prepare_gpu_frame(*frame, value_count)) return false;
      free_frames.push_back(std::move(frame));
      ++allocated_frames;
    }
    for (auto &frame : free_frames)
      if (!prepare_gpu_frame(*frame, value_count)) return false;
    device_views = views;
    cached_atom_kokkos = static_cast<void *>(atom);
    device_ready = true;
    return true;
  }

  bool eligible()
  {
    const bool result = fast_binary && binary && !compressed && !multifile && !multiproc &&
        nprocs == 1 && me == 0 && filewriter && fp && igroup == 0 &&
        atom->nlocal >= minimum_nodes && static_cast<bigint>(atom->nlocal) == atom->natoms &&
        sort_flag && sortcol == 0 && sortorder == 0 && reorderflag && idlo == 1 &&
        ntotal_reorder == atom->natoms && nme_reorder == atom->nlocal &&
        !balance_flag && !pbcflag && !delay_flag && !skipflag && !refreshflag &&
        idregion == nullptr && nthresh == 0 && ncompute == 0 && nfix == 0 &&
        nvariable == 0 && ncustom == 0 && triclinic_general == 0 &&
        size_one == 14 && nfield == 14 && columns_default && columns &&
        std::strcmp(columns_default, EXPECTED_COLUMNS) == 0 &&
        std::strcmp(columns, EXPECTED_COLUMNS) == 0 &&
        atom->tag && atom->type && atom->x && atom->v && atom->f && atom->image;
    if (result && device_binary && sync_downgrade) active_device_dump = this;
    if (timing && !eligibility_reported) {
      eligibility_reported = true;
      std::fprintf(stderr,
          "BINARY_STATE_FAST_DUMP eligibility result=%d fast=%d binary=%d compressed=%d "
          "multifile=%d multiproc=%d nprocs=%d me=%d writer=%d fp=%d igroup=%d "
          "nlocal=%d natoms=%lld sort=%d sortcol=%d sortorder=%d reorder=%d "
          "idlo=%lld ntotal_reorder=%lld nme_reorder=%d balance=%d pbc=%d "
          "delay=%d skip=%d refresh=%d region=%d nthresh=%d ncompute=%d nfix=%d "
          "nvariable=%d ncustom=%d triclinic_general=%d size_one=%d nfield=%d "
          "columns_default=[%s] columns=[%s]\n",
          result ? 1 : 0, fast_binary ? 1 : 0, binary, compressed, multifile,
          multiproc, nprocs, me, filewriter, fp ? 1 : 0, igroup, atom->nlocal,
          static_cast<long long>(atom->natoms), sort_flag, sortcol, sortorder,
          reorderflag, static_cast<long long>(idlo),
          static_cast<long long>(ntotal_reorder), nme_reorder, balance_flag,
          pbcflag, delay_flag, skipflag, refreshflag, idregion ? 1 : 0,
          nthresh, ncompute, nfix, nvariable, ncustom, triclinic_general,
          size_one, nfield, columns_default ? columns_default : "<null>",
          columns ? columns : "<null>");
    }
    return result;
  }

  void set_box_bounds()
  {
    if (domain->triclinic == 0) {
      boxxlo = domain->boxlo[0];
      boxxhi = domain->boxhi[0];
      boxylo = domain->boxlo[1];
      boxyhi = domain->boxhi[1];
      boxzlo = domain->boxlo[2];
      boxzhi = domain->boxhi[2];
    } else {
      boxxlo = domain->boxlo_bound[0];
      boxxhi = domain->boxhi_bound[0];
      boxylo = domain->boxlo_bound[1];
      boxyhi = domain->boxhi_bound[1];
      boxzlo = domain->boxlo_bound[2];
      boxzhi = domain->boxhi_bound[2];
      boxxy = domain->xy;
      boxxz = domain->xz;
      boxyz = domain->yz;
    }
  }

  void allocate_fast_buffer(int count)
  {
    const bigint values = static_cast<bigint>(count) * size_one;
    if (values > MAXSMALLINT)
      error->all(FLERR, Error::NOLASTLINE, "Too much per-proc data for BINARY_STATE dump");
    if (values > maxbuf) {
      maxbuf = static_cast<int>(values);
      memory->destroy(buf);
      memory->create(buf, maxbuf, "dump:buf");
    }
  }

  bool scatter_rows(double *target)
  {
    std::atomic<int> invalid{0};
    tagint *tag = atom->tag;
    int *type = atom->type;
    imageint *image = atom->image;
    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    const int count = atom->nlocal;
    parallel_for(count, [&](int begin, int end) {
      for (int source = begin; source < end; ++source) {
        const tagint id = tag[source];
        if (id < 1 || id > static_cast<tagint>(count)) {
          invalid.store(1, std::memory_order_relaxed);
          continue;
        }
        double *row = target + static_cast<size_t>(id - 1) * 14;
        row[0] = static_cast<double>(id);
        row[1] = static_cast<double>(type[source]);
        row[2] = x[source][0];
        row[3] = x[source][1];
        row[4] = x[source][2];
        row[5] = v[source][0];
        row[6] = v[source][1];
        row[7] = v[source][2];
        row[8] = f[source][0];
        row[9] = f[source][1];
        row[10] = f[source][2];
        row[11] = static_cast<double>((image[source] & IMGMASK) - IMGMAX);
        row[12] = static_cast<double>((image[source] >> IMGBITS & IMGMASK) - IMGMAX);
        row[13] = static_cast<double>((image[source] >> IMG2BITS) - IMGMAX);
      }
    });
    return invalid.load(std::memory_order_relaxed) == 0;
  }

  void validate_same_snapshot(double *target)
  {
    // Build the reference with the stock count/pack/sort implementation.
    nme = count();
    ntotal = nme;
    allocate_fast_buffer(nme);
    if (nme > maxids) {
      maxids = nme;
      memory->destroy(ids);
      memory->create(ids, maxids, "dump:ids");
    }
    pack(ids);
    if (nme > 1) sort();
    std::vector<double> reference(buf, buf + static_cast<size_t>(nme) * size_one);
    if (!scatter_rows(target))
      error->all(FLERR, Error::NOLASTLINE, "BINARY_STATE dump saw an out-of-range atom ID");
    const size_t bytes = reference.size() * sizeof(double);
    if (std::memcmp(reference.data(), target, bytes) != 0)
      error->all(FLERR, Error::NOLASTLINE,
                 "BINARY_STATE one-pass dump does not match stock pack/sort on the same snapshot");
    validated_once = true;
  }

  void write_fast_binary()
  {
    const auto total_begin = Clock::now();
    set_box_bounds();
    if (fp) clearerr(fp);
    nme = atom->nlocal;
    ntotal = nme;
    allocate_fast_buffer(nme);

    const auto pack_begin = Clock::now();
    if (validate && !validated_once) validate_same_snapshot(buf);
    else if (!scatter_rows(buf)) {
      // No bytes have been emitted yet, so the stock path is still safe.
      return DumpCustom::write();
    }
    const auto packed = Clock::now();

    if (filewriter && write_header_flag) write_header(ntotal);
    const auto header_done = Clock::now();
    write_data(nme, buf);
    const auto data_done = Clock::now();
    if (flush_flag && fp) fflush(fp);
    if (filewriter && fp != nullptr) write_footer();
    if (fp && ferror(fp))
      error->one(FLERR, Error::NOLASTLINE, "Error writing dump {}: {}", id,
                 utils::getsyserror());
    const auto done = Clock::now();

    if (timing)
      std::fprintf(stderr,
                   "BINARY_STATE_FAST_DUMP fast total_ms=%.3f pack_ms=%.3f header_ms=%.3f "
                   "data_ms=%.3f flush_ms=%.3f threads=%d validated=%d\n",
                   milliseconds(done - total_begin), milliseconds(packed - pack_begin),
                   milliseconds(header_done - packed), milliseconds(data_done - header_done),
                   milliseconds(done - data_done), threads, validated_once ? 1 : 0);
  }

  int dump_every() const
  {
    for (int index = 0; index < output->ndump; ++index)
      if (output->dump[index] == this) return output->every_dump[index];
    return 0;
  }

  bool is_last_dump_in_run() const
  {
    const int every = dump_every();
    return every <= 0 || update->ntimestep + every > update->endstep;
  }

  bool shared_snapshot_due() const
  {
    if (!shared_output_snapshot || validate || is_last_dump_in_run() ||
        !output->restart_flag_single || output->restart_flag_double ||
        output->next_restart != update->ntimestep ||
        output->next_restart_single != update->ntimestep ||
        output->last_restart == update->ntimestep || neighbor->build_once ||
        modify->restart_pbc_any || output->next != update->ntimestep ||
        output->any_time_dumps || output->var_thermo || modify->n_energy_global)
      return false;
    bool owns_scheduled_state = false;
    for (int index = 0; index < output->ndump; ++index) {
      if (output->next_dump[index] != update->ntimestep) continue;
      auto *scheduled = dynamic_cast<DumpCustomFastDEVICE_DUMP *>(output->dump[index]);
      if (scheduled == this) { owns_scheduled_state = true; continue; }
      if (!scheduled || !scheduled->fast_scalar || !scheduled->scalar_eligible())
        return false;
    }
    if (!owns_scheduled_state) return false;
    static auto restart_available = reinterpret_cast<RestartAvailableFn>(
        dlsym(RTLD_DEFAULT, "dpa_device_restart_device_restart_available"));
    if (!restart_available || !restart_available(atom)) return false;
    static auto available = reinterpret_cast<SharedSnapshotAvailableFn>(
        dlsym(RTLD_DEFAULT, "dpa_zerocopy_restart_shared_snapshot_available"));
    return available && output->restart1 && available(output->restart1);
  }

  void writer_loop()
  {
    for (;;) {
      std::unique_ptr<AsyncFrame> frame;
      {
        std::unique_lock<std::mutex> lock(writer_mutex);
        writer_cv.wait(lock, [&] { return writer_stopping || !writer_queue.empty(); });
        if (writer_stopping && writer_queue.empty()) return;
        frame = std::move(writer_queue.front());
        writer_queue.pop_front();
        writer_busy = true;
      }
      const auto begin = Clock::now();
      double device_wait_ms = 0.0;
      bool ready = true;
      if (frame->gpu) {
        const auto wait_begin = Clock::now();
        ready = cudaEventSynchronize(frame->copied_event) == cudaSuccess &&
            frame->pinned_invalid && *frame->pinned_invalid == 0;
        device_wait_ms = milliseconds(Clock::now() - wait_begin);
      }
      const double *payload = frame->gpu ? frame->pinned_values : frame->values.data();
      const bool count_ok =
          ready && std::fwrite(&frame->value_count, sizeof(int), 1, fp) == 1;
      const bool data_ok = count_ok && std::fwrite(
          payload, sizeof(double), frame->value_count, fp) ==
              static_cast<size_t>(frame->value_count);
      const bool flush_ok = !flush_flag || std::fflush(fp) == 0;
      bool ok = data_ok && flush_ok && !std::ferror(fp);
      if (frame->shared_snapshot) {
        static auto release = reinterpret_cast<SharedSnapshotReleaseFn>(
            dlsym(RTLD_DEFAULT, "dpa_device_restart_release_state_snapshot"));
        ok = release && release(frame->pinned_values) == 1 && ok;
        frame->shared_snapshot = false;
      }
      const auto done = Clock::now();
      if (timing)
        std::fprintf(stderr,
                     "DEVICE_DUMP_ASYNC_DUMP write_ms=%.3f device_wait_ms=%.3f values=%d "
                     "gpu=%d ok=%d\n",
                     milliseconds(done - begin), device_wait_ms, frame->value_count,
                     frame->gpu ? 1 : 0, ok ? 1 : 0);
      {
        std::lock_guard<std::mutex> lock(writer_mutex);
        writer_failed = writer_failed || !ok;
        writer_busy = false;
        free_frames.push_back(std::move(frame));
      }
      writer_cv.notify_all();
    }
  }

  void ensure_writer()
  {
    std::lock_guard<std::mutex> lock(writer_mutex);
    if (writer_started) return;
    writer_started = true;
    writer_thread = std::thread([this] { writer_loop(); });
  }

  std::unique_ptr<AsyncFrame> acquire_frame()
  {
    ensure_writer();
    std::unique_lock<std::mutex> lock(writer_mutex);
    if (!free_frames.empty()) {
      auto frame = std::move(free_frames.front());
      free_frames.pop_front();
      return frame;
    }
    if (allocated_frames < 2) {
      ++allocated_frames;
      return std::make_unique<AsyncFrame>();
    }
    writer_cv.wait(lock, [&] { return writer_failed || !free_frames.empty(); });
    if (writer_failed) return nullptr;
    auto frame = std::move(free_frames.front());
    free_frames.pop_front();
    return frame;
  }

  bool flush_writer()
  {
    if (!writer_started) return true;
    std::unique_lock<std::mutex> lock(writer_mutex);
    writer_cv.wait(lock, [&] {
      return writer_failed || (writer_queue.empty() && !writer_busy);
    });
    return !writer_failed;
  }

  void stop_writer()
  {
    if (!writer_started) return;
    flush_writer();
    {
      std::lock_guard<std::mutex> lock(writer_mutex);
      writer_stopping = true;
    }
    writer_cv.notify_all();
    if (writer_thread.joinable()) writer_thread.join();
    for (auto &frame : free_frames) release_gpu_frame(*frame);
    free_frames.clear();
  }

  void validate_device_snapshot(const double *candidate, int count)
  {
    nme = this->count();
    if (nme != count)
      error->all(FLERR, Error::NOLASTLINE,
                 "DEVICE_DUMP device dump count differs from stock dump count");
    ntotal = nme;
    allocate_fast_buffer(nme);
    if (nme > maxids) {
      maxids = nme;
      memory->destroy(ids);
      memory->create(ids, maxids, "dump:ids");
    }
    pack(ids);
    if (nme > 1) sort();
    const size_t bytes = static_cast<size_t>(nme) * size_one * sizeof(double);
    if (std::memcmp(buf, candidate, bytes) != 0)
      error->all(FLERR, Error::NOLASTLINE,
                 "DEVICE_DUMP device dump does not match stock pack/sort on the same snapshot");
    validated_once = true;
  }

  void write_device_binary()
  {
    const auto total_begin = Clock::now();
    const auto prior_begin = Clock::now();
    if (!flush_writer())
      error->one(FLERR, Error::NOLASTLINE, "DEVICE_DUMP prior asynchronous dump write failed");
    const auto prior_done = Clock::now();
    if (!ensure_device_ready() || !cached_views_match())
      error->one(FLERR, Error::NOLASTLINE, "DEVICE_DUMP device state views are unavailable");
    set_box_bounds();
    if (fp) clearerr(fp);
    nme = atom->nlocal;
    ntotal = nme;
    auto frame = acquire_frame();
    if (!frame)
      error->one(FLERR, Error::NOLASTLINE, "DEVICE_DUMP could not acquire dump buffer");
    frame->value_count = nme * size_one;
    frame->gpu = true;
    frame->shared_snapshot = false;
    const bool share = shared_snapshot_due();
    const size_t state_bytes = static_cast<size_t>(frame->value_count) * sizeof(double);
    const size_t snapshot_bytes = state_bytes + (share ? static_cast<size_t>(nme) * 3 * sizeof(int32_t) : 0);
    const size_t allocated_values = (snapshot_bytes + sizeof(double) - 1) / sizeof(double);
    if (!prepare_gpu_frame(*frame, allocated_values))
      error->one(FLERR, Error::NOLASTLINE, "DEVICE_DUMP could not allocate device dump buffers");
    // Growing the shared payload releases the previous GPU allocation and
    // clears this ownership flag.  Restore it only after preparation succeeds.
    frame->gpu = true;

    const auto pack_begin = Clock::now();
    const size_t bytes = snapshot_bytes;
    const int block_size = 256;
    const int blocks = (nme + block_size - 1) / block_size;
    if (cudaMemsetAsync(frame->device_invalid, 0, sizeof(int), nullptr) != cudaSuccess)
      error->one(FLERR, Error::NOLASTLINE, "DEVICE_DUMP could not clear device dump status");
    if (share) {
      const int32_t *mask = device_pointer<int32_t>(atom, 0x8b0);
      pack_shared_device_state<<<blocks, block_size, 0, nullptr>>>(
          device_views, mask, nme, frame->device_values, frame->device_invalid);
    } else {
      pack_device_state<<<blocks, block_size, 0, nullptr>>>(
          device_views, nme, frame->device_values, frame->device_invalid);
    }
    if (cudaGetLastError() != cudaSuccess ||
        cudaEventRecord(frame->packed_event, nullptr) != cudaSuccess ||
        cudaStreamWaitEvent(frame->copy_stream, frame->packed_event, 0) != cudaSuccess ||
        cudaMemcpyAsync(frame->pinned_values, frame->device_values, bytes,
                        cudaMemcpyDeviceToHost, frame->copy_stream) != cudaSuccess ||
        cudaMemcpyAsync(frame->pinned_invalid, frame->device_invalid, sizeof(int),
                        cudaMemcpyDeviceToHost, frame->copy_stream) != cudaSuccess ||
        cudaEventRecord(frame->copied_event, frame->copy_stream) != cudaSuccess)
      error->one(FLERR, Error::NOLASTLINE, "DEVICE_DUMP device dump submission failed");
    const auto submitted_copy = Clock::now();
    if (share) {
      static auto register_snapshot = reinterpret_cast<SharedSnapshotRegisterFn>(
          dlsym(RTLD_DEFAULT, "dpa_device_restart_register_state_snapshot"));
      const auto *metadata = reinterpret_cast<const int32_t *>(
          frame->pinned_values + static_cast<size_t>(nme) * 14);
      if (register_snapshot)
        frame->shared_snapshot = register_snapshot(atom, update->ntimestep, nme,
            frame->pinned_values, metadata, frame->pinned_invalid, frame->copied_event) == 1;
      if (std::getenv("DPA_SHARED_OUTPUT_TIMING"))
        std::fprintf(stderr,"SHARED_OUTPUT_STATE_SUBMIT step=%lld atoms=%d state_bytes=%zu "
                     "snapshot_bytes=%zu registered=%d\n",static_cast<long long>(update->ntimestep),
                     nme,state_bytes,bytes,frame->shared_snapshot ? 1 : 0);
      const char *validate_shared = std::getenv("DPA_SHARED_OUTPUT_VALIDATE");
      if (frame->shared_snapshot && validate_shared && std::strcmp(validate_shared,"0") != 0) {
        if (cudaEventSynchronize(frame->copied_event) != cudaSuccess || *frame->pinned_invalid)
          error->all(FLERR,Error::NOLASTLINE,"Shared state snapshot copy failed");
        static auto original_sync = reinterpret_cast<SyncFn>(dlsym(RTLD_NEXT,
            "_ZN9LAMMPS_NS10AtomKokkos4syncENS_14ExecutionSpaceEj"));
        if (!original_sync) error->all(FLERR,Error::NOLASTLINE,"Missing original atom sync");
        original_sync(atom,static_cast<int>(Host),ALL_ATOM_MASK);
        validate_device_snapshot(frame->pinned_values,nme);
        std::fprintf(stderr,"SHARED_OUTPUT_STATE_BYTE_VALIDATE step=%lld bytes=%zu exact=1\n",
                     static_cast<long long>(update->ntimestep),state_bytes);
      }
    }

    if (validate && !validated_once) {
      const cudaError_t wait_status = cudaEventSynchronize(frame->copied_event);
      if (wait_status != cudaSuccess || !frame->pinned_invalid || *frame->pinned_invalid)
        error->all(FLERR, Error::NOLASTLINE, "DEVICE_DUMP device dump saw an invalid atom ID");
      validate_device_snapshot(frame->pinned_values, nme);
    }

    if (filewriter && write_header_flag) write_header(ntotal);
    if (filewriter && fp != nullptr) write_footer();
    const auto header_done = Clock::now();
    {
      std::lock_guard<std::mutex> lock(writer_mutex);
      writer_queue.push_back(std::move(frame));
    }
    writer_cv.notify_one();
    const auto submitted = Clock::now();
    double final_wait_ms = 0.0;
    if (is_last_dump_in_run()) {
      const auto final_begin = Clock::now();
      if (!flush_writer())
        error->one(FLERR, Error::NOLASTLINE, "DEVICE_DUMP final asynchronous dump write failed");
      final_wait_ms = milliseconds(Clock::now() - final_begin);
    }
    if (timing)
      std::fprintf(stderr,
                   "DEVICE_DUMP_DEVICE_DUMP total_ms=%.3f prior_wait_ms=%.3f submit_copy_ms=%.3f "
                   "header_ms=%.3f queue_ms=%.3f final_wait_ms=%.3f values=%d "
                   "validated=%d\n",
                   milliseconds(Clock::now() - total_begin),
                   milliseconds(prior_done - prior_begin),
                   milliseconds(submitted_copy - pack_begin),
                   milliseconds(header_done - submitted_copy),
                   milliseconds(submitted - header_done), final_wait_ms,
                   nme * size_one, validated_once ? 1 : 0);
  }

  void write_async_binary()
  {
    const auto total_begin = Clock::now();
    const auto prior_begin = Clock::now();
    if (!flush_writer())
      error->one(FLERR, Error::NOLASTLINE, "ASYNC_DUMP prior asynchronous dump write failed");
    const auto prior_done = Clock::now();
    set_box_bounds();
    if (fp) clearerr(fp);
    nme = atom->nlocal;
    ntotal = nme;
    auto frame = acquire_frame();
    if (!frame)
      error->one(FLERR, Error::NOLASTLINE, "ASYNC_DUMP could not acquire dump buffer");
    frame->gpu = false;
    frame->value_count = nme * size_one;
    frame->values.resize(static_cast<size_t>(frame->value_count));

    const auto pack_begin = Clock::now();
    if (validate && !validated_once) validate_same_snapshot(frame->values.data());
    else if (!scatter_rows(frame->values.data())) return DumpCustom::write();
    const auto packed = Clock::now();

    if (filewriter && write_header_flag) write_header(ntotal);
    if (filewriter && fp != nullptr) write_footer();
    const auto header_done = Clock::now();
    {
      std::lock_guard<std::mutex> lock(writer_mutex);
      writer_queue.push_back(std::move(frame));
    }
    writer_cv.notify_one();
    const auto submitted = Clock::now();
    double final_wait_ms = 0.0;
    if (is_last_dump_in_run()) {
      const auto final_begin = Clock::now();
      if (!flush_writer())
        error->one(FLERR, Error::NOLASTLINE, "ASYNC_DUMP final asynchronous dump write failed");
      final_wait_ms = milliseconds(Clock::now() - final_begin);
    }
    if (timing)
      std::fprintf(stderr,
                   "ASYNC_DUMP_ASYNC_DUMP submit total_ms=%.3f prior_wait_ms=%.3f "
                   "pack_ms=%.3f header_ms=%.3f submit_ms=%.3f final_wait_ms=%.3f "
                   "threads=%d validated=%d\n",
                   milliseconds(Clock::now() - total_begin),
                   milliseconds(prior_done - prior_begin), milliseconds(packed - pack_begin),
                   milliseconds(header_done - packed), milliseconds(submitted - header_done),
                   final_wait_ms, threads, validated_once ? 1 : 0);
  }

  void write_data(int n, double *mybuf) override
  {
    const auto begin = Clock::now();
    DumpCustom::write_data(n, mybuf);
    t_data += milliseconds(Clock::now() - begin);
  }

  int convert_string(int n, double *mybuf) override
  {
    const auto begin = Clock::now();
    const int result = format_text(n, mybuf);
    t_convert += milliseconds(Clock::now() - begin);
    return result;
  }

  int format_text(int n, double *mybuf)
  {
    std::vector<Field> fields(nfield);
    for (int column = 0; column < nfield; ++column)
      if (!parse(vformat[column], vtype[column] == INT, vtype[column] == DOUBLE,
                 fields[column]))
        return DumpCustom::convert_string(n, mybuf);

    const unsigned hardware = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    const int use_threads = n < 512 ? 1 : static_cast<int>(hardware);
    const size_t row_max = static_cast<size_t>(nfield) * 50 + 2;
    if (chunks.size() < static_cast<size_t>(use_threads)) {
      chunks.resize(use_threads);
      capacity.resize(use_threads, 0);
    }
    std::vector<size_t> sizes(use_threads, 0);
    auto work = [&](int thread) {
      const int begin = static_cast<int>(static_cast<long>(n) * thread / use_threads);
      const int end = static_cast<int>(static_cast<long>(n) * (thread + 1) / use_threads);
      const size_t need = row_max * (end - begin) + 1;
      if (capacity[thread] < need) {
        capacity[thread] = need + need / 4;
        chunks[thread].reset(new char[capacity[thread]]);
      }
      char *out = chunks[thread].get();
      const double *row = mybuf + static_cast<size_t>(begin) * nfield;
      for (int index = begin; index < end; ++index, row += nfield) {
        for (int column = 0; column < nfield; ++column) {
          const Field &field = fields[column];
          if (field.kind == 0)
            out = std::to_chars(out, out + 16, static_cast<int>(row[column])).ptr;
          else
            out = put_double(out, row[column], field.precision);
          if (field.space) *out++ = ' ';
        }
        *out++ = '\n';
      }
      sizes[thread] = out - chunks[thread].get();
    };
    auto parallel = [use_threads](const auto &function) {
      if (use_threads == 1) return function(0);
      std::vector<std::thread> workers;
      for (int thread = 0; thread < use_threads; ++thread)
        workers.emplace_back(function, thread);
      for (auto &worker : workers) worker.join();
    };
    parallel(work);
    std::vector<size_t> offsets(use_threads + 1, 0);
    for (int thread = 0; thread < use_threads; ++thread)
      offsets[thread + 1] = offsets[thread] + sizes[thread];
    const size_t total = offsets[use_threads];
    if (total + 1 > static_cast<size_t>(MAXSMALLINT)) return -1;
    if (static_cast<size_t>(maxsbuf) < total + 1) {
      maxsbuf = static_cast<int>(total + 1);
      memory->grow(sbuf, maxsbuf, "dump:sbuf");
    }
    parallel([&](int thread) {
      std::memcpy(sbuf + offsets[thread], chunks[thread].get(), sizes[thread]);
    });
    sbuf[total] = '\0';
    return static_cast<int>(total);
  }
};

Dump *create_dump(LAMMPS *lmp, int narg, char **arg)
{
  return new DumpCustomFastDEVICE_DUMP(lmp, narg, arg);
}

}  // namespace

extern "C" void device_dump_atom_sync(void *, int, unsigned int)
    asm("_ZN9LAMMPS_NS10AtomKokkos4syncENS_14ExecutionSpaceEj");
extern "C" void device_dump_atom_sync(void *object, int space, unsigned int mask)
{
  static auto original = reinterpret_cast<SyncFn>(
      dlsym(RTLD_NEXT, "_ZN9LAMMPS_NS10AtomKokkos4syncENS_14ExecutionSpaceEj"));
  if (!original) std::abort();
  const bool output_sync = active_device_dump &&
      active_device_dump->is_output_sync(object, space, mask);
  unsigned int replacement = mask;
  if (active_device_dump &&
      active_device_dump->replace_output_sync(object, space, mask, replacement)) {
    if (replacement) original(object, space, replacement);
    return;
  }
  const auto begin = std::chrono::steady_clock::now();
  original(object, space, mask);
  if (output_sync)
    active_device_dump->report_original_output_sync(
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count());
}

extern "C" void lammpsplugin_init(void *ptr, void *, void *)
{
  const char *enabled = std::getenv("DPA_FAST_DUMP");
  if (enabled && std::strcmp(enabled, "0") == 0) return;
  auto &map = *static_cast<LAMMPS *>(ptr)->output->dump_map;
  if (std::getenv("DPA_FAST_DUMP_REF") && map.count("custom"))
    map["custom/ref"] = map["custom"];
  map["custom"] = &create_dump;
}
