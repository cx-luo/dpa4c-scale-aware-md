#include "atom.h"
#include "domain.h"
#include "fix.h"
#include "lmptype.h"
#include "modify.h"
#include "update.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <deque>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace LAMMPS_NS;

namespace {
using PostForceFn = void (*)(void *, int);
using ModifyParamsFn = void (*)(void *, int, char **);
using SyncFn = void (*)(void *, int, unsigned int);
using Clock = std::chrono::steady_clock;
using PACKED_NVTMaterializeFn = int (*)(void*);

constexpr unsigned X_MASK = 0x00000001;
constexpr unsigned V_MASK = 0x00000002;
constexpr unsigned F_MASK = 0x00000004;
constexpr unsigned TAG_MASK = 0x00000008;
constexpr unsigned TYPE_MASK = 0x00000010;
constexpr unsigned IMAGE_MASK = 0x00000040;
constexpr unsigned OBSERVER_MASK = X_MASK | V_MASK | F_MASK | TAG_MASK | TYPE_MASK | IMAGE_MASK;

struct Row {
  int64_t id, type, image[3];
  double x[3], v[3], f[3];
};
static_assert(sizeof(Row) == 112, "observer wire format changed");

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

__global__ void pack_device_rows(DeviceViews views, int count, Row *rows, int *invalid)
{
  const int source = blockIdx.x * blockDim.x + threadIdx.x;
  if (source >= count) return;
  const int32_t id = views.tag[source];
  if (id < 1 || id > count) {
    atomicExch(invalid, 1);
    return;
  }
  Row &row = rows[id - 1];
  row.id = static_cast<int64_t>(id);
  row.type = static_cast<int64_t>(views.type[source]);
  const int32_t image = views.image[source];
  row.image[0] = static_cast<int64_t>((image & 1023) - 512);
  row.image[1] = static_cast<int64_t>(((image >> 10) & 1023) - 512);
  row.image[2] = static_cast<int64_t>((image >> 20) - 512);
  const size_t offset = static_cast<size_t>(source) * 3;
  for (int axis = 0; axis < 3; ++axis) {
    row.x[axis] = views.x[offset + axis];
    row.v[axis] = views.v[offset + axis];
    row.f[axis] = views.f[offset + axis];
  }
}

class ModifyAccess : public Modify {
 public:
  int post_count() const { return n_post_force; }
  int post_group_count() const { return n_post_force_group; }
  int *post_list() const { return list_post_force; }
  Atom *atom_pointer() const { return atom; }
  void *atom_kokkos_pointer() const { return atomKK; }
  Update *update_pointer() const { return update; }
  Domain *domain_pointer() const { return domain; }
};

class FixAccess : public Fix {
 public:
  Atom *atom_pointer() const { return atom; }
};

struct CaptureState {
  Fix *fix = nullptr;
  bool active = false;
  int64_t begin = 0;
  int64_t end = 0;
  std::string prefix;
  std::set<int64_t> selected;
};

CaptureState capture;
std::vector<Row> rows;
int minimum_nodes = 16384;
int worker_threads = 8;
bool enabled = false;
bool timing = false;
bool validate_writer = false;
bool async_write = false;
bool device_pack = false;
int device_minimum_nodes = 32768;
int device_small_maximum_nodes = -1;

struct AsyncFrame {
  std::string path;
  std::array<int64_t, 4> header;
  std::array<double, 12> box;
  std::vector<Row> rows;
  Row *device_rows = nullptr;
  Row *pinned_rows = nullptr;
  int *device_invalid = nullptr;
  int *pinned_invalid = nullptr;
  cudaStream_t copy_stream = nullptr;
  cudaEvent_t packed_event = nullptr;
  cudaEvent_t copied_event = nullptr;
  size_t gpu_capacity = 0;
  int gpu_count = 0;
  bool gpu = false;
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
bool device_ids_valid = false;
void *cached_atom_kokkos = nullptr;
DeviceViews cached_views;

void ensure_writer();

int packed_nvt_materialize_force() {
  static auto materialize = reinterpret_cast<PACKED_NVTMaterializeFn>(
      dlsym(RTLD_DEFAULT, "dpa_chain_small_nvt_materialize_force"));
  if (!materialize) return 0;
  return materialize(nullptr);
}

double millis(Clock::duration duration)
{
  return std::chrono::duration<double, std::milli>(duration).count();
}

template <class Function>
void parallel_for(int count, Function function)
{
  const int threads = count < 32768 ? 1 : std::max(1, worker_threads);
  if (threads == 1) {
    function(0, count);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(threads));
  for (int thread = 0; thread < threads; ++thread) {
    const int begin = static_cast<int>(static_cast<long long>(count) * thread / threads);
    const int end = static_cast<int>(static_cast<long long>(count) * (thread + 1) / threads);
    workers.emplace_back(function, begin, end);
  }
  for (auto &worker : workers) worker.join();
}

void release_gpu_frame(AsyncFrame &frame)
{
  if (frame.copied_event) cudaEventDestroy(frame.copied_event);
  if (frame.packed_event) cudaEventDestroy(frame.packed_event);
  if (frame.copy_stream) cudaStreamDestroy(frame.copy_stream);
  if (frame.device_invalid) cudaFree(frame.device_invalid);
  if (frame.pinned_invalid) cudaFreeHost(frame.pinned_invalid);
  if (frame.device_rows) cudaFree(frame.device_rows);
  if (frame.pinned_rows) cudaFreeHost(frame.pinned_rows);
  frame.copied_event = nullptr;
  frame.packed_event = nullptr;
  frame.copy_stream = nullptr;
  frame.device_invalid = nullptr;
  frame.pinned_invalid = nullptr;
  frame.device_rows = nullptr;
  frame.pinned_rows = nullptr;
  frame.gpu_capacity = 0;
  frame.gpu_count = 0;
  frame.gpu = false;
}

bool prepare_gpu_frame(AsyncFrame &frame, int count)
{
  if (frame.gpu_capacity >= static_cast<size_t>(count) && frame.device_rows &&
      frame.pinned_rows && frame.device_invalid && frame.pinned_invalid &&
      frame.copy_stream && frame.packed_event && frame.copied_event)
    return true;
  release_gpu_frame(frame);
  const size_t bytes = static_cast<size_t>(count) * sizeof(Row);
  if (cudaMalloc(reinterpret_cast<void **>(&frame.device_rows), bytes) != cudaSuccess ||
      cudaHostAlloc(reinterpret_cast<void **>(&frame.pinned_rows), bytes,
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
  frame.gpu_capacity = static_cast<size_t>(count);
  return true;
}

bool validate_contiguous_tags(Atom *atom)
{
  const int count = atom->nlocal;
  if (count <= 0 || static_cast<bigint>(count) != atom->natoms || !atom->tag) return false;
  std::vector<unsigned char> seen(static_cast<size_t>(count), 0);
  for (int index = 0; index < count; ++index) {
    const tagint id = atom->tag[index];
    if (id < 1 || id > static_cast<tagint>(count) || seen[static_cast<size_t>(id - 1)])
      return false;
    seen[static_cast<size_t>(id - 1)] = 1;
  }
  return true;
}

bool prepare_device_pool(Atom *atom)
{
  device_ids_valid = validate_contiguous_tags(atom);
  cached_atom_kokkos = atom;
  if (!device_ids_valid || !get_device_views(atom, atom->nlocal, cached_views)) {
    device_ids_valid = false;
    cached_atom_kokkos = nullptr;
    return false;
  }
  ensure_writer();
  std::lock_guard<std::mutex> lock(writer_mutex);
  while (allocated_frames < 3) {
    auto frame = std::make_unique<AsyncFrame>();
    if (!prepare_gpu_frame(*frame, atom->nlocal)) return false;
    free_frames.push_back(std::move(frame));
    ++allocated_frames;
  }
  for (auto &frame : free_frames)
    if (!prepare_gpu_frame(*frame, atom->nlocal)) return false;
  return true;
}

bool cached_views_match(void *atom_kokkos, int count)
{
  return device_ids_valid && atom_kokkos == cached_atom_kokkos && count > 0 &&
      static_cast<size_t>(count) <= cached_views.capacity &&
      device_pointer<int32_t>(atom_kokkos, 0x820) == cached_views.tag &&
      device_pointer<int32_t>(atom_kokkos, 0x868) == cached_views.type &&
      device_pointer<int32_t>(atom_kokkos, 0x8f8) == cached_views.image &&
      device_pointer<double>(atom_kokkos, 0x940) == cached_views.x &&
      device_pointer<double>(atom_kokkos, 0x998) == cached_views.v &&
      device_pointer<double>(atom_kokkos, 0x9f0) == cached_views.f;
}

__attribute__((noinline, cold)) bool refresh_cached_views(void *atom_kokkos, int count,
                                                          int64_t step)
{
  DeviceViews refreshed_views;
  if (!get_device_views(atom_kokkos, count, refreshed_views)) return false;
  cached_atom_kokkos = atom_kokkos;
  cached_views = refreshed_views;
  if (timing)
    std::fprintf(stderr,
                 "DEVICE_VIEW_DEVICE_VIEW_REFRESH step=%lld rows=%d capacity=%zu\n",
                 static_cast<long long>(step), count, cached_views.capacity);
  return true;
}

bool write_all(int descriptor, const void *data, size_t bytes)
{
  auto *cursor = static_cast<const char *>(data);
  while (bytes) {
    const ssize_t written = ::write(descriptor, cursor, bytes);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return false;
    cursor += written;
    bytes -= static_cast<size_t>(written);
  }
  return true;
}

bool pack_rows(Atom *atom, std::vector<Row> &packed)
{
  const int count = atom->nlocal;
  packed.resize(static_cast<size_t>(count));
  parallel_for(count, [atom, &packed](int begin, int end) {
    for (int old_index = begin; old_index < end; ++old_index) {
      const tagint id = atom->tag[old_index];
      if (id < 1 || id > static_cast<tagint>(atom->nlocal)) continue;
      Row &row = packed[static_cast<size_t>(id - 1)];
      row.id = static_cast<int64_t>(id);
      row.type = static_cast<int64_t>(atom->type[old_index]);
      const imageint image = atom->image[old_index];
      row.image[0] = static_cast<int64_t>((image & IMGMASK) - IMGMAX);
      row.image[1] = static_cast<int64_t>(((image >> IMGBITS) & IMGMASK) - IMGMAX);
      row.image[2] = static_cast<int64_t>((image >> IMG2BITS) - IMGMAX);
      for (int axis = 0; axis < 3; ++axis) {
        row.x[axis] = atom->x[old_index][axis];
        row.v[axis] = atom->v[old_index][axis];
        row.f[axis] = atom->f[old_index][axis];
      }
    }
  });
  for (int index = 0; index < count; ++index)
    if (packed[static_cast<size_t>(index)].id != static_cast<int64_t>(index + 1)) return false;
  return true;
}

bool write_payload(const std::string &path, const int64_t *header, const double *box,
                   const Row *packed, size_t count)
{
  const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (descriptor < 0) return false;
  const char magic[8] = {'D', 'P', 'A', 'F', 'R', 'C', '0', '1'};
  const bool ok = write_all(descriptor, magic, sizeof(magic)) &&
      write_all(descriptor, header, 4 * sizeof(int64_t)) &&
      write_all(descriptor, box, 12 * sizeof(double)) &&
      write_all(descriptor, packed, count * sizeof(Row));
  const int close_status = ::close(descriptor);
  return ok && close_status == 0;
}

bool write_payload(const std::string &path, const int64_t *header, const double *box,
                   const std::vector<Row> &packed)
{
  return write_payload(path, header, box, packed.data(), packed.size());
}

void writer_loop()
{
  for (;;) {
    std::unique_ptr<AsyncFrame> frame;
    {
      std::unique_lock<std::mutex> lock(writer_mutex);
      writer_cv.wait(lock, [] { return writer_stopping || !writer_queue.empty(); });
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
      device_wait_ms = millis(Clock::now() - wait_begin);
    }
    const Row *payload = frame->gpu ? frame->pinned_rows : frame->rows.data();
    const size_t count = frame->gpu ? static_cast<size_t>(frame->gpu_count) : frame->rows.size();
    const bool ok = ready &&
        write_payload(frame->path, frame->header.data(), frame->box.data(), payload, count);
    const auto end = Clock::now();
    if (timing)
      std::fprintf(stderr,
                   "DEVICE_OBSERVER_ASYNC_WRITE total_ms=%.3f device_wait_ms=%.3f rows=%zu "
                   "gpu=%d ok=%d path=%s\n",
                   millis(end - begin), device_wait_ms, count, frame->gpu ? 1 : 0,
                   ok ? 1 : 0, frame->path.c_str());
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
  writer_thread = std::thread(writer_loop);
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
  if (allocated_frames < 3) {
    ++allocated_frames;
    return std::make_unique<AsyncFrame>();
  }
  writer_cv.wait(lock, [] { return writer_failed || !free_frames.empty(); });
  if (writer_failed) return nullptr;
  auto frame = std::move(free_frames.front());
  free_frames.pop_front();
  return frame;
}

bool enqueue_frame(std::unique_ptr<AsyncFrame> frame)
{
  std::lock_guard<std::mutex> lock(writer_mutex);
  if (writer_failed) return false;
  writer_queue.push_back(std::move(frame));
  writer_cv.notify_one();
  return true;
}

bool flush_writer()
{
  if (!writer_started) return true;
  std::unique_lock<std::mutex> lock(writer_mutex);
  writer_cv.wait(lock, [] { return writer_failed || (writer_queue.empty() && !writer_busy); });
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

bool write_frame(ModifyAccess *modify, int64_t step, const char *suffix = "")
{
  Atom *atom = modify->atom_pointer();
  Domain *domain = modify->domain_pointer();
  if (atom->nlocal <= 0 || static_cast<bigint>(atom->nlocal) != atom->natoms) return false;
  const auto packed_begin = Clock::now();
  if (!pack_rows(atom, rows)) return false;
  const auto packed_end = Clock::now();

  const std::string path = capture.prefix + "-" + std::to_string(step) + suffix + ".bin";
  const int64_t header[4] = {step, capture.begin, capture.end, atom->nlocal};
  const double box[12] = {domain->boxlo[0], domain->boxlo[1], domain->boxlo[2],
                          domain->xprd, 0, 0, domain->xy, domain->yprd, 0,
                          domain->xz, domain->yz, domain->zprd};
  const bool ok = write_payload(path, header, box, rows);
  const auto written_end = Clock::now();
  if (timing) {
    std::fprintf(stderr,
                 "FUSED_OBSERVER_FUSED_OBSERVER step=%lld pack_ms=%.3f write_ms=%.3f rows=%zu threads=%d ok=%d\n",
                 static_cast<long long>(step), millis(packed_end - packed_begin),
                 millis(written_end - packed_end), rows.size(), worker_threads,
                 ok ? 1 : 0);
  }
  return ok;
}

bool enqueue_async_frame(ModifyAccess *modify, int64_t step, const char *suffix = "")
{
  auto frame = acquire_frame();
  if (!frame) return false;
  Atom *atom = modify->atom_pointer();
  Domain *domain = modify->domain_pointer();
  frame->path = capture.prefix + "-" + std::to_string(step) + suffix + ".bin";
  frame->header = {step, capture.begin, capture.end, atom->nlocal};
  frame->box = {domain->boxlo[0], domain->boxlo[1], domain->boxlo[2],
                domain->xprd, 0, 0, domain->xy, domain->yprd, 0,
                domain->xz, domain->yz, domain->zprd};
  if (!pack_rows(atom, frame->rows)) return false;
  return enqueue_frame(std::move(frame));
}

bool enqueue_device_frame(ModifyAccess *modify, int64_t step, const char *suffix = "")
{
  auto frame = acquire_frame();
  if (!frame) return false;
  auto fail = [&]() {
    if (frame && frame->gpu) cudaDeviceSynchronize();
    std::lock_guard<std::mutex> lock(writer_mutex);
    free_frames.push_back(std::move(frame));
    writer_cv.notify_all();
    return false;
  };
  Atom *atom = modify->atom_pointer();
  Domain *domain = modify->domain_pointer();
  if (!prepare_gpu_frame(*frame, atom->nlocal)) return fail();
  void *atom_kokkos = modify->atom_kokkos_pointer();
  if (__builtin_expect(!cached_views_match(atom_kokkos, atom->nlocal), 0) &&
      !refresh_cached_views(atom_kokkos, atom->nlocal, step))
    return fail();
  frame->path = capture.prefix + "-" + std::to_string(step) + suffix + ".bin";
  frame->header = {step, capture.begin, capture.end, atom->nlocal};
  frame->box = {domain->boxlo[0], domain->boxlo[1], domain->boxlo[2],
                domain->xprd, 0, 0, domain->xy, domain->yprd, 0,
                domain->xz, domain->yz, domain->zprd};
  frame->gpu = true;
  frame->gpu_count = atom->nlocal;
  const size_t bytes = static_cast<size_t>(atom->nlocal) * sizeof(Row);
  if (cudaMemsetAsync(frame->device_invalid, 0, sizeof(int), nullptr) != cudaSuccess)
    return fail();
  const int threads = 256;
  const int blocks = (atom->nlocal + threads - 1) / threads;
  pack_device_rows<<<blocks, threads, 0, nullptr>>>(
      cached_views, atom->nlocal, frame->device_rows, frame->device_invalid);
  if (cudaGetLastError() != cudaSuccess ||
      cudaEventRecord(frame->packed_event, nullptr) != cudaSuccess ||
      cudaStreamWaitEvent(frame->copy_stream, frame->packed_event, 0) != cudaSuccess ||
      cudaMemcpyAsync(frame->pinned_rows, frame->device_rows, bytes,
                      cudaMemcpyDeviceToHost, frame->copy_stream) != cudaSuccess ||
      cudaMemcpyAsync(frame->pinned_invalid, frame->device_invalid, sizeof(int),
                      cudaMemcpyDeviceToHost, frame->copy_stream) != cudaSuccess ||
      cudaEventRecord(frame->copied_event, frame->copy_stream) != cudaSuccess) {
    cudaGetLastError();
    return fail();
  }
  return enqueue_frame(std::move(frame));
}

bool observer_only(ModifyAccess *modify, Fix *&observer)
{
  if (modify->post_group_count() != 0 || modify->post_count() != 1) return false;
  const int index = modify->post_list()[0];
  observer = modify->fix[index];
  return observer && observer->id && observer->style &&
      std::strcmp(observer->id, "organizer_force_observer") == 0 &&
      std::strcmp(observer->style, "dpa/force-observer") == 0;
}
}  // namespace

extern "C" __attribute__((visibility("default")))
int dpa_force_observer_needs_force(long long step)
{
  return enabled && capture.active && capture.selected.count(step) != 0;
}

extern "C" void fused_observer_fix_modify_params(void *, int, char **)
    asm("_ZN9LAMMPS_NS3Fix13modify_paramsEiPPc");
extern "C" void fused_observer_fix_modify_params(void *opaque, int narg, char **arg)
{
  static auto original = reinterpret_cast<ModifyParamsFn>(
      dlsym(RTLD_NEXT, "_ZN9LAMMPS_NS3Fix13modify_paramsEiPPc"));
  if (!original) std::abort();
  auto *fix = static_cast<Fix *>(opaque);
  const bool target = enabled && narg >= 1 && fix->id &&
      std::strcmp(fix->id, "organizer_force_observer") == 0;
  if (target && async_write &&
      (std::strcmp(arg[0], "clear") == 0 || std::strcmp(arg[0], "capture") == 0) &&
      !flush_writer())
    std::abort();
  original(opaque, narg, arg);
  if (!target) return;
  if (std::strcmp(arg[0], "clear") == 0) {
    capture = CaptureState{};
    device_ids_valid = false;
    cached_atom_kokkos = nullptr;
    cached_views = DeviceViews{};
    return;
  }
  if (narg < 5 || std::strcmp(arg[0], "capture") != 0) return;
  CaptureState next;
  next.fix = fix;
  next.active = true;
  next.prefix = arg[1];
  next.begin = std::stoll(arg[3]);
  next.end = std::stoll(arg[4]);
  std::ifstream schedule(arg[2]);
  int64_t step = 0;
  while (schedule >> step) next.selected.insert(step);
  if (!schedule.eof() || next.selected.empty()) return;
  capture = std::move(next);
  const int observer_nodes = reinterpret_cast<FixAccess *>(fix)->atom_pointer()->nlocal;
  if (device_pack &&
      (observer_nodes <= device_small_maximum_nodes ||
       observer_nodes >= device_minimum_nodes)) {
    Atom *atom = reinterpret_cast<FixAccess *>(fix)->atom_pointer();
    if (!prepare_device_pool(atom)) device_ids_valid = false;
  } else {
    device_ids_valid = false;
  }
  if (timing)
    std::fprintf(stderr,
                 "DEVICE_OBSERVER_DEVICE_OBSERVER_CONFIG begin=%lld end=%lld selected=%zu "
                 "device_ready=%d prefix=%s\n",
                 static_cast<long long>(capture.begin), static_cast<long long>(capture.end),
                 capture.selected.size(), device_ids_valid ? 1 : 0, capture.prefix.c_str());
}

extern "C" void fused_observer_fused_post_force(void *, int)
    asm("_ZN9LAMMPS_NS12ModifyKokkos10post_forceEi");
extern "C" void fused_observer_fused_post_force(void *opaque, int vflag)
{
  static auto original = reinterpret_cast<PostForceFn>(
      dlsym(RTLD_NEXT, "_ZN9LAMMPS_NS12ModifyKokkos10post_forceEi"));
  static auto sync = reinterpret_cast<SyncFn>(
      dlsym(RTLD_NEXT, "_ZN9LAMMPS_NS10AtomKokkos4syncENS_14ExecutionSpaceEj"));
  if (!original || !sync) std::abort();
  if (!enabled || !capture.active) return original(opaque, vflag);

  auto *modify = static_cast<ModifyAccess *>(opaque);
  Fix *observer = nullptr;
  const int64_t step = modify->update_pointer()->ntimestep;
  Atom *atom = modify->atom_pointer();
  const bool selected = capture.selected.count(step) != 0;
  if (selected && packed_nvt_materialize_force() < 0) {
    std::fprintf(stderr, "PACKED_NVT observer force materialization failed\n");
    std::abort();
  }
  const bool eligible = atom->nlocal >= minimum_nodes &&
      static_cast<bigint>(atom->nlocal) == atom->natoms;
  if (!selected) {
    original(opaque, vflag);
    if (async_write && step >= capture.end) {
      const auto flush_begin = Clock::now();
      if (!flush_writer()) std::abort();
      if (timing)
        std::fprintf(stderr, "FUSED_OBSERVER_ASYNC_FLUSH step=%lld wait_ms=%.3f\n",
                     static_cast<long long>(step), millis(Clock::now() - flush_begin));
    }
    return;
  }
  if (!eligible || !observer_only(modify, observer) || observer != capture.fix ||
      (observer->datamask_read & OBSERVER_MASK) != OBSERVER_MASK)
    return original(opaque, vflag);

  const auto begin = Clock::now();
  const bool use_device = device_pack && async_write && device_ids_valid &&
      (atom->nlocal <= device_small_maximum_nodes ||
       atom->nlocal >= device_minimum_nodes);
  auto synchronized = begin;
  bool ok = false;
  if (use_device) {
    ok = enqueue_device_frame(modify, step, validate_writer ? ".device_observer" : "");
  } else {
    sync(modify->atom_kokkos_pointer(), static_cast<int>(Host), OBSERVER_MASK);
    synchronized = Clock::now();
    ok = async_write ? enqueue_async_frame(modify, step, validate_writer ? ".fused_observer" : "") :
                       write_frame(modify, step, validate_writer ? ".fused_observer" : "");
  }
  if (!ok) return original(opaque, vflag);
  if (validate_writer) {
    original(opaque, vflag);
    return;
  }
  observer->datamask_read = capture.selected.count(step + 1) ? OBSERVER_MASK : 0;
  if (timing)
    std::fprintf(stderr,
                 "DEVICE_OBSERVER_DEVICE_OBSERVER_TOTAL step=%lld sync_ms=%.3f submit_ms=%.3f "
                 "device=%d async=%d\n",
                 static_cast<long long>(step), millis(synchronized - begin),
                 millis(Clock::now() - synchronized), use_device ? 1 : 0,
                 async_write ? 1 : 0);
}

__attribute__((constructor)) static void initialize()
{
  const char *value = std::getenv("DPA_FUSED_OBSERVER_FUSED_OBSERVER");
  enabled = value && std::strcmp(value, "0") != 0;
  timing = std::getenv("DPA_FUSED_OBSERVER_FUSED_OBSERVER_TIMING") != nullptr;
  validate_writer = std::getenv("DPA_FUSED_OBSERVER_VALIDATE_WRITER") != nullptr;
  async_write = std::getenv("DPA_FUSED_OBSERVER_ASYNC_WRITE") != nullptr;
  device_pack = std::getenv("DPA_DEVICE_OBSERVER_DEVICE_OBSERVER") != nullptr;
  if (const char *nodes = std::getenv("DPA_DEVICE_OBSERVER_DEVICE_OBSERVER_MIN_NODES"))
    device_minimum_nodes = std::max(0, std::atoi(nodes));
  if (const char *nodes = std::getenv("DPA_OBSERVER_SCOPE_DEVICE_OBSERVER_SMALL_MAX_NODES"))
    device_small_maximum_nodes = std::max(-1, std::atoi(nodes));
  if (const char *nodes = std::getenv("DPA_FUSED_OBSERVER_FUSED_OBSERVER_MIN_NODES"))
    minimum_nodes = std::max(0, std::atoi(nodes));
  if (const char *threads = std::getenv("DPA_FUSED_OBSERVER_FUSED_OBSERVER_THREADS"))
    worker_threads = std::max(1, std::atoi(threads));
}

__attribute__((destructor)) static void finalize()
{
  stop_writer();
}
