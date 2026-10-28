#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "modify.h"
#include "update.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <memory>

using namespace LAMMPS_NS;

namespace {

using PackFn = int (*)(void *, int, double *);
using Clock = std::chrono::steady_clock;

constexpr int kRestartValues = 11;
constexpr size_t kTagViewOffset = 0x820;
constexpr size_t kTypeViewOffset = 0x868;
constexpr size_t kMaskViewOffset = 0x8b0;
constexpr size_t kImageViewOffset = 0x8f8;
constexpr size_t kXViewOffset = 0x940;
constexpr size_t kVViewOffset = 0x998;

class AtomVecAccess : public AtomVec {
 public:
  Atom *atom_pointer() const { return atom; }
  Modify *modify_pointer() const { return modify; }
  Comm *comm_pointer() const { return comm; }
  int restart_fields() const { return nrestart; }
  Update *update_pointer() const { return update; }
};

struct DeviceViews {
  const int32_t *tag = nullptr;
  const int32_t *type = nullptr;
  const int32_t *mask = nullptr;
  const int32_t *image = nullptr;
  const double *x = nullptr;
  const double *v = nullptr;
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

bool is_device_pointer(const void *pointer)
{
  cudaPointerAttributes attributes{};
  if (!pointer || cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return attributes.type == cudaMemoryTypeDevice;
}

bool read_device_views(Atom *atom, DeviceViews &views)
{
  void *object = static_cast<void *>(atom);
  views.tag = device_pointer<int32_t>(object, kTagViewOffset);
  views.type = device_pointer<int32_t>(object, kTypeViewOffset);
  views.mask = device_pointer<int32_t>(object, kMaskViewOffset);
  views.image = device_pointer<int32_t>(object, kImageViewOffset);
  views.x = device_pointer<double>(object, kXViewOffset);
  views.v = device_pointer<double>(object, kVViewOffset);
  views.capacity = view_extent(object, kXViewOffset, 0);
  if (atom->nlocal <= 0 || static_cast<size_t>(atom->nlocal) > views.capacity ||
      view_extent(object, kXViewOffset, 1) != 3 ||
      view_extent(object, kVViewOffset, 1) != 3)
    return false;
  const void *pointers[] = {
      views.tag, views.type, views.mask, views.image, views.x, views.v};
  return std::all_of(std::begin(pointers), std::end(pointers), is_device_pointer);
}

__device__ double integer_bits(int32_t value)
{
  return __longlong_as_double(static_cast<long long>(value));
}

__global__ void pack_restart_records(DeviceViews views, int count,
                                     double *records, int *invalid)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= count) return;
  const int32_t tag = views.tag[atom];
  const int32_t type = views.type[atom];
  if (tag <= 0 || type <= 0) atomicExch(invalid, 1);
  double *row = records + static_cast<size_t>(atom) * kRestartValues;
  const size_t xyz = static_cast<size_t>(atom) * 3;
  row[0] = static_cast<double>(kRestartValues);
  row[1] = views.x[xyz];
  row[2] = views.x[xyz + 1];
  row[3] = views.x[xyz + 2];
  row[4] = integer_bits(tag);
  row[5] = integer_bits(type);
  row[6] = integer_bits(views.mask[atom]);
  row[7] = integer_bits(views.image[atom]);
  row[8] = views.v[xyz];
  row[9] = views.v[xyz + 1];
  row[10] = views.v[xyz + 2];
}

bool enabled()
{
  static const bool value = [] {
    const char *flag = std::getenv("DPA_DEVICE_RESTART_DEVICE_RESTART");
    return flag && std::strcmp(flag, "0") != 0;
  }();
  return value;
}

bool timing_enabled()
{
  static const bool value = [] {
    const char *flag = std::getenv("DPA_DEVICE_RESTART_RESTART_TIMING");
    return flag && std::strcmp(flag, "0") != 0;
  }();
  return value;
}

bool validation_enabled()
{
  static const bool value = [] {
    const char *flag = std::getenv("DPA_DEVICE_RESTART_DEVICE_RESTART_VALIDATE");
    return flag && std::strcmp(flag, "0") != 0;
  }();
  return value;
}

bool async_restart_enabled()
{
  static const bool value = [] {
    const char *flag = std::getenv("DPA_DEVICE_RESTART_ASYNC_HANDOFF");
    return flag && std::strcmp(flag, "0") != 0;
  }();
  return value;
}

int minimum_nodes()
{
  static const int value = [] {
    const char *text = std::getenv("DPA_DEVICE_RESTART_DEVICE_RESTART_MIN_NODES");
    return text && *text ? std::max(1, std::atoi(text)) : 16384;
  }();
  return value;
}

double milliseconds(Clock::duration duration)
{
  return std::chrono::duration<double, std::milli>(duration).count();
}

PackFn original_pack()
{
  static auto function = reinterpret_cast<PackFn>(
      dlsym(RTLD_NEXT, "_ZN9LAMMPS_NS7AtomVec12pack_restartEiPd"));
  if (!function) {
    std::fprintf(stderr, "DEVICE_RESTART_DEVICE_RESTART missing original pack_restart: %s\n", dlerror());
    std::abort();
  }
  return function;
}

bool shared_output_enabled()
{
  static const bool value = [] {
    const char *flag = std::getenv("DPA_SHARED_OUTPUT_SNAPSHOT");
    return flag && std::strcmp(flag, "0") != 0;
  }();
  return value;
}

bool shared_output_validate()
{
  static const bool value = [] {
    const char *flag = std::getenv("DPA_SHARED_OUTPUT_VALIDATE");
    return flag && std::strcmp(flag, "0") != 0;
  }();
  return value;
}

// A state dump owns the pinned snapshot and the copied event.  Its writer may
// return that frame to its pool only after this registry has finished the
// restart payload conversion.  No model/force snapshot enters this registry.
struct SharedOutputRegistry {
  struct Image {
    AtomVec *vec = nullptr;
    int64_t step = -1;
    int count = 0;
    const double *values = nullptr;
    const int32_t *metadata = nullptr;
    const int *invalid = nullptr;
    cudaEvent_t ready = nullptr;
    double *destination = nullptr;
    size_t bytes = 0;
    bool active = false;
    bool borrowed = false;
    bool completed = false;
    std::shared_ptr<std::vector<double>> reference;
  };
  std::array<Image, 3> images{};
  std::mutex mutex;
  std::condition_variable cv;

  bool register_image(AtomVec *vec, int64_t step, int count,
                      const double *values, const int32_t *metadata,
                      const int *invalid, cudaEvent_t ready,
                      const double *reference)
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &image : images)
      if (image.active && image.values == values) return false;
    for (auto &image : images) {
      if (image.active) continue;
      image = Image{};
      image.vec = vec;
      image.step = step;
      image.count = count;
      image.values = values;
      image.metadata = metadata;
      image.invalid = invalid;
      image.ready = ready;
      image.active = true;
      if (reference) {
        const size_t length = static_cast<size_t>(count) * kRestartValues;
        image.reference = std::make_shared<std::vector<double>>(reference, reference + length);
      }
      return true;
    }
    return false;
  }

  bool borrow(AtomVec *vec, int64_t step, int count, double *destination, size_t bytes)
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &image : images) {
      if (!image.active || image.borrowed || image.completed ||
          image.vec != vec || image.step != step || image.count != count)
        continue;
      image.destination = destination;
      image.bytes = bytes;
      image.borrowed = true;
      if (timing_enabled())
        std::fprintf(stderr,
                     "SHARED_OUTPUT_ASYNC_BORROW step=%lld atoms=%d bytes=%zu\n",
                     static_cast<long long>(step), count, bytes);
      return true;
    }
    return false;
  }

  void cancel_unborrowed(AtomVec *vec, int64_t step)
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &image : images)
      if (image.active && !image.borrowed && image.vec == vec && image.step == step)
        image.completed = true;
    cv.notify_all();
  }

  int wait_release(const double *values)
  {
    std::unique_lock<std::mutex> lock(mutex);
    for (auto &image : images) {
      if (!image.active || image.values != values) continue;
      cv.wait(lock, [&] { return image.completed; });
      image = Image{};
      return 1;
    }
    return 0;
  }

  int finish_payload(void *destination, size_t bytes)
  {
    Image copy;
    Image *matched = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex);
      for (auto &image : images) {
        if (!image.active || !image.borrowed || image.completed ||
            image.destination != destination || image.bytes != bytes)
          continue;
        copy = image;
        matched = &image;
        break;
      }
    }
    if (!matched) return 0;
    const auto begin = Clock::now();
    bool ok = cudaEventSynchronize(copy.ready) == cudaSuccess &&
        copy.invalid && *copy.invalid == 0 &&
        bytes == static_cast<size_t>(copy.count) * kRestartValues * sizeof(double);
    const auto copied = Clock::now();
    std::atomic<bool> invalid{false};
    const int threads = copy.count < 65536 ? 1 : [] {
      const char *text = std::getenv("DPA_SHARED_OUTPUT_THREADS");
      return text && *text ? std::clamp(std::atoi(text), 1, 8) : 4;
    }();
    auto convert = [&](int first, int last) {
      for (int source = first; source < last; ++source) {
        const int32_t *metadata = copy.metadata + static_cast<size_t>(source) * 3;
        const int32_t id = metadata[0];
        if (id < 1 || id > copy.count) { invalid.store(true); continue; }
        const double *state = copy.values + static_cast<size_t>(id - 1) * 14;
        double *row = copy.destination + static_cast<size_t>(source) * kRestartValues;
        const int64_t integers[] = {id, static_cast<int32_t>(state[1]), metadata[1], metadata[2]};
        row[0] = static_cast<double>(kRestartValues);
        std::memcpy(row + 1, state + 2, 3 * sizeof(double));
        std::memcpy(row + 4, integers, sizeof(integers));
        std::memcpy(row + 8, state + 5, 3 * sizeof(double));
      }
    };
    if (ok) {
      if (threads == 1) convert(0, copy.count);
      else {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(threads));
        for (int worker = 0; worker < threads; ++worker) {
          const int first = static_cast<int>(static_cast<int64_t>(copy.count) * worker / threads);
          const int last = static_cast<int>(static_cast<int64_t>(copy.count) * (worker + 1) / threads);
          workers.emplace_back(convert, first, last);
        }
        for (auto &worker : workers) worker.join();
      }
      ok = !invalid.load();
      if (copy.reference) {
        const bool exact = std::memcmp(copy.destination, copy.reference->data(), bytes) == 0;
        std::fprintf(stderr,"SHARED_OUTPUT_BYTE_VALIDATE step=%lld bytes=%zu exact=%d\n",
                     static_cast<long long>(copy.step), bytes, exact ? 1 : 0);
        ok = ok && exact;
      }
    }
    if (timing_enabled())
      std::fprintf(stderr,
                   "SHARED_OUTPUT_RESTART_REPACK step=%lld wait_ms=%.6f convert_ms=%.6f "
                   "atoms=%d threads=%d bytes=%zu ok=%d\n",
                   static_cast<long long>(copy.step), milliseconds(copied - begin),
                   milliseconds(Clock::now() - copied), copy.count, threads, bytes, ok ? 1 : 0);
    {
      std::lock_guard<std::mutex> lock(mutex);
      matched->completed = true;
    }
    cv.notify_all();
    return ok ? 1 : -1;
  }
};

SharedOutputRegistry &shared_output_registry()
{
  static SharedOutputRegistry registry;
  return registry;
}

struct RestartState {
  struct AsyncTransfer {
    void *destination = nullptr;
    size_t bytes = 0;
    size_t capacity = 0;
    double *device_records = nullptr;
    int *device_invalid = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t packed = nullptr;
    cudaEvent_t ready = nullptr;
    int *pinned_invalid = nullptr;
    bool pending = false;
  };

  AtomVec *armed_vec = nullptr;
  Atom *armed_atom = nullptr;
  DeviceViews views{};
  double *device_records = nullptr;
  double *pinned_records = nullptr;
  int *device_invalid = nullptr;
  int *pinned_invalid = nullptr;
  size_t capacity = 0;
  int packed_count = 0;
  bool packed = false;
  bool async_batch = false;
  bool reported = false;
  long long validation_records = 0;
  long long validation_mismatches = 0;
  std::array<AsyncTransfer, 3> transfers{};
  std::mutex transfer_mutex;

  ~RestartState()
  {
    for (auto &transfer : transfers) {
      if (transfer.pending && transfer.ready) cudaEventSynchronize(transfer.ready);
      if (transfer.copy_stream) cudaStreamDestroy(transfer.copy_stream);
      if (transfer.packed) cudaEventDestroy(transfer.packed);
      if (transfer.ready) cudaEventDestroy(transfer.ready);
      if (transfer.device_records) cudaFree(transfer.device_records);
      if (transfer.device_invalid) cudaFree(transfer.device_invalid);
      if (transfer.pinned_invalid) cudaFreeHost(transfer.pinned_invalid);
    }
    if (device_records) cudaFree(device_records);
    if (pinned_records) cudaFreeHost(pinned_records);
    if (device_invalid) cudaFree(device_invalid);
    if (pinned_invalid) cudaFreeHost(pinned_invalid);
  }

  bool strict_eligibility(AtomVec *vec, Atom *expected_atom = nullptr)
  {
    if (!enabled() || !vec) return false;
    auto *access = reinterpret_cast<AtomVecAccess *>(vec);
    Atom *atom = access->atom_pointer();
    Modify *modify = access->modify_pointer();
    Comm *comm = access->comm_pointer();
    if (!atom || (expected_atom && atom != expected_atom) || atom->avec != vec ||
        atom->nlocal < minimum_nodes() || atom->natoms != atom->nlocal ||
        access->molecular != 0 || access->bonus_flag != 0 ||
        access->restart_fields() != 0 || atom->nextra_restart != 0 ||
        !modify || modify->restart_pbc_any != 0 || !comm || comm->nprocs != 1)
      return false;
    return true;
  }

  bool ensure_capacity(size_t count)
  {
    if (capacity >= count && device_records && pinned_records &&
        device_invalid && pinned_invalid)
      return true;
    if (device_records) cudaFree(device_records);
    if (pinned_records) cudaFreeHost(pinned_records);
    if (device_invalid) cudaFree(device_invalid);
    if (pinned_invalid) cudaFreeHost(pinned_invalid);
    device_records = pinned_records = nullptr;
    device_invalid = pinned_invalid = nullptr;
    capacity = 0;
    const size_t bytes = count * kRestartValues * sizeof(double);
    if (cudaMalloc(reinterpret_cast<void **>(&device_records), bytes) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&pinned_records), bytes,
                      cudaHostAllocPortable) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&device_invalid), sizeof(int)) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void **>(&pinned_invalid), sizeof(int),
                      cudaHostAllocPortable) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    capacity = count;
    return true;
  }

  bool arm(Atom *atom)
  {
    AtomVec *vec = atom ? atom->avec : nullptr;
    if (!strict_eligibility(vec, atom)) return false;
    DeviceViews candidate{};
    if (!read_device_views(atom, candidate) ||
        !ensure_capacity(static_cast<size_t>(atom->nlocal)))
      return false;
    armed_vec = vec;
    armed_atom = atom;
    views = candidate;
    packed = false;
    async_batch = false;
    packed_count = 0;
    if (!reported) {
      std::fprintf(stderr,
                   "DEVICE_RESTART_DEVICE_RESTART eligible nlocal=%d capacity=%zu values=%d\n",
                   atom->nlocal, views.capacity, kRestartValues);
      reported = true;
    }
    return true;
  }

  bool pinned_destination(void *destination)
  {
    cudaPointerAttributes attributes{};
    if (!destination || cudaPointerGetAttributes(&attributes, destination) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return attributes.type == cudaMemoryTypeHost;
  }

  void reset_transfer_resources(AsyncTransfer &transfer)
  {
    if (transfer.copy_stream) cudaStreamDestroy(transfer.copy_stream);
    if (transfer.packed) cudaEventDestroy(transfer.packed);
    if (transfer.ready) cudaEventDestroy(transfer.ready);
    if (transfer.device_records) cudaFree(transfer.device_records);
    if (transfer.device_invalid) cudaFree(transfer.device_invalid);
    if (transfer.pinned_invalid) cudaFreeHost(transfer.pinned_invalid);
    transfer = AsyncTransfer{};
  }

  bool ensure_transfer_resources(AsyncTransfer &transfer, size_t count)
  {
    if (!transfer.copy_stream &&
        cudaStreamCreateWithFlags(&transfer.copy_stream, cudaStreamNonBlocking) != cudaSuccess)
      return false;
    if (!transfer.packed &&
        cudaEventCreateWithFlags(&transfer.packed, cudaEventDisableTiming) != cudaSuccess)
      return false;
    if (!transfer.ready &&
        cudaEventCreateWithFlags(&transfer.ready, cudaEventDisableTiming) != cudaSuccess)
      return false;
    if (!transfer.pinned_invalid &&
        cudaHostAlloc(reinterpret_cast<void **>(&transfer.pinned_invalid), sizeof(int),
                      cudaHostAllocPortable) != cudaSuccess)
      return false;
    if (!transfer.device_invalid &&
        cudaMalloc(reinterpret_cast<void **>(&transfer.device_invalid), sizeof(int)) != cudaSuccess)
      return false;
    if (transfer.capacity >= count && transfer.device_records) return true;
    if (transfer.device_records) {
      cudaFree(transfer.device_records);
      transfer.device_records = nullptr;
      transfer.capacity = 0;
    }
    const size_t bytes = count * kRestartValues * sizeof(double);
    if (cudaMalloc(reinterpret_cast<void **>(&transfer.device_records), bytes) != cudaSuccess)
      return false;
    transfer.capacity = count;
    return true;
  }

  AsyncTransfer *acquire_transfer(void *destination, size_t bytes, size_t count)
  {
    std::lock_guard<std::mutex> lock(transfer_mutex);
    for (auto &transfer : transfers) {
      if (transfer.pending) continue;
      if (!ensure_transfer_resources(transfer, count)) {
        reset_transfer_resources(transfer);
        cudaGetLastError();
        return nullptr;
      }
      transfer.destination = destination;
      transfer.bytes = bytes;
      transfer.pending = true;
      return &transfer;
    }
    return nullptr;
  }

  void abandon_transfer(AsyncTransfer *transfer)
  {
    if (!transfer) return;
    std::lock_guard<std::mutex> lock(transfer_mutex);
    transfer->destination = nullptr;
    transfer->bytes = 0;
    transfer->pending = false;
  }

  bool pack_async(AtomVec *vec, double *destination)
  {
    if (!async_restart_enabled() || validation_enabled() ||
        !pinned_destination(destination))
      return false;
    auto *access = reinterpret_cast<AtomVecAccess *>(vec);
    Atom *atom = access->atom_pointer();
    if (!strict_eligibility(vec) || !arm(atom))
      fail("async_eligibility_changed");
    const int count = atom->nlocal;
    const size_t bytes = static_cast<size_t>(count) * kRestartValues * sizeof(double);
    if (shared_output_enabled() && access->update_pointer() &&
        shared_output_registry().borrow(vec, access->update_pointer()->ntimestep,
                                        count, destination, bytes)) {
      packed_count = count;
      packed = true;
      async_batch = true;
      validation_records = 0;
      validation_mismatches = 0;
      return true;
    }
    AsyncTransfer *transfer = acquire_transfer(destination, bytes, static_cast<size_t>(count));
    if (!transfer) return false;
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    const auto begin = Clock::now();
    if (cudaMemsetAsync(transfer->device_invalid, 0, sizeof(int), nullptr) != cudaSuccess) {
      abandon_transfer(transfer);
      fail("async_clear");
    }
    pack_restart_records<<<blocks, threads, 0, nullptr>>>(
        views, count, transfer->device_records, transfer->device_invalid);
    if (cudaGetLastError() != cudaSuccess ||
        cudaEventRecord(transfer->packed, nullptr) != cudaSuccess ||
        cudaStreamWaitEvent(transfer->copy_stream, transfer->packed, 0) != cudaSuccess ||
        cudaMemcpyAsync(destination, transfer->device_records, bytes,
                        cudaMemcpyDeviceToHost, transfer->copy_stream) != cudaSuccess ||
        cudaMemcpyAsync(transfer->pinned_invalid, transfer->device_invalid, sizeof(int),
                        cudaMemcpyDeviceToHost, transfer->copy_stream) != cudaSuccess ||
        cudaEventRecord(transfer->ready, transfer->copy_stream) != cudaSuccess) {
      abandon_transfer(transfer);
      fail("async_pack_or_copy");
    }
    packed_count = count;
    packed = true;
    async_batch = true;
    validation_records = 0;
    validation_mismatches = 0;
    if (timing_enabled())
      std::fprintf(stderr,
                   "DEVICE_RESTART_ASYNC_SUBMIT elapsed_ms=%.6f atoms=%d bytes=%zu "
                   "destination=%p\n",
                   milliseconds(Clock::now() - begin), count, bytes, destination);
    return true;
  }

  int wait_async(void *destination, size_t bytes)
  {
    if (shared_output_enabled()) {
      const int shared = shared_output_registry().finish_payload(destination, bytes);
      if (shared != 0) return shared;
    }
    AsyncTransfer *match = nullptr;
    {
      std::lock_guard<std::mutex> lock(transfer_mutex);
      for (auto &transfer : transfers)
        if (transfer.pending && transfer.destination == destination &&
            transfer.bytes == bytes) {
          match = &transfer;
          break;
        }
    }
    if (!match) return 0;
    const auto begin = Clock::now();
    const bool ok = cudaEventSynchronize(match->ready) == cudaSuccess &&
        match->pinned_invalid && *match->pinned_invalid == 0;
    {
      std::lock_guard<std::mutex> lock(transfer_mutex);
      match->destination = nullptr;
      match->bytes = 0;
      match->pending = false;
    }
    if (timing_enabled())
      std::fprintf(stderr,
                   "DEVICE_RESTART_ASYNC_WAIT elapsed_ms=%.6f bytes=%zu ok=%d\n",
                   milliseconds(Clock::now() - begin), bytes, ok ? 1 : 0);
    return ok ? 1 : -1;
  }

  [[noreturn]] void fail(const char *stage)
  {
    std::fprintf(stderr, "DEVICE_RESTART_DEVICE_RESTART fatal stage=%s cuda=%s\n",
                 stage, cudaGetErrorString(cudaGetLastError()));
    std::fflush(stderr);
    std::_Exit(89);
  }

  void pack(AtomVec *vec)
  {
    auto *access = reinterpret_cast<AtomVecAccess *>(vec);
    Atom *atom = access->atom_pointer();
    if (shared_output_enabled() && access->update_pointer())
      shared_output_registry().cancel_unborrowed(vec, access->update_pointer()->ntimestep);
    if (!strict_eligibility(vec) || !arm(atom))
      fail("eligibility_changed_after_sync_downgrade");
    const int count = atom->nlocal;
    const size_t bytes = static_cast<size_t>(count) * kRestartValues * sizeof(double);
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    const auto begin = Clock::now();
    if (cudaMemsetAsync(device_invalid, 0, sizeof(int), nullptr) != cudaSuccess)
      fail("clear");
    pack_restart_records<<<blocks, threads, 0, nullptr>>>(
        views, count, device_records, device_invalid);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(pinned_records, device_records, bytes,
                        cudaMemcpyDeviceToHost, nullptr) != cudaSuccess ||
        cudaMemcpyAsync(pinned_invalid, device_invalid, sizeof(int),
                        cudaMemcpyDeviceToHost, nullptr) != cudaSuccess ||
        cudaStreamSynchronize(nullptr) != cudaSuccess)
      fail("pack_or_copy");
    if (*pinned_invalid) fail("invalid_atom_metadata");
    packed_count = count;
    packed = true;
    validation_records = 0;
    validation_mismatches = 0;
    if (timing_enabled())
      std::fprintf(stderr,
                   "DEVICE_RESTART_DEVICE_RESTART_PACK elapsed_ms=%.6f atoms=%d bytes=%zu\n",
                   milliseconds(Clock::now() - begin), count, bytes);
  }
};

RestartState &state()
{
  static RestartState instance;
  return instance;
}

}  // namespace

extern "C" int dpa_device_restart_device_restart_available(void *atom_object)
{
  return state().arm(static_cast<Atom *>(atom_object)) ? 1 : 0;
}

extern "C" int dpa_device_restart_wait_async_payload(void *destination, size_t bytes)
{
  return state().wait_async(destination, bytes);
}

extern "C" int dpa_device_restart_register_state_snapshot(
    void *atom_object, int64_t step, int count, const double *values,
    const int32_t *metadata, const int *invalid, cudaEvent_t ready)
{
  Atom *atom = static_cast<Atom *>(atom_object);
  RestartState &current = state();
  if (!shared_output_enabled() || !async_restart_enabled() || validation_enabled() ||
      !atom || atom->nlocal != count || !values || !metadata || !invalid || !ready ||
      !current.strict_eligibility(atom->avec, atom)) return 0;
  auto *access = reinterpret_cast<AtomVecAccess *>(atom->avec);
  if (!access->update_pointer() || access->update_pointer()->ntimestep != step) return 0;
  const double *reference = nullptr;
  if (shared_output_validate()) {
    current.pack(atom->avec);
    reference = current.pinned_records;
  }
  return shared_output_registry().register_image(atom->avec, step, count,
      values, metadata, invalid, ready, reference) ? 1 : 0;
}

extern "C" int dpa_device_restart_release_state_snapshot(const double *values)
{
  return shared_output_registry().wait_release(values);
}

extern "C" int dpa_device_restart_pack_batch(void *object, double *buffer,
                                                     int atom_count,
                                                     int expected_values)
{
  auto *vec = static_cast<AtomVec *>(object);
  RestartState &current = state();
  if (!enabled() || !vec || !buffer || !current.strict_eligibility(vec)) return -1;
  auto *access = reinterpret_cast<AtomVecAccess *>(vec);
  Atom *atom = access->atom_pointer();
  if (!atom || atom_count != atom->nlocal || atom_count <= 0 ||
      static_cast<size_t>(expected_values) !=
          static_cast<size_t>(atom_count) * kRestartValues)
    return -1;
  if (!current.pack_async(vec, buffer)) return -1;
  if (timing_enabled())
    std::fprintf(stderr,
                 "DEVICE_RESTART_BATCH_BYPASS atoms=%d values=%d destination=%p\n",
                 atom_count, expected_values, buffer);
  return expected_values;
}

extern "C" int device_restart_pack_restart(void *, int, double *)
    asm("_ZN9LAMMPS_NS7AtomVec12pack_restartEiPd");

extern "C" int device_restart_pack_restart(void *object, int atom_index, double *buffer)
{
  auto *vec = static_cast<AtomVec *>(object);
  RestartState &current = state();
  if (!enabled() || !current.strict_eligibility(vec))
    return original_pack()(object, atom_index, buffer);
  if (atom_index == 0) {
    if (current.pack_async(vec, buffer))
      return static_cast<int>(static_cast<size_t>(current.packed_count) * kRestartValues);
    current.pack(vec);
  }
  if (!current.packed || current.armed_vec != vec || atom_index < 0 ||
      atom_index >= current.packed_count)
    return original_pack()(object, atom_index, buffer);
  if (current.async_batch) return 0;
  if (validation_enabled()) {
    double stock[kRestartValues]{};
    const int result = original_pack()(object, atom_index, stock);
    const double *candidate = current.pinned_records +
        static_cast<size_t>(atom_index) * kRestartValues;
    ++current.validation_records;
    const bool mismatch = result != kRestartValues ||
        std::memcmp(stock, candidate, kRestartValues * sizeof(double)) != 0;
    if (mismatch) {
      ++current.validation_mismatches;
      static bool reported = false;
      if (!reported) {
        int column = 0;
        while (column < kRestartValues &&
               std::memcmp(stock + column, candidate + column, sizeof(double)) == 0)
          ++column;
        unsigned long long stock_bits = 0;
        unsigned long long device_bits = 0;
        if (column < kRestartValues) {
          std::memcpy(&stock_bits, stock + column, sizeof(stock_bits));
          std::memcpy(&device_bits, candidate + column, sizeof(device_bits));
        }
        std::fprintf(stderr,
                     "DEVICE_RESTART_DEVICE_RESTART_VALIDATE mismatch atom=%d column=%d "
                     "result=%d stock_bits=%016llx device_bits=%016llx\n",
                     atom_index, column, result, stock_bits, device_bits);
        reported = true;
      }
    }
    if (atom_index + 1 == current.packed_count)
      std::fprintf(stderr,
                   "DEVICE_RESTART_DEVICE_RESTART_VALIDATE records=%lld mismatches=%lld exact=%d\n",
                   current.validation_records, current.validation_mismatches,
                   current.validation_mismatches == 0 ? 1 : 0);
    if (atom_index == 0) {
      const size_t values = static_cast<size_t>(current.packed_count) * kRestartValues;
      std::memcpy(buffer, current.pinned_records, values * sizeof(double));
      return static_cast<int>(values);
    }
    return 0;
  }
  // WriteRestart allocates one contiguous buffer using size_restart() and then
  // accumulates the return value of each per-atom call.  For the strictly
  // gated 11-double atomic layout, copy the complete batch on the first call
  // and return zero for the remaining loop iterations.  The final accumulated
  // count is unchanged, while the hot path performs one vectorized memcpy
  // instead of nlocal tiny copies.
  if (atom_index == 0) {
    const size_t values = static_cast<size_t>(current.packed_count) * kRestartValues;
    std::memcpy(buffer, current.pinned_records, values * sizeof(double));
    return static_cast<int>(values);
  }
  return 0;
}
