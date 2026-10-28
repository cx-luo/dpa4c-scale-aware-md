#include <sys/mman.h>
#include <sys/sendfile.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;
using WriteRestartFn = void (*)(void *, const std::string *);
using SmallocFn = void *(*)(void *, long, const char *);
using SfreeFn = void (*)(void *, void *);
using FwriteFn = size_t (*)(const void *, size_t, size_t, FILE *);
using MemsetFn = void *(*)(void *, int, size_t);
using WaitAsyncPayloadFn = int (*)(void *, size_t);

double milliseconds(Clock::duration duration)
{
  return std::chrono::duration<double, std::milli>(duration).count();
}

bool enabled()
{
  static const bool value = [] {
    const char *flag = std::getenv("DPA_ZEROCOPY_RESTART_ZEROCOPY_RESTART");
    return flag && std::strcmp(flag, "0") != 0;
  }();
  return value;
}

bool timing_enabled()
{
  static const bool value = [] {
    const char *flag = std::getenv("DPA_ZEROCOPY_RESTART_RESTART_TIMING");
    return flag && std::strcmp(flag, "0") != 0;
  }();
  return value;
}

off_t minimum_async_bytes()
{
  static const off_t value = [] {
    constexpr off_t fallback = 1 << 20;
    const char *text = std::getenv("DPA_ZEROCOPY_RESTART_RESTART_MIN_BYTES");
    if (!text || !*text) return fallback;
    char *end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(text, &end, 10);
    return errno == 0 && end && *end == '\0' && parsed > 0
        ? static_cast<off_t>(parsed) : fallback;
  }();
  return value;
}

bool periodic_restart(const std::string &path)
{
  const size_t slash = path.find_last_of('/');
  const std::string name = path.substr(slash == std::string::npos ? 0 : slash + 1);
  constexpr const char *prefix = "native-state.";
  constexpr const char *suffix = ".restart";
  return name.size() > std::strlen(prefix) + std::strlen(suffix) &&
      name.compare(0, std::strlen(prefix), prefix) == 0 &&
      name.compare(name.size() - std::strlen(suffix), std::strlen(suffix), suffix) == 0;
}

std::string absolute_path(const std::string &path)
{
  if (!path.empty() && path.front() == '/') return path;
  char cwd[4096];
  if (!getcwd(cwd, sizeof(cwd))) std::abort();
  return std::string(cwd) + "/" + path;
}

WriteRestartFn original_write_restart()
{
  static auto function = reinterpret_cast<WriteRestartFn>(dlsym(
      RTLD_NEXT,
      "_ZN9LAMMPS_NS12WriteRestart5writeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"));
  if (!function) std::abort();
  return function;
}

SmallocFn original_smalloc()
{
  static auto function = reinterpret_cast<SmallocFn>(
      dlsym(RTLD_NEXT, "_ZN9LAMMPS_NS6Memory7smallocElPKc"));
  if (!function) std::abort();
  return function;
}

SfreeFn original_sfree()
{
  static auto function = reinterpret_cast<SfreeFn>(
      dlsym(RTLD_NEXT, "_ZN9LAMMPS_NS6Memory5sfreeEPv"));
  if (!function) std::abort();
  return function;
}

FwriteFn original_fwrite()
{
  static auto function = reinterpret_cast<FwriteFn>(dlsym(RTLD_NEXT, "fwrite"));
  if (!function) std::abort();
  return function;
}

MemsetFn original_memset()
{
  static auto function = reinterpret_cast<MemsetFn>(dlsym(RTLD_NEXT, "memset"));
  if (!function) std::abort();
  return function;
}

WaitAsyncPayloadFn wait_async_payload()
{
  static auto function = reinterpret_cast<WaitAsyncPayloadFn>(
      dlsym(RTLD_DEFAULT, "dpa_device_restart_wait_async_payload"));
  return function;
}

struct Slot {
  void *data = nullptr;
  size_t capacity = 0;
  size_t requested = 0;
  bool busy = false;
};

class BufferPool {
 public:
  Slot *acquire(size_t bytes)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] {
      return std::any_of(std::begin(slots_), std::end(slots_),
                         [](const Slot &slot) { return !slot.busy; });
    });
    Slot *slot = nullptr;
    for (auto &candidate : slots_)
      if (!candidate.busy) {
        slot = &candidate;
        break;
      }
    if (!slot) std::abort();
    if (slot->capacity < bytes) {
      if (slot->data) cudaFreeHost(slot->data);
      void *next = nullptr;
      if (cudaHostAlloc(&next, bytes, cudaHostAllocPortable) != cudaSuccess || !next)
        std::abort();
      slot->data = next;
      slot->capacity = bytes;
    }
    slot->requested = bytes;
    slot->busy = true;
    return slot;
  }

  void release(Slot *slot)
  {
    if (!slot) return;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      slot->requested = 0;
      slot->busy = false;
    }
    cv_.notify_all();
  }

  Slot *find(const void *pointer)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &slot : slots_)
      if (slot.data == pointer) return &slot;
    return nullptr;
  }

  void destroy()
  {
    for (auto &slot : slots_) {
      if (slot.data) cudaFreeHost(slot.data);
      slot = Slot{};
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  Slot slots_[3];
};

BufferPool &pool()
{
  static auto *instance = new BufferPool();
  return *instance;
}

struct Capture {
  Slot *slot = nullptr;
  FILE *stream = nullptr;
  off_t payload_offset = -1;
  size_t payload_bytes = 0;
  bool memset_skipped = false;
};

thread_local Capture *active_capture = nullptr;

struct Job {
  int source_fd = -1;
  off_t total_bytes = 0;
  off_t payload_offset = -1;
  size_t payload_bytes = 0;
  Slot *slot = nullptr;
  std::string target;
};

class AsyncWriter {
 public:
  void submit(Job job)
  {
    ensure_started();
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return failed_ || outstanding_ < 3; });
    check_failure_locked();
    queue_.push_back(std::move(job));
    ++outstanding_;
    cv_.notify_all();
  }

  void flush()
  {
    if (!started_) return;
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return failed_ || outstanding_ == 0; });
    check_failure_locked();
  }

  void stop()
  {
    if (!started_) return;
    flush();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    started_ = false;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Job> queue_;
  std::thread thread_;
  bool started_ = false;
  bool stopping_ = false;
  bool failed_ = false;
  int failure_errno_ = 0;
  int outstanding_ = 0;

  void ensure_started()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_) return;
    started_ = true;
    thread_ = std::thread([this] { loop(); });
  }

  [[noreturn]] void fail_now_locked()
  {
    std::fprintf(stderr, "ZEROCOPY_RESTART_ZEROCOPY_RESTART fatal errno=%d text=%s\n",
                 failure_errno_, std::strerror(failure_errno_));
    std::fflush(stderr);
    std::_Exit(91);
  }

  void check_failure_locked()
  {
    if (failed_) fail_now_locked();
  }

  static bool send_range(int output, int input, off_t begin, off_t end,
                         int &error_value)
  {
    off_t offset = begin;
    while (offset < end) {
      const ssize_t copied = sendfile(output, input, &offset,
                                      static_cast<size_t>(end - offset));
      if (copied > 0) continue;
      if (copied < 0 && errno == EINTR) continue;
      error_value = copied == 0 ? EIO : errno;
      return false;
    }
    return true;
  }

  static bool write_payload(int output, const void *source, size_t bytes,
                            int &error_value)
  {
    const auto *data = static_cast<const unsigned char *>(source);
    size_t offset = 0;
    while (offset < bytes) {
      const ssize_t written = write(output, data + offset, bytes - offset);
      if (written > 0) {
        offset += static_cast<size_t>(written);
        continue;
      }
      if (written < 0 && errno == EINTR) continue;
      error_value = written == 0 ? EIO : errno;
      return false;
    }
    return true;
  }

  static bool copy_to_file(const Job &job, int &error_value, double &device_wait_ms)
  {
    const int output = open(job.target.c_str(),
                            O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (output < 0) {
      error_value = errno;
      return false;
    }
    bool ok = true;
    if (job.slot && job.payload_offset >= 0 && job.payload_bytes) {
      const off_t suffix = job.payload_offset +
          static_cast<off_t>(job.payload_bytes);
      ok = send_range(output, job.source_fd, 0, job.payload_offset, error_value);
      if (ok) {
        const auto wait_begin = Clock::now();
        if (auto function = wait_async_payload()) {
          const int wait_result = function(job.slot->data, job.payload_bytes);
          if (wait_result < 0) {
            error_value = EIO;
            ok = false;
          }
        }
        device_wait_ms = milliseconds(Clock::now() - wait_begin);
      }
      ok = ok && write_payload(output, job.slot->data, job.payload_bytes, error_value) &&
          send_range(output, job.source_fd, suffix, job.total_bytes, error_value);
    } else {
      ok = send_range(output, job.source_fd, 0, job.total_bytes, error_value);
    }
    if (close(output) != 0 && ok) {
      error_value = errno;
      ok = false;
    }
    return ok;
  }

  void loop()
  {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (stopping_ && queue_.empty()) return;
        job = std::move(queue_.front());
        queue_.pop_front();
      }
      const auto begin = Clock::now();
      int error_value = 0;
      double device_wait_ms = 0.0;
      const bool ok = copy_to_file(job, error_value, device_wait_ms);
      close(job.source_fd);
      pool().release(job.slot);
      if (timing_enabled())
        std::fprintf(stderr,
                     "ZEROCOPY_RESTART_ZEROCOPY_WRITE elapsed_ms=%.6f device_wait_ms=%.6f "
                     "bytes=%lld payload=%zu ok=%d path=%s\n",
                     milliseconds(Clock::now() - begin),
                     device_wait_ms,
                     static_cast<long long>(job.total_bytes), job.payload_bytes,
                     ok ? 1 : 0, job.target.c_str());
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ok) {
          failed_ = true;
          failure_errno_ = error_value;
        }
        --outstanding_;
      }
      cv_.notify_all();
    }
  }
};

AsyncWriter &writer()
{
  static auto *instance = new AsyncWriter();
  return *instance;
}

int &periodic_mode()
{
  static int mode = 0;  // 0 unknown, 1 stock, 2 zero-copy async
  return mode;
}

}  // namespace

extern "C" int dpa_zerocopy_restart_shared_snapshot_available(const char *path)
{
  return enabled() && periodic_mode() == 2 && path && periodic_restart(path) ? 1 : 0;
}

extern "C" void *zerocopy_restart_smalloc(void *, long, const char *)
    asm("_ZN9LAMMPS_NS6Memory7smallocElPKc");

extern "C" void *zerocopy_restart_smalloc(void *object, long bytes, const char *name)
{
  if (enabled() && active_capture && bytes > 0 && name &&
      std::strcmp(name, "write_restart:buf") == 0) {
    Slot *slot = pool().acquire(static_cast<size_t>(bytes));
    active_capture->slot = slot;
    return slot->data;
  }
  return original_smalloc()(object, bytes, name);
}

extern "C" void zerocopy_restart_sfree(void *, void *)
    asm("_ZN9LAMMPS_NS6Memory5sfreeEPv");

extern "C" void zerocopy_restart_sfree(void *object, void *pointer)
{
  if (enabled() && pool().find(pointer)) return;
  original_sfree()(object, pointer);
}

extern "C" void *memset(void *destination, int value, size_t bytes)
{
  if (enabled() && active_capture && active_capture->slot && value == 0 &&
      destination == active_capture->slot->data &&
      bytes == active_capture->slot->requested) {
    active_capture->memset_skipped = true;
    return destination;
  }
  return original_memset()(destination, value, bytes);
}

extern "C" size_t fwrite(const void *pointer, size_t size, size_t count,
                         FILE *stream)
{
  if (enabled() && active_capture && active_capture->slot && count && size &&
      pointer == active_capture->slot->data &&
      size <= SIZE_MAX / count &&
      size * count == active_capture->slot->requested &&
      active_capture->payload_offset < 0) {
    const off_t offset = ftello(stream);
    if (offset < 0 || fseeko(stream, static_cast<off_t>(size * count), SEEK_CUR) != 0)
      return 0;
    active_capture->stream = stream;
    active_capture->payload_offset = offset;
    active_capture->payload_bytes = size * count;
    return count;
  }
  return original_fwrite()(pointer, size, count, stream);
}

extern "C" void zerocopy_restart_write_restart(void *, const std::string *)
    asm("_ZN9LAMMPS_NS12WriteRestart5writeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");

extern "C" void zerocopy_restart_write_restart(void *object, const std::string *path)
{
  if (!enabled() || !path || !periodic_restart(*path)) {
    const auto flush_begin = Clock::now();
    if (enabled()) writer().flush();
    const auto original_begin = Clock::now();
    original_write_restart()(object, path);
    if (timing_enabled())
      std::fprintf(stderr,
                   "ZEROCOPY_RESTART_ZEROCOPY_SYNC flush_ms=%.6f original_ms=%.6f path=%s\n",
                   milliseconds(original_begin - flush_begin),
                   milliseconds(Clock::now() - original_begin),
                   path ? path->c_str() : "<null>");
    return;
  }

  if (periodic_mode() == 0) {
    const auto begin = Clock::now();
    original_write_restart()(object, path);
    struct stat status {};
    const std::string target = absolute_path(*path);
    const bool measured = stat(target.c_str(), &status) == 0 && status.st_size > 0;
    periodic_mode() = measured && status.st_size >= minimum_async_bytes() ? 2 : 1;
    if (timing_enabled())
      std::fprintf(stderr,
                   "ZEROCOPY_RESTART_ZEROCOPY_GATE mode=%s elapsed_ms=%.6f bytes=%lld "
                   "threshold=%lld path=%s\n",
                   periodic_mode() == 2 ? "zerocopy" : "stock",
                   milliseconds(Clock::now() - begin),
                   static_cast<long long>(measured ? status.st_size : 0),
                   static_cast<long long>(minimum_async_bytes()), path->c_str());
    return;
  }
  if (periodic_mode() == 1) {
    original_write_restart()(object, path);
    return;
  }

  const auto begin = Clock::now();
  const int memory_fd = memfd_create("dpa-zerocopy_restart-restart", MFD_CLOEXEC);
  if (memory_fd < 0) {
    original_write_restart()(object, path);
    return;
  }
  const std::string memory_path = "/proc/self/fd/" + std::to_string(memory_fd);
  Capture capture;
  active_capture = &capture;
  original_write_restart()(object, &memory_path);
  active_capture = nullptr;
  struct stat status {};
  if (fstat(memory_fd, &status) != 0 || status.st_size <= 0 || !capture.slot ||
      capture.payload_offset < 0 || !capture.payload_bytes) {
    if (capture.slot) pool().release(capture.slot);
    close(memory_fd);
    std::abort();
  }
  const auto packed = Clock::now();
  writer().submit(Job{memory_fd, status.st_size, capture.payload_offset,
                      capture.payload_bytes, capture.slot, absolute_path(*path)});
  if (timing_enabled())
    std::fprintf(stderr,
                 "ZEROCOPY_RESTART_ZEROCOPY_SUBMIT total_ms=%.6f pack_ms=%.6f bytes=%lld "
                 "payload=%zu offset=%lld memset_skipped=%d path=%s\n",
                 milliseconds(Clock::now() - begin), milliseconds(packed - begin),
                 static_cast<long long>(status.st_size), capture.payload_bytes,
                 static_cast<long long>(capture.payload_offset),
                 capture.memset_skipped ? 1 : 0, path->c_str());
}

__attribute__((destructor)) static void zerocopy_restart_finish()
{
  if (enabled()) {
    writer().stop();
    pool().destroy();
  }
}
