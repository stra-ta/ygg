// norn_capture.cpp - standalone Ygg capture of REAL Norn MPMC queue contention.
//
// This program is intentionally standalone: it is NOT part of the Norn build.
// It includes Norn's header-only queue (real Vyukov mpmc_ring, real atomics,
// real CAS/spin behavior) and links Ygg's instrumentation static library to
// emit application-level YGG events while exercising the queue under
// contention.
//
// Four backoff regimes are exercised by the *capture program's* retry loop,
// approximating the contention strategies a caller might choose around a real
// Norn queue:
//   tight        - busy-spin on contention (no yield)
//   yield        - std::this_thread::yield() on contention
//   bounded      - capped spin, then yield past the cap
//   exponential  - doubling spin (1,2,4,... up to a ceiling)
//
// Event kinds (Application base 1000):
//   1000 Push        arg0 = approximate queue depth after push
//   1001 Pop         arg0 = approximate queue depth after pop
//   1002 CAS retry   arg0 = cumulative retry count since last success
//   1003 Yield       arg0 = 0
//   1004 Spin        arg0 = spin iterations performed
//
// macOS lacks eBPF, so only application-level events (YGG_EVENT) are captured.
// That is acceptable for V0.1 contention (application-level). The collector
// spills to the configured output path (or /tmp/ygg-<pid>.events by default).
//
// Usage: norn_capture <policy> <grid> <out_path> [duration_ms]
//   grid      - producers = consumers = grid (1,2,4,8)
//   out_path  - spill file path (32-byte header + 48-byte records)

// Ygg's public header uses POSIX struct timespec / CLOCK_MONOTONIC in its
// inline timestamp_ns(); ensure those are declared before including it.
#include <ctime>

#include <ygg/ygg.h>

#include <norn/queue/mpmc_ring.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

enum class Policy { Tight, Yield, Bounded, Exponential };

// Fixed event kinds (Application base 1000).
constexpr uint16_t KIND_PUSH = 1000;
constexpr uint16_t KIND_POP = 1001;
constexpr uint16_t KIND_CAS_RETRY = 1002;
constexpr uint16_t KIND_YIELD = 1003;
constexpr uint16_t KIND_SPIN = 1004;

constexpr std::size_t kQueueCapacity = 1024;

// Bounds on total emitted volume so the per-thread ring (65536 events) never
// overflows badly and the spill files stay a manageable size.
constexpr uint64_t kGlobalEmitCap = 150'000;  // hard stop for the whole run
constexpr uint64_t kLocalEmitCap = 30'000;   // per-thread ceiling (< ring size)

Policy g_policy = Policy::Tight;
std::atomic<uint64_t> g_emitted{0};
std::atomic<bool> g_stop{false};
// Approximate concurrent depth (racy by design; only used as an event arg).
std::atomic<int64_t> g_depth{0};

norn::mpmc_ring<uint64_t, kQueueCapacity> g_queue;

// Emit one event, but only while both the per-thread and global caps allow it.
// Returns true if an event was actually written.
inline bool maybe_emit(uint16_t kind, uint64_t a0, uint64_t a1, uint64_t a2,
                       uint64_t* local) {
  if (*local >= kLocalEmitCap) return false;
  if (g_emitted.load(std::memory_order_relaxed) >= kGlobalEmitCap) {
    g_stop.store(true, std::memory_order_relaxed);
    return false;
  }
  ygg_emit(kind, a0, a1, a2);
  ++(*local);
  g_emitted.fetch_add(1, std::memory_order_relaxed);
  return true;
}

// Cheap busy-spin of n iterations (volatile store so it is not optimized away).
inline void busy_spin(uint32_t n) {
  volatile uint32_t sink = 0;
  for (uint32_t i = 0; i < n; ++i) {
    sink = i;
  }
  (void)sink;
}

// Apply the regime-specific backoff after a failed push/pop, emitting Spin or
// Yield events plus a CAS-retry event carrying the cumulative retry count.
inline void backoff(Policy p, uint32_t& retry, uint64_t* local) {
  const uint32_t capped_retry = retry < 65535 ? retry : 65535;
  switch (p) {
    case Policy::Tight: {
      const uint32_t n = 16;
      busy_spin(n);
      maybe_emit(KIND_SPIN, n, 0, 0, local);
      ++retry;
      maybe_emit(KIND_CAS_RETRY, capped_retry, 0, 0, local);
      break;
    }
    case Policy::Yield: {
      std::this_thread::yield();
      maybe_emit(KIND_YIELD, 0, 0, 0, local);
      ++retry;
      maybe_emit(KIND_CAS_RETRY, capped_retry, 0, 0, local);
      break;
    }
    case Policy::Bounded: {
      if (retry < 64) {
        const uint32_t n = 8;
        busy_spin(n);
        maybe_emit(KIND_SPIN, n, 0, 0, local);
      } else {
        std::this_thread::yield();
        maybe_emit(KIND_YIELD, 0, 0, 0, local);
        retry = 0;
      }
      ++retry;
      maybe_emit(KIND_CAS_RETRY, capped_retry, 0, 0, local);
      break;
    }
    case Policy::Exponential: {
      uint32_t e = retry < 20 ? retry : 20;
      uint32_t n = (1u << e);
      if (n > 4096) n = 4096;
      busy_spin(n);
      maybe_emit(KIND_SPIN, n, 0, 0, local);
      ++retry;
      maybe_emit(KIND_CAS_RETRY, capped_retry, 0, 0, local);
      break;
    }
  }
}

void producer(int tid, uint64_t* local) {
  uint64_t counter = static_cast<uint64_t>(tid) * 0x9E3779B97F4A7C15ULL + 1;
  uint32_t retry = 0;
  while (!g_stop.load(std::memory_order_relaxed) &&
         g_emitted.load(std::memory_order_relaxed) < kGlobalEmitCap) {
    uint64_t item = counter++;
    if (g_queue.try_push(item)) {
      int64_t d = g_depth.fetch_add(1, std::memory_order_relaxed) + 1;
      maybe_emit(KIND_PUSH, static_cast<uint64_t>(d), 0, 0, local);
      retry = 0;
    } else {
      backoff(g_policy, retry, local);
    }
  }
}

void consumer(int tid, uint64_t* local) {
  (void)tid;
  uint32_t retry = 0;
  uint64_t got = 0;
  while (!g_stop.load(std::memory_order_relaxed) &&
         g_emitted.load(std::memory_order_relaxed) < kGlobalEmitCap) {
    if (g_queue.try_pop(got)) {
      int64_t d = g_depth.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (d < 0) d = 0;
      maybe_emit(KIND_POP, static_cast<uint64_t>(d), 0, 0, local);
      retry = 0;
    } else {
      backoff(g_policy, retry, local);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <policy> <grid> <out_path> [duration_ms]\n"
                 "  policy in {tight,yield,bounded,exponential}\n"
                 "  grid   producers = consumers = grid (1,2,4,8)\n",
                 argv[0]);
    return 2;
  }

  std::string policy_s = argv[1];
  if (policy_s == "tight")
    g_policy = Policy::Tight;
  else if (policy_s == "yield")
    g_policy = Policy::Yield;
  else if (policy_s == "bounded")
    g_policy = Policy::Bounded;
  else if (policy_s == "exponential")
    g_policy = Policy::Exponential;
  else {
    std::fprintf(stderr, "unknown policy '%s'\n", policy_s.c_str());
    return 2;
  }

  int grid = std::atoi(argv[2]);
  if (grid < 1) grid = 1;
  const char* out_path = argv[3];
  int dur_ms = argc > 4 ? std::atoi(argv[4]) : 3000;
  if (dur_ms < 100) dur_ms = 100;

  // Direct the spill file to the requested path *before* init so the collector
  // writes there instead of the default /tmp/ygg-<pid>.events.
  ygg_collector_set_output(out_path);

  if (ygg_init("norn") != 0) {
    std::fprintf(stderr, "ygg_init failed\n");
    return 1;
  }

  const auto start = std::chrono::steady_clock::now();
  const int nprod = grid;
  const int ncons = grid;

  std::vector<std::thread> threads;
  std::vector<uint64_t> local_emit(static_cast<std::size_t>(nprod + ncons), 0);
  int idx = 0;
  for (int i = 0; i < nprod; ++i)
    threads.emplace_back(producer, i, &local_emit[static_cast<std::size_t>(idx++)]);
  for (int i = 0; i < ncons; ++i)
    threads.emplace_back(consumer, nprod + i, &local_emit[static_cast<std::size_t>(idx++)]);

  // Monitor loop: stop on global emit cap or duration.
  while (!g_stop.load(std::memory_order_relaxed)) {
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    if (ms >= dur_ms ||
        g_emitted.load(std::memory_order_relaxed) >= kGlobalEmitCap) {
      g_stop.store(true, std::memory_order_relaxed);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  for (auto& t : threads) t.join();

  uint64_t dropped = ygg_collector_dropped();
  ygg_shutdown();

  // Best-effort spill-file size report (confirms the collector wrote it).
  long spill_bytes = -1;
  std::FILE* f = std::fopen(out_path, "rb");
  if (f) {
    std::fseek(f, 0, SEEK_END);
    spill_bytes = std::ftell(f);
    std::fclose(f);
  }

  std::printf(
      "capture done: policy=%s grid=%d threads=%d events=%llu dropped=%llu "
      "spill_bytes=%ld path=%s\n",
      policy_s.c_str(), grid, nprod + ncons,
      static_cast<unsigned long long>(g_emitted.load()),
      static_cast<unsigned long long>(dropped), spill_bytes, out_path);
  return 0;
}
