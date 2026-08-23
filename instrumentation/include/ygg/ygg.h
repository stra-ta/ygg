#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>

/**
 * Ygg Instrumentation Library
 *
 * Ultra-low-overhead application event emission for trace collection.
 * Hot path: ~5-10 cycles, no allocation, no locks.
 *
 * Usage:
 *   YGG_EVENT(ParseFrame, bytes);
 *   YGG_EVENT(AdmissionAccepted, queue_depth);
 *
 * Events are written to a thread-local ring buffer and drained
 * by a single collector thread via shared memory.
 */

#ifdef __cplusplus
extern "C" {
#endif

// Event kind registration
typedef uint16_t ygg_event_kind_t;

// Opaque handle for registered event types
typedef struct ygg_event_registry *ygg_event_registry_t;

// Initialize the instrumentation library.
// Must be called once per process before any YGG_EVENT macros.
int ygg_init(const char *process_name);

// Register a custom event type. Returns assigned kind (>= 1000).
// name: human-readable event name (e.g., "ParseFrame")
// Returns: event kind ID, or 0 on failure
ygg_event_kind_t ygg_register_event(const char *name);

// Emit an event with up to 3 uint64 arguments.
// This is the hot path - inlined, no locks, no allocation.
// kind: event kind from ygg_register_event or built-in kinds
// arg0, arg1, arg2: event payload (interpretation depends on kind)
static inline void ygg_emit(ygg_event_kind_t kind, uint64_t arg0, uint64_t arg1, uint64_t arg2);

// Shutdown and flush remaining events.
void ygg_shutdown(void);

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ API
// ============================================================================

#ifdef __cplusplus

namespace ygg {

// Built-in event kinds (matching schema/event.fbs)
enum class BuiltinKind : uint16_t {
    // Application events start at 1000
    AppBase = 1000,

    // Syscall events (kernel-side, but can be emitted from user space too)
    SysEnter = 2000,
    SysExit = 2001,

    // Scheduler events
    SchedSwitch = 3000,
    SchedWakeup = 3001,
    SchedMigrate = 3002,

    // Block I/O
    BlockRqIssue = 4000,
    BlockRqComplete = 4001,

    // Network
    TcpSendmsg = 5000,
    TcpRecvmsg = 5001,

    // Memory
    PageFault = 6000,
    PageFaultMajor = 6001,

    // Hardware counters
    PerfCycles = 7000,
    PerfInstructions = 7001,
    PerfCacheMisses = 7002,
    PerfBranchMisses = 7003,
    PerfContextSwitches = 7004,

    // Loki fault injection
    LokiInject = 8000,

    // Dynamic custom events
    Custom = 9000,
};

// Event registry for type-safe event emission
class EventRegistry {
public:
    explicit EventRegistry(const char *name);
    ~EventRegistry();

    // Non-copyable, movable
    EventRegistry(const EventRegistry&) = delete;
    EventRegistry& operator=(const EventRegistry&) = delete;
    EventRegistry(EventRegistry&&) noexcept;
    EventRegistry& operator=(EventRegistry&&) noexcept;

    // Register a custom event, returns assigned kind
    [[nodiscard]] uint16_t register_event(const char *name);

    // Emit event with 0-3 arguments (overloads for convenience)
    inline void emit(uint16_t kind) const noexcept {
        ygg_emit(kind, 0, 0, 0);
    }
    inline void emit(uint16_t kind, uint64_t arg0) const noexcept {
        ygg_emit(kind, arg0, 0, 0);
    }
    inline void emit(uint16_t kind, uint64_t arg0, uint64_t arg1) const noexcept {
        ygg_emit(kind, arg0, arg1, 0);
    }
    inline void emit(uint16_t kind, uint64_t arg0, uint64_t arg1, uint64_t arg2) const noexcept {
        ygg_emit(kind, arg0, arg1, arg2);
    }

    // Access to the underlying C handle
    ygg_event_registry_t handle() const noexcept { return registry_; }

private:
    ygg_event_registry_t registry_ = nullptr;
};

// Thread-local event sink for zero-overhead emission
class ThreadLocalSink {
public:
    ThreadLocalSink() = default;
    ~ThreadLocalSink() = default;

    // Non-copyable, non-movable (thread-local)
    ThreadLocalSink(const ThreadLocalSink&) = delete;
    ThreadLocalSink& operator=(const ThreadLocalSink&) = delete;

    // Try to emit an event. Returns false if ring buffer is full.
    [[nodiscard]] bool try_emit(uint16_t kind, uint64_t arg0, uint64_t arg1, uint64_t arg2) noexcept;

    // Flush any pending events to the collector
    void flush() noexcept;
};

// Get the thread-local sink (creates on first access)
ThreadLocalSink& thread_local_sink();

} // namespace ygg

// ============================================================================
// Convenience macro for event emission
// ============================================================================

// Usage:
//   YGG_EVENT(MyEvent, arg0, arg1, arg2);
//   YGG_EVENT(ParseFrame, bytes);
//   YGG_EVENT(AdmissionAccepted, queue_depth);
//
// The macro expands to a call that:
// 1. Registers the event type on first use (via static variable)
// 2. Emits the event via thread-local sink
// 3. Falls back to direct emission if sink is unavailable

#define YGG_EVENT_IMPL_0(name) \
    do { \
        static ygg::EventRegistry registry(#name); \
        static uint16_t kind = registry.register_event(#name); \
        ygg::thread_local_sink().try_emit(kind, 0, 0, 0); \
    } while (0)

#define YGG_EVENT_IMPL_1(name, arg0) \
    do { \
        static ygg::EventRegistry registry(#name); \
        static uint16_t kind = registry.register_event(#name); \
        ygg::thread_local_sink().try_emit(kind, static_cast<uint64_t>(arg0), 0, 0); \
    } while (0)

#define YGG_EVENT_IMPL_2(name, arg0, arg1) \
    do { \
        static ygg::EventRegistry registry(#name); \
        static uint16_t kind = registry.register_event(#name); \
        ygg::thread_local_sink().try_emit(kind, \
            static_cast<uint64_t>(arg0), \
            static_cast<uint64_t>(arg1), \
            0); \
    } while (0)

#define YGG_EVENT_IMPL_3(name, arg0, arg1, arg2) \
    do { \
        static ygg::EventRegistry registry(#name); \
        static uint16_t kind = registry.register_event(#name); \
        ygg::thread_local_sink().try_emit(kind, \
            static_cast<uint64_t>(arg0), \
            static_cast<uint64_t>(arg1), \
            static_cast<uint64_t>(arg2)); \
    } while (0)

#define YGG_EVENT_GET_MACRO(_0, _1, _2, _3, NAME, ...) NAME
#define YGG_EVENT(...) \
    YGG_EVENT_GET_MACRO(__VA_ARGS__, \
        YGG_EVENT_IMPL_3, \
        YGG_EVENT_IMPL_2, \
        YGG_EVENT_IMPL_1, \
        YGG_EVENT_IMPL_0)(__VA_ARGS__)

// ============================================================================
// Timestamp helpers (rdtsc/rdtscp with calibration)
// ============================================================================

namespace ygg {

// Get current timestamp in nanoseconds using calibrated TSC
// This is the same clock domain as the eBPF collector
[[nodiscard]] inline uint64_t timestamp_ns() noexcept {
    // Implementation uses rdtsc/rdtscp with calibration constants
    // set by ygg_init() via shared memory
    extern uint64_t ygg_tsc_freq_hz;
    extern uint64_t ygg_monotonic_offset_ns;

    unsigned int aux;
    uint64_t tsc = __builtin_ia32_rdtscp(&aux);
    return (tsc * 1000000000ull) / ygg_tsc_freq_hz + ygg_monotonic_offset_ns;
}

// Calibration constants (populated by ygg_init from collector)
extern uint64_t ygg_tsc_freq_hz;
extern uint64_t ygg_monotonic_offset_ns;

} // namespace ygg

#endif // __cplusplus