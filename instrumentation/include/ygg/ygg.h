#pragma once

/*
 * Ygg Instrumentation Library - public C/C++ API
 *
 * Ultra-low-overhead application event emission for trace collection.
 * Hot path: ~5-10 cycles, no allocation, no locks.
 *
 * Usage:
 *   YGG_EVENT(ParseFrame, bytes);
 *   YGG_EVENT(AdmissionAccepted, queue_depth);
 *
 * Events are written to a thread-local SPSC ring buffer and drained by a
 * single collector thread via shared memory (/dev/shm/ygg-<pid>).
 *
 * This header is valid as both C (for the C API) and C++ (for the C++ API).
 * The C++ API (EventRegistry, ThreadLocalSink, timestamp_ns) is header-only
 * and calls into the C ABI implemented in src/ygg.c, src/ring_buffer.c and
 * src/collector_thread.c.
 */

#if defined(__cplusplus)
#include <cstdint>
#else
#include <stdint.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Event kind registration */
typedef uint16_t ygg_event_kind_t;

/* Opaque handle for registered event types (reserved for future use). */
typedef struct ygg_event_registry *ygg_event_registry_t;

/* Initialize the instrumentation library.
 * Must be called once per process before any YGG_EVENT macros. */
int ygg_init(const char *process_name);

/* Register a custom event type. Returns assigned kind (>= 1000).
 * name: human-readable event name (e.g., "ParseFrame").
 * Returns: event kind ID, or 0 on failure. */
ygg_event_kind_t ygg_register_event(const char *name);

/* Emit an event with up to 3 uint64 arguments.
 * This is the hot path - inlined, no locks, no allocation.
 * kind: event kind from ygg_register_event or built-in kinds. */
void ygg_emit(ygg_event_kind_t kind, uint64_t arg0, uint64_t arg1, uint64_t arg2);

/* Try to emit an event. Returns 1 on success, 0 if the ring was full (dropped). */
int ygg_try_emit(ygg_event_kind_t kind, uint64_t arg0, uint64_t arg1, uint64_t arg2);

/* Explicit flush point. Rings drain asynchronously via the collector thread. */
void ygg_flush(void);

/* Configure the collector's forwarding target.
 *   socket_path: Unix socket the events are forwarded to (Rust collector).
 *   file_path:   raw 48-byte-per-event spill file (fallback / verification).
 * Either may be NULL to disable that sink. */
void ygg_collector_set_socket(const char *socket_path);
void ygg_collector_set_output(const char *file_path);

/* Number of events dropped (ring full or no reachable sink). */
uint64_t ygg_collector_dropped(void);

/* Calibration helpers (defined in src/ygg.c). */
uint64_t ygg_read_tsc(void);
uint64_t ygg_calibrate_tsc(void);

/* Shutdown and flush remaining events. */
void ygg_shutdown(void);

/* Calibration constants, populated by ygg_init(). Referenced by the inline
 * timestamp_ns() in the C++ API below. Defined in src/ygg.c. */
extern uint64_t ygg_tsc_freq_hz;
extern uint64_t ygg_tsc_to_ns_mult;
extern uint64_t ygg_monotonic_offset_ns;

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * C++ API
 * ============================================================================ */

#ifdef __cplusplus

namespace ygg {

/* Built-in event kinds (matching schema/event.fbs). */
enum class BuiltinKind : uint16_t {
    AppBase = 1000,

    SysEnter = 2000,
    SysExit  = 2001,

    SchedSwitch  = 3000,
    SchedWakeup  = 3001,
    SchedMigrate = 3002,

    BlockRqIssue    = 4000,
    BlockRqComplete = 4001,

    TcpSendmsg = 5000,
    TcpRecvmsg = 5001,

    PageFault       = 6000,
    PageFaultMajor  = 6001,

    PerfCycles          = 7000,
    PerfInstructions    = 7001,
    PerfCacheMisses     = 7002,
    PerfBranchMisses    = 7003,
    PerfContextSwitches = 7004,

    LokiInject = 8000,

    Custom = 9000,
};

/* Event registry for type-safe event emission. */
class EventRegistry {
public:
    explicit EventRegistry(const char *name) : name_(name) {}
    ~EventRegistry() = default;

    EventRegistry(const EventRegistry &) = delete;
    EventRegistry &operator=(const EventRegistry &) = delete;
    EventRegistry(EventRegistry &&) noexcept = default;
    EventRegistry &operator=(EventRegistry &&) noexcept = default;

    /* Register a custom event, returns the assigned kind. */
    [[nodiscard]] uint16_t register_event(const char *name) {
        return ygg_register_event(name);
    }

    /* Emit event with 0-3 arguments (convenience overloads). */
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

    /* Access to the underlying event name. */
    const char *name() const noexcept { return name_; }

private:
    const char *name_;
};

/* Thread-local event sink for zero-overhead emission. */
class ThreadLocalSink {
public:
    ThreadLocalSink() = default;
    ~ThreadLocalSink() = default;

    ThreadLocalSink(const ThreadLocalSink &) = delete;
    ThreadLocalSink &operator=(const ThreadLocalSink &) = delete;

    /* Try to emit an event. Returns false if the ring buffer is full. */
    [[nodiscard]] bool try_emit(uint16_t kind, uint64_t arg0,
                                uint64_t arg1, uint64_t arg2) noexcept {
        return ygg_try_emit(kind, arg0, arg1, arg2) != 0;
    }

    /* Flush any pending events to the collector. */
    void flush() noexcept { ygg_flush(); }
};

/* Get the thread-local sink (creates on first access). */
inline ThreadLocalSink &thread_local_sink() {
    thread_local ThreadLocalSink sink;
    return sink;
}

/* Get current timestamp in nanoseconds using the calibrated TSC.
 * Same clock domain as the eBPF collector. Hot path: rdtscp + fixed-point mul. */
[[nodiscard]] inline uint64_t timestamp_ns() noexcept {
    unsigned int aux = 0;
#if defined(__x86_64__) || defined(__i386__)
    uint64_t tsc = static_cast<uint64_t>(__builtin_ia32_rdtscp(&aux));
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t tsc = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull
                 + static_cast<uint64_t>(ts.tv_nsec);
#endif
    uint64_t ns = static_cast<uint64_t>(
        (static_cast<unsigned __int128>(tsc) * ygg_tsc_to_ns_mult) >> 32);
    return ns + ygg_monotonic_offset_ns;
}

} /* namespace ygg */

/* ============================================================================
 * Convenience macro for event emission
 * ============================================================================ */

/* Usage:
 *   YGG_EVENT(MyEvent, arg0, arg1, arg2);
 *   YGG_EVENT(ParseFrame, bytes);
 *   YGG_EVENT(AdmissionAccepted, queue_depth);
 *
 * The macro expands to a call that:
 *   1. Registers the event type on first use (via a function-local static).
 *   2. Emits the event via the thread-local sink. */
#define YGG_EVENT_IMPL_0(name)                                   \
    do {                                                         \
        static ygg::EventRegistry registry(#name);               \
        static uint16_t kind = registry.register_event(#name);   \
        ygg::thread_local_sink().try_emit(kind, 0, 0, 0);        \
    } while (0)

#define YGG_EVENT_IMPL_1(name, arg0)                             \
    do {                                                         \
        static ygg::EventRegistry registry(#name);               \
        static uint16_t kind = registry.register_event(#name);   \
        ygg::thread_local_sink().try_emit(                       \
            kind, static_cast<uint64_t>(arg0), 0, 0);           \
    } while (0)

#define YGG_EVENT_IMPL_2(name, arg0, arg1)                      \
    do {                                                         \
        static ygg::EventRegistry registry(#name);               \
        static uint16_t kind = registry.register_event(#name);   \
        ygg::thread_local_sink().try_emit(                       \
            kind, static_cast<uint64_t>(arg0),                   \
            static_cast<uint64_t>(arg1), 0);                    \
    } while (0)

#define YGG_EVENT_IMPL_3(name, arg0, arg1, arg2)                \
    do {                                                         \
        static ygg::EventRegistry registry(#name);               \
        static uint16_t kind = registry.register_event(#name);   \
        ygg::thread_local_sink().try_emit(                       \
            kind, static_cast<uint64_t>(arg0),                   \
            static_cast<uint64_t>(arg1),                         \
            static_cast<uint64_t>(arg2));                       \
    } while (0)

#define YGG_EVENT_GET_MACRO(_0, _1, _2, _3, NAME, ...) NAME
#define YGG_EVENT(...)                                                   \
    YGG_EVENT_GET_MACRO(__VA_ARGS__,                                    \
        YGG_EVENT_IMPL_3,                                               \
        YGG_EVENT_IMPL_2,                                               \
        YGG_EVENT_IMPL_1,                                               \
        YGG_EVENT_IMPL_0)(__VA_ARGS__)

#endif /* __cplusplus */
