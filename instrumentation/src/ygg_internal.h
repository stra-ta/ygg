#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Internal shared declarations for the Ygg instrumentation C sources.
 *
 * This header is included only by the C translation units compiled by
 * build.rs (ygg.c, ring_buffer.c, collector_thread.c). It is NOT part of the
 * public ygg/ygg.h API and is not C++-compatible (it uses C11 _Atomic).
 *
 * Memory model
 * ------------
 * Each emitting thread owns a private "ring slot" inside the shared header
 * region and a private ring-data shared memory object. The producer thread
 * writes `head`; the single collector thread writes `tail`. They live on
 * separate cache lines. This is a classic SPSC ring, safe across processes
 * because both ends map the same shared memory.
 */

#define YGG_RING_SIZE       (1u << 16)    /* 65536 events per thread            */
#define YGG_RING_MASK       (YGG_RING_SIZE - 1)
#define YGG_MAX_RINGS       4096
#define YGG_CACHE_LINE      64
#define YGG_SHM_NAME_MAX    48

/* Fixed 48-byte event record (matches schema/event.fbs and EVENT_SCHEMA.md). */
struct ygg_event {
    uint64_t timestamp_ns;
    uint32_t cpu;
    uint32_t pid;
    uint32_t tid;
    uint16_t kind;
    uint16_t padding;
    uint64_t arg0;
    uint64_t arg1;
    uint64_t arg2;
};
_Static_assert(sizeof(struct ygg_event) == 48, "event record must be 48 bytes");

/*
 * Per-thread ring slot, lives inside the shared header region.
 *   head  -> owned (written) by the producer thread
 *   tail  -> owned (written) by the collector thread
 * They sit on separate cache lines to avoid false sharing.
 */
struct ygg_ring_slot {
    _Atomic uint64_t head;
    char _pad_head[YGG_CACHE_LINE - sizeof(_Atomic uint64_t)];
    _Atomic uint64_t tail;
    char _pad_tail[YGG_CACHE_LINE - sizeof(_Atomic uint64_t)];
    uint32_t tid;
    int32_t  cpu;
    _Atomic uint32_t active;
    uint32_t reserved;
    char     shm_name[YGG_SHM_NAME_MAX];
};
_Static_assert(sizeof(struct ygg_ring_slot) % YGG_CACHE_LINE == 0,
                "slot must be a whole number of cache lines");

#define YGG_SHM_MAGIC 0x5967673100000001ull   /* "Ygg1" */

struct ygg_shm_header {
    uint64_t magic;
    uint64_t tsc_freq_hz;
    uint64_t tsc_to_ns_mult;       /* 32.32 fixed point: ns = (tsc * mult) >> 32 */
    uint64_t monotonic_offset_ns;
    uint32_t pid;
    uint32_t collector_pid;
    _Atomic uint32_t shutdown_flag;
    _Atomic uint32_t ring_count;
    uint32_t max_rings;
    uint32_t reserved;
    _Alignas(YGG_CACHE_LINE) struct ygg_ring_slot slots[YGG_MAX_RINGS];
};

/* Global shared header, mapped in ygg_init (defined in ygg.c). */
extern struct ygg_shm_header *g_ygg_shm;

/* Calibration constants (defined in ygg.c; also exported for the C++ header). */
extern uint64_t ygg_tsc_freq_hz;
extern uint64_t ygg_tsc_to_ns_mult;
extern uint64_t ygg_monotonic_offset_ns;

/* Read a TSC/clock value. On x86 returns raw TSC; elsewhere CLOCK_MONOTONIC ns
 * (with mult=1, offset=0 the same conversion formula applies). */
uint64_t ygg_read_tsc(void);

/* Calibrate TSC against CLOCK_MONOTONIC_RAW; populates the three globals. */
uint64_t ygg_calibrate_tsc(void);

/* Per-thread ring emit. Returns 1 on success, 0 if the ring was full (dropped). */
int ygg_ring_try_emit(uint16_t kind, uint64_t a0, uint64_t a1, uint64_t a2);

/* Map a per-thread ring-data shared memory object by name. Returns base or NULL. */
struct ygg_event *ygg_map_ring_data(const char *shm_name);

/* Current thread id (Linux gettid, portable fallback). */
uint32_t ygg_gettid(void);

/* Collector thread lifecycle (defined in collector_thread.c). */
int  ygg_collector_start(void);
void ygg_collector_stop(void);
void ygg_collector_set_socket(const char *socket_path);
void ygg_collector_set_output(const char *file_path);
uint64_t ygg_collector_dropped(void);

#ifdef __cplusplus
}
#endif
