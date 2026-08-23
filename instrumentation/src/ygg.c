/*
 * ygg.c - process lifecycle, TSC calibration, shared-memory setup.
 *
 * ygg_init():
 *   - calibrates the TSC against CLOCK_MONOTONIC_RAW over ~100 ms
 *   - creates the shared header region (/dev/shm/ygg-<pid> via shm_open)
 *   - starts the single collector thread
 *   - publishes calibration constants for the inline C++ hot path
 *
 * ygg_shutdown(): signals the collector, joins it (final drain), then
 * unmaps/unlinks the shared regions.
 */

#include "ygg/ygg.h"
#include "ygg_internal.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <stdatomic.h>

#ifdef __linux__
#include <sched.h>
#include <sys/syscall.h>
#endif

/* Definitions of the calibration globals (satisfy the header externs and the
 * inline timestamp_ns() in ygg.h). */
uint64_t ygg_tsc_freq_hz         = 1000000000ull;
uint64_t ygg_tsc_to_ns_mult      = (1ull << 32);
uint64_t ygg_monotonic_offset_ns = 0;

struct ygg_shm_header *g_ygg_shm = NULL;
static int g_shm_fd = -1;

uint64_t ygg_calibrate_tsc(void) {
#if defined(__x86_64__) || defined(__i386__)
    struct timespec t0, t1;
    uint64_t c0, c1;

    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    c0 = ygg_read_tsc();

    struct timespec req = { 0, 100000000 };   /* 100 ms */
    nanosleep(&req, NULL);

    c1 = ygg_read_tsc();
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    uint64_t ns  = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull
                 + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
    uint64_t dt  = c1 - c0;
    uint64_t freq = (dt * 1000000000ull) / ns;

    ygg_tsc_freq_hz    = freq;
    ygg_tsc_to_ns_mult = (1000000000ull << 32) / freq;

    /* Offset so ts = (c * mult) >> 32 + offset == current monotonic ns. */
    uint64_t now_ns = (uint64_t)t1.tv_sec * 1000000000ull + (uint64_t)t1.tv_nsec;
    uint64_t cnow   = ygg_read_tsc();
    uint64_t est    = (uint64_t)(((unsigned __int128)cnow * ygg_tsc_to_ns_mult) >> 32);
    ygg_monotonic_offset_ns = now_ns - est;
#else
    /* Non-x86 fallback: use the monotonic clock directly. */
    ygg_tsc_freq_hz         = 1000000000ull;
    ygg_tsc_to_ns_mult      = (1ull << 32);
    ygg_monotonic_offset_ns = 0;
#endif
    return ygg_tsc_freq_hz;
}

int ygg_init(const char *process_name) {
    (void)process_name;
    if (g_ygg_shm) return 0;   /* already initialized */

    ygg_calibrate_tsc();

    uint32_t pid = (uint32_t)getpid();
    char name[64];
    snprintf(name, sizeof(name), "ygg-%u", pid);

    g_shm_fd = shm_open(name, O_CREAT | O_RDWR, 0600);
    if (g_shm_fd < 0) return -1;

    size_t sz = sizeof(struct ygg_shm_header);
    if (ftruncate(g_shm_fd, (off_t)sz) < 0) {
        close(g_shm_fd);
        g_shm_fd = -1;
        return -1;
    }

    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
    if (p == MAP_FAILED) {
        close(g_shm_fd);
        g_shm_fd = -1;
        return -1;
    }

    g_ygg_shm = (struct ygg_shm_header *)p;
    memset(g_ygg_shm, 0, sz);
    g_ygg_shm->magic               = YGG_SHM_MAGIC;
    g_ygg_shm->tsc_freq_hz         = ygg_tsc_freq_hz;
    g_ygg_shm->tsc_to_ns_mult      = ygg_tsc_to_ns_mult;
    g_ygg_shm->monotonic_offset_ns = ygg_monotonic_offset_ns;
    g_ygg_shm->pid                 = pid;
    g_ygg_shm->collector_pid       = pid;
    atomic_store_explicit(&g_ygg_shm->shutdown_flag, 0, memory_order_relaxed);
    atomic_store_explicit(&g_ygg_shm->ring_count, 0, memory_order_relaxed);
    g_ygg_shm->max_rings           = YGG_MAX_RINGS;

    if (ygg_collector_start() != 0) {
        /* Not fatal: the library still works, events are just dropped. */
        fprintf(stderr, "ygg: collector thread failed to start\n");
    }
    return 0;
}

static _Atomic uint16_t g_next_kind = 1000;   /* start at BuiltinKind::AppBase */

ygg_event_kind_t ygg_register_event(const char *name) {
    (void)name;   /* names are recorded by the collector/registry out of scope */
    return atomic_fetch_add_explicit(&g_next_kind, 1, memory_order_relaxed);
}

void ygg_emit(ygg_event_kind_t kind, uint64_t arg0, uint64_t arg1, uint64_t arg2) {
    ygg_ring_try_emit(kind, arg0, arg1, arg2);
}

int ygg_try_emit(uint16_t kind, uint64_t arg0, uint64_t arg1, uint64_t arg2) {
    return ygg_ring_try_emit(kind, arg0, arg1, arg2);
}

void ygg_flush(void) {
    /* Rings are drained asynchronously by the collector thread; nothing to do.
     * Provided so callers have an explicit flush point if the design changes. */
}

void ygg_shutdown(void) {
    if (g_ygg_shm) {
        atomic_store_explicit(&g_ygg_shm->shutdown_flag, 1, memory_order_release);
    }
    ygg_collector_stop();   /* joins collector (final drain) */

    if (g_ygg_shm) {
        uint32_t pid = g_ygg_shm->pid;
        char name[64];
        snprintf(name, sizeof(name), "ygg-%u", pid);

        /* Unlink per-thread ring shm objects created during the run. */
        for (uint32_t i = 0; i < g_ygg_shm->max_rings; i++) {
            struct ygg_ring_slot *slot = &g_ygg_shm->slots[i];
            if (atomic_load_explicit(&slot->active, memory_order_acquire)
                && slot->shm_name[0]) {
                shm_unlink(slot->shm_name);
            }
        }

        munmap(g_ygg_shm, sizeof(struct ygg_shm_header));
        g_ygg_shm = NULL;

        if (g_shm_fd >= 0) { close(g_shm_fd); g_shm_fd = -1; }
        shm_unlink(name);
    }
}

/* ---- C++ ABI shims (the C++ API in ygg.h is header-only and calls these) ---- */
#ifdef __cplusplus
extern "C" {
#endif

int ygg_thread_local_try_emit(uint16_t kind, uint64_t a0, uint64_t a1, uint64_t a2) {
    return ygg_ring_try_emit(kind, a0, a1, a2);
}

void ygg_thread_local_flush(void) {
    ygg_flush();
}

#ifdef __cplusplus
}
#endif
