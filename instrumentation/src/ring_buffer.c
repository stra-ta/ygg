/*
 * ring_buffer.c - lock-free SPSC per-thread ring buffer.
 *
 * Hot path: ygg_ring_try_emit() performs a thread-local lookup, an atomic
 * acquire-load of the collector's tail, a fixed-point TSC->ns conversion, an
 * event write, and a release-store of head. No locks, no allocation.
 *
 * Ring data lives in a POSIX shared memory object so the collector (which may
 * run in a separate process after a fork) can map it by name.
 */

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

#ifdef __linux__
#include <sched.h>
#endif

/* ---- thread-local ring state (not shared) ---- */
static __thread int        tls_slot      = -1;
static __thread struct ygg_event *tls_ring_base = NULL;

uint32_t ygg_gettid(void) {
#ifdef __linux__
    return (uint32_t)syscall(SYS_gettid);
#elif defined(__APPLE__)
    uint64_t tid = 0;
    pthread_threadid_np(NULL, &tid);
    return (uint32_t)tid;
#else
    return (uint32_t)getpid();
#endif
}

uint64_t ygg_read_tsc(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int aux = 0;
    return (uint64_t)__builtin_ia32_rdtscp(&aux);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

struct ygg_event *ygg_map_ring_data(const char *shm_name) {
    int fd = shm_open(shm_name, O_RDWR, 0);
    if (fd < 0) return NULL;
    size_t sz = sizeof(struct ygg_event) * YGG_RING_SIZE;
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    return (struct ygg_event *)p;
}

/* Claim a slot in the shared header and create the per-thread ring shm. */
static int ensure_ring(void) {
    if (tls_slot >= 0) return tls_slot;
    if (!g_ygg_shm) return -1;

    uint32_t idx = atomic_fetch_add_explicit(&g_ygg_shm->ring_count, 1,
                                             memory_order_relaxed);
    if (idx >= g_ygg_shm->max_rings) {
        /* Roll back the claim so we never exceed the array. */
        atomic_fetch_sub_explicit(&g_ygg_shm->ring_count, 1, memory_order_relaxed);
        return -1;
    }

    uint32_t tid = ygg_gettid();
    char name[YGG_SHM_NAME_MAX];
    snprintf(name, sizeof(name), "ygg-%u-ring-%u", g_ygg_shm->pid, tid);

    int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        atomic_fetch_sub_explicit(&g_ygg_shm->ring_count, 1, memory_order_relaxed);
        return -1;
    }

    size_t sz = sizeof(struct ygg_event) * YGG_RING_SIZE;
    if (ftruncate(fd, (off_t)sz) < 0) {
        close(fd);
        shm_unlink(name);
        atomic_fetch_sub_explicit(&g_ygg_shm->ring_count, 1, memory_order_relaxed);
        return -1;
    }
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        shm_unlink(name);
        atomic_fetch_sub_explicit(&g_ygg_shm->ring_count, 1, memory_order_relaxed);
        return -1;
    }

    struct ygg_ring_slot *slot = &g_ygg_shm->slots[idx];
    atomic_store_explicit(&slot->head, 0, memory_order_relaxed);
    atomic_store_explicit(&slot->tail, 0, memory_order_relaxed);
    slot->tid   = tid;
#ifdef __linux__
    slot->cpu   = sched_getcpu();
#else
    slot->cpu   = 0;
#endif
    atomic_store_explicit(&slot->active, 1, memory_order_release);
    memset(slot->shm_name, 0, YGG_SHM_NAME_MAX);
    strncpy(slot->shm_name, name, YGG_SHM_NAME_MAX - 1);
    atomic_thread_fence(memory_order_release);

    tls_ring_base = (struct ygg_event *)p;
    tls_slot = (int)idx;
    return idx;
}

int ygg_ring_try_emit(uint16_t kind, uint64_t a0, uint64_t a1, uint64_t a2) {
    int idx = ensure_ring();
    if (idx < 0) return 0;

    struct ygg_ring_slot *slot = &g_ygg_shm->slots[idx];

    uint64_t head = atomic_load_explicit(&slot->head, memory_order_relaxed);
    uint64_t tail = atomic_load_explicit(&slot->tail, memory_order_acquire);
    if (head - tail >= YGG_RING_SIZE) {
        return 0;   /* ring full: drop event (backpressure) */
    }

    uint64_t tsc = ygg_read_tsc();
    uint64_t ts_ns = (uint64_t)(((unsigned __int128)tsc * ygg_tsc_to_ns_mult) >> 32);
    ts_ns += ygg_monotonic_offset_ns;

    struct ygg_event *ev = &tls_ring_base[head & YGG_RING_MASK];
    ev->timestamp_ns = ts_ns;
    ev->cpu   = (uint32_t)slot->cpu;
    ev->pid   = g_ygg_shm->pid;
    ev->tid   = slot->tid;
    ev->kind  = kind;
    ev->padding = 0;
    ev->arg0  = a0;
    ev->arg1  = a1;
    ev->arg2  = a2;

    atomic_store_explicit(&slot->head, head + 1, memory_order_release);
    return 1;
}
