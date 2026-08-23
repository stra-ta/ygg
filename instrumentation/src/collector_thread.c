/*
 * collector_thread.c - single background collector.
 *
 * One pthread per process. It is the sole consumer of every thread ring:
 *   - scans the shared header's ring-slot registry
 *   - maps each per-thread ring-data shm by name
 *   - drains events (head -> tail) and forwards them
 *   - writes a 32-byte stream header followed by 48-byte event records
 *
 * Forwarding targets (in priority order):
 *   1. A Unix socket (to the Rust collector / ygg-collector), if configured.
 *   2. A raw spill file of 48-byte records, if configured.
 *
 * If no sink is reachable, events are still consumed (tail advanced) and counted
 * as dropped, so producers never stall waiting for a collector that is down.
 */

#include "ygg_internal.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>

/* ---- configuration (set via ygg_collector_set_*) ---- */
static char   g_socket_path[256];
static char   g_file_path[256];
static int    g_have_socket = 0;
static int    g_have_file   = 0;

/* ---- runtime state ---- */
static pthread_t g_collector_thread;
static int       g_collector_running = 0;
static int       g_collector_stop    = 0;

static int       g_sock_fd = -1;
static int       g_file_fd  = -1;
static int       g_stream_hdr_sent = 0;

/* Cached ring-data mappings, indexed by slot index. */
static struct ygg_event *g_ring_maps[YGG_MAX_RINGS];
static int               g_ring_mapped[YGG_MAX_RINGS];

static uint64_t g_dropped = 0;

void ygg_collector_set_socket(const char *socket_path) {
    if (socket_path && *socket_path) {
        strncpy(g_socket_path, socket_path, sizeof(g_socket_path) - 1);
        g_socket_path[sizeof(g_socket_path) - 1] = 0;
        g_have_socket = 1;
    } else {
        g_have_socket = 0;
    }
}

void ygg_collector_set_output(const char *file_path) {
    if (file_path && *file_path) {
        strncpy(g_file_path, file_path, sizeof(g_file_path) - 1);
        g_file_path[sizeof(g_file_path) - 1] = 0;
        g_have_file = 1;
    } else {
        g_have_file = 0;
    }
}

uint64_t ygg_collector_dropped(void) {
    return g_dropped;
}

/* ---- helpers ---- */

static void ygg_sleep_us(long us) {
    struct timespec req;
    req.tv_sec  = us / 1000000;
    req.tv_nsec = (us % 1000000) * 1000;
    nanosleep(&req, NULL);
}

static int write_all(int fd, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (w == 0) return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static int try_connect(void) {
    if (!g_have_socket) return -1;

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, g_socket_path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    g_sock_fd = fd;
    g_stream_hdr_sent = 0;   /* new connection: re-send the stream header */
    return 0;
}

struct ygg_stream_header {
    uint64_t magic;
    uint64_t tsc_freq_hz;
    uint64_t tsc_to_ns_mult;
    uint64_t monotonic_offset_ns;
};

static int send_stream_header(void) {
    if (g_stream_hdr_sent) return 0;
    struct ygg_stream_header h;
    h.magic                = YGG_SHM_MAGIC;
    h.tsc_freq_hz          = ygg_tsc_freq_hz;
    h.tsc_to_ns_mult       = ygg_tsc_to_ns_mult;
    h.monotonic_offset_ns  = ygg_monotonic_offset_ns;

    if (g_sock_fd >= 0) {
        if (write_all(g_sock_fd, &h, sizeof(h)) < 0) { close(g_sock_fd); g_sock_fd = -1; return -1; }
    }
    if (g_file_fd >= 0) {
        if (write_all(g_file_fd, &h, sizeof(h)) < 0) { close(g_file_fd); g_file_fd = -1; return -1; }
    }
    g_stream_hdr_sent = 1;
    return 0;
}

/* Forward one event. Returns 0 on success, -1 if it could not be written. */
static int write_event(const struct ygg_event *ev) {
    if (g_sock_fd >= 0) {
        if (send_stream_header() < 0) return -1;
        return write_all(g_sock_fd, ev, sizeof(*ev));
    }
    if (g_file_fd >= 0) {
        if (send_stream_header() < 0) return -1;
        return write_all(g_file_fd, ev, sizeof(*ev));
    }
    return -1;   /* no sink configured: caller counts a drop */
}

/* Drain every registered ring once. Always advances tails (freeing producer
 * space); only counts a drop when an event could not be forwarded. */
static void drain_all(void) {
    if (!g_ygg_shm) return;

    for (uint32_t i = 0; i < g_ygg_shm->max_rings; i++) {
        struct ygg_ring_slot *slot = &g_ygg_shm->slots[i];
        if (!atomic_load_explicit(&slot->active, memory_order_acquire)) continue;

        if (!g_ring_mapped[i]) {
            g_ring_maps[i]  = ygg_map_ring_data(slot->shm_name);
            g_ring_mapped[i] = (g_ring_maps[i] != NULL) ? 1 : 0;
        }
        struct ygg_event *base = g_ring_maps[i];
        if (!base) continue;

        uint64_t tail = atomic_load_explicit(&slot->tail, memory_order_relaxed);
        uint64_t head = atomic_load_explicit(&slot->head, memory_order_acquire);

        while (tail != head) {
            struct ygg_event *ev = &base[tail & YGG_RING_MASK];
            if (write_event(ev) < 0) g_dropped++;
            tail++;
        }
        atomic_store_explicit(&slot->tail, tail, memory_order_release);
    }
}

static void *collector_main(void *arg) {
    (void)arg;

    /* Open the raw spill file up front if configured. */
    if (g_have_file && g_file_fd < 0) {
        g_file_fd = open(g_file_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    }

    for (;;) {
        int shutdown = g_ygg_shm
            ? atomic_load_explicit(&g_ygg_shm->shutdown_flag, memory_order_acquire)
            : 1;

        if (g_sock_fd < 0 && g_have_socket) {
            try_connect();   /* lazy (re)connect to the Rust collector */
        }

        drain_all();

        if (g_collector_stop || shutdown) break;
        ygg_sleep_us(2000);   /* ~2 ms poll interval */
    }

    /* Final drain so in-flight events are captured before we exit. */
    drain_all();
    drain_all();

    if (g_sock_fd >= 0) { close(g_sock_fd); g_sock_fd = -1; }
    if (g_file_fd  >= 0) { close(g_file_fd);  g_file_fd  = -1; }
    return NULL;
}

int ygg_collector_start(void) {
    if (g_collector_running) return 0;
    g_collector_stop = 0;

    /* Sensible default so events are captured even without an external
     * collector: spill to /tmp/ygg-<pid>.events when nothing is configured. */
    if (!g_have_socket && !g_have_file) {
        snprintf(g_file_path, sizeof(g_file_path), "/tmp/ygg-%u.events",
                 (unsigned)(getpid()));
        g_have_file = 1;
    }

    if (pthread_create(&g_collector_thread, NULL, collector_main, NULL) != 0) {
        return -1;
    }
    g_collector_running = 1;
    return 0;
}

void ygg_collector_stop(void) {
    if (!g_collector_running) return;
    g_collector_stop = 1;
    pthread_join(g_collector_thread, NULL);
    g_collector_running = 0;
}
