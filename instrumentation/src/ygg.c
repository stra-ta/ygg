#include "ygg/ygg.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/futex.h>
#include <sys/syscall.h>

// ============================================================================
// Ring buffer implementation (lock-free, single-producer per thread)
// ============================================================================

#define YGG_RING_SIZE (1 << 16)  // 64K events per thread
#define YGG_RING_MASK (YGG_RING_SIZE - 1)

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

struct ygg_thread_ring {
    _Atomic(uint64_t) head;  // producer index
    _Atomic(uint64_t) tail;  // consumer index
    struct ygg_event events[YGG_RING_SIZE];
    pthread_t thread_id;
    int cpu;
    struct ygg_thread_ring *next;
};

// Global registry of thread rings (for collector to drain)
static _Atomic(struct ygg_thread_ring *) g_thread_rings = NULL;
static pthread_mutex_t g_registry_mutex = PTHREAD_MUTEX_INITIALIZER;

// Shared memory region for collector coordination
struct ygg_shared_state {
    uint64_t tsc_freq_hz;
    uint64_t monotonic_offset_ns;
    uint32_t collector_pid;
    _Atomic(uint32_t) shutdown_flag;
};

static struct ygg_shared_state *g_shared = NULL;
static int g_shm_fd = -1;

// Thread-local ring buffer
static __thread struct ygg_thread_ring *tls_ring = NULL;

// ============================================================================
// TSC calibration
// ============================================================================

static uint64_t calibrate_tsc(void) {
    // Measure TSC frequency by sampling over 100ms
    struct timespec start, end;
    uint64_t tsc_start, tsc_end;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    unsigned int aux;
    tsc_start = __builtin_ia32_rdtscp(&aux);

    usleep(100000);  // 100ms

    tsc_end = __builtin_ia32_rdtscp(&aux);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    uint64_t ns_elapsed = (end.tv_sec - start.tv_sec) * 1000000000ull +
                          (end.tv_nsec - start.tv_nsec);
    uint64_t tsc_elapsed = tsc_end - tsc_start;

    return (tsc_elapsed * 1000000000ull) / ns_elapsed;
}

// ============================================================================
// Ring buffer operations
// ============================================================================

static struct ygg_thread_ring *get_or_create_ring(void) {
    if (tls_ring) return tls_ring;

    struct ygg_thread_ring *ring = mmap(NULL, sizeof(struct ygg_thread_ring),
                                         PROT_READ | PROT_WRITE,
                                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ring == MAP_FAILED) return NULL;

    ring->head = 0;
    ring->tail = 0;
    ring->thread_id = pthread_self();
    ring->cpu = 0;  // Will be set by collector
    ring->next = NULL;

    // Register globally
    pthread_mutex_lock(&g_registry_mutex);
    ring->next = (struct ygg_thread_ring *)atomic_load(&g_thread_rings);
    atomic_store(&g_thread_rings, ring);
    pthread_mutex_unlock(&g_registry_mutex);

    tls_ring = ring;
    return ring;
}

bool ygg_try_emit(uint16_t kind, uint64_t arg0, uint64_t arg1, uint64_t arg2) {
    struct ygg_thread_ring *ring = get_or_create_ring();
    if (!ring) return false;

    uint64_t head = atomic_load_explicit(&ring->head, memory_order_relaxed);
    uint64_t next_head = head + 1;

    // Check if ring is full
    uint64_t tail = atomic_load_explicit(&ring->tail, memory_order_acquire);
    if (next_head - tail >= YGG_RING_SIZE) {
        return false;  // Ring full, drop event
    }

    struct ygg_event *event = &ring->events[head & YGG_RING_MASK];

    // Get timestamp using calibrated TSC
    extern uint64_t ygg_tsc_freq_hz;
    extern uint64_t ygg_monotonic_offset_ns;
    unsigned int aux;
    uint64_t tsc = __builtin_ia32_rdtscp(&aux);
    event->timestamp_ns = (tsc * 1000000000ull) / ygg_tsc_freq_hz + ygg_monotonic_offset_ns;

    event->cpu = ring->cpu;
    event->pid = getpid();
    event->tid = syscall(SYS_gettid);
    event->kind = kind;
    event->padding = 0;
    event->arg0 = arg0;
    event->arg1 = arg1;
    event->arg2 = arg2;

    atomic_store_explicit(&ring->head, next_head, memory_order_release);
    return true;
}

// ============================================================================
// C API implementation
// ============================================================================

struct ygg_event_registry {
    uint16_t next_kind;
    pthread_mutex_t mutex;
};

int ygg_init(const char *process_name) {
    (void)process_name;

    // Calibrate TSC
    uint64_t tsc_freq = calibrate_tsc();

    // Create shared memory for calibration constants
    g_shm_fd = shm_open("/ygg-calibration", O_CREAT | O_RDWR, 0600);
    if (g_shm_fd < 0) return -1;

    if (ftruncate(g_shm_fd, sizeof(struct ygg_shared_state)) < 0) {
        close(g_shm_fd);
        return -1;
    }

    g_shared = mmap(NULL, sizeof(struct ygg_shared_state),
                     PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
    if (g_shared == MAP_FAILED) {
        close(g_shm_fd);
        return -1;
    }

    g_shared->tsc_freq_hz = tsc_freq;

    // Get monotonic offset
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    unsigned int aux;
    uint64_t tsc_now = __builtin_ia32_rdtscp(&aux);
    g_shared->monotonic_offset_ns = ts.tv_sec * 1000000000ull + ts.tv_nsec -
                                     (tsc_now * 1000000000ull) / tsc_freq;
    g_shared->collector_pid = getpid();
    atomic_init(&g_shared->shutdown_flag, 0);

    // Export for inline functions
    extern uint64_t ygg_tsc_freq_hz;
    extern uint64_t ygg_monotonic_offset_ns;
    ygg_tsc_freq_hz = tsc_freq;
    ygg_monotonic_offset_ns = g_shared->monotonic_offset_ns;

    return 0;
}

ygg_event_kind_t ygg_register_event(const char *name) {
    (void)name;  // In this simple impl, we just assign sequential IDs
    static _Atomic(uint16_t) next_kind = 1000;  // Start at AppBase
    return atomic_fetch_add_explicit(&next_kind, 1, memory_order_relaxed);
}

void ygg_emit(ygg_event_kind_t kind, uint64_t arg0, uint64_t arg1, uint64_t arg2) {
    ygg_try_emit(kind, arg0, arg1, arg2);
}

void ygg_shutdown(void) {
    if (g_shared) {
        atomic_store_explicit(&g_shared->shutdown_flag, 1, memory_order_release);
        munmap(g_shared, sizeof(struct ygg_shared_state));
        g_shared = NULL;
    }
    if (g_shm_fd >= 0) {
        close(g_shm_fd);
        shm_unlink("/ygg-calibration");
        g_shm_fd = -1;
    }

    // Clean up thread rings
    pthread_mutex_lock(&g_registry_mutex);
    struct ygg_thread_ring *ring = (struct ygg_thread_ring *)atomic_load(&g_thread_rings);
    while (ring) {
        struct ygg_thread_ring *next = ring->next;
        munmap(ring, sizeof(struct ygg_thread_ring));
        ring = next;
    }
    atomic_store(&g_thread_rings, NULL);
    pthread_mutex_unlock(&g_registry_mutex);
}

// ============================================================================
// C++ ThreadLocalSink implementation
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// These are called from C++ inline functions
bool ygg_thread_local_try_emit(uint16_t kind, uint64_t arg0, uint64_t arg1, uint64_t arg2) {
    return ygg_try_emit(kind, arg0, arg1, arg2);
}

void ygg_thread_local_flush(void) {
    // Flush is implicit in ring buffer - collector reads directly
}

#ifdef __cplusplus
}
#endif