/*
 * bench.c - Ygg instrumentation hot-path benchmark.
 *
 * The public header claims a "~5-10 cycles" emit hot path. That is an empirical
 * claim, so this file measures it instead of asserting it. It runs several
 * scenarios, each one printing a single JSON object on stdout (one per line)
 * so a runner (run.rs) can build, execute, parse, and aggregate the results.
 *
 * Scenarios (selected by argv[1]):
 *   baseline      - empty loop overhead, no YGG_EVENT at all
 *   no_collector  - YGG_EVENT with the collector stopped (ring fills -> drops)
 *   draining      - YGG_EVENT with an active collector draining to a spill file
 *   at_capacity   - YGG_EVENT burst faster than the collector drains (drops)
 *   threads       - N threads each emitting (runs 4 and 8 by default)
 *
 * Timing is portable:
 *   x86_64   -> rdtscp (raw TSC ticks, reported in "cycles")
 *   macos/arm64 -> mach_absolute_time (converted to nanoseconds)
 *   generic  -> CLOCK_MONOTONIC nanoseconds
 *
 * Build (the runner does this for you):
 *   cc bench.c -O2 -I../include -I../src \
 *      -L<libdir> -lygg_instrumentation -lpthread -o bench
 *
 * This file only links the existing instrumentation library; it does not
 * modify it.
 */

#include "ygg/ygg.h"
/* ygg_internal.h exposes ygg_collector_stop(), which we call to model the
 * "collector NOT draining" scenario. It is an existing symbol in the static
 * library; we only link against it, never modify the library. */
#include "ygg_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

/* --------------------------------------------------------------------------
 * Portable high-resolution clock
 * -------------------------------------------------------------------------- */

#if defined(__x86_64__) || defined(__i386__)
#  include <x86intrin.h>
   static inline uint64_t bench_clock(void) {
       unsigned int aux = 0;
       return (uint64_t)__rdtscp(&aux);
   }
   /* Samples are raw TSC ticks; the reporting unit is "cycles" (no convert). */
#  define BENCH_UNIT      "cycles"
#  define BENCH_PLATFORM  "x86_64"
   static inline double bench_to_unit(uint64_t sample) { return (double)sample; }

#elif defined(__APPLE__) && defined(__aarch64__)
#  include <mach/mach.h>
#  include <mach/mach_time.h>
   static mach_timebase_info_data_t g_tb;
   static inline uint64_t bench_clock(void) { return mach_absolute_time(); }
   /* mach ticks -> nanoseconds via the timebase. */
   static inline double bench_to_unit(uint64_t sample) {
       return (double)(sample * (uint64_t)g_tb.numer / (uint64_t)g_tb.denom);
   }
#  define BENCH_UNIT      "ns"
#  define BENCH_PLATFORM  "macos-arm64"

#else
   static inline uint64_t bench_clock(void) {
       struct timespec ts;
       clock_gettime(CLOCK_MONOTONIC, &ts);
       return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
   }
   static inline double bench_to_unit(uint64_t sample) { return (double)sample; }
#  define BENCH_UNIT      "ns"
#  define BENCH_PLATFORM  "generic"
#endif

/* --------------------------------------------------------------------------
 * Defaults
 * -------------------------------------------------------------------------- */

#define DEFAULT_EVENTS 1000000u
#define BATCH          1000u    /* events per measured batch (amortizes the clock read) */
#define WARMUP         2000u    /* untimed events to allocate the ring and reach steady state */
/*
 * The per-thread ring holds YGG_RING_SIZE (65536) events. To measure the pure
 * hot path WITH an active collector and guarantee zero drops, the draining and
 * multithreaded scenarios emit a single burst that stays under that capacity:
 * the ring can never fill within the burst no matter how slow the collector is,
 * so every event exercises the non-drop fast path. (YGG_RING_SIZE is internal;
 * 60000 leaves margin.) The overload / drop behavior is covered separately by
 * the no_collector and ring_at_capacity scenarios.
 */
#define DRAIN_TOTAL    60000u   /* under-capacity burst for the "active collector" hot path */
#define DRAIN_BATCH    1000u    /* measurement batch size within the under-cap burst */
#define THREAD_EVENTS  DRAIN_TOTAL  /* per-thread under-cap burst */

/* --------------------------------------------------------------------------
 * Sample buffers + percentiles
 * -------------------------------------------------------------------------- */

typedef struct {
    double   *samples;
    size_t    cap;
    size_t    count;
    uint64_t  dropped;
    uint64_t  accepted;
} sample_buf;

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/* arr must be sorted ascending; q in [0,1]. Linear interpolation between ranks. */
static double percentile(const double *arr, size_t n, double q) {
    if (n == 0) return 0.0;
    double idx = q * (double)(n - 1);
    size_t lo = (size_t)idx;
    size_t hi = (lo + 1 < n) ? lo + 1 : lo;
    double frac = idx - (double)lo;
    return arr[lo] + (arr[hi] - arr[lo]) * frac;
}

/* --------------------------------------------------------------------------
 * Emission loop (the timed hot path)
 * -------------------------------------------------------------------------- */

/*
 * Emit `total` events in batches of `batch`. For each batch we read the clock
 * around the whole batch (amortizing the read over `batch` events) and record
 * cycles/event (or ns/event). `pace_us`, if non-zero, is slept BETWEEN batches
 * (outside the measured region) so an active collector can drain and the ring
 * never fills. try_emit's return value is used to count dropped events.
 */
static void run_emission(sample_buf *sb, uint64_t total, uint64_t batch,
                         uint64_t warmup, uint64_t pace_us, uint16_t kind) {
    uint64_t dropped = 0, accepted = 0;

    /* Untimed warmup: allocate the per-thread ring and reach steady state. */
    for (uint64_t i = 0; i < warmup; i++) {
        if (ygg_try_emit(kind, i, 0, 0)) accepted++; else dropped++;
    }

    uint64_t done = 0;
    while (done < total) {
        uint64_t n = (total - done < batch) ? (total - done) : batch;

        uint64_t t0 = bench_clock();
        for (uint64_t i = 0; i < n; i++) {
            if (ygg_try_emit(kind, i, 0, 0)) accepted++; else dropped++;
        }
        uint64_t t1 = bench_clock();

        double per = bench_to_unit(t1 - t0) / (double)n;
        if (sb->count < sb->cap) sb->samples[sb->count++] = per;

        done += n;
        if (pace_us) usleep((useconds_t)pace_us);
    }

    sb->dropped = dropped;
    sb->accepted = accepted;
}

/* Empty-loop floor: same control flow, no YGG_EVENT. */
static void run_baseline(sample_buf *sb, uint64_t total, uint64_t batch,
                         uint64_t warmup) {
    for (uint64_t i = 0; i < warmup; i++) { volatile uint64_t s = i; (void)s; }

    uint64_t done = 0;
    while (done < total) {
        uint64_t n = (total - done < batch) ? (total - done) : batch;

        uint64_t t0 = bench_clock();
        volatile uint64_t sink = 0;
        for (uint64_t i = 0; i < n; i++) sink += i;   /* prevent loop elimination */
        uint64_t t1 = bench_clock();
        (void)sink;

        double per = bench_to_unit(t1 - t0) / (double)n;
        if (sb->count < sb->cap) sb->samples[sb->count++] = per;

        done += n;
    }
    sb->dropped = 0;
    sb->accepted = total;
}

/* --------------------------------------------------------------------------
 * JSON output
 * -------------------------------------------------------------------------- */

static void print_json(const char *scenario, uint64_t events, int threads,
                       sample_buf *sb, int collector_active,
                       const double *per_thread_median, int per_thread_n) {
    qsort(sb->samples, sb->count, sizeof(double), cmp_double);
    double median = percentile(sb->samples, sb->count, 0.50);
    double p95    = percentile(sb->samples, sb->count, 0.95);
    double p99    = percentile(sb->samples, sb->count, 0.99);

    printf("{");
    printf("\"scenario\":\"%s\",", scenario);
    printf("\"events\":%llu,", (unsigned long long)events);
    printf("\"threads\":%d,", threads);
    printf("\"median_cycles_per_event\":%.4f,", median);
    printf("\"p95_cycles_per_event\":%.4f,", p95);
    printf("\"p99_cycles_per_event\":%.4f,", p99);
    printf("\"dropped_events\":%llu,", (unsigned long long)sb->dropped);
    printf("\"collector_active\":%s,", collector_active ? "true" : "false");
    printf("\"platform\":\"%s\",", BENCH_PLATFORM);
    printf("\"unit\":\"%s\",", BENCH_UNIT);
    printf("\"samples\":%zu", sb->count);
    if (per_thread_median && per_thread_n > 0) {
        printf(",\"per_thread_cycles_per_event\":[");
        for (int i = 0; i < per_thread_n; i++) {
            printf("%s%.4f", i ? "," : "", per_thread_median[i]);
        }
        printf("]");
    }
    printf("}\n");
    fflush(stdout);
}

/* --------------------------------------------------------------------------
 * Scenario: baseline
 * -------------------------------------------------------------------------- */

static void scenario_baseline(uint64_t events) {
    sample_buf sb = { malloc(sizeof(double) * (events / BATCH + 16)),
                      (size_t)(events / BATCH + 16), 0, 0, 0 };
    run_baseline(&sb, events, BATCH, WARMUP);
    print_json("baseline_disabled", events, 1, &sb, 0, NULL, 0);
    free(sb.samples);
}

/* --------------------------------------------------------------------------
 * Scenario: YGG_EVENT with the collector stopped (ring fills -> drops)
 * -------------------------------------------------------------------------- */

static void scenario_no_collector(uint64_t events) {
    ygg_init("ygg-bench-no-collector");
    ygg_collector_stop();   /* drainer is gone; the ring fills and try_emit drops */
    uint16_t kind = (uint16_t)ygg_register_event("bench_event");

    sample_buf sb = { malloc(sizeof(double) * (events / BATCH + 16)),
                      (size_t)(events / BATCH + 16), 0, 0, 0 };
    run_emission(&sb, events, BATCH, WARMUP, 0, kind);
    print_json("ygg_event_no_collector", events, 1, &sb, 0, NULL, 0);

    free(sb.samples);
    ygg_shutdown();
}

/* --------------------------------------------------------------------------
 * Scenario: YGG_EVENT with an active collector draining to a spill file
 * -------------------------------------------------------------------------- */

static void scenario_draining(void) {
    char spill[256];
    snprintf(spill, sizeof(spill), "/tmp/ygg-bench-drain-%u.events", (unsigned)getpid());
    ygg_collector_set_output(spill);

    ygg_init("ygg-bench-draining");
    uint16_t kind = (uint16_t)ygg_register_event("bench_event");

    sample_buf sb = { malloc(sizeof(double) * (DRAIN_TOTAL / DRAIN_BATCH + 16)),
                      (size_t)(DRAIN_TOTAL / DRAIN_BATCH + 16), 0, 0, 0 };
    /* Single burst under the ring capacity: the ring can never fill, so every
     * event takes the non-drop hot path while the collector is active. */
    run_emission(&sb, DRAIN_TOTAL, DRAIN_BATCH, WARMUP, 0, kind);
    print_json("ygg_event_draining", DRAIN_TOTAL, 1, &sb, 1, NULL, 0);

    free(sb.samples);
    ygg_shutdown();
    unlink(spill);
}

/* --------------------------------------------------------------------------
 * Scenario: YGG_EVENT burst faster than the collector drains (drops)
 * -------------------------------------------------------------------------- */

static void scenario_at_capacity(uint64_t events) {
    ygg_init("ygg-bench-at-capacity");
    uint16_t kind = (uint16_t)ygg_register_event("bench_event");

    sample_buf sb = { malloc(sizeof(double) * (events / BATCH + 16)),
                      (size_t)(events / BATCH + 16), 0, 0, 0 };
    /* No pacing: a tight burst that overflows the ring before the collector drains. */
    run_emission(&sb, events, BATCH, WARMUP, 0, kind);
    print_json("ring_at_capacity", events, 1, &sb, 1, NULL, 0);

    free(sb.samples);
    ygg_shutdown();
}

/* --------------------------------------------------------------------------
 * Scenario: multiple threads, each emitting on its own per-thread ring
 * -------------------------------------------------------------------------- */

typedef struct {
    uint64_t total;
    uint16_t kind;
    double  *samples;
    size_t   cap;
    size_t   count;
    uint64_t dropped;
} thread_ctx;

static void *thread_main(void *arg) {
    thread_ctx *c = (thread_ctx *)arg;
    sample_buf sb = { c->samples, c->cap, 0, 0, 0 };
    run_emission(&sb, c->total, DRAIN_BATCH, WARMUP, 0, c->kind);
    c->count   = sb.count;
    c->dropped = sb.dropped;
    return NULL;
}

static void scenario_threads(int nthreads) {
    ygg_init("ygg-bench-threads");
    uint16_t kind = (uint16_t)ygg_register_event("bench_event");

    pthread_t  *ths  = calloc((size_t)nthreads, sizeof(pthread_t));
    thread_ctx *ctxs = calloc((size_t)nthreads, sizeof(thread_ctx));
    double     *combined = malloc(sizeof(double) *
                                  ((size_t)nthreads * (DRAIN_TOTAL / DRAIN_BATCH + 16)));
    double     *per_thread = malloc(sizeof(double) * (size_t)nthreads);

    size_t combined_cap = (size_t)nthreads * (size_t)(DRAIN_TOTAL / DRAIN_BATCH + 16);
    sample_buf comb = { combined, combined_cap, 0, 0, 0 };
    uint64_t dropped_total = 0;

    for (int t = 0; t < nthreads; t++) {
        ctxs[t].total   = THREAD_EVENTS;
        ctxs[t].kind    = kind;
        ctxs[t].cap     = (size_t)(DRAIN_TOTAL / DRAIN_BATCH + 16);
        ctxs[t].samples = malloc(sizeof(double) * ctxs[t].cap);
        ctxs[t].count   = 0;
        ctxs[t].dropped = 0;
        pthread_create(&ths[t], NULL, thread_main, &ctxs[t]);
    }

    for (int t = 0; t < nthreads; t++) {
        pthread_join(ths[t], NULL);
        /* Per-thread median. */
        qsort(ctxs[t].samples, ctxs[t].count, sizeof(double), cmp_double);
        per_thread[t] = percentile(ctxs[t].samples, ctxs[t].count, 0.50);
        for (size_t i = 0; i < ctxs[t].count; i++) {
            if (comb.count < comb.cap) comb.samples[comb.count++] = ctxs[t].samples[i];
        }
        dropped_total += ctxs[t].dropped;
        free(ctxs[t].samples);
    }

    uint64_t total_events = (uint64_t)nthreads * THREAD_EVENTS;
    comb.dropped = dropped_total;
    print_json("ygg_event_threads", total_events, nthreads, &comb, 1,
               per_thread, nthreads);

    free(combined);
    free(per_thread);
    free(ctxs);
    free(ths);
    ygg_shutdown();
}

/* --------------------------------------------------------------------------
 * main
 * -------------------------------------------------------------------------- */

int main(int argc, char **argv) {
#if defined(__APPLE__) && defined(__aarch64__)
    mach_timebase_info(&g_tb);
#endif

    const char *scenario = (argc > 1) ? argv[1] : "baseline";
    uint64_t events = (argc > 2) ? strtoull(argv[2], NULL, 10) : DEFAULT_EVENTS;

    if (strcmp(scenario, "baseline") == 0) {
        scenario_baseline(events);
    } else if (strcmp(scenario, "no_collector") == 0) {
        scenario_no_collector(events);
    } else if (strcmp(scenario, "draining") == 0) {
        scenario_draining();
    } else if (strcmp(scenario, "at_capacity") == 0) {
        scenario_at_capacity(events);
    } else if (strcmp(scenario, "threads") == 0) {
        /* Run 4 then 8 threads (override with argv[2]=<N> to run a single width). */
        if (argc > 2) {
            int n = (int)strtol(argv[2], NULL, 10);
            if (n > 0) scenario_threads(n);
        } else {
            scenario_threads(4);
            scenario_threads(8);
        }
    } else {
        fprintf(stderr,
                "unknown scenario '%s' (want: baseline|no_collector|draining|"
                "at_capacity|threads)\n", scenario);
        return 2;
    }
    return 0;
}
