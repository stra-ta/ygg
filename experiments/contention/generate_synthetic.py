#!/usr/bin/env python3
"""
Synthetic Norn contention trace generator (V0.1 validation harness).

Produces Parquet traces for the 16-cell matrix:

    threads x policy:
        1x1, 2x2, 4x4, 8x8  x  {tight, yield, bounded, exponential}

Each trace is a sequence of events with the schema defined in
``schema/EVENT_SCHEMA.md`` (8 columns, Arrow types
timestamp_ns u64, cpu u32, pid u32, tid u32, kind u16, arg0 u64, arg1 u64, arg2 u64).

The four regimes differ *genuinely* in their event composition and argument
statistics so that a label-free encoder can later learn to separate them:

    tight       high CAS retries (50-200), low yield, high spin, high cache
                misses, frequent scheduler migrations
    yield       low CAS retries, frequent Yield events, low spin, moderate migrations
    bounded     moderate CAS retries with bounded backoff, occasional yield, low migrations
    exponential CAS retries follow exponential backoff (retry count grows), rare
                yield, moderate migrations, high cache misses under oversubscription (8x8)

Thread count also shifts the statistics: more threads -> more scheduler switches,
more migrations, and higher queue-occupancy variance.

Output: experiments/contention/synthetic/<regime>_<threads>x<threads>.parquet

This is a *synthetic* stand-in. The target V0.1 figure uses real Norn traces
collected via the C++ instrumentation (see experiments/contention/README.md).
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Application event base. The schema reserves 1000+ for user-defined semantic
# events; AppBase + 0..4 map to the Norn push/pop/cas/yield/spin vocabulary.
APP_BASE = 1000

# Event kinds (must match schema/EVENT_SCHEMA.md ranges).
PUSH = APP_BASE + 0          # 1000  arg0 = queue_depth
POP = APP_BASE + 1           # 1001  arg0 = queue_depth
CAS_RETRY = APP_BASE + 2      # 1002  arg0 = retry_count
YIELD = APP_BASE + 3          # 1003  arg0 = 0
SPIN = APP_BASE + 4           # 1004  arg0 = spin_iterations
SCHED_SWITCH = 3000           # arg0 = prev_tid, arg1 = next_tid
SCHED_MIGRATE = 3002          # arg0 = tid, arg1 = from_cpu, arg2 = to_cpu
PERF_CACHE_MISSES = 7002      # arg0 = delta

EVENT_SCHEMA = pa.schema([
    ("timestamp_ns", pa.uint64()),
    ("cpu", pa.uint32()),
    ("pid", pa.uint32()),
    ("tid", pa.uint32()),
    ("kind", pa.uint16()),
    ("arg0", pa.uint64()),
    ("arg1", pa.uint64()),
    ("arg2", pa.uint64()),
])

# Per-regime event mixture (probabilities, normalised at runtime).
# Categories: push, pop, cas, yield, spin, cache, migrate, sched.
REGIME_WEIGHTS = {
    "tight": {
        "push": 0.15, "pop": 0.15, "cas": 0.30, "yield": 0.01,
        "spin": 0.18, "cache": 0.06, "migrate": 0.07, "sched": 0.08,
    },
    "yield": {
        "push": 0.18, "pop": 0.18, "cas": 0.05, "yield": 0.32,
        "spin": 0.02, "cache": 0.06, "migrate": 0.08, "sched": 0.11,
    },
    "bounded": {
        "push": 0.18, "pop": 0.18, "cas": 0.20, "yield": 0.12,
        "spin": 0.10, "cache": 0.07, "migrate": 0.04, "sched": 0.11,
    },
    "exponential": {
        "push": 0.18, "pop": 0.18, "cas": 0.26, "yield": 0.03,
        "spin": 0.11, "cache": 0.07, "migrate": 0.05, "sched": 0.12,
    },
}

CATEGORIES = ["push", "pop", "cas", "yield", "spin", "cache", "migrate", "sched"]


def generate_trace(regime: str, threads: int, n_events: int, seed: int) -> pa.Table:
    """Return an Arrow Table of ``n_events`` synthetic contention events."""
    rng = np.random.default_rng(seed)
    n_cpus = max(1, min(threads, 64))

    weights = dict(REGIME_WEIGHTS[regime])
    # More threads -> more scheduler activity and migrations.
    scale = 0.5 + threads / 16.0
    weights["sched"] *= scale
    weights["migrate"] *= scale

    probs = np.array([weights[c] for c in CATEGORIES], dtype=float)
    probs = probs / probs.sum()
    choices = rng.choice(len(CATEGORIES), size=n_events, p=probs)

    timestamps = np.empty(n_events, dtype=np.uint64)
    cpus = np.empty(n_events, dtype=np.uint32)
    pids = np.full(n_events, 1000, dtype=np.uint32)
    tids = np.empty(n_events, dtype=np.uint32)
    kinds = np.empty(n_events, dtype=np.uint16)
    arg0 = np.zeros(n_events, dtype=np.uint64)
    arg1 = np.zeros(n_events, dtype=np.uint64)
    arg2 = np.zeros(n_events, dtype=np.uint64)

    # Exponential-backoff retry counter state (only used by the exponential regime).
    exp_k = 0
    exp_cap = 400
    exp_base = 2

    ts = 0
    for i in range(n_events):
        cat = CATEGORIES[choices[i]]

        # Monotonic inter-event timing (clipped to [1ns, 1us] per event).
        ts += int(rng.integers(1, 1000))
        timestamps[i] = ts
        cpu = int(rng.integers(0, n_cpus))
        cpus[i] = cpu
        tid = int(rng.integers(0, threads)) if threads > 1 else 0
        tids[i] = tid

        if cat == "push":
            kinds[i] = PUSH
            qd = int(np.clip(rng.normal(threads, threads * 0.5 + 1.0), 0, 2 * threads + 5))
            arg0[i] = qd
        elif cat == "pop":
            kinds[i] = POP
            qd = int(np.clip(rng.normal(threads, threads * 0.5 + 1.0), 0, 2 * threads + 5))
            arg0[i] = qd
        elif cat == "cas":
            kinds[i] = CAS_RETRY
            if regime == "tight":
                rc = int(rng.integers(50, 201))
            elif regime == "yield":
                rc = int(rng.integers(1, 6))
            elif regime == "bounded":
                rc = int(rng.integers(5, 31))  # capped -> bounded backoff
            else:  # exponential
                rc = int(min(exp_cap, exp_base * (2 ** exp_k)))
                exp_k += 1
                if rc >= exp_cap:
                    exp_k = 0
            arg0[i] = rc
        elif cat == "yield":
            kinds[i] = YIELD
            arg0[i] = 0
        elif cat == "spin":
            kinds[i] = SPIN
            if regime == "tight":
                sp = int(rng.integers(1000, 5001))
            elif regime == "yield":
                sp = int(rng.integers(0, 51))
            elif regime == "bounded":
                sp = int(rng.integers(100, 801))
            else:
                sp = int(rng.integers(200, 1501))
            arg0[i] = sp
        elif cat == "cache":
            kinds[i] = PERF_CACHE_MISSES
            if regime == "tight":
                cd = int(rng.integers(5000, 20001))
            elif regime == "exponential":
                cd = int(rng.integers(4000, 18001))
                if threads >= 16:  # oversubscription -> higher misses
                    cd = int(rng.integers(8000, 30001))
            else:
                cd = int(rng.integers(1000, 5001))
            arg0[i] = cd
        elif cat == "migrate":
            kinds[i] = SCHED_MIGRATE
            arg0[i] = tid
            arg1[i] = cpu
            arg2[i] = int(rng.integers(0, n_cpus))
        else:  # sched
            kinds[i] = SCHED_SWITCH
            arg0[i] = tid
            arg1[i] = int(rng.integers(0, threads))

    return pa.table(
        {
            "timestamp_ns": timestamps,
            "cpu": cpus,
            "pid": pids,
            "tid": tids,
            "kind": kinds,
            "arg0": arg0,
            "arg1": arg1,
            "arg2": arg2,
        },
        schema=EVENT_SCHEMA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Norn contention traces")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "synthetic"),
        help="Output directory for the synthetic parquet traces",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=1024,
        help="Base number of events per trace (scaled by thread count)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regimes = ["tight", "yield", "bounded", "exponential"]
    grids = [1, 2, 4, 8]  # matrix dimension; threads = grid * grid

    written = 0
    for regime in regimes:
        for grid in grids:
            threads = grid * grid
            n_events = args.events + threads * 32
            seed = args.seed + hash((regime, grid)) % (2 ** 31)
            table = generate_trace(regime, threads, n_events, seed)
            path = out_dir / f"{regime}_{threads}x{threads}.parquet"
            pq.write_table(table, path)
            written += 1
            print(
                f"[generate] {path.name}: {table.num_rows} events, "
                f"threads={threads}, regime={regime}"
            )

    print(f"[generate] wrote {written} synthetic traces to {out_dir}")


if __name__ == "__main__":
    main()
