"""
generate_switch_trace.py
=======================

Synthesizes two Ygg execution traces with the canonical 8-column event schema
(timestamp_ns u64, cpu u32, pid u32, tid u32, kind u16, arg0 u64, arg1 u64,
arg2 u64):

  * ``healthy.parquet``   - bounded backoff throughout (the reference behavior).
  * ``switched.parquet``  - bounded backoff for the first 50% of events, then a
                            hard policy switch to tight spin (high CAS retries,
                            long spin loops, frequent scheduler migrations, high
                            cache-miss deltas) for the second 50%.

The switch happens at event index ``N // 2``. The TRUE switch point (event
index + timestamp) is recorded to ``switch_meta.json`` so localization can be
scored against ground truth without ever feeding the label into the model.

Run:
    python experiments/divergence/generate_switch_trace.py

Outputs (under experiments/divergence/):
    healthy.parquet
    switched.parquet
    switch_meta.json
"""

from __future__ import annotations

import json
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- #
# Event kinds (see schema/EVENT_SCHEMA.md)
# --------------------------------------------------------------------------- #
APP_BASE = 1000
KIND_PUSH = APP_BASE + 0          # arg0 = queue_depth
KIND_POP = APP_BASE + 1           # arg0 = queue_depth
KIND_CAS_RETRY = APP_BASE + 2     # arg0 = retry_count
KIND_YIELD = APP_BASE + 3         # arg0 = 0
KIND_SPIN = APP_BASE + 4          # arg0 = spin_iterations
KIND_SCHED_SWITCH = 3000          # arg0 = prev_tid, arg1 = next_tid, arg2 = prev_state
KIND_SCHED_MIGRATE = 3002         # arg0 = tid, arg1 = from_cpu, arg2 = to_cpu
KIND_CACHE_MISSES = 7002          # arg0 = delta cache misses

# Canonical arrow schema (matches schema/EVENT_SCHEMA.md).
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

HERE = os.path.dirname(os.path.abspath(__file__))


def _rng() -> np.random.Generator:
    return np.random.default_rng(0xC0FFEE)


def _sample_event(rng: np.random.Generator, kind: int, phase: str) -> tuple[int, int, int, int, int]:
    """
    Return (kind, arg0, arg1, arg2, tid) for one event given its kind and the
    behavioral phase ('healthy' or 'spinning').
    """
    tid = int(rng.integers(1000, 9999))
    if kind == KIND_PUSH or kind == KIND_POP:
        return kind, int(rng.integers(0, 32)), 0, 0, tid
    if kind == KIND_CAS_RETRY:
        if phase == "spinning":
            retry = int(rng.integers(50, 200))      # tight contention
        else:
            retry = int(rng.integers(1, 6))         # bounded backoff
        return kind, retry, 0, 0, tid
    if kind == KIND_YIELD:
        return kind, 0, 0, 0, tid
    if kind == KIND_SPIN:
        if phase == "spinning":
            iters = int(rng.integers(500, 2000))    # long tight spin
        else:
            iters = int(rng.integers(10, 60))       # brief backoff spin
        return kind, iters, 0, 0, tid
    if kind == KIND_SCHED_SWITCH:
        return kind, int(rng.integers(1000, 9999)), int(rng.integers(1000, 9999)), int(rng.integers(0, 3)), tid
    if kind == KIND_SCHED_MIGRATE:
        return kind, tid, int(rng.integers(0, 15)), int(rng.integers(0, 15)), tid
    if kind == KIND_CACHE_MISSES:
        if phase == "spinning":
            delta = int(rng.integers(1500, 6000))   # high cache-miss pressure
        else:
            delta = int(rng.integers(100, 600))     # moderate
        return kind, delta, 0, 0, tid
    # Fallback (should not happen).
    return kind, 0, 0, 0, tid


def _phase_weights(phase: str) -> dict[int, float]:
    """Per-event kind sampling weights for a behavioral phase."""
    if phase == "spinning":
        return {
            KIND_PUSH: 0.30,
            KIND_POP: 0.30,
            KIND_CAS_RETRY: 0.18,   # elevated
            KIND_YIELD: 0.02,
            KIND_SPIN: 0.12,        # elevated
            KIND_SCHED_SWITCH: 0.05,
            KIND_SCHED_MIGRATE: 0.10,  # frequent migrations
            KIND_CACHE_MISSES: 0.03,   # high cache pressure
        }
    # healthy / bounded-backoff
    return {
        KIND_PUSH: 0.40,
        KIND_POP: 0.40,
        KIND_CAS_RETRY: 0.07,
        KIND_YIELD: 0.03,
        KIND_SPIN: 0.04,
        KIND_SCHED_SWITCH: 0.03,
        KIND_SCHED_MIGRATE: 0.02,   # occasional
        KIND_CACHE_MISSES: 0.01,
    }


def generate_pair(
    n_events: int, switch_at: int, nominal_seed: int, spin_seed: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a (healthy, switched) trace pair.

    The ``healthy`` trace is a *nominal* execution: bounded backoff throughout.
    The ``switched`` trace reuses the exact same nominal generative stream for
    its first ``switch_at`` events (so the reference and the candidate agree
    perfectly up to the switch) and then switches to the 'spinning' phase for
    the remainder. Sharing the nominal stream only guarantees that the baseline
    is clean; the *location* of the switch is never revealed to any consumer of
    the resulting parquet files.

    Returns (healthy_rows [n, 8], switched_rows [n, 8], true_switch_timestamp_ns).
    """
    nrng = np.random.default_rng(nominal_seed)   # nominal (bounded-backoff) stream
    srng = np.random.default_rng(spin_seed)       # spinning stream

    kinds = list(_phase_weights("healthy").keys())
    w_healthy = np.array([_phase_weights("healthy")[k] for k in kinds], dtype=float)
    w_spin = np.array([_phase_weights("spinning")[k] for k in kinds], dtype=float)
    w_healthy /= w_healthy.sum()
    w_spin /= w_spin.sum()

    healthy = np.zeros((n_events, 8), dtype=np.uint64)
    switched = np.zeros((n_events, 8), dtype=np.uint64)
    t_h = 0          # healthy clock
    t_s = 0          # switched clock
    true_switch_ts = -1
    pid = np.uint32(1234)

    for idx in range(n_events):
        # --- healthy side: always nominal, driven by nrng ---
        k_h = int(nrng.choice(len(kinds), p=w_healthy))
        kind_h = int(kinds[k_h])
        ko_h, a0_h, a1_h, a2_h, tid_h = _sample_event(nrng, kind_h, "healthy")
        cpu_h = int(nrng.integers(0, 16))
        t_h += int(nrng.integers(500, 5000))

        healthy[idx, 0] = np.uint64(t_h)
        healthy[idx, 1] = np.uint32(cpu_h)
        healthy[idx, 2] = pid
        healthy[idx, 3] = np.uint32(tid_h)
        healthy[idx, 4] = np.uint16(ko_h)
        healthy[idx, 5] = np.uint64(a0_h)
        healthy[idx, 6] = np.uint64(a1_h)
        healthy[idx, 7] = np.uint64(a2_h)

        # --- switched side ---
        if idx < switch_at:
            # Identical to the healthy event at this index (shared stream), and
            # carry the switched clock forward so the spin phase continues it.
            switched[idx] = healthy[idx]
            t_s = int(healthy[idx, 0])
        else:
            # Continue the switched clock from the first half, then advance.
            t_s += int(srng.integers(500, 5000))
            if idx == switch_at:
                true_switch_ts = t_s
            k_s = int(srng.choice(len(kinds), p=w_spin))
            kind_s = int(kinds[k_s])
            ko_s, a0_s, a1_s, a2_s, tid_s = _sample_event(srng, kind_s, "spinning")
            cpu_s = int(srng.integers(0, 16))
            t_s += int(srng.integers(500, 5000))

            switched[idx, 0] = np.uint64(t_s)
            switched[idx, 1] = np.uint32(cpu_s)
            switched[idx, 2] = pid
            switched[idx, 3] = np.uint32(tid_s)
            switched[idx, 4] = np.uint16(ko_s)
            switched[idx, 5] = np.uint64(a0_s)
            switched[idx, 6] = np.uint64(a1_s)
            switched[idx, 7] = np.uint64(a2_s)

    return healthy, switched, int(true_switch_ts)


def write_parquet(rows: np.ndarray, path: str) -> None:
    table = pa.table({
        "timestamp_ns": pa.array(rows[:, 0], pa.uint64()),
        "cpu": pa.array(rows[:, 1], pa.uint32()),
        "pid": pa.array(rows[:, 2], pa.uint32()),
        "tid": pa.array(rows[:, 3], pa.uint32()),
        "kind": pa.array(rows[:, 4], pa.uint16()),
        "arg0": pa.array(rows[:, 5], pa.uint64()),
        "arg1": pa.array(rows[:, 6], pa.uint64()),
        "arg2": pa.array(rows[:, 7], pa.uint64()),
    }, schema=EVENT_SCHEMA)
    pq.write_table(table, path)


def main() -> dict:
    n_events = 20000
    switch_at = n_events // 2

    healthy_rows, switched_rows, true_ts = generate_pair(
        n_events, switch_at, nominal_seed=1, spin_seed=2
    )

    healthy_path = os.path.join(HERE, "healthy.parquet")
    switched_path = os.path.join(HERE, "switched.parquet")
    meta_path = os.path.join(HERE, "switch_meta.json")

    write_parquet(healthy_rows, healthy_path)
    write_parquet(switched_rows, switched_path)

    meta = {
        "n_events": n_events,
        "true_switch_event_index": switch_at,
        "true_switch_timestamp_ns": true_ts,
        "total_duration_ns": int(switched_rows[-1, 0]),
        "healthy_path": "healthy.parquet",
        "switched_path": "switched.parquet",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {healthy_path} ({len(healthy_rows)} events)")
    print(f"wrote {switched_path} ({len(switched_rows)} events)")
    print(f"true switch at event index {switch_at}, timestamp {true_ts} ns "
          f"({true_ts / 1e9:.4f} s)")
    return meta


if __name__ == "__main__":
    main()
