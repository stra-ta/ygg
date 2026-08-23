# Ygg

**Representation learning for systems execution.**

A trace intelligence engine that learns the shape of healthy and pathological execution from low-level system telemetry.

## What Ygg Does

Ygg watches real programs run, turns their execution into structured traces, learns embeddings of system behavior, and lets you ask:

- Is this execution normal?
- Where did behavior diverge?
- What class of failure does this resemble?
- Which executions are behaviorally similar?
- Did a code change create a new execution regime?
- Did Loki discover a failure mode we've never seen before?

The important word is **representation**. Ygg doesn't start with labels like "deadlock" or "disk stall." It learns:

```
execution → vector
```

where executions with similar underlying behavior cluster together.

## Stack Position

```
                     ┌──────────────┐
                     │     Ygg      │
                     │  understands │
                     └──────▲───────┘
                            │
                         traces
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       ┌──┴──┐           ┌──┴──┐          ┌──┴──┐
       │Norn │           │Weir │          │Orda │
       └──▲──┘           └──▲──┘          └──▲──┘
          │                 │                 │
          └────────────┬────┴─────┬──────────┘
                       │          │
                    ┌──┴──┐    ┌──┴───┐
                    │Loki │    │Fenrir│
                    │fault│    │load  │
                    └──┬──┘    └──┬───┘
                       │          │
                       └────┬─────┘
                            │
                         ┌──┴──┐
                         │Kiln │
                         │proof│
                         └─────┘
```

## Quick Start

```bash
# Build the Rust collector (Linux recommended for full eBPF/perf support)
# macOS builds but the eBPF/perf data path is a compile-time no-op.
cargo build --release

# Record a trace (requires Linux + root for eBPF/perf)
sudo ./target/release/ygg-collector --output trace.parquet -- ./weir-server

# Confirm the model training entry point loads
python -m model.train --help

# Confirm the analysis package imports
python -c "import analysis"
```

> **Platform note:** The collector's eBPF (aya) and `perf_event_open` hardware counters are Linux-only, gated behind `cfg(target_os = "linux")`. On macOS the crate compiles but performs no collection. eBPF also requires clang and kernel BTF (`vmlinux.h`); see `DEVELOPMENT.md`.

## Current Status

What is implemented and verified versus what is still scaffolded:

| Component | State | Notes |
|-----------|-------|-------|
| `collector/` | Implemented | Rust + eBPF (aya) + `perf_event_open`. `cargo build --release` works. Linux-only telemetry; macOS build is a no-op. Depends on the `ygg-schema` crate for the `Event` type (the old `schema.rs` was deleted). |
| `instrumentation/` | Implemented | C++20 header (`ygg.h`) + C implementation + Rust FFI wrapper. Thread-local SPSC ring buffers, `/dev/shm/ygg-<pid>` shared memory, 100ms TSC calibration, single collector thread with Unix socket forwarding + spill-file fallback. Builds `libygg_instrumentation.a`/`.so`. 2/2 tests pass. |
| `schema/` | Implemented | `ygg-schema` Rust crate with `Event`, `EventKind`, and Arrow/Parquet schemas. |
| `model/` | Implemented | JAX/Flax: `config.py`, `encoder.py`, `hierarchical.py`, `objectives.py`, `dataset.py`, `train.py`. `python -m model.train --help` works. |
| `analysis/` | Implemented | `divergence.py`, `clustering.py`, `attribution.py`, `viz.py`. `python -c "import analysis"` works. |
| `integrations/` | Scaffolded | Kiln/Loki/Norn/Weir integration points exist but are not yet wired to the finished components above. |
| `experiments/` | V0.1 synthetic evidence | Contention V0.1 synthetic pipeline implemented (generate → masked-only train → UMAP → silhouette 0.30, no policy labels). Real Norn campaign pending. V0.2/V0.3 still scaffolded. |

## Components

| Component | Language | Purpose |
|-----------|----------|---------|
| `collector/` | Rust + eBPF | Kernel/system trace collection |
| `instrumentation/` | C++20 | Application event emission (zero-overhead) |
| `schema/` | Rust (`ygg-schema`) | Event schema & Arrow/Parquet serialization |
| `model/` | JAX + Flax | Representation learning |
| `analysis/` | Python | Clustering, divergence, attribution |
| `integrations/` | Various | Kiln, Loki, Norn, Weir, Orda |

## Data Flow

```
Application (YGG_EVENT)          Kernel (eBPF)
        │                             │
        ▼                             ▼
┌───────────────┐              ┌───────────────┐
│ Thread-local  │              │ Per-CPU ring  │
│ ring buffer   │              │ buffer        │
└───────┬───────┘              └───────┬───────┘
        │                              │
        └──────────────┬───────────────┘
                       ▼
              ┌───────────────┐
              │ Rust collector│
              │ (mmap + poll) │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │ Arrow/Parquet │
              │ (ZSTD comp.)  │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │  JAX model    │
              │ (Transformer) │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │ Embeddings →  │
              │ Analysis CLI  │
              └───────────────┘
```

## Self-Supervised Objectives

1. **Masked Event Modeling** — predict masked events in a sequence
2. **Next-Event Prediction** — predict likely next events; divergence = anomaly
3. **Contrastive Execution Learning** — healthy executions close, failures separate
4. **Temporal Consistency** — adjacent windows evolve smoothly; jumps = phase transitions

## Research Questions

- **RQ1:** Can self-supervised models learn reusable representations of concurrent system execution?
- **RQ2:** Do learned representations separate workload changes from actual failure modes?
- **RQ3:** Can behavioral embeddings identify previously unseen faults?
- **RQ4:** Can the model locate the transition point between healthy and pathological execution?
- **RQ5:** Do models trained on one stra-ta system transfer to another?

## Philosophy

- No LLM log summarization
- No generic OpenTelemetry ingestion
- No cloud monitoring dashboards
- No "AI observability" SaaS nonsense

Ygg is a **research instrument**. Think: `perf` met representation learning and developed opinions about causality.

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — system design, data model, collector/instrumentation architecture, milestones.
- [DEVELOPMENT.md](DEVELOPMENT.md) — build, test, and run instructions.

## Development

```bash
# Build the Rust workspace release artifacts (Linux recommended for eBPF/perf)
cargo build --release

# Run the instrumentation crate's tests
cargo test -p ygg-instrumentation --release

# Confirm the model training entry point loads
python -m model.train --help

# Confirm the analysis package imports
python -c "import analysis"
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for full build, test, and run instructions.

## License

MIT — but the aura is proprietary.