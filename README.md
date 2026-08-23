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
# Build collector
cd collector && cargo build --release

# Record a trace
./target/release/ygg-collector --output trace.parquet -- ./weir-server

# Inspect (Python)
python -m ygg.inspect trace.parquet

# Compare executions
python -m ygg.diff baseline.parquet experiment.parquet
```

## Components

| Component | Language | Purpose |
|-----------|----------|---------|
| `collector/` | Rust + eBPF | Kernel/system trace collection |
| `instrumentation/` | C++20 | Application event emission (zero-overhead) |
| `schema/` | FlatBuffers + Arrow | Event schema & Parquet serialization |
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

## License

MIT — but the aura is proprietary.