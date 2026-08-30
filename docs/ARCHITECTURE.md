# Ygg Architecture

> **Ygg learns representations of concurrent system execution from application, kernel, scheduler, and hardware traces.**

## Overview

Ygg is a trace intelligence engine that learns embeddings of system behavior from low-level telemetry. It sits at the center of the stra-ta stack, consuming traces from Norn, Weir, Orda, Loki, and Kiln to answer:

- Is this execution normal?
- Where did behavior diverge?
- What class of failure does this resemble?
- Which executions are behaviorally similar?
- Did a code change create a new execution regime?

## System Context

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

## Data Model

### Event Schema

Fixed 48-byte event structure (defined in `schema/event.fbs`):

```cpp
struct Event {
    uint64_t timestamp_ns;  // CLOCK_MONOTONIC_RAW, calibrated from TSC
    uint32_t cpu;           // CPU core ID
    uint32_t pid;           // Process ID
    uint32_t tid;           // Thread ID
    uint16_t kind;          // EventKind enum
    uint16_t padding;       // Alignment
    uint64_t arg0;          // Kind-dependent payload
    uint64_t arg1;
    uint64_t arg2;
};
```

### Event Kinds

| Range | Category | Examples |
|-------|----------|----------|
| 1000+ | Application | User-defined semantic events |
| 2000-2001 | Syscall | `SysEnter`, `SysExit` |
| 3000-3002 | Scheduler | `SchedSwitch`, `SchedWakeup`, `SchedMigrate` |
| 4000-4001 | Block I/O | `BlockRqIssue`, `BlockRqComplete` |
| 5000-5001 | Network | `TcpSendmsg`, `TcpRecvmsg` |
| 6000-6001 | Memory | `PageFault`, `PageFaultMajor` |
| 7000-7004 | Hardware Counters | `PerfCycles`, `PerfInstructions`, `PerfCacheMisses`, `PerfBranchMisses`, `PerfContextSwitches` |
| 8000 | Loki Injection | `LokiInject` |
| 9000 | Custom | Dynamic registration |

### Execution Metadata

```cpp
struct ExecutionMetadata {
    string git_sha;
    string machine_fingerprint;
    string kernel_version;
    string compiler;
    string workload;
    string loki_fault_plan;
};
```

### Storage Format

**Apache Arrow / Parquet** — columnar, compressed, Arrow columnar interoperability via Arrow/Parquet (zero-copy is an Arrow property, not a Ygg guarantee).

File layout:
- Row group 1: metadata (single row)
- Row groups 2+: events (partitioned by time windows)

## Collector

### Linux Telemetry (eBPF)

The collector uses eBPF probes attached to kernel tracepoints and kprobes:

```
sched_switch      → SchedSwitch
sched_wakeup      → SchedWakeup
sched_migrate_task → SchedMigrate
sys_enter         → SysEnter
sys_exit          → SysExit
block_rq_issue    → BlockRqIssue
block_rq_complete → BlockRqComplete
tcp_sendmsg       → TcpSendmsg
tcp_recvmsg       → TcpRecvmsg
page_fault_user   → PageFault
page_fault_kernel → PageFaultMajor
```

### Hardware Counters (perf_event_open)

Sampled at configurable periods (default 1M cycles):
- `PERF_COUNT_HW_CPU_CYCLES`
- `PERF_COUNT_HW_INSTRUCTIONS`
- `PERF_COUNT_HW_CACHE_MISSES`
- `PERF_COUNT_HW_BRANCH_MISSES`
- `PERF_COUNT_SW_CONTEXT_SWITCHES`

### Data Path

```
eBPF probes / perf_event_open
           ↓
    Kernel ring buffer (per-CPU)
           ↓
    Rust collector (mmap + epoll)
           ↓
    Timestamp normalization (TSC → MONOTONIC_RAW)
           ↓
    Event encoding (fixed schema)
           ↓
    Arrow RecordBatch
           ↓
    Parquet (ZSTD compressed)
```

**No async on the data path.** The collector uses synchronous ring buffer polling with a dedicated writer thread.

## Application Instrumentation

### C++ API

Header-only (`instrumentation/include/ygg/ygg.h`):

```cpp
// Register event type (once per process)
ygg::EventRegistry registry("my-server");
auto kind = registry.register_event("ParseFrame");

// Hot path: ~5-10 cycles, no allocation, no locks
YGG_EVENT(ParseFrame, bytes);
YGG_EVENT(AdmissionAccepted, queue_depth);
YGG_EVENT(WalAppend, lsn);
YGG_EVENT(DurableCommit, lsn);
```

### Implementation

- Thread-local fixed-size ring buffers (64K events)
- Single collector thread drains via shared memory
- `rdtsc`/`rdtscp` timestamps calibrated to `CLOCK_MONOTONIC_RAW`
- Cache-line aligned storage
- SPSC ring buffer per thread (single-producer/single-consumer, no formal lock-free progress claim; see `docs/LIMITATIONS.md`)

## Model

### Representation Learning Objectives

**Objective 1: Masked Event Modeling (BERT-style)**
```
RECV → PARSE → [MASK] → WAL_APPEND → SYNC → COMMIT → ACK
                          ↓
                    Predict: ADMIT
```

**Objective 2: Next-Event Prediction (GPT-style)**
```
RECV → PARSE → ADMIT → WAL_APPEND → [predict next]
Expected: FDATASYNC (0.71), WAL_APPEND (0.18), COMMIT (0.08)
Observed: SCHED_SWITCH → divergence signal
```

**Objective 3: Contrastive Execution Learning**
- Positive pairs: segments from equivalent healthy executions
- Negative pairs: healthy vs. Loki-corrupted
- Learns behavioral regime separation

**Objective 4: Temporal Consistency**
- Adjacent windows should evolve smoothly
- Sudden embedding movement = behavioral phase transition

### Architecture

**Phase 1: Transformer Encoder**
```
Events (512) → Token Embeddings → 6-layer Transformer (d_model=256, 8 heads)
                                                          ↓
                                                      Pooling
                                                          ↓
                                                    Execution Embedding (256-d)
```

Token composition:
```
event_type_embedding
    + thread_embedding (bucketed)
    + cpu_embedding
    + Δt_projection (time since prev event)
    + metric_projection (arg0, arg1, arg2)
    ↓
token
```

**Later: Hierarchical Traces**
```
[512 events] → Local Transformer → Window Vector
[512 events] → Local Transformer → Window Vector
[512 events] → Local Transformer → Window Vector
                     ↓
            Global Transformer → Execution Embedding
```

**Future: Causal Graph Representation**
Model concurrent execution as partial-order graph:
- Nodes: events
- Edges: program order, synchronization, wake-up, queue transfer, request lineage, I/O completion

Use Graph Transformer / Temporal Graph Networks to learn representations of **happens-before structure**.

### Training Stack

- **JAX** (not PyTorch) — `jit`, `vmap`, `pmap`, XLA
- **Flax / Equinox** for model definition
- **Polars / NumPy / DuckDB / Arrow** for data loading
- **CUDA** via Vanta (when earned)

## Analysis Capabilities

### CLI Interface

```bash
# Record a trace
ygg record -- ./weir-server

# Inspect trace
ygg inspect trace.parquet

# Compare executions
ygg diff baseline.trace bad.trace

# Find behavioral neighbors
ygg neighbors trace

# Cluster campaign
ygg cluster campaign/

# Embed entire campaign
ygg embed campaign/

# Explain divergence
ygg explain trace
```

### Example Output

```
$ ygg inspect trace.parquet
execution 01JYGGW03M...

events             8,913,441
duration            42.81 s
threads             12
cpus                 8

behavioral distance from baseline:
0.118

nearest executions:
  01JYGG...   healthy/weir/high-load       0.081
  01JYGG...   healthy/weir/high-load       0.093
  01JYGG...   loki/fsync-delay/2ms          0.391

largest divergence:
  31.208s → 31.346s
```

```
$ ygg diff baseline.trace bad.trace
Divergence begins at 31.217843 s

dominant changes:
  + block IO completion latency
  + scheduler migration frequency
  + WAL queue occupancy
  - commit frequency

likely behavioral family:
  storage-stall cluster
```

## Integrations

### Kiln

```toml
[instrumentation.ygg]
enabled = true
events = ["scheduler", "block_io", "syscalls", "application"]
perf = ["cycles", "instructions", "cache-misses"]
```

Kiln artifacts:
```
campaign/
├── run.json
├── metrics.json
├── stdout.log
├── ygg.trace.parquet
└── manifest
```

```bash
kiln compare A B --ygg
```

Output:
```
Benchmark difference: p99 +18.1%
Behavioral difference: embedding distance 0.43
New execution regime detected: yes
```

### Loki

Loki generates controlled fault campaigns. Ygg gets labeled data for free:
- Disk latency injection
- Packet delay
- Partial writes
- Allocator failure
- Syscall errors
- Scheduler interference
- CPU starvation

### Norn

Contention regime dataset:
- CAS retries
- Thread scheduling
- Queue occupancy
- CPU migration
- Cycles, cache misses
- Per-worker progress

Embeddings separate: tight spin, bounded backoff, yield, oversubscribed

### Orda

Workload shape embeddings:
- Price-level churn
- Matching bursts
- Allocation behavior
- Cache pressure
- Workload phase shifts

Test: cross-heavy vs modify-heavy vs cancel-heavy vs sweep

## Research Questions

| ID | Question |
|----|----------|
| RQ1 | Can self-supervised models learn reusable representations of concurrent system execution? |
| RQ2 | Do learned representations separate workload changes from actual failure modes? |
| RQ3 | Can behavioral embeddings identify previously unseen faults? |
| RQ4 | Can the model locate the transition point between healthy and pathological execution? |
| RQ5 | Do models trained on one stra-ta system transfer to another? |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Collector | Rust + Aya (eBPF) + perf_event_open |
| Instrumentation | C++20 header-only + shared memory |
| Schema | FlatBuffers + Arrow/Parquet |
| Training | JAX + Flax/Equinox |
| Data | Polars + NumPy + DuckDB + Arrow |
| Visualization | Static SVGs, UMAP/PCA (no dashboards) |

## Repository Structure

```
ygg/
├── collector/
│   ├── ebpf/probes.bpf.c
│   └── src/{main,ebpf,perf,writer}.rs
├── instrumentation/
│   ├── include/ygg/ygg.h
│   ├── src/{ygg,ring_buffer,collector_thread,ygg_internal}.c
│   ├── src/lib.rs
│   └── build.rs
├── schema/
│   ├── event.fbs
│   ├── EVENT_SCHEMA.md
│   └── src/lib.rs          # ygg-schema crate
├── model/
│   ├── config.py
│   ├── encoder.py
│   ├── hierarchical.py
│   ├── objectives.py
│   ├── dataset.py
│   └── train.py
├── analysis/
│   ├── __init__.py
│   ├── divergence.py
│   ├── clustering.py
│   ├── attribution.py
│   └── viz.py
├── integrations/
│   ├── kiln.rs
│   ├── loki.rs
│   ├── norn.rs
│   └── weir.rs
├── experiments/
│   ├── README.md
│   ├── contention/{README.md,config.toml}
│   ├── storage-stall/{README.md,config.toml}
│   └── workload-shift/{README.md,config.toml}
└── docs/
    ├── ARCHITECTURE.md
    ├── TRACE-MODEL.md
    ├── REPRESENTATION.md
    └── EXPERIMENTS.md
```

## Milestones

### V0.1: Norn Contention Traces
- Instrument Norn push/pop/CAS/yield/spin/queue occupancy
- Collect scheduler switches, CPU migrations, cycles, cache misses
- Kiln campaigns: 1x1, 2x2, 4x4, 8x8 × {tight, yield, bounded, exponential}
- Train tiny encoder → embeddings separate contention regimes (no labels)

### V0.2: Loki Fault Campaigns (Weir)
- Healthy vs fsync +250µs/+1ms/+5ms vs CPU starvation
- Verify: separation, magnitude ordering, phase transition detection

### V0.3: Cross-System Transfer
- Train: Norn + Weir
- Evaluate: Orda
- Test: contention/failure structure transfer

## Current Implementation Status

This section reflects what is actually built today (see `GUIDE.md` for working commands). It augments, not replaces, the vision above.

| Component | State | Key facts |
|-----------|-------|-----------|
| Collector | Implemented | Rust + eBPF (aya) + `perf_event_open`; files `main.rs`, `ebpf.rs`, `perf.rs`, `writer.rs`. Linux-only (macOS no-op). `cargo build --release` works. The collector uses the `ygg-schema` crate for event definitions. |
| Instrumentation | Implemented | C++20 `ygg.h` + C sources + Rust FFI (`lib.rs`) built via `build.rs` (cc). Thread-local SPSC ring buffers, `/dev/shm/ygg-<pid>`, 100ms TSC calibration, single collector pthread with Unix socket + spill-file fallback. 2/2 tests pass. |
| Schema | Implemented | `ygg-schema` Rust crate: `Event`, `EventKind`, Arrow/Parquet schemas. Resolved as part of the cargo workspace build graph. |
| Model | Implemented | JAX/Flax: `config.py`, `encoder.py`, `hierarchical.py`, `objectives.py` (4 losses), `dataset.py` (streaming Parquet, Kiln discovery), `train.py` (multi-objective, validation, checkpointing, pmap). `python -m model.train --help` works. |
| Analysis | Implemented | `divergence.py` (DTW, PELT/BOCPD, sustained divergence), `clustering.py` (HDBSCAN, UMAP/PCA, FAISS), `attribution.py` (integrated gradients), `viz.py` (static SVGs). `python -c "import analysis"` works. |
| Integrations | Scaffolded | Kiln/Loki/Norn/Weir points exist but not yet wired to finished components. |
| Experiments | V0.1 synthetic evidence | Contention V0.1 synthetic pipeline implemented (generation → masked-only training → UMAP → silhouette 0.30, no policy labels). Real Norn campaign pending. V0.2/V0.3 scaffolded. |

> The `ygg record/inspect/diff/neighbors/cluster/embed/explain` CLI commands and the example outputs above describe the target interface, not yet-implemented behavior.
