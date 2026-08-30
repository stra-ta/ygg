# Ygg

**Representation learning for systems execution.**

A trace intelligence engine that watches programs run, turns execution into structured traces, and learns embeddings of system behavior - so you can ask whether an execution is normal, where it diverged, and what it resembles, without hand-written labels.

## What Ygg Does

Ygg learns:

```
execution → vector
```

where executions with similar underlying behavior land close together. From that representation it answers:

- Is this execution normal?
- Where did behavior diverge?
- What class of failure does this resemble?
- Did a code change create a new execution regime?

The important word is **representation**. Ygg doesn't start from labels like "deadlock" or "disk stall". It learns the shape of execution first, then answers questions in that space.

## Quick Start

```bash
# Build the Rust collector (Linux recommended for full eBPF/perf support)
cargo build --release

# Record a trace (requires Linux + root for eBPF/perf)
sudo ./target/release/ygg-collector --output trace.parquet -- ./weir-server

# Confirm the model training entry point loads
python -m model.train --help

# Confirm the analysis package imports
python -c "import analysis"
```

> **Platform note:** eBPF (aya) and `perf_event_open` hardware counters are Linux-only, gated behind `cfg(target_os = "linux")`. On macOS the crate compiles but performs no kernel collection. Application-level C instrumentation works on both platforms. See [DEVELOPMENT.md](DEVELOPMENT.md).

## Measured Results

All numbers below are from committed artifacts in this repo, not projections.

### Instrumentation overhead (macOS arm64)

Per-event cost of `YGG_EVENT` emission, measured by `instrumentation/bench`:

| Scenario | Threads | p50 | p95 | p99 | Dropped |
|---|---|---:|---:|---:|---:|
| baseline (no events) | 1 | 5 ns | 5 ns | 5 ns | 0 |
| event, collector stopped | 1 | 7 ns | 63 ns | 88 ns | 936k / 1M |
| event, collector draining | 1 | 80 ns | 82 ns | 108 ns | 0 |
| ring at capacity | 1 | 9 ns | 61 ns | 80 ns | 936k / 1M |
| multi-threaded draining | 4 | 103 ns | 223 ns | 979 ns | 0 |
| multi-threaded draining | 8 | 82 ns | 285 ns | 2015 ns | 0 |

The active-path cost is ~80 ns/event single-threaded. The p99 tail grows sharply under concurrent emission (~2 us at 8 threads); that tail is itself a study target.

Source: [instrumentation/bench/results.json](instrumentation/bench/results.json), plotted in [figures/overhead.svg](figures/overhead.svg).

### Learned representations: what worked and what did not

| Study | Data | Result |
|---|---|---|
| Synthetic contention regimes | generated traces | silhouette **+0.30** - regimes separate under masked-only training |
| Blind policy-switch localization | injected switch | detected 0.028 s vs true 0.028 s (**0.06% error**), BOCPD independently at 0.019 s |
| Real Norn regimes, objective ablation | 160 captured traces (10 runs x 16 cells) | all five objective variants land at **-0.06 to -0.09**: adding timing / argument / contrastive supervision does not rescue separation |

The honest reading: on application-level events alone, real backoff policies do not separate regardless of self-supervised objective. The discriminating signal likely lives at the system level (scheduler, preemption, migration), which requires the Linux eBPF path. Full analysis: [figures/README.md](figures/README.md).

## Components

| Component | Language | State | Purpose |
|-----------|----------|-------|---------|
| `collector/` | Rust + eBPF | Implemented | Kernel/system trace collection. Linux-only telemetry; macOS build is a no-op. |
| `instrumentation/` | C++20 | Implemented | Application event emission via thread-local SPSC rings and shared memory. 2/2 tests pass. |
| `schema/` | Rust (`ygg-schema`) | Implemented | Event schema, Arrow/Parquet serialization. |
| `model/` | JAX + Flax | Implemented | Hierarchical transformer encoder, six self-supervised objectives, training + checkpointing. |
| `analysis/` | Python | Implemented | DTW divergence, PELT/BOCPD change points, clustering, attribution, SVG figures. |
| `experiments/` | Python | Active | Versioned campaign definitions. Contention campaign has synthetic evidence plus a real Norn run and a V0.2 ablation. |
| `integrations/` | various | Scaffolded | Kiln/Loki/Norn/Weir points exist, not yet wired. |

## Development

```bash
cargo build --release                          # Rust workspace
cargo test -p ygg-instrumentation --release    # instrumentation tests
python -m pytest tests/ -q                     # model/objective tests (19)
python experiments/contention/train_synthetic.py   # end-to-end V0.1 pipeline
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for full instructions.

## Reproducing the studies

```sh
cd experiments/contention/real
N_RUNS=10 bash run_capture.sh        # capture 160 traces from the local Norn library (~12 min)
python3 train_v02.py --variant event --epochs 12
python3 run_ablation.py              # trains missing variants, writes table + figure
```

Every figure in [figures/](figures/) links to its generator script and raw data. Figures are regenerated, never hand-edited.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - system design, data model, component architecture, milestones.
- [docs/TRACE-MODEL.md](docs/TRACE-MODEL.md) - fixed-width event record shared by collector, instrumentation, and model.
- [docs/REPRESENTATION.md](docs/REPRESENTATION.md) - token composition and the hierarchical encoder design.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) - experiment framework: how campaigns pin model, data, and validation to a hypothesis.
- [figures/README.md](figures/README.md) - every figure with its generator, source data, and honest interpretation.
- [DEVELOPMENT.md](DEVELOPMENT.md) - build, test, run.

## Limitations

- Kernel telemetry (eBPF, perf counters) is Linux-only. On macOS, only application-level events are available, which the V0.2 ablation shows is insufficient to separate real backoff regimes.
- The CLI commands described in ARCHITECTURE.md describe the target interface, not shipped behavior.
- Integrations with Kiln/Loki/Norn/Weir are scaffolded, not wired.

## License

MIT
