# Ygg Development

## Prerequisites

- Rust 1.78+
- clang (for eBPF compilation)
- Linux kernel headers (for vmlinux.h)
- Python 3.11+ with JAX
- C++20 compiler

### Linux Kernel Headers (for eBPF)

```bash
# Ubuntu/Debian
sudo apt install linux-headers-$(uname -r) clang llvm

# Arch
sudo pacman -S linux-headers clang llvm

# Fedora
sudo dnf install kernel-devel clang llvm
```

### vmlinux.h Generation

```bash
cd collector/ebpf
bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h
```

## Building

The Rust workspace (`collector`, `instrumentation`) builds together with Cargo. The `instrumentation` crate compiles its C/C++ sources via `build.rs` (the `cc` crate) and emits `libygg_instrumentation.a`/`.so` — there is no CMake step.

```bash
# Build everything in the Rust workspace (collector + instrumentation)
cargo build --release

# Run on Linux with root for eBPF/perf
sudo ./target/release/ygg-collector --output trace.parquet -- ./weir-server
```

> **Platform note:** eBPF (aya) and `perf_event_open` are Linux-only, gated by `cfg(target_os = "linux")`. The collector compiles on macOS but the telemetry data path is a no-op there.

```bash
# Install Python deps for model/ + analysis/
pip install -e .
```

The `schema/` directory is a standalone Rust crate (`ygg-schema`). Cargo resolves it as part of the workspace build graph (via path dependency); `cargo build --release` from the workspace root builds it alongside the collector and instrumentation.

## Testing

```bash
# Instrumentation unit tests (2/2 pass)
cargo test -p ygg-instrumentation

# Confirm the model training entry point loads
python -m model.train --help

# Confirm the analysis package imports
python -c "import analysis"
```

There is no `ygg` CLI yet (the `ygg record/inspect/diff/...` commands in ARCHITECTURE.md are aspirational). Use the commands above to exercise the implemented surface.

## Status

Per-component implementation status:

- **collector/** — *Implemented.* Rust + eBPF (aya) + `perf_event_open`. Key files: `src/main.rs` (CLI/orchestration), `src/ebpf.rs` (eBPF loading + ring buffer polling), `src/perf.rs` (hardware counters), `src/writer.rs` (Arrow/Parquet ZSTD writer), `src/schema.rs` (being replaced by `ygg-schema`). Linux-only; macOS no-op.
- **instrumentation/** — *Implemented.* C++20 header `include/ygg/ygg.h` + `src/ygg.c`, `src/ring_buffer.c`, `src/collector_thread.c`, `src/ygg_internal.h`, Rust FFI `src/lib.rs`, `build.rs` (cc). Thread-local SPSC ring buffers, `/dev/shm/ygg-<pid>`, 100ms TSC calibration, single collector pthread with Unix socket + spill-file fallback. 2/2 tests pass.
- **schema/** — *Implemented.* `ygg-schema` crate with `Event`, `EventKind`, Arrow/Parquet schemas.
- **model/** — *Implemented.* JAX/Flax: `config.py`, `encoder.py`, `hierarchical.py`, `objectives.py` (4 self-supervised losses), `dataset.py` (streaming Parquet + Kiln campaign discovery), `train.py` (multi-objective, validation, checkpointing, pmap). `python -m model.train --help` works.
- **analysis/** — *Implemented.* `divergence.py` (DTW, PELT/BOCPD, sustained divergence), `clustering.py` (HDBSCAN, UMAP/PCA, FAISS), `attribution.py` (integrated gradients), `viz.py` (static SVGs).
- **integrations/** — *Scaffolded.* Kiln/Loki/Norn/Weir points exist but not yet wired to finished components.
- **experiments/** — *Scaffolded.* Campaign directories only.

## Running the Collector

```bash
# Basic trace collection
sudo ./target/release/ygg-collector --output trace.parquet -- ./weir-server

# With specific events
sudo ./target/release/ygg-collector \
    --output trace.parquet \
    --sched \
    --syscalls \
    --block-io \
    --perf \
    --perf-events cycles,instructions,cache-misses \
    -- ./weir-server

# Duration-limited
sudo ./target/release/ygg-collector --output trace.parquet --duration 60 -- ./weir-server
```

## Using the Instrumentation Library

```cpp
#include <ygg/ygg.h>

int main() {
    ygg::EventRegistry registry("my-server");
    auto parse_kind = registry.register_event("ParseFrame");
    auto admit_kind = registry.register_event("AdmissionAccepted");

    // Hot path
    YGG_EVENT(ParseFrame, bytes);
    YGG_EVENT(AdmissionAccepted, queue_depth);

    ygg_shutdown();
    return 0;
}
```

Link with `-lygg_instrumentation`.

## Training

```bash
# Masked event modeling
python -m model.train traces/*.parquet --objective masked --epochs 10

# Next-event prediction
python -m model.train traces/*.parquet --objective next --epochs 10

# Contrastive (requires healthy + faulty pairs)
python -m model.train healthy/*.parquet faulty/*.parquet --objective contrastive
```

## Analysis

```python
from analysis import diff_traces, embed_campaign, cluster_embeddings
from model.encoder import create_embedder, EncoderConfig
from flax.training import checkpoints

# Load model
config = EncoderConfig()
embedder = create_embedder(config)
params = checkpoints.restore_checkpoint("checkpoints", target=None)["params"]

# Diff two traces
result = diff_traces(embedder, params, "baseline.parquet", "experiment.parquet")
print(result)

# Cluster campaign
embeddings = embed_campaign(embedder, params, "campaign/")
labels, _ = cluster_embeddings(embeddings)
plot_clusters(embeddings, labels, "clusters.png")
```

## Project Structure

```
ygg/
├── collector/           # Rust eBPF/perf collector
│   ├── ebpf/
│   │   ├── probes.bpf.c    # eBPF probes
│   │   └── vmlinux.h       # Kernel BTF (generated)
│   ├── perf/           # perf_event_open wrapper
│   └── src/
│       └── main.rs     # Collector entry point
├── instrumentation/     # C++ application instrumentation
│   ├── include/ygg/
│   │   └── ygg.h       # Header-only API
│   └── src/
│       └── ygg.c       # Implementation
├── schema/              # Event schema
│   ├── event.fbs       # FlatBuffers schema
│   └── EVENT_SCHEMA.md
├── model/               # JAX/Flax models
│   ├── encoder.py      # Transformer encoder
│   ├── dataset.py      # Parquet dataset loading
│   └── train.py        # Training entry point
├── analysis/            # Python analysis tools
│   ├── divergence.py   # Divergence detection
│   ├── clustering.py   # Behavioral clustering
│   └── attribution.py  # Feature attribution
├── integrations/        # Stra-ta stack integrations
│   ├── kiln.rs
│   ├── loki.rs
│   ├── norn.rs
│   └── weir.rs
├── experiments/         # Version-controlled experiments
│   ├── contention/
│   ├── storage-stall/
│   └── workload-shift/
└── docs/
    ├── ARCHITECTURE.md
    ├── TRACE-MODEL.md
    ├── REPRESENTATION.md
    └── EXPERIMENTS.md
```