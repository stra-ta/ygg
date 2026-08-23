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

```bash
# Build Rust collector
cargo build --release

# Build C++ instrumentation
cd instrumentation && mkdir build && cd build
cmake .. && make

# Install Python deps
pip install -e .
```

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