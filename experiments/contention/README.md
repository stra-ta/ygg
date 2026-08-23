# Contention Traces (V0.1 / Norn)

## Hypothesis
A tiny encoder trained with a masked objective over Norn scheduler contention
traces will learn representations that separate distinct contention regimes
without any labels. The regimes are defined by grid size and workload shape,
not by human annotation.

## Data source
- Instrumentation: `norn` (push/pop/CAS/yield/spin/queue-occupancy events).
- Collected signals: scheduler switches, CPU migrations, cycles, cache misses.
- Campaign grid: `1x1`, `2x2`, `4x4`, `8x8` crossed with
  `{tight, yield, bounded, exponential}`.
- Trace layout: `campaigns/norn/*/ygg.trace.parquet` (matches `config.toml`
  `[data].trace_glob`).

## Success criteria
- `metric = "silhouette"`, `expected = "separate"`.
- Embeddings of the 16 regime combinations form clearly separated clusters
  (no labels used during training).
- Silhouette score is meaningfully above the random-baseline for the same
  grid count.

## How to run
1. Collect Norn contention traces for every grid x workload combination:
   ```
   ygg-collector --output campaigns/norn/<id>/ygg.trace.parquet -- ./norn-bench
   ```
2. Train the encoder using this experiment config:
   ```
   ygg-train --config experiments/contention/config.toml
   ```
3. Inspect the clustering report / embedding projection produced by the
   analysis harness (see `analysis/clustering.py`).
