# Storage Stall (V0.2 / Loki Fault Campaigns, Weir)

## Hypothesis
A contrastive encoder trained over Loki fault campaigns (run through Weir)
will (a) separate healthy from faulted traces, (b) order faulted traces by
the magnitude of the injected storage fault, and (c) detect the point at which
a system transitions from healthy to degraded behavior.

## Data source
- Fault injection: `loki` fault plans (e.g. fsync delay).
- Campaign runner: `weir` (healthy baseline vs faulted runs).
- Campaigns: healthy, fsync `+250us` / `+1ms` / `+5ms`, and CPU starvation.
- Trace layout: `campaigns/loki/*/ygg.trace.parquet` (matches
  `config.toml` `[data].trace_glob`).

## Success criteria
- `metric = "knn_accuracy"`, `expected = "ordered"`.
- Separation: healthy vs each fault class is cleanly distinguishable.
- Magnitude ordering: fsync `+250us` < `+1ms` < `+5ms` in embedding distance
  from the healthy centroid.
- Phase transition: the encoder flags the degradation boundary as fault
  magnitude increases.

## How to run
1. Generate Loki fault plans and run the Weir campaign for each plan:
   ```
   ygg-collector --output campaigns/loki/<plan>/ygg.trace.parquet -- ./weir-server
   ```
2. Train the contrastive encoder:
   ```
   ygg-train --config experiments/storage-stall/config.toml
   ```
3. Run magnitude-ordering and phase-transition checks via the analysis
   harness (see `analysis/clustering.py` and `analysis/viz.py`).
