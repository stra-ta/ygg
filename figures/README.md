# Figures

Generated figures for the Ygg project.

Each figure links to its generator script and the measured data it is built from.
Figures are regenerated from source data; they are not hand-edited bitmaps.

## Index

| Figure | File | Generator | Source data | Description |
| ------ | ---- | --------- | ----------- | ----------- |
| 1 | [embedding_map.svg](embedding_map.svg) | [../experiments/contention/real/embed_real.py](../experiments/contention/real/embed_real.py) | [../experiments/contention/real/\*.parquet](../experiments/contention/real/), [metrics.json](../experiments/contention/real/metrics.json) | Label-free V0.1 encoder embeddings of 16 real Norn contention traces (window-level), colored by backoff regime. Honest result: masked-only self-supervision does not yet separate regimes (silhouette -0.31). |
| 2 | [overhead.svg](overhead.svg) | [_make_overhead_svg.py](_make_overhead_svg.py) | [../instrumentation/bench/results.json](../instrumentation/bench/results.json) | Measured instrumentation hot-path overhead: median/p95/p99 per-event cost and dropped-event counts across the 5 benchmark scenarios. |
| 3 | [divergence_localization.svg](divergence_localization.svg) | [../experiments/divergence/run_localization.py](../experiments/divergence/run_localization.py) | [../experiments/divergence/healthy.parquet](../experiments/divergence/healthy.parquet), [switched.parquet](../experiments/divergence/switched.parquet), [switch_meta.json](../experiments/divergence/switch_meta.json) | Blind detection of a mid-execution backoff-policy switch (bounded to tight-spin at the midpoint) via DTW + PELT/BOCPD change-point localization. |

## Figure 2 - Instrumentation overhead

`overhead.svg` plots the measured accept cost of `YGG_EVENT` emission.

It shows grouped bars of the median (p50), p95, and p99 per-event cost for each of the 5 benchmark scenarios:

- `baseline` - empty-loop measurement floor, no `YGG_EVENT`.
- `no collector` - ring fills because the collector is stopped; events are dropped.
- `draining` - active collector draining to a spill file; the real accept hot path.
- `ring at capacity` - burst faster than the collector drains; events are dropped.
- `threads` - multi-threaded emission (4T and 8T runs averaged).

Dropped-event counts are annotated on the two drop-path scenarios.

### Honesty note on units

The benchmark reports its unit in the data, and the figure labels it accordingly.

On `macos-arm64` the unit is `ns` (nanoseconds via `mach_absolute_time`), so the figure labels the axis `ns` and never claims "cycles".

The `~5-10 cycles` figure in `instrumentation/include/ygg/ygg.h` is an estimate for x86-64 `rdtscp`. On x86-64 the benchmark reports raw TSC ticks and the figure would label the axis `cycles`; on this macOS run it does not.

The active-collector scenarios (`draining`, `threads`) measure ~80-100 ns per event, consistent with the header's documented 70-80 ns on macOS/arm64.

### Regenerating

```sh
cd instrumentation/bench && cargo run --release   # writes results.json
python3 figures/_make_overhead_svg.py             # writes figures/overhead.svg
```

## Figure 1 - Real Norn embedding map (label-free V0.1)

`embedding_map.svg` is the V0.1 "does it learn anything real" figure.

It embeds 16 real Norn contention traces captured by the standalone capture
program (`experiments/contention/real/norn_capture.cpp`, which links the Norn
library and the Ygg C instrumentation). The encoder was trained with the
masked-event objective only, no policy labels.

Each point is one local window of events from a real trace, reduced with UMAP
(PCA fallback) and colored by backoff regime (tight / yield / bounded /
exponential), shaped by thread count (2 / 4 / 8 / 16).

### Honest result

The label-free V0.1 encoder does not yet separate the real backoff regimes.

The window-level silhouette by regime is -0.31 (per-regime: yield 0.30, tight
-0.36, exponential -0.24, bounded -0.96). Per-trace and per-thread-count
silhouettes are also negative. This is expected, not a bug: real backoff
policies emit the same application-level event types (kinds 1000-1004) and
differ mainly in timing and argument patterns, which the masked-event-type
loss weakly supervises.

By contrast, the synthetic V0.1 run separates at silhouette 0.30 because its
regimes differ in event-type structure, confirming the encoder learns when a
signal exists. This is exactly the gap the temporal (next-event dt) and
contrastive objectives target for V0.2: they supervise the timing/argument
structure that distinguishes real policies.

The pipeline itself is validated end to end: real traces are captured,
embedded, and visualized without any labels.

### Regenerating

```sh
cd experiments/contention/real && ./build.sh && ./run_capture.sh   # writes *.parquet
python3 train_real.py                                          # writes results/encoder_real.msgpack
python3 embed_real.py                                          # writes ../../figures/embedding_map.svg
```

## Figure 3 - Divergence localization (blind policy switch)

`divergence_localization.svg` shows blind detection of a mid-execution
backoff-policy switch.

A synthetic healthy trace is generated, then a policy switch is injected at the
midpoint (bounded backoff to tight spin). The analysis (DTW windowed
divergence + PELT/BOCPD change-point detection) localizes the switch with no
knowledge of where it was inserted.

Measured result: detected 0.028 s vs true 0.028 s, error 0.0000 s (0.06%),
with BOCPD independently landing at 0.019 s.

### Regenerating

```sh
cd experiments/divergence && python3 generate_switch_trace.py   # writes healthy.parquet, switched.parquet, switch_meta.json
python3 run_localization.py                                     # writes ../../figures/divergence_localization.svg
```
