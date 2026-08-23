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
| 4 | [ablation_silhouette.svg](ablation_silhouette.svg) | [../experiments/contention/real/run_ablation.py](../experiments/contention/real/run_ablation.py) | [../experiments/contention/real/results/ablation_results.json](../experiments/contention/real/results/ablation_results.json) | V0.2 objective ablation on real Norn traces: does adding timing / argument / contrastive supervision rescue regime separation? Answer: no. All five variants land at silhouette -0.06 to -0.09 on held-out runs. |

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

## Figure 4 - V0.2 objective ablation (does richer supervision rescue separation?)

`ablation_silhouette.svg` answers the V0.2 hypothesis directly.

The question: the V0.1 masked-event objective is blind to timing and argument
structure, which is what actually distinguishes real backoff policies. Does
adding temporal / argument / contrastive supervision fix regime separation on
real Norn traces?

The study: 160 independent captures (4 regimes x 4 thread configs x 10 runs),
label-free V0.2 encoder trained per variant, silhouette by regime measured on
held-out runs only. Splits are by run, never by window, so no model can lean
on machine/run fingerprints.

| variant | silhouette | yield | bounded | exponential | tight |
| --- | ---: | ---: | ---: | ---: | ---: |
| event (V0.1 baseline) | -0.068 | +0.084 | +0.028 | -0.008 | -0.018 |
| event+timing | -0.081 | +0.071 | +0.031 | +0.002 | -0.012 |
| event+args | -0.086 | +0.073 | +0.025 | +0.008 | -0.012 |
| event+timing+args | -0.088 | +0.073 | +0.027 | +0.008 | -0.011 |
| event+contrastive | -0.060 | +0.084 | +0.023 | -0.008 | -0.014 |

### Honest result

The hypothesis is falsified.

Adding masked-dt reconstruction, argument regression, next-step prediction,
and InfoNCE contrastive training does not separate real backoff regimes.
Every variant lands between -0.06 and -0.09, statistically indistinguishable,
with `yield` the only consistently positive regime (+0.07 to +0.08) across all
five variants. The objectives change what the encoder predicts; they do not
change whether the embedding separates policies.

This also corrects Figure 1's headline number. That figure reported -0.31 under
the older protocol (raw arg tokens up to 1e9 dominating the embedding, windows
aggregated without run-level splits). Under corrected metric scaling and
run-level evaluation the true picture is near zero, not strongly negative.
Figure 4 supersedes it as the rigorous measurement of the same question.

### What this means

Application-level events alone appear insufficient to distinguish these real
backoff regimes, regardless of self-supervised objective. The discriminating
signal likely lives at the system level: scheduler behavior, preemption,
migrations, cache pressure. Capturing that requires the Linux eBPF collector,
which macOS cannot run. That is the concrete next lever, not more objective
engineering on app-level data.

### Regenerating

```sh
cd experiments/contention/real
N_RUNS=10 bash run_capture.sh        # 160 traces (~12 min), writes *_run<k>.parquet
python3 train_v02.py --variant <name> --epochs 12   # per-variant checkpoints
python3 run_ablation.py              # trains missing variants, evaluates, writes table + figure
```

Checkpoints (`results/encoder_v02_*.msgpack`) are gitignored; the ablation
retrains any missing variant automatically.
