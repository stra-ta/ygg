# Ygg experiment framework.

Experiments are version-controlled definitions that pin a model, training
regime, data source, and validation criterion to a hypothesis. They live
under `experiments/` and are the contract between the trace campaigns and the
training/evaluation code.

## Layout

Each experiment is its own directory containing:

- `config.toml` - machine-readable model / training / data / validation
  settings.
- `README.md` - hypothesis, data source, success criteria, and run steps.

Keeping both files in the repo means an experiment is reproducible from its
definition alone (and reviewable as a diff).

## Config schema

Every `config.toml` follows the same five-section shape (see
`experiments/README.md` for the index):

```toml
[experiment]   # name, version, objective (masked|contrastive|next), description
[model]        # d_model, n_layers, n_heads, window_size, n_windows
[training]     # epochs, batch_size, lr
[data]         # source (norn|loki|weir|...), trace_glob
[validation]   # metric (silhouette|knn_accuracy), expected (separate|ordered|detected)
```

The `objective` selects which self-supervised loss dominates the run
(see `docs/REPRESENTATION.md`); `validation.metric` and
`validation.expected` define what "success" means for that experiment.

## Milestones

The roadmap in `experiments/README.md` defines three milestones, each with a
real `config.toml` + `README.md` already scaffolded:

### V0.1 - Norn Contention (`experiments/contention/`)

- `source = norn`, `objective = masked`
- `validation.metric = silhouette`, `validation.expected = separate`
- Hypothesis: a masked encoder over Norn scheduler-contention traces learns
  embeddings that separate distinct contention regimes (grid size x workload
  shape) with no labels.
- Success: silhouette of the 16 regime combinations clearly above the
  random baseline.

### V0.2 - Loki Fault Campaigns (`experiments/storage-stall/`)

- `source = loki`, `objective = contrastive`
- `validation.metric = knn_accuracy`, `validation.expected = ordered`
- Hypothesis: a contrastive encoder over Loki/Weir fault campaigns learns
  embeddings that separate fault regimes and order by injected fault
  magnitude (and detect phase transitions).
- Success: 1-NN accuracy recovers the fault-magnitude ordering.

### V0.3 - Cross-System Transfer (`experiments/workload-shift/`)

- `source = norn`, `objective = contrastive`
- `validation.metric = knn_accuracy`, `validation.expected = detected`
- Hypothesis: a contrastive encoder trained on Norn (and Weir failure
  structure) transfers to a different system, **Orda**, as a zero-shot
  target. Orda traces are held out as the evaluation set; no Orda labels are
  used in training.
- Success: contention/failure structure learned on Norn + Weir is recovered
  on Orda (kNN accuracy on Orda regime labels above chance, zero-shot).

## How to run

A typical experiment flow:

1. **Collect** traces with the collector:

   ```bash
   ygg-collector --output campaigns/norn/<id>/ygg.trace.parquet -- ./norn-bench
   ```

   (Collector CLI and Parquet layout: `docs/TRACE-MODEL.md`.)

2. **Train** the encoder with the experiment config:

   ```bash
   model.train --config experiments/contention/config.toml
   ```

   (The `experiments/*/README.md` files invoke this as `ygg-train`; both
   refer to the same training entry point in `model/train.py`.)

3. **Analyze** the resulting embeddings with the `analysis` package
   (clustering, divergence, attribution, and viz under `analysis/`,
   driven by the `validation.metric` from `config.toml`).

## See also

- `experiments/README.md` - the experiment index, roadmap, and "how to add a
  new experiment".
- `experiments/contention/`, `experiments/storage-stall/`,
  `experiments/workload-shift/` - the scaffolded milestone definitions.
- `docs/REPRESENTATION.md` - objectives and training that consume these
  configs.
- `docs/TRACE-MODEL.md` - the trace format the `data.trace_glob` points at.
