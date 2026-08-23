# Ygg Experiments

Version-controlled definitions for the Ygg embedding experiments. Each experiment
lives in its own directory with a `config.toml` (model / training / data /
validation) and a `README.md` (hypothesis, data source, success criteria, how to
run).

## Experiments

| Experiment | Version | Source | Objective | Validation | Status |
| --- | --- | --- | --- | --- | --- |
| [contention](contention/) | 0.1.0 | norn | masked | silhouette / separate | scaffolded |
| [storage-stall](storage-stall/) | 0.2.0 | loki | contrastive | knn_accuracy / ordered | scaffolded |
| [workload-shift](workload-shift/) | 0.3.0 | norn | contrastive | knn_accuracy / detected | scaffolded |

## Roadmap (from the Ygg vision)
- **V0.1 Norn Contention Traces** - instrument Norn, run Kiln campaigns over
  grid sizes x workload shapes, train a masked encoder, verify label-free
  separation of contention regimes. (`contention/`)
- **V0.2 Loki Fault Campaigns (Weir)** - healthy vs fsync latency vs CPU
  starvation; verify separation, magnitude ordering, phase-transition
  detection. (`storage-stall/`)
- **V0.3 Cross-System Transfer** - train Norn + Weir, evaluate Orda; test
  whether contention / failure structure transfers. (`workload-shift/`)

## Config schema
Every `config.toml` follows the same shape:
```toml
[experiment]   # name, version, objective (masked|contrastive|next), description
[model]        # d_model, n_layers, n_heads, window_size, n_windows
[training]     # epochs, batch_size, lr
[data]         # source (kiln|norn|loki|weir), trace_glob
[validation]   # metric (silhouette|knn_accuracy), expected (separate|ordered|detected)
```

## How to add a new experiment
1. Create `experiments/<name>/`.
2. Add `config.toml` using the schema above (pick `objective` and
   `validation.expected` that match your hypothesis).
3. Add `README.md` describing hypothesis, data source, success criteria, and
   run steps.
4. Add a row to the table above with status `scaffolded` (or `active` /
   `completed` as work proceeds).
5. Commit the directory so the definition stays version-controlled alongside
   the code that consumes it.
