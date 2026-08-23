# Workload Shift (V0.3 / Cross-System Transfer)

## Hypothesis
A contrastive encoder trained on Norn contention traces and Weir fault
campaigns will learn structure that transfers to a different system
(`orda`): embeddings of Orda traces under analogous contention and failure
conditions should align with the Norn/Weir regime geometry without being
retrained on Orda.

## Data source
- Primary training traces: `norn` contention campaigns
  (`campaigns/norn/*/ygg.trace.parquet`).
- Auxiliary labels/signals: `weir` fault campaigns for failure structure.
- Transfer target: `orda` traces collected under matched conditions.
- Note: the `config.toml` `[data].source` records the primary training source
  (`norn`); Orda traces are held out as the evaluation set, not used in
  training.

## Success criteria
- `metric = "knn_accuracy"`, `expected = "detected"`.
- Contention/failure structure learned on Norn + Weir is recovered on Orda
  (kNN accuracy on Orda regime labels above chance).
- Transfer is measured zero-shot: no Orda labels used during training.

## How to run
1. Collect Norn + Weir training traces and a separate Orda evaluation set:
   ```
   ygg-collector --output campaigns/norn/<id>/ygg.trace.parquet -- ./norn-bench
   ygg-collector --output campaigns/orda/<id>/ygg.trace.parquet -- ./orda-server
   ```
2. Train the contrastive encoder:
   ```
   ygg-train --config experiments/workload-shift/config.toml
   ```
3. Evaluate transfer on the Orda held-out set (zero-shot kNN against the
   Norn/Weir embedding space).
