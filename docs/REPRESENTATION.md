# Representation learning approach for Ygg execution embeddings.

Ygg learns fixed-dimensional embeddings of a concurrent-system execution
from its trace. The model is implemented in JAX/Flax under `model/`:
`model/encoder.py` (token embedding, transformer block, heads),
`model/hierarchical.py` (two-stage encoder), and `model/objectives.py`
(self-supervised losses).

## Token composition

A single event becomes a `token_dim = 7` vector consumed by
`TokenEmbedding` in `model/encoder.py`:

```text
[0] event_type  - categorical (vocab_size)
[1] thread      - categorical (bucketed, 256)
[2] cpu         - categorical (64)
[3] log_dt      - continuous, log-scaled time delta
[4:7] arg0,arg1,arg2 - continuous metrics
```

Each component is projected into `d_model` and summed:

```text
token = event_type_emb
      + thread_emb        (nn.Embed, 256)
      + cpu_emb           (nn.Embed, 64)
      + dt_proj(log_dt)   (nn.Dense)
      + metric_proj(arg0,arg1,arg2)  (nn.Dense)
      + pos_emb           (learned, window-local index)
```

`dt` is the log-scaled inter-event delta (`log_dt`); `pos_emb` is a learned
positional embedding indexed by position modulo `max_seq_len`, so sequences
longer than one window still receive a valid cyclic position.

## Hierarchical encoder

Implemented in `model/hierarchical.py` (`HierarchicalEncoder`). Two stages,
both built from the pre-LayerNorm `TransformerBlock` in `model/encoder.py`.

1. **LocalEncoder** - transformer over the events within a single window
   (`window_size` events). `local_layers` blocks produce per-event context;
   a masked mean over the window yields a per-window embedding
   (`window`).
2. **GlobalEncoder** - transformer over the sequence of `N` window
   embeddings (`global_layers` blocks, plus a window positional embedding),
   producing contextualized window embeddings (`window_ctx`) and a masked
   mean over real windows giving the execution-level embedding (`exec`).

The top-level `YggModel` returns `event`, `window`, `window_ctx`, `exec`
plus the head outputs `masked_logits`, `next_logits`, and `proj` (the
projected execution embedding used by the contrastive objective).

## Config

Defaults from `model/config.py` (`ModelConfig`):

```text
d_model        = 256
n_layers       = 6        # total transformer depth reference
n_heads        = 8
d_ff           = 1024
local_layers   = 6
global_layers  = 4
window_size    = 512
n_windows      = 4        # sequence length = n_windows * window_size
vocab_size     = 10000
max_seq_len    = 512
dropout        = 0.1
```

Objective weights and hyperparameters (also in `ModelConfig`):

```text
masked_weight      = 1.0
next_weight        = 1.0
contrastive_weight = 0.5
temporal_weight    = 0.3
mask_ratio         = 0.15
mask_token_id      = 0
temperature        = 0.07
```

## Self-supervised objectives

Defined in `model/objectives.py`. All are pure functions composed inside a
single `jax.value_and_grad` graph via `combine_losses` (weighted sum,
zero-weight terms skipped).

1. **Masked Event Modeling** - `masked_event_loss`. Randomly mask
   `mask_ratio` (0.15) of event-type positions (`mask_token_id`); predict
   the original event type with cross-entropy over masked positions only.
   Bidirectional context.

2. **Next-Event Prediction** - `next_event_loss` (autoregressive). With a
   causal attention mask, predict the next event type at each position;
   cross-entropy over shifted labels (padding uses `ignore_index`).

3. **Contrastive** - `contrastive_loss` (InfoNCE). L2-normalized anchor
   (`proj`) has one positive (a healthy execution embedding) and uses every
   other positive plus every negative in the batch as distractors. Negatives
   are corrupted executions. Returns `(loss, accuracy)`. Temperature is
   `config.temperature` (0.07).

4. **Temporal Consistency** - `temporal_consistency_loss`. Pushes adjacent
   window embeddings (`window`, pre-global) toward high cosine similarity;
   minimizes the negative mean cosine over `[B, N-1]` adjacent pairs.

## Training

Implemented in `model/train.py`:

- **Multi-objective** - `compute_loss` runs the four heads and combines them
  with `combine_losses` using the configured weights.
- **Gradient accumulation** - `grad_accum` micro-batches are summed then
  averaged before one optimizer step (`_train_single`).
- **Schedule** - `optax.warmup_cosine_decay_schedule` with
  `warmup` steps then cosine decay; optimizer is AdamW with
  `weight_decay` and global-norm clipping (`max_grad_norm`).
- **Validation** - `evaluate` embeds the validation set and reports
  silhouette score (`sklearn.metrics.silhouette_score`) and leave-one-out
  1-NN accuracy in embedding space (`_nearest_neighbour_accuracy`). Labels
  come from the corruption/campaign splits.
- **Checkpointing** - msgpack via `flax.serialization`; a `best.msgpack`
  (best by validation loss) and periodic `step_*.msgpack` / `last.msgpack`
  are written atomically. `--resume` reloads `best.msgpack`.
- **Multi-GPU** - when `config.distributed` is set and
  `jax.device_count() > 1`, training uses data-parallel `jax.pmap`
  (`_train_pmap`, gradient `pmean` across devices; `grad_accum` forced to
  1).

## See also

- `model/encoder.py` - token embedding, transformer block, objective heads,
  `YggModel`.
- `model/hierarchical.py` - local/global two-stage encoder.
- `model/objectives.py` - the four loss functions and `combine_losses`.
- `model/config.py` - `ModelConfig` (single source of truth).
- `model/train.py` - training loop, validation, checkpointing, pmap.
- `docs/TRACE-MODEL.md` - the event schema these embeddings are built from.
