#!/usr/bin/env python3
"""
Ygg training entry point.

Trains the hierarchical execution encoder against four self-supervised
objectives with configurable weights:

    1. Masked Event Modeling   (reconstruct masked event types)
    2. Next-Event Prediction   (autoregressive event-type distribution)
    3. Contrastive             (InfoNCE: healthy-healthy vs healthy-corrupted)
    4. Temporal Consistency    (adjacent windows should be similar)

Features: gradient accumulation, warmup + cosine LR, validation (silhouette +
nearest-neighbour accuracy), best-by-validation-loss checkpointing, JSONL and
optional WandB logging, and data-parallel ``jax.pmap`` for multi-GPU ("Vanta").
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax import serialization as fser
import jax.tree_util

from model.config import ModelConfig
from model.encoder import YggModel
from model import objectives as O
from model.dataset import StreamingTraceDataset, split_campaign


# ---------------------------------------------------------------------------
# Checkpointing (msgpack via flax.serialization; avoids the TensorFlow-dependent
# flax.training.checkpoints import).
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, params, opt_state, step: int, config: ModelConfig) -> None:
    blob = fser.to_bytes(
        {
            "params": params,
            "opt_state": opt_state,
            "step": step,
            "config": config.to_dict(),
        }
    )
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path)


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        blob = f.read()
    target = {"params": None, "opt_state": None, "step": 0, "config": {}}
    return fser.from_bytes(target, blob)


def maybe_resume(state, checkpoint_dir: str):
    best = os.path.join(checkpoint_dir, "best.msgpack")
    if os.path.exists(best):
        print(f"[checkpoint] resuming from {best}")
        ck = load_checkpoint(best)
        return state.replace(params=ck["params"], opt_state=ck["opt_state"]), int(ck["step"])
    return state, 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _nearest_neighbour_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out 1-NN accuracy in embedding space."""
    if X.shape[0] < 2 or len(np.unique(y)) < 2:
        return float("nan")
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    pred = np.argmax(sim, axis=1)
    return float(np.mean(y[pred] == y))


def evaluate(params, val_dataset: StreamingTraceDataset, config: ModelConfig, rng, max_batches: int = 64):
    """Embed the validation set and report embedding-quality + loss metrics."""
    from sklearn.metrics import silhouette_score

    model = YggModel(config)
    Xs, ys = [], []
    losses = []
    for i, batch in enumerate(val_dataset):
        if i >= max_batches:
            break
        rng, er = jax.random.split(rng)
        out = model.apply({"params": params}, batch["tokens"], batch["mask"], causal=False, train=False)
        Xs.append(np.asarray(out["exec"]))
        ys.append(np.asarray(batch["corrupt"]))
        l, _ = compute_loss(params, batch, er, config)
        losses.append(float(l))
    if not Xs:
        return {"val_loss": float("nan"), "silhouette": float("nan"), "nn_accuracy": float("nan")}
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)

    nn_acc = _nearest_neighbour_accuracy(X, y)
    try:
        sil = float(silhouette_score(X, y)) if len(np.unique(y)) > 1 and X.shape[0] > 2 else float("nan")
    except Exception:
        sil = float("nan")
    return {"val_loss": float(np.mean(losses)), "silhouette": sil, "nn_accuracy": nn_acc}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(params, batch, rng, config: ModelConfig):
    """Returns (total_loss, metrics_dict).

    Metrics stay as JAX arrays so this is safe inside ``value_and_grad``; the
    training loop converts them to Python floats after the fact.
    """
    weights = {
        "masked": config.masked_weight,
        "next": config.next_weight,
        "contrastive": config.contrastive_weight,
        "temporal": config.temporal_weight,
    }
    metrics: Dict[str, Any] = {}
    losses: Dict[str, jnp.ndarray] = {}

    k0, k1, k2, k3, k4 = jax.random.split(rng, 5)

    # 1. Masked event modeling (bidirectional context on the masked sequence).
    out_m = YggModel(config).apply(
        {"params": params},
        batch["masked_tokens"], batch["mask"], causal=False, train=True,
        rngs={"dropout": k0},
    )
    m_loss = O.masked_event_loss(out_m["masked_logits"], batch["masked_labels"])
    losses["masked"] = m_loss
    metrics["masked_loss"] = m_loss

    # Reuse the original-sequence forward for temporal + contrastive anchor.
    out = YggModel(config).apply(
        {"params": params},
        batch["tokens"], batch["mask"], causal=False, train=True,
        rngs={"dropout": k1},
    )
    t_loss = O.temporal_consistency_loss(out["window"])
    losses["temporal"] = t_loss
    metrics["temporal_loss"] = t_loss

    a_proj = out["proj"]
    p_proj = YggModel(config).apply(
        {"params": params},
        batch["pos_tokens"], batch["pos_mask"], causal=False, train=True,
        rngs={"dropout": k2},
    )["proj"]
    n_proj = YggModel(config).apply(
        {"params": params},
        batch["neg_tokens"], batch["neg_mask"], causal=False, train=True,
        rngs={"dropout": k3},
    )["proj"]
    c_loss, c_acc = O.contrastive_loss(a_proj, p_proj, n_proj, config.temperature)
    losses["contrastive"] = c_loss
    metrics["contrastive_loss"] = c_loss
    metrics["contrastive_acc"] = c_acc

    # 4. Next-event prediction (causal context).
    if config.next_weight:
        out_n = YggModel(config).apply(
            {"params": params},
            batch["tokens"], batch["mask"], causal=True, train=True,
            rngs={"dropout": k4},
        )
        n_loss = O.next_event_loss(out_n["next_logits"], batch["next_labels"])
        losses["next"] = n_loss
        metrics["next_loss"] = n_loss

    total = O.combine_losses(losses, weights)
    metrics["loss"] = total
    return total, metrics


# ---------------------------------------------------------------------------
# Optimizer / state
# ---------------------------------------------------------------------------

def make_tx(config: ModelConfig, total_steps: int):
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=max(1, config.warmup),
        decay_steps=max(config.warmup + 1, total_steps),
    )
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(sched, weight_decay=config.weight_decay),
    )


def create_train_state(rng, config: ModelConfig, total_steps: int, seq_len: int):
    model = YggModel(config)
    dummy_tokens = jnp.ones((1, seq_len, config.token_dim), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, seq_len), dtype=jnp.float32)
    params = model.init(rng, dummy_tokens, dummy_mask, causal=False, train=True)["params"]
    tx = make_tx(config, total_steps)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# __APPEND_B__


# ---------------------------------------------------------------------------
# Single-device step (with gradient accumulation)
# ---------------------------------------------------------------------------

def compute_grads(params, batch, rng, config):
    (loss, metrics), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        params, batch, rng, config
    )
    return loss, metrics, grads


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401

        return True
    except Exception:
        return False


class Logger:
    def __init__(self, config: ModelConfig, use_wandb: bool):
        self.use_wandb = use_wandb and _wandb_available()
        os.makedirs(config.log_dir, exist_ok=True)
        self.jsonl = os.path.join(config.log_dir, f"{config.run_name}.jsonl")
        config.to_json(os.path.join(config.log_dir, f"{config.run_name}.config.json"))
        if self.use_wandb:
            import wandb

            wandb.init(project="ygg", name=config.run_name, config=config.to_dict())

    def log(self, record: Dict[str, Any]):
        clean = {}
        for k, v in record.items():
            if isinstance(v, (jnp.ndarray, np.ndarray)):
                clean[k] = float(v) if v.size == 1 else v.tolist()
            elif isinstance(v, (np.floating, float)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                clean[k] = int(v)
            else:
                clean[k] = v
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(clean) + "\n")
        if self.use_wandb:
            import wandb

            wandb.log(clean)


# ---------------------------------------------------------------------------
# Main training loop (single device / host)
# ---------------------------------------------------------------------------

def train(config: ModelConfig, trace_paths: List[str], corrupt_paths: List[str], resume: bool):
    seq_len = config.n_windows * config.window_size
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    healthy = list(trace_paths)
    corrupt = list(corrupt_paths)
    if not corrupt:
        for p in trace_paths:
            if os.path.isdir(p):
                h, c = split_campaign(p)
                if c:
                    healthy, corrupt = h, c
                    break

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    total_steps = config.total_steps or max(1, config.epochs * max(1, len(healthy) * 4))

    state = create_train_state(init_rng, config, total_steps, seq_len)
    start_step = 0
    if resume:
        state, start_step = maybe_resume(state, config.checkpoint_dir)

    n_devices = jax.device_count()
    use_pmap = config.distributed and n_devices > 1
    if use_pmap:
        print(f"[train] data-parallel pmap over {n_devices} devices")
        state = jax.device_put_replicated(state, jax.devices())
    else:
        print(f"[train] single-device ({jax.default_backend()})")

    logger = Logger(config, config.use_wandb)
    best_val = float("inf")

    train_ds = StreamingTraceDataset(
        healthy_paths=healthy,
        corrupt_paths=corrupt,
        seq_len=seq_len,
        batch_size=config.batch_size,
        mask_ratio=config.mask_ratio,
        seed=config.seed,
        windows_per_file=256,
        epochs=config.epochs,
    )
    val_ds = StreamingTraceDataset(
        healthy_paths=healthy,
        corrupt_paths=corrupt,
        seq_len=seq_len,
        batch_size=config.batch_size,
        seed=config.seed + 12345,
        windows_per_file=32,
        epochs=1,
    )

    if use_pmap:
        _train_pmap(state, train_ds, config, logger, rng, total_steps, best_val, val_ds, start_step, n_devices)
    else:
        _train_single(state, train_ds, config, logger, rng, total_steps, best_val, val_ds, start_step)


def _train_single(state, train_ds, config, logger, rng, total_steps, best_val, val_ds, start_step):
    step = start_step
    agg_grads = None
    agg_metrics: Dict[str, float] = {}
    agg_count = 0
    t0 = time.time()

    for epoch in range(config.epochs):
        for batch in train_ds:
            rng, sr = jax.random.split(rng)
            loss, metrics, grads = compute_grads(state.params, batch, sr, config)

            if agg_grads is None:
                agg_grads = grads
                agg_metrics = dict(metrics)
            else:
                agg_grads = jax.tree_util.tree_map(lambda a, b: a + b, agg_grads, grads)
                agg_metrics = {k: agg_metrics[k] + metrics[k] for k in metrics}
            agg_count += 1

            if agg_count >= config.grad_accum:
                avg_grads = jax.tree_util.tree_map(lambda g: g / agg_count, agg_grads)
                state = state.apply_gradients(grads=avg_grads)
                avg_metrics = {k: v / agg_count for k, v in agg_metrics.items()}
                step += 1

                if step % config.log_every == 0:
                    avg_metrics["step"] = step
                    avg_metrics["epoch"] = epoch
                    avg_metrics["sec_per_step"] = (time.time() - t0) / max(1, config.log_every)
                    logger.log(avg_metrics)
                    print(
                        f"step {step} loss {avg_metrics['loss']:.4f} "
                        f"mask {avg_metrics.get('masked_loss', 0):.4f} "
                        f"next {avg_metrics.get('next_loss', 0):.4f} "
                        f"contr {avg_metrics.get('contrastive_loss', 0):.4f} "
                        f"temp {avg_metrics.get('temporal_loss', 0):.4f}"
                    )
                    t0 = time.time()

                if step % config.eval_every == 0:
                    rng, vr = jax.random.split(rng)
                    val = evaluate(state.params, val_ds, config, vr)
                    val["step"] = step
                    val["epoch"] = epoch
                    logger.log(val)
                    print(f"[eval] step {step} {val}")
                    if val["val_loss"] < best_val:
                        best_val = val["val_loss"]
                        save_checkpoint(
                            os.path.join(config.checkpoint_dir, "best.msgpack"),
                            state.params, state.opt_state, step, config,
                        )
                        print(f"[checkpoint] new best val_loss {best_val:.4f}")

                if step % config.save_every == 0:
                    save_checkpoint(
                        os.path.join(config.checkpoint_dir, f"step_{step}.msgpack"),
                        state.params, state.opt_state, step, config,
                    )

                agg_grads, agg_metrics, agg_count = None, {}, 0

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    save_checkpoint(
        os.path.join(config.checkpoint_dir, "last.msgpack"),
        state.params, state.opt_state, step, config,
    )
    print("[train] done.")


# __APPEND_C__


def _train_pmap(state, train_ds, config, logger, rng, total_steps, best_val, val_ds, start_step, n_devices):
    """Data-parallel pmap path (grad_accum forced to 1, best-by-val tracked)."""
    step = start_step

    def pmap_step(state, batch, rng):
        (loss, metrics), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            state.params, batch, rng, config
        )
        grads = jax.lax.pmean(grads, axis_name="data")
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics

    pmap_step = jax.pmap(pmap_step, axis_name="data")

    def shard(batch):
        out = {}
        for k, v in batch.items():
            v = np.asarray(v)
            out[k] = jnp.array(v.reshape(n_devices, -1, *v.shape[1:]))
        return out

    for epoch in range(config.epochs):
        for batch in train_ds:
            rng, sr = jax.random.split(rng)
            sr = jax.random.split(sr, n_devices)
            sharded = shard(batch)
            state, loss, metrics = pmap_step(state, sharded, sr)
            loss = float(jnp.mean(loss))
            metrics = {k: float(jnp.mean(v)) for k, v in metrics.items()}
            step += 1
            if step % config.log_every == 0:
                metrics["step"] = step
                metrics["epoch"] = epoch
                logger.log(metrics)
                print(f"step {step} loss {metrics['loss']:.4f}")
            if step % config.eval_every == 0:
                params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
                val = evaluate(params, val_ds, config, rng)
                val["step"] = step
                val["epoch"] = epoch
                logger.log(val)
                print(f"[eval] step {step} {val}")
                if val["loss"] < best_val:
                    best_val = val["loss"]
                    save_checkpoint(
                        os.path.join(config.checkpoint_dir, "best.msgpack"),
                        params, state.opt_state, step, config,
                    )
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
    save_checkpoint(
        os.path.join(config.checkpoint_dir, "last.msgpack"),
        params, state.opt_state, step, config,
    )
    print("[train] done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_config(args) -> ModelConfig:
    if args.config:
        config = ModelConfig.from_json(args.config)
    else:
        config = ModelConfig()

    # Apply CLI overrides for any field that was explicitly set.
    overrides = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "n_windows": args.n_windows,
        "window_size": args.window_size,
        "local_layers": args.local_layers,
        "global_layers": args.global_layers,
        "vocab_size": args.vocab_size,
        "max_seq_len": args.max_seq_len,
        "dropout": args.dropout,
        "masked_weight": args.masked_weight,
        "next_weight": args.next_weight,
        "contrastive_weight": args.contrastive_weight,
        "temporal_weight": args.temporal_weight,
        "mask_ratio": args.mask_ratio,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "warmup": args.warmup,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "log_every": args.log_every,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "use_wandb": args.wandb,
        "distributed": args.distributed,
        "run_name": args.run_name,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
    }
    config = config.from_dict({**config.to_dict(), **{k: v for k, v in overrides.items() if v is not None}})
    return config


def main():
    p = argparse.ArgumentParser(description="Train the Ygg hierarchical execution encoder")
    p.add_argument("traces", nargs="+", help="Parquet trace files or Kiln campaign directories")
    p.add_argument("--corrupt", nargs="*", default=[], help="Corrupted/negative trace files or dirs")
    p.add_argument("--config", help="Path to a ModelConfig JSON to load")
    p.add_argument("--resume", action="store_true", help="Resume from best checkpoint")

    # Model / hierarchy
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--n-windows", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--local-layers", type=int, default=None)
    p.add_argument("--global-layers", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)

    # Objective weights / hyperparameters
    p.add_argument("--masked-weight", type=float, default=None)
    p.add_argument("--next-weight", type=float, default=None)
    p.add_argument("--contrastive-weight", type=float, default=None)
    p.add_argument("--temporal-weight", type=float, default=None)
    p.add_argument("--mask-ratio", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)

    # Training
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)

    # Logging / distributed
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--distributed", action="store_true", help="Data-parallel pmap over GPUs")
    p.add_argument("--run-name", default="ygg-run")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--config-out", help="Write the resolved config to this JSON path")

    args = p.parse_args()
    config = _build_config(args)

    if args.config_out:
        config.to_json(args.config_out)

    print("Resolved config:")
    print(str(config))

    train(config, args.traces, args.corrupt, args.resume)


if __name__ == "__main__":
    main()


