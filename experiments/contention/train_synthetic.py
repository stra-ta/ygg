#!/usr/bin/env python3
"""
Train the hierarchical encoder on synthetic Norn contention traces.

This is the *training* stage of the V0.1 validation harness. It loads the
synthetic Parquet traces produced by ``generate_synthetic.py`` and trains the
hierarchical encoder with the **masked-event objective only** -- no policy or
regime labels are used. The goal is to prove the pipeline (generate -> train ->
embed -> separate) works end-to-end so that, when real Norn traces arrive, the
same harness can reproduce the actual V0.1 figure.

Outputs:
    experiments/contention/results/encoder.msgpack   (trained params + config)
    experiments/contention/results/train_log.jsonl   (per-step loss log)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the repo root (which owns the `model` package) is importable when this
# file is run as a script: python experiments/contention/train_synthetic.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization as fser
from flax.training import train_state

import polars as pl

from model.config import ModelConfig
from model.hierarchical import HierarchicalEncoder
from model.encoder import MaskedEventHead
from model import objectives as O
from model.dataset import events_to_tokens, _mask_labels

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SYNTHETIC_DIR = Path(__file__).resolve().parent / "synthetic"

# Token layout (must match model.encoder.TokenEmbedding / dataset.events_to_tokens).
TOKEN_DIM = 7
# Keep argument magnitudes meaningful: retry counts / spin iterations / cache
# deltas sit in the tens-to-thousands range, so scale them by 1.0 (not the
# default 1e6) to preserve regime signal for the encoder.
METRIC_SCALE = 1.0


class MaskedOnlyModel(nn.Module):
    """Slim model: hierarchical encoder -> masked-event head (no other heads)."""

    config: ModelConfig

    @nn.compact
    def __call__(self, tokens, mask, train):
        enc = HierarchicalEncoder(self.config, name="hier")
        out = enc(tokens, mask=mask, causal=False, train=train)
        logits = MaskedEventHead(self.config, name="masked_head")(out["event"])
        return logits


def load_tokens(path: Path, seq_len: int):
    """Read a synthetic parquet trace and return its full token matrix."""
    df = pl.read_parquet(path)
    cols = ["timestamp_ns", "cpu", "pid", "tid", "kind", "arg0", "arg1", "arg2"]
    present = [c for c in cols if c in df.columns]
    events = df.select(present).to_numpy()
    if events.shape[1] < 8:
        pad = np.zeros((events.shape[0], 8 - events.shape[1]), dtype=events.dtype)
        events = np.concatenate([events, pad], axis=1)
    return events_to_tokens(events, metric_scale=METRIC_SCALE).astype(np.float32)


def build_batches(traces_tokens, seq_len, batch_size, windows_per_file, rng):
    """Sample fixed-length windows from each trace and collate into batches."""
    windows = []
    for tokens_full in traces_tokens:
        S = tokens_full.shape[0]
        if S == 0:
            continue
        for _ in range(windows_per_file):
            if S >= seq_len:
                start = int(rng.integers(0, S - seq_len + 1))
                w = tokens_full[start : start + seq_len].astype(np.float32)
                mask = np.ones(seq_len, dtype=np.float32)
            else:
                w = np.zeros((seq_len, TOKEN_DIM), dtype=np.float32)
                w[:S] = tokens_full.astype(np.float32)
                mask = np.zeros(seq_len, dtype=np.float32)
                mask[:S] = 1.0
            windows.append((w, mask))

    rng.shuffle(windows)
    batches = []
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        if len(chunk) < batch_size:
            continue
        toks = np.stack([c[0] for c in chunk])
        masks = np.stack([c[1] for c in chunk])
        masked_toks, masked_labels = _mask_labels(toks, masks, 0.15, 0, rng)
        batches.append(
            {
                "tokens": jnp.asarray(toks),
                "mask": jnp.asarray(masks),
                "masked_tokens": jnp.asarray(masked_toks),
                "masked_labels": jnp.asarray(masked_labels),
            }
        )
    return batches


def save_checkpoint(path: str, params, opt_state, step: int, config: ModelConfig) -> None:
    # Store config as a JSON *string*: flax.serialization restores nested dicts
    # against an empty-template structure and would otherwise drop the fields.
    blob = fser.to_bytes(
        {
            "params": params,
            "opt_state": opt_state,
            "step": step,
            "config": json.dumps(config.to_dict()),
        }
    )
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path)


def make_config() -> ModelConfig:
    """Small settings for the synthetic validation run (masked objective only)."""
    return ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=8,
        d_ff=256,
        local_layers=2,
        global_layers=2,
        window_size=128,
        n_windows=4,
        vocab_size=10000,
        token_dim=7,
        masked_weight=1.0,
        next_weight=0.0,
        contrastive_weight=0.0,
        temporal_weight=0.0,
        mask_ratio=0.15,
        mask_token_id=0,
        batch_size=16,
        lr=3e-4,
        warmup=50,
        epochs=20,
        weight_decay=0.01,
        max_grad_norm=1.0,
        dropout=0.1,
        max_seq_len=512,
        seed=42,
        pool="mean",
    )


def loss_fn(params, batch, rng, config):
    model = MaskedOnlyModel(config)
    logits = model.apply(
        {"params": params},
        batch["masked_tokens"],
        batch["mask"],
        train=True,
        rngs={"dropout": rng},
    )
    m_loss = O.masked_event_loss(logits, batch["masked_labels"])
    return m_loss, {"masked_loss": m_loss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train encoder on synthetic contention traces")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--windows-per-file", type=int, default=8)
    args = parser.parse_args()

    config = make_config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size

    seq_len = config.n_windows * config.window_size
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trace_paths = sorted(SYNTHETIC_DIR.glob("*.parquet"))
    if not trace_paths:
        raise SystemExit(
            f"No synthetic traces found in {SYNTHETIC_DIR}. "
            "Run generate_synthetic.py first."
        )
    print(f"[train] loading {len(trace_paths)} synthetic traces", flush=True)
    traces_tokens = [load_tokens(p, seq_len) for p in trace_paths]

    rng = np.random.default_rng(config.seed)
    jrng = jax.random.PRNGKey(config.seed)

    model = MaskedOnlyModel(config)
    dummy_tokens = jnp.ones((1, seq_len, TOKEN_DIM), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, seq_len), dtype=jnp.float32)
    params = model.init(jrng, dummy_tokens, dummy_mask, train=True)["params"]

    batches_per_epoch = max(1, len(trace_paths) * args.windows_per_file // args.batch_size)
    total_steps = max(1, config.epochs * batches_per_epoch)
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=max(1, config.warmup),
        decay_steps=max(config.warmup + 1, total_steps),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(sched, weight_decay=config.weight_decay),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    log_path = RESULTS_DIR / "train_log.jsonl"
    with open(log_path, "w") as logf:
        step = 0
        for epoch in range(config.epochs):
            np_rng = np.random.default_rng(config.seed + epoch)
            batches = build_batches(
                traces_tokens, seq_len, args.batch_size, args.windows_per_file, np_rng
            )
            for batch in batches:
                jrng, dr = jax.random.split(jrng)
                (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    state.params, batch, dr, config
                )
                state = state.apply_gradients(grads=grads)
                step += 1
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss),
                    "masked_loss": float(metrics["masked_loss"]),
                }
                logf.write(json.dumps(record) + "\n")
                logf.flush()
                if step % 10 == 0 or step == 1:
                    print(
                        f"step {step} epoch {epoch} loss {float(loss):.4f}",
                        flush=True,
                    )

    save_checkpoint(
        str(RESULTS_DIR / "encoder.msgpack"),
        state.params,
        state.opt_state,
        step,
        config,
    )
    print(f"[train] done. checkpoint -> {RESULTS_DIR / 'encoder.msgpack'}", flush=True)
    print(f"[train] log -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
