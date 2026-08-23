#!/usr/bin/env python3
"""
train_real.py - train the hierarchical encoder on REAL Norn contention traces.

Mirrors ``train_synthetic.py`` exactly in method: the masked-event objective
only, no regime or policy labels. The difference is purely the data source --
here the traces come from a standalone capture program that drives a real Norn
MPMC queue (norn::mpmc_ring) under four backoff regimes and four thread grids,
emitting Ygg application-level events.

The point: prove the same label-free pipeline that separated synthetic regimes
also learns a representation from real Norn traces. We do NOT peek at the
regime labels during training.

Output:
    experiments/contention/real/results/encoder_real.msgpack
    experiments/contention/real/results/train_real_log.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Make the repo root (which owns the `model` package) importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DATA_DIR = HERE  # the 16 <policy>_<grid>x<grid>.parquet live here

TOKEN_DIM = 7
METRIC_SCALE = 1.0  # must match embed_real.py

REGIMES = ["tight", "yield", "bounded", "exponential"]
GRIDS = [1, 2, 4, 8]


def _is_real_trace(p: Path) -> bool:
    """Keep only the canonical <regime>_<N>x<N>.parquet traces."""
    stem = p.stem
    if stem.startswith("_"):
        return False
    parts = stem.split("_")
    if len(parts) != 2:
        return False
    regime, combo = parts
    if regime not in REGIMES:
        return False
    # combo must be of the form "<N>x<N>" (e.g. "2x2", "8x8").
    if "x" not in combo:
        return False
    left, _, right = combo.partition("x")
    return left == right and left.isdigit()


class MaskedOnlyModel(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, tokens, mask, train):
        enc = HierarchicalEncoder(self.config, name="hier")
        out = enc(tokens, mask=mask, causal=False, train=train)
        logits = MaskedEventHead(self.config, name="masked_head")(out["event"])
        return logits


def load_tokens(path: Path, seq_len: int):
    df = pl.read_parquet(path)
    cols = ["timestamp_ns", "cpu", "pid", "tid", "kind", "arg0", "arg1", "arg2"]
    present = [c for c in cols if c in df.columns]
    events = df.select(present).to_numpy()
    if events.shape[1] < 8:
        pad = np.zeros((events.shape[0], 8 - events.shape[1]), dtype=events.dtype)
        events = np.concatenate([events, pad], axis=1)
    return events_to_tokens(events, metric_scale=METRIC_SCALE).astype(np.float32)


def build_batches(traces_tokens, seq_len, batch_size, windows_per_file, rng):
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
    parser = argparse.ArgumentParser(description="Train encoder on REAL Norn contention traces")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--windows-per-file", type=int, default=8)
    args = parser.parse_args()

    config = make_config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size

    seq_len = config.n_windows * config.window_size
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trace_paths = sorted([p for p in DATA_DIR.glob("*.parquet") if _is_real_trace(p)])
    if not trace_paths:
        raise SystemExit(
            f"No real traces found in {DATA_DIR}. Run spill_to_parquet.py first."
        )
    print(f"[train_real] loading {len(trace_paths)} real traces", flush=True)
    for p in trace_paths:
        print(f"  {p.name}", flush=True)
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

    log_path = RESULTS_DIR / "train_real_log.jsonl"
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
        str(RESULTS_DIR / "encoder_real.msgpack"),
        state.params,
        state.opt_state,
        step,
        config,
    )
    print(f"[train_real] done. checkpoint -> {RESULTS_DIR / 'encoder_real.msgpack'}", flush=True)
    print(f"[train_real] log -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
