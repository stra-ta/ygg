#!/usr/bin/env python3
"""
train_v02.py - V0.2 multi-objective ablation training + evaluation.

Trains one of five ablation variants of the verified ``model.v02.V02Model`` on
the REAL Norn contention traces, using a run-level (not window-level) train/eval
split so the 3200 correlated windows are never both trained and evaluated.

Variants (see VARIANT_WEIGHTS):
    event             : masked_event only
    event+timing      : + masked_dt, next_dt
    event+args        : + masked_arg, next_arg
    event+timing+args : all of the above (no contrastive)
    event+contrastive : masked_event + contrastive (two-view InfoNCE)

Checkpoints: results/encoder_v02_<variant>.msgpack
Train log:     results/train_v02_<variant>.jsonl
Embeddings:    embed_eval_windows(variant, eval_files) -> (X, regimes, silhouette)
"""
import argparse
import json
import os
import re
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
from model.v02 import V02Model, mask_tokens_v02, next_targets
from model import objectives as O

# Import the existing, verified data pipeline (do NOT modify train_real.py).
from train_real import save_checkpoint
from model.dataset import events_to_tokens

# V0.2 uses sane metric scaling. Real arg values top out around 1e4, so
# metric_scale=1e6 keeps token magnitudes bounded (~1e-2 for valid args) and
# stops sentinel-clipped values (<=1e9) from exploding the per-event embedding
# that the dt/arg/next heads read. METRIC_SCALE=1.0 (used by V0.1 train_real)
# fed those heads 1e9-magnitude inputs and produced init MSE ~4e4.
METRIC_SCALE_V02 = 1e6


def load_tokens_v02(path, seq_len):
    df = pl.read_parquet(path)
    cols = ["timestamp_ns", "cpu", "pid", "tid", "kind", "arg0", "arg1", "arg2"]
    present = [c for c in cols if c in df.columns]
    events = df.select(present).to_numpy()
    if events.shape[1] < 8:
        pad = np.zeros((events.shape[0], 8 - events.shape[1]), dtype=events.dtype)
        events = np.concatenate([events, pad], axis=1)
    return events_to_tokens(events, metric_scale=METRIC_SCALE_V02).astype(np.float32)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

TOKEN_DIM = 7

# Matches: bounded_1x1_run0.parquet -> (regime="bounded", grid="1x1", run=0)
RUN_RE = re.compile(r"^(\w+)_(\d+)x(\d+)_run(\d+)\.parquet$")

# Enabled objective weights per ablation variant. Weights are heuristic:
# timing/arg regression targets are pre-normalised to ~[0,1] and we want them
# comparable in scale to the (already ~unit) masked-event cross-entropy, so we
# up-weight the continuous objectives by 3x; the contrastive term is a weaker
# regulariser at 0.5 (it shares the masked_event backbone).
VARIANT_WEIGHTS = {
    "event": {
        "masked_event": 1.0,
    },
    "event+timing": {
        "masked_event": 1.0,
        "masked_dt": 3.0,
        "next_dt": 3.0,
    },
    "event+args": {
        "masked_event": 1.0,
        "masked_arg": 3.0,
        "next_arg": 3.0,
    },
    "event+timing+args": {
        "masked_event": 1.0,
        "masked_dt": 3.0,
        "masked_arg": 3.0,
        "next_dt": 3.0,
        "next_arg": 3.0,
    },
    "event+contrastive": {
        "masked_event": 1.0,
        "contrastive": 0.5,
    },
}


def make_config() -> ModelConfig:
    """V0.2 ablation model + training config (mirrors train_real.make_config)."""
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
        dropout=0.1,
        max_seq_len=512,
        mask_ratio=0.15,
        mask_token_id=0,
        batch_size=16,
        lr=3e-4,
        warmup=50,
        epochs=12,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        pool="mean",
    )


# ---------------------------------------------------------------------------
# Run-level split (critical: split by RUN, not by window)
# ---------------------------------------------------------------------------
def discover_run_files(data_dir: Path):
    """Group run-indexed parquet files by (regime, grid) -> sorted [(run, path)]."""
    cells = {}
    for p in sorted(data_dir.glob("*_run*.parquet")):
        m = RUN_RE.match(p.name)
        if not m:
            continue
        regime, n = m.group(1), m.group(2)
        grid = f"{n}x{n}"
        run = int(m.group(4))
        cells.setdefault((regime, grid), []).append((run, p))
    return cells


def run_level_split(data_dir: Path):
    """Return (train_files, eval_files, n_runs_total) using a per-cell run split.

    For each (regime, grid) cell: sort its run indices, take the first
    70% (min 1) as train runs, the rest as eval runs. Windows built from a
    train-run file are NEVER evaluated, and vice-versa.
    """
    cells = discover_run_files(data_dir)
    train_files, eval_files = [], []
    n_runs_total = 0
    for (regime, grid), runs in cells.items():
        runs_sorted = sorted(runs, key=lambda x: x[0])
        n_runs = len(runs_sorted)
        n_train = max(1, int(0.7 * n_runs))
        n_runs_total += n_runs
        for _, p in runs_sorted[:n_train]:
            train_files.append(p)
        for _, p in runs_sorted[n_train:]:
            eval_files.append(p)
    return train_files, eval_files, n_runs_total


def regime_of(path) -> str:
    m = RUN_RE.match(Path(path).name)
    if m:
        return m.group(1)
    return Path(path).stem


# ---------------------------------------------------------------------------
# Windowing / batching (consistent with train_real.build_batches)
# ---------------------------------------------------------------------------
def build_batches(traces_tokens, seq_len, batch_size, windows_per_file, rng):
    """Random-window (or pad+mask) batching over a list of per-trace token mats.

    Each trace token matrix is [S, 7]; we draw ``windows_per_file`` windows of
    length ``seq_len`` (random offset if S >= seq_len, else pad+mask). Returns a
    list of {"tokens": np[B,S,7], "mask": np[B,S]} batches (full batches only).
    """
    if seq_len is None:
        raise ValueError("seq_len required")
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
        batches.append({"tokens": toks, "mask": masks})
    return batches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# JIT-safe mirrors of model.objectives' masked losses. The upstream functions
# guard with `if valid.sum() == 0:` (a Python bool on a traced array), which
# breaks jax.jit. These replicate the exact math (mean CE/MSE over valid
# positions) without the traced-bool branch, so the training step stays jitted.
def jevent(logits, labels):
    lf = logits.reshape(-1, logits.shape[-1])
    lf2 = labels.reshape(-1).astype(jnp.int32)
    valid = (lf2 != -1).astype(lf.dtype)
    ce = optax.softmax_cross_entropy_with_integer_labels(lf, lf2)
    denom = valid.sum()
    return (ce * valid).sum() / jnp.maximum(denom, 1.0)


def jmse(pred, target):
    if pred.ndim == 3:
        pf = pred.reshape(-1, pred.shape[-1])
        tf = target.reshape(-1, target.shape[-1])
        valid = (tf != -1.0).any(axis=-1).astype(pf.dtype)
        se = ((pf - tf) ** 2) * valid[:, None]
        denom = valid.sum() * pf.shape[-1]
    else:
        pf = pred.reshape(-1)
        tf = target.reshape(-1)
        valid = (tf != -1.0).astype(pf.dtype)
        se = (pf - tf) ** 2 * valid
        denom = valid.sum()
    return se.sum() / jnp.maximum(denom, 1.0)


def _make_optimizer(config, total_steps):
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=max(1, config.warmup),
        # Mirror train_real: keep the decay tail valid even if total_steps < warmup.
        decay_steps=max(config.warmup + 1, total_steps),
    )
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(sched, weight_decay=config.weight_decay),
    )


def train_variant(variant: str, epochs: int = 12, batch_size: int = 16, windows_per_file: int = 2) -> str:
    if variant not in VARIANT_WEIGHTS:
        raise ValueError(f"unknown variant {variant!r}; known={list(VARIANT_WEIGHTS)}")
    w = VARIANT_WEIGHTS[variant]

    config = make_config()
    config.epochs = epochs
    config.batch_size = batch_size
    seq_len = config.n_windows * config.window_size  # 512
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_files, eval_files, n_runs = run_level_split(HERE)
    print(f"[train_v02:{variant}] {len(train_files)} train files, {len(eval_files)} eval files "
          f"({n_runs} runs total)", flush=True)
    for p in train_files:
        print(f"  train {p.name}", flush=True)

    traces_tokens = [load_tokens_v02(p, seq_len) for p in train_files]

    rng = np.random.default_rng(config.seed)
    jrng = jax.random.PRNGKey(config.seed)
    model = V02Model(config)
    dummy_tokens = jnp.ones((1, seq_len, TOKEN_DIM), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, seq_len), dtype=jnp.float32)
    params = model.init(jrng, dummy_tokens, dummy_mask, causal=True, train=True)["params"]

    batches_per_epoch = max(1, len(train_files) * windows_per_file // batch_size)
    total_steps = max(1, epochs * batches_per_epoch)
    tx = _make_optimizer(config, total_steps)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    log_path = RESULTS_DIR / f"train_v02_{variant}.jsonl"
    enabled = set(w.keys())

    @jax.jit
    def fwd(params, corr, mask, tgt_event, tgt_dt, tgt_arg, nxt_event, nxt_dt, nxt_arg, rng):
        out = model.apply(
            {"params": params}, corr, mask, causal=True, train=True, rngs={"dropout": rng}
        )
        losses = {}
        if "masked_event" in enabled:
            losses["masked_event"] = jevent(out["masked_event"], tgt_event)
        if "masked_dt" in enabled:
            losses["masked_dt"] = jmse(out["masked_dt"][..., 0], tgt_dt)
        if "masked_arg" in enabled:
            losses["masked_arg"] = jmse(out["masked_arg"], tgt_arg)
        if "next_dt" in enabled:
            # Slice to S-1 to align with the model's next_dt[:, :-1] output.
            losses["next_dt"] = jmse(out["next_dt"][:, :-1, 0], nxt_dt[:, :-1])
        if "next_arg" in enabled:
            losses["next_arg"] = jmse(out["next_arg"][:, :-1], nxt_arg[:, :-1])
        total = sum(w[k] * losses[k] for k in enabled)
        return total, losses

    @jax.jit
    def fwd_c(params, corr1, corr2, mask, tgt_event, rng):
        out1 = model.apply(
            {"params": params}, corr1, mask, causal=True, train=True, rngs={"dropout": rng}
        )
        out2 = model.apply(
            {"params": params}, corr2, mask, causal=True, train=True, rngs={"dropout": rng}
        )
        le = jevent(out1["masked_event"], tgt_event)
        lc, _ = O.contrastive_loss(out1["proj"], out2["proj"], out2["proj"])
        total = w["masked_event"] * le + w["contrastive"] * lc
        return total, {"masked_event": le, "contrastive": lc}

    is_contrastive = "contrastive" in enabled

    with open(log_path, "w") as logf:
        step = 0
        for epoch in range(epochs):
            np_rng = np.random.default_rng(config.seed + epoch)
            batches = build_batches(
                traces_tokens, seq_len, batch_size, windows_per_file, np_rng
            )
            for batch in batches:
                tokens_np = np.asarray(batch["tokens"], dtype=np.float32)
                mask_np = np.asarray(batch["mask"], dtype=np.float32)
                mask_j = jnp.asarray(mask_np)
                tgt = next_t = None
                if is_contrastive:
                    jrng, r1, r2, rd = jax.random.split(jrng, 4)
                    corr1, t1 = mask_tokens_v02(tokens_np, mask_np, r1)
                    corr2, _ = mask_tokens_v02(tokens_np, mask_np, r2)
                    tgt_event = jnp.asarray(t1["event"])
                    (loss, losses), grads = jax.value_and_grad(fwd_c, has_aux=True)(
                        state.params,
                        jnp.asarray(corr1),
                        jnp.asarray(corr2),
                        mask_j,
                        tgt_event,
                        rd,
                    )
                else:
                    jrng, rc, rd = jax.random.split(jrng, 3)
                    corr, tgt = mask_tokens_v02(tokens_np, mask_np, rc)
                    next_t = next_targets(tokens_np)
                    (loss, losses), grads = jax.value_and_grad(fwd, has_aux=True)(
                        state.params,
                        jnp.asarray(corr),
                        mask_j,
                        jnp.asarray(tgt["event"]),
                        jnp.asarray(tgt["dt"]),
                        jnp.asarray(tgt["arg"]),
                        jnp.asarray(next_t["event"]),
                        jnp.asarray(next_t["dt"]),
                        jnp.asarray(next_t["arg"]),
                        rd,
                    )
                state = state.apply_gradients(grads=grads)
                step += 1
                rec = {"step": step, "epoch": epoch, "loss": float(loss)}
                rec.update({k: float(v) for k, v in losses.items()})
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if step % 10 == 0 or step == 1:
                    parts = " ".join(f"{k}={float(v):.4f}" for k, v in losses.items())
                    print(f"[{variant}] step {step} epoch {epoch} loss {float(loss):.4f} {parts}",
                          flush=True)

    ckpt_path = str(RESULTS_DIR / f"encoder_v02_{variant}.msgpack")
    save_checkpoint(ckpt_path, state.params, state.opt_state, step, config)
    print(f"[train_v02:{variant}] checkpoint -> {ckpt_path}", flush=True)
    return ckpt_path


# ---------------------------------------------------------------------------
# Checkpoint load + embedding / evaluation
# ---------------------------------------------------------------------------
def _load_params(variant: str, config: ModelConfig):
    path = RESULTS_DIR / f"encoder_v02_{variant}.msgpack"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint for variant {variant!r} at {path}")
    model = V02Model(config)
    seq_len = config.n_windows * config.window_size
    dummy_tokens = jnp.ones((1, seq_len, TOKEN_DIM), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, seq_len), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), dummy_tokens, dummy_mask, causal=False, train=False)["params"]
    # Template for flax deserialization (mirrors save_checkpoint structure).
    sched = optax.warmup_cosine_decay_schedule(0.0, config.lr, 1, 2)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(sched, weight_decay=config.weight_decay),
    )
    opt_state = tx.init(params)
    template = {"params": params, "opt_state": opt_state, "step": 0, "config": ""}
    with open(path, "rb") as f:
        blob = f.read()
    ckpt = fser.from_bytes(template, blob)
    return ckpt["params"]


def _silhouette(X: np.ndarray, regimes: np.ndarray):
    from sklearn.metrics import silhouette_score

    uniq = np.unique(regimes)
    if len(uniq) < 2 or X.shape[0] < 2:
        return float("nan")
    mapping = {r: i for i, r in enumerate(uniq)}
    labels = np.array([mapping[r] for r in regimes])
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")


def embed_eval_windows(variant: str, eval_files):
    """Embed every eval window with the trained variant and score regime separation.

    Returns (X, regimes, silhouette):
        X        : [n_windows*N, d] contextual window embeddings (N = n_windows)
        regimes  : [n_windows*N] string regime per embedding
        silhouette: float (nan if <2 regimes or degenerate)
    """
    config = make_config()
    seq_len = config.n_windows * config.window_size
    params = _load_params(variant, config)
    model = V02Model(config)

    X_parts = []
    reg_parts = []
    for p in eval_files:
        regime = regime_of(p)
        toks = load_tokens_v02(p, seq_len)  # [S, 7]
        S = toks.shape[0]
        windows = []
        if S >= seq_len:
            for start in range(0, S - seq_len + 1, seq_len):
                windows.append(toks[start : start + seq_len])
        else:
            w = np.zeros((seq_len, TOKEN_DIM), dtype=np.float32)
            w[:S] = toks
            windows.append(w)
        if not windows:
            continue
        for i in range(0, len(windows), 64):
            chunk = np.stack(windows[i : i + 64])
            mask = np.ones((chunk.shape[0], seq_len), dtype=np.float32)
            out = model.apply(
                {"params": params},
                jnp.asarray(chunk, dtype=jnp.float32),
                jnp.asarray(mask),
                causal=False,
                train=False,
            )
            wc = np.asarray(out["window_ctx"]).reshape(-1, config.d_model)
            X_parts.append(wc)
            reg_parts.extend([regime] * wc.shape[0])

    if not X_parts:
        return np.zeros((0, config.d_model), dtype=np.float32), np.array([]), float("nan")

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    regimes = np.array(reg_parts)

    # Keep the silhouette fast: sample <=3000 points deterministically.
    if X.shape[0] > 3000:
        srng = np.random.default_rng(42)
        idx = srng.choice(X.shape[0], 3000, replace=False)
        X = X[idx]
        regimes = regimes[idx]

    sil = _silhouette(X, regimes)
    return X, regimes, sil


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train one V0.2 ablation variant on REAL Norn traces")
    parser.add_argument("--variant", required=True, choices=list(VARIANT_WEIGHTS.keys()))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--windows-per-file", type=int, default=2)
    args = parser.parse_args()

    train_variant(
        args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        windows_per_file=args.windows_per_file,
    )


if __name__ == "__main__":
    main()
