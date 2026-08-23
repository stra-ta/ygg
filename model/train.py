#!/usr/bin/env python3
"""
Ygg Training Entry Point

Trains the execution encoder with self-supervised objectives.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from flax import jax_utils

from model.encoder import (
    EncoderConfig,
    ExecutionEncoder,
    MaskedEventHead,
    NextEventHead,
    ContrastiveModel,
    create_model,
    create_embedder,
    create_contrastive_model,
)
from model.dataset import (
    TraceDataset,
    masked_event_modeling_batch,
    next_event_prediction_batch,
    contrastive_pairs,
)


class TrainState(train_state.TrainState):
    """Extended train state with EMA params."""
    ema_params: Optional[dict] = None
    ema_decay: float = 0.999


def create_train_state(
    rng: jax.random.PRNGKey,
    config: EncoderConfig,
    learning_rate: float,
    weight_decay: float,
) -> TrainState:
    """Initialize model and optimizer."""
    model = create_model(config)
    dummy_tokens = jnp.ones((1, config.max_seq_len, config.token_dim), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, config.max_seq_len), dtype=jnp.float32)

    params = model.init(rng, dummy_tokens, dummy_mask, train=True)["params"]

    tx = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        mask=lambda p: "bias" not in p,  # No weight decay on biases
    )

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def masked_modeling_loss_fn(params, batch, rng, model, head, config):
    """Loss for masked event modeling."""
    masked_batch, labels = masked_event_modeling_batch(batch)
    logits = head.apply({"params": params}, model.apply({"params": params}, masked_batch.tokens, masked_batch.mask, train=True))

    # Flatten for cross-entropy
    logits_flat = logits.reshape(-1, config.vocab_size)
    labels_flat = labels.reshape(-1)

    # Ignore -1 labels (non-masked positions)
    mask = labels_flat != -1
    logits_flat = logits_flat[mask]
    labels_flat = labels_flat[mask]

    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat)
    )
    return loss


def next_event_loss_fn(params, batch, rng, model, head, config):
    """Loss for next-event prediction."""
    input_batch, labels = next_event_prediction_batch(batch)
    logits = head.apply({"params": params}, model.apply({"params": params}, input_batch.tokens, input_batch.mask, train=True))

    logits_flat = logits.reshape(-1, config.vocab_size)
    labels_flat = labels.reshape(-1)

    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat)
    )
    return loss


def contrastive_loss_fn(params, anchor, positive, negative, mask, model, config):
    """Loss for contrastive learning."""
    contrastive_model = create_contrastive_model(config)
    loss, acc = contrastive_model.apply(
        {"params": params},
        anchor.tokens, positive.tokens, negative.tokens,
        mask, train=True,
    )
    return loss, acc


@jax.jit
def train_step_masked(state: TrainState, batch, rng):
    """Single training step for masked event modeling."""
    head = MaskedEventHead(state.config)
    model = ExecutionEncoder(state.config)

    def loss_fn(params):
        return masked_modeling_loss_fn(params, batch, rng, model, head, state.config)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def train_step_next(state: TrainState, batch, rng):
    """Single training step for next-event prediction."""
    head = NextEventHead(state.config)
    model = ExecutionEncoder(state.config)

    def loss_fn(params):
        return next_event_loss_fn(params, batch, rng, model, head, state.config)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def train_step_contrastive(state: TrainState, anchor, positive, negative, mask, rng):
    """Single training step for contrastive learning."""
    model = ExecutionEncoder(state.config)

    def loss_fn(params):
        return contrastive_loss_fn(params, anchor, positive, negative, mask, model, state.config)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, acc), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


def train(
    trace_paths: list,
    config: EncoderConfig,
    objective: str = "masked",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    checkpoint_dir: str = "checkpoints",
    log_every: int = 100,
):
    """Main training loop."""
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, config, learning_rate, weight_decay)
    state = jax_utils.replicate(state)

    dataset = TraceDataset(trace_paths, config.max_seq_len, config.max_seq_len // 2, batch_size)

    step = 0
    for epoch in range(epochs):
        for batch in dataset:
            rng, step_rng = jax.random.split(rng)
            step_rng = jax.random.fold_in(step_rng, step)

            if objective == "masked":
                state, loss = train_step_masked(state, batch, step_rng)
            elif objective == "next":
                state, loss = train_step_next(state, batch, step_rng)
            elif objective == "contrastive":
                # Need contrastive pairs - simplified for now
                continue

            step += 1

            if step % log_every == 0:
                loss = jax_utils.unreplicate(loss)
                print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")

        # Checkpoint
        if checkpoint_dir:
            checkpoints.save_checkpoint(
                checkpoint_dir,
                jax_utils.unreplicate(state),
                step=epoch,
                keep=3,
            )

    return jax_utils.unreplicate(state)


def main():
    parser = argparse.ArgumentParser(description="Train Ygg execution encoder")
    parser.add_argument("traces", nargs="+", help="Paths to Parquet trace files or directories")
    parser.add_argument("--objective", choices=["masked", "next", "contrastive"], default="masked")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--config-out", help="Save config to JSON")

    args = parser.parse_args()

    # Expand directories to trace files
    trace_paths = []
    for p in args.traces:
        path = Path(p)
        if path.is_dir():
            trace_paths.extend(str(f) for f in path.glob("**/*.parquet"))
        else:
            trace_paths.append(str(path))

    print(f"Found {len(trace_paths)} trace files")

    config = EncoderConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
    )

    if args.config_out:
        with open(args.config_out, "w") as f:
            json.dump(config.__dict__, f, indent=2)

    train(
        trace_paths=trace_paths,
        config=config,
        objective=args.objective,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()