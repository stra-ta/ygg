"""
Loss functions for the four Ygg training objectives.

All functions are pure (no module state) so they can be composed freely inside
a single ``jax.value_and_grad`` graph. They expect already-shaped model outputs
(see :class:`model.encoder.YggModel`).
"""

import jax
import jax.numpy as jnp
import optax


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)


def masked_event_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_index: int = -1,
) -> jnp.ndarray:
    """Cross-entropy over masked positions only.

    logits: [B, S, V]
    labels: [B, S]   integer event types; ``ignore_index`` marks unmasked.
    """
    logits_f = logits.reshape(-1, logits.shape[-1])
    labels_f = labels.reshape(-1).astype(jnp.int32)
    valid = labels_f != ignore_index
    if valid.sum() == 0:
        return jnp.array(0.0, dtype=logits.dtype)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_f[valid], labels_f[valid]
    )
    return loss.mean()


def next_event_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    ignore_index: int = -1,
) -> jnp.ndarray:
    """Autoregressive next-event cross-entropy.

    logits: [B, S-1, V]
    labels: [B, S-1]  integer next event types; padding uses ``ignore_index``.
    """
    return masked_event_loss(logits, labels, ignore_index=ignore_index)


def masked_mse(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    ignore_index: float = -1.0,
) -> jnp.ndarray:
    """Mean squared error over masked / valid positions only.

    Used for the continuous V0.2 regression targets (masked/next dt and args).
    ``target`` carries ``ignore_index`` where the position should not contribute.

    pred:   [B, S] or [B, S, C]
    target: same shape, with ``ignore_index`` marking ignored positions.
    """
    if pred.ndim == 3:
        pred_f = pred.reshape(-1, pred.shape[-1])
        target_f = target.reshape(-1, target.shape[-1])
        valid = (target_f != ignore_index).any(axis=-1)
    else:
        pred_f = pred.reshape(-1)
        target_f = target.reshape(-1)
        valid = target_f != ignore_index
    if valid.sum() == 0:
        return jnp.array(0.0, dtype=pred.dtype)
    se = (pred_f[valid] - target_f[valid]) ** 2
    return se.mean()


def contrastive_loss(
    z_anchor: jnp.ndarray,
    z_positive: jnp.ndarray,
    z_negative: jnp.ndarray,
    temperature: float = 0.07,
) -> tuple:
    """InfoNCE contrastive loss with in-batch negatives.

    Each row of ``z_anchor`` has exactly one positive (``z_positive`` at the same
    row) and uses every other positive and every negative in the batch as a
    distractor. Returns ``(loss, accuracy)``.

    Shapes: all [B, d].
    """
    a = _l2_normalize(z_anchor)
    p = _l2_normalize(z_positive)
    n = _l2_normalize(z_negative)
    B = a.shape[0]

    sim_ap = (a * p).sum(-1) / temperature                       # [B]
    sim_ap_all = (a @ p.T) / temperature                         # [B, B]
    sim_an_all = (a @ n.T) / temperature                         # [B, B]

    # Numerically stable InfoNCE via logsumexp over the full distractor set.
    # All logits per row i: [sim(a_i, p_j) for j] ++ [sim(a_i, n_j) for j] -> [B, 2B].
    logits = jnp.concatenate([sim_ap_all, sim_an_all], axis=1)   # [B, 2B]
    labels = jnp.arange(B)                                       # own positive at index i
    # Stable cross-entropy: -logits[label] + logsumexp(logits)
    loss = -logits[jnp.arange(B), labels] + jax.nn.logsumexp(logits, axis=-1)
    loss = loss.mean()

    # For row i the correct match is positive j == i (first block).
    combined = logits
    pred = jnp.argmax(combined, axis=1)
    acc = (pred == jnp.arange(B)).mean()
    return loss, acc


def temporal_consistency_loss(
    window_embeddings: jnp.ndarray,
) -> jnp.ndarray:
    """Push adjacent window embeddings toward high cosine similarity.

    window_embeddings: [B, N, d]
    """
    w = _l2_normalize(window_embeddings)
    a = w[:, :-1, :]
    b = w[:, 1:, :]
    sim = (a * b).sum(-1)  # [B, N-1]
    # High similarity (identical adjacent windows) -> loss 0; orthogonal -> loss 1.
    return (1.0 - sim).mean()


def combine_losses(
    losses: dict,
    weights: dict,
) -> jnp.ndarray:
    """Weighted sum of the enabled objectives, ignoring zero-weight terms."""
    total = jnp.array(0.0)
    for name, loss in losses.items():
        w = weights.get(name, 0.0)
        if w:
            total = total + w * loss
    return total
