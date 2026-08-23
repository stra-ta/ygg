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

    denom = jnp.exp(sim_ap_all).sum(-1) + jnp.exp(sim_an_all).sum(-1)
    num = jnp.exp(sim_ap)
    loss = -jnp.log(num / denom.clip(min=1e-8) + 1e-8)
    loss = loss.mean()

    # For row i the correct match is positive j == i (first block).
    combined = jnp.concatenate([sim_ap_all, sim_an_all], axis=1)  # [B, 2B]
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
    # High similarity is good -> minimise its negative.
    return -sim.mean()


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
