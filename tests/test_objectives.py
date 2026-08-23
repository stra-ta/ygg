"""
Tests for the Ygg objective functions in :mod:`model.objectives`.

Run with::

    python -m pytest tests/ -v

All losses are pure JAX functions, so gradients are checked with
:func:`jax.grad` and finiteness is asserted everywhere.
"""

import jax
import jax.numpy as jnp
import pytest

from model.objectives import (
    combine_losses,
    contrastive_loss,
    masked_event_loss,
    next_event_loss,
    temporal_consistency_loss,
)


# --------------------------------------------------------------------------- #
# masked_event_loss
# --------------------------------------------------------------------------- #
def test_masked_event_loss_finite(key):
    logits = jax.random.normal(key, (4, 8, 10))
    labels = jax.random.randint(jax.random.fold_in(key, 1), (4, 8), 0, 10)
    loss = masked_event_loss(logits, labels)
    assert jnp.isfinite(loss)


def test_masked_event_loss_grad_finite(key):
    logits = jax.random.normal(key, (4, 8, 10))
    labels = jax.random.randint(jax.random.fold_in(key, 1), (4, 8), 0, 10)
    grad = jax.grad(lambda x: masked_event_loss(x, labels))(logits)
    assert jnp.all(jnp.isfinite(grad))


def test_masked_event_loss_ignores_masked(key):
    B, S, V = 2, 4, 5
    logits = jax.random.normal(key, (B, S, V))
    labels = jax.random.randint(jax.random.fold_in(key, 1), (B, S), 0, V)
    labels = labels.at[:, -1].set(-1)  # last column masked out

    loss = masked_event_loss(logits, labels)
    # Logits at masked positions must not contribute to the loss.
    logits2 = logits.at[:, -1].mul(1e6)
    loss2 = masked_event_loss(logits2, labels)
    assert jnp.allclose(loss, loss2)

    # Masking an additional valid position does change the set of valid entries.
    labels3 = labels.at[:, -2].set(-1)
    loss3 = masked_event_loss(logits, labels3)
    assert not jnp.allclose(loss, loss3)


def test_masked_event_loss_all_masked(key):
    B, S, V = 3, 5, 7
    logits = jax.random.normal(key, (B, S, V))
    labels = jnp.full((B, S), -1, dtype=jnp.int32)
    loss = masked_event_loss(logits, labels)
    assert jnp.isfinite(loss)
    assert float(loss) == 0.0


def test_masked_event_loss_all_masked_grad(key):
    B, S, V = 3, 5, 7
    logits = jax.random.normal(key, (B, S, V))
    labels = jnp.full((B, S), -1, dtype=jnp.int32)
    grad = jax.grad(lambda x: masked_event_loss(x, labels))(logits)
    assert jnp.all(jnp.isfinite(grad))


# --------------------------------------------------------------------------- #
# next_event_loss
# --------------------------------------------------------------------------- #
def test_next_event_loss_causal_targets(key):
    # labels are the next token after each position (shifted sequence).
    seq = jnp.array([[0, 1, 2, 3, 0], [1, 2, 3, 0, 1]])
    labels = seq[:, 1:]  # [B, S-1]
    B, T, V = seq.shape[0], labels.shape[1], 4
    # Perfect (one-hot) next-token logits should yield ~0 loss.
    logits = jnp.zeros((B, T, V)).at[
        jnp.arange(B)[:, None], jnp.arange(T)[None, :], labels
    ].set(1e3)
    loss = next_event_loss(logits, labels)
    assert jnp.allclose(loss, 0.0, atol=1e-3)


def test_next_event_loss_finite(key):
    logits = jax.random.normal(key, (4, 6, 8))
    labels = jax.random.randint(jax.random.fold_in(key, 1), (4, 6), 0, 8)
    loss = next_event_loss(logits, labels)
    assert jnp.isfinite(loss)


def test_next_event_loss_finite_grad(key):
    logits = jax.random.normal(key, (4, 6, 8))
    labels = jax.random.randint(jax.random.fold_in(key, 1), (4, 6), 0, 8)
    grad = jax.grad(lambda x: next_event_loss(x, labels))(logits)
    assert jnp.all(jnp.isfinite(grad))


def test_next_event_loss_padding_invariance(key):
    B, T, V = 2, 5, 4
    logits = jax.random.normal(key, (B, T, V))
    labels = jax.random.randint(jax.random.fold_in(key, 1), (B, T), 0, V)
    labels = labels.at[:, -1].set(-1)  # padded position
    loss = next_event_loss(logits, labels)
    # Wildly different logits at the padded position must not change the loss.
    logits2 = logits.at[:, -1].mul(1e6)
    loss2 = next_event_loss(logits2, labels)
    assert jnp.allclose(loss, loss2)


# --------------------------------------------------------------------------- #
# contrastive_loss
# --------------------------------------------------------------------------- #
def _orthogonal_to(vec, onto):
    """Return per-row vectors orthogonal to ``onto`` (Gram-Schmidt, unnormalized)."""
    scale = (vec * onto).sum(-1, keepdims=True) / (onto * onto).sum(-1, keepdims=True)
    return vec - scale * onto


def test_contrastive_positive_alignment(key):
    B, d = 8, 16
    z_a = jax.random.normal(key, (B, d))
    z_p = z_a  # identical positive
    z_n = _orthogonal_to(jax.random.normal(jax.random.fold_in(key, 1), (B, d)), z_a)
    loss, acc = contrastive_loss(z_a, z_p, z_n, temperature=0.07)
    assert jnp.isfinite(loss)
    assert float(loss) < 1e-2  # positive clearly favored -> ~0 loss
    assert float(acc) == 1.0


def test_contrastive_negative_separation(key):
    B, d = 8, 16
    z_a = jax.random.normal(key, (B, d))
    z_n = z_a  # negative identical to anchor
    z_p = _orthogonal_to(jax.random.normal(jax.random.fold_in(key, 1), (B, d)), z_a)
    loss, acc = contrastive_loss(z_a, z_p, z_n, temperature=0.07)
    assert jnp.isfinite(loss)
    assert float(loss) > 5.0  # negative closer than positive -> high loss
    assert float(acc) == 0.0


def test_contrastive_logsumexp_stability(key):
    B, d = 4, 8
    # Huge embeddings with a tiny temperature push the raw similarities to ~1e4,
    # which would overflow a naive exp. logsumexp keeps it finite.
    z = jax.random.normal(key, (B, d)) * 1000.0
    z_p = z
    z_n = jax.random.normal(jax.random.fold_in(key, 1), (B, d))
    loss, acc = contrastive_loss(z, z_p, z_n, temperature=1e-4)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(acc)
    # Also stable at the default temperature with large-magnitude embeddings.
    loss2, _ = contrastive_loss(z * 1e3, z * 1e3, z_n * 1e3, temperature=0.07)
    assert jnp.isfinite(loss2)


def test_contrastive_temperature_scaling(key):
    B, d = 8, 16
    z_a = jax.random.normal(key, (B, d))
    # Positive orthogonal, negative nearly identical -> negative currently favored.
    z_p = _orthogonal_to(jax.random.normal(jax.random.fold_in(key, 1), (B, d)), z_a)
    z_n = z_a + 0.01 * jax.random.normal(jax.random.fold_in(key, 2), (B, d))
    loss_tiny = contrastive_loss(z_a, z_p, z_n, temperature=0.01)[0]
    loss_big = contrastive_loss(z_a, z_p, z_n, temperature=1.0)[0]
    # Higher temperature softens the similarities -> lower loss in this regime.
    assert float(loss_big) < float(loss_tiny)
    # The loss is always non-negative.
    assert float(loss_big) >= 0.0


# --------------------------------------------------------------------------- #
# temporal_consistency_loss
# --------------------------------------------------------------------------- #
def test_temporal_identical_windows(key):
    B, N, d = 4, 6, 16
    base = jax.random.normal(jax.random.fold_in(key, 1), (B, 1, d))
    windows = jnp.broadcast_to(base, (B, N, d))  # all adjacent windows identical
    loss = temporal_consistency_loss(windows)
    assert jnp.isfinite(loss)
    assert jnp.allclose(loss, 0.0, atol=1e-4)


def test_temporal_orthogonal_windows(key):
    B, N, d = 4, 6, 16
    # Assign each window a distinct one-hot direction so neighbours are orthogonal.
    oh = jax.nn.one_hot(jnp.arange(N) % d, d)  # [N, d]
    windows = jnp.broadcast_to(oh, (B, N, d))
    loss = temporal_consistency_loss(windows)
    assert jnp.isfinite(loss)
    assert float(loss) > 0.5  # 1 - cos ~ 1 for orthogonal neighbours


def test_temporal_finite_grad(key):
    windows = jax.random.normal(key, (4, 6, 16))
    grad = jax.grad(temporal_consistency_loss)(windows)
    assert jnp.all(jnp.isfinite(grad))


# --------------------------------------------------------------------------- #
# combine_losses
# --------------------------------------------------------------------------- #
def test_combine_losses_weighted_sum():
    losses = {
        "masked": jnp.array(1.0),
        "next": jnp.array(2.0),
        "contrastive": jnp.array(3.0),
    }
    weights = {"masked": 1.0, "next": 0.5, "contrastive": 2.0}
    total = combine_losses(losses, weights)
    expected = 1.0 * 1.0 + 0.5 * 2.0 + 2.0 * 3.0
    assert jnp.allclose(total, expected)


def test_combine_losses_zero_weights():
    losses = {"masked": jnp.array(5.0), "next": jnp.array(7.0)}
    weights = {"masked": 0.0, "next": 0.0}
    total = combine_losses(losses, weights)
    assert jnp.allclose(total, 0.0)


def test_combine_losses_all_zero_embeddings(key):
    B, S, V, d = 2, 4, 5, 8
    logits = jnp.zeros((B, S, V))
    labels = jax.random.randint(jax.random.fold_in(key, 0), (B, S), 0, V)
    m = masked_event_loss(logits, labels)
    n = next_event_loss(logits, labels)

    z = jnp.zeros((B, d))
    c, _ = contrastive_loss(z, z, z)

    windows = jnp.zeros((B, 4, d))
    t = temporal_consistency_loss(windows)

    weights = {"masked": 1.0, "next": 1.0, "contrastive": 1.0, "temporal": 1.0}
    losses = {"masked": m, "next": n, "contrastive": c, "temporal": t}
    total = combine_losses(losses, weights)
    assert jnp.isfinite(total)
