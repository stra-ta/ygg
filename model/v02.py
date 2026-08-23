"""
V0.2 multi-objective model.

Extends the V0.1 masked-event encoder with the objectives the real-Norn study
showed were missing:

    masked event type      (classification, already in V0.1)
    masked dt  (log-scaled time delta)   -> regression
    masked args (arg0/1/2)               -> regression
    next  event type                     -> classification
    next  dt                             -> regression
    next  args                           -> regression
    contrastive execution embedding      -> InfoNCE (two-view)

The encoder trunk (``HierarchicalEncoder``) is shared with V0.1, so the
``hier`` subtree in a V0.2 checkpoint is structurally identical and can be
reused by :class:`model.encoder.ExecutionEmbedder` / ``WindowEmbedder``.

All predictive heads read from the per-event contextual embedding ``event``
(``[B, S, d]``) produced by the local encoder, so a single forward pass
supplies every objective.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from model.config import ModelConfig
from model.encoder import MaskedEventHead, NextEventHead, ProjectionHead
from model.hierarchical import HierarchicalEncoder

# Normalisation constants for the continuous regression targets.
# log_dt in events_to_tokens is clipped to [1, 1e9] then log() -> roughly [0, 20.7].
DT_NORM = 20.0
# Raw arg values are clipped to [0, 1e9] in events_to_tokens; valid args top out
# around 1e4, so clip targets to that and scale to [0, 1] for stable MSE.
ARG_NORM = 1e4


class V02Model(nn.Module):
    """Multi-objective execution encoder for the V0.2 ablation."""

    config: ModelConfig

    @nn.compact
    def __call__(self, tokens, mask=None, causal=True, train=False):
        enc = HierarchicalEncoder(self.config, name="hier")
        out = enc(tokens, mask=mask, causal=causal, train=train)

        event = out["event"]          # [B, S, d]
        window_ctx = out["window_ctx"]  # [B, N, d]
        exec_emb = out["exec"]        # [B, d]

        masked_event = MaskedEventHead(self.config, name="masked_head")(event)
        next_event = NextEventHead(self.config, name="next_head")(event)
        masked_dt = nn.Dense(1, name="masked_dt_head")(event)       # [B, S, 1]
        masked_arg = nn.Dense(3, name="masked_arg_head")(event)    # [B, S, 3]
        next_dt = nn.Dense(1, name="next_dt_head")(event)
        next_arg = nn.Dense(3, name="next_arg_head")(event)
        proj = ProjectionHead(self.config, name="proj_head")(exec_emb, train=train)

        return {
            "event": event,
            "window_ctx": window_ctx,
            "exec": exec_emb,
            "masked_event": masked_event,
            "masked_dt": masked_dt,
            "masked_arg": masked_arg,
            "next_event": next_event,
            "next_dt": next_dt,
            "next_arg": next_arg,
            "proj": proj,
        }


def mask_tokens_v02(tokens, mask, rng, mask_ratio=0.15, mask_token_id=0):
    """NumPy BERT-style corruption across every predictive token dimension.

    Returns ``(corrupted_tokens, targets)`` where ``targets`` holds the true
    values at masked positions and ``-1`` elsewhere so the regression /
    classification losses can ignore unmasked positions.

    tokens: [B, S, 7]  (event_type, thread, cpu, log_dt, arg0, arg1, arg2)
    mask:   [B, S]     1 = real event, 0 = padding
    """
    tokens = np.asarray(tokens, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    B, S, _ = tokens.shape

    rng_slot, rng_t, rng_a = jax.random.split(rng, 3)
    sel = jax.random.bernoulli(rng_slot, mask_ratio, shape=(B, S))
    sel = np.asarray(sel) & (mask > 0)
    sel3 = sel[:, :, None]

    corr = tokens.copy()
    corr[..., 0] = np.where(sel, mask_token_id, tokens[..., 0])
    corr[..., 3] = np.where(sel, 0.0, tokens[..., 3])
    corr[..., 4:7] = np.where(sel3, 0.0, tokens[..., 4:7])

    event_t = np.where(sel, tokens[..., 0].astype(np.int32), -1)
    dt_true = np.clip(tokens[..., 3], 0.0, DT_NORM) / DT_NORM
    dt_t = np.where(sel, dt_true, -1.0)
    arg_true = np.clip(tokens[..., 4:7], 0.0, ARG_NORM) / ARG_NORM
    arg_t = np.where(sel3, arg_true, -1.0)

    targets = {"event": event_t, "dt": dt_t, "arg": arg_t}
    return corr, targets


def next_targets(tokens):
    """Shifted targets for the autoregressive next-* objectives.

    For position ``i`` the target is the value at position ``i + 1``; the final
    position is ``-1`` (ignored).
    """
    tokens = np.asarray(tokens, dtype=np.float32)
    B, S, _ = tokens.shape
    mask_id = -1

    event_t = np.full((B, S), mask_id, dtype=np.int32)
    event_t[:, :-1] = tokens[:, 1:, 0].astype(np.int32)

    dt_true = np.clip(tokens[:, 1:, 3], 0.0, DT_NORM) / DT_NORM
    dt_t = np.full((B, S), -1.0, dtype=np.float32)
    dt_t[:, :-1] = dt_true

    arg_true = np.clip(tokens[:, 1:, 4:7], 0.0, ARG_NORM) / ARG_NORM
    arg_t = np.full((B, S, 3), -1.0, dtype=np.float32)
    arg_t[:, :-1, :] = arg_true

    return {"event": event_t, "dt": dt_t, "arg": arg_t}
