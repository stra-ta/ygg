"""
Ygg Transformer Encoder

JAX/Flax implementation of the execution embedding model.

This module owns the token embedding, the (reusable) transformer block, the
prediction heads, and the top-level ``YggModel`` that wires the hierarchical
encoder (see :mod:`model.hierarchical`) to all four objective heads.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Dict, Tuple

from model.config import ModelConfig


# Backwards-compatibility alias used by analysis tooling / DEVELOPMENT.md.
EncoderConfig = ModelConfig


def _pool(x: jnp.ndarray, mask: jnp.ndarray, method: str) -> jnp.ndarray:
    """Masked pooling over the sequence axis (axis=1).

    x:    [B, L, D]
    mask: [B, L]  (1 = real, 0 = padding)
    """
    if method == "max":
        return x.max(axis=1)
    if method == "cls":
        return x[:, 0, :]
    # mean (masked)
    m = mask[:, :, None]
    return (x * m).sum(axis=1) / m.sum(axis=1).clip(min=1.0)


class TokenEmbedding(nn.Module):
    """Compose a token from categorical + continuous event features.

    Input token layout (``config.token_dim`` == 7):
        [0] event_type   - categorical (vocab)
        [1] thread       - categorical (bucketed 256)
        [2] cpu          - categorical (64)
        [3] log_dt       - continuous (log-scaled time delta)
        [4:7] arg0,arg1,arg2 - continuous metrics
    """

    config: ModelConfig

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # [batch, seq_len, 7]
        event_type = tokens[:, :, 0].astype(jnp.int32)
        thread_id = tokens[:, :, 1].astype(jnp.int32)
        cpu_id = tokens[:, :, 2].astype(jnp.int32)
        dt = tokens[:, :, 3:4]                       # [B, S, 1]
        metrics = tokens[:, :, 4:]                   # [B, S, n_metrics]

        event_emb = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            name="event_type_emb",
        )(event_type)
        thread_emb = nn.Embed(
            num_embeddings=256,
            features=self.config.d_model,
            name="thread_emb",
        )(thread_id)
        cpu_emb = nn.Embed(
            num_embeddings=64,
            features=self.config.d_model,
            name="cpu_emb",
        )(cpu_id)
        dt_proj = nn.Dense(self.config.d_model, name="dt_proj")(dt)
        metric_proj = nn.Dense(self.config.d_model, name="metric_proj")(metrics)

        x = event_emb + thread_emb + cpu_emb + dt_proj + metric_proj

        # Learned positional embedding: window-local index so sequences longer
        # than one window still receive a valid (cyclic) position.
        pos = jnp.arange(tokens.shape[1]) % self.config.max_seq_len
        pos_emb = nn.Embed(
            num_embeddings=self.config.max_seq_len,
            features=self.config.d_model,
            name="pos_emb",
        )(pos.astype(jnp.int32))
        x = x + pos_emb

        return nn.Dropout(self.config.dropout)(x, deterministic=not train)


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block (reused by local + global stacks)."""

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attn_mask: Optional[jnp.ndarray] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_heads,
            qkv_features=self.config.d_model,
            out_features=self.config.d_model,
            dropout_rate=self.config.dropout,
            deterministic=not train,
            name="attention",
        )(x, mask=attn_mask)

        x = x + nn.Dropout(self.config.dropout)(attn_out, deterministic=not train)
        x = nn.LayerNorm(name="ln1")(x)

        ff = nn.Dense(self.config.d_ff, name="ff1")(x)
        ff = nn.gelu(ff)
        ff = nn.Dropout(self.config.dropout)(ff, deterministic=not train)
        ff = nn.Dense(self.config.d_model, name="ff2")(ff)

        x = x + nn.Dropout(self.config.dropout)(ff, deterministic=not train)
        x = nn.LayerNorm(name="ln2")(x)
        return x


class MaskedEventHead(nn.Module):
    """Predict the masked event_type at each position."""

    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.config.d_model, name="dense")(x)
        x = nn.gelu(x)
        x = nn.LayerNorm(name="ln")(x)
        return nn.Dense(self.config.vocab_size, name="output")(x)


class NextEventHead(nn.Module):
    """Predict the next event_type distribution at each position."""

    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.config.d_model, name="dense")(x)
        x = nn.gelu(x)
        x = nn.LayerNorm(name="ln")(x)
        return nn.Dense(self.config.vocab_size, name="output")(x)


class ProjectionHead(nn.Module):
    """Non-linear projection used for the contrastive objective."""

    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Dense(self.config.d_model, name="proj1")(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.config.dropout)(x, deterministic=not train)
        return nn.Dense(self.config.d_model, name="proj2")(x)


class YggModel(nn.Module):
    """Full multi-objective model.

    One forward produces everything the four objectives need:

        event      [B, S, d]   contextual per-event embeddings
        window     [B, N, d]   per-window embeddings (local pool)
        window_ctx [B, N, d]   window embeddings after global context
        exec       [B, d]      execution-level embedding
        masked_logits [B, S, V]
        next_logits   [B, S, V]
        proj          [B, d]   projected exec embedding (contrastive)
    """

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        causal: bool = False,
        train: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        from model.hierarchical import HierarchicalEncoder

        enc = HierarchicalEncoder(self.config, name="hier")
        out = enc(tokens, mask=mask, causal=causal, train=train)

        event = out["event"]
        window = out["window"]
        window_ctx = out["window_ctx"]
        exec_emb = out["exec"]

        masked_logits = MaskedEventHead(self.config, name="masked_head")(event)
        next_logits = NextEventHead(self.config, name="next_head")(event)
        proj = ProjectionHead(self.config, name="proj_head")(exec_emb, train=train)

        return {
            "event": event,
            "window": window,
            "window_ctx": window_ctx,
            "exec": exec_emb,
            "masked_logits": masked_logits,
            "next_logits": next_logits,
            "proj": proj,
        }


def create_model(config: ModelConfig) -> YggModel:
    return YggModel(config)


def create_embedder(config: ModelConfig, pool: str = "mean") -> "ExecutionEmbedder":
    return ExecutionEmbedder(config, pool=pool)


class ExecutionEmbedder(nn.Module):
    """Inference-only embedder: returns the execution embedding."""

    config: ModelConfig
    pool: str = "mean"

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        from model.hierarchical import HierarchicalEncoder

        enc = HierarchicalEncoder(self.config, name="hier")
        out = enc(tokens, mask=mask, causal=False, train=train)
        return out["exec"]
