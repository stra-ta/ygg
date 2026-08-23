"""
Ygg Transformer Encoder

JAX/Flax implementation of the execution embedding model.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    vocab_size: int = 10000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    token_dim: int = 5  # Input token dimension (see dataset.py)


class TokenEmbedding(nn.Module):
    """Compose token from categorical + continuous features."""
    config: EncoderConfig

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """
        tokens: [batch, seq_len, token_dim]
        Returns: [batch, seq_len, d_model]
        """
        # Event type embedding
        event_type = tokens[:, :, 0].astype(jnp.int32)
        event_emb = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            name="event_type_emb",
        )(event_type)

        # Thread embedding (bucketed)
        thread_id = tokens[:, :, 1].astype(jnp.int32)
        thread_emb = nn.Embed(
            num_embeddings=256,
            features=self.config.d_model,
            name="thread_emb",
        )(thread_id)

        # CPU embedding
        cpu_id = tokens[:, :, 2].astype(jnp.int32)
        cpu_emb = nn.Embed(
            num_embeddings=64,
            features=self.config.d_model,
            name="cpu_emb",
        )(cpu_id)

        # Time delta projection
        dt = tokens[:, :, 3:4]  # [batch, seq_len, 1]
        dt_proj = nn.Dense(self.config.d_model, name="dt_proj")(dt)

        # Metric projection (arg0, can extend to arg1, arg2)
        metrics = tokens[:, :, 4:]  # [batch, seq_len, n_metrics]
        metric_proj = nn.Dense(self.config.d_model, name="metric_proj")(metrics)

        # Sum all embeddings
        x = event_emb + thread_emb + cpu_emb + dt_proj + metric_proj

        # Positional encoding
        pos = jnp.arange(self.config.max_seq_len)[None, :, None]
        pos_emb = nn.Embed(
            num_embeddings=self.config.max_seq_len,
            features=self.config.d_model,
            name="pos_emb",
        )(pos.astype(jnp.int32))
        x = x + pos_emb

        return nn.Dropout(self.config.dropout)(x, deterministic=not self.is_mutable_collection('params'))


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    config: EncoderConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # Self-attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_heads,
            qkv_features=self.config.d_model,
            out_features=self.config.d_model,
            dropout_rate=self.config.dropout,
            deterministic=deterministic,
            name="attention",
        )(x, mask=mask)

        x = x + nn.Dropout(self.config.dropout)(attn_out, deterministic=deterministic)
        x = nn.LayerNorm(name="ln1")(x)

        # Feed-forward
        ff_out = nn.Dense(self.config.d_ff, name="ff1")(x)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dropout(self.config.dropout)(ff_out, deterministic=deterministic)
        ff_out = nn.Dense(self.config.d_model, name="ff2")(ff_out)

        x = x + nn.Dropout(self.config.dropout)(ff_out, deterministic=deterministic)
        x = nn.LayerNorm(name="ln2")(x)

        return x


class ExecutionEncoder(nn.Module):
    """Transformer encoder for execution traces."""
    config: EncoderConfig

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        """
        tokens: [batch, seq_len, token_dim]
        mask: [batch, seq_len] - 1 for real, 0 for padding
        Returns: [batch, seq_len, d_model] - contextualized token embeddings
        """
        x = TokenEmbedding(self.config)(tokens)

        # Convert mask to attention mask
        if mask is not None:
            # [batch, 1, 1, seq_len] for broadcasting
            attn_mask = mask[:, None, None, :].astype(jnp.bool_)
        else:
            attn_mask = None

        for i in range(self.config.n_layers):
            x = TransformerBlock(self.config, name=f"layer_{i}")(
                x, mask=attn_mask, deterministic=not train
            )

        return x


class MaskedEventHead(nn.Module):
    """Prediction head for masked event modeling."""
    config: EncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [batch, seq_len, d_model] -> logits [batch, seq_len, vocab_size]"""
        x = nn.Dense(self.config.d_model, name="dense")(x)
        x = nn.gelu(x)
        x = nn.LayerNorm(name="ln")(x)
        logits = nn.Dense(self.config.vocab_size, name="output")(x)
        return logits


class NextEventHead(nn.Module):
    """Prediction head for next-event prediction."""
    config: EncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [batch, seq_len, d_model] -> logits [batch, seq_len, vocab_size]"""
        x = nn.Dense(self.config.d_model, name="dense")(x)
        x = nn.gelu(x)
        x = nn.LayerNorm(name="ln")(x)
        logits = nn.Dense(self.config.vocab_size, name="output")(x)
        return logits


class ExecutionEmbedder(nn.Module):
    """Full model: encoder + pooling for execution-level embedding."""
    config: EncoderConfig
    pool: str = "mean"  # "mean", "cls", "max"

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        """
        Returns: [batch, d_model] - execution embedding
        """
        x = ExecutionEncoder(self.config)(tokens, mask, train=train)

        if self.pool == "mean":
            if mask is not None:
                mask = mask[:, :, None]
                x = (x * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
            else:
                x = x.mean(axis=1)
        elif self.pool == "max":
            x = x.max(axis=1)
        elif self.pool == "cls":
            x = x[:, 0, :]  # First token

        return x


class ContrastiveModel(nn.Module):
    """Siamese model for contrastive learning."""
    config: EncoderConfig
    temperature: float = 0.07

    @nn.compact
    def __call__(
        self,
        anchor: jnp.ndarray,
        positive: jnp.ndarray,
        negative: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns: (loss, accuracy)
        """
        embedder = ExecutionEmbedder(self.config, name="embedder")

        z_a = embedder(anchor, mask, train=train)
        z_p = embedder(positive, mask, train=train)
        z_n = embedder(negative, mask, train=train)

        # Normalize
        z_a = z_a / jnp.linalg.norm(z_a, axis=-1, keepdims=True)
        z_p = z_p / jnp.linalg.norm(z_p, axis=-1, keepdims=True)
        z_n = z_n / jnp.linalg.norm(z_n, axis=-1, keepdims=True)

        # Similarities
        sim_ap = jnp.sum(z_a * z_p, axis=-1) / self.temperature
        sim_an = jnp.sum(z_a * z_n, axis=-1) / self.temperature

        # Contrastive loss (InfoNCE-style)
        logits = jnp.stack([sim_ap, sim_an], axis=-1)  # [batch, 2]
        labels = jnp.zeros(logits.shape[0], dtype=jnp.int32)  # Positive is index 0

        loss = jnp.mean(
            nn.softmax_cross_entropy_with_integer_labels(logits, labels)
        )

        # Accuracy
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)

        return loss, acc


def create_model(config: EncoderConfig) -> ExecutionEncoder:
    """Factory function."""
    return ExecutionEncoder(config)


def create_embedder(config: EncoderConfig, pool: str = "mean") -> ExecutionEmbedder:
    """Factory for execution embedder."""
    return ExecutionEmbedder(config, pool=pool)


def create_contrastive_model(config: EncoderConfig) -> ContrastiveModel:
    """Factory for contrastive model."""
    return ContrastiveModel(config)