"""
Hierarchical execution encoder.

Two stages:

    Local  : window of ``window_size`` events -> ``local_layers`` transformer
             blocks -> per-event context + a single window embedding.
    Global : sequence of ``N`` window embeddings -> ``global_layers``
             transformer blocks -> contextualised window embeddings and a
             single execution-level embedding.

Both stages reuse the pre-LN :class:`TransformerBlock` from
:mod:`model.encoder`.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Dict

from model.encoder import TokenEmbedding, TransformerBlock
from model.config import ModelConfig


class LocalEncoder(nn.Module):
    """Transformer over events *within* a single window."""

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        pad_mask: jnp.ndarray,
        causal: bool = False,
        train: bool = True,
    ) -> jnp.ndarray:
        # x:        [B, N, W, d]
        # pad_mask: [B, N, W]  (1 = real event, 0 = padding)
        B, N, W, _ = x.shape

        pad_f = (pad_mask.reshape(B * N, W) > 0)[:, None, None, :]  # [BN, 1, 1, W]
        if causal:
            causal_m = jnp.tril(jnp.ones((W, W), dtype=jnp.bool_))[None, None, :, :]
            attn = pad_f & causal_m  # [BN, 1, W, W]
        else:
            attn = pad_f  # [BN, 1, 1, W]

        x_f = x.reshape(B * N, W, self.config.d_model)
        for i in range(self.config.local_layers):
            x_f = TransformerBlock(self.config, name=f"local_{i}")(
                x_f, attn_mask=attn, train=train
            )

        return x_f.reshape(B, N, W, self.config.d_model)


class GlobalEncoder(nn.Module):
    """Transformer over the sequence of window embeddings."""

    config: ModelConfig
    max_windows: int = 4096

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        window_mask: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        # x:           [B, N, d]
        # window_mask: [B, N]  (1 = real window)
        B, N, _ = x.shape

        pos = jnp.arange(N)[None, :]
        pos_emb = nn.Embed(
            num_embeddings=self.max_windows,
            features=self.config.d_model,
            name="window_pos",
        )(pos.astype(jnp.int32))
        x = x + pos_emb

        attn = window_mask[:, None, None, :]  # [B, 1, 1, N]
        for i in range(self.config.global_layers):
            x = TransformerBlock(self.config, name=f"global_{i}")(
                x, attn_mask=attn, train=train
            )
        return x


class HierarchicalEncoder(nn.Module):
    """Local + global encoder returning event/window/execution embeddings."""

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        causal: bool = False,
        train: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        B, S, _ = tokens.shape
        W = self.config.window_size
        d = self.config.d_model

        # Pad the sequence to a multiple of the window size.
        pad_len = (W - S % W) % W
        if pad_len:
            tokens = jnp.pad(tokens, ((0, 0), (0, pad_len), (0, 0)))
            if mask is None:
                mask = jnp.ones((B, S), dtype=jnp.float32)
            mask = jnp.pad(mask, ((0, 0), (0, pad_len)), constant_values=0.0)

        S2 = tokens.shape[1]
        N = S2 // W

        if mask is None:
            mask = jnp.ones((B, S2), dtype=jnp.float32)
        mask = mask.astype(jnp.float32)

        x = TokenEmbedding(self.config, name="token_emb")(tokens, train=train)
        # [B, N, W, d]
        xw = x.reshape(B, N, W, d)
        mw = mask.reshape(B, N, W)

        local_out = LocalEncoder(self.config, name="local")(
            xw, pad_mask=mw, causal=causal, train=train
        )  # [B, N, W, d]

        # Per-window embedding: masked mean over the window axis.
        wmask = mw[:, :, :, None]
        window = (local_out * wmask).sum(axis=2) / wmask.sum(axis=2).clip(min=1.0)

        # Real-window indicator for the global attention / pooling.
        real_win = (mw.sum(axis=2) > 0).astype(jnp.float32)  # [B, N]

        window_ctx = GlobalEncoder(self.config, name="global")(
            window, window_mask=real_win, train=train
        )  # [B, N, d]

        # Execution embedding: masked mean over windows.
        wm = real_win[:, :, None]
        exec_emb = (window_ctx * wm).sum(axis=1) / wm.sum(axis=1).clip(min=1.0)

        # Event embeddings: flatten local output back to the sequence.
        event = local_out.reshape(B, S2, d)[:, :S, :]

        return {
            "event": event,
            "window": window,
            "window_ctx": window_ctx,
            "exec": exec_emb,
        }
