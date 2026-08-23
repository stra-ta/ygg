"""
Ygg Gradient Attribution

Attribute divergence between two executions to input features using
Integrated Gradients taken with respect to the *token embedding* matrix
(continuous), then roll per-token attributions up to the event-kind / CPU /
thread level.

Why the token embedding (not the raw token indices)?
    The raw token is ``[event_type, thread, cpu, log_dt, arg0, arg1, arg2]``.
    The first three columns are discrete indices fed through ``nn.Embed``; their
    gradient is identically zero, so IG over the raw indices can never attribute
    to event kind / CPU / thread. Taking the gradient with respect to the
    composed token-embedding vectors (the output of ``TokenEmbedding``) makes
    every feature differentiable and yields meaningful per-token attributions
    for all input features.

The attribution target is the cosine distance between the two executions'
pooled embeddings. We compute the gradient of that distance with respect to
trace-1's token-embedding matrix (holding trace-2 fixed) and apply the
Integrated Gradients path integral from a baseline.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import polars as pl

from model.encoder import ExecutionEmbedder
from model.dataset import events_to_tokens


def _tokens_around_timestamp(
    trace_path: str,
    timestamp_ns: int,
    segment_len: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a ``segment_len``-event token window centered near a timestamp.

    Returns ``(tokens [segment_len, 7], mask [segment_len])``. If the trace has
    fewer than ``segment_len`` events it is left-padded with zeros and the mask
    marks the real prefix.
    """
    df = pl.read_parquet(trace_path)
    events = df.select(
        ["timestamp_ns", "cpu", "pid", "tid", "kind", "arg0", "arg1", "arg2"]
    ).to_numpy()
    n = len(events)
    if n == 0:
        return np.zeros((segment_len, 7), dtype=np.float32), np.zeros(segment_len, dtype=np.float32)

    ts = events[:, 0].astype(np.int64)
    center = int(np.argmin(np.abs(ts - timestamp_ns)))
    half = segment_len // 2
    start = max(0, center - half)
    end = min(n, start + segment_len)
    start = max(0, end - segment_len)  # keep length == segment_len

    window = events[start:end]
    tokens = events_to_tokens(window).astype(np.float32)
    mask = np.ones(len(window), dtype=np.float32)

    if len(window) < segment_len:
        pad = segment_len - len(window)
        tokens = np.concatenate([np.zeros((pad, 7), dtype=np.float32), tokens], axis=0)
        mask = np.concatenate([np.zeros(pad, dtype=np.float32), mask], axis=0)

    return tokens, mask


def _exec_from_embeddings(
    E: jnp.ndarray,
    mask: jnp.ndarray,
    cfg,
    local_params: dict,
    global_params: dict,
) -> jnp.ndarray:
    """
    Run the local + global encoder (pooling) from a token-embedding matrix.

    Mirrors :class:`model.hierarchical.HierarchicalEncoder` exactly, but takes
    the per-token embeddings ``E`` directly instead of raw tokens, so that the
    embedding matrix can be the differentiated variable.
    """
    E = E[None]  # [1, S, d]
    mask = mask[None]  # [1, S]
    B, S, d = E.shape
    W = cfg.window_size

    pad_len = (W - S % W) % W
    if pad_len:
        E = jnp.pad(E, ((0, 0), (0, pad_len), (0, 0)))
        mask = jnp.pad(mask, ((0, 0), (0, pad_len)), constant_values=0.0)

    S2 = E.shape[1]
    N = S2 // W
    mw = mask.reshape(B, N, W)
    xw = E.reshape(B, N, W, d)

    from model.hierarchical import LocalEncoder, GlobalEncoder

    local_out = LocalEncoder(cfg).apply({"params": local_params}, xw, pad_mask=mw, train=False)
    wmask = mw[:, :, :, None]
    window = (local_out * wmask).sum(axis=2) / wmask.sum(axis=2).clip(min=1.0)

    real_win = (mw.sum(axis=2) > 0).astype(jnp.float32)
    window_ctx = GlobalEncoder(cfg).apply(
        {"params": global_params}, window, window_mask=real_win, train=False
    )
    wm = real_win[:, :, None]
    exec_emb = (window_ctx * wm).sum(axis=1) / wm.sum(axis=1).clip(min=1.0)
    return exec_emb[0]


def _build_token_exec(
    model: ExecutionEmbedder, params: dict
) -> Tuple[Callable, Callable, bool]:
    """
    Build ``(embed_tokens, exec_from_E, is_embedding_space)``.

    In embedding space (preferred), the differentiated variable is the composed
    token-embedding matrix. Falls back to raw-token space (continuous features
    only) if the hierarchical parameter tree is unavailable.
    """
    cfg = getattr(model, "config", None)
    try:
        if cfg is None or "hier" not in params:
            raise KeyError("no hierarchical params")
        from model.encoder import TokenEmbedding

        te_params = params["hier"]["token_emb"]
        local_params = params["hier"]["local"]
        global_params = params["hier"]["global"]
        token_emb = TokenEmbedding(cfg)

        def embed_tokens(tokens: np.ndarray, mask) -> np.ndarray:
            E = token_emb.apply({"params": te_params}, tokens[None], train=False)
            return np.asarray(E[0])

        def exec_from_E(E: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            return _exec_from_embeddings(E, mask, cfg, local_params, global_params)

        return embed_tokens, exec_from_E, True
    except Exception:
        # Fallback: attribute over raw tokens via the model forward.
        def embed_tokens(tokens: np.ndarray, mask) -> np.ndarray:
            return tokens.astype(np.float32)

        def exec_from_E(tokens: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            e = model.apply({"params": params}, tokens[None], mask[None], train=False)
            return e[0]

        return embed_tokens, exec_from_E, False


def integrated_gradients(
    embed_tokens: Callable,
    exec_from_E: Callable,
    var1: np.ndarray,
    var2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    *,
    steps: int = 50,
    baseline: str = "counterfactual",
) -> np.ndarray:
    """
    Integrated Gradients of the divergence w.r.t. ``var1`` (trace-1 embedding
    or token matrix). Returns an array shaped like ``var1``.
    """
    var1 = np.asarray(var1, dtype=np.float32)
    var2 = np.asarray(var2, dtype=np.float32)
    if baseline == "zero":
        base = np.zeros_like(var1)
    else:
        base = var2  # counterfactual

    def dist_fn(v: jnp.ndarray) -> jnp.ndarray:
        e1 = exec_from_E(v, jnp.array(mask1))
        e2 = exec_from_E(jnp.array(var2), jnp.array(mask2))
        e1 = e1 / jnp.linalg.norm(e1, axis=-1, keepdims=True)
        e2 = e2 / jnp.linalg.norm(e2, axis=-1, keepdims=True)
        return 1.0 - jnp.sum(e1 * e2, axis=-1)

    grad_fn = jax.grad(dist_fn)

    alphas = np.linspace(0.0, 1.0, steps)
    total = np.zeros_like(var1)
    for a in alphas:
        interpolated = base + a * (var1 - base)
        g = np.asarray(grad_fn(jnp.array(interpolated)))
        total += g
    total *= (var1 - base) / steps  # IG formula
    total = total * mask1[:, None]  # zero out padding positions
    return total


def per_token_attribution(ig: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-token attribution magnitude (L2 norm across embedding dims)."""
    mag = np.linalg.norm(np.asarray(ig, dtype=float), axis=-1)
    if mask is not None:
        mag = mag * np.asarray(mask, dtype=float)
    return mag


def aggregate_attribution(
    tokens: np.ndarray,
    per_token: np.ndarray,
) -> Dict[str, float]:
    """
    Roll per-token (embedding-gradient) attributions up to semantic levels.

    Produces keys:
        ``kind_<k>``    - aggregated by event kind
        ``cpu_<c>``     - aggregated by CPU id
        ``thread_<t>``  - aggregated by thread id

    Continuous features (dt, arg0/1/2) are attributed separately via a raw-token
    Integrated Gradients pass; see :func:`aggregate_continuous`.
    """
    tokens = np.asarray(tokens, dtype=float)
    agg: Dict[str, float] = {}

    kind_col = tokens[:, 0].astype(int)
    for k in np.unique(kind_col):
        agg[f"kind_{int(k)}"] = float(per_token[kind_col == k].sum())

    cpu_col = tokens[:, 2].astype(int)
    for c in np.unique(cpu_col):
        agg[f"cpu_{int(c)}"] = float(per_token[cpu_col == c].sum())

    thread_col = tokens[:, 1].astype(int)
    for t in np.unique(thread_col):
        agg[f"thread_{int(t)}"] = float(per_token[thread_col == t].sum())

    return agg


def aggregate_continuous(raw_ig: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Aggregate a raw-token Integrated Gradients tensor to continuous features.

    ``raw_ig`` is the gradient of the distance w.r.t. the raw token matrix; the
    categorical columns (0,1,2) are discrete and contribute ~0, so only the
    continuous columns are reported: ``dt`` (col 3), ``arg0/1/2`` (cols 4-6).
    """
    raw_ig = np.asarray(raw_ig, dtype=float)
    if mask is not None:
        raw_ig = raw_ig * mask[:, None]
    out: Dict[str, float] = {}
    if raw_ig.shape[1] > 3:
        out["dt"] = float(np.abs(raw_ig[:, 3]).sum())
    if raw_ig.shape[1] > 4:
        out["arg0"] = float(np.abs(raw_ig[:, 4]).sum())
    if raw_ig.shape[1] > 5:
        out["arg1"] = float(np.abs(raw_ig[:, 5]).sum())
    if raw_ig.shape[1] > 6:
        out["arg2"] = float(np.abs(raw_ig[:, 6]).sum())
    return out


def integrated_gradients_raw(
    model: ExecutionEmbedder,
    params: dict,
    tok1: np.ndarray,
    tok2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    *,
    steps: int = 50,
    baseline: str = "counterfactual",
) -> np.ndarray:
    """
    Integrated Gradients of the divergence w.r.t. the raw token matrix.

    Continuous columns (dt, arg0/1/2) are differentiable through the model's
    projection layers; categorical columns (kind/cpu/thread) are discrete and
    yield ~0 gradient (use the embedding-space pass for those).
    """
    base = tok2.astype(np.float32) if baseline == "counterfactual" else np.zeros_like(tok1)

    def dist_fn(v: jnp.ndarray) -> jnp.ndarray:
        e1 = model.apply({"params": params}, v[None], mask1[None], train=False)
        e2 = model.apply({"params": params}, tok2[None], mask2[None], train=False)
        e1 = e1[0] / jnp.linalg.norm(e1[0], axis=-1, keepdims=True)
        e2 = e2[0] / jnp.linalg.norm(e2[0], axis=-1, keepdims=True)
        return 1.0 - jnp.sum(e1 * e2, axis=-1)

    grad_fn = jax.grad(dist_fn)
    tok1 = tok1.astype(np.float32)
    total = np.zeros_like(tok1)
    for a in np.linspace(0.0, 1.0, steps):
        interpolated = base + a * (tok1 - base)
        g = np.asarray(grad_fn(jnp.array(interpolated)))
        total += g
    total *= (tok1 - base) / steps
    total = total * mask1[:, None]
    return total


def attribute_divergence(
    model: ExecutionEmbedder,
    params: dict,
    trace1: str,
    trace2: str,
    start,
    *,
    window_ns: int = 10_000_000,
    steps: int = 50,
    baseline: str = "counterfactual",
    top_k: int = 30,
    segment_len: int = 512,
) -> Dict[str, float]:
    """
    Attribute a divergence (identified by ``start.timestamp_ns``) to features.

    Returns the top-``top_k`` features by absolute aggregated attribution,
    sorted descending.
    """
    ts = int(getattr(start, "timestamp_ns", start))
    tok1, mask1 = _tokens_around_timestamp(trace1, ts, segment_len)
    tok2, mask2 = _tokens_around_timestamp(trace2, ts, segment_len)

    embed_tokens, exec_from_E, _ = _build_token_exec(model, params)
    E1 = embed_tokens(tok1, mask1)
    E2 = embed_tokens(tok2, mask2)

    ig = integrated_gradients(
        embed_tokens, exec_from_E, E1, E2, mask1, mask2,
        steps=steps, baseline=baseline,
    )
    per_token = per_token_attribution(ig, mask1)
    agg = aggregate_attribution(tok1, per_token)

    # Continuous features via a raw-token IG pass (categorical cols are discrete).
    raw_ig = integrated_gradients_raw(
        model, params, tok1, tok2, mask1, mask2,
        steps=steps, baseline=baseline,
    )
    agg.update(aggregate_continuous(raw_ig, mask1))

    ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return dict(ranked[:top_k])


def causal_attribution(graph_model, trace1: str, trace2: str, start) -> Dict:
    """
    Attribute divergence to causal edges in a learned causal graph.

    FUTURE WORK: requires a trained causal graph model. The signature is
    reserved so downstream tooling can call it once such a model exists.
    """
    raise NotImplementedError(
        "Causal attribution requires a learned causal graph model (future work). "
        "Graph model integration point reserved at analysis.attribution.causal_attribution."
    )
