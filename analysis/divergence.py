"""
Ygg Divergence Analysis

Locates behavioral divergence points between executions.
"""

import jax.numpy as jnp
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from model.encoder import ExecutionEmbedder, EncoderConfig
from model.dataset import events_to_tokens, create_segments


@dataclass
class DivergencePoint:
    """A detected divergence between two executions."""
    timestamp_ns: int
    window_start: int
    window_end: int
    distance: float
    dominant_features: Dict[str, float]


def load_embeddings(model: ExecutionEmbedder, params: dict, trace_path: str, segment_len: int = 512, stride: int = 256) -> Tuple[jnp.ndarray, np.ndarray]:
    """
    Compute embeddings for all segments in a trace.

    Returns: (embeddings [n_segments, d_model], timestamps [n_segments])
    """
    df = pl.read_parquet(trace_path)
    events = df.select([
        "timestamp_ns", "cpu", "pid", "tid", "kind",
        "arg0", "arg1", "arg2"
    ]).to_numpy()

    tokens = events_to_tokens(events)
    segments = create_segments(tokens, segment_len, stride)

    embeddings = []
    timestamps = []
    for seg in segments:
        batch = seg.events[None, ...]  # Add batch dim
        mask = jnp.ones((1, segment_len))
        emb = model.apply({"params": params}, batch, mask, train=False)
        embeddings.append(emb[0])
        # Use middle event timestamp as segment representative
        mid_idx = segment_len // 2
        timestamps.append(events[seg.start_idx + mid_idx, 0])

    return jnp.stack(embeddings), np.array(timestamps)


def compute_divergence(
    emb1: jnp.ndarray,
    emb2: jnp.ndarray,
    ts1: np.ndarray,
    ts2: np.ndarray,
    window_size: int = 10,
) -> List[DivergencePoint]:
    """
    Compute sliding-window divergence between two embedding sequences.

    Uses cosine distance between window-averaged embeddings.
    """
    n1, n2 = len(emb1), len(emb2)
    min_len = min(n1, n2)

    # Align by time (simplified - assumes similar duration)
    # In practice, use DTW or timestamp alignment
    emb1 = emb1[:min_len]
    emb2 = emb2[:min_len]
    ts = ts1[:min_len]

    # Normalize
    emb1 = emb1 / jnp.linalg.norm(emb1, axis=-1, keepdims=True)
    emb2 = emb2 / jnp.linalg.norm(emb2, axis=-1, keepdims=True)

    divergences = []
    half_window = window_size // 2

    for i in range(half_window, min_len - half_window):
        w1 = emb1[i - half_window:i + half_window].mean(axis=0)
        w2 = emb2[i - half_window:i + half_window].mean(axis=0)

        # Cosine distance
        dist = 1.0 - jnp.dot(w1, w2)

        if dist > 0.1:  # Threshold
            divergences.append(DivergencePoint(
                timestamp_ns=int(ts[i]),
                window_start=int(ts[i - half_window]),
                window_end=int(ts[i + half_window]),
                distance=float(dist),
                dominant_features={},  # Filled by attribution
            ))

    return divergences


def find_divergence_start(
    divergences: List[DivergencePoint],
    min_duration_ns: int = 1_000_000,  # 1ms
) -> Optional[DivergencePoint]:
    """
    Find the first sustained divergence point.

    Requires divergence to persist for min_duration_ns.
    """
    if not divergences:
        return None

    current_start = divergences[0]
    for d in divergences[1:]:
        if d.timestamp_ns - current_start.timestamp_ns > min_duration_ns:
            return current_start
        if d.distance < current_start.distance * 0.5:
            current_start = d

    return current_start if divergences[-1].timestamp_ns - divergences[0].timestamp_ns > min_duration_ns else None


def attribute_divergence(
    model: ExecutionEmbedder,
    params: dict,
    trace1: str,
    trace2: str,
    divergence_ts: int,
    window_ns: int = 10_000_000,  # 10ms window
) -> Dict[str, float]:
    """
    Attribute divergence to feature changes.

    Compares feature statistics in a window around the divergence point.
    """
    df1 = pl.read_parquet(trace1)
    df2 = pl.read_parquet(trace2)

    # Filter to window
    w_start = divergence_ts - window_ns // 2
    w_end = divergence_ts + window_ns // 2

    df1_w = df1.filter((pl.col("timestamp_ns") >= w_start) & (pl.col("timestamp_ns") <= w_end))
    df2_w = df2.filter((pl.col("timestamp_ns") >= w_start) & (pl.col("timestamp_ns") <= w_end))

    # Compute feature statistics
    features = {}

    # Event kind distribution
    for kind in df1_w["kind"].unique().to_list():
        c1 = df1_w.filter(pl.col("kind") == kind).height
        c2 = df2_w.filter(pl.col("kind") == kind).height
        total1 = df1_w.height
        total2 = df2_w.height
        if total1 > 0 and total2 > 0:
            features[f"kind_{kind}_rate"] = (c2 / total2) - (c1 / total1)

    # Arg0 statistics (per kind)
    for kind in df1_w["kind"].unique().to_list():
        vals1 = df1_w.filter(pl.col("kind") == kind)["arg0"]
        vals2 = df2_w.filter(pl.col("kind") == kind)["arg0"]
        if len(vals1) > 10 and len(vals2) > 10:
            features[f"kind_{kind}_arg0_mean_delta"] = float(vals2.mean() - vals1.mean())
            features[f"kind_{kind}_arg0_p99_delta"] = float(vals2.quantile(0.99) - vals1.quantile(0.99))

    # CPU utilization
    for cpu in df1_w["cpu"].unique().to_list():
        c1 = df1_w.filter(pl.col("cpu") == cpu).height
        c2 = df2_w.filter(pl.col("cpu") == cpu).height
        total1 = df1_w.height
        total2 = df2_w.height
        if total1 > 0 and total2 > 0:
            features[f"cpu_{cpu}_util_delta"] = (c2 / total2) - (c1 / total1)

    # Sort by magnitude
    return dict(sorted(features.items(), key=lambda x: abs(x[1]), reverse=True))


def diff_traces(
    model: ExecutionEmbedder,
    params: dict,
    trace1: str,
    trace2: str,
    segment_len: int = 512,
    stride: int = 256,
) -> Dict:
    """
    Full diff between two traces.

    Returns dict with divergence points, attribution, and summary.
    """
    print(f"Loading embeddings for {trace1}...")
    emb1, ts1 = load_embeddings(model, params, trace1, segment_len, stride)

    print(f"Loading embeddings for {trace2}...")
    emb2, ts2 = load_embeddings(model, params, trace2, segment_len, stride)

    print("Computing divergence...")
    divergences = compute_divergence(emb1, emb2, ts1, ts2)

    print("Finding divergence start...")
    start = find_divergence_start(divergences)

    result = {
        "trace1": trace1,
        "trace2": trace2,
        "num_divergences": len(divergences),
        "divergence_start": None,
        "attribution": {},
        "top_divergences": [],
    }

    if start:
        print(f"Divergence starts at {start.timestamp_ns} ns")
        result["divergence_start"] = {
            "timestamp_ns": start.timestamp_ns,
            "window_start_ns": start.window_start,
            "window_end_ns": start.window_end,
            "distance": start.distance,
        }

        print("Attributing divergence...")
        attribution = attribute_divergence(model, params, trace1, trace2, start.timestamp_ns)
        result["attribution"] = attribution

        # Top 5 divergences
        top = sorted(divergences, key=lambda d: d.distance, reverse=True)[:5]
        result["top_divergences"] = [
            {"timestamp_ns": d.timestamp_ns, "distance": d.distance}
            for d in top
        ]

    return result