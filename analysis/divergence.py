"""
Ygg Divergence Analysis

Locates behavioral divergence points between two executions.

Pipeline:
  1. Embed each trace into a sequence of segment-level vectors (one per window).
  2. Align the two embedding sequences with Dynamic Time Warping so that
     executions at different speeds are compared point-to-point.
  3. Build a distance time-series (cosine distance per aligned pair, smoothed
     with a sliding window).
  4. Detect change points with PELT (or Bayesian online change point detection).
  5. Report the first *sustained* divergence: distance above threshold for
     longer than ``min_duration_ns``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp
import polars as pl

from model.encoder import ExecutionEmbedder
from model.dataset import events_to_tokens


def _window_tokens(
    tokens: np.ndarray, window_size: int, stride: int
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Sliding windows over a token sequence.

    Returns a list of ``(window_tokens, mask, start_idx)``. Short sequences are
    zero-padded to ``window_size`` with a mask marking real positions. A trailing
    window anchored at the end guarantees coverage of the tail.
    """
    n = len(tokens)
    if n == 0:
        return []
    if n < window_size:
        pad = window_size - n
        w = np.concatenate(
            [tokens, np.zeros((pad, tokens.shape[1]), dtype=tokens.dtype)], axis=0
        )
        mask = np.concatenate(
            [np.ones(n, dtype=np.float32), np.zeros(pad, dtype=np.float32)]
        )
        return [(w, mask, 0)]

    out: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        out.append((tokens[start:end], np.ones(window_size, dtype=np.float32), start))

    # Ensure the tail is covered.
    tail_start = n - window_size
    if out and out[-1][2] < tail_start:
        out.append((tokens[tail_start:], np.ones(window_size, dtype=np.float32), tail_start))
    if not out:
        out.append((tokens, np.ones(n, dtype=np.float32), 0))
    return out


@dataclass
class DivergencePoint:
    """A detected divergence between two executions at a point in time."""

    timestamp_ns: int
    window_start_ns: int
    window_end_ns: int
    distance: float
    confidence: float
    # Aligned source indices in the DTW path (for downstream attribution).
    aligned_idx1: int = -1
    aligned_idx2: int = -1
    # Optional human-readable dominant features (filled by attribution).
    dominant_features: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Embedding loading
# --------------------------------------------------------------------------- #
def load_embeddings(
    model: ExecutionEmbedder,
    params: dict,
    trace_path: str,
    segment_len: int = 512,
    stride: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute embeddings for all segments in a trace.

    Returns ``(embeddings [n_segments, d_model], timestamps [n_segments])``.
    """
    df = pl.read_parquet(trace_path)
    events = df.select(
        ["timestamp_ns", "cpu", "pid", "tid", "kind", "arg0", "arg1", "arg2"]
    ).to_numpy()

    if len(events) == 0:
        return np.empty((0, 0)), np.empty((0,))

    tokens = events_to_tokens(events)
    windows = _window_tokens(tokens, segment_len, stride)

    embeddings: List[np.ndarray] = []
    timestamps: List[int] = []
    for w, mask, start in windows:
        batch = w[None, ...]  # [1, seq_len, token_dim]
        emb = model.apply({"params": params}, batch, mask[None, :], train=False)
        embeddings.append(np.asarray(emb[0]))
        mid = start + w.shape[0] // 2
        timestamps.append(int(events[min(mid, len(events) - 1), 0]))

    if not embeddings:
        return np.empty((0, 0)), np.empty((0,))
    return np.stack(embeddings), np.array(timestamps)


# --------------------------------------------------------------------------- #
# Dynamic Time Warping
# --------------------------------------------------------------------------- #
def dtw_align(
    emb1: np.ndarray,
    emb2: np.ndarray,
    band_frac: Optional[float] = 0.25,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Align two embedding sequences with (optionally banded) DTW.

    Returns the warping path ``[(i, j), ...]`` and the precomputed cosine
    distance matrix ``cost`` of shape ``(n1, n2)``.
    """
    if emb1.size == 0 or emb2.size == 0:
        return [], np.empty((0, 0))

    n1, n2 = emb1.shape[0], emb2.shape[0]
    a = emb1 / np.linalg.norm(emb1, axis=-1, keepdims=True)
    b = emb2 / np.linalg.norm(emb2, axis=-1, keepdims=True)
    sim = a @ b.T  # [n1, n2]
    cost = 1.0 - np.clip(sim, -1.0, 1.0)

    if band_frac is None:
        band = max(n1, n2)
    else:
        band = max(1, int(band_frac * max(n1, n2)))

    D = np.full((n1 + 1, n2 + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n1 + 1):
        j_lo = max(1, i - band)
        j_hi = min(n2 + 1, i + band + 1)
        ci = cost[i - 1]
        Di = D[i]
        Dim1 = D[i - 1]
        for j in range(j_lo, j_hi):
            c = ci[j - 1] + min(Dim1[j], Di[j - 1], Dim1[j - 1])
            Di[j] = c

    # Backtrack.
    path: List[Tuple[int, int]] = []
    i, j = n1, n2
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        d_diag = D[i - 1, j - 1]
        d_up = D[i - 1, j]
        d_left = D[i, j - 1]
        if d_diag <= d_up and d_diag <= d_left:
            i -= 1
            j -= 1
        elif d_up <= d_left:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path, cost


# --------------------------------------------------------------------------- #
# Change point detection
# --------------------------------------------------------------------------- #
def _pelt_gaussian(series: np.ndarray, penalty: Optional[float]) -> List[int]:
    """PELT with a Gaussian (sum-of-squares) segment cost. Exact offline."""
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n < 4:
        return []

    if penalty is None:
        var = float(y.var()) + 1e-12
        penalty = 2.0 * np.log(n) * var

    # Prefix sums for O(1) segment variance cost: sum (y - mean)^2.
    ps = np.concatenate([[0.0], np.cumsum(y)])
    ps2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def seg_cost(s: int, t: int) -> float:
        m = t - s
        if m <= 0:
            return 0.0
        s_sum = ps[t] - ps[s]
        s_sq = ps2[t] - ps2[s]
        cost = s_sq - (s_sum * s_sum) / m
        return max(0.0, cost)  # guard against float round-off

    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    cp = np.zeros(n + 1, dtype=int)

    for t in range(1, n + 1):
        best = np.inf
        best_s = 0
        for s in range(t):
            c = F[s] + seg_cost(s, t) + penalty
            if c < best:
                best = c
                best_s = s
        F[t] = best
        cp[t] = best_s

    changes: List[int] = []
    t = n
    while t > 0:
        s = cp[t]
        if s != 0:
            changes.append(s)
        t = s
    changes.reverse()
    return changes


def _student_t_logpdf(x: float, mu: float, nu: float, sigma2: float) -> float:
    """Log of Student-t density (location mu, dof nu, scale^2 sigma2)."""
    if sigma2 <= 0:
        sigma2 = 1e-12
    log_z = (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi * sigma2)
    )
    return log_z - 0.5 * (nu + 1.0) * math.log1p((x - mu) ** 2 / (nu * sigma2))


def _bocpd(series: np.ndarray, hazard: float) -> np.ndarray:
    """
    Bayesian online change point detection (Adams & Mackay 2007) with a
    Normal-Inverse-Gamma conjugate prior over a Gaussian observation model.

    Returns the run-length-0 probability (changepoint probability) per step.
    """
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n == 0:
        return np.empty(0)

    # Prior hyperparameters (alpha0 > 1 so the prior mean variance is finite).
    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 2.0, 1.0
    log_h = math.log(1.0 - hazard)

    # Posterior state per active run length.
    mu = [mu0]
    kappa = [kappa0]  # placeholder; replaced below
    alpha = [alpha0]
    beta = [beta0]
    kappa = [kappa0]
    R = np.array([1.0])  # belief over run lengths (index = run length)
    cps = np.zeros(n)

    for t in range(n):
        # Predictive log-probability for each run length (Student-t marginal).
        nu = 2.0 * np.array(alpha)
        sigma2 = (np.array(beta) / np.array(alpha)) * (1.0 + 1.0 / np.array(kappa))
        pred = np.array(
            [_student_t_logpdf(y[t], mu[r], nu[r], sigma2[r]) for r in range(len(mu))]
        )

        # Growth (no change) and change (run length resets to 0).
        R_growth = R * np.exp(pred + log_h)
        R_change = np.exp(pred[0] + math.log(hazard)) * np.sum(R)
        R_new = np.concatenate([[R_change], R_growth])
        total = R_new.sum()
        if total > 0:
            R_new = R_new / total
        cps[t] = R_new[0]

        # Update posteriors. A growth from run length r (at t-1) to r+1 (at t)
        # incorporates y[t]; the just-reset run (length 0) starts from the
        # prior again, so the prior is inserted at index 0 and the rest shift.
        mu_n = [mu0]
        kappa_n = [kappa0]
        alpha_n = [alpha0]
        beta_n = [beta0]
        for r in range(len(mu)):
            kappa_n.append(kappa[r] + 1.0)
            mu_n.append((kappa[r] * mu[r] + y[t]) / (kappa[r] + 1.0))
            alpha_n.append(alpha[r] + 0.5)
            beta_n.append(
                beta[r]
                + 0.5 * (y[t] - mu[r]) ** 2
                + 0.5 * kappa[r] / (kappa[r] + 1.0) * (y[t] - mu[r]) ** 2
            )
        mu = mu_n
        kappa = kappa_n
        alpha = alpha_n
        beta = beta_n
        R = R_new

    return cps


def detect_change_points(
    series: Sequence[float],
    method: str = "pelt",
    penalty: Optional[float] = None,
    hazard: Optional[float] = None,
    min_size: int = 2,
) -> List[int]:
    """
    Detect change points in a 1D series.

    ``method="pelt"`` uses exact PELT with a Gaussian cost. ``method="bocpd"``
    uses Bayesian online change point detection; change points are the steps
    whose changepoint probability exceeds 0.5.
    """
    y = np.asarray(series, dtype=float)
    if method == "pelt":
        changes = _pelt_gaussian(y, penalty)
        # Enforce minimum segment size by merging tiny segments.
        if min_size > 1 and changes:
            merged: List[int] = []
            last = 0
            for c in changes:
                if c - last >= min_size:
                    merged.append(c)
                    last = c
            changes = merged
        return changes
    elif method == "bocpd":
        if hazard is None:
            # Adaptive default: expected run length scales with series length.
            hazard = 1.0 / max(10, len(y) // 4)
        cps = _bocpd(y, hazard)
        return [int(i) for i in range(1, len(cps)) if cps[i] > 0.5]
    else:
        raise ValueError(f"Unknown change point method: {method}")


# --------------------------------------------------------------------------- #
# Sliding window + divergence construction
# --------------------------------------------------------------------------- #
def _sliding_mean(series: np.ndarray, half: int) -> np.ndarray:
    n = len(series)
    out = np.empty(n)
    for k in range(n):
        lo = max(0, k - half)
        hi = min(n, k + half + 1)
        out[k] = series[lo:hi].mean()
    return out


def _confidence(distance: float, threshold: float) -> float:
    """Map a raw distance to a 0..1 confidence relative to the threshold."""
    if distance <= 0:
        return 0.0
    if threshold <= 0:
        return float(np.clip(distance, 0.0, 1.0))
    c = (distance - threshold) / (1.0 - threshold + 1e-9)
    return float(np.clip(c, 0.0, 1.0))


def find_divergence(
    trace1: str,
    trace2: str,
    model: ExecutionEmbedder,
    params: dict,
    *,
    segment_len: int = 512,
    stride: int = 256,
    window_size: int = 10,
    threshold: float = 0.1,
    dtw_band_frac: Optional[float] = 0.25,
    change_method: str = "pelt",
    change_penalty: Optional[float] = None,
    change_hazard: float = 1.0 / 250.0,
) -> List[DivergencePoint]:
    """
    Detect divergence points between two traces.

    Returns a list of :class:`DivergencePoint` (one per DTW-aligned step),
    each carrying the smoothed cosine distance and a per-step confidence.
    """
    emb1, ts1 = load_embeddings(model, params, trace1, segment_len, stride)
    emb2, ts2 = load_embeddings(model, params, trace2, segment_len, stride)
    if emb1.size == 0 or emb2.size == 0:
        return []

    path, cost = dtw_align(emb1, emb2, band_frac=dtw_band_frac)

    dist_series: List[float] = []
    t_series: List[int] = []
    idx1: List[int] = []
    idx2: List[int] = []
    for (i, j) in path:
        dist_series.append(float(cost[i, j]))
        t_series.append(int((int(ts1[i]) + int(ts2[j])) // 2))
        idx1.append(i)
        idx2.append(j)

    dist_series = np.array(dist_series)
    t_series = np.array(t_series)

    # Smooth with a sliding window over the aligned distance series.
    half = max(1, window_size // 2)
    wmean = _sliding_mean(dist_series, half)

    # Keep the series ordered by (mid) time for stable change point detection.
    order = np.argsort(t_series)
    t_sorted = t_series[order]
    d_sorted = wmean[order]
    i1_sorted = np.array(idx1)[order]
    i2_sorted = np.array(idx2)[order]

    points = [
        DivergencePoint(
            timestamp_ns=int(t_sorted[k]),
            window_start_ns=int(t_sorted[max(0, k - half)]),
            window_end_ns=int(t_sorted[min(len(t_sorted) - 1, k + half)]),
            distance=float(d_sorted[k]),
            confidence=_confidence(float(d_sorted[k]), threshold),
            aligned_idx1=int(i1_sorted[k]),
            aligned_idx2=int(i2_sorted[k]),
        )
        for k in range(len(t_sorted))
    ]
    return points


def find_divergence_start(
    divergences: List[DivergencePoint],
    *,
    threshold: float = 0.1,
    min_duration_ns: int = 1_000_000,
    change_method: str = "pelt",
    change_penalty: Optional[float] = None,
) -> Optional[DivergencePoint]:
    """
    Return the first *sustained* divergence point.

    A divergence is sustained when the distance stays above ``threshold`` for
    at least ``min_duration_ns``. Confidence reflects the separation between
    the diverging segment and the preceding baseline.
    """
    if not divergences:
        return None

    # Sort by time; guard against duplicate timestamps.
    pts = sorted(divergences, key=lambda d: d.timestamp_ns)
    ts = np.array([p.timestamp_ns for p in pts], dtype=np.int64)
    dist = np.array([p.distance for p in pts], dtype=float)
    if len(ts) < 2:
        return None

    changes = detect_change_points(dist, method=change_method, penalty=change_penalty)
    # Segment boundaries in index space.
    bounds = [0] + changes + [len(ts)]

    for seg_i in range(1, len(bounds) - 1):
        s, e = bounds[seg_i], bounds[seg_i + 1]
        seg_ts = ts[s:e]
        seg_dist = dist[s:e]

        # Sustained = a contiguous run above threshold spanning >= min_duration.
        run, run_span = _longest_above_threshold(seg_ts, seg_dist, threshold)
        if run is None or run_span <= min_duration_ns:
            continue

        run_start, run_end = run
        # Preceding baseline segment.
        prev_s, prev_e = bounds[seg_i - 1], s
        baseline = dist[prev_s:prev_e]
        base_mean = float(baseline.mean()) if len(baseline) else 0.0
        base_std = float(baseline.std()) + 1e-6
        run_mean = float(seg_dist[run_start:run_end + 1].mean())
        sep = (run_mean - base_mean) / base_std
        confidence = float(1.0 / (1.0 + math.exp(-sep)))
        start_pt = pts[s + run_start]
        return DivergencePoint(
            timestamp_ns=int(seg_ts[run_start]),
            window_start_ns=int(seg_ts[run_start]),
            window_end_ns=int(seg_ts[run_end]),
            distance=run_mean,
            confidence=confidence,
            aligned_idx1=start_pt.aligned_idx1,
            aligned_idx2=start_pt.aligned_idx2,
        )

    return None


def _longest_above_threshold(
    ts: np.ndarray, dist: np.ndarray, threshold: float
) -> Tuple[Optional[Tuple[int, int]], int]:
    """Longest contiguous run where ``dist > threshold``; returns (run, span_ns)."""
    best: Optional[Tuple[int, int]] = None
    best_span = -1
    cur_start = None
    for k in range(len(dist)):
        if dist[k] > threshold:
            if cur_start is None:
                cur_start = k
            if k == len(dist) - 1 or dist[k + 1] <= threshold:
                span = int(ts[k] - ts[cur_start])
                if span > best_span:
                    best_span = span
                    best = (cur_start, k)
                cur_start = None
        else:
            cur_start = None
    return best, best_span


# --------------------------------------------------------------------------- #
# Attribution hook
# --------------------------------------------------------------------------- #
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
    Attribute a divergence to input features.

    Delegates to :mod:`analysis.attribution` (integrated gradients). ``start``
    is a :class:`DivergencePoint` (or any object exposing ``timestamp_ns``).
    """
    from .attribution import attribute_divergence as _attr

    return _attr(
        model,
        params,
        trace1,
        trace2,
        start,
        window_ns=window_ns,
        steps=steps,
        baseline=baseline,
        top_k=top_k,
        segment_len=segment_len,
    )


# --------------------------------------------------------------------------- #
# Convenience full diff (backwards compatible API)
# --------------------------------------------------------------------------- #
def diff_traces(
    model: ExecutionEmbedder,
    params: dict,
    trace1: str,
    trace2: str,
    *,
    segment_len: int = 512,
    stride: int = 256,
    threshold: float = 0.1,
    window_size: int = 10,
) -> Dict:
    """Full diff between two traces: points, start, attribution, summary."""
    divergences = find_divergence(
        trace1, trace2, model, params,
        segment_len=segment_len, stride=stride,
        window_size=window_size, threshold=threshold,
    )
    start = find_divergence_start(divergences, threshold=threshold)

    result: Dict = {
        "trace1": trace1,
        "trace2": trace2,
        "num_divergences": len(divergences),
        "divergence_start": None,
        "attribution": {},
        "top_divergences": [],
    }

    if start is not None:
        result["divergence_start"] = {
            "timestamp_ns": start.timestamp_ns,
            "window_start_ns": start.window_start_ns,
            "window_end_ns": start.window_end_ns,
            "distance": start.distance,
            "confidence": start.confidence,
        }
        result["attribution"] = attribute_divergence(
            model, params, trace1, trace2, start
        )
        top = sorted(divergences, key=lambda d: d.distance, reverse=True)[:5]
        result["top_divergences"] = [
            {"timestamp_ns": d.timestamp_ns, "distance": d.distance}
            for d in top
        ]

    return result
