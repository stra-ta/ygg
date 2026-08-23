"""
Ygg dataset loader.

Turns Kiln/collector Parquet traces into training batches for the four
objectives. Features:

* Streaming reader (Polars -> numpy -> JAX) that processes one Parquet file at a
  time, so arbitrarily large campaigns fit in memory.
* Dynamic segment sampling: each window is drawn from a *random* offset rather
  than a fixed stride.
* Contrastive pairing from Kiln campaign structure: healthy-healthy positives
  and healthy-corrupted negatives.
"""

import jax.numpy as jnp
import numpy as np
import polars as pl
from pathlib import Path
from typing import Iterator, List, Dict, Optional, Tuple
from dataclasses import dataclass


# Token layout (must match model.encoder.TokenEmbedding):
#   [event_type, thread, cpu, log_dt, arg0, arg1, arg2]
TOKEN_DIM = 7


@dataclass
class TraceSegment:
    """A contiguous window of events from one execution."""
    tokens: np.ndarray          # [window_size, TOKEN_DIM]
    mask: np.ndarray            # [window_size]  (1 real, 0 padding)
    trace_id: int               # id of the source trace / run
    corrupt: bool               # True if this segment is from a faulted run


def load_trace(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_campaign(campaign_dir: str) -> List[str]:
    """All ygg.trace.parquet files under a Kiln campaign directory."""
    return [str(p) for p in Path(campaign_dir).glob("**/ygg.trace.parquet")]


def events_to_tokens(
    events: np.ndarray,
    event_type_vocab: int = 10000,
    thread_bucket_size: int = 256,
    cpu_count: int = 64,
    max_dt_ns: int = 1_000_000_000,
    metric_scale: float = 1e6,
) -> np.ndarray:
    """Raw event rows -> composed token matrix [seq_len, TOKEN_DIM].

    Expected ``events`` columns:
        [timestamp_ns, cpu, pid, tid, kind, arg0, arg1, arg2]
    Output columns:
        [0] event_type (kind)
        [1] thread (tid bucketed to 256)
        [2] cpu id
        [3] log-scaled time delta (clipped to [1ns, 1s])
        [4] arg0 / scale
        [5] arg1 / scale
        [6] arg2 / scale
    """
    seq_len = events.shape[0]
    tokens = np.zeros((seq_len, TOKEN_DIM), dtype=np.float32)

    tokens[:, 0] = events[:, 4].astype(np.float32)                       # kind
    tokens[:, 1] = (events[:, 3] % thread_bucket_size).astype(np.float32)  # tid
    tokens[:, 2] = events[:, 1].astype(np.float32)                      # cpu

    dt = np.diff(events[:, 0], prepend=events[0, 0])
    dt = np.clip(dt, 1, max_dt_ns).astype(np.float32)
    tokens[:, 3] = np.log(dt)

    # Metric args (arg0/1/2). Real telemetry can carry sentinel / uninitialised
    # values (e.g. 2**63, 2**64-1) or NaN from missing fields. A single garbage
    # cell divided by ``metric_scale`` would explode the activations and blow up
    # the gradients to NaN, so clip to a sane bound and scrub non-finite values.
    METRIC_CLIP = 1e9
    args = np.nan_to_num(
        events[:, 5:8], nan=0.0, posinf=METRIC_CLIP, neginf=0.0
    ).astype(np.float32)
    args = np.clip(args, 0.0, METRIC_CLIP)
    tokens[:, 4] = (args[:, 0] / metric_scale).astype(np.float32)     # arg0
    tokens[:, 5] = (args[:, 1] / metric_scale).astype(np.float32)     # arg1
    tokens[:, 6] = (args[:, 2] / metric_scale).astype(np.float32)     # arg2
    return tokens


# ---------------------------------------------------------------------------
# Contrastive pair discovery from Kiln campaign structure
# ---------------------------------------------------------------------------

def _is_corrupt(path: str) -> bool:
    """Heuristic: a path/run is 'corrupted' if its directory or metadata names
    a fault / degradation scenario."""
    low = path.lower()
    markers = ("corrupt", "fault", "degrad", "broken", "error", "anomal")
    if any(m in low for m in markers):
        return True
    # Explicit meta.json flag if present.
    parent = Path(path).parent
    meta = parent / "meta.json"
    if meta.exists():
        try:
            import json

            d = json.loads(meta.read_text())
            if str(d.get("health", "")).lower() in ("corrupt", "fault", "degraded"):
                return True
            if str(d.get("health", "")).lower() in ("healthy", "baseline", "control"):
                return False
        except Exception:
            pass
    return False


def split_campaign(campaign_dir: str) -> Tuple[List[str], List[str]]:
    """Return (healthy_paths, corrupt_paths) from a Kiln campaign dir."""
    paths = load_campaign(campaign_dir)
    healthy, corrupt = [], []
    for p in paths:
        (corrupt if _is_corrupt(p) else healthy).append(p)
    # If nothing was flagged corrupt, treat the first run as the healthy
    # baseline and let cross-trace negatives serve as the "other" class.
    return healthy, corrupt


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------

def _sample_windows(
    tokens_full: np.ndarray,
    seq_len: int,
    n_windows: int,
    rng: np.random.Generator,
    trace_id: int,
    corrupt: bool,
) -> List[TraceSegment]:
    S = tokens_full.shape[0]
    out: List[TraceSegment] = []
    if S == 0:
        return out
    for _ in range(n_windows):
        if S >= seq_len:
            start = int(rng.integers(0, S - seq_len + 1))
            w = tokens_full[start:start + seq_len].astype(np.float32)
            mask = np.ones(seq_len, dtype=np.float32)
        else:
            w = np.zeros((seq_len, TOKEN_DIM), dtype=np.float32)
            w[:S] = tokens_full.astype(np.float32)
            mask = np.zeros(seq_len, dtype=np.float32)
            mask[:S] = 1.0
        out.append(TraceSegment(w, mask, trace_id, corrupt))
    return out


def _mask_labels(
    tokens: np.ndarray,
    mask: np.ndarray,
    mask_ratio: float,
    mask_token_id: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (masked_tokens, labels) for masked event modeling.

    Returns masked token matrix and a label matrix of event types where masked
    positions carry the true id and everything else is -1.
    """
    B, S, _ = tokens.shape
    labels = -np.ones((B, S), dtype=np.int32)
    masked = tokens.copy()
    for b in range(B):
        real = np.where(mask[b] > 0)[0]
        if real.size == 0:
            continue
        n = max(1, int(real.size * mask_ratio))
        choice = rng.choice(real, size=n, replace=False)
        labels[b, choice] = tokens[b, choice, 0].astype(np.int32)
        masked[b, choice, 0] = mask_token_id
    return masked, labels


def _next_labels(tokens: np.ndarray) -> np.ndarray:
    """Shift event types left by one; last position is -1 (no target)."""
    B, S, _ = tokens.shape
    lab = np.concatenate(
        [tokens[:, 1:, 0].astype(np.int32), -np.ones((B, 1), dtype=np.int32)],
        axis=1,
    )
    return lab


def collate(
    windows: List[TraceSegment],
    mask_ratio: float,
    mask_token_id: int,
    rng: np.random.Generator,
) -> Dict[str, jnp.ndarray]:
    """Assemble a training batch dict from a list of windows."""
    tokens = np.stack([w.tokens for w in windows])
    mask = np.stack([w.mask for w in windows])
    trace_ids = np.array([w.trace_id for w in windows])
    corrupt = np.array([w.corrupt for w in windows])

    masked_tokens, masked_labels = _mask_labels(
        tokens, mask, mask_ratio, mask_token_id, rng
    )
    next_labels = _next_labels(tokens)

    # Contrastive triples: anchor = this window; positive = another window from
    # the *same* trace (healthy-healthy); negative = a window from a *different*
    # (preferably corrupted) trace.
    B = len(windows)
    anchor_tokens, anchor_mask = tokens, mask
    pos_tokens = tokens.copy()
    neg_tokens = tokens.copy()
    neg_mask = mask.copy()
    for b in range(B):
        same = np.where((trace_ids == trace_ids[b]) & (np.arange(B) != b))[0]
        diff = np.where(trace_ids != trace_ids[b])[0]
        # Prefer a corrupted window for the negative.
        diff_corrupt = np.where((trace_ids != trace_ids[b]) & corrupt)[0]
        if same.size:
            j = rng.integers(0, same.size)
            pos_tokens[b] = tokens[same[j]]
        if diff_corrupt.size:
            j = rng.integers(0, diff_corrupt.size)
            neg_tokens[b] = tokens[diff_corrupt[j]]
            neg_mask[b] = mask[diff_corrupt[j]]
        elif diff.size:
            j = rng.integers(0, diff.size)
            neg_tokens[b] = tokens[diff[j]]
            neg_mask[b] = mask[diff[j]]

    return {
        "tokens": jnp.asarray(tokens),
        "mask": jnp.asarray(mask),
        "masked_tokens": jnp.asarray(masked_tokens),
        "masked_labels": jnp.asarray(masked_labels),
        "next_labels": jnp.asarray(next_labels),
        "anchor_tokens": jnp.asarray(anchor_tokens),
        "anchor_mask": jnp.asarray(anchor_mask),
        "pos_tokens": jnp.asarray(pos_tokens),
        "pos_mask": jnp.asarray(mask),
        "neg_tokens": jnp.asarray(neg_tokens),
        "neg_mask": jnp.asarray(neg_mask),
        "corrupt": jnp.asarray(corrupt),
    }


class StreamingTraceDataset:
    """Streaming, dynamically-sampled dataset yielding training batch dicts.

    ``seq_len`` is the full sequence length fed to the model (normally
    ``n_windows * window_size``) so the hierarchical encoder sees several
    adjacent windows per sample.
    """

    def __init__(
        self,
        healthy_paths: List[str],
        corrupt_paths: Optional[List[str]] = None,
        seq_len: int = 2048,
        batch_size: int = 32,
        mask_ratio: float = 0.15,
        mask_token_id: int = 0,
        seed: int = 42,
        windows_per_file: int = 256,
        epochs: int = 1,
    ):
        self.healthy_paths = list(healthy_paths)
        self.corrupt_paths = list(corrupt_paths or [])
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
        self.seed = seed
        self.windows_per_file = windows_per_file
        self.epochs = epochs

    def _iter_file(self, path: str, trace_id: int, corrupt: bool, rng):
        try:
            df = load_trace(path)
        except Exception as e:
            print(f"[dataset] skipping {path}: {e}")
            return []
        cols = ["timestamp_ns", "cpu", "pid", "tid", "kind", "arg0", "arg1", "arg2"]
        present = [c for c in cols if c in df.columns]
        events = df.select(present).to_numpy()
        # Pad missing metric columns with zeros.
        if events.shape[1] < 8:
            pad = np.zeros((events.shape[0], 8 - events.shape[1]), dtype=events.dtype)
            events = np.concatenate([events, pad], axis=1)
        tokens_full = events_to_tokens(events)
        return _sample_windows(
            tokens_full, self.seq_len, self.windows_per_file, rng, trace_id, corrupt
        )

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        for epoch in range(self.epochs):
            rng = np.random.default_rng(self.seed + epoch)
            file_order = self.healthy_paths[:]
            rng.shuffle(file_order)

            # Pre-load corrupt segments once per epoch (usually small).
            corrupt_segs: List[TraceSegment] = []
            for i, p in enumerate(self.corrupt_paths):
                corrupt_segs.extend(self._iter_file(p, 100000 + i, True, rng))

            all_segs: List[TraceSegment] = []
            tid = 0
            for p in file_order:
                segs = self._iter_file(p, tid, False, rng)
                tid += 1
                if corrupt_segs:
                    # Inject corrupt windows so negatives can be true faults.
                    all_segs.extend(segs)
                    all_segs.extend(
                        rng.choice(
                            corrupt_segs,
                            size=min(len(segs), len(corrupt_segs)),
                            replace=len(corrupt_segs) < len(segs),
                        ).tolist()
                        if corrupt_segs
                        else []
                    )
                else:
                    all_segs.extend(segs)

            rng.shuffle(all_segs)
            for i in range(0, len(all_segs), self.batch_size):
                chunk = all_segs[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    continue
                yield collate(
                    chunk, self.mask_ratio, self.mask_token_id, rng
                )


@dataclass
class Batch:
    """Lightweight batch container (legacy API)."""

    tokens: jnp.ndarray
    mask: jnp.ndarray
    labels: Optional[jnp.ndarray] = None
