"""
Ygg Dataset Loader

Loads Parquet traces into JAX-compatible format for training.
"""

import jax.numpy as jnp
import numpy as np
import polars as pl
from pathlib import Path
from typing import Iterator, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class TraceSegment:
    """A contiguous segment of events from a single execution."""
    events: jnp.ndarray          # [seq_len, 8] - timestamp, cpu, pid, tid, kind, arg0, arg1, arg2
    metadata: dict               # Execution metadata
    start_idx: int               # Global event index
    end_idx: int                 # Global event index (exclusive)


@dataclass
class Batch:
    """Training batch."""
    tokens: jnp.ndarray          # [batch, seq_len, token_dim]
    mask: jnp.ndarray            # [batch, seq_len] - 1 for real, 0 for padding
    labels: Optional[jnp.ndarray] = None  # For supervised objectives


def load_trace(parquet_path: str) -> pl.DataFrame:
    """Load a single Parquet trace file."""
    return pl.read_parquet(parquet_path)


def load_campaign(campaign_dir: str) -> List[pl.DataFrame]:
    """Load all traces from a Kiln campaign directory."""
    traces = []
    for p in Path(campaign_dir).glob("*/ygg.trace.parquet"):
        traces.append(pl.read_parquet(p))
    return traces


def events_to_tokens(
    events: np.ndarray,
    event_type_vocab: int = 10000,
    thread_bucket_size: int = 256,
    cpu_count: int = 64,
    max_dt_ns: int = 1_000_000_000,  # 1 second
    metric_scale: float = 1e6,
) -> np.ndarray:
    """
    Convert raw events to token embeddings.

    Each event becomes a composed token:
    - event_type_embedding [event_type_vocab]
    - thread_embedding [thread_bucket_size]
    - cpu_embedding [cpu_count]
    - dt_projection [1] - log-scaled time delta
    - metric_projection [3] - scaled arg0, arg1, arg2

    Returns: [seq_len, token_dim]
    """
    seq_len = events.shape[0]
    token_dim = 5  # Will be projected to d_model by embedding layer

    tokens = np.zeros((seq_len, token_dim), dtype=np.float32)

    # Event type (categorical)
    tokens[:, 0] = events[:, 4]  # kind

    # Thread ID (bucketed)
    tokens[:, 1] = (events[:, 3] % thread_bucket_size).astype(np.float32)

    # CPU ID
    tokens[:, 2] = events[:, 1].astype(np.float32)

    # Time delta (log-scaled)
    dt = np.diff(events[:, 0], prepend=events[0, 0])
    dt = np.clip(dt, 1, max_dt_ns)
    tokens[:, 3] = np.log(dt.astype(np.float32))

    # Metrics (scaled)
    tokens[:, 4] = (events[:, 5] / metric_scale).astype(np.float32)  # arg0

    return tokens


def create_segments(
    events: np.ndarray,
    segment_len: int = 512,
    stride: int = 256,
) -> List[TraceSegment]:
    """Split events into overlapping segments."""
    segments = []
    for start in range(0, len(events) - segment_len + 1, stride):
        end = start + segment_len
        seg_events = events[start:end]
        segments.append(TraceSegment(
            events=jnp.array(seg_events),
            metadata={},
            start_idx=start,
            end_idx=end,
        ))
    return segments


class TraceDataset:
    """Iterable dataset for training."""

    def __init__(
        self,
        trace_paths: List[str],
        segment_len: int = 512,
        stride: int = 256,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.trace_paths = trace_paths
        self.segment_len = segment_len
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._segments = []

    def __iter__(self) -> Iterator[Batch]:
        # Load all segments
        for path in self.trace_paths:
            df = load_trace(path)
            events = df.select([
                "timestamp_ns", "cpu", "pid", "tid", "kind",
                "arg0", "arg1", "arg2"
            ]).to_numpy()
            tokens = events_to_tokens(events)
            segments = create_segments(tokens, self.segment_len, self.stride)
            self._segments.extend(segments)

        if self.shuffle:
            np.random.shuffle(self._segments)

        # Yield batches
        for i in range(0, len(self._segments), self.batch_size):
            batch_segments = self._segments[i:i + self.batch_size]
            if len(batch_segments) < self.batch_size:
                continue

            batch_tokens = jnp.stack([jnp.array(s.events) for s in batch_segments])
            batch_mask = jnp.ones((self.batch_size, self.segment_len))

            yield Batch(tokens=batch_tokens, mask=batch_mask)

    def __len__(self) -> int:
        return len(self._segments) // self.batch_size


def masked_event_modeling_batch(
    batch: Batch,
    mask_ratio: float = 0.15,
    mask_token_id: int = 0,
    vocab_size: int = 10000,
) -> Tuple[Batch, jnp.ndarray]:
    """
    Prepare batch for masked event modeling (BERT-style).

    Returns: (masked_batch, labels)
    """
    tokens = batch.tokens
    batch_size, seq_len, _ = tokens.shape

    # Create mask
    mask = jnp.zeros_like(tokens[:, :, 0], dtype=bool)
    for b in range(batch_size):
        n_mask = int(seq_len * mask_ratio)
        mask_indices = np.random.choice(seq_len, n_mask, replace=False)
        mask = mask.at[b, mask_indices].set(True)

    # Labels are the original event types at masked positions
    labels = jnp.where(mask, tokens[:, :, 0], -1)

    # Replace masked tokens with [MASK] token
    masked_tokens = tokens.at[:, :, 0].set(
        jnp.where(mask, mask_token_id, tokens[:, :, 0])
    )

    return Batch(tokens=masked_tokens, mask=batch.mask), labels


def next_event_prediction_batch(
    batch: Batch,
) -> Tuple[Batch, jnp.ndarray]:
    """
    Prepare batch for next-event prediction (GPT-style).

    Returns: (input_batch, next_event_labels)
    """
    tokens = batch.tokens
    # Input: all but last
    inputs = tokens[:, :-1, :]
    # Labels: event types of next positions
    labels = tokens[:, 1:, 0]

    return Batch(tokens=inputs, mask=batch.mask[:, :-1]), labels


def contrastive_pairs(
    healthy_traces: List[str],
    corrupted_traces: List[str],
    segment_len: int = 512,
    stride: int = 256,
) -> Iterator[Tuple[Batch, Batch, jnp.ndarray]]:
    """
    Generate contrastive pairs for Objective 3.

    Yields: (anchor_batch, positive_batch, negative_batch), labels
    where labels = 1 for (anchor, positive), 0 for (anchor, negative)
    """
    healthy_dataset = TraceDataset(healthy_traces, segment_len, stride, shuffle=False)
    corrupted_dataset = TraceDataset(corrupted_traces, segment_len, stride, shuffle=False)

    healthy_iter = iter(healthy_dataset)
    corrupted_iter = iter(corrupted_dataset)

    while True:
        try:
            anchor = next(healthy_iter)
            positive = next(healthy_iter)
            negative = next(corrupted_iter)

            # Labels: 1 for similar, 0 for dissimilar
            labels = jnp.array([1] * len(anchor.tokens) + [0] * len(anchor.tokens))

            yield anchor, positive, negative, labels
        except StopIteration:
            break