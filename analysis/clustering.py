"""
Ygg Clustering Analysis

Cluster executions by behavioral similarity, tune HDBSCAN automatically,
reduce dimensions for visualization, characterize clusters, and search
neighbors with a FAISS index (with a sklearn fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import jax.numpy as jnp

import umap
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA

from model.encoder import ExecutionEmbedder
from model.dataset import events_to_tokens
from .divergence import _window_tokens


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionEmbedding:
    """Embedding for a single execution."""

    trace_path: str
    embedding: np.ndarray  # [d_model]
    metadata: dict = field(default_factory=dict)
    n_segments: int = 0


def compute_execution_embedding(
    model: ExecutionEmbedder,
    params: dict,
    trace_path: str,
    segment_len: int = 512,
    stride: int = 256,
    pool: str = "mean",
) -> ExecutionEmbedding:
    """Compute a single embedding for an entire execution."""
    df = pl.read_parquet(trace_path)

    metadata: dict = {}
    if "metadata" in df.columns:
        meta_df = df.select("metadata").to_dicts()
        if meta_df and meta_df[0].get("metadata"):
            metadata = meta_df[0]["metadata"]

    events = df.select(
        ["timestamp_ns", "cpu", "pid", "tid", "kind", "arg0", "arg1", "arg2"]
    ).to_numpy()

    if len(events) == 0:
        return ExecutionEmbedding(trace_path, np.zeros(256), metadata, 0)

    tokens = events_to_tokens(events)
    windows = _window_tokens(tokens, segment_len, stride)

    if not windows:
        return ExecutionEmbedding(trace_path, np.zeros(256), metadata, 0)

    embeddings = []
    for w, mask, _ in windows:
        batch = w[None, ...]
        emb = model.apply({"params": params}, batch, mask[None, :], train=False)
        embeddings.append(np.asarray(emb[0]))

    emb_array = np.stack(embeddings)

    if pool == "mean":
        exec_emb = emb_array.mean(axis=0)
    elif pool == "max":
        exec_emb = emb_array.max(axis=0)
    else:
        exec_emb = emb_array[0]

    return ExecutionEmbedding(trace_path, exec_emb, metadata, len(embeddings))


def embed_campaign(
    model: ExecutionEmbedder,
    params: dict,
    campaign_dir: str,
    segment_len: int = 512,
    stride: int = 256,
) -> List[ExecutionEmbedding]:
    """Embed all traces in a Kiln campaign directory."""
    embeddings: List[ExecutionEmbedding] = []
    for trace_path in Path(campaign_dir).glob("*/ygg.trace.parquet"):
        embeddings.append(
            compute_execution_embedding(model, params, str(trace_path), segment_len, stride)
        )
    return embeddings


# --------------------------------------------------------------------------- #
# Dimensionality reduction
# --------------------------------------------------------------------------- #
def reduce_dimensions(
    embeddings: Sequence[ExecutionEmbedding],
    method: str = "umap",
    n_components: int = 2,
) -> np.ndarray:
    """Reduce embeddings to ``n_components`` for visualization."""
    X = np.stack([np.asarray(e.embedding, dtype=float) for e in embeddings])
    if X.shape[0] < 2:
        return X

    if method == "umap":
        reducer = umap.UMAP(
            n_components=n_components, n_neighbors=15, min_dist=0.1, metric="euclidean"
        )
    elif method == "pca":
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(X)


# --------------------------------------------------------------------------- #
# HDBSCAN clustering with stability-based auto-tuning
# --------------------------------------------------------------------------- #
def _stability_score(clusterer: HDBSCAN, labels: np.ndarray) -> float:
    """Composite stability score: persistence * (1 - noise fraction)."""
    n = len(labels)
    noise = int((labels == -1).sum())
    noise_frac = noise / n if n else 1.0

    persistence = getattr(clusterer, "cluster_persistence_", None)
    if persistence is None:
        pers = 0.0
    elif isinstance(persistence, dict):
        pers = float(np.mean(list(persistence.values()))) if persistence else 0.0
    else:
        pers = float(np.mean(persistence)) if len(persistence) else 0.0

    n_clusters = len(set(labels.tolist()) - {-1})
    # Reward clean, well-separated, multi-cluster structure.
    return pers * (1.0 - noise_frac) + 0.01 * n_clusters


def tune_hdbscan(
    embeddings: Sequence[ExecutionEmbedding],
    min_cluster_size_range: Sequence[int] = (3, 5, 8, 12, 16, 24, 32),
    min_samples: int = 2,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, HDBSCAN, int]:
    """
    Auto-tune ``min_cluster_size`` via stability selection.

    Sweeps candidate sizes, fits HDBSCAN for each, and returns the labels and
    clusterer for the configuration with the highest stability score.
    """
    X = np.stack([np.asarray(e.embedding, dtype=float) for e in embeddings])
    n = len(X)
    if n < 2:
        return np.zeros(n, dtype=int), HDBSCAN(), 0

    best: Optional[Tuple[float, np.ndarray, HDBSCAN, int]] = None
    for mcs in min_cluster_size_range:
        if mcs < 2 or mcs > n:
            continue
        clusterer = HDBSCAN(
            min_cluster_size=int(mcs),
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(X)
        score = _stability_score(clusterer, labels)
        if best is None or score > best[0]:
            best = (score, labels, clusterer, int(mcs))

    if best is None:
        labels = np.zeros(n, dtype=int)
        return labels, HDBSCAN(), 0
    return best[1], best[2], best[3]


def cluster_embeddings(
    embeddings: Sequence[ExecutionEmbedding],
    min_cluster_size: Optional[int] = None,
    min_samples: int = 2,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, HDBSCAN]:
    """
    Cluster execution embeddings using HDBSCAN.

    If ``min_cluster_size`` is None, it is auto-tuned via
    :func:`tune_hdbscan`. Returns ``(labels, clusterer)``.
    """
    X = np.stack([np.asarray(e.embedding, dtype=float) for e in embeddings])
    n = len(X)
    if n < 2:
        return np.zeros(n, dtype=int), HDBSCAN()

    if min_cluster_size is None:
        labels, clusterer, _ = tune_hdbscan(
            embeddings, min_samples=min_samples, metric=metric
        )
        return labels, clusterer

    clusterer = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X)
    return labels, clusterer


# --------------------------------------------------------------------------- #
# Cluster characterization
# --------------------------------------------------------------------------- #
def _medoid_index(X: np.ndarray) -> int:
    """Index of the medoid (min average distance to all other points)."""
    n = X.shape[0]
    if n == 1:
        return 0
    # Use cosine distance on L2-normalized rows.
    Xn = X / np.linalg.norm(X, axis=-1, keepdims=True)
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    return int(np.argmin(dist.sum(axis=1)))


def _feature_statistics(trace_path: str) -> Dict:
    """Per-trace behavioral statistics used to characterize clusters."""
    try:
        df = pl.read_parquet(trace_path)
    except Exception:
        return {}

    if df.height == 0:
        return {}

    ts = df["timestamp_ns"].to_numpy().astype(np.int64)
    duration_ns = int(ts.max() - ts.min())
    duration_s = duration_ns / 1e9 if duration_ns > 0 else 1.0

    stats: Dict = {
        "events": int(df.height),
        "duration_ns": duration_ns,
        "events_per_s": df.height / duration_s,
    }

    # Event kind distribution.
    kind_counts = df["kind"].value_counts()
    total = df.height
    stats["kind_rates"] = {
        int(r["kind"]): float(r["count"] / total) for r in kind_counts.iter_rows(named=True)
    }

    # CPU utilization (fraction of events per cpu).
    if "cpu" in df.columns:
        cpu_counts = df["cpu"].value_counts()
        stats["cpu_util"] = {
            int(r["cpu"]): float(r["count"] / total) for r in cpu_counts.iter_rows(named=True)
        }

    # arg0 distribution per dominant kind.
    if "arg0" in df.columns and df.height > 0:
        stats["arg0_mean"] = float(df["arg0"].mean())
        stats["arg0_p99"] = float(df["arg0"].quantile(0.99))

    return stats


def analyze_clusters(
    embeddings: Sequence[ExecutionEmbedding],
    labels: np.ndarray,
) -> Dict:
    """Analyze cluster composition and per-cluster characteristics."""
    X = np.stack([np.asarray(e.embedding, dtype=float) for e in embeddings])
    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    analysis: Dict = {}
    for label, members in clusters.items():
        if label == -1:
            continue
        embs = [embeddings[i] for i in members]
        workloads = [m.metadata.get("workload", "unknown") for m in embs]
        faults = [m.metadata.get("loki_fault_plan", "none") for m in embs]
        git_shas = [str(m.metadata.get("git_sha", ""))[:8] for m in embs]

        member_X = X[members]
        medoid = _medoid_index(member_X)
        prototype = embs[medoid].trace_path

        # Aggregate feature statistics across members.
        feat: Dict = {"events_per_s": [], "kind_rates": {}, "cpu_util": {}}
        for i in members:
            fs = _feature_statistics(embeddings[i].trace_path)
            if not fs:
                continue
            feat["events_per_s"].append(fs.get("events_per_s", 0.0))
            for k, v in fs.get("kind_rates", {}).items():
                feat["kind_rates"][k] = feat["kind_rates"].get(k, 0.0) + v
            for c, v in fs.get("cpu_util", {}).items():
                feat["cpu_util"][c] = feat["cpu_util"].get(c, 0.0) + v

        n_members = max(1, len(feat["events_per_s"]))
        analysis[f"cluster_{label}"] = {
            "size": len(members),
            "workloads": {w: workloads.count(w) for w in set(workloads)},
            "faults": {f: faults.count(f) for f in set(faults)},
            "git_shas": {g: git_shas.count(g) for g in set(git_shas)},
            "prototype": prototype,
            "medoid_idx": members[medoid],
            "mean_events_per_s": float(np.mean(feat["events_per_s"])) if feat["events_per_s"] else 0.0,
            "kind_rates": {k: v / n_members for k, v in feat["kind_rates"].items()},
            "cpu_util": {k: v / n_members for k, v in feat["cpu_util"].items()},
            "members": [m.trace_path for m in embs],
        }

    return analysis


# --------------------------------------------------------------------------- #
# Neighbor search (FAISS with sklearn fallback)
# --------------------------------------------------------------------------- #
def build_neighbor_index(
    embeddings: Sequence[ExecutionEmbedding],
    use_faiss: bool = True,
):
    """
    Build a k-NN index over execution embeddings.

    Prefers FAISS (IndexFlatL2 on L2-normalized vectors so that squared
    distance maps cleanly to cosine similarity). Falls back to sklearn
    ``NearestNeighbors`` if FAISS is unavailable or fails.
    """
    X = np.stack([np.asarray(e.embedding, dtype=float) for e in embeddings]).astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)

    if use_faiss:
        try:
            import faiss

            index = faiss.IndexFlatL2(Xn.shape[1])
            index.add(Xn)
            return ("faiss", index, Xn)
        except Exception:
            use_faiss = False

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(Xn)
    return ("sklearn", nn, Xn)


def find_neighbors(
    embeddings: Sequence[ExecutionEmbedding],
    query_idx: int,
    k: int = 5,
    index=None,
) -> List[Tuple[int, float]]:
    """
    Find the ``k`` nearest neighbors of ``query_idx``.

    Returns a list of ``(index, similarity)`` sorted by descending cosine
    similarity. ``index`` may be a prebuilt index from
    :func:`build_neighbor_index`.
    """
    if index is None:
        index = build_neighbor_index(embeddings)

    kind, idx, Xn = index
    q = Xn[query_idx]

    if kind == "faiss":
        import faiss

        D, I = idx.search(q[None].astype(np.float32), k + 1)
        out = []
        for d, i in zip(D[0], I[0]):
            i = int(i)
            if i == query_idx or i < 0:
                continue
            sim = 1.0 - float(d) / 2.0  # L2^2 = 2(1 - cos) for unit vectors
            out.append((i, sim))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    dists, inds = idx.kneighbors(Xn[query_idx][None])
    out = []
    for d, i in zip(dists[0], inds[0]):
        i = int(i)
        if i == query_idx:
            continue
        sim = 1.0 - float(d)  # cosine distance -> similarity
        out.append((i, sim))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:k]


# --------------------------------------------------------------------------- #
# Visualization (kept for parity; SVG reports live in analysis.viz)
# --------------------------------------------------------------------------- #
def plot_clusters(
    embeddings: Sequence[ExecutionEmbedding],
    labels: np.ndarray,
    output_path: str,
    method: str = "umap",
    title: str = "Execution Clusters",
) -> str:
    """Generate a cluster visualization, choosing format from the extension."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = reduce_dimensions(embeddings, method=method, n_components=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        if label == -1:
            label_str = "noise"
            color = (0.5, 0.5, 0.5, 0.5)
        else:
            label_str = f"cluster {label}"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], label=label_str, alpha=0.7, s=50,
        )

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    fmt = "svg" if str(output_path).endswith(".svg") else "png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", format=fmt)
    plt.close()
    return str(output_path)
