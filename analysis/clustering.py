"""
Ygg Clustering Analysis

Clusters executions by behavioral similarity.
"""

import jax.numpy as jnp
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import HDBSCAN
from sklearn.manifold import UMAP
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.encoder import ExecutionEmbedder, EncoderConfig
from model.dataset import events_to_tokens, create_segments


@dataclass
class ExecutionEmbedding:
    """Embedding for a single execution."""
    trace_path: str
    embedding: jnp.ndarray  # [d_model]
    metadata: dict
    n_segments: int


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

    # Extract metadata
    metadata = {}
    if "metadata" in df.columns:
        meta_df = df.select("metadata").to_dicts()
        if meta_df:
            metadata = meta_df[0]

    events = df.select([
        "timestamp_ns", "cpu", "pid", "tid", "kind",
        "arg0", "arg1", "arg2"
    ]).to_numpy()

    if len(events) == 0:
        return ExecutionEmbedding(trace_path, jnp.zeros(256), metadata, 0)

    tokens = events_to_tokens(events)
    segments = create_segments(tokens, segment_len, stride)

    if not segments:
        return ExecutionEmbedding(trace_path, jnp.zeros(256), metadata, 0)

    embeddings = []
    for seg in segments:
        batch = seg.events[None, ...]
        mask = jnp.ones((1, segment_len))
        emb = model.apply({"params": params}, batch, mask, train=False)
        embeddings.append(emb[0])

    emb_array = jnp.stack(embeddings)

    # Pool across segments
    if pool == "mean":
        exec_emb = emb_array.mean(axis=0)
    elif pool == "max":
        exec_emb = emb_array.max(axis=0)
    else:
        exec_emb = emb_array[0]

    return ExecutionEmbedding(trace_path, exec_emb, metadata, len(segments))


def embed_campaign(
    model: ExecutionEmbedder,
    params: dict,
    campaign_dir: str,
    segment_len: int = 512,
    stride: int = 256,
) -> List[ExecutionEmbedding]:
    """Embed all traces in a Kiln campaign directory."""
    embeddings = []
    for trace_path in Path(campaign_dir).glob("*/ygg.trace.parquet"):
        emb = compute_execution_embedding(model, params, str(trace_path), segment_len, stride)
        embeddings.append(emb)
    return embeddings


def cluster_embeddings(
    embeddings: List[ExecutionEmbedding],
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> Tuple[np.ndarray, HDBSCAN]:
    """
    Cluster execution embeddings using HDBSCAN.

    Returns: (labels, clusterer)
    """
    X = np.stack([e.embedding for e in embeddings])

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)

    return labels, clusterer


def reduce_dimensions(
    embeddings: List[ExecutionEmbedding],
    method: str = "umap",
    n_components: int = 2,
) -> np.ndarray:
    """Reduce embeddings for visualization."""
    X = np.stack([e.embedding for e in embeddings])

    if method == "umap":
        reducer = UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, metric="euclidean")
    elif method == "pca":
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(X)


def plot_clusters(
    embeddings: List[ExecutionEmbedding],
    labels: np.ndarray,
    output_path: str,
    method: str = "umap",
    title: str = "Execution Clusters",
):
    """Generate cluster visualization."""
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
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            label=label_str,
            alpha=0.7,
            s=50,
        )

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved cluster plot to {output_path}")


def analyze_clusters(
    embeddings: List[ExecutionEmbedding],
    labels: np.ndarray,
) -> Dict:
    """Analyze cluster composition."""
    clusters = {}
    for emb, label in zip(embeddings, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(emb)

    analysis = {}
    for label, members in clusters.items():
        if label == -1:
            continue

        # Collect metadata
        workloads = [m.metadata.get("workload", "unknown") for m in members]
        faults = [m.metadata.get("loki_fault_plan", "none") for m in members]
        git_shas = [m.metadata.get("git_sha", "")[:8] for m in members]

        analysis[f"cluster_{label}"] = {
            "size": len(members),
            "workloads": {w: workloads.count(w) for w in set(workloads)},
            "faults": {f: faults.count(f) for f in set(faults)},
            "git_shas": {g: git_shas.count(g) for g in set(git_shas)},
            "members": [m.trace_path for m in members],
        }

    return analysis


def find_neighbors(
    embeddings: List[ExecutionEmbedding],
    query_idx: int,
    k: int = 5,
) -> List[Tuple[int, float]]:
    """Find k nearest neighbors by cosine similarity."""
    query_emb = embeddings[query_idx].embedding
    query_norm = query_emb / jnp.linalg.norm(query_emb)

    distances = []
    for i, emb in enumerate(embeddings):
        if i == query_idx:
            continue
        emb_norm = emb.embedding / jnp.linalg.norm(emb.embedding)
        sim = float(jnp.dot(query_norm, emb_norm))
        distances.append((i, sim))

    distances.sort(key=lambda x: x[1], reverse=True)
    return distances[:k]