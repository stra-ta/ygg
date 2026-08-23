"""
Ygg Analysis Package

Provides divergence detection, clustering, and attribution.
"""

from .divergence import (
    DivergencePoint,
    load_embeddings,
    compute_divergence,
    find_divergence_start,
    attribute_divergence,
    diff_traces,
)

from .clustering import (
    ExecutionEmbedding,
    compute_execution_embedding,
    embed_campaign,
    cluster_embeddings,
    reduce_dimensions,
    plot_clusters,
    analyze_clusters,
    find_neighbors,
)

__all__ = [
    "DivergencePoint",
    "load_embeddings",
    "compute_divergence",
    "find_divergence_start",
    "attribute_divergence",
    "diff_traces",
    "ExecutionEmbedding",
    "compute_execution_embedding",
    "embed_campaign",
    "cluster_embeddings",
    "reduce_dimensions",
    "plot_clusters",
    "analyze_clusters",
    "find_neighbors",
]