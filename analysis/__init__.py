"""
Ygg Analysis Package

Divergence detection, gradient attribution, clustering, and static SVG
visualization for execution traces.
"""

from .divergence import (
    DivergencePoint,
    load_embeddings,
    find_divergence,
    find_divergence_start,
    attribute_divergence,
    diff_traces,
    dtw_align,
    detect_change_points,
)

from .clustering import (
    ExecutionEmbedding,
    compute_execution_embedding,
    embed_campaign,
    cluster_embeddings,
    tune_hdbscan,
    reduce_dimensions,
    plot_clusters,
    analyze_clusters,
    build_neighbor_index,
    find_neighbors,
)

from .attribution import (
    integrated_gradients,
    per_token_attribution,
    aggregate_attribution,
    aggregate_continuous,
    integrated_gradients_raw,
    attribute_divergence as attribute_divergence_grad,
    causal_attribution,
)

from .viz import (
    scatter_clusters_svg,
    divergence_timeline_svg,
    attribution_bars_svg,
    embedding_trajectory_svg,
    kiln_comparison_svg,
    generate_report,
)

__all__ = [
    # Divergence
    "DivergencePoint",
    "load_embeddings",
    "find_divergence",
    "find_divergence_start",
    "attribute_divergence",
    "diff_traces",
    "dtw_align",
    "detect_change_points",
    # Clustering
    "ExecutionEmbedding",
    "compute_execution_embedding",
    "embed_campaign",
    "cluster_embeddings",
    "tune_hdbscan",
    "reduce_dimensions",
    "plot_clusters",
    "analyze_clusters",
    "build_neighbor_index",
    "find_neighbors",
    # Attribution
    "integrated_gradients",
    "per_token_attribution",
    "aggregate_attribution",
    "aggregate_continuous",
    "integrated_gradients_raw",
    "attribute_divergence_grad",
    "causal_attribution",
    # Visualization
    "scatter_clusters_svg",
    "divergence_timeline_svg",
    "attribution_bars_svg",
    "embedding_trajectory_svg",
    "kiln_comparison_svg",
    "generate_report",
]
