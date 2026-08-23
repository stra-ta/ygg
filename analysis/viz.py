"""
Ygg Visualization (static SVG only)

Renders analysis results as standalone SVG files - no Streamlit, no
interactive dashboards. All charts are produced with matplotlib's Agg backend
and written to SVG.

Functions:
    scatter_clusters_svg     - UMAP/PCA scatter colored by cluster/fault/workload
    divergence_timeline_svg  - distance vs time with change points marked
    attribution_bars_svg     - top-k attributed features
    embedding_trajectory_svg - PCA of window embeddings over time
    kiln_comparison_svg      - side-by-side trace comparison
    generate_report          - orchestrates a full two-trace report
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _save(fig, output_path: str) -> str:
    fmt = "svg" if str(output_path).endswith(".svg") else "png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return str(output_path)


def scatter_clusters_svg(
    embeddings: Sequence,
    labels: np.ndarray,
    output_path: str,
    *,
    color_by: str = "cluster",
    method: str = "umap",
    title: str = "Execution Clusters",
) -> str:
    """Cluster scatter colored by cluster label, fault plan, or workload."""
    from .clustering import reduce_dimensions

    coords = reduce_dimensions(embeddings, method=method, n_components=2)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    if color_by in ("fault", "workload"):
        values = [
            str(getattr(e, "metadata", {}).get(
                "loki_fault_plan" if color_by == "fault" else "workload", "unknown"
            ))
            for e in embeddings
        ]
        unique = sorted(set(values))
        cmap = plt.get_cmap("tab20")
        color_map = {v: cmap(i / max(1, len(unique))) for i, v in enumerate(unique)}
        for v in unique:
            mask = np.array([x == v for x in values])
            ax.scatter(coords[mask, 0], coords[mask, 1], label=v, s=50, alpha=0.8,
                       color=color_map[v])
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        unique = np.unique(labels)
        cmap = plt.get_cmap("tab20")
        for label in unique:
            mask = labels == label
            if label == -1:
                ax.scatter(coords[mask, 0], coords[mask, 1], c=[(0.5, 0.5, 0.5, 0.5)],
                           label="noise", s=50)
            else:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           color=cmap(int(label) % 20), label=f"cluster {label}", s=50)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    plt.tight_layout()
    return _save(fig, output_path)


def divergence_timeline_svg(
    divergences: Sequence,
    output_path: str,
    *,
    threshold: float = 0.1,
    change_points: Optional[List[int]] = None,
    start: Optional = None,
    title: str = "Divergence Timeline",
) -> str:
    """Distance vs time with the threshold, change points, and start marked."""
    ts = np.array([float(d.timestamp_ns) for d in divergences]) / 1e6  # ms
    dist = np.array([float(d.distance) for d in divergences])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ts, dist, lw=1.0, color="#1f77b4", label="distance")
    ax.axhline(threshold, color="red", ls="--", lw=1.0, label=f"threshold ({threshold})")

    if change_points:
        for cp in change_points:
            cp = min(cp, len(ts) - 1)
            ax.axvline(ts[cp], color="orange", ls=":", lw=1.0)

    if start is not None:
        st = float(getattr(start, "timestamp_ns", start)) / 1e6
        ax.axvline(st, color="green", ls="-.", lw=1.5, label="divergence start")

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("cosine distance")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return _save(fig, output_path)


def attribution_bars_svg(
    attribution: Dict[str, float],
    output_path: str,
    *,
    top_k: int = 20,
    title: str = "Divergence Attribution",
) -> str:
    """Horizontal bar chart of the top-k attributed features."""
    items = sorted(attribution.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    if not items:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "no attribution", ha="center")
        return _save(fig, output_path)

    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]

    fig, ax = plt.subplots(figsize=(9, max(2, 0.35 * len(names) + 1)))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("attribution (red = negative contribution)")
    ax.set_title(title)
    plt.tight_layout()
    return _save(fig, output_path)


def embedding_trajectory_svg(
    emb1: np.ndarray,
    ts1: np.ndarray,
    emb2: np.ndarray,
    ts2: np.ndarray,
    output_path: str,
    *,
    title: str = "Embedding Trajectory",
) -> str:
    """PCA of window embeddings over time, drawn as two trajectories."""
    emb1 = np.asarray(emb1, dtype=float)
    emb2 = np.asarray(emb2, dtype=float)
    combined = np.vstack([emb1, emb2])
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)
    p1 = proj[: len(emb1)]
    p2 = proj[len(emb1):]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(p1[:, 0], p1[:, 1], color="#1f77b4", lw=1.5, label="trace1")
    ax.plot(p2[:, 0], p2[:, 1], color="#ff7f0e", lw=1.5, label="trace2")
    if len(p1):
        ax.scatter([p1[0, 0]], [p1[0, 1]], color="#1f77b4", s=80, marker="o", zorder=5)
        ax.scatter([p1[-1, 0]], [p1[-1, 1]], color="#1f77b4", s=80, marker="x", zorder=5)
    if len(p2):
        ax.scatter([p2[0, 0]], [p2[0, 1]], color="#ff7f0e", s=80, marker="o", zorder=5)
        ax.scatter([p2[-1, 0]], [p2[-1, 1]], color="#ff7f0e", s=80, marker="x", zorder=5)

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    return _save(fig, output_path)


def kiln_comparison_svg(
    trace1: str,
    trace2: str,
    output_path: str,
    *,
    title: str = "Kiln Comparison",
) -> str:
    """Side-by-side comparison of two traces' behavioral statistics."""
    from .clustering import _feature_statistics

    f1 = _feature_statistics(trace1)
    f2 = _feature_statistics(trace2)

    kind1 = f1.get("kind_rates", {})
    kind2 = f2.get("kind_rates", {})
    all_kinds = sorted(set(kind1) | set(kind2), key=lambda x: str(x))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.4 * len(all_kinds) + 2)))
    for ax, fs, name in zip(axes, (f1, f2), ("trace1", "trace2")):
        rates = fs.get("kind_rates", {})
        names = [str(k) for k in all_kinds]
        vals = [rates.get(k, 0.0) for k in all_kinds]
        ax.barh(range(len(names)), vals, color="#1f77b4")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{name}\n{fs.get('events', 0)} events, "
                     f"{fs.get('events_per_s', 0.0):.0f}/s")
        ax.set_xlabel("event kind rate")

    fig.suptitle(title)
    plt.tight_layout()
    return _save(fig, output_path)


def generate_report(
    trace1: str,
    trace2: str,
    model,
    params: dict,
    output_dir: str,
    *,
    segment_len: int = 512,
    stride: int = 256,
    threshold: float = 0.1,
    change_method: str = "pelt",
) -> Dict:
    """
    Produce a full two-trace report as static SVGs in ``output_dir``.

    Writes: divergence_timeline.svg, attribution_bars.svg,
    embedding_trajectory.svg, kiln_comparison.svg. Returns a summary dict.
    """
    from . import divergence as div
    from .clustering import _feature_statistics

    output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Divergence.
    divergences = div.find_divergence(
        trace1, trace2, model, params,
        segment_len=segment_len, stride=stride, threshold=threshold,
    )
    start = div.find_divergence_start(divergences, threshold=threshold)

    change_points: Optional[List[int]] = None
    if divergences:
        import numpy as _np
        dist = _np.array([d.distance for d in divergences])
        change_points = div.detect_change_points(dist, method=change_method)

    timeline_path = str(Path(output_dir) / "divergence_timeline.svg")
    divergence_timeline_svg(
        divergences, timeline_path, threshold=threshold,
        change_points=change_points, start=start,
    )

    # Attribution.
    attribution: Dict[str, float] = {}
    if start is not None:
        attribution = div.attribute_divergence(model, params, trace1, trace2, start)
        attr_path = str(Path(output_dir) / "attribution_bars.svg")
        attribution_bars_svg(attribution, attr_path, top_k=20)

    # Trajectory.
    traj_path = str(Path(output_dir) / "embedding_trajectory.svg")
    emb1, ts1 = div.load_embeddings(model, params, trace1, segment_len, stride)
    emb2, ts2 = div.load_embeddings(model, params, trace2, segment_len, stride)
    embedding_trajectory_svg(np.asarray(emb1), np.asarray(ts1),
                              np.asarray(emb2), np.asarray(ts2), traj_path)

    # Kiln comparison.
    kiln_path = str(Path(output_dir) / "kiln_comparison.svg")
    kiln_comparison_svg(trace1, trace2, kiln_path)

    return {
        "output_dir": output_dir,
        "num_divergences": len(divergences),
        "divergence_start": (
            {
                "timestamp_ns": start.timestamp_ns,
                "distance": start.distance,
                "confidence": start.confidence,
            }
            if start is not None else None
        ),
        "attribution": attribution,
        "files": {
            "divergence_timeline": timeline_path,
            "attribution_bars": str(Path(output_dir) / "attribution_bars.svg") if attribution else None,
            "embedding_trajectory": traj_path,
            "kiln_comparison": kiln_path,
        },
        "trace1_stats": _feature_statistics(trace1),
        "trace2_stats": _feature_statistics(trace2),
    }
