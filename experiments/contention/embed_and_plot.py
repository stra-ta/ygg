#!/usr/bin/env python3
"""
Compute execution embeddings for synthetic Norn contention traces and plot them.

This is the *verification* stage of the V0.1 validation harness. It loads the
encoder trained by ``train_synthetic.py`` (no regime labels were used during
training), computes one execution-level embedding per synthetic trace, reduces
them to 2D, and plots them colored by regime and shaped by thread count.

The point of the figure: if the label-free encoder has learned a useful
representation, the four contention regimes should form visibly separated
clusters. UMAP is used when available; PCA is the deterministic fallback.

Outputs:
    experiments/contention/results/contention_embeddings.svg
    experiments/contention/results/metrics.json
"""

import json
import sys
from pathlib import Path

# Ensure the repo root (which owns the `model` package) is importable when this
# file is run as a script: python experiments/contention/embed_and_plot.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")  # static SVG only; no display backend
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import jax.numpy as jnp

from model.config import ModelConfig
from model.encoder import ExecutionEmbedder
from model.dataset import events_to_tokens

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SYNTHETIC_DIR = Path(__file__).resolve().parent / "synthetic"
CHECKPOINT = RESULTS_DIR / "encoder.msgpack"

TOKEN_DIM = 7
METRIC_SCALE = 1.0  # must match train_synthetic.py

REGIMES = ["tight", "yield", "bounded", "exponential"]
GRIDS = [1, 2, 4, 8]
REGIME_COLORS = {
    "tight": "#e6194b",
    "yield": "#3cb44b",
    "bounded": "#4363d8",
    "exponential": "#f58231",
}
# Marker per thread count (grid 1x1..8x8 -> threads 1,4,16,64).
THREAD_GRID = {1: "o", 2: "s", 4: "^", 8: "D"}


def load_checkpoint(path: Path):
    from flax import serialization as fser

    with open(path, "rb") as f:
        blob = f.read()
    target = {"params": None, "opt_state": None, "step": 0, "config": ""}
    ck = fser.from_bytes(target, blob)
    # Config was stored as a JSON string (flax drops nested-dict structure on
    # restore), so decode it back into a dict here.
    ck["config"] = json.loads(ck["config"])
    return ck


def embed_trace(path: Path, embedder, params, config) -> np.ndarray:
    """Return the execution-level embedding [d] for one trace."""
    df = pl.read_parquet(path)
    cols = ["timestamp_ns", "cpu", "pid", "tid", "kind", "arg0", "arg1", "arg2"]
    present = [c for c in cols if c in df.columns]
    events = df.select(present).to_numpy()
    if events.shape[1] < 8:
        pad = np.zeros((events.shape[0], 8 - events.shape[1]), dtype=events.dtype)
        events = np.concatenate([events, pad], axis=1)
    tokens = events_to_tokens(events, metric_scale=METRIC_SCALE).astype(np.float32)
    tokens = jnp.asarray(tokens)[None]  # [1, S, 7]
    mask = jnp.ones((1, tokens.shape[1]), dtype=jnp.float32)
    emb = embedder.apply({"params": params}, tokens, mask, train=False)
    return np.asarray(emb)[0]


def reduce(X: np.ndarray):
    """UMAP (preferred) with PCA fallback. Returns (XY [n,2], method)."""
    try:
        from umap import UMAP

        reducer = UMAP(
            n_neighbors=min(4, X.shape[0] - 1),
            min_dist=0.3,
            n_components=2,
            random_state=0,
            metric="euclidean",
        )
        XY = reducer.fit_transform(X)
        return XY, "umap"
    except Exception as e:  # noqa: BLE001 - fall back to PCA on any UMAP failure
        print(f"[plot] UMAP unavailable ({e}); falling back to PCA")
        from sklearn.decomposition import PCA

        XY = PCA(n_components=2, random_state=0).fit_transform(X)
        return XY, "pca"


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"No checkpoint at {CHECKPOINT}. Run train_synthetic.py first.")

    ck = load_checkpoint(CHECKPOINT)
    config = ModelConfig.from_dict(ck["config"])
    params = ck["params"]

    embedder = ExecutionEmbedder(config, pool=config.pool)

    X_rows = []
    regimes = []
    thread_counts = []
    for regime in REGIMES:
        for grid in GRIDS:
            threads = grid * grid
            path = SYNTHETIC_DIR / f"{regime}_{threads}x{threads}.parquet"
            if not path.exists():
                raise SystemExit(f"Missing synthetic trace {path}")
            emb = embed_trace(path, embedder, params, config)
            X_rows.append(emb)
            regimes.append(regime)
            thread_counts.append(threads)

    X = np.stack(X_rows, axis=0)  # [n, d]
    regimes = np.array(regimes)
    thread_counts = np.array(thread_counts)

    XY, method = reduce(X)

    # --- Silhouette by regime ---
    from sklearn.metrics import silhouette_score

    try:
        sil = float(silhouette_score(X, regimes))
    except Exception as e:  # noqa: BLE001
        print(f"[plot] silhouette failed: {e}")
        sil = float("nan")

    separated = not np.isnan(sil) and sil > 0.25

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    legend_handles = []
    for regime in REGIMES:
        for grid in GRIDS:
            threads = grid * grid
            idx = [
                i
                for i in range(len(regimes))
                if regimes[i] == regime and thread_counts[i] == threads
            ]
            if not idx:
                continue
                ax.scatter(
                XY[idx, 0],
                XY[idx, 1],
                c=REGIME_COLORS[regime],
                marker=THREAD_GRID[grid],
                s=90,
                edgecolors="black",
                linewidths=0.6,
                label=None,
            )
    # Explicit legends (color = regime, marker = threads) to avoid duplicate labels.
    from matplotlib.lines import Line2D

    regime_handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=REGIME_COLORS[r],
            markersize=10, markeredgecolor="black", label=r,
        )
        for r in REGIMES
    ]
    thread_handles = [
        Line2D(
            [0], [0], marker=THREAD_GRID[g], color="w", markerfacecolor="gray",
            markersize=10, markeredgecolor="black", label=f"{g*g}t",
        )
        for g in GRIDS
    ]
    first_legend = ax.legend(
        handles=regime_handles, title="regime", loc="upper left", fontsize=9
    )
    ax.add_artist(first_legend)
    ax.legend(handles=thread_handles, title="threads", loc="lower right", fontsize=9)

    ax.set_xlabel(f"{method} dim 1")
    ax.set_ylabel(f"{method} dim 2")
    ax.set_title(
        f"Synthetic Norn contention embeddings (label-free)\n"
        f"silhouette by regime = {sil:.2f}"
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    svg_path = RESULTS_DIR / "contention_embeddings.svg"
    fig.savefig(svg_path, format="svg")
    plt.close(fig)

    metrics = {
        "silhouette_by_regime": sil,
        "reduction_method": method,
        "separated": separated,
        "n_traces": int(len(regimes)),
        "regimes": REGIMES,
        "thread_counts": [int(t) for t in thread_counts],
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Silhouette by regime: {sil:.2f}")
    print(f"Clusters visually separated: {separated} (method={method})")
    print(f"[plot] wrote {svg_path}")
    print(f"[plot] wrote {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
