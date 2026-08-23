#!/usr/bin/env python3
"""
embed_real.py - compute execution / window embeddings for REAL Norn contention
traces and render the Figure 1 embedding map.

The encoder is trained label-free (masked objective only). We embed every
real trace at the *window* level (one vector per local window of events), which
gives hundreds of points per regime and a stable clustering signal, then reduce
with UMAP (PCA fallback) and colour by backoff regime / shape by thread count.

V0.1 note: macOS lacks eBPF, so only application-level Ygg events (kinds
1000-1004) are captured. That is sufficient for V0.1 contention analysis.

Outputs:
    ~/Projects/ygg/figures/embedding_map.svg
    experiments/contention/real/metrics.json
"""
import json
import sys
from pathlib import Path

# Make the repo root (which owns the `model` package) importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")  # static SVG only
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import polars as pl

import jax
import jax.numpy as jnp
from flax import linen as nn

from model.config import ModelConfig
from model.dataset import events_to_tokens

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DATA_DIR = HERE
CHECKPOINT = RESULTS_DIR / "encoder_real.msgpack"
FIGURE_DIR = Path.home() / "Projects" / "ygg" / "figures"
FIGURE_PATH = FIGURE_DIR / "embedding_map.svg"
METRICS_PATH = HERE / "metrics.json"

TOKEN_DIM = 7
METRIC_SCALE = 1.0  # must match train_real.py

REGIMES = ["tight", "yield", "bounded", "exponential"]
GRIDS = [1, 2, 4, 8]
MAX_WINDOWS_PER_TRACE = 200  # cap for stable, fast clustering + plotting
REGIME_COLORS = {
    "tight": "#e6194b",
    "yield": "#3cb44b",
    "bounded": "#4363d8",
    "exponential": "#f58231",
}
THREAD_GRID = {1: "o", 2: "s", 4: "^", 8: "D"}


class WindowEmbedder(nn.Module):
    """Inference-only embedder returning per-window contextual embeddings.

    Uses the same ``hier`` encoder subtree stored in the trained checkpoint.
    """

    config: ModelConfig

    @nn.compact
    def __call__(self, tokens, mask):
        from model.hierarchical import HierarchicalEncoder

        enc = HierarchicalEncoder(self.config, name="hier")
        out = enc(tokens, mask=mask, causal=False, train=False)
        return out["window_ctx"]  # [1, N, d]


def load_checkpoint(path: Path):
    from flax import serialization as fser

    with open(path, "rb") as f:
        blob = f.read()
    target = {"params": None, "opt_state": None, "step": 0, "config": ""}
    ck = fser.from_bytes(target, blob)
    ck["config"] = json.loads(ck["config"])
    return ck


def embed_trace(path: Path, embedder, params, config) -> np.ndarray:
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
    emb = embedder.apply({"params": params}, tokens, mask)  # [1, N, d]
    return np.asarray(emb)[0]  # [N, d]


def reduce(X: np.ndarray):
    try:
        from umap import UMAP

        reducer = UMAP(
            n_neighbors=min(15, X.shape[0] - 1),
            min_dist=0.3,
            n_components=2,
            random_state=0,
            metric="euclidean",
        )
        XY = reducer.fit_transform(X)
        return XY, "umap"
    except Exception as e:  # noqa: BLE001
        print(f"[embed_real] UMAP unavailable ({e}); falling back to PCA")
        from sklearn.decomposition import PCA

        XY = PCA(n_components=2, random_state=0).fit_transform(X)
        return XY, "pca"


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"No checkpoint at {CHECKPOINT}. Run train_real.py first.")

    ck = load_checkpoint(CHECKPOINT)
    config = ModelConfig.from_dict(ck["config"])
    params = ck["params"]

    embedder = WindowEmbedder(config)

    X_rows = []
    regimes = []
    grids = []
    rng = np.random.default_rng(0)
    for regime in REGIMES:
        for grid in GRIDS:
            # Files are named <regime>_<grid>x<grid>; the capture spawns
            # 2*grid threads (producers + consumers).
            path = DATA_DIR / f"{regime}_{grid}x{grid}.parquet"
            if not path.exists():
                raise SystemExit(f"Missing real trace {path}")
            emb = embed_trace(path, embedder, params, config)  # [N, d]
            if emb.shape[0] > MAX_WINDOWS_PER_TRACE:
                idx = rng.choice(emb.shape[0], MAX_WINDOWS_PER_TRACE, replace=False)
                emb = emb[idx]
            X_rows.append(emb)
            regimes.extend([regime] * emb.shape[0])
            grids.extend([grid] * emb.shape[0])

    X = np.concatenate(X_rows, axis=0)
    regimes = np.array(regimes)
    grids = np.array(grids)

    XY, method = reduce(X)

    from sklearn.metrics import silhouette_score, silhouette_samples

    try:
        sil = float(silhouette_score(X, regimes))
        samples = silhouette_samples(X, regimes)
        per_regime = {r: float(np.mean(samples[regimes == r])) for r in REGIMES}
    except Exception as e:  # noqa: BLE001
        print(f"[embed_real] silhouette failed: {e}")
        sil = float("nan")
        per_regime = {r: float("nan") for r in REGIMES}

    separated = not np.isnan(sil) and sil > 0.25

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for regime in REGIMES:
        for grid in GRIDS:
            idx = [
                i
                for i in range(len(regimes))
                if regimes[i] == regime and grids[i] == grid
            ]
            if not idx:
                continue
            ax.scatter(
                XY[idx, 0],
                XY[idx, 1],
                c=REGIME_COLORS[regime],
                marker=THREAD_GRID[grid],
                s=22,
                alpha=0.7,
                edgecolors="none",
                label=None,
            )

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
            markersize=10, markeredgecolor="black", label=f"{2 * g}t",
        )
        for g in GRIDS
    ]
    first_legend = ax.legend(
        handles=regime_handles, title="regime (backoff)", loc="upper left", fontsize=9
    )
    ax.add_artist(first_legend)
    ax.legend(handles=thread_handles, title="threads", loc="lower right", fontsize=9)

    ax.set_xlabel(f"{method} dim 1")
    ax.set_ylabel(f"{method} dim 2")
    status = "separated" if separated else "not yet separated"
    ax.set_title(
        f"REAL Norn contention embeddings (label-free, window-level)\n"
        f"silhouette by regime = {sil:.2f}  ({status}, method={method})\n"
        f"application-level events only (no eBPF on macOS)"
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, format="svg")
    plt.close(fig)

    metrics = {
        "n_windows": int(X.shape[0]),
        "n_regimes": len(REGIMES),
        "method": method,
        "silhouette_overall": sil,
        "silhouette_by_regime": per_regime,
        "separated": separated,
        "note": (
            "Window-level embeddings from the label-free V0.1 encoder. "
            "Positive silhouette => regimes form distinct clusters without "
            "any policy labels."
        ),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[embed_real] figure -> {FIGURE_PATH}")
    print(f"[embed_real] metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
