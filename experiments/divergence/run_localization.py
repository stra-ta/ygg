"""
run_localization.py
===================

Divergence-localization experiment (Figure 3) for Ygg.

We do NOT need a trained Transformer to validate the *localization
methodology*. We feed Ygg's real divergence pipeline functions
(``analysis.divergence.dtw_align`` and ``analysis.divergence.detect_change_points``)
a lightweight, magnitude-aware window embedding:

    window -> [ per-kind fraction histogram (8)
               + mean CAS-retry arg0
               + mean spin-iteration arg0
               + mean cache-miss delta
               + scheduler-migration rate
               + mean queue depth ]                    (13 raw features)

The two phases (bounded backoff vs tight spin) differ mainly in the *magnitude*
of those arg-bearing features (CAS 3 -> 125, spin 35 -> 1250, cache 350 -> 3750),
so we measure divergence with a standardized (Mahalanobis) distance from each
``switched`` window to the nominal ``healthy`` reference. The ``healthy`` trace
defines "normal"; the switch location is never shown to the algorithm.

Pipeline (all real Ygg functions):
  1. Embed both traces into window feature vectors.
  2. ``dtw_align`` the healthy embedding sequence against the switched one
     (aligns the two executions point-to-point despite speed differences).
  3. Build a per-window divergence series from the aligned pairs.
  4. ``detect_change_points`` with PELT (and cross-check with BOCPD).
  5. The earliest sustained change point is the localized divergence.

No switch label is given to the model. The ground-truth switch (recorded in
``switch_meta.json`` at generation time) is used ONLY to score the result and to
draw the "true policy switch" reference line in the figure.

Run:
    python experiments/divergence/run_localization.py

Produces:
    figures/divergence_localization.svg
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import polars as pl

# Make the project root importable whether run as a script or imported.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.divergence import dtw_align, detect_change_points  # noqa: E402

# Event kinds (mirror schema/EVENT_SCHEMA.md).
APP_BASE = 1000
KIND_PUSH = APP_BASE + 0
KIND_POP = APP_BASE + 1
KIND_CAS_RETRY = APP_BASE + 2
KIND_YIELD = APP_BASE + 3
KIND_SPIN = APP_BASE + 4
KIND_SCHED_SWITCH = 3000
KIND_SCHED_MIGRATE = 3002
KIND_CACHE_MISSES = 7002

KIND_ORDER = [
    KIND_PUSH, KIND_POP, KIND_CAS_RETRY, KIND_YIELD, KIND_SPIN,
    KIND_SCHED_SWITCH, KIND_SCHED_MIGRATE, KIND_CACHE_MISSES,
]

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "figures")

WIN = 512
STRIDE = 256


def window_features(path: str, win: int = WIN, stride: int = STRIDE):
    """
    Raw feature vector per window of a trace.

    Returns (features [M, 13], window_mid_timestamp_ns [M]).
    Features are intentionally left unnormalized; the divergence metric below
    standardizes against the healthy reference, which keeps the two phases
    separable by magnitude.
    """
    df = pl.read_parquet(path).select(["timestamp_ns", "kind", "arg0"])
    ts = df["timestamp_ns"].to_numpy().astype(np.int64)
    kind = df["kind"].to_numpy().astype(np.int64)
    arg0 = df["arg0"].to_numpy().astype(np.int64)
    n = len(ts)

    feats: list[np.ndarray] = []
    mid_ts: list[int] = []
    for s in range(0, n - win + 1, stride):
        e = s + win
        seg_kind = kind[s:e]
        seg_arg = arg0[s:e]

        hist = np.array(
            [(seg_kind == k).sum() for k in KIND_ORDER], dtype=float
        ) / win

        cas = seg_kind == KIND_CAS_RETRY
        spin = seg_kind == KIND_SPIN
        cm = seg_kind == KIND_SCHED_MIGRATE
        pf = seg_kind == KIND_CACHE_MISSES
        q = (seg_kind == KIND_PUSH) | (seg_kind == KIND_POP)

        mean_cas = float(seg_arg[cas].mean()) if cas.any() else 0.0
        mean_spin = float(seg_arg[spin].mean()) if spin.any() else 0.0
        mean_pf = float(seg_arg[pf].mean()) if pf.any() else 0.0
        mig_rate = float(cm.sum()) / win
        mean_depth = float(seg_arg[q].mean()) if q.any() else 0.0

        vec = np.array(
            [*hist, mean_cas, mean_spin, mean_pf, mig_rate, mean_depth],
            dtype=float,
        )
        feats.append(vec)
        mid_ts.append(int(ts[s + win // 2]))

    return np.stack(feats, axis=0), np.array(mid_ts, dtype=np.int64)


def divergence_series(feat_ref, ts_ref, feat_tgt, ts_tgt, band_frac=0.25):
    """
    Align the healthy reference to the switched target with DTW, then build a
    per-target-window divergence series using a Mahalanobis distance from each
    aligned pair to the healthy reference distribution.

    Returns (dist [M_tgt], ts_tgt [M_tgt]).
    """
    # L2-normalize per window so cosine in DTW reflects compositional direction.
    def _l2(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(n == 0, 1.0, n)

    emb_ref = _l2(feat_ref)
    emb_tgt = _l2(feat_tgt)

    path, _ = dtw_align(emb_ref, emb_tgt, band_frac=band_frac)

    # Standardized distance metric: Mahalanobis to the healthy reference cloud.
    mu = feat_ref.mean(axis=0)
    cov = np.cov(feat_ref.T) + 1e-6 * np.eye(feat_ref.shape[1])
    inv_cov = np.linalg.inv(cov)

    def _maha(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.sqrt(d @ inv_cov @ d))

    per_j: dict[int, list[float]] = {}
    for i, j in path:
        j = int(j)
        per_j.setdefault(j, []).append(_maha(feat_tgt[j], feat_ref[i]))

    M = feat_tgt.shape[0]
    dist = np.full(M, np.nan)
    for j in range(M):
        if j in per_j:
            dist[j] = float(np.mean(per_j[j]))
    if np.isnan(dist).any():
        idx = np.arange(M)
        good = ~np.isnan(dist)
        dist = np.interp(idx[good], idx[good], dist[good])
    return dist, ts_tgt


def localize(dist: np.ndarray, ts: np.ndarray, method: str = "pelt"):
    """
    Detect the earliest divergence point on the distance series.

    Returns (detected_timestamp_ns, change_index, all_change_indices).
    Falls back to the argmax of the distance if no change point is found.
    """
    changes = detect_change_points(dist, method=method, min_size=2)
    if changes:
        c = int(changes[0])  # earliest sustained change
        return int(ts[c]), c, [int(x) for x in changes]
    c = int(np.argmax(dist))
    return int(ts[c]), c, []


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    healthy_path = os.path.join(HERE, "healthy.parquet")
    switched_path = os.path.join(HERE, "switched.parquet")
    meta_path = os.path.join(HERE, "switch_meta.json")

    with open(meta_path) as f:
        meta = json.load(f)
    true_idx = int(meta["true_switch_event_index"])
    true_ts = int(meta["true_switch_timestamp_ns"])
    total_ns = int(meta["total_duration_ns"])

    feat_ref, ts_ref = window_features(healthy_path)
    feat_tgt, ts_tgt = window_features(switched_path)

    dist, ts_tgt = divergence_series(feat_ref, ts_ref, feat_tgt, ts_tgt)

    det_pelt, det_idx_pelt, chg_pelt = localize(dist, ts_tgt, method="pelt")
    det_bocpd, det_idx_bocpd, chg_bocpd = localize(dist, ts_tgt, method="bocpd")

    err_pelt = abs(det_pelt - true_ts)
    err_bocpd = abs(det_bocpd - true_ts)
    err_frac = err_pelt / total_ns if total_ns else 0.0

    print(f"true switch    : event {true_idx}, {true_ts/1e9:.4f} s")
    print(f"PELT detected  : window {det_idx_pelt}, {det_pelt/1e9:.4f} s "
          f"(error {err_pelt/1e9:.4f} s, {err_frac*100:.2f}% of trace)")
    print(f"BOCPD detected : window {det_idx_bocpd}, {det_bocpd/1e9:.4f} s "
          f"(error {err_bocpd/1e9:.4f} s)")
    print(f"PELT changes   : {chg_pelt}")
    print(f"BOCPD changes  : {chg_bocpd}")

    # ----------------------------------------------------------------- #
    # Static SVG figure
    # ----------------------------------------------------------------- #
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "divergence_localization.svg")

    t_sec = ts_tgt.astype(float) / 1e9
    true_sec = true_ts / 1e9
    det_sec = det_pelt / 1e9

    fig, ax = plt.subplots(figsize=(11, 5.2))

    ax.plot(t_sec, dist, color="#3b6ea5", lw=1.6, alpha=0.9,
            label="divergence vs healthy reference (DTW + Mahalanobis)")
    ax.axvline(true_sec, color="#c0392b", ls="--", lw=2.0,
               label=f"true policy switch ({true_sec:.3f} s)")
    ax.axvline(det_sec, color="#27ae60", ls=":", lw=2.4,
               label=f"detected - PELT ({det_sec:.3f} s)")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("divergence (Mahalanobis distance)")
    ax.set_title("Figure 3 - Divergence localization: blind phase-transition "
                 "detection (DTW + PELT/BOCPD)")
    caption = (f"Detected: {det_sec:.3f} s   True: {true_sec:.3f} s   "
               f"Error: {err_pelt/1e9:.4f} s ({err_frac*100:.2f}%)   "
               f"[BOCPD: {det_bocpd/1e9:.3f} s]")
    ax.text(0.5, -0.22, caption, transform=ax.transAxes, ha="center",
            va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="#f4f6f8", ec="#cccccc"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
