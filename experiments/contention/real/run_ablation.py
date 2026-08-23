#!/usr/bin/env python3
"""
run_ablation.py - run the full V0.2 ablation study and report regime separation.

For each of the five variants:
  1. If results/encoder_v02_<variant>.msgpack already exists, skip training.
  2. Else train it with train_variant (run-level split, ~2h for the full study).
  3. Embed the EVAL-run windows and compute silhouette score by regime.

Outputs:
  experiments/contention/real/ablation_results.json
  a Markdown table to stdout
  ~/Projects/ygg/figures/ablation_silhouette.svg  (bar chart, red if sil<=0 else green)

This script is the orchestrator's full-study driver; for the smoke test only
train_v02.py is run directly. Do NOT commit or push.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from train_v02 import (
    VARIANT_WEIGHTS,
    make_config,
    train_variant,
    embed_eval_windows,
    run_level_split,
    RESULTS_DIR,
    HERE,
)

# HERE = .../experiments/contention/real -> parents[2] is the repo root.
FIG_DIR = HERE.parents[2] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = list(VARIANT_WEIGHTS.keys())


def per_regime_silhouette(X: np.ndarray, regimes: np.ndarray):
    from sklearn.metrics import silhouette_score

    uniq = sorted(np.unique(regimes))
    out = {}
    for r in uniq:
        binlab = (regimes == r).astype(int)
        if int((binlab == 1).sum()) < 2 or int((binlab == 0).sum()) < 1:
            out[r] = float("nan")
            continue
        try:
            out[r] = float(silhouette_score(X, binlab))
        except Exception:
            out[r] = float("nan")
    return out


def main() -> None:
    _, eval_files, n_runs = run_level_split(HERE)
    print(f"[run_ablation] {len(eval_files)} eval files, {n_runs} total runs", flush=True)

    results = []
    for variant in VARIANTS:
        ckpt = RESULTS_DIR / f"encoder_v02_{variant}.msgpack"
        if ckpt.exists():
            print(f"[run_ablation:{variant}] checkpoint exists, skipping training", flush=True)
        else:
            print(f"[run_ablation:{variant}] training...", flush=True)
            train_variant(variant)

        X, regimes, sil = embed_eval_windows(variant, eval_files)
        per = per_regime_silhouette(X, regimes)
        rec = {
            "variant": variant,
            "silhouette": sil,
            "per_regime_silhouette": per,
            "n_eval_windows": int(X.shape[0]),
        }
        results.append(rec)
        print(f"[run_ablation:{variant}] silhouette={sil:.4f} n_windows={rec['n_eval_windows']}",
              flush=True)

    # JSON results
    out_json = RESULTS_DIR / "ablation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"[run_ablation] wrote {out_json}", flush=True)

    # Markdown table to stdout
    print("\n## V0.2 ablation: regime separation (silhouette)\n")
    print("| variant | silhouette | n_eval_windows |")
    print("|---|---|---|")
    for r in results:
        sil = r["silhouette"]
        sil_s = "nan" if sil != sil else f"{sil:.4f}"
        print(f"| {r['variant']} | {sil_s} | {r['n_eval_windows']} |")

    # Bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        variants = [r["variant"] for r in results]
        sils = [r["silhouette"] for r in results]
        colors = ["red" if (s != s or s <= 0) else "green" for s in sils]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(variants, sils, color=colors)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel("silhouette (regime separation)")
        ax.set_title(
            f"V0.2 ablation silhouette by variant (run-level split, {n_runs} runs)"
        )
        ax.set_ylim(min(-0.2, min(sils) if len(sils) else -0.2), max(0.6, max(sils) if len(sils) else 0.6))
        plt.xticks(rotation=20, ha="right")
        for b, s in zip(bars, sils):
            label = "nan" if s != s else f"{s:.3f}"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), label,
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig_path = FIG_DIR / "ablation_silhouette.svg"
        fig.savefig(fig_path)
        print(f"[run_ablation] wrote {fig_path}", flush=True)
    except Exception as e:
        print(f"[run_ablation] chart skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
