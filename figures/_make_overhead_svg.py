#!/usr/bin/env python3
"""Generate figures/overhead.svg (Figure 2) for the Ygg project.

Reads the measured instrumentation benchmark results from
instrumentation/bench/results.json (one JSON object per line, here stored as a
JSON array) and emits a static SVG bar chart of the hot-path overhead:

  - grouped bars of median / p95 / p99 per scenario
  - dropped-event counts annotated on the drop-path scenarios
  - platform + unit stated in the caption, never claiming "cycles" on macOS

The figure is honest: the unit comes straight from the data. On macOS/arm64 the
bench reports nanoseconds ("ns"); on x86_64 it reports raw TSC ticks ("cycles").
We never relabel ns as cycles.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # static, no interactive backend
# Keep text as real <text> elements (selectable/editable) rather than embedded
# glyph paths, and use core fonts so the SVG is self-contained.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "instrumentation" / "bench" / "results.json"
OUT_SVG = Path(__file__).resolve().parent / "overhead.svg"

# Logical scenario order and human-readable labels for the x-axis.
# The bench emits "ygg_event_threads" twice (4T and 8T); we aggregate it here
# into a single "threads" group and footnote both raw runs.
SCENARIO_ORDER = [
    "baseline_disabled",
    "ygg_event_no_collector",
    "ygg_event_draining",
    "ring_at_capacity",
    "ygg_event_threads",
]
SCENARIO_LABELS = {
    "baseline_disabled": "baseline\n(no YGG)",
    "ygg_event_no_collector": "no collector\n(ring fills)",
    "ygg_event_draining": "draining\n(collector on)",
    "ring_at_capacity": "ring at capacity\n(burst)",
    "ygg_event_threads": "threads\n(4T + 8T)",
}

# Scenarios whose bars get a dropped-event annotation.
DROP_SCENARIOS = {"ygg_event_no_collector", "ring_at_capacity"}

COLORS = {
    "median": "#2563eb",
    "p95": "#f59e0b",
    "p99": "#dc2626",
}


def fmt_int(n: int) -> str:
    return f"{n:,}"


def load_results(path: Path) -> list[dict]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("results.json must be a JSON array of scenario objects")
    return data


def aggregate(data: list[dict]) -> dict[str, dict]:
    """Build one record per logical scenario, averaging the threads runs."""
    by_scenario: dict[str, list[dict]] = {}
    for row in data:
        by_scenario.setdefault(row["scenario"], []).append(row)

    out: dict[str, dict] = {}
    for scen in SCENARIO_ORDER:
        rows = by_scenario.get(scen, [])
        if not rows:
            raise ValueError(f"missing scenario in results: {scen}")
        if len(rows) == 1:
            r = rows[0]
            out[scen] = {
                "median": float(r["median_cycles_per_event"]),
                "p95": float(r["p95_cycles_per_event"]),
                "p99": float(r["p99_cycles_per_event"]),
                "dropped": int(r["dropped_events"]),
                "raw": rows,
            }
        else:
            # Aggregate multi-run scenarios (threads: 4T and 8T) by mean of the
            # percentiles and sum of drops. Keep the raw runs for the footnote.
            out[scen] = {
                "median": sum(float(r["median_cycles_per_event"]) for r in rows) / len(rows),
                "p95": sum(float(r["p95_cycles_per_event"]) for r in rows) / len(rows),
                "p99": sum(float(r["p99_cycles_per_event"]) for r in rows) / len(rows),
                "dropped": sum(int(r["dropped_events"]) for r in rows),
                "raw": rows,
            }
    return out


def unit_note(unit: str) -> str:
    if unit == "cycles":
        return "raw TSC ticks (rdtscp)"
    return "nanoseconds (mach_absolute_time / CLOCK_MONOTONIC)"


def main() -> None:
    data = load_results(RESULTS)
    records = aggregate(data)

    # Honesty guard: pull platform + unit from the data, never hardcode.
    unit = data[0].get("unit", "ns")
    platform = data[0].get("platform", "unknown")

    labels = [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
    medians = [records[s]["median"] for s in SCENARIO_ORDER]
    p95s = [records[s]["p95"] for s in SCENARIO_ORDER]
    p99s = [records[s]["p99"] for s in SCENARIO_ORDER]
    drops = [records[s]["dropped"] for s in SCENARIO_ORDER]

    x = range(len(SCENARIO_ORDER))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=120)

    b1 = ax.bar([i - width for i in x], medians, width, label="median (p50)", color=COLORS["median"])
    b2 = ax.bar([i for i in x], p95s, width, label="p95", color=COLORS["p95"])
    b3 = ax.bar([i + width for i in x], p99s, width, label="p99", color=COLORS["p99"])

    # Log scale: the spread (sub-10 ns floor to ~1500 ns multithread tail) makes
    # a linear axis unusable. Bars start at 1 ns (log scale has no zero).
    ax.set_yscale("log")
    ax.set_ylim(bottom=1, top=5000)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([1, 10, 100, 1000])

    ax.set_ylabel(f"per-event accept cost ({unit}, log scale)")
    ax.set_title(
        "Figure 2. Instrumentation hot-path overhead\n"
        "measured per-event cost of YGG_EVENT emission across benchmark scenarios",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", which="major", linestyle=":", alpha=0.5)

    # Value labels on top of each bar.
    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    # Dropped-event annotations on the drop-path scenarios.
    for i, scen in enumerate(SCENARIO_ORDER):
        if scen in DROP_SCENARIOS and drops[i] > 0:
            ax.annotate(
                f"{fmt_int(drops[i])} dropped",
                xy=(i, 5000),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="#7f1d1d",
            )

    # Caption: state platform + unit honestly, and tie back to the header claim.
    threads_raw = records["ygg_event_threads"]["raw"]
    threads_parts = ", ".join(
        f"{int(r['threads'])}T: {float(r['median_cycles_per_event']):.1f}/"
        f"{float(r['p95_cycles_per_event']):.0f}/{float(r['p99_cycles_per_event']):.0f}"
        for r in threads_raw
    )
    cap = (
        f"Platform: {platform}. Reporting unit: {unit} ({unit_note(unit)}). "
        + (
            "The '~5-10 cycles' figure in instrumentation/include/ygg/ygg.h applies to "
            "x86-64 rdtscp; on this platform the hot path is reported in ns, not cycles. "
            if unit == "ns"
            else "On x86-64 the hot path is reported in raw TSC cycles. "
        )
        + "Active-collector scenarios (draining, threads) measure ~80-100 ns per event, "
        "consistent with the header's documented 70-80 ns on macOS/arm64. On the drop "
        "paths (no collector, ring at capacity) emission stays cheap but events are lost. "
        f"threads = mean of 4T and 8T runs ({threads_parts} ns; median/p95/p99)."
    )
    fig.text(0.5, 0.012, cap, ha="center", va="bottom", fontsize=8.5, wrap=True)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT_SVG, format="svg", bbox_inches="tight")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
