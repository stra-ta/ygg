#!/usr/bin/env bash
#
# run_capture.sh - run the standalone Norn capture across all (regime, grid)
# cells, emitting MANY independent captures per cell so downstream training can
# split by run (not by window) and avoid correlated-sample bias.
#
# Layout: 4 backoff regimes (tight yield bounded exponential) x
# {1x1, 2x2, 4x4, 8x8} grids, repeated N_RUNS times.
#
# Output: ./raw/<policy>_<grid>x<grid>_run<k>.events
#         ./<policy>_<grid>x<grid>_run<k>.parquet  (after spill_to_parquet)
#
# Env overrides:
#   N_RUNS=10          number of independent runs per cell (default 10)
#   DURATION_MS=3000   per-run capture duration (default 3000)
#   SMOKE=1            quick test: N_RUNS=2, DURATION_MS=400

set -euo pipefail

cd "$(dirname "$0")"

# --- env-driven configuration ---
if [ "${SMOKE:-0}" = "1" ]; then
  N_RUNS="${N_RUNS:-2}"
  DURATION_MS="${DURATION_MS:-400}"
else
  N_RUNS="${N_RUNS:-10}"
  DURATION_MS="${DURATION_MS:-3000}"
fi

POLICIES=(tight yield bounded exponential)
GRIDS=(1 2 4 8)

echo "=== config: N_RUNS=$N_RUNS DURATION_MS=$DURATION_MS ==="

# --- ensure the binary exists (build it if missing) ---
if [ ! -x ./norn_capture ]; then
  echo "=== norn_capture missing; building via ./build.sh ==="
  ./build.sh
fi
if [ ! -x ./norn_capture ]; then
  echo "!!! build failed: ./norn_capture still missing" >&2
  exit 1
fi

mkdir -p raw

total=$(( ${#POLICIES[@]} * ${#GRIDS[@]} * N_RUNS ))
done=0
for policy in "${POLICIES[@]}"; do
  for grid in "${GRIDS[@]}"; do
    for ((k = 0; k < N_RUNS; k++)); do
      out="raw/${policy}_${grid}x${grid}_run${k}.events"
      done=$((done + 1))
      echo "=== [$done/$total] capture: policy=$policy grid=$grid run=$k -> $out ==="
      ./norn_capture "$policy" "$grid" "$out" "$DURATION_MS"
    done
  done
done

echo "=== all captures done ($total files) ==="
ls -la raw/

# --- convert raw spills to parquet (run-suffixed stems preserved) ---
echo "=== converting spills to parquet ==="
python3 spill_to_parquet.py raw .

echo "=== done ==="
