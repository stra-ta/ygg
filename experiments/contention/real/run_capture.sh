#!/usr/bin/env bash
#
# run_capture.sh - run the standalone Norn capture for all 16 combos
# (4 backoff regimes x {1x1, 2x2, 4x4, 8x8}) and write the raw spill files.
#
# Output: ./raw/<policy>_<grid>x<grid>.events

set -euo pipefail

cd "$(dirname "$0")"

POLICIES=(tight yield bounded exponential)
GRIDS=(1 2 4 8)
DURATION_MS="${DURATION_MS:-3000}"

mkdir -p raw

for policy in "${POLICIES[@]}"; do
  for grid in "${GRIDS[@]}"; do
    out="raw/${policy}_${grid}x${grid}.events"
    echo "=== capture: policy=$policy grid=$grid -> $out ==="
    ./norn_capture "$policy" "$grid" "$out" "$DURATION_MS"
  done
done

echo "=== all captures done ==="
ls -la raw/
