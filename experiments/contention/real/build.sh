#!/usr/bin/env bash
#
# build.sh - compile the standalone Norn capture program.
#
# Norn is header-only (no .a needed); we just need its include path.
# Ygg's instrumentation static library is linked directly. On macOS the Rust
# staticlib can pull in system frameworks; if that fails we fall back to
# compiling Ygg's C sources directly (same code, no Rust runtime to link).
#
# Output: ./norn_capture

set -euo pipefail

# --- locate roots (override with env if needed) ---
YGG_ROOT="${YGG_ROOT:-$HOME/Projects/ygg}"
NORN_ROOT="${NORN_ROOT:-$HOME/Projects/norn}"

YGG_INST_INC="$YGG_ROOT/instrumentation/include"
YGG_SRC="$YGG_ROOT/instrumentation/src"
NORN_INC="$NORN_ROOT/include"

# The static lib lives in the workspace target dir (cargo -p ygg-instrumentation).
YGG_LIB="$(ls "$YGG_ROOT"/target/release/libygg_instrumentation.a 2>/dev/null || true)"
if [ -z "$YGG_LIB" ]; then
  YGG_LIB="$(ls "$YGG_ROOT"/instrumentation/target/release/libygg_instrumentation.a 2>/dev/null || true)"
fi

echo "[build] YGG_ROOT=$YGG_ROOT"
echo "[build] NORN_ROOT=$NORN_ROOT"
echo "[build] YGG include=$YGG_INST_INC"
echo "[build] NORN include=$NORN_INC"
echo "[build] YGG lib=$YGG_LIB"

CXX="${CXX:-clang++}"
CXXFLAGS=(-std=c++20 -O2 -Wall -Wextra)
INCLUDES=(-I"$YGG_INST_INC" -I"$NORN_INC")

if [ -z "$YGG_LIB" ]; then
  echo "[build] ERROR: libygg_instrumentation.a not found. Build it with:" >&2
  echo "         cargo build --release -p ygg-instrumentation" >&2
  exit 1
fi

# --- attempt 1: link the Rust staticlib ---
echo "[build] linking against $YGG_LIB"
if "$CXX" "${CXXFLAGS[@]}" "${INCLUDES[@]}" \
    norn_capture.cpp -o norn_capture \
    "$YGG_LIB" -lpthread \
    -framework Security -framework Foundation -framework CoreFoundation \
    2>build.log; then
  echo "[build] OK (staticlib link)"
  exit 0
fi

echo "[build] staticlib link failed; trying direct C-source compile:" >&2
cat build.log >&2

# --- attempt 2: compile Ygg's C sources directly (no Rust runtime) ---
"$CXX" "${CXXFLAGS[@]}" "${INCLUDES[@]}" -I"$YGG_SRC" \
  norn_capture.cpp \
  "$YGG_SRC/ygg.c" "$YGG_SRC/ring_buffer.c" "$YGG_SRC/collector_thread.c" \
  -o norn_capture -lpthread \
  && echo "[build] OK (direct C sources)" \
  && exit 0

echo "[build] FAILED" >&2
cat build.log >&2
exit 1
