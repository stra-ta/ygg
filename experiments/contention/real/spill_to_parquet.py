#!/usr/bin/env python3
"""
spill_to_parquet.py - convert Ygg raw spill files to Parquet.

A raw spill file is a 32-byte stream header followed by 48-byte event records:

    struct ygg_stream_header {     // 32 bytes
        uint64_t magic;            // 0x5967673100000001  ("Ygg1")
        uint64_t tsc_freq_hz;
        uint64_t tsc_to_ns_mult;   // 32.32 fixed point
        uint64_t monotonic_offset_ns;
    };
    // then N x 48-byte records:
    struct ygg_event {             // 48 bytes
        uint64_t timestamp_ns;
        uint32_t cpu;
        uint32_t pid;
        uint32_t tid;
        uint16_t kind;
        uint16_t padding;
        uint64_t arg0;
        uint64_t arg1;
        uint64_t arg2;
    };

We parse both and emit one Parquet file per spill, matching the 8-column
schema in schema/EVENT_SCHEMA.md:

    timestamp_ns u64, cpu u32, pid u32, tid u32, kind u16,
    arg0 u64, arg1 u64, arg2 u64

Usage:
    python3 spill_to_parquet.py [raw_dir] [out_dir]
Defaults: raw_dir=./raw  out_dir=. (the real/ directory)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Must match ygg_internal.h YGG_SHM_MAGIC.
EXPECTED_MAGIC = 0x5967673100000001
HEADER_BYTES = 32
RECORD_BYTES = 48

EVENT_DTYPE = np.dtype(
    [
        ("timestamp_ns", "<u8"),
        ("cpu", "<u4"),
        ("pid", "<u4"),
        ("tid", "<u4"),
        ("kind", "<u2"),
        ("padding", "<u2"),
        ("arg0", "<u8"),
        ("arg1", "<u8"),
        ("arg2", "<u8"),
    ]
)

# Arrow schema (no padding column) per schema/EVENT_SCHEMA.md.
PARQUET_SCHEMA = pa.schema(
    [
        ("timestamp_ns", pa.uint64()),
        ("cpu", pa.uint32()),
        ("pid", pa.uint32()),
        ("tid", pa.uint32()),
        ("kind", pa.uint16()),
        ("arg0", pa.uint64()),
        ("arg1", pa.uint64()),
        ("arg2", pa.uint64()),
    ]
)


def convert_one(spill_path: Path) -> tuple[pa.Table, int]:
    data = spill_path.read_bytes()
    if len(data) < HEADER_BYTES:
        raise ValueError(f"{spill_path}: file too small ({len(data)} bytes)")

    header = np.frombuffer(data[:HEADER_BYTES], dtype="<u8")
    magic = int(header[0])
    if magic != EXPECTED_MAGIC:
        raise ValueError(
            f"{spill_path}: bad magic 0x{magic:016x} != 0x{EXPECTED_MAGIC:016x}"
        )

    body = data[HEADER_BYTES:]
    if len(body) % RECORD_BYTES != 0:
        raise ValueError(
            f"{spill_path}: body size {len(body)} not a multiple of {RECORD_BYTES}"
        )
    n_records = len(body) // RECORD_BYTES

    if n_records == 0:
        # Empty trace (header only) -> emit a zero-row table with the schema.
        table = pa.table(
            {name: pa.array([], type=typ) for name, typ in zip(
                PARQUET_SCHEMA.names, PARQUET_SCHEMA.types)}
        )
        return table, 0

    events = np.frombuffer(body, dtype=EVENT_DTYPE, count=n_records)
    table = pa.table(
        {
            "timestamp_ns": events["timestamp_ns"].astype(np.uint64),
            "cpu": events["cpu"].astype(np.uint32),
            "pid": events["pid"].astype(np.uint32),
            "tid": events["tid"].astype(np.uint32),
            "kind": events["kind"].astype(np.uint16),
            "arg0": events["arg0"].astype(np.uint64),
            "arg1": events["arg1"].astype(np.uint64),
            "arg2": events["arg2"].astype(np.uint64),
        },
        schema=PARQUET_SCHEMA,
    )
    return table, n_records


def main() -> None:
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./raw")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    spill_files = sorted(raw_dir.glob("*.events"))
    if not spill_files:
        raise SystemExit(f"No spill files found in {raw_dir}")

    total_rows = 0
    print(f"[spill_to_parquet] {len(spill_files)} spill file(s) -> {out_dir}")
    for sp in spill_files:
        table, n = convert_one(sp)
        out_path = out_dir / (sp.stem + ".parquet")
        pq.write_table(table, out_path)
        total_rows += n

        # Kind histogram for a quick sanity check of regime signal.
        kinds = table.column("kind").to_numpy()
        hist = {}
        for k in kinds:
            hist[int(k)] = hist.get(int(k), 0) + 1
        kind_str = " ".join(f"{k}:{c}" for k, c in sorted(hist.items()))
        print(
            f"  {sp.name:28s} rows={n:7d} bytes={sp.stat().st_size:9d} "
            f"kinds={{ {kind_str} }} -> {out_path.name}"
        )

    print(f"[spill_to_parquet] done. total rows={total_rows}")


if __name__ == "__main__":
    main()
