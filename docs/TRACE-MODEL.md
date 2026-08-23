# Trace data model and event schema for Ygg.

All on-wire events are a single fixed-width record. The model and the
collector share this layout through `schema/src/lib.rs` (Rust) and the
documentation in `schema/EVENT_SCHEMA.md`.

## Event struct

Defined as `Event` in `schema/src/lib.rs`. The on-wire record is 48 bytes
(`EVENT_SIZE` / `KERNEL_RECORD_SIZE`). Note: the Rust `#[repr(C)]` struct is
padded by the compiler, so `size_of::<Event>()` is larger than 48; the
collector serializes field-by-field from the eBPF ring buffer via
`Event::from_kernel_record`.

```rust
pub struct Event {
    pub timestamp_ns: u64,  // 8 bytes, normalized to CLOCK_MONOTONIC_RAW
    pub cpu:          u32,  // 4 bytes
    pub pid:          u32,  // 4 bytes
    pub tid:          u32,  // 4 bytes
    pub kind:         u16,  // 2 bytes, EventKind discriminant
    pub padding:      u16,  // 2 bytes, alignment only
    pub arg0:         u64,  // 8 bytes, event-specific
    pub arg1:         u64,  // 8 bytes
    pub arg2:         u64,  // 8 bytes
}
// Total: 48 bytes
```

The 2-byte `padding` field exists only for on-wire alignment. It is dropped
from the column schema (see below).

## EventKind enum

`EventKind` is a `#[repr(u16)]` discriminant for `Event::kind`. The ranges
below are reserved; the concrete values defined in code are:

| Range   | Domain               | Defined kinds |
| ---     | ---                  | --- |
| 1000+   | Application          | `AppBase = 1000` |
| 2000+   | Syscalls             | `SysEnter = 2000`, `SysExit = 2001` |
| 3000+   | Scheduler            | `SchedSwitch = 3000`, `SchedWakeup = 3001`, `SchedMigrate = 3002` |
| 4000+   | Block I/O            | `BlockRqIssue = 4000`, `BlockRqComplete = 4001` |
| 5000+   | Network              | `TcpSendmsg = 5000`, `TcpRecvmsg = 5001` |
| 6000+   | Memory               | `PageFault = 6000`, `PageFaultMajor = 6001` |
| 7000+   | Hardware counters    | `PerfCycles = 7000` .. `PerfContextSwitches = 7004` |
| 8000+   | Loki fault injection | `LokiInject = 8000` |
| 9000+   | Custom               | `Custom = 9000` |

The `arg0/1/2` meaning is per-kind and documented in
`schema/EVENT_SCHEMA.md` (e.g. `SchedSwitch` uses arg0=prev_tid,
arg1=next_tid, arg2=prev_state; `PerfCycles` arg0=delta since last sample).

## Arrow / Parquet schema

Produced by `arrow_schema()` in `schema/src/lib.rs`. The `padding` field is
intentionally omitted from the column schema; only the real columns are
stored.

```text
timestamp_ns : UInt64
cpu          : UInt32
pid          : UInt32
tid          : UInt32
kind         : UInt16
arg0         : UInt64
arg1         : UInt64
arg2         : UInt64
```

Metadata for the whole execution is stored as a single-row leading row group
via `metadata_schema()`:

```text
git_sha           : Utf8
machine_fingerprint : Utf8
kernel_version    : Utf8
compiler          : Utf8
workload          : Utf8
loki_fault_plan   : Utf8
```

See `schema/EVENT_SCHEMA.md` for the Python `pyarrow` equivalent and the
full file layout.

## ExecutionMetadata

`ExecutionMetadata` (Rust, also `metadata_schema()` columns) records
provenance for one collected execution:

- `git_sha` - source revision the workload was built from
- `machine_fingerprint` - stable identifier for the host
- `kernel_version` - OS kernel at collection time
- `compiler` - toolchain that built the workload
- `workload` - workload / scenario identifier
- `loki_fault_plan` - Loki fault-injection plan used, if any

An empty record (`ExecutionMetadata::new()`) has all-blank fields; this is
the default written when provenance is unavailable.

## Timestamp normalization

All timestamps are normalized to `CLOCK_MONOTONIC_RAW` nanoseconds at
collector startup. The collector calibrates the TSC against
`clock_gettime(CLOCK_MONOTONIC_RAW)` using `rdtsc`/`rdtscp` and records:

- `tsc_freq_hz` - calibrated TSC frequency
- `monotonic_offset_ns` - offset between TSC and the monotonic clock at
  calibration

Conversion applied per event:

```text
timestamp_ns = (rdtsc() * 1e9) / tsc_freq_hz + monotonic_offset_ns
```

This keeps events from different CPUs comparable without wall-clock
dependency.

## Storage layout

The collector (`collector/src/main.rs`) encodes events as Arrow
`RecordBatch`es (8192 rows per batch) and writes them to Parquet with ZSTD
compression.

- Columnar Parquet, ZSTD compression.
- Row group 1: the single `ExecutionMetadata` row.
- Row groups 2+: event rows, partitioned per time window.

This layout makes the metadata a cheap first read and lets downstream
tooling stream per-window batches.

## See also

- `schema/EVENT_SCHEMA.md` - full event layout, per-kind arg semantics, and
  the Arrow/Parquet schema in Python.
- `schema/src/lib.rs` - `Event`, `EventKind`, `ExecutionMetadata`,
  `arrow_schema()`, `metadata_schema()` definitions.
- `collector/src/main.rs` - the eBPF/perf collection and Parquet writer
  pipeline that produces these files.
