# Ygg Event Schema

## Event Layout

Each event is a fixed 48-byte structure:

```cpp
struct Event {
    uint64_t timestamp_ns;  // 8 bytes
    uint32_t cpu;           // 4 bytes
    uint32_t pid;           // 4 bytes
    uint32_t tid;           // 4 bytes
    uint16_t kind;          // 2 bytes
    uint16_t padding;       // 2 bytes (alignment)
    uint64_t arg0;          // 8 bytes
    uint64_t arg1;          // 8 bytes
    uint64_t arg2;          // 8 bytes
    // Total: 48 bytes
};
```

## Event Kinds

### Application Events (1000+)
Reserved for user-defined semantic events. Applications register their own event types at runtime via the instrumentation library.

### Syscall Events (2000-2001)
- `SysEnter` (2000): arg0 = syscall number, arg1-2 = first two args
- `SysExit` (2001): arg0 = syscall number, arg1 = return value

### Scheduler Events (3000-3002)
- `SchedSwitch` (3000): arg0 = prev_tid, arg1 = next_tid, arg2 = prev_state
- `SchedWakeup` (3001): arg0 = target_tid, arg1 = target_cpu
- `SchedMigrate` (3002): arg0 = tid, arg1 = from_cpu, arg2 = to_cpu

### Block I/O Events (4000-4001)
- `BlockRqIssue` (4000): arg0 = sector, arg1 = bytes, arg2 = rw_flag
- `BlockRqComplete` (4001): arg0 = sector, arg1 = bytes, arg2 = latency_ns

### Network Events (5000-5001)
- `TcpSendmsg` (5000): arg0 = bytes, arg1 = socket_fd
- `TcpRecvmsg` (5001): arg0 = bytes, arg1 = socket_fd

### Memory Events (6000-6001)
- `PageFault` (6000): arg0 = address, arg1 = error_code
- `PageFaultMajor` (6001): arg0 = address, arg1 = error_code

### Hardware Counters (7000-7004)
Sampled via `perf_event_open()` at configurable intervals.
- `PerfCycles` (7000): arg0 = delta since last sample
- `PerfInstructions` (7001): arg0 = delta
- `PerfCacheMisses` (7002): arg0 = delta
- `PerfBranchMisses` (7003): arg0 = delta
- `PerfContextSwitches` (7004): arg0 = delta

### Loki Fault Injection (8000)
- `LokiInject` (8000): arg0 = fault_type_id, arg1 = param0, arg2 = param1

### Custom Events (9000)
Dynamic application events registered at runtime.

## Arrow/Parquet Schema

The Parquet file uses the following Arrow schema:

```python
import pyarrow as pa

event_schema = pa.schema([
    ('timestamp_ns', pa.uint64()),
    ('cpu', pa.uint32()),
    ('pid', pa.uint32()),
    ('tid', pa.uint32()),
    ('kind', pa.uint16()),
    ('arg0', pa.uint64()),
    ('arg1', pa.uint64()),
    ('arg2', pa.uint64()),
])

metadata_schema = pa.schema([
    ('git_sha', pa.string()),
    ('machine_fingerprint', pa.string()),
    ('kernel_version', pa.string()),
    ('compiler', pa.string()),
    ('workload', pa.string()),
    ('loki_fault_plan', pa.string()),
])

# File layout:
# - Row group 1: metadata (single row)
# - Row group 2+: events (partitioned by time windows)
```

## Timestamp Normalization

All timestamps are normalized to `CLOCK_MONOTONIC_RAW` nanoseconds at collector startup using `rdtsc`/`rdtscp` calibration against `clock_gettime(CLOCK_MONOTONIC_RAW)`.

The collector records:
- `tsc_freq_hz`: calibrated TSC frequency
- `monotonic_offset_ns`: offset between TSC and monotonic clock at calibration

Conversion: `timestamp_ns = (rdtsc() * 1e9) / tsc_freq_hz + monotonic_offset_ns`