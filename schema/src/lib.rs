//! Ygg event schema and Arrow/Parquet definitions.
//!
//! Defines the fixed 48-byte on-wire `Event` layout, the `EventKind`
//! discriminant enum, the Arrow `Schema` for the event row groups, and the
//! `ExecutionMetadata` row for the leading metadata row group.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

/// Size in bytes of the fixed `Event` record as serialized to the Parquet
/// event row groups. The C layout documented in `EVENT_SCHEMA.md` sums to
/// 48 bytes.
pub const EVENT_SIZE: usize = 48;

/// A single fixed-width telemetry record.
///
/// The field order and widths match the C `struct Event` in
/// `EVENT_SCHEMA.md`. Note: under `#[repr(C)]` the trailing `arg*` u64 fields
/// are naturally 8-byte aligned, so `size_of::<Event>()` is larger than
/// `EVENT_SIZE`; consumers must serialize field-by-field (or via packed
/// views) rather than relying on `size_of`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Event {
    pub timestamp_ns: u64,
    pub cpu: u32,
    pub pid: u32,
    pub tid: u32,
    pub kind: u16,
    pub padding: u16, // alignment
    pub arg0: u64,
    pub arg1: u64,
    pub arg2: u64,
}

impl Event {
    /// Size of the raw kernel ring buffer record (48 bytes).
    pub const KERNEL_RECORD_SIZE: usize = 48;

    /// Decode a 48-byte kernel record from the eBPF ring buffer.
    /// The record layout matches the kernel-side `struct event` from probes.bpf.c:
    /// timestamp_ns (u64), cpu (u32), pid (u32), tid (u32), kind (u16), padding (u16), arg0/1/2 (u64).
    #[cfg(target_os = "linux")]
    pub fn from_kernel_record(record: &[u8]) -> Option<Self> {
        if record.len() < Self::KERNEL_RECORD_SIZE {
            return None;
        }
        use std::convert::TryInto;
        let timestamp_ns = u64::from_ne_bytes(record[0..8].try_into().ok()?);
        let cpu = u32::from_ne_bytes(record[8..12].try_into().ok()?);
        let pid = u32::from_ne_bytes(record[12..16].try_into().ok()?);
        let tid = u32::from_ne_bytes(record[16..20].try_into().ok()?);
        let kind = u16::from_ne_bytes(record[20..22].try_into().ok()?);
        // bytes 22..24 = padding, skip
        let arg0 = u64::from_ne_bytes(record[24..32].try_into().ok()?);
        let arg1 = u64::from_ne_bytes(record[32..40].try_into().ok()?);
        let arg2 = u64::from_ne_bytes(record[40..48].try_into().ok()?);
        Some(Self { timestamp_ns, cpu, pid, tid, kind, padding: 0, arg0, arg1, arg2 })
    }
}

/// Discriminant for `Event::kind`, matching `event.fbs`.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    AppBase = 1000,
    SysEnter = 2000,
    SysExit = 2001,
    SchedSwitch = 3000,
    SchedWakeup = 3001,
    SchedMigrate = 3002,
    BlockRqIssue = 4000,
    BlockRqComplete = 4001,
    TcpSendmsg = 5000,
    TcpRecvmsg = 5001,
    PageFault = 6000,
    PageFaultMajor = 6001,
    PerfCycles = 7000,
    PerfInstructions = 7001,
    PerfCacheMisses = 7002,
    PerfBranchMisses = 7003,
    PerfContextSwitches = 7004,
    LokiInject = 8000,
    Custom = 9000,
}

impl EventKind {
    /// All event kinds in declaration order.
    pub const ALL: &'static [EventKind] = &[
        EventKind::AppBase,
        EventKind::SysEnter,
        EventKind::SysExit,
        EventKind::SchedSwitch,
        EventKind::SchedWakeup,
        EventKind::SchedMigrate,
        EventKind::BlockRqIssue,
        EventKind::BlockRqComplete,
        EventKind::TcpSendmsg,
        EventKind::TcpRecvmsg,
        EventKind::PageFault,
        EventKind::PageFaultMajor,
        EventKind::PerfCycles,
        EventKind::PerfInstructions,
        EventKind::PerfCacheMisses,
        EventKind::PerfBranchMisses,
        EventKind::PerfContextSwitches,
        EventKind::LokiInject,
        EventKind::Custom,
    ];
}

/// Metadata describing a single execution, stored as the leading single-row
/// row group of the Parquet file.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionMetadata {
    pub git_sha: String,
    pub machine_fingerprint: String,
    pub kernel_version: String,
    pub compiler: String,
    pub workload: String,
    pub loki_fault_plan: String,
}

impl ExecutionMetadata {
    /// Construct an empty metadata record (all fields blank).
    pub fn new() -> Self {
        ExecutionMetadata {
            git_sha: String::new(),
            machine_fingerprint: String::new(),
            kernel_version: String::new(),
            compiler: String::new(),
            workload: String::new(),
            loki_fault_plan: String::new(),
        }
    }
}

impl Default for ExecutionMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Arrow schema for the event row groups.
///
/// Matches the Parquet layout in `EVENT_SCHEMA.md`. The `padding` field is
/// omitted from the column schema; it exists only in the on-wire struct for
/// alignment.
pub fn arrow_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("timestamp_ns", DataType::UInt64, false),
        Field::new("cpu", DataType::UInt32, false),
        Field::new("pid", DataType::UInt32, false),
        Field::new("tid", DataType::UInt32, false),
        Field::new("kind", DataType::UInt16, false),
        Field::new("arg0", DataType::UInt64, false),
        Field::new("arg1", DataType::UInt64, false),
        Field::new("arg2", DataType::UInt64, false),
    ]))
}

/// Arrow schema for the `ExecutionMetadata` row group (single row).
pub fn metadata_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("git_sha", DataType::Utf8, false),
        Field::new("machine_fingerprint", DataType::Utf8, false),
        Field::new("kernel_version", DataType::Utf8, false),
        Field::new("compiler", DataType::Utf8, false),
        Field::new("workload", DataType::Utf8, false),
        Field::new("loki_fault_plan", DataType::Utf8, false),
    ]))
}
