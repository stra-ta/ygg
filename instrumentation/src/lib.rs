//! Ygg instrumentation FFI crate.
//!
//! The ultra-low-overhead event emission hot path is implemented in C
//! (`src/ygg.c`, `src/ring_buffer.c`, `src/collector_thread.c`) and linked into
//! this crate by `build.rs`. This Rust crate wraps the C ABI and provides a
//! small helper for receiving the event stream a collector forwards over a
//! Unix socket (the "forwards to Rust collector via Unix socket" path).
//!
//! Building this crate produces `libygg_instrumentation.a` (staticlib) and
//! `libygg_instrumentation.so` (cdylib).

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_ulonglong};

extern "C" {
    pub fn ygg_init(process_name: *const c_char) -> c_int;
    pub fn ygg_shutdown();
    pub fn ygg_register_event(name: *const c_char) -> c_uint;
    pub fn ygg_emit(kind: c_uint, arg0: c_ulonglong, arg1: c_ulonglong, arg2: c_ulonglong);
    pub fn ygg_try_emit(
        kind: c_uint,
        arg0: c_ulonglong,
        arg1: c_ulonglong,
        arg2: c_ulonglong,
    ) -> c_int;
    pub fn ygg_flush();
    pub fn ygg_collector_set_socket(socket_path: *const c_char);
    pub fn ygg_collector_set_output(file_path: *const c_char);
    pub fn ygg_collector_dropped() -> c_ulonglong;
    pub fn ygg_read_tsc() -> c_ulonglong;
    pub fn ygg_calibrate_tsc() -> c_ulonglong;

    /* Calibration constants published by ygg_init(). */
    pub static ygg_tsc_freq_hz: u64;
    pub static ygg_tsc_to_ns_mult: u64;
    pub static ygg_monotonic_offset_ns: u64;
}

/* -------------------------------------------------------------------------- */
/* Safe Rust wrappers                                                         */
/* -------------------------------------------------------------------------- */

/// Initialize the library. Must be called once before emitting events.
pub fn init(name: &str) -> Result<(), i32> {
    let c = CString::new(name).map_err(|_| -1)?;
    let r = unsafe { ygg_init(c.as_ptr()) };
    if r == 0 {
        Ok(())
    } else {
        Err(r)
    }
}

/// Shut the collector down and flush remaining events.
pub fn shutdown() {
    unsafe { ygg_shutdown() }
}

/// Register a custom event type, returning the assigned kind id.
pub fn register_event(name: &str) -> u16 {
    let c = CString::new(name).expect("event name must not contain NUL");
    unsafe { ygg_register_event(c.as_ptr()) as u16 }
}

/// Emit an event unconditionally (drops silently when the ring is full).
pub fn emit(kind: u16, arg0: u64, arg1: u64, arg2: u64) {
    unsafe { ygg_emit(kind as c_uint, arg0, arg1, arg2) }
}

/// Try to emit an event. Returns `false` if the ring was full.
pub fn try_emit(kind: u16, arg0: u64, arg1: u64, arg2: u64) -> bool {
    unsafe { ygg_try_emit(kind as c_uint, arg0, arg1, arg2) != 0 }
}

/// Explicit flush point (rings drain asynchronously via the collector).
pub fn flush() {
    unsafe { ygg_flush() }
}

/// Forward events to a Unix socket (the Rust `ygg-collector`).
pub fn set_collector_socket(path: &str) {
    let c = CString::new(path).expect("socket path must not contain NUL");
    unsafe { ygg_collector_set_socket(c.as_ptr()) }
}

/// Spill raw 48-byte-per-event records to a file (fallback / verification).
pub fn set_collector_output(path: &str) {
    let c = CString::new(path).expect("file path must not contain NUL");
    unsafe { ygg_collector_set_output(c.as_ptr()) }
}

/// Number of events dropped (ring full or no reachable sink).
pub fn dropped() -> u64 {
    unsafe { ygg_collector_dropped() }
}

/// Raw TSC / monotonic reading.
pub fn read_tsc() -> u64 {
    unsafe { ygg_read_tsc() }
}

/// Calibrated TSC frequency in Hz.
pub fn tsc_freq_hz() -> u64 {
    unsafe { ygg_tsc_freq_hz }
}

/// Fixed-point TSC -> ns multiplier (32.32).
pub fn tsc_to_ns_mult() -> u64 {
    unsafe { ygg_tsc_to_ns_mult }
}

/// Monotonic offset (ns) applied after the TSC -> ns conversion.
pub fn monotonic_offset_ns() -> u64 {
    unsafe { ygg_monotonic_offset_ns }
}

/* -------------------------------------------------------------------------- */
/* Receiver helper: read the Unix-socket / spill-file stream                 */
/* -------------------------------------------------------------------------- */

/// A single 48-byte event record (matches the C `struct ygg_event`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub timestamp_ns: u64,
    pub cpu: u32,
    pub pid: u32,
    pub tid: u32,
    pub kind: u16,
    pub padding: u16,
    pub arg0: u64,
    pub arg1: u64,
    pub arg2: u64,
}

/// Header written once at the start of a stream (socket or spill file).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StreamHeader {
    pub magic: u64,
    pub tsc_freq_hz: u64,
    pub tsc_to_ns_mult: u64,
    pub monotonic_offset_ns: u64,
}

const YGG_SHM_MAGIC: u64 = 0x5967673100000001;
const EVENT_SIZE: usize = 48;
const HEADER_SIZE: usize = 32;

/// Per-thread ring capacity (matches YGG_RING_SIZE in the C sources).
pub const YGG_RING_SIZE: usize = 1 << 16;

/// True if the stream header carries the expected Ygg magic value.
pub fn is_valid_magic(h: &StreamHeader) -> bool {
    h.magic == YGG_SHM_MAGIC
}

fn read_exact_or_eof(reader: &mut dyn std::io::Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        match reader.read(&mut buf[total..])? {
            0 => break,
            n => total += n,
        }
    }
    Ok(total)
}

/// Parse a raw 48-byte little-endian event record.
pub fn parse_event(bytes: &[u8; EVENT_SIZE]) -> Event {
    let u64le = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let u32le = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u16le = |o: usize| u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap());
    Event {
        timestamp_ns: u64le(0),
        cpu: u32le(8),
        pid: u32le(12),
        tid: u32le(16),
        kind: u16le(20),
        padding: u16le(22),
        arg0: u64le(24),
        arg1: u64le(32),
        arg2: u64le(40),
    }
}

/// Parse the 32-byte stream header.
pub fn parse_header(bytes: &[u8; HEADER_SIZE]) -> StreamHeader {
    let u64le = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    StreamHeader {
        magic: u64le(0),
        tsc_freq_hz: u64le(8),
        tsc_to_ns_mult: u64le(16),
        monotonic_offset_ns: u64le(24),
    }
}

/// Read the event stream from any reader (socket or spill file).
///
/// Calls `on_header` once with the stream header and `on_event` for each event.
/// Returns when the stream ends (EOF) or an error occurs.
pub fn read_stream<R: std::io::Read>(
    mut reader: R,
    mut on_header: impl FnMut(StreamHeader),
    mut on_event: impl FnMut(Event),
) -> std::io::Result<()> {
    let mut hbuf = [0u8; HEADER_SIZE];
    if read_exact_or_eof(&mut reader, &mut hbuf)? < HEADER_SIZE {
        return Ok(()); // empty / truncated
    }
    let header = parse_header(&hbuf);
    on_header(header);

    let mut ebuf = [0u8; EVENT_SIZE];
    loop {
        let n = read_exact_or_eof(&mut reader, &mut ebuf)?;
        if n < EVENT_SIZE {
            break; // clean EOF or trailing partial record
        }
        on_event(parse_event(&ebuf));
    }
    Ok(())
}

/// Read a spill file produced by the collector into a vector of events.
pub fn read_spill_file(path: &std::path::Path) -> std::io::Result<(StreamHeader, Vec<Event>)> {
    let mut file = std::fs::File::open(path)?;
    let mut header = None;
    let mut events = Vec::new();
    read_stream(
        &mut file,
        |h| header = Some(h),
        |e| events.push(e),
    )?;
    Ok((header.expect("spill file has a header"), events))
}

/* -------------------------------------------------------------------------- */
/* Tests                                                                      */
/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /* The library is process-global (single shm region + single collector
     * thread), so the tests must not run concurrently. */
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn emit_and_collect() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let pid = std::process::id();
        let spill = PathBuf::from(format!("/tmp/ygg-{}.events", pid));
        let _ = std::fs::remove_file(&spill);

        // Configure the spill file explicitly (collector config is process-
        // global and may persist across init/shutdown within the process).
        set_collector_socket("");
        set_collector_output(spill.to_str().unwrap());

        let drops_before = dropped();
        init("smoke").expect("init");
        let n: u64 = 2000;
        for i in 0..n {
            emit(1000, i, i.wrapping_mul(2), i.wrapping_mul(3));
        }
        // Let the collector drain at least once.
        std::thread::sleep(std::time::Duration::from_millis(60));
        shutdown();

        let (header, events) = read_spill_file(&spill).expect("read spill");
        assert!(is_valid_magic(&header), "bad stream magic");
        assert_eq!(
            events.len() as u64, n,
            "expected {} events, got {}", n, events.len()
        );
        assert_eq!(events[0].kind, 1000);
        assert_eq!(events[0].arg0, 0);
        let last = &events[events.len() - 1];
        assert_eq!(last.arg0, n - 1);
        assert_eq!(
            dropped() - drops_before, 0,
            "low-volume run should not drop"
        );
        println!(
            "captured {} events, tsc_freq_hz={}",
            events.len(),
            tsc_freq_hz()
        );
        let _ = std::fs::remove_file(&spill);
    }

    #[test]
    fn drop_when_no_sink() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let pid = std::process::id();

        // Point the collector at a Unix socket nothing is listening on, so
        // every drained event is dropped and counted by the collector.
        let sock = format!("/tmp/ygg-no-listener-{}.sock", pid);
        let _ = std::fs::remove_file(&sock);
        set_collector_socket("");
        set_collector_output("");
        set_collector_socket(&sock);

        let drops_before = dropped();
        init("smoke2").expect("init");

        let n: u64 = 5000;
        for i in 0..n {
            emit(1000, i, 0, 0);
        }
        // Let the collector drain (and drop) everything it can.
        std::thread::sleep(std::time::Duration::from_millis(80));
        shutdown();

        let delta = dropped() - drops_before;
        assert!(delta > 0, "collector should count dropped events");
        assert!(delta >= n, "all {} emitted events should be dropped", n);
        let _ = std::fs::remove_file(&sock);
    }
}
