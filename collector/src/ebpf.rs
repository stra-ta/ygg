//! eBPF loading and per-CPU ring-buffer polling.
//!
//! On Linux this module uses `aya` to load `probes.bpf.o`, attach the
//! tracepoints / kprobes declared in the BPF program, and poll the shared
//! `events` ring buffer (via `aya::maps::RingBuf`, which is per-CPU under the
//! hood and uses epoll on the underlying perf fds). Kernel records are decoded
//! into [`Event`]s and forwarded to the writer thread.
//!
//! On non-Linux hosts (e.g. macOS, where `aya` cannot compile) the collector is
//! built without kernel tracing: `run` simply idles until shutdown. This keeps
//! the crate buildable on any target while the real implementation is selected
//! automatically on Linux.

/// Calibration between the time-stamp counter (TSC) and
/// `CLOCK_MONOTONIC_RAW`, captured at startup.
///
/// Kernel BPF events already carry monotonic nanoseconds (`bpf_ktime_get_ns`),
/// but hardware perf samples and any userspace timestamp need a TSC -> monotonic
/// conversion, so the collector records `tsc_freq_hz` and `monotonic_offset_ns`.
#[derive(Debug, Clone, Copy)]
pub struct TscCalibration {
    pub tsc_freq_hz: f64,
    pub monotonic_offset_ns: i64,
    pub available: bool,
}

impl TscCalibration {
    /// A degenerate calibration (identity mapping) used on platforms without a
    /// usable TSC / `CLOCK_MONOTONIC_RAW`.
    pub fn unavailable() -> Self {
        Self {
            tsc_freq_hz: 1.0,
            monotonic_offset_ns: 0,
            available: false,
        }
    }

    /// Convert a raw TSC value to `CLOCK_MONOTONIC_RAW` nanoseconds.
    #[allow(dead_code)]
    pub fn tsc_to_monotonic_ns(&self, tsc: u64) -> u64 {
        if !self.available {
            return tsc;
        }
        let ns = (tsc as f64 / self.tsc_freq_hz * 1e9) as i64 + self.monotonic_offset_ns;
        ns.max(0) as u64
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod tsc_impl {
    use super::TscCalibration;
    use std::time::Duration;

    #[inline]
    unsafe fn rdtsc() -> u64 {
        std::arch::x86_64::_rdtsc()
    }

    fn mono_raw_ns() -> i64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut ts) };
        ts.tv_sec as i64 * 1_000_000_000 + ts.tv_nsec as i64
    }

    /// Measure TSC frequency by sampling TSC against `CLOCK_MONOTONIC_RAW` over a
    /// short busy window, then compute the offset at the first sample.
    pub fn calibrate() -> TscCalibration {
        let t0 = unsafe { rdtsc() };
        let mono0 = mono_raw_ns();

        let spin_start = mono_raw_ns();
        while mono_raw_ns() - spin_start < 5_000_000 {
            std::thread::yield_now();
        }

        let t1 = unsafe { rdtsc() };
        let mono1 = mono_raw_ns();

        let dt_ns = (mono1 - mono0) as f64;
        let dt_tsc = (t1 - t0) as f64;
        let freq = if dt_ns > 0.0 {
            dt_tsc / dt_ns * 1e9
        } else {
            1.0
        };

        let tsc0_ns = (t0 as f64 / freq * 1e9) as i64;
        let offset = mono0 - tsc0_ns;

        TscCalibration {
            tsc_freq_hz: freq,
            monotonic_offset_ns: offset,
            available: true,
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
impl TscCalibration {
    /// Calibrate the TSC against `CLOCK_MONOTONIC_RAW` (x86_64 Linux).
    pub fn calibrate() -> Self {
        tsc_impl::calibrate()
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
impl TscCalibration {
    /// Calibration is unavailable on this target; returns a no-op mapping.
    pub fn calibrate() -> Self {
        Self::unavailable()
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::TscCalibration;
    use ygg_schema::Event;
    use crate::Args;
    use anyhow::{Context, Result};
    use crossbeam::channel::Sender;
    use std::path::PathBuf;
    use std::time::Duration;
    use tracing::warn;

    /// Locate the compiled BPF object. Honors `YGG_BPF_OBJECT`, then falls back
    /// to a few conventional locations.
    fn find_bpf_object() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("YGG_BPF_OBJECT") {
            let pb = PathBuf::from(p);
            if pb.exists() {
                return Some(pb);
            }
        }
        for cand in &[
            "target/bpfel-unknown-none/release/ygg-probes.o",
            "target/bpfel-unknown-none/release/probes.bpf.o",
            "collector/target/bpfel-unknown-none/release/ygg-probes.o",
            "/usr/lib/ygg/probes.bpf.o",
            "/usr/local/lib/ygg/probes.bpf.o",
        ] {
            let pb = PathBuf::from(cand);
            if pb.exists() {
                return Some(pb);
            }
        }
        None
    }

    fn attach_btf(
        bpf: &mut aya::Bpf,
        name: &str,
        tp: &str,
    ) -> Result<aya::programs::Link> {
        use aya::programs::BtfTracePoint;
        let prog: &mut BtfTracePoint = bpf
            .program_mut(name)
            .with_context(|| format!("eBPF program `{name}` not found in object"))?
            .try_into()?;
        prog.load()?;
        Ok(prog.attach(tp)?)
    }

    fn attach_kprobe(
        bpf: &mut aya::Bpf,
        name: &str,
        ret: bool,
    ) -> Result<aya::programs::Link> {
        if ret {
            use aya::programs::KRetProbe;
            let prog: &mut KRetProbe = bpf
                .program_mut(name)
                .with_context(|| format!("eBPF program `{name}` not found in object"))?
                .try_into()?;
            prog.load()?;
            Ok(prog.attach(name, 0)?)
        } else {
            use aya::programs::KProbe;
            let prog: &mut KProbe = bpf
                .program_mut(name)
                .with_context(|| format!("eBPF program `{name}` not found in object"))?
                .try_into()?;
            prog.load()?;
            Ok(prog.attach(name, 0)?)
        }
    }

    pub fn run(
        sender: Sender<Event>,
        args: &Args,
        shutdown: &Arc<AtomicBool>,
        _calib: &TscCalibration,
    ) -> Result<()> {
        let obj = find_bpf_object().context(
            "probes.bpf.o not found; build the BPF object (see build.rs) and set YGG_BPF_OBJECT",
        )?;
        let mut bpf = aya::Bpf::load_file(&obj)?;

        // Keep every link alive for the lifetime of the collector; dropping a
        // link detaches the program.
        let mut _links: Vec<aya::programs::Link> = Vec::new();

        if args.sched {
            _links.push(attach_btf(&mut bpf, "sched_switch", "sched/sched_switch")?);
            _links.push(attach_btf(&mut bpf, "sched_wakeup", "sched/sched_wakeup")?);
            _links.push(attach_btf(
                &mut bpf,
                "sched_migrate_task",
                "sched/sched_migrate_task",
            )?);
        }
        if args.syscalls {
            _links.push(attach_btf(&mut bpf, "sys_enter", "syscalls/sys_enter")?);
            _links.push(attach_btf(&mut bpf, "sys_exit", "syscalls/sys_exit")?);
        }
        if args.block_io {
            _links.push(attach_btf(&mut bpf, "block_rq_issue", "block/block_rq_issue")?);
            _links.push(attach_btf(
                &mut bpf,
                "block_rq_complete",
                "block/block_rq_complete",
            )?);
        }
        if args.network {
            _links.push(attach_kprobe(&mut bpf, "tcp_sendmsg", false)?);
            _links.push(attach_kprobe(&mut bpf, "tcp_recvmsg", true)?);
        }
        if args.page_faults {
            _links.push(attach_btf(
                &mut bpf,
                "page_fault_user",
                "exceptions/page_fault_user",
            )?);
            _links.push(attach_btf(
                &mut bpf,
                "page_fault_kernel",
                "exceptions/page_fault_kernel",
            )?);
        }

        // Per-CPU ring buffer (BPF_MAP_TYPE_RINGBUF); aya reads all CPUs.
        let mut events = bpf.map_mut("events").context("`events` ring buffer map missing")?;
        let mut ring = aya::maps::RingBuf::try_from(&*events)?;

        info!("eBPF collection started");
        let poll = Duration::from_millis(100);
        while !shutdown.load(Ordering::Relaxed) {
            if let Err(e) = ring.poll(poll) {
                warn!("ring buffer poll error: {e}");
                break;
            }
            while let Some(item) = ring.next() {
                if let Some(ev) = Event::from_kernel_record(item) {
                    if args.pid != 0 && ev.pid != args.pid {
                        continue;
                    }
                    if sender.send(ev).is_err() {
                        // Writer is gone; shut down.
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::TscCalibration;
    use ygg_schema::Event;
    use crate::Args;
    use anyhow::Result;
    use crossbeam::channel::Sender;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tracing::warn;

    pub fn run(
        _sender: Sender<Event>,
        _args: &Args,
        shutdown: &Arc<AtomicBool>,
        _calib: &TscCalibration,
    ) -> Result<()> {
        warn!(
            "eBPF collection is only supported on Linux; this build runs without kernel tracing"
        );
        while !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(200));
        }
        Ok(())
    }
}

pub use imp::run;
