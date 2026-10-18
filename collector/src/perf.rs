//! Hardware counter sampling via `perf_event_open`.
//!
//! For each requested counter we open a `PERF_TYPE_HARDWARE` event, `mmap` its
//! ring buffer, and add its fd to an `epoll` set. On wakeup we walk the ring
//! buffer, parse `PERF_RECORD_SAMPLE` records, compute the delta since the last
//! sample, and emit an [`Event`] tagged with the matching kind (7000..7004).
//!
//! Only compiled on Linux (it relies on Linux-specific `libc` interfaces). On
//! other platforms `run` is a no-op that idles until shutdown.

#[cfg(target_os = "linux")]
mod imp {
    use crate::ebpf::TscCalibration;
    use crate::Args;
    use anyhow::{anyhow, Result};
    use crossbeam::channel::Sender;
    use std::ptr;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tracing::{info, warn};
    use ygg_schema::Event;

    // libc does not export the perf_event uapi surface, so declare the
    // subset this collector touches. Layout must match the kernel's
    // <uapi/linux/perf_event.h>.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PerfEventAttr {
        type_: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        bits: u64,
        wakeup_events: u32,
        bp_type: u32,
        bp_addr_or_config1: u64,
        bp_len_or_config2: u64,
        branch_sample_type: u64,
        sample_regs_user: u64,
        sample_stack_user: u32,
        clockid: i32,
        sample_regs_intr: u64,
        aux_watermark: u32,
        sample_max_stack: u16,
        reserved2: u16,
        aux_sample_size: u32,
        reserved3: u32,
    }

    #[repr(C)]
    struct PerfEventMmapPage {
        version: u32,
        compat_version: u32,
        lock: u32,
        index: u32,
        offset: i64,
        time_enabled: u64,
        time_running: u64,
        capabilities: u64,
        pmc_width: u16,
        time_shift: u16,
        time_mult: u32,
        time_zero: u64,
        time_offset: u64,
        time_cycles: u64,
        time_mask: u64,
        cap_user_time_zero: u16,
        cap_user_time_short: u16,
        reserved: [u32; 5],
        data_head: u64,
        data_tail: u64,
        data_offset: u64,
        data_size: u64,
    }

    // glibc exports the perf_event_open wrapper on only a few targets, so go
    // through the raw syscall instead; libc always provides the number.
    fn perf_event_open(
        attr: *mut PerfEventAttr,
        pid: libc::pid_t,
        cpu: libc::c_int,
        group_fd: libc::c_int,
        flags: libc::c_ulong,
    ) -> libc::c_int {
        unsafe {
            libc::syscall(libc::SYS_perf_event_open, attr, pid, cpu, group_fd, flags) as libc::c_int
        }
    }

    const PERF_TYPE_HARDWARE: u32 = 0;
    const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
    const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
    const PERF_COUNT_HW_CACHE_MISSES: u64 = 4;
    const PERF_COUNT_HW_BRANCH_MISSES: u64 = 5;
    const PERF_COUNT_HW_CONTEXT_SWITCHES: u64 = 3;

    const PERF_SAMPLE_IP: u64 = 1 << 0;
    const PERF_SAMPLE_TMESTAMP: u64 = 1 << 1;
    const PERF_SAMPLE_READ: u64 = 1 << 3;
    const PERF_SAMPLE_TID: u64 = 1 << 4;
    const PERF_SAMPLE_CPU: u64 = 1 << 5;

    const PERF_RECORD_SAMPLE: u32 = 9;

    const PERF_EVENT_IOC_ENABLE: libc::c_ulong = 0x2400;

    fn counter_config(name: &str) -> Result<u64> {
        Ok(match name {
            "cycles" => PERF_COUNT_HW_CPU_CYCLES,
            "instructions" => PERF_COUNT_HW_INSTRUCTIONS,
            "cache-misses" => PERF_COUNT_HW_CACHE_MISSES,
            "branch-misses" => PERF_COUNT_HW_BRANCH_MISSES,
            "context-switches" => PERF_COUNT_HW_CONTEXT_SWITCHES,
            other => return Err(anyhow!("unknown perf event: {other}")),
        })
    }

    fn kind_for(name: &str, idx: usize) -> u16 {
        match name {
            "cycles" => 7000,
            "instructions" => 7001,
            "cache-misses" => 7002,
            "branch-misses" => 7003,
            "context-switches" => 7004,
            _ => 7000 + idx as u16,
        }
    }

    struct Counter {
        kind: u16,
        fd: i32,
        base: u64,
        mmap: *mut libc::c_void,
        mmap_len: usize,
    }

    // The raw mmap pointer is only ever touched from the owning thread.
    unsafe impl Send for Counter {}

    impl Drop for Counter {
        fn drop(&mut self) {
            unsafe {
                if !self.mmap.is_null() {
                    libc::munmap(self.mmap, self.mmap_len);
                }
                if self.fd >= 0 {
                    libc::close(self.fd);
                }
            }
        }
    }

    fn open_counter(name: &str, config: u64, pid: i32, period: u64) -> Result<Counter> {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let order: usize = 4; // 16 data pages
        let n_pages = 1usize << order;
        let mmap_len = page_size * (1 + n_pages);

        let mut attr: PerfEventAttr = unsafe { std::mem::zeroed() };
        attr.type_ = PERF_TYPE_HARDWARE;
        attr.size = std::mem::size_of::<PerfEventAttr>() as u32;
        attr.config = config;
        attr.sample_period = period;
        attr.sample_type = PERF_SAMPLE_IP
            | PERF_SAMPLE_TMESTAMP
            | PERF_SAMPLE_READ
            | PERF_SAMPLE_TID
            | PERF_SAMPLE_CPU;
        // disabled = 1 (the first bit of the flags word) until we ioctl ENABLE.
        attr.bits = 1;
        attr.wakeup_events = 1;

        let fd = perf_event_open(&mut attr, pid, -1, -1, 0);
        if fd < 0 {
            return Err(anyhow!(
                "perf_event_open({name}) failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        let mmap = unsafe {
            libc::mmap(
                ptr::null_mut(),
                mmap_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if mmap == libc::MAP_FAILED {
            unsafe { libc::close(fd) };
            return Err(anyhow!(
                "mmap ring buffer for {name} failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        unsafe { libc::ioctl(fd, PERF_EVENT_IOC_ENABLE, 0 as libc::c_ulong) };

        Ok(Counter {
            kind: 0,
            fd,
            base: 0,
            mmap,
            mmap_len,
        })
    }

    /// Read `n` bytes from the circular data buffer at absolute offset `pos`,
    /// wrapping around `size`. Returns a heap buffer (records are small).
    unsafe fn read_bytes(base: *const u8, size: usize, pos: usize, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = *base.add((pos + i) % size);
        }
        out
    }

    unsafe fn read_u64(base: *const u8, size: usize, pos: usize) -> u64 {
        let b = read_bytes(base, size, pos, 8);
        u64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }

    unsafe fn read_u32(base: *const u8, size: usize, pos: usize) -> u32 {
        let b = read_bytes(base, size, pos, 4);
        u32::from_ne_bytes([b[0], b[1], b[2], b[3]])
    }

    /// Drain one counter's ring buffer, emitting an [`Event`] per sample.
    unsafe fn drain(c: &mut Counter, sender: &Sender<Event>) -> Result<()> {
        let page = c.mmap as *mut PerfEventMmapPage;
        let data_head = (*page).data_head as usize;
        let data_tail = (*page).data_tail as usize;
        let page_size = libc::sysconf(libc::_SC_PAGESIZE) as usize;
        let data_size = page_size * (c.mmap_len / page_size - 1);
        let data = (c.mmap as *const u8).add(page_size);

        let mut tail = data_tail;
        while tail + 8 <= data_head {
            let hdr = read_bytes(data, data_size, tail, 8);
            let rtype = u32::from_ne_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
            let rsize = u16::from_ne_bytes([hdr[4], hdr[5]]) as usize;
            if rsize < 8 {
                break;
            }
            if rtype == PERF_RECORD_SAMPLE {
                // Layout (in sample_type bit order): ip, time, read, tid, cpu.
                let mut off = tail + 8;
                let _ip = read_u64(data, data_size, off);
                off += 8;
                let time = read_u64(data, data_size, off);
                off += 8;
                let value = read_u64(data, data_size, off);
                off += 8;
                let pid = read_u32(data, data_size, off);
                off += 4;
                let tid = read_u32(data, data_size, off);
                off += 4;
                let cpu = read_u32(data, data_size, off);
                off += 4;
                let _reserved = read_u32(data, data_size, off);

                let delta = value.wrapping_sub(c.base);
                c.base = value;

                let ev = Event {
                    timestamp_ns: time,
                    cpu,
                    pid,
                    tid,
                    kind: c.kind,
                    padding: 0,
                    arg0: delta,
                    arg1: 0,
                    arg2: 0,
                };
                if sender.send(ev).is_err() {
                    return Ok(());
                }
            }
            tail = (tail + rsize) % data_size;
        }
        (*page).data_tail = tail as u64;
        Ok(())
    }

    pub fn run(
        sender: Sender<Event>,
        args: &Args,
        shutdown: &Arc<AtomicBool>,
        _calib: &TscCalibration,
    ) -> Result<()> {
        let names: Vec<String> = if args.perf_events.is_empty() {
            vec![
                "cycles".into(),
                "instructions".into(),
                "cache-misses".into(),
                "branch-misses".into(),
                "context-switches".into(),
            ]
        } else {
            args.perf_events.clone()
        };

        let (pid_arg, _cpu_arg) = if args.pid == 0 {
            (-1, -1)
        } else {
            (args.pid as i32, -1)
        };

        let mut counters: Vec<Counter> = Vec::new();
        for (i, name) in names.iter().enumerate() {
            let cfg = counter_config(name)?;
            let mut c = open_counter(name, cfg, pid_arg, args.perf_period)?;
            c.kind = kind_for(name, i);
            counters.push(c);
        }
        if counters.is_empty() {
            return Ok(());
        }

        let epoll = unsafe { libc::epoll_create1(0) };
        if epoll < 0 {
            return Err(anyhow!(
                "epoll_create1 failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        let mut events: Vec<libc::epoll_event> =
            vec![libc::epoll_event { events: 0, u64: 0 }; counters.len()];
        for (i, c) in counters.iter().enumerate() {
            let mut ev = libc::epoll_event {
                events: libc::EPOLLIN as u32,
                u64: (i + 1) as u64,
            };
            unsafe { libc::epoll_ctl(epoll, libc::EPOLL_CTL_ADD, c.fd, &mut ev) };
        }

        info!("perf collection started for {} counter(s)", counters.len());
        let timeout_ms = 100;
        while !shutdown.load(Ordering::Relaxed) {
            let n = unsafe {
                libc::epoll_wait(epoll, events.as_mut_ptr(), events.len() as i32, timeout_ms)
            };
            if n < 0 {
                warn!("epoll_wait failed: {}", std::io::Error::last_os_error());
                break;
            }
            for ev in events.iter().take(n as usize) {
                let idx = (ev.u64 as usize).wrapping_sub(1);
                if idx < counters.len() {
                    if let Err(e) = unsafe { drain(&mut counters[idx], &sender) } {
                        warn!("perf drain error: {e}");
                        unsafe { libc::close(epoll) };
                        return Ok(()); // writer gone
                    }
                }
            }
        }

        unsafe { libc::close(epoll) };
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use crate::ebpf::TscCalibration;
    use crate::Args;
    use anyhow::Result;
    use crossbeam::channel::Sender;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tracing::warn;
    use ygg_schema::Event;

    pub fn run(
        _sender: Sender<Event>,
        _args: &Args,
        shutdown: &Arc<AtomicBool>,
        _calib: &TscCalibration,
    ) -> Result<()> {
        warn!(
            "perf_event_open sampling is only supported on Linux; this build runs without hardware counters"
        );
        while !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(200));
        }
        Ok(())
    }
}

pub use imp::run;
