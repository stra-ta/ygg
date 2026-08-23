//! Ygg trace collector entry point.
//!
//! Architecture:
//! ```text
//! eBPF ringbuf + perf mmap
//!         |  (polled by sync threads, epoll on Linux)
//!         v
//! normalize timestamps (TSC -> MONOTONIC_RAW)
//!         |
//!         v
//! encode to Event  ---> crossbeam channel ---> writer thread (sync)
//!                                              |
//!                                              v
//!                                        Arrow RecordBatch (8192 rows)
//!                                              |
//!                                              v
//!                                        Parquet (ZSTD)
//! ```
//!
//! Tokio is used only for signal handling / graceful shutdown. All data-path
//! work runs on `spawn_blocking` sync threads so the hot loop never touches the
//! async runtime.

mod ebpf;
mod perf;
mod writer;

use anyhow::Result;
use clap::Parser;
use crossbeam::channel::bounded;
use ygg_schema::Event;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tracing::{info, warn};
use writer::{ParquetWriter, EVENT_BATCH_SIZE};

#[derive(Parser, Debug)]
#[command(name = "ygg-collector", version, about = "Ygg eBPF/perf trace collector")]
struct Args {
    /// Output Parquet file
    #[arg(short, long, default_value = "trace.parquet")]
    output: PathBuf,

    /// Enable scheduler events
    #[arg(long, default_value = "true")]
    sched: bool,

    /// Enable syscall events
    #[arg(long, default_value = "true")]
    syscalls: bool,

    /// Enable block I/O events
    #[arg(long, default_value = "true")]
    block_io: bool,

    /// Enable network events
    #[arg(long, default_value = "true")]
    network: bool,

    /// Enable page fault events
    #[arg(long, default_value = "true")]
    page_faults: bool,

    /// Enable hardware counter sampling (perf_event_open)
    #[arg(long)]
    perf: bool,

    /// Hardware counters to sample
    #[arg(long, value_delimiter = ',')]
    perf_events: Vec<String>,

    /// Perf sampling period (events between samples)
    #[arg(long, default_value = "1000000")]
    perf_period: u64,

    /// Target PID to trace (0 = all)
    #[arg(long, default_value = "0")]
    pid: u32,

    /// Duration to collect (seconds, 0 = until Ctrl-C)
    #[arg(long, default_value = "0")]
    duration: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Arc::new(Args::parse());

    let calib = ebpf::TscCalibration::calibrate();
    if calib.available {
        info!(
            "TSC calibration: {:.3} GHz, monotonic_offset_ns = {}",
            calib.tsc_freq_hz / 1e9,
            calib.monotonic_offset_ns
        );
    } else {
        warn!("TSC calibration unavailable on this platform; raw timestamps used as-is");
    }

    let writer = ParquetWriter::new(args.output.clone(), EVENT_BATCH_SIZE)?;
    let (tx, rx) = bounded::<Event>(100_000);
    let shutdown = Arc::new(AtomicBool::new(false));

    // Writer thread (sync).
    let w_shutdown = shutdown.clone();
    let writer_handle = tokio::task::spawn_blocking(move || writer.run(rx, &w_shutdown));

    // eBPF collector thread (sync).
    let e_args = args.clone();
    let e_shutdown = shutdown.clone();
    let e_tx = tx.clone();
    let e_calib = calib;
    let ebpf_handle = tokio::task::spawn_blocking(move || {
        ebpf::run(e_tx, &e_args, &e_shutdown, &e_calib)
    });

    // perf collector thread (sync), only if requested.
    let mut perf_handle = None;
    if args.perf {
        let p_args = args.clone();
        let p_shutdown = shutdown.clone();
        let p_tx = tx.clone();
        let p_calib = calib;
        perf_handle = Some(tokio::task::spawn_blocking(move || {
            perf::run(p_tx, &p_args, &p_shutdown, &p_calib)
        }));
    }

    // Block until duration elapses or Ctrl-C (Tokio owns only this path).
    if args.duration > 0 {
        tokio::time::sleep(Duration::from_secs(args.duration)).await;
    } else {
        signal::ctrl_c().await?;
    }

    info!("Shutdown requested; flushing collectors and finalizing Parquet");
    shutdown.store(true, Ordering::Relaxed);
    drop(tx); // signal writer + collectors that no more events are coming

    let count = writer_handle.await??;
    ebpf_handle.await??;
    if let Some(h) = perf_handle {
        h.await??;
    }

    info!("Wrote {} events to {}", count, args.output.display());
    Ok(())
}
