use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use arrow::array::{UInt64Array, UInt32Array, UInt16Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use crossbeam::channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::PathBuf;
use tokio::signal;
use tracing::{info, warn, error};

mod schema;
use schema::Event;

const EVENT_BATCH_SIZE: usize = 8192;
const FLUSH_INTERVAL_MS: u64 = 100;

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

    /// Perf sampling period (cycles between samples)
    #[arg(long, default_value = "1000000")]
    perf_period: u64,

    /// Target PID to trace (0 = all)
    #[arg(long, default_value = "0")]
    pid: u32,

    /// Duration to collect (seconds, 0 = infinite)
    #[arg(long, default_value = "0")]
    duration: u64,
}

struct Collector {
    tx: Sender<Event>,
    rx: Receiver<Event>,
    schema: SchemaRef,
    writer: Mutex<Option<ArrowWriter<File>>>,
    event_count: Mutex<u64>,
    start_time: std::time::Instant,
}

impl Collector {
    fn new(output: PathBuf, schema: SchemaRef) -> Result<Self> {
        let (tx, rx) = bounded(100000);
        let file = File::create(&output)?;
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD)
            .set_max_row_group_size(EVENT_BATCH_SIZE as i64)
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        Ok(Self {
            tx,
            rx,
            schema,
            writer: Mutex::new(Some(writer)),
            event_count: Mutex::new(0),
            start_time: std::time::Instant::now(),
        })
    }

    fn sender(&self) -> Sender<Event> {
        self.tx.clone()
    }

    fn process_batch(&self, events: &[Event]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }

        let n = events.len();
        let timestamps: Vec<u64> = events.iter().map(|e| e.timestamp_ns).collect();
        let cpus: Vec<u32> = events.iter().map(|e| e.cpu).collect();
        let pids: Vec<u32> = events.iter().map(|e| e.pid).collect();
        let tids: Vec<u32> = events.iter().map(|e| e.tid).collect();
        let kinds: Vec<u16> = events.iter().map(|e| e.kind).collect();
        let arg0s: Vec<u64> = events.iter().map(|e| e.arg0).collect();
        let arg1s: Vec<u64> = events.iter().map(|e| e.arg1).collect();
        let arg2s: Vec<u64> = events.iter().map(|e| e.arg2).collect();

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(UInt64Array::from(timestamps)),
                Arc::new(UInt32Array::from(cpus)),
                Arc::new(UInt32Array::from(pids)),
                Arc::new(UInt32Array::from(tids)),
                Arc::new(UInt16Array::from(kinds)),
                Arc::new(UInt64Array::from(arg0s)),
                Arc::new(UInt64Array::from(arg1s)),
                Arc::new(UInt64Array::from(arg2s)),
            ],
        )?;

        let mut writer = self.writer.lock();
        if let Some(w) = writer.as_mut() {
            w.write(&batch)?;
            *self.event_count.lock() += n as u64;
        }

        Ok(())
    }

    fn run_writer(&self) -> Result<()> {
        let mut batch = Vec::with_capacity(EVENT_BATCH_SIZE);
        let flush_interval = Duration::from_millis(FLUSH_INTERVAL_MS);

        loop {
            match self.rx.recv_timeout(flush_interval) {
                Ok(event) => {
                    batch.push(event);
                    if batch.len() >= EVENT_BATCH_SIZE {
                        self.process_batch(&batch)?;
                        batch.clear();
                    }
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    if !batch.is_empty() {
                        self.process_batch(&batch)?;
                        batch.clear();
                    }
                }
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                    if !batch.is_empty() {
                        self.process_batch(&batch)?;
                    }
                    break;
                }
            }
        }

        // Finalize writer
        let mut writer = self.writer.lock();
        if let Some(w) = writer.take() {
            w.close()?;
        }

        let count = *self.event_count.lock();
        let elapsed = self.start_time.elapsed().as_secs_f64();
        info!("Collection complete: {} events in {:.2}s ({:.0} events/s)", count, elapsed, count as f64 / elapsed);

        Ok(())
    }
}

fn build_schema() -> SchemaRef {
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let schema = build_schema();
    let collector = Collector::new(args.output, schema)?;
    let sender = collector.sender();

    // Spawn writer task
    let writer_handle = tokio::task::spawn_blocking(move || {
        collector.run_writer()
    });

    // Spawn eBPF collector
    let ebpf_handle = tokio::task::spawn_blocking(move || {
        run_ebpf_collection(sender, &args)
    });

    // Duration timeout
    if args.duration > 0 {
        tokio::time::sleep(Duration::from_secs(args.duration)).await;
        info!("Duration elapsed, shutting down...");
    } else {
        signal::ctrl_c().await?;
        info!("Ctrl-C received, shutting down...");
    }

    // Drop sender to signal writer to finish
    drop(sender);

    // Wait for completion
    writer_handle.await??;
    ebpf_handle.await??;

    Ok(())
}

fn run_ebpf_collection(sender: Sender<Event>, args: &Args) -> Result<()> {
    // This is a scaffold - real implementation uses aya to load the eBPF program
    // and read from the ring buffer, converting kernel events to Event structs

    use aya::programs::{TracePoint, KProbe, KRetProbe};
    use aya::{include_bytes_aligned, Bpf};
    use aya_log::BpfLogger;

    let mut bpf = Bpf::load(include_bytes_aligned!(
        "../../target/bpfel-unknown-none/release/ygg-probes"
    ))?;

    if let Err(e) = BpfLogger::init(&mut bpf) {
        warn!("Failed to initialize eBPF logger: {}", e);
    }

    // Attach tracepoints
    if args.sched {
        let prog: &mut TracePoint = bpf.program_mut("sched_switch").unwrap().try_into()?;
        prog.load()?.attach("sched", "sched_switch")?;

        let prog: &mut TracePoint = bpf.program_mut("sched_wakeup").unwrap().try_into()?;
        prog.load()?.attach("sched", "sched_wakeup")?;

        let prog: &mut TracePoint = bpf.program_mut("sched_migrate_task").unwrap().try_into()?;
        prog.load()?.attach("sched", "sched_migrate_task")?;
    }

    if args.syscalls {
        let prog: &mut TracePoint = bpf.program_mut("sys_enter").unwrap().try_into()?;
        prog.load()?.attach("syscalls", "sys_enter")?;

        let prog: &mut TracePoint = bpf.program_mut("sys_exit").unwrap().try_into()?;
        prog.load()?.attach("syscalls", "sys_exit")?;
    }

    if args.block_io {
        let prog: &mut TracePoint = bpf.program_mut("block_rq_issue").unwrap().try_into()?;
        prog.load()?.attach("block", "block_rq_issue")?;

        let prog: &mut TracePoint = bpf.program_mut("block_rq_complete").unwrap().try_into()?;
        prog.load()?.attach("block", "block_rq_complete")?;
    }

    if args.network {
        let prog: &mut KProbe = bpf.program_mut("tcp_sendmsg").unwrap().try_into()?;
        prog.load()?.attach("tcp_sendmsg", 0)?;

        let prog: &mut KRetProbe = bpf.program_mut("tcp_recvmsg").unwrap().try_into()?;
        prog.load()?.attach("tcp_recvmsg", 0)?;
    }

    if args.page_faults {
        let prog: &mut TracePoint = bpf.program_mut("page_fault_user").unwrap().try_into()?;
        prog.load()?.attach("exceptions", "page_fault_user")?;

        let prog: &mut TracePoint = bpf.program_mut("page_fault_kernel").unwrap().try_into()?;
        prog.load()?.attach("exceptions", "page_fault_kernel")?;
    }

    // Ring buffer polling loop
    let mut ring_buffers = Vec::new();
    // ... attach ring buffer consumers ...

    info!("eBPF collection started");
    loop {
        // Poll ring buffers and send events via sender
        std::thread::sleep(Duration::from_millis(10));
    }
}