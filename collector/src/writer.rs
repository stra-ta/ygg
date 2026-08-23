//! Arrow / Parquet batch writer.
//!
//! Events flow in over a crossbeam channel. The writer thread batches them into
//! Arrow [`RecordBatch`]es of `EVENT_BATCH_SIZE` rows and flushes to a Parquet
//! file with ZSTD compression. The data path is fully synchronous; the caller
//! is expected to run `run` inside `tokio::task::spawn_blocking`.

use ygg_schema::Event;
use anyhow::Result;
use arrow::array::{UInt16Array, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use crossbeam::channel::{Receiver, RecvTimeoutError};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;

/// Number of events per Arrow `RecordBatch` / Parquet row group.
pub const EVENT_BATCH_SIZE: usize = 8192;
/// Idle flush interval when fewer than `EVENT_BATCH_SIZE` events are pending.
pub const FLUSH_INTERVAL_MS: u64 = 100;

/// Build the Arrow schema for the events stream.
pub fn build_schema() -> SchemaRef {
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

/// Encodes a slice of [`Event`]s into an Arrow [`RecordBatch`].
fn record_batch(schema: SchemaRef, events: &[Event]) -> Result<RecordBatch> {
    let n = events.len();
    let mut timestamps = Vec::with_capacity(n);
    let mut cpus = Vec::with_capacity(n);
    let mut pids = Vec::with_capacity(n);
    let mut tids = Vec::with_capacity(n);
    let mut kinds = Vec::with_capacity(n);
    let mut arg0s = Vec::with_capacity(n);
    let mut arg1s = Vec::with_capacity(n);
    let mut arg2s = Vec::with_capacity(n);
    for e in events {
        timestamps.push(e.timestamp_ns);
        cpus.push(e.cpu);
        pids.push(e.pid);
        tids.push(e.tid);
        kinds.push(e.kind);
        arg0s.push(e.arg0);
        arg1s.push(e.arg1);
        arg2s.push(e.arg2);
    }

    let batch = RecordBatch::try_new(
        schema,
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
    Ok(batch)
}

/// Owns the Parquet file and drains the event channel.
pub struct ParquetWriter {
    schema: SchemaRef,
    writer: ArrowWriter<File>,
    batch_size: usize,
    event_count: u64,
    start: Instant,
}

impl ParquetWriter {
    pub fn new(path: PathBuf, batch_size: usize) -> Result<Self> {
        let file = File::create(&path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .set_max_row_group_size(batch_size)
            .build();
        let writer = ArrowWriter::try_new(file, build_schema(), Some(props))?;
        Ok(Self {
            schema: build_schema(),
            writer,
            batch_size,
            event_count: 0,
            start: Instant::now(),
        })
    }

    /// Append a batch of events to the Parquet file.
    fn write_batch(&mut self, events: &[Event]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        let batch = record_batch(self.schema.clone(), events)?;
        self.writer.write(&batch)?;
        self.event_count += events.len() as u64;
        Ok(())
    }

    /// Consume the channel until the sender disconnects or `shutdown` is set.
    /// Returns the total number of events written.
    pub fn run(mut self, rx: Receiver<Event>, shutdown: &AtomicBool) -> Result<u64> {
        let mut batch = Vec::with_capacity(self.batch_size);
        let flush = Duration::from_millis(FLUSH_INTERVAL_MS);

        loop {
            match rx.recv_timeout(flush) {
                Ok(event) => {
                    batch.push(event);
                    if batch.len() >= self.batch_size {
                        self.write_batch(&batch)?;
                        batch.clear();
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if !batch.is_empty() {
                        self.write_batch(&batch)?;
                        batch.clear();
                    }
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    if !batch.is_empty() {
                        self.write_batch(&batch)?;
                    }
                    break;
                }
            }
        }

        self.writer.close()?;

        let count = self.event_count;
        let elapsed = self.start.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            count as f64 / elapsed
        } else {
            0.0
        };
        info!(
            "Collection complete: {} events in {:.2}s ({:.0} events/s)",
            count, elapsed, rate
        );
        Ok(count)
    }
}
