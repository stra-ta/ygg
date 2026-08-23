// Weir integration: Durability traces

pub struct WeirEvent {
    pub op: String,           // "recv", "parse", "admit", "wal_append", "sync", "commit", "ack"
    pub connection_id: u64,
    pub bytes: u64,
    pub lsn: u64,
    pub queue_depth: u32,
    pub timestamp_ns: u64,
}

pub fn instrument_weir(engine: &mut WeirEngine) {
    // YGG_EVENT(WeirRecv, bytes);
    // YGG_EVENT(WeirParse, frame_type);
    // YGG_EVENT(WeirAdmit, queue_depth);
    // YGG_EVENT(WeirWalAppend, bytes);
    // YGG_EVENT(WeirSync, 0);
    // YGG_EVENT(WeirCommit, lsn);
    // YGG_EVENT(WeirAck, connection_id);
    todo!()
}

pub fn run_fault_campaign(config: FaultCampaignConfig) -> anyhow::Result<CampaignResults> {
    // Run Kiln campaign with Loki faults on Weir
    // Healthy, fsync +250µs, +1ms, +5ms, CPU starvation
    todo!()
}

pub struct FaultCampaignConfig {
    pub fault_types: Vec<FaultType>,
    pub magnitudes: Vec<u64>,  // e.g., microseconds of delay
    pub durations_sec: u64,
}

pub enum FaultType {
    FsyncDelay,
    CpuStarvation,
    NetworkDelay,
    DiskError,
}