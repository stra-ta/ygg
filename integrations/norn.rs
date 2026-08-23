// Norn integration: Contention regime traces

pub struct NornEvent {
    pub op: String,           // "push", "pop", "cas_retry", "yield", "spin"
    pub queue_id: u32,
    pub thread_id: u32,
    pub cpu: u32,
    pub timestamp_ns: u64,
    pub queue_depth: u32,
    pub cas_retries: u32,
}

pub fn instrument_norn_queue(queue: &mut NornQueue) {
    // Add YGG_EVENT calls to Norn queue operations
    // YGG_EVENT(NornPush, queue_depth);
    // YGG_EVENT(NornPop, queue_depth);
    // YGG_EVENT(NornCasRetry, retries);
    // YGG_EVENT(NornYield, 0);
    // YGG_EVENT(NornSpin, iterations);
    todo!()
}

pub fn run_contention_campaign(config: ContentionConfig) -> anyhow::Result<CampaignResults> {
    // Run Kiln campaign over Norn contention parameters
    // 1x1, 2x2, 4x4, 8x8 x {tight, yield, bounded, exponential}
    todo!()
}

pub struct ContentionConfig {
    pub thread_counts: Vec<usize>,
    pub backoff_policies: Vec<BackoffPolicy>,
    pub durations_sec: u64,
}

pub enum BackoffPolicy {
    Tight,
    Yield,
    Bounded { max: u32 },
    Exponential { base: u32, max: u32 },
}

pub struct CampaignResults {
    pub trace_paths: Vec<PathBuf>,
    pub configs: Vec<ContentionConfig>,
}