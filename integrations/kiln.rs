// Kiln integration: Ygg trace collection

use crate::experiments::ExperimentConfig;

pub struct YggConfig {
    pub enabled: bool,
    pub events: Vec<String>,
    pub perf: Vec<String>,
    pub output_path: String,
}

impl Default for YggConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            events: vec!["scheduler".into(), "block_io".into(), "syscalls".into(), "application".into()],
            perf: vec!["cycles".into(), "instructions".into(), "cache-misses".into()],
            output_path: "ygg.trace.parquet".into(),
        }
    }
}

pub fn setup_ygg_collection(config: &YggConfig, experiment: &ExperimentConfig) -> anyhow::Result<()> {
    // 1. Generate ygg-collector command line
    // 2. Start collector as child process
    // 3. Set up shared memory for calibration
    // 4. Return handle for cleanup
    todo!()
}

pub fn collect_ygg_artifact(run_dir: &Path) -> anyhow::Result<PathBuf> {
    // Move trace.parquet to run_dir/ygg.trace.parquet
    todo!()
}

pub fn compare_with_ygg(run_a: &RunArtifacts, run_b: &RunArtifacts) -> anyhow::Result<YggComparison> {
    // Load both traces
    // Compute embeddings
    // Return distance + divergence analysis
    todo!()
}

pub struct YggComparison {
    pub embedding_distance: f64,
    pub divergence_point_ns: Option<u64>,
    pub behavioral_family: Option<String>,
    pub new_regime: bool,
}