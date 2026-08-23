/*
 * run.rs - benchmark runner for the Ygg instrumentation hot path.
 *
 * What it does:
 *   1. Builds the existing `ygg-instrumentation` static library (so we link the
 *      shipped library, never modify it).
 *   2. Compiles bench.c against that static library using the system C compiler.
 *   3. Runs each benchmark scenario, capturing its JSON lines on stdout.
 *   4. Prints a human-readable summary table to stderr.
 *   5. Writes the full results array to results.json.
 *
 * Run it with:  cargo run --release
 */

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let bench_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let instr_dir = bench_dir.join(".."); // instrumentation/ (the library crate)
    let workspace_root = bench_dir.join("..").join(".."); // ygg/ (workspace root, holds target/)

    // ---- 1. Build the instrumentation static library -----------------------
    eprintln!("building ygg-instrumentation staticlib ...");
    let mut build = Command::new("cargo");
    build
        .args(["build", "--release", "--manifest-path"])
        .arg(instr_dir.join("Cargo.toml"));
    let status = build.status().expect("failed to spawn cargo");
    if !status.success() {
        eprintln!("error: cargo build of ygg-instrumentation failed");
        std::process::exit(1);
    }

    // ---- 2. Locate the static library --------------------------------------
    let lib = locate_lib(&instr_dir, &workspace_root)
        .unwrap_or_else(|| {
            eprintln!(
                "error: could not find libygg_instrumentation.a (looked under {} and {})",
                instr_dir.join("target/release").display(),
                workspace_root.join("target/release").display()
            );
            std::process::exit(1);
        });
    eprintln!("using static lib: {}", lib.display());

    // ---- 3. Compile bench.c -------------------------------------------------
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let bench_c = instr_dir.join("bench").join("bench.c");
    let bin = std::env::temp_dir().join("ygg_bench_bin");

    let mut cc = Command::new(&compiler);
    cc.arg(&bench_c)
        .arg("-O2")
        .arg("-I")
        .arg(instr_dir.join("include"))
        .arg("-I")
        .arg(instr_dir.join("src"));

    match env::consts::OS {
        "macos" => {
            // Rust staticlibs must be force-loaded on Apple platforms.
            cc.arg(format!("-Wl,-force_load,{}", lib.display()));
        }
        "linux" => {
            cc.arg("-Wl,--whole-archive")
                .arg("-L")
                .arg(lib.parent().unwrap())
                .arg("-lygg_instrumentation")
                .arg("-Wl,--no-whole-archive");
        }
        other => {
            eprintln!(
                "warning: unknown OS '{}'; linking may need adjustments (using -L/-l)",
                other
            );
            cc.arg("-L").arg(lib.parent().unwrap()).arg("-lygg_instrumentation");
        }
    }
    cc.arg("-lpthread");
    if env::consts::OS != "macos" {
        cc.arg("-ldl"); // Rust std on Linux/glibc
    }
    cc.arg("-o").arg(&bin);

    let out = cc.output().expect("failed to spawn C compiler");
    if !out.status.success() {
        eprintln!("error: failed to compile bench.c");
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        std::process::exit(1);
    }

    // ---- 4. Run each scenario, collect JSON lines --------------------------
    let scenarios: &[(&str, Option<&str>)] = &[
        ("baseline", None),
        ("no_collector", None),
        ("draining", None),
        ("at_capacity", None),
        ("threads", None),
    ];

    let mut results: Vec<serde_json::Value> = Vec::new();
    for (name, arg) in scenarios {
        eprintln!("running scenario: {}", name);
        let mut run = Command::new(&bin);
        run.arg(name);
        if let Some(a) = arg {
            run.arg(a);
        }
        let out = run.output().expect("failed to run benchmark");
        if !out.status.success() {
            eprintln!("warning: scenario '{}' exited non-zero", name);
            eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        }
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(line) {
                Ok(v) => results.push(v),
                Err(e) => eprintln!("warning: could not parse JSON line '{}': {}", line, e),
            }
        }
    }

    // ---- 5. Summary table to stderr ----------------------------------------
    print_summary(&results);

    // ---- 6. Write full results ---------------------------------------------
    let out_path = bench_dir.join("results.json");
    let pretty = serde_json::to_string_pretty(&results).expect("serialize");
    std::fs::write(&out_path, pretty).expect("write results.json");
    eprintln!("wrote {}", out_path.display());
}

fn locate_lib(instr_dir: &Path, workspace_root: &Path) -> Option<PathBuf> {
    let candidates = [
        instr_dir.join("target/release/libygg_instrumentation.a"),
        workspace_root.join("target/release/libygg_instrumentation.a"),
        instr_dir
            .join("..")
            .join("target/release/libygg_instrumentation.a"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn f64_of(v: &serde_json::Value, key: &str) -> String {
    match v.get(key).and_then(|x| x.as_f64()) {
        Some(x) => format!("{:.2}", x),
        None => "-".to_string(),
    }
}

fn print_summary(results: &[serde_json::Value]) {
    eprintln!();
    eprintln!(
        "{:<22} {:>10} {:>7} {:>9} {:>9} {:>9} {:>12} {:>5} {:>10}",
        "scenario", "events", "thr", "median", "p95", "p99", "dropped", "unit", "coll?"
    );
    eprintln!("{}", "-".repeat(96));
    for r in results {
        let scenario = r.get("scenario").and_then(|x| x.as_str()).unwrap_or("?");
        let events = r.get("events").and_then(|x| x.as_u64()).unwrap_or(0);
        let threads = r.get("threads").and_then(|x| x.as_u64()).unwrap_or(1);
        let median = f64_of(r, "median_cycles_per_event");
        let p95 = f64_of(r, "p95_cycles_per_event");
        let p99 = f64_of(r, "p99_cycles_per_event");
        let dropped = r.get("dropped_events").and_then(|x| x.as_u64()).unwrap_or(0);
        let unit = r.get("unit").and_then(|x| x.as_str()).unwrap_or("?");
        let coll = if r.get("collector_active").and_then(|x| x.as_bool()).unwrap_or(false) {
            "yes"
        } else {
            "no"
        };
        eprintln!(
            "{:<22} {:>10} {:>7} {:>9} {:>9} {:>9} {:>12} {:>5} {:>10}",
            scenario, events, threads, median, p95, p99, dropped, unit, coll
        );
    }
    eprintln!();
    if let Some(plat) = results.first().and_then(|r| r.get("platform")).and_then(|x| x.as_str()) {
        eprintln!("platform: {}", plat);
        if plat != "x86_64" {
            eprintln!(
                "note: the '5-10 cycles' claim is for x86 rdtsc. On {} the unit is '{}' \
                 (nanoseconds via mach_absolute_time / CLOCK_MONOTONIC), not CPU cycles.",
                plat,
                results
                    .first()
                    .and_then(|r| r.get("unit"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("?")
            );
        }
    }
}
