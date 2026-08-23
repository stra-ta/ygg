use std::process::Command;
use std::path::PathBuf;

fn main() {
    // Compile eBPF probes
    let probe_src = PathBuf::from("collector/ebpf/probes.bpf.c");
    let probe_out = PathBuf::from("target/bpfel-unknown-none/release/ygg-probes");

    // Ensure output directory exists
    std::fs::create_dir_all(probe_out.parent().unwrap()).unwrap();

    // Check if we're in a Cargo build context
    let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| "target".to_string());
    let target_dir = PathBuf::from(&out_dir).join("bpfel-unknown-none/release");

    // Compile with clang
    let status = Command::new("clang")
        .args([
            "-target", "bpfel-unknown-none",
            "-O2",
            "-g",
            "-I.",  // For vmlinux.h
            "-c", probe_src.to_str().unwrap(),
            "-o", target_dir.join("ygg-probes.o").to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run clang");

    if !status.success() {
        panic!("eBPF probe compilation failed");
    }

    // Also copy to collector target dir for runtime loading
    std::fs::create_dir_all(&target_dir).unwrap();
    std::fs::copy(
        target_dir.join("ygg-probes.o"),
        PathBuf::from("collector/target/bpfel-unknown-none/release/ygg-probes"),
    ).ok();

    println!("cargo:rerun-if-changed=collector/ebpf/probes.bpf.c");
    println!("cargo:rustc-link-search=native={}", target_dir.display());
}