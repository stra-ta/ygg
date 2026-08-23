/*
 * build.rs - compile the C instrumentation sources into the Rust crate.
 *
 * The ultra-low-overhead hot path (ring buffer, TSC calibration, collector
 * thread) lives in C and is linked into both the staticlib and cdylib
 * (`libygg_instrumentation.a` / `libygg_instrumentation.so`) produced by Cargo.
 */
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    let mut build = cc::Build::new();
    build
        .include(manifest.join("include"))
        .include(manifest.join("src"))
        .file(manifest.join("src/ygg.c"))
        .file(manifest.join("src/ring_buffer.c"))
        .file(manifest.join("src/collector_thread.c"))
        .define("YGG_BUILD", None)
        .warnings(true)
        .extra_warnings(true)
        .flag_if_supported("-pthread")
        .flag_if_supported("-std=c11");

    build.compile("ygg");

    println!("cargo:rerun-if-changed=src/ygg.c");
    println!("cargo:rerun-if-changed=src/ring_buffer.c");
    println!("cargo:rerun-if-changed=src/collector_thread.c");
    println!("cargo:rerun-if-changed=src/ygg_internal.h");
    println!("cargo:rerun-if-changed=include/ygg/ygg.h");
}
