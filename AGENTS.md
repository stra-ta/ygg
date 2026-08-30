# Ygg engineering guide

## Purpose

Ygg turns versioned application and Linux kernel traces into execution-level diagnostics.
It keeps capture, schema, representation learning, change-point analysis, and stra-ta integrations separate.

## Invariants

- Event records carry a schema version and incompatible major versions fail closed.
- Dropped or lost event counts are data and remain visible to training and evaluation.
- Train, validation, and test splits occur at execution-group level, never by adjacent windows.
- Seed, run, and group identities cannot cross split boundaries.
- Linear probes are diagnostics, not training supervision.
- Runtime length, event count, and other trivial features are measured as collapse probes.
- Python ML dependencies never enter low-level C++ or Rust target builds.
- A larger model requires evidence that the system-level signal exists.

## Verification

Use `./scripts/verify` for the Rust functional suite.
Use `./scripts/confidence` for formatting, Clippy, tests, and a release build.
Linux eBPF campaigns require a compatible kernel and privileges and are reported separately from functional CI.

## Lab-wide contracts

- See https://github.com/stra-ta/.github/blob/main/LAB_RULES.md and https://github.com/stra-ta/.github/blob/main/EVIDENCE.md and https://github.com/stra-ta/.github/blob/main/COMPATIBILITY.md for lab-wide naming, evidence, and schema contracts.
- Per https://github.com/stra-ta/.github/blob/main/CONTRIBUTING.md, contributions require the target repo's AGENTS.md, README, and relevant design note, preserve repo boundaries, add the narrowest regression test, run one-command verification, and keep performance claims tied to committed manifests.
