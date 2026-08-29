---
type: folder
title: molff/src/bin
description: Benchmark binary for the molff crate — parameter sweep of all RAFF solver modes.
tags: [rust, binary, benchmark, raff, solver, projective-dynamics, xpbd, fire]
timestamp: 2026-08-29
---

# molff/src/bin — Benchmark binary

Standalone CLI binaries for the `molff` crate. Run with `cargo run -p molff --bin <name>`.

## Binaries

- **`raff_bench.rs`** — RAFF solver benchmark: parameter sweep of `{dt, iters, over_relax}` × `{PBD, XPBD, Projective, ForceMD}` on CH4/water/tree-20/tree-100. Reports `n_steps`, `n_port_evals`, `t_wall_us` (single-thread). Run: `cargo run --release -p molff --bin raff_bench`. See [`/notes/reports/2026-08-28_raff_solver_benchmark_report.md`](/notes/reports/2026-08-28_raff_solver_benchmark_report.md)

## See also

- [`../README.md`](../README.md) — molff crate overview (raff.rs module)
- [`/userguide/raff.md`](/userguide/raff.md) — RAFF solver modes end-user guide (performance comparison table)
- [`/debug/raff_bench/README.md`](/debug/raff_bench/README.md) — benchmark output plots
