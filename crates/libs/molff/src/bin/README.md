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
- **`rarff2d_demo.rs`** — RARFF-2D repair demo: corrupted naphthalene (2 fused hexagons + ring entities), displaced atom ± spurious atom in a ring hole; overdamped GD relax + FD checks + CSV/XYZ dumps. Run: `cargo run --release -p molff --bin rarff2d_demo` → `debug/rarff2d_dyn/`
- **`rarff2d_assemble.rs`** — RARFF-2D self-assembly demo: N random sp2 atoms in soft arena → damped MD → bonded graph with emergent hexagons; `--seed/--n/--arena/--out`; dumps traj.xyz + detected bonds. Run: `cargo run --release -p molff --bin rarff2d_assemble -- --seed 7` → `debug/rarff2d_assemble*/`
- **`raff_reactive_demo.rs`** — 3D reactive-RAFF movie generator (bonds emerge from geometry, no bond list): `--scene methane|cloud|pair` (tetrahedral C+4H self-assembly / 12C+12H box self-assembly / 2-atom orientation alignment), `--steps/--stride/--dt/--cdamp/--grid/--out`; writes `traj.xyz` + `energy.tsv` + emerged-bond list. Run: `cargo run --release -p molff --bin raff_reactive_demo -- --scene methane --steps 1500` → `debug/raff_reactive_demo/<scene>/`
- **`rarff2d_bench.rs`** — broad-phase benchmark: O(N²) vs `Grid2d` (dx=rcut/3/1) vs COG `Groups2d` on a hexagonal lattice (a=1.3, 70% fill, jitter — non-commensurate w/ accel grid); packed-baseline efficiency (`ninside·t_packed/measured`), `dbg_npair`/`dbg_ninside` call counts, per-pass timings, parity asserts; `--n/--reps`; dumps `debug/rarff2d_bench/{scenario,groups}.csv` for `scripts/plot_rarff2d_bench.py`. Run: `cargo run --release -p molff --bin rarff2d_bench -- --n 800 --reps 30`

## See also

- [`../README.md`](../README.md) — molff crate overview (raff.rs module)
- [`/userguide/raff.md`](/userguide/raff.md) — RAFF solver modes end-user guide (performance comparison table)
- [`/debug/raff_bench/README.md`](/debug/raff_bench/README.md) — benchmark output plots
