---
type: folder
title: molff/tests
description: Integration tests for the molff crate — RAFF physical invariants, solver convergence, broad-phase parity, multigrid parity, benzene diagnostic, rigid_sp3 baseline.
tags: [rust, test, raff, convergence, parity, broad-phase, multigrid, benzene, rigid-sp3]
timestamp: 2026-08-29
---

# molff/tests — Integration tests

Integration tests for the `molff` crate. Run with `cargo test -p molff`.

## Test files

- **`test_raff.rs`** (607 LOC) — 22 tests: port force parity, Wahba rotation convergence, energy conservation, momentum conservation, XPBD constraint satisfaction, collision resolution, adiabatic torque residual. All passing
- **`test_raff_convergence.rs`** (216 LOC) — 9 tests: ForceMD + all 3 position-based solvers converge to same geometry (Kabsch RMSD < 1e-3). Kabsch invariants, chain4 dihedral null space, PD outer-inertia retention. All passing
- **`test_broad_phase.rs`** (177 LOC) — 3 tests: `eval_broad` vs `eval` (NonBondedFF), far molecules (0 BP pairs), `eval_nonbonded_broad` vs `eval_nonbonded` (RAFF). All passing
- **`test_multigrid.rs`** (~200 LOC) — 7 tests: matvec parity (TrussOp vs dense), diagonal-block parity, direct-solve parity (MG vs Gaussian), 3.9× fewer smoothing steps on 8×8 grid (144 vs 561), cached coarse force parity, fitted modal quadratic parity, bend/twist orthonormality. All passing
- **`test_multigrid_molecules.rs`** (~165 LOC) — 4 molecule benchmarks: pentacene/hexadecane/DiTriptyceno with bond-only `TrussOp`, pentacene with full UFF `UffHessianOp`. All passing
- **`test_benzene_diag.rs`** (178 LOC) — regression test: per-atom ARAP port geometry gives E_port=0 and stable benzene. Documents the bug where idealized sp2 ports caused geometrically inconsistent port-to-neighbor assignment
- **`test_rigid_sp3.rs`** (110 LOC) — tetrahedral sp3 center (CH4-like) + water test for the legacy `RigidSp3FF`

## Running

```bash
cargo test -p molff                          # all tests
cargo test -p molff --test test_raff         # RAFF physical invariants
cargo test -p molff --test test_raff_convergence  # solver convergence
cargo test -p molff --test test_broad_phase  # broad-phase parity
cargo test -p molff --test test_multigrid    # multigrid parity
```

## See also

- [`../README.md`](../README.md) — molff crate overview
- [`../src/README.md`](../src/README.md) — source modules
- [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) — RAFF cross-implementation map
- [`/doc/topical_audit/multigrid.md`](/doc/topical_audit/multigrid.md) — multigrid cross-implementation map
- [`/doc/topical_audit/spatial_acceleration.md`](/doc/topical_audit/spatial_acceleration.md) — broad-phase collision map
