---
type: topical-audit
title: Multigrid Solver
description: Cross-implementation map for coarse-grained molecular relaxation in SurfMol — linear V-cycle solver and modal methods. Contract-separated pentacene benchmarks give 57× for pure in-manifold relaxation, 53.8× for staged canonical decoding, and 1.66× for additive preservation of a mixed atomistic state. Force-projection Galerkin V-shape remains the nonlinear fallback.
tags: [multigrid, solver, truss-elasticity, uff-hessian, prolongation, coarse-space, jacobi, v-cycle, modal, bend-twist, topical-audit]
timestamp: 2026-09-29
---

# Multigrid Solver — Cross-Implementation Map

## Topic

Multigrid V-cycle solver for linearized molecular elasticity `A·δx = F`, where `A = K + diag(M/Δt²)`. Two operators via the `LinearOp` trait:
- **`TrussOp`**: matrix-free bond-only axial-spring stiffness (`K = Σ 2k·n⊗n`). Fast, scalable, but **zero out-of-plane bending stiffness at planar equilibrium** — axial springs only resist along the bond direction.
- **`UffHessianOp`**: full UFF Hessian (bonds + angles + dihedrals + inversions) via central finite differences. Captures aromatic bending/torsion stiffness from inversion (improper torsion) and dihedral terms — **1000× more out-of-plane stiffness than bond-only** on pentacene (7.67 vs 0.007).

V-cycle: pre-smooth (block Jacobi with optional heavy-ball momentum) → restrict residual to coarse space → coarse solve (dense Cholesky) → prolongate correction → post-smooth.

## Where it lives

### SurfMol (Rust CPU — authoritative)

| Component | File | Function/Struct |
|-----------|------|-----------------|
| `LinearOp` trait (natoms, matvec, diagonal_blocks, assemble_dense) | `crates/libs/molff/src/multigrid.rs` | `LinearOp` |
| Truss operator (bond-only, matrix-free) | `crates/libs/molff/src/multigrid.rs` | `TrussOp` impl `LinearOp` |
| Full UFF Hessian operator (finite-difference, dense) | `crates/libs/molff/src/multigrid.rs` | `UffHessianOp` impl `LinearOp` |
| Block Jacobi smoother (with heavy-ball momentum) | `crates/libs/molff/src/multigrid.rs` | `jacobi_smooth`, `jacobi_smooth_momentum`, `invert_3x3_blocks` |
| Geometric prolongation (pivot BFS + inverse-distance) | `crates/libs/molff/src/multigrid.rs` | `select_pivots_maximin`, `build_pivot_prolongation` |
| Galerkin coarse operator + cached level | `crates/libs/molff/src/multigrid.rs` | `galerkin_coarse`, `GalerkinLevel` |
| V-cycle (two-grid + outer loop) | `crates/libs/molff/src/multigrid.rs` | `solve_two_grid`, `solve_multigrid`, `solve_coarse_first`, `coarse_correct` |
| Fitted coupled modal quadratic model | `crates/libs/molff/src/multigrid.rs` | `ModalQuadratic::fit_central`, `solve_force`, `project_force` |
| Bend/twist orthonormal mode generator | `crates/libs/molff/src/multigrid.rs` | `build_bend_twist_modes` |
| Dense direct solve (test reference) | `crates/libs/molff/src/multigrid.rs` | `dense_solve` |
| f64 Cholesky factor + solve (coarse solve) | `crates/libs/numcore/src/math/linalg.rs` | `cholesky_factor_f64`, `cholesky_solve_f64`, `dense_matvec_f64` |
| Parity + convergence tests | `crates/libs/molff/tests/test_multigrid.rs` | T1–T7 (7 tests) |
| Molecule benchmarks (bond-only + full UFF) | `crates/libs/molff/tests/test_multigrid_molecules.rs` | pentacene (bond + full UFF), hexadecane, DiTriptyceno (4 tests) |

### SurfMol (OpenCL — copied, not yet wired)

| Component | File | Origin |
|-----------|------|--------|
| Prolongation (tiled P) | `opencl/multigrid.cl` | `NumericalMathPlayground/.../kernels_multigrid.cl` |
| Restriction (R=Pᵀ tree-reduce) | `opencl/multigrid.cl` | same |
| Coarse Cholesky solve | `opencl/multigrid.cl` | same |
| Block Jacobi smoother (local-memory patches) | `opencl/block_jacobi.cl` | `NumericalMathPlayground/.../kernels_block_jacobi.cl` |
| Residual computation | `opencl/block_jacobi.cl` | same |
| Diagonal Dinv computation | `opencl/block_jacobi.cl` | same |

### Reference repo: NumericalMathPlayground

| Component | File |
|-----------|------|
| Python MG core (spectral + geometric prolongation, Galerkin, V-cycle) | `topics/LinarElasticity/MultiGrid.py` |
| Truss solver (matvec, diagonal, Jacobi, direct solve) | `topics/LinarElasticity/TrussSolver.py` |
| Demo + benchmark | `topics/LinarElasticity/demo_MultiGrid.py` |
| OpenCL MG kernels | `topics/LinarElasticity/kernels_multigrid.cl` |
| OpenCL block Jacobi kernels | `topics/LinarElasticity/kernels_block_jacobi.cl` |

## Parity status

| Component | Status |
|-----------|--------|
| TrussOp matvec | ✅ Parity verified (T1: max err 1.5e-8 vs dense) |
| Diagonal blocks | ✅ Parity verified (T2: exact) |
| Direct solve | ✅ Parity verified (T3: max err 2.9e-13 vs Gaussian elimination) |
| V-cycle convergence (regular grid) | ✅ 3.9× fewer smoothing steps than Jacobi on 8×8 grid (T4: 144 vs 561) |
| Cached coarse force parity | ✅ GalerkinLevel vs exact coarse correction (T5) |
| Fitted modal quadratic parity | ✅ K fitted to 4.4e-16, response to 2.2e-16 (T6) |
| Bend/twist orthonormality | ✅ Gram matrix = I to 1e-14 (T7) |
| Full UFF Hessian (UffHessianOp) | ✅ 1000× more out-of-plane stiffness than bond-only on pentacene (7.67 vs 0.007) |
| V-cycle convergence (full UFF molecules) | ✅ MG converges in 2 V-cycles on pentacene with full UFF Hessian |
| Spectral prolongation | ❌ Not implemented (needs eigensolver) |
| OpenCL wiring | ❌ Kernels copied, not loaded by any Rust crate |
| Newton-step wrapper (`relax_uff_multigrid`) | ❌ Not implemented |
| RAFF integration | ❌ Not started |

## Operators

| Operator | Stiffness terms | Cost | Use case |
|----------|----------------|------|----------|
| `TrussOp` | Bonds only (axial `k·n⊗n`) | O(n_bonds) per matvec | Bond-only relaxation, large molecules, linear diagnostic |
| `UffHessianOp` | Bonds + angles + dihedrals + inversions | O(n²) per matvec, 2·n_dof force evals to build | Aromatic molecules (pentacene), bending/torsion, physical accuracy |

**Key insight**: at a planar equilibrium, axial bonds have **zero transverse stiffness** (the bond direction is in-plane). Without inversion/dihedral terms, the operator cannot resist out-of-plane bending — the mass term is the only resistance. `UffHessianOp` captures the full UFF stiffness and is required for physically meaningful multigrid on aromatic molecules.

## Prolongation / coarse-space strategies

| Strategy | Status | Where | Notes |
|----------|--------|-------|-------|
| Geometric pivots (BFS farthest-point + inverse-distance) | ✅ Implemented | `build_pivot_prolongation` | Retained as diagnostic; not physically motivated for molecules |
| Spectral (vibration eigenmodes as P) | ❌ Planned | needs dense `eigh` for K | Quality target for linear V-cycle |
| Beam (coarse sticks with bending modes) | ❌ Future | for elongated molecules | Related to RAFF rigid-cluster coarse graining |
| **Fitted staged modal decoder** | ✅ Implemented benchmark | `build_bend_twist_modes` + `ModalQuadratic` | **57× pure in-manifold; 53.8× mixed canonical decode** |
| Additive fitted modal correction | ✅ Implemented benchmark | same + `CoarseContract::Additive` | **57× pure in-manifold; 1.66× mixed atomistic state** |
| Force-projection Galerkin V-shape (no K fitting) | ❌ Designed | uses `project_force` + projected line search/FIRE | Nonlinear fallback; preserves full atomistic state |

## See also

- `notes/reports/2026-08-29_multigrid_consolidated_report.md` — **consolidated report**: full history, corrected benchmark data, conceptual insights, negative results, and prioritized improvements
- `notes/designs/2026-08-29_modal_relaxation_design_spec.md` — **design spec**: both approaches (fitted modal + force-projection Galerkin V-shape), RAFF/GPU roadmap, test molecule assignments
- `userguide/uff_spff.md` — UFF/SPFF end-user guide (parameter pipeline, real UFF setup, pentacene relaxation tests)
- `opencl/README.md` — kernel listing
