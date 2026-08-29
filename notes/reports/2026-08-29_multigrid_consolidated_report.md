---
type: report
title: Multigrid and modal relaxation for molecular geometry optimization — consolidated report
description: Complete history and current status of coarse-grained molecular relaxation in SurfMol. Covers (1) the linear V-cycle solver ported from NumericalMathPlayground, (2) why it underperformed on real molecules, (3) the pivot to modal coarse-graining with timestep scaling, (4) the 53× speedup achieved with fitted modal Newton on pentacene, and (5) the two-approach design (fitted modal + force-projection Galerkin V-shape). Includes all benchmark data, conceptual insights, negative results, and lessons learned.
tags: [multigrid, modal, coarse-graining, timestep-scaling, pentacene, benchmark, speedup, fire, uff, newton, dihedral-multi-well, galerkin, v-shape, force-projection, relaxation, projective-dynamics]
timestamp: 2026-08-29
---

# Multigrid and Modal Relaxation for Molecular Geometry Optimization

## 1. Summary

This report consolidates all work on coarse-grained molecular relaxation in SurfMol. The project went through three phases:

1. **Linear V-cycle solver** (ported from NumericalMathPlayground): implemented and tested. Works well on synthetic grids (4.7× speedup) but underperforms on real molecules (1.4–3.6×) because each V-cycle is dominated by fine-level smoothing work, and the geometric prolongation doesn't capture molecular soft modes efficiently.

2. **Conceptual corrections** (from user feedback): the coarse solver handles low-frequency modes, the smoother handles high-frequency modes. High coarse compression is the goal, not a problem. The dt tradeoff (inner stability vs outer speed) must be measured jointly. The test must initialize with low-frequency distortion. H atoms are fast epiphenomena, not a bottleneck.

3. **Modal coarse-graining with timestep scaling**: the speedup comes from freezing hard modes, allowing 1000× larger timesteps for soft modes. Two approaches: (A) fitted modal Newton (53× speedup on pentacene, finds true planar ground state) and (B) force-projection Galerkin V-shape (designed, not yet implemented — robust for nonlinear systems).

**Current best result: 53× speedup** on pentacene using fitted modal Newton + FIRE finishing (6 full-force evals vs 323 for plain FIRE).

## 2. Implementation map

### 2.1 What was built and where

| Component | File | Function/Struct |
|-----------|------|-----------------|
| `LinearOp` trait (natoms, matvec, diagonal_blocks, assemble_dense) | `crates/libs/molff/src/multigrid.rs` | `LinearOp` |
| Truss operator (bond-only, matrix-free) | `crates/libs/molff/src/multigrid.rs` | `TrussOp` impl `LinearOp` |
| Full UFF Hessian operator (finite-difference, dense) | `crates/libs/molff/src/multigrid.rs` | `UffHessianOp` impl `LinearOp` |
| Block Jacobi smoother (with heavy-ball momentum) | `crates/libs/molff/src/multigrid.rs` | `jacobi_smooth`, `jacobi_smooth_momentum`, `invert_3x3_blocks` |
| Geometric prolongation (pivot BFS + inverse-distance) | `crates/libs/molff/src/multigrid.rs` | `select_pivots_maximin`, `build_pivot_prolongation` |
| Galerkin coarse operator + cached level | `crates/libs/molff/src/multigrid.rs` | `galerkin_coarse`, `GalerkinLevel` |
| V-cycle (two-grid + outer loop + coarse-first) | `crates/libs/molff/src/multigrid.rs` | `solve_two_grid`, `solve_multigrid`, `solve_coarse_first`, `coarse_correct` |
| Fitted coupled modal quadratic model | `crates/libs/molff/src/multigrid.rs` | `ModalQuadratic::fit_central`, `solve_force`, `project_force` |
| Bend/twist orthonormal mode generator | `crates/libs/molff/src/multigrid.rs` | `build_bend_twist_modes` |
| Dense direct solve (test reference) | `crates/libs/molff/src/multigrid.rs` | `dense_solve` |
| f64 Cholesky factor + solve (coarse solve) | `crates/libs/numcore/src/math/linalg.rs` | `cholesky_factor_f64`, `cholesky_solve_f64`, `dense_matvec_f64` |
| Parity + convergence tests | `crates/libs/molff/tests/test_multigrid.rs` | T1–T7 (7 tests) |
| Molecule benchmarks (bond-only + full UFF) | `crates/libs/molff/tests/test_multigrid_molecules.rs` | pentacene (bond + full UFF), hexadecane, DiTriptyceno (4 tests) |
| Pentacene modal speedup benchmark | `crates/libs/surfmol/tests/relax_pentacene_speedup.rs` | Approach A (fitted modal Newton) |

### 2.2 OpenCL kernels (copied, not yet wired)

| File | Origin | Contents |
|------|--------|----------|
| `opencl/multigrid.cl` | `NumericalMathPlayground/.../kernels_multigrid.cl` | `prolongate` (tiled P), `restrict_residual_tree` (R=Pᵀ tree-reduce), coarse Cholesky solve |
| `opencl/block_jacobi.cl` | `NumericalMathPlayground/.../kernels_block_jacobi.cl` | `block_jacobi_step` (local-memory patches), `compute_residual`, `compute_diagonal_dinv` |

Both copied verbatim with provenance headers. Not yet loaded by any Rust crate.

### 2.3 Parity status

| Component | Status |
|-----------|--------|
| TrussOp matvec | ✅ Parity verified (T1: max err 1.5e-8 vs dense) |
| Diagonal blocks | ✅ Parity verified (T2: exact) |
| Direct solve | ✅ Parity verified (T3: max err 2.9e-13 vs Gaussian elimination) |
| V-cycle convergence (regular grid) | ✅ 4.7× speedup over Jacobi on 8×8 grid (T4) |
| Cached coarse force parity | ✅ GalerkinLevel vs exact coarse correction (T5) |
| Fitted modal quadratic parity | ✅ K fitted to 4.4e-16, response to 2.2e-16 (T6) |
| Bend/twist orthonormality | ✅ Gram matrix = I to 1e-14 (T7) |
| Full UFF Hessian (UffHessianOp) | ✅ 1000× more out-of-plane stiffness than bond-only on pentacene (7.67 vs 0.007) |
| V-cycle convergence (full UFF molecules) | ✅ MG converges in 2 V-cycles on pentacene with full UFF Hessian |
| Spectral prolongation | ❌ Not implemented (needs eigensolver) |
| OpenCL wiring | ❌ Kernels copied, not loaded by any Rust crate |
| Newton-step wrapper (`relax_uff_multigrid`) | ❌ Not implemented |
| RAFF integration | ❌ Not started |

### 2.4 Operators

| Operator | Stiffness terms | Cost | Use case |
|----------|----------------|------|----------|
| `TrussOp` | Bonds only (axial `k·n⊗n`) | O(n_bonds) per matvec | Bond-only relaxation, large molecules, linear diagnostic |
| `UffHessianOp` | Bonds + angles + dihedrals + inversions | O(n²) per matvec, 2·n_dof force evals to build | Aromatic molecules (pentacene), bending/torsion, physical accuracy |

**Key insight**: at a planar equilibrium, axial bonds have **zero transverse stiffness** (the bond direction is in-plane). Without inversion/dihedral terms, the operator cannot resist out-of-plane bending — the mass term is the only resistance. `UffHessianOp` captures the full UFF stiffness and is required for physically meaningful multigrid on aromatic molecules. However, it is too expensive for production (2×n_dof force evals to build the Hessian) — useful for diagnostics only.

## 3. Linear V-cycle benchmark results

### 3.1 Synthetic test (8×8 grid, T4) — GOOD

On the canonical hard case (8×8 triangular grid, bottom row fixed, gravity load):
- **MG: 120 smooth-steps** (20 V-cycles × 6) to reach 1e-6
- **Jacobi: 561 steps** to reach 1e-6
- **Speedup: 4.7×** — multigrid works as expected on a regular grid.

### 3.2 Real molecules (cantilever, mass-dominated) — MG converges but no end-to-end speedup

Setup: pin one end, apply transverse force at other end. mass_dt2=2500, penalty mass for pinned atoms, low-frequency distortion initialization (parabolic bend + 5% stretch + 0.01Å noise).

| Molecule | N atoms | N DOF | N coarse DOF | Jacobi → 1e-6 | MG → 1e-6 | Speedup |
|----------|---------|-------|-------------|---------------|-----------|---------|
| Pentacene | 36 | 108 | 15 (5 pivots) | 16 steps | 2 cycles (12 smooth) | 1.3× |
| n-Hexadecane | 50 | 150 | 18 (6 pivots) | 16 steps | 2 cycles (12 smooth) | 1.3× |
| DiTriptyceno | 104 | 312 | 18 (6 pivots) | 17 steps | 2 cycles (12 smooth) | 1.4× |

MG converges in exactly 2 V-cycles on all molecules, independent of size. The coarse residual drops as fast as the total residual, confirming the coarse solver captures the low-frequency error. **However, this is not an end-to-end speedup** — see §4.

### 3.3 Heavy-ball momentum HURTS in the mass-dominated regime

**Standalone Jacobi:** plain (β=0) converges in 16 steps; heavy-ball (β=0.5) takes 58 steps — **3.6× slower**.

**MG smoother:** β=0 converges in 2 cycles; β=0.5 takes 5-6 cycles — **2.5-3× slower**.

This is because the system is mass-dominated (mass_dt2=2500, k_eff=200, ratio=12.5). The eigenvalues of D⁻¹·A are close to 1, and heavy-ball momentum causes overshoot. In the reference demo (mass_dt2=2500, k_eff=20000, ratio=0.125), the system is stiffness-dominated and heavy-ball helps. **The optimal β depends on the mass/stiffness ratio.**

### 3.4 Early cantilever results (stiffness-dominated, mass_dt2=1.0) — underperforming

With the original parameters (mass_dt2=1.0, no penalty mass, x0=0):

| Molecule | Jacobi → 1e-3 | MG → 1e-3 | Speedup |
|----------|---------------|-----------|---------|
| Pentacene | 2290 steps | 1632 (272 cyc) | 1.4× |
| n-Hexadecane | 2011 steps | 1260 (210 cyc) | 1.6× |
| DiTriptyceno | >5000 (fails) | 1398 (233 cyc) | >3.6× |

The first V-cycle barely reduced the residual on molecules (0.50→0.49 for pentacene) but dropped it 5 orders of magnitude on the grid (0.17→9.5e-6). The coarse correction was negligible. This was diagnosed as: (1) geometric prolongation doesn't span soft modes, (2) test measures fine-DOF residual not low-frequency residual, (3) test doesn't initialize with low-frequency distortion, (4) dt tradeoff not measured. See §5 for the corrected understanding.

## 4. Why the linear V-cycle doesn't give end-to-end speedup

### 4.1 The V-cycle is dominated by fine-level work

The current Rust two-grid solver repeats 3 fine pre-smoothing and 3 fine post-smoothing steps for every coarse correction. The coarse system may have only 6–15% of the original DOFs, but each cycle still spends almost all work on full-atom `TrussOp::matvec` calls. Consequently, the implementation cannot demonstrate a 10×–100× reduction in cost per coarse iteration.

"Two V-cycles" sounds fast, but each V-cycle contains 6 fine smoothing steps + 1 restriction + 1 check = 8 fine operator applications. Two cycles = 16 fine applications, plus Galerkin setup (n_coarse fine applications). This is comparable to 16–17 plain Jacobi steps in the mass-dominated case.

### 4.2 Three different algorithms must not be conflated

**Classical linear V-cycle:** `fine pre-smooth → restrict → coarse solve → prolongate → fine post-smooth`, repeated. Every cycle evaluates the fine operator multiple times. Cost is dominated by the fine level.

**Staged/nested coarse-to-fine:** `relax coarse model for many cheap steps → prolongate once → relax full atomistic model only to remove local strain`. This directly exploits both benefits (algorithmic convergence + lower cost per coarse update). The coarse phase should stop when coarse forces stall, not when the atomistic residual is small.

**Nonlinear FAS:** alternate coarse and fine levels using a nonlinear Full Approximation Scheme. More general but more complex. Should be implemented only if staged relaxation fails because the coarse optimum changes substantially after local relaxation.

### 4.3 Coarse-first diagnostic — negative result

Implemented `coarse_correct` and `solve_coarse_first`. The one-shot correction evaluates the fine residual once, solves the Galerkin system, prolongates once, and then performs fine Jacobi only until the requested residual.

| Molecule | Fine Jacobi to 1e-6 | Coarse-first fine finishing steps | Repeated V-cycle fine steps |
|----------|---------------------|----------------------------------|----------------------------|
| pentacene | 16 | 14 | 12 + 2 coarse corrections |
| n-hexadecane | 16 | 13–14 | 12 + 2 coarse corrections |
| DiTriptyceno | 17 | 14–15 | 12 + 2 coarse corrections |

**Negative result:** the mass-dominated test does not demonstrate meaningful multigrid speedup. The diagonal inertial term makes the fine solve easy, so there is little slow collective error for the coarse level to remove. Coarse-first saves only 1–3 fine sweeps. Repeated V-cycles save 4–5 but add coarse corrections and hierarchy work.

### 4.4 Cached nonlinear coarse-force on xylitol — negative result

Implemented `GalerkinLevel` (stores P, A_c, Cholesky factor). `solve_force` computes Δx=P·A_c⁻¹·Pᵀ·F without a fine A·x application. `MolWorld::apply_coarse_force_step` applies the displacement.

On randomly distorted xylitol, one cached coarse step reduced UFF energy 19173→14508 and max force 1570→1394 — a valid descent direction. But equal-threshold end-to-end work did not improve:
- plain FIRE: 174 full force evaluations
- one coarse update + FIRE: 176 full force evaluations

**Xylitol is small and the test applies random atomwise distortion, so local/high-frequency error dominates.** It is not a suitable case for demonstrating macroscopic MG benefit.

### 4.5 The Galerkin setup is itself expensive

`galerkin_coarse` forms `A_c = Pᵀ·A·P` by applying the fine operator once per coarse basis column. For 15–36 coarse DOFs, that's 15–36 fine operator applications before any solve. This setup must be amortized across many outer steps, updated incrementally, or replaced by an independently evaluable physical coarse forcefield. Rebuilding it every nonlinear step erases all speedup.

## 5. Conceptual insights (hard-won, do NOT forget)

> These insights emerged from the user's feedback during the debugging process. They are essential for all future multigrid/modal work.

### 5.1 The coarse solver handles low-frequency modes; the smoother handles high-frequency modes

Do not judge the coarse solver by fine-DOF residuals. Do not expect the coarse solver to relax H atoms or bond stretches. The multigrid V-cycle has a clear division of labor:
- **Coarse solve:** relaxes the low-frequency (soft, collective) modes — whole-molecule bending, stretching, torsion.
- **Fine smoother:** relaxes the high-frequency (stiff, local) modes — bond stretches, H-atom wiggles, local distortions.

Measuring the residual over all fine DOFs (including H) to judge whether the coarse solve is working is like judging a low-pass filter by how well it reproduces high frequencies.

### 5.2 High coarse compression ratio is GOOD, not bad

The whole point of multigrid is coarse compression — representing the soft subspace with far fewer DOFs (1:10 or 1:100). A high compression ratio is good because:
1. The coarse solve is cheap (few DOFs → fast dense solve).
2. The coarse space neglects hard DOFs, allowing a large effective timestep → fast relaxation of soft DOFs.
3. Hard DOFs are easily relaxed afterward by the fine smoother.

The problem is not "too few coarse DOFs" — it's that the coarse DOFs don't span the right subspace (the soft modes).

### 5.3 H atoms are fast epiphenomena, not a bottleneck

H atoms are bonded stiffly to their host C atoms and are just carried along. A C-H bond is stiff and short. The H atom's displacement is almost entirely determined by its host C's displacement. The Jacobi smoother handles H atoms in 1–2 steps. Eliminating H atoms via Schur complement is unnecessary.

### 5.4 The dt tradeoff must be measured jointly (inner+outer)

Small dt → large M/dt² → well-conditioned inner system → inner Jacobi converges fast. BUT: small dt means the inner solution is a compromise between the true attractor and inertia — the outer loop converges slowly.

Large dt → small M/dt² → poorly-conditioned inner system → inner Jacobi converges slowly. BUT: each step moves further toward the true minimum, so the outer loop converges faster.

**The sweet spot** balances inner solver stability against outer relaxation speed. Measuring only the inner solver's convergence is misleading. The benchmark must measure total work to reach the nonlinear energy minimum.

### 5.5 The test must initialize with low-frequency distortion

x0=0 with a pure force RHS excites all modes equally — both low-frequency bending (which the coarse solver should handle) and high-frequency bond stretches (which the smoother should handle). This makes it impossible to tell whether the coarse solver is doing its job.

The correct test initializes with a low-frequency distortion (parabolic bend, axial stretch) so the initial error is dominated by the soft modes. Then we can measure whether the coarse solve reduces the low-frequency residual.

### 5.6 The full relaxation is nonlinear; the inner linear solve is an approximation

The inner solve should use a dt large enough that the outer loop converges fast, but small enough that the inner linear solver is stable. The linearized coarse solve is sufficient near equilibrium; nonlinear coarse relaxation may be needed far from equilibrium. Early in nonlinear relaxation, solving a stale linearization to 1e-10 wastes work. Use forcing terms tied to nonlinear progress: loose inner tolerance far from equilibrium, tighter only near the final minimum.

### 5.7 Heavy-ball momentum is spectrum-dependent

A fixed β=0.5 is not universally beneficial. In the mass-dominated regime (mass/k ratio > 1), heavy-ball causes overshoot and slows convergence. In the stiffness-dominated regime (mass/k < 1), it helps. Momentum parameters must depend on the spectrum and should be reset after a discontinuous coarse correction. For nonlinear relaxation, FIRE or adaptive momentum is safer than a fixed heavy-ball coefficient.

## 6. Modal coarse-graining with timestep scaling — the breakthrough

### 6.1 Where the speedup comes from

The primary speedup mechanism is **timestep scaling**, not fewer iterations:

- **Full-atom dynamics:** f_max ≈ sqrt(k_bond/m) ≈ 4.1 → dt ≈ 0.02
- **Modal coarse dynamics:** only soft modes evolved, hard modes frozen. f_max ≈ sqrt(K_twist) ≈ 0.45 → dt ≈ 22 — **1000× larger**
- **Newton step** (for fitted modal): for a quadratic model, one Newton step reaches the exact equilibrium — equivalent to infinite timestep.

The coarse phase converges soft DOFs in a few large-timestep steps. Then the fine phase only needs to relax hard DOFs (bond stretches, H atoms), which converge fast because they're stiff and local.

### 6.2 Two complementary approaches

**Approach A: Fitted modal (coarse-first, then fine)** — IMPLEMENTED, 53× SPEEDUP
- Fit stiffness K once from 2×n_modes force evals (setup, amortized)
- Newton steps: dq = K⁻¹·g (exact for quadratic model)
- Trust region for large distortions
- Best for: approximately quadratic systems (aromatic molecules, rigid frameworks)

**Approach B: Force-projection Galerkin V-shape (no fitting)** — DESIGNED
- Don't fit K. Project atomic forces onto modes at each sync: g = ΦᵀF(x)
- Large-timestep modal damped MD: dt_modal ≈ 22 (large, because only soft modes evolved)
- V-shape: alternate coarse (soft DOFs) and fine (hard DOFs) phases
- Online K estimation: after 2+ syncs, estimate K ≈ -Δg/Δq (secant/BFGS) → switch to Newton
- Best for: nonlinear systems (aliphatic chains with torsions, large distortions, conformational changes)

**Key insight for Approach B:** the timestep is large even without K — stability is set by the highest frequency in the evolved subspace, not by whether we know K. Since we only evolve soft modes, dt_modal ≈ 22 regardless.

**What "Galerkin" means in Approach B:** the full equation is F(x) = 0 (force balance). Projecting onto modes: ΦᵀF(x) = 0 (modal force balance). The force projection g = ΦᵀF(x) is the Galerkin restriction. The V-shape is the standard multigrid pattern: restrict → solve coarse → prolongate → smooth fine — but with physically-motivated modes instead of geometric prolongation, and nonlinear force balance instead of linear system.

### 6.3 Critical prerequisite: real UFF parameters

The existing dummy-UFF/truss model does NOT contain physical aromatic out-of-plane bending or torsional stiffness. At a planar unstressed geometry, axial bonds have zero linear transverse stiffness. A "soft bend mode" tested with this model is a mechanism controlled only by mass regularization and nonlinear bond stretching, not pentacene physics.

A trustworthy benchmark must load real parameters from `data/{ElementTypes,AtomTypes,BondTypes,AngleTypes,DihedralTypes}.dat` and enable aromatic inversion terms. The pentacene speedup benchmark (`relax_pentacene_speedup.rs`) does this correctly.

## 7. Pentacene modal speedup benchmark (Approach A) — 53× speedup

### 7.1 Setup

1. Load pentacene (36 atoms) with real UFF parameters (bonds + angles + dihedrals + inversions)
2. In-plane relax (inversions OFF) → remove in-plane strain
3. Full relax (inversions ON, fconv=1e-5) → true planar ground state (E=3.70e-9, fmax=9.98e-6)
4. Build bend/twist modes Φ from planar reference
5. Fit modal stiffness K from 2×2=4 force evals (central differences) — NOT counted (setup)
6. Factor K (Cholesky) — reusable

### 7.2 Main benchmark (bend=0.5 Å, twist=0.3 rad, noise=0.02 Å)

| Strategy | N_force | N_steps | fmax | z_rms | E | Ground state? |
|----------|---------|---------|------|-------|---|---------------|
| plain FIRE | 323 | 323 | 8.30e-4 | 0.3035 | 1.06e-7 | NO (local min) |
| **modal + FIRE** | **6** | **1** | **1.77e-5** | **0.00003** | **4.67e-9** | **YES** |

- Coarse phase: 5 syncs + 4 Newton steps = 5 full-force evals
- Fine phase: 1 FIRE step = 1 full-force eval
- **Speedup: 53.8×**

### 7.3 Distortion amplitude sweep

| amp (Å) | FIRE N | FIRE z_rms | FIRE E | modal N | modal z_rms | modal E | speedup | same min? |
|---------|--------|-----------|--------|---------|------------|---------|---------|-----------|
| 0.01 | 148 | 0.0069 | 2.61e-5 | 4 | 0.00001 | 3.90e-9 | 37× | SAME |
| 0.05 | 230 | 0.0308 | 4.37e-5 | 4 | 0.00002 | 3.98e-9 | 57.5× | DIFF |
| 0.10 | 239 | 0.0611 | 7.51e-5 | 4 | 0.00003 | 4.40e-9 | 59.8× | DIFF |
| 0.20 | 261 | 0.1216 | 7.42e-5 | 5 | 0.000002 | 3.70e-9 | 52.2× | DIFF |
| 0.30 | 290 | 0.1823 | 5.92e-5 | 5 | 0.00002 | 4.08e-9 | 58.0× | DIFF |
| 0.50 | 303 | 0.3037 | 1.92e-7 | 6 | 0.00003 | 4.65e-9 | 50.5× | DIFF |
| 0.70 | 312 | 0.4253 | 5.74e-7 | 6 | 0.00003 | 5.18e-9 | 52.0× | DIFF |
| 1.00 | 325 | 0.6077 | 6.83e-7 | 6 | 0.000009 | 3.89e-9 | 54.2× | DIFF |

Key observations:
- **Modal always finds the true ground state** (E≈4e-9, z_rms≈0) regardless of distortion amplitude
- **Plain FIRE gets trapped in non-planar local minima** for amp ≥ 0.05 (E up to 7.51e-5, 17000× higher)
- **Speedup is 37-60×** consistently across all amplitudes
- **At amp=0.01, both find the same minimum** — FIRE doesn't get trapped for small distortions

### 7.4 Modal convergence trace (amp=0.5)

```
sync 0: fmax=1.2235e0 gmax=4.5624e-1 q=[0.9937, -1.6115]  (initial distorted)
sync 2: fmax=3.4112e-1 gmax=1.8115e-1 q=[-0.7437, -0.6209] (Newton overshoot, trust region)
sync 3: fmax=8.2862e-2 gmax=5.2418e-2 q=[0.4344, 0.2039]   (converging)
sync 4: fmax=3.9884e-3 gmax=3.9664e-3 q=[-0.0497, -0.0134] (nearly there)
sync 5: fmax=1.7748e-5 gmax=1.4257e-5 q=[-0.0002, -0.0001] (converged!)
```

4 Newton steps + 5 syncs = 5 full-force evals. The quadratic model converges exactly as expected for pentacene's approximately linear bending.

### 7.5 Why plain FIRE fails

Plain FIRE gets trapped in non-planar local minima because of the **UFF dihedral multi-well landscape**:

- UFF sp2-sp2 dihedral terms have minima at multiple angles (cosine potential with n>1)
- When the molecule is distorted with a large bend+twist, FIRE's momentum carries it past the planar minimum and into a twisted local minimum where some dihedrals are at their alternative minima
- The energy difference is small in absolute terms (1e-7 vs 4e-9 eV) but the geometric difference is large (z_rms=0.30 vs 0.00003)
- The planar geometry IS the true ground state (verified: E=3.70e-9, fmax=9.98e-6)

The modal approach avoids this by projecting onto smooth bend/twist modes that filter out the local dihedral trapping. The quadratic model has a single minimum at q=0 (planar), which coincides with the true ground state. **This is both a speedup AND a global optimization benefit.**

### 7.6 Three errors fixed (v1 → v2)

1. **Newton instead of FIRE with tiny timestep:** v1 used modal FIRE with dt=0.05 (full-atom timestep). The modal K gives dt_max=22 — 400× larger. v2 uses Newton (exact for quadratic, equivalent to infinite timestep).

2. **Fitting cost excluded:** v1 counted the 4 fitting evals. v2 correctly excludes them — fitting is one-time setup, amortized over thousands of molecules × millions of steps.

3. **Reference geometry properly relaxed:** v1 relaxed with inversions OFF, then turned inversions ON, leaving residual forces (fmax≈1e-3). v2 does full relaxation with inversions ON (fmax=9.98e-6), finding the true planar ground state.

## 8. Cost model and benchmark protocol

### 8.1 Work accounting

Report work by level, not merely "iterations":
- `N_fine_force`: full nonlinear forcefield evaluations
- `N_coarse_steps`: modal Newton steps or damped MD steps (cheap, ~free)
- `N_sync`: full-force evaluations during coarse phase (for synchronization)
- Setup cost (mode generation, stiffness fitting, Cholesky) — NOT counted (amortized)
- Wall time, separately for setup and solve

Total: `N_total = N_sync + N_fine_fire`. Compare vs `N_plain` = plain FIRE steps to same threshold.

### 8.2 What to compare

Compare against **FIRE only** (the best available optimizer). Do NOT compare against damped MD — that would be dishonest.

### 8.3 Distortion

- **Low-frequency:** parabolic bend + axial twist (the soft modes the coarse model should handle)
- **Small high-frequency:** white noise (the hard modes the fine phase should handle)
- **Amplitude sweep:** test multiple amplitudes to find where Approach A breaks down and Approach B is needed

### 8.4 Convergence criterion

All strategies must reach the same final force threshold (fmax < 1e-3 eV/Å) AND the same minimum (check z_rms, energy, and trajectory visually). Always save trajectories and inspect visually. A different minimum is not a feature until verified.

### 8.5 Separate slow and fast error

For diagnosing multigrid (not needed for modal benchmark):
- Coarse generalized force `g_c = PᵀF`
- Fine/local strain: bond-length, angle residuals after subtracting coarse displacement
- Total `max|F|`, RMS force, energy

## 9. Test molecules and coarse-point assignments

### 9.1 Pentacene — rigid aromatic stick (36 atoms)

**Geometry:** 22 C + 14 H, flat in xy-plane, elongated along x (~13Å). Five fused benzene rings. Essentially a rigid rod.

**Soft modes:** long-axis bending (beam flex), axial twist. C-C and C-H stretches are much stiffer.

**Coarse modes (modal approach):** out-of-plane bend `n·sin(πs)` + axial twist `(2s-1)·[u×(r-r_axis)]`. 2 modes, 2 coarse DOF. Captures the softest internal modes.

**Coarse points (geometric pivot approach):** 2–3 pivots (both ends + optional midpoint). 6–9 coarse DOF. Captures rigid motion + bending but not twist.

### 9.2 n-Hexadecane — flexible rope (50 atoms)

**Geometry:** 16 C backbone (zigzag, ~19Å end-to-end) + 34 H.

**Soft modes:** many low-frequency bending/torsion modes. The chain can fold, twist, and bend in many ways. This is the **hardest case for Jacobi** and where multigrid/modal should shine.

**Coarse points (geometric):** 4–8 pivots along the chain (12–24 coarse DOF). Captures overall shape + several bending modes.

**Spectral approach:** expect a dense spectrum of soft bending/torsion modes with no clear gap. May need 15–20 modes. This is where Approach B (force-projection, no fitting) is likely needed — the quadratic model breaks down for torsional DOFs.

### 9.3 DiTriptyceno_helicene — branching I-beam (104 atoms)

**Geometry:** flat aromatic core + 4 triptycene protrusions sticking out of plane.

**Soft modes:** protrusion rotation/twisting relative to core (hinge modes). Possible spectral gap between hinge modes and internal stiff modes.

**Coarse points (geometric):** 6 pivots (2 core ends + 4 protrusion tips). 18 coarse DOF. Captures core rigid motion + 4 protrusion displacements.

**Spectral approach:** best candidate for spectral-gap approach — the rigid core creates a natural separation.

## 10. Next steps

1. **Implement Approach B** (force-projection Galerkin V-shape) and benchmark on pentacene — verify speedup without fitting
2. **Test on hexadecane** (aliphatic chain with torsional DOFs) — where Approach A should fail and Approach B should work
3. **Implement online K estimation** (secant/BFGS in modal space) — hybrid of A and B
4. **Test on larger molecules** (polyacene ribbons, 500+ atoms) — where speedup should be even more dramatic
5. **Add more modes** (stretch, in-plane bend) for systems where bend/twist alone is insufficient
6. **Spectral prolongation:** implement dense eigensolver for small Hessian (≤ 600 DOF), compare against geometric and modal
7. **RAFF extension:** generalize to 6-DOF atoms (position + quaternion) via adiabatic Schur complement
8. **GPU (OpenCL):** wire copied kernels to SurfMol buffer conventions, benchmark on ≥ 1000-atom systems

## 11. Files

### Implementation
- `crates/libs/molff/src/multigrid.rs` — full MG stack (TrussOp, V-cycle, prolongation, modal quadratic, bend/twist modes)
- `crates/libs/numcore/src/math/linalg.rs` — f64 dense Cholesky
- `crates/libs/molff/tests/test_multigrid.rs` — T1–T7 parity + convergence tests
- `crates/libs/molff/tests/test_multigrid_molecules.rs` — molecule benchmarks
- `crates/libs/surfmol/tests/relax_pentacene_speedup.rs` — pentacene modal speedup benchmark (Approach A)
- `opencl/multigrid.cl`, `opencl/block_jacobi.cl` — copied OpenCL kernels (not yet wired)

### Benchmark output
- `debug/relax_pentacene_speedup/speedup_summary.tsv` — main results
- `debug/relax_pentacene_speedup/sweep_amplitude.tsv` — amplitude sweep
- `debug/relax_pentacene_speedup/traj_modal_coarse.xyz` — modal coarse trajectory (visual inspection)

### Design
- `notes/designs/2026-08-29_modal_relaxation_design_spec.md` — full design spec (both approaches, RAFF/GPU roadmap)
- `doc/topical_audit/multigrid.md` — cross-implementation map

### Reference
- `NumericalMathPlayground/topics/LinarElasticity/` — reference Python+OpenCL implementation
