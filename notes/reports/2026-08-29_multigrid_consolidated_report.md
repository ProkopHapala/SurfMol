---
type: report
title: Multigrid and modal relaxation for molecular geometry optimization — consolidated report
description: Complete history and current status of coarse-grained molecular relaxation in SurfMol. Separates two valid contracts: staged coarse-manifold decoding (57× on pure in-manifold pentacene; 53.8× on mixed canonical decode) and additive same-state Galerkin relaxation (1.66× on mixed input). Covers fitted modal Newton, timestep/stiffness separation, nonlinear force-projection V-shape fallback, and implementation priorities.
tags: [multigrid, modal, coarse-graining, timestep-scaling, pentacene, benchmark, speedup, fire, uff, newton, dihedral-multi-well, galerkin, v-shape, force-projection, relaxation, projective-dynamics]
timestamp: 2026-08-29
---

# Multigrid and Modal Relaxation for Molecular Geometry Optimization

## 1. Summary

This report consolidates all work on coarse-grained molecular relaxation in SurfMol. The project went through three phases:

1. **Linear V-cycle solver** (ported from NumericalMathPlayground): implemented and tested. Works well on synthetic grids (current T4: 3.9× fewer smoothing steps) but underperforms on real molecules (1.4–3.6×) because each V-cycle is dominated by fine-level smoothing work, and the geometric prolongation doesn't capture molecular soft modes efficiently.

2. **Conceptual corrections** (from user feedback): the coarse solver handles low-frequency modes, the smoother handles high-frequency modes. High coarse compression is the goal, not a problem. The dt tradeoff (inner stability vs outer speed) must be measured jointly. The test must initialize with low-frequency distortion. H atoms are fast epiphenomena, not a bottleneck.

3. **Modal coarse-graining**: two complementary approaches remain promising: (A) fitted reduced stiffness + safeguarded Newton, and (B) true force-projection Galerkin updates alternating with fine smoothing. Projection can permit larger update scales because atomistic stiff directions are excluded, but the stable scale is governed by the current reduced Hessian and must be estimated or globalized.

**Both speedups are meaningful under different contracts.** Additive same-state relaxation preserves unresolved atomistic coordinates and gives **1.66×** on the mixed input (323→194). Staged coarse-manifold simulation intentionally retains only q and decodes canonical atom positions; it gives **53.8×** on the mixed workflow (323→6). On a pure in-manifold input, where no coordinates are discarded, both additive and decoder updates give **57×** (285→5). Thus the order-of-magnitude modal acceleration is real when the deformation is represented by the coarse state; decoder use must simply not be mislabeled as atomistic microstate preservation.

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

### 2.4 Operators

| Operator | Stiffness terms | Cost | Use case |
|----------|----------------|------|----------|
| `TrussOp` | Bonds only (axial `k·n⊗n`) | O(n_bonds) per matvec | Bond-only relaxation, large molecules, linear diagnostic |
| `UffHessianOp` | Bonds + angles + dihedrals + inversions | O(n²) per matvec, 2·n_dof force evals to build | Aromatic molecules (pentacene), bending/torsion, physical accuracy |

**Key insight**: at a planar equilibrium, axial bonds have **zero transverse stiffness** (the bond direction is in-plane). Without inversion/dihedral terms, the operator cannot resist out-of-plane bending — the mass term is the only resistance. `UffHessianOp` captures the full UFF stiffness and is required for physically meaningful multigrid on aromatic molecules. However, it is too expensive for production (2×n_dof force evals to build the Hessian) — useful for diagnostics only.

## 3. Linear V-cycle benchmark results

### 3.1 Synthetic test (8×8 grid, T4) — GOOD

On the canonical hard case (8×8 triangular grid, bottom row fixed, gravity load):
- **MG: 144 smooth-steps** (24 V-cycles × 6) to reach 1e-6
- **Jacobi: 561 steps** to reach 1e-6
- **Smoothing-step reduction: 3.9×** — multigrid works on the regular grid, before counting coarse/setup work.

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

## 6. Modal coarse-graining: corrected algorithmic picture

### 6.1 Where acceleration can come from

Projection excludes atomistic directions outside the coarse span. The relevant curvature is therefore the reduced Hessian `H_c=ΦᵀHΦ` (and reduced mass `M_c=ΦᵀMΦ` for dynamics), which can be much better conditioned than the full atomistic problem. This permits larger **validated** update scales and direct movement of collective modes. It does not justify a universal `dt≈22`: nonlinearity changes `H_c`, and a projected gradient sample does not determine its spectrum.

The second potential gain is cost: a reusable fitted or independently evaluable coarse model can take cheap internal steps between fine-force synchronizations. Without such a model, every true nonlinear projected-force step still costs a full force evaluation.

### 6.2 Two complementary approaches

**Approach A: fitted staged coarse-manifold/decoder**
- The simulation state is q; atomistic microstate outside q is intentionally absent.
- Fit reduced internal K/E_c once, evolve q with Newton or large-step modal FIRE, decode `x=D(q)`, then refine.
- Canonical decoding is desirable because it restores valid local bond/angle geometry without carrying unnecessary high-frequency noise through coarse global search.
- For production, use fitted internal forces cheaply between sparse full synchronizations; only external/surface/intermolecular coarse forces need updating during the coarse loop.
- Best when the decoder spans relevant conformations and one model is reused across many molecular instances.

**Approach B: force-projection Galerkin V-shape**
- Evaluate the true reduced gradient `g=ΦᵀF(x)` without a pre-fitted Hessian.
- Use projected gradient/modal FIRE with backtracking or a safeguarded online BB/BFGS scale.
- Alternate with fine smoothing only when soft-hard coupling regenerates coarse force.
- Best for nonlinear landscapes where a fixed K is unreliable, although it still fails if Φ does not span the important deformation.

The full force-balance equation is `F(x)=0`; `ΦᵀF(x)=0` is its Galerkin restriction. A nonlinear coarse update requires a new force evaluation at its trial point unless a fitted/online model or independent coarse forcefield supplies a validated prediction.

### 6.3 Critical prerequisite: real UFF parameters

The dummy-UFF/truss model does not contain physical aromatic out-of-plane bending or torsional stiffness. At planar equilibrium, axial bonds have zero linear transverse stiffness. The benchmark therefore loads real UFF bonds, angles, dihedrals, and aromatic inversions.

## 7. Contract-separated pentacene modal benchmark

### 7.1 Why the original 53× result needs a contract, not deletion

`x=x_ref+Φq` intentionally maps a coarse state to canonical atom coordinates. This is valid for staged coarse-grained simulation, where q is the complete molecular state during the coarse phase. It is not equivalent to relaxing an arbitrary existing atomistic microstate, because `(I−ΦΦᵀ)(x−x_ref)` is discarded by restriction/decoding.

The benchmark now exposes both contracts:
- **Additive:** `x←x+ΦΔq`, asserting that the fine complement is unchanged.
- **Decoder:** `x←D(q)=x_ref+Φq`, intentionally replacing unresolved atomistic coordinates.

Both use the same fitted K, true-energy acceptance, trust region, and full-force accounting.

### 7.2 Mixed bend/twist/noise input

| Strategy | N_force | Fine FIRE | fmax | z_rms | reported E | Interpretation |
|----------|--------:|----------:|-----:|------:|-----------:|----------------|
| plain FIRE | 323 | 323 | 8.30e-4 | 0.30348 | 1.06e-7 | Atomistic baseline |
| modal additive | 194 | 184 | 8.32e-4 | 0.30350 | 1.42e-5 | Same-state correction: **1.66×** |
| modal decoder | 6 | 1 | 1.77e-5 | 0.000027 | 4.67e-9 | Canonical coarse-to-fine: **53.8×** |

The mixed input has complementary RMS `0.30504 Å`. Additive mode preserves it; decoder mode intentionally resets it. Therefore 53.8× is valid for the decoder workflow but must not be described as preserving the same atomistic microstate.

### 7.3 Pure coarse-manifold input: fair common test

A second input was generated exactly as `x=x_ref+Φq` with `q=[1.0,-1.6]` and no fine noise. Initial complement RMS is approximately `7e-17`, so additive and decoder updates are mathematically the same operation.

| Strategy | N_force | Fine FIRE | fmax | reported E | Speedup |
|----------|--------:|----------:|-----:|-----------:|--------:|
| plain FIRE | 285 | 285 | 8.70e-4 | 4.20e-7 | 1× |
| modal additive | 5 | 1 | 1.29e-5 | 4.09e-9 | **57×** |
| modal decoder | 5 | 1 | 1.29e-5 | 4.09e-9 | **57×** |

This establishes that the order-of-magnitude speedup is not solely caused by deleting random noise. When the slow deformation lies in the fitted coarse subspace, three Newton steps plus four force synchronizations remove it, while explicit full-atom FIRE requires 285 evaluations.

### 7.4 Additive amplitude sweep on mixed inputs

| amp | FIRE N | corrected modal N | speedup | modal final z_rms | modal reported E |
|----:|-------:|------------------:|--------:|------------------:|-----------------:|
| 0.01 | 148 | 130 | 1.14× | 0.00638 | 7.92e-6 |
| 0.05 | 230 | 131 | 1.76× | 0.03026 | 7.73e-6 |
| 0.10 | 239 | 132 | 1.81× | 0.06060 | 7.59e-6 |
| 0.20 | 261 | 140 | 1.86× | 0.12137 | 8.86e-6 |
| 0.30 | 290 | 140 | 2.07× | 0.18216 | 1.26e-5 |
| 0.50 | 303 | 176 | 1.72× | 0.30373 | 6.57e-6 |
| 0.70 | 312 | 266 | 1.17× | 0.42542 | 9.62e-5 |
| 1.00 | 325 | 223 | 1.46× | 0.60769 | 4.70e-6 |

The large-amplitude cases expose fixed-K model breakdown and repeated line-search rejection. More importantly, the intended finite twist is a curvilinear rotation while the current twist column is only its tangent at the planar reference. For the main input, the two linear modes leave `fine_rms≈0.305 Å`; the coarse solver cannot remove a deformation it does not represent.

### 7.5 Current optimization priorities

1. **Use the fitted model during the coarse loop:** internal `g_int≈−Kq` is cheap; do not call full UFF at every coarse step. Update only coarse external/surface/intermolecular forces, then synchronize once before refinement. This is the main production speed mechanism.
2. **Curvilinear decoder:** represent finite ring/fragment rotations and bends with rigid transforms or internal coordinates rather than a linear tangent Φ. This keeps decoded geometries chemically valid over large amplitudes.
3. **Large reduced steps:** for relaxation, Newton/trust-region steps dominate explicit time integration. For coarse dynamics, use the reduced mass and reduced curvature to permit a larger stable timestep than full atoms.
4. **Nonlinear Galerkin fallback:** projected gradient/FIRE + backtracking and online SPD BFGS when the fitted model ceases to predict the true reduced force.
5. **Optimize synchronization and handoff:** minimize expensive external/full-force evaluations and fine refinement work, not nominal coarse-step count.
6. **Benchmark same physical endpoint:** use tighter convergence/common post-processing before comparing final energies or basins.

Mechanical parameter sweeps, table generation, larger test runs, and helper extraction can be delegated to cheaper agents. Curvilinear coordinates, reduced-mass/stability derivation, force/Jacobian consistency, and basin interpretation require expert attention.

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
