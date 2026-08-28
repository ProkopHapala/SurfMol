---
type: report
title: RAFF position-based solver benchmark — FIRE vs PBD vs XPBD vs Projective
description: Systematic comparison of force-based (FIRE, damped MD) and position-based (PBD-compliance, XPBD, Projective Dynamics) relaxation solvers on the RAFF port-based rigid force field. Steps-to-convergence, dt sweep, stability limits, and wall time.
tags: [raff, benchmark, pbd, xpbd, projective-dynamics, fire, relaxation, convergence]
timestamp: 2026-08-28
---

# RAFF Position-Based Solver Benchmark

## 1. Goal

Compare the convergence speed of position-based dynamics (PBD) variants against force-based molecular dynamics (FIRE, damped Euler) for geometry relaxation in the RAFF port-based rigid force field. The question: **can position-based solvers beat force-based methods for interactive pre-optimization and global relaxation?**

## 2. Setup

### 2.1 Solvers tested

| Solver | Type | Key parameters |
|--------|------|----------------|
| **FIRE** | Force-MD, momentum + adaptive dt | dt0=0.001, dtmax=0.01 and 0.02, alpha=0.1→0.099 |
| **Damped MD** | Force-MD, constant damping | dt=0.005, cdamp=0.95 (old baseline) |
| **PBD-compliance or=1.9** | Position-based, over-relaxed | dt=0.05, 0.1, 0.2; iters=16; over_relax=1.9 |
| **XPBD-lagged** | Position-based, compliance | dt=0.05, 0.1, 0.2; iters=16; over_relax=1.0 |
| **Projective** | Position-based, Jacobi global | dt=0.05, 0.1, 0.2; iters=16; over_relax=1.0 |

FIRE implementation: Bitzek et al. 2006. Momentum-based, `dot(v,F)<0` velocity reset, adaptive dt increase (×1.1 after 5 positive steps), dt decrease (×0.5 on uphill), alpha decay (×0.99). Uses Dynamic orientation mode (smooth rotation integration — Adiabatic mode's discontinuous snaps pump energy into momentum).

### 2.2 Molecules

| Name | Atoms | Description |
|------|-------|-------------|
| CH4 | 5 | Methane (rigid, no free DOFs) |
| water | 3 | H2O (rigid, no free DOFs) |
| H2O2 | 4 | Hydrogen peroxide (1 free dihedral DOF) |
| tree20 | 20 | Random tree molecule (multiple branch rotation DOFs) |
| tree100 | 100 | Random tree molecule (many branch rotation DOFs) |

### 2.3 Distortions

| Name | Description | What it tests |
|------|-------------|---------------|
| **D1 random** | Gaussian displacement, σ=0.1 Å | High-frequency noise (easy) |
| **D2 stretch** | 1.3× stretch along PCA principal axis | Low-frequency collective mode (hard — long narrow valley) |
| **D3a dihedral** | 60° rotation around a bond | Free-DOF perturbation (tests torsion null space) |

### 2.4 Convergence targets (force-based, DOF-independent)

| Target | Threshold | Use case |
|--------|-----------|----------|
| **T2 (rough)** | max\|F\| < 0.1 eV/Å (100 meV/Å) | Interactive pre-optimization |
| **T1 (accurate)** | max\|F\| < 1e-3 eV/Å (1 meV/Å) | Geometry optimization |

**Primary metric: max|F|** (force residual). DOF-independent — when E=0, |F|=0 regardless of which point in the free-DOF manifold the solver reaches. RMSD vs reference is secondary and meaningless for molecules with free dihedral/branch-rotation DOFs.

### 2.5 Reference geometry

For each (molecule, distortion) pair: relax with FIRE to |F| < 1e-3 eV/Å → reference geometry. This is the force field's attractor from the perturbed state, **not** the input geometry. All solvers are then relaxed from the same perturbed state and compared.

### 2.6 Budget

- Max 10,000 steps per solver (divergent solvers hit this cap)
- Max 10,000 steps for reference relaxation
- Release mode (`cargo run --release`)
- Full benchmark: 5 molecules × 3 distortions × 12 solver configs = 180 runs, completes in <1 minute

## 3. Results — steps to T2 (rough: |F| < 0.1 eV/Å)

Best solver per row in **bold**. "--" = did not converge in 10k steps.

| Molecule | Distortion | FIRE dtmax=0.02 | Damped MD | PBD or=1.9 dt=0.05 | XPBD dt=0.2 | Proj dt=0.2 |
|---|---|---|---|---|---|---|
| CH4 | D1 random | 77 | 189 | **2** | 6 | 4 |
| CH4 | D2 stretch | 68 | 180 | **2** | 6 | 4 |
| CH4 | D3a dihedral | 119 | 161 | **2** | 8 | 3 |
| water | D1 random | 79 | 158 | **2** | 5 | 3 |
| water | D2 stretch | 92 | 158 | **2** | 6 | 4 |
| H2O2 | D1 random | 74 | 179 | **2** | 5 | 3 |
| H2O2 | D2 stretch | 77 | 205 | **2** | 7 | 5 |
| H2O2 | D3a dihedral | 1 | 1 | 1 | 1 | 1 |
| tree20 | D1 random | 117 | 427 | **6** | 16 | 7 |
| tree20 | D2 stretch | 130 | 1513 | **25** | 46 | 23 |
| tree20 | D3a dihedral | 166 | 1082 | **25** | 42 | 15 |
| tree100 | D1 random | 179 | 930 | 36 | 73 | **15** |
| tree100 | D2 stretch | 314 | 3464(†) | 138 | 178 | **50** |
| tree100 | D3a dihedral | 284 | 2939(†) | 179 | 232 | **46** |

(†) Damped MD did not converge to T1 in 10k steps.

## 4. Results — steps to T1 (accurate: |F| < 1e-3 eV/Å)

| Molecule | Distortion | FIRE dtmax=0.02 | Damped MD | PBD or=1.9 dt=0.05 | XPBD dt=0.2 | Proj dt=0.2 |
|---|---|---|---|---|---|---|
| CH4 | D1 random | 132 | 333 | **4** | 12 | 8 |
| CH4 | D2 stretch | 122 | 329 | **4** | 12 | 9 |
| water | D1 random | 140 | 338 | **3** | 10 | 7 |
| H2O2 | D1 random | 128 | 354 | **3** | 11 | 8 |
| H2O2 | D2 stretch | 128 | 367 | **4** | 12 | 10 |
| tree20 | D1 random | 221 | 2519 | **32** | 65 | 35 |
| tree20 | D2 stretch | 252 | 4524 | **61** | 107 | 67 |
| tree20 | D3a dihedral | 284 | 2634 | **66** | 107 | 40 |
| tree100 | D1 random | 461 | 7847 | 256 | 264 | **112** |
| tree100 | D2 stretch | 581 | --(†) | **365** | 492 | 251 |
| tree100 | D3a dihedral | 541 | --(†) | 411 | 515 | **226** |

## 5. dt sweep — stability limits

### PBD-compliance or=1.9

| dt | CH4 | water | H2O2 | tree20 | tree100 |
|----|-----|-------|------|--------|---------|
| 0.05 | **stable** (2-4 steps) | **stable** (2-3) | **stable** (2-4) | **stable** (6-25) | **stable** (36-179) |
| 0.1 | **DIVERGES** (E→158) | stable (5-9) | stable (5-9) | **DIVERGES** (E→14) | **DIVERGES** (E→168) |
| 0.2 | **DIVERGES** (E→197) | stable (9-19) | **DIVERGES** (E→42) | **DIVERGES** (E→319) | **DIVERGES** (E→351) |

PBD-or1.9 is stable only at dt=0.05 on stiff molecules (CH4, tree). On water (flexible, 3 atoms) it's stable up to dt=0.2. **Over-relaxation (or=1.9) + large dt = unstable.**

### XPBD (or=1.0)

| dt | CH4 | water | H2O2 | tree20 | tree100 |
|----|-----|-------|------|--------|---------|
| 0.05 | stable (30-41) | stable (27-35) | stable (31-47) | stable (98-385) | stable (247-1228) |
| 0.1 | stable (12-13) | stable (10-12) | stable (11-15) | stable (31-124) | stable (136-538) |
| 0.2 | **stable** (6-8) | **stable** (5-6) | **stable** (5-7) | **stable** (16-46) | **stable** (73-232) |

XPBD is stable at all tested dt values. Larger dt = fewer steps. **dt=0.2 is optimal.**

### Projective (or=1.0)

| dt | CH4 | water | H2O2 | tree20 | tree100 |
|----|-----|-------|------|--------|---------|
| 0.05 | stable (24-36) | stable (22-31) | stable (27-44) | stable (90-311) | stable (192-697) |
| 0.1 | stable (8-11) | stable (7-10) | stable (8-13) | stable (24-79) | stable (49-176) |
| 0.2 | **stable** (3-4) | **stable** (3-4) | **stable** (3-5) | **stable** (7-23) | **stable** (15-50) |

Projective is stable at all dt values and benefits most from large dt. **dt=0.2 is optimal and the fastest overall on large molecules.**

## 6. Wall time

Wall time per step varies significantly between solvers. Position-based solvers do more work per step (16 iterations of constraint solving) than force-MD (1 force evaluation).

| Molecule | Solver | Steps to T2 | Wall time (µs) | µs/step |
|----------|--------|------------:|----------------:|--------:|
| CH4 | FIRE dtmax=0.02 | 77 | 177 | 2.3 |
| CH4 | PBD dt=0.05 | 2 | 62 | 31 |
| CH4 | XPBD dt=0.2 | 6 | 195 | 33 |
| CH4 | Proj dt=0.2 | 4 | 103 | 26 |
| tree100 | FIRE dtmax=0.02 | 314 | 4031 | 13 |
| tree100 | PBD dt=0.05 | 138 | 381190 | 2762 |
| tree100 | XPBD dt=0.2 | 178 | 523988 | 2944 |
| tree100 | Proj dt=0.2 | 50 | 273240 | 5465 |

**Key observation**: on small molecules, PBD wins both in steps and wall time. On tree100, FIRE wins in wall time (4ms vs 273-524ms for position-based) despite needing 6× more steps — the per-step cost of position-based solvers scales poorly with atom count. **This is the main bottleneck to fix.**

## 7. Key findings

1. **PBD-compliance or=1.9 dt=0.05 is the fastest solver for small/medium molecules** (2-25 steps to T2). It converges in 2 steps on CH4/water/H2O2 — essentially one relaxation sweep. But it's **unstable at dt≥0.1 on stiff molecules**.

2. **Projective dt=0.2 is the fastest on large molecules** (tree100: 15-50 steps to T2 vs 36-179 for PBD, 179-314 for FIRE). The global Jacobi solve handles collective modes better than Gauss-Seidel (PBD/XPBD).

3. **FIRE is 2× faster than damped MD** (CH4: 77 vs 189 steps to T2). Momentum + adaptive dt helps. But it's still 10-60× slower than PBD on small molecules.

4. **Damped MD doesn't converge to T1 on tree100** in 10k steps (|F| stalls at 1e-3). FIRE converges in 461-823 steps. **Momentum-based methods are essential for large molecules.**

5. **The D2 stretch pathology is confirmed**: all solvers slow down on D2 vs D1 (tree100 Damped MD: 930→3464 steps to T2). The low-frequency collective mode is the bottleneck. PBD is less affected (36→138) than force-MD (930→3464).

6. **H2O2 D3a dihedral = 1 step for all solvers**: the dihedral is a genuine free DOF in the port model (only bond lengths + angles are constrained, not torsions). The perturbation rotates atoms around the O-O bond, but since the port model doesn't constrain the dihedral, the geometry is already at E≈0. **This documents the missing torsion term.**

7. **Wall time bottleneck**: position-based solvers have 10-200× higher per-step cost than FIRE on large molecules. The 16-iteration constraint solve doesn't scale well. **Optimization target: reduce per-step cost or reduce iteration count.**

## 8. Recommendations

| Use case | Recommended solver | Parameters |
|----------|-------------------|------------|
| Interactive pre-opt (small molecules) | PBD-compliance or=1.9 | dt=0.05, iters=16 |
| Interactive pre-opt (large molecules) | Projective | dt=0.2, iters=16 |
| Accurate geometry optimization | PBD-compliance or=1.9 (small) / Projective (large) | dt=0.05 / 0.2 |
| Reference relaxation | FIRE | dt0=0.001, dtmax=0.02 |

## 9. ⚠ CRITICAL CAVEAT — position-based solvers are missing the outer inertial loop

**All PBD/XPBD/Projective results in this report are WITHOUT inertia and WITHOUT heavy-ball momentum.** They measure pure constraint projection (Gauss-Seidel/Jacobi iteration), not real Projective Dynamics. See labbook Session 2 for details.

### The architectural error

The point of implicit-Euler / position-based methods (PD, XPBD) is to do fast **linear sub-steps within one global inertial step**. The outer loop carries real dynamics (inertia), and the inner loop often uses heavy-ball momentum acceleration. Position-based and force-based should not be confused — the linear sub-steps are local smoothers that cannot by themselves solve nonlinear problems (rotations, dihedral torsions), but they're much cheaper because they don't require costly O(n²) long-range interactions (Coulomb, PME). Typically only 1-2 PD linear sub-steps per global nonlinear inertial step is sufficient.

Three bugs in our `step_xpbd` (raff.rs:1115-1146), identified by comparison with FireCore's `ProjectiveDynamics_d::run_LinSolve` (FireCore/cpp/common/math/ProjectiveDynamics_d.cpp:686-801):

1. **No outer-loop inertia.** When `cdamp=0` (used for all position-based solvers in the benchmark), the predict step `x += v*dt` is **skipped entirely** and velocity is always zero. Our "PD" is just repeated constraint projection with zero momentum — not Projective Dynamics. FireCore always does `ps_pred = points + vel*dt + forces*dt²/m`.

2. **No heavy-ball momentum in the inner solver.** FireCore's `updateIterativeMomentum` applies `p_{k+1} = p'_k + bmix·d_k` (bmix ramps 0→0.75 via `SmartMixer`). Our `solve_projective_jacobi` does plain Jacobi with no momentum.

3. **`cdamp` semantics are wrong.** In FireCore, `Cdrag=0` = no drag (full inertia). In our code, `cdamp=0` = no velocity at all (kill inertia). The velocity update `v = (x_new - x_old) * (cdamp/dt)` with cdamp=0 gives v=0. Should be `v = (x_new - x_old) / dt` always.

### Impact on results

- The "2 steps to convergence" for PBD on CH4 is **2 projection sweeps, not 2 inertial steps**. Real PD with inertia would behave differently.
- The comparison with FIRE was **unfair** — FIRE has inertia, our "PD" doesn't.
- The poor scaling of position-based solvers on large molecules may be because without inertia, the inner solver does all the work (16 iters), whereas with inertia, 1-2 inner iters per outer step might suffice.
- **All position-based results in §3-§7 must be re-measured after fixing the outer loop.**

### Reference implementation

FireCore/cpp/common/math/ProjectiveDynamics_d.h + .cpp:
- `run_LinSolve` — outer loop: predict → solve → corrector (line 686-801)
- `updateIterativeMomentum` — inner loop with heavy-ball momentum + SmartMixer (line 461-503)
- `SmartMixer` — bmix scheduling: 0 for first 3 iters, 0.75 after (line 43-59)
- `updateJacobi_fly` / `updateGaussSeidel_fly` — nonlinear variant (recompute RHS each iter)

## 10. Artifacts

- **CSVs**: `debug/raff_bench/{molecule}_{distortion}_{solver}.csv` (180 files, per-step step/rmsd/max_f/n_evals)
- **PNGs**: `debug/raff_bench/{molecule}_{distortion}_{plot}.png` (46 files)
  - `force_vs_step.png` — max|F| vs macrostep (PRIMARY)
  - `force_vs_evals.png` — max|F| vs soft evals (PRIMARY, cross-solver)
  - `rmsd_vs_step.png` — RMSD vs macrostep (SECONDARY — plateaus for free-DOF molecules)
  - `summary_steps_to_target.png` — bar chart overview
- **Plotting script**: `scripts/plot_raff_bench.py`
- **Benchmark binary**: `crates/libs/molff/src/bin/raff_bench.rs`
- **FIRE implementation**: `crates/libs/molff/src/raff.rs` (`step_fire`, `FireState`)

## 11. Next steps

1. **Fix the outer inertial loop in `step_xpbd`** (CRITICAL) — always predict, always update velocity, separate cdamp from predict/corrector. Add dot(v,F)<0 velocity reset for relaxation.
2. **Add heavy-ball momentum to the inner Jacobi/GS solver** — port FireCore's SmartMixer (bmix 0→0.75).
3. **Re-run benchmark with proper PD** — compare with/without inertia, 1-2 inner iters vs 16.
4. **Add torsion term to RAFF** — the H2O2 D3a result shows the port model has no dihedral constraint.
5. **Test on real molecules** (benzoic acid dimer, surface adsorbates).
6. **Consider the "fly" variant** — recompute RHS each iteration (nonlinear), may converge better for port model rotations.
