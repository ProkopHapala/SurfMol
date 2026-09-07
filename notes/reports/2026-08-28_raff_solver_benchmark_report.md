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

## 9. ⚠ Historical-result caveat — original inertial-PD and heavy-ball labels were invalid

**All original PBD/XPBD/Projective results in this report are projection-only baselines.** The first follow-up runs labeled `PD+inertia` were also not valid inertial-PD measurements: their benchmark configuration passed `cdamp=0.0`, while RAFF defines `cdamp` as a velocity-retention multiplier (`0` kills velocity, `1` preserves it). Those runs therefore discarded velocity after every outer step. In addition, the `i4` cases used `bmix_istart=3` while mixing was disabled on the last iteration, so heavy-ball was never active. The focused corrected subset below supersedes those labels; see labbook Sessions 2–4.

### The architectural error

The point of implicit-Euler / position-based methods (PD, XPBD) is to do fast **linear sub-steps within one global inertial step**. The outer loop carries real dynamics (inertia), and the inner loop often uses heavy-ball momentum acceleration. Position-based and force-based should not be confused — the linear sub-steps are local smoothers that cannot by themselves solve nonlinear problems (rotations, dihedral torsions), but they're much cheaper because they don't require costly O(n²) long-range interactions (Coulomb, PME). Typically only 1-2 PD linear sub-steps per global nonlinear inertial step is sufficient.

Three bugs in our `step_xpbd` (raff.rs:1115-1146), identified by comparison with FireCore's `ProjectiveDynamics_d::run_LinSolve` (FireCore/cpp/common/math/ProjectiveDynamics_d.cpp:686-801):

1. **No outer-loop inertia.** When `cdamp=0` (used for all position-based solvers in the benchmark), the predict step `x += v*dt` is **skipped entirely** and velocity is always zero. Our "PD" is just repeated constraint projection with zero momentum — not Projective Dynamics. FireCore always does `ps_pred = points + vel*dt + forces*dt²/m`.

2. **No heavy-ball momentum in the inner solver.** FireCore's `updateIterativeMomentum` applies `p_{k+1} = p'_k + bmix·d_k` (bmix ramps 0→0.75 via `SmartMixer`). Our `solve_projective_jacobi` does plain Jacobi with no momentum.

3. **The benchmark used the wrong `cdamp` convention.** FireCore expresses drag as a damping amount, while RAFF's `cdamp` is a velocity-retention multiplier (`0` kills velocity, `1` preserves it). The corrector itself now computes `v = (x_new - x_old)/dt`, after which RAFF applies the retention factor. Therefore true undamped inertia requires `cdamp=1`, not zero.

### Impact on results

- The "2 steps to convergence" for PBD on CH4 is **2 projection sweeps, not 2 inertial steps**. Real PD with inertia would behave differently.
- The comparison with FIRE was **unfair** — FIRE has inertia, our "PD" doesn't.
- The poor scaling of position-based solvers on large molecules may be because without inertia, the inner solver does all the work (16 iters), whereas with inertia, 1-2 inner iters per outer step might suffice.
- **All position-based results in §3-§7 must be re-measured after fixing the outer loop.**

### Focused corrected result (tree100 / D2 stretch)

A short follow-up corrected the retention factor (`cdamp=1.0` for full inertia), activated HB inside the four-iteration solve, and compared controls:

| Solver | T2 steps | T1 steps | Wall to T1 |
|---|---:|---:|---:|
| Legacy Projective dt=.2 i16, projection-only | 50 | 251 | 267 ms |
| True PD dt=.1 i4, reset, HB=.75 | **32** | 152 | 46.1 ms |
| True PD dt=.1 i4, reset, no HB | 38 | **109** | **33.8 ms** |
| FIRE dtmax=.02 | 314 | 581 | 4.14 ms |

Correct outer inertia improves Projective T1 step count by 2.3× and wall time by 7.9× relative to the prior Projective baseline. Fixed HB helps rough convergence but hurts accurate convergence, indicating that momentum should restart/disable automatically when the residual stops decreasing. FIRE remains faster in wall time because a PD outer step performs several local/global updates and repeated orientation solves.

### Follow-up inner-loop optimization

The redundant overwritten Jacobi pass was removed; the constant diagonal is precomputed; owned/incoming RHS contributions are fused into one port traversal; and inner buffers are reused. All convergence diagnostics pass and the focused case preserves T2/T1 exactly at 38/109. Wall time changed only 33.8→33.1 ms (~2%), proving that this waste was not the dominant bottleneck. Repeated adiabatic Wahba solves and diagnostic force evaluations are stronger suspects.

A proposed automatic HB restart based on `Jacobi correction · previous momentum > 0` failed its diagnostic: it disabled every HB update on CH4 (`|x_hb-x_plain|=0`). The change was reverted. A valid restart must use the true linear residual `||b-Ax||` (or spectral bounds), not displacement alignment.

### Orientation algorithm correction and comparison

The old adiabatic Wahba solver used warm-started power iteration. A 180° bad start can be an exact non-dominant eigenvector, leaving the solver trapped at `E=266.7` regardless of iteration count. It was replaced by deterministic cyclic Jacobi diagonalization of the full symmetric Davenport 4×4 matrix. The bad-start case now reaches `E=0`; equilibrium energy is `1.54e-29`; perturbed torque residual is `1.22e-14`.

Projective dynamic orientation was also implemented: once per outer step it evaluates torque, advances `ω`, integrates `q`, then holds `q` fixed during four translational inner sweeps. Convergence now requires both max force and max torque thresholds.

| Orientation | dt | T2 outer | T1 outer | force evals | inner sweeps | orientation ops |
|---|---:|---:|---:|---:|---:|---:|
| Adiabatic Davenport/Wahba | .10 | **38** | **113** | 226 | **452** | 566 |
| Dynamic torque/ω | .01 | 523 | 1203 | 3609 | 4812 | 1203 |
| Dynamic torque/ω | .02 | 242 | 593 | 1779 | 2372 | 593 |
| Dynamic torque/ω | .05 | **109** | **251** | **753** | **1004** | **251** |
| Dynamic torque/ω | .10 | -- | -- | 30000 | 40000 | 10000 |

Adiabatic orientation is the stronger relaxation preconditioner for tree100/D2. Dynamic orientation preserves physical angular memory but has an explicit rotational stability limit between dt=.05 and .1. **Multirate rotational subcycling was tested (Session 9) and does NOT help:** subcycling the rotation at `dt_rot = dt/n` while keeping the translational dt large causes divergence even at dt=0.05/sub2 (single-rate dt=0.05 converges). Root cause: the port constraints couple rotational and translational displacement — subcycling reduces `Δθ` without reducing `Δx`, breaking the coupled displacement ratio. **The correct fix (Session 10): inner-coupled rotational Jacobi.** The inner loop accumulates both translational RHS and torque in ONE port traversal, then updates both x and q together. No outer torque integration — the inner loop IS the substep for both DOFs. This extends the dynamic stability limit from dt=.05–.1 to dt=.1–.15, and dynamic PD at dt=0.1/i8+HB now beats adiabatic in T2 steps (30 vs 32) at 4× lower wall time (5.7ms vs 23ms).

### Reference implementation

FireCore/cpp/common/math/ProjectiveDynamics_d.h + .cpp:
- `run_LinSolve` — outer loop: predict → solve → corrector (line 686-801)
- `updateIterativeMomentum` — inner loop with heavy-ball momentum + SmartMixer (line 461-503)
- `SmartMixer` — bmix scheduling: 0 for first 3 iters, 0.75 after (line 43-59)
- `updateJacobi_fly` / `updateGaussSeidel_fly` — nonlinear variant (recompute RHS each iter)

## 10. Artifacts

- **CSVs**: `debug/raff_bench/{molecule}_{distortion}_{solver}.csv` (generated configurations vary; columns step/rmsd/max_f/n_evals)
- **PNGs**: `debug/raff_bench/{molecule}_{distortion}_{plot}.png` (46 plot files for the 15 molecule/distortion groups plus summary)
  - `force_vs_step.png` — max|F| vs macrostep (PRIMARY)
  - `force_vs_evals.png` — currently max|F| vs an outer-step proxy; **not a valid cross-family work metric** until soft/local/linear evaluations are counted separately
  - `rmsd_vs_step.png` — RMSD vs macrostep (SECONDARY — plateaus for free-DOF molecules)
  - `summary_steps_to_target.png` — bar chart overview
- **Plotting script**: `scripts/plot_raff_bench.py`
- **Benchmark binary**: `crates/libs/molff/src/bin/raff_bench.rs`
- **FIRE implementation**: `crates/libs/molff/src/raff.rs` (`step_fire`, `FireState`)

## 11. Prioritized next steps after solver audit

1. **Completed for the focused tree100/D2 case: validate true outer inertia.** With the current retention-factor convention, `cdamp=1.0` preserves velocity. Reset stabilized the tested outer loop; the no-reset `dt=.2` case diverged.
2. **Partially completed: test genuinely active heavy-ball.** Active HB improved rough T2 convergence (38→32 steps) but worsened T1 convergence (109→152). A displacement-alignment restart failed and was reverted; the next restart must monitor the true linear residual `||b-Ax||`.
3. **Make work counters physically meaningful.** Record expensive soft-force evaluations separately from local projections, linear iterations, Wahba solves, and final diagnostic force evaluations. Current `n_evals=outer_steps` makes the cross-family evaluation plot misleading.
4. **Completed, low impact: remove inner-loop waste.** Fused RHS accumulation, constant diagonal, and reused inner buffers preserve 38/109 convergence but improve this focused wall time by only ~2%. Profile orientation and diagnostic evaluations next.
5. **Prefer a reusable global solve over more Jacobi tuning.** The PD matrix is constant for fixed topology/stiffness/dt; sparse LDLT/Cholesky or preconditioned CG can be reused. FireCore already provides the reference architecture. This is especially promising for bounded-degree molecular trees.
6. **Implement the intended IMEX split.** Evaluate long-range nonbonded forces once per outer step, build `y=x+v·dt+M⁻¹F_soft·dt²`, and keep only ports/contact projections inside the cheap loop.
7. **Automate parameters from diagnostics.** The focused comparison supports `i4` for minimum outer iterations and `i3` for similar CPU work, but two sweeps are insufficient. ~~Estimate rotational stability separately (`dt_rot ≲ c/sqrt(k_rot/I)`) and use subcycling/multirate integration.~~ **Subcycling tested and rejected (Session 9); inner-coupled rotational Jacobi implemented (Session 10).** The inner loop now updates both translation and rotation in one port traversal, extending the dynamic stability limit to dt=.1–.15. Dynamic PD at dt=0.1/i8+HB beats adiabatic in T2 steps (30 vs 32) at 4× lower wall time. Stop inner iterations using the true linear residual.
8. **Add the missing torsion physics and test real molecules.** Without an explicit torsion term, dihedral benchmarks probe a null space rather than nonlinear torsional convergence.
