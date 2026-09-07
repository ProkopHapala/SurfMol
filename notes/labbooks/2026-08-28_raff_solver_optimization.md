# RAFF Solver Systematic Optimization — Labbook

## Goal

Systematically optimize the position-based solvers (PBD, XPBD, Projective) and force-based solvers (FIRE) in RAFF to achieve fast geometry relaxation. The benchmark report (`notes/reports/2026-08-28_raff_solver_benchmark_report.md`) established the baseline. This labbook tracks the optimization work.

## Baseline summary (2026-08-28)

See `notes/reports/2026-08-28_raff_solver_benchmark_report.md` for full data.

**Best results so far:**
- Small molecules (CH4, water, H2O2): PBD-or1.9 dt=0.05 → 2-4 steps to T2 (|F|<0.1 eV/Å)
- Large molecules (tree100): Projective dt=0.2 → 15-50 steps to T2
- Force-MD: FIRE dtmax=0.02 → 77-314 steps to T2 (2× faster than damped MD)

**Main bottlenecks:**
1. PBD-or1.9 unstable at dt≥0.1 on stiff molecules (over-relaxation + large dt diverges)
2. Position-based per-step cost scales poorly with atom count (2762-5465 µs/step on tree100 vs 13 µs/step for FIRE)
3. Damped MD doesn't converge to T1 on tree100 in 10k steps

## Timeline

### 2026-08-28 — Session 1: FIRE implementation + benchmark fixes

**What was done:**
- Implemented FIRE (Bitzek et al. 2006) in `raff.rs`: `FireState` struct + `step_fire` function
- Bug: initially used old velocity for `dot(v,F)` — FIRE requires the **updated** velocity. Fixed.
- FIRE diverged with dtmax=0.1 (Adiabatic orientation snaps pump energy into momentum). Switched to Dynamic orientation mode + dtmax=0.01-0.02 → stable.
- Fixed benchmark reference: was using input geometry (wrong for free-DOF molecules). Now uses FIRE-relaxed geometry from same perturbed state.
- Fixed convergence thresholds: was |F|<1.6e-5 (machine precision). Now T2=|F|<0.1, T1=|F|<1e-3 (1 meV/Å).
- Added dt sweep: PBD/XPBD/Projective at dt=0.05, 0.1, 0.2.
- Reduced max steps from 200k to 10k. Full benchmark now completes in <1 minute.

**What happened (key numbers):**
- FIRE converges: CH4/D1 77 steps to T2 (dtmax=0.02), tree100/D2 314 steps
- PBD-or1.9 dt=0.05: 2 steps on CH4, 138 steps on tree100/D2
- PBD-or1.9 dt≥0.1: DIVERGES on CH4 (E→158), tree20, tree100
- Projective dt=0.2: 4 steps on CH4, 50 steps on tree100/D2 — **best for large molecules**
- Damped MD: does NOT converge to T1 on tree100 in 10k steps

**What it means:**
- PBD's over-relaxation (or=1.9) is powerful but fragile — only stable at small dt on stiff molecules
- Projective's global solve is more stable and benefits from large dt — best for large molecules
- FIRE is the proper force-MD baseline (2× faster than damped MD) but still 10-60× slower than PBD on small molecules
- The per-step cost of position-based solvers (16 iterations) is the wall-time bottleneck on large molecules

**Dead ends:**
- FIRE with dtmax=0.1 diverges (rotational DOFs too stiff). dtmax=0.02 is the practical limit.
- FIRE with Adiabatic orientation mode diverges (discontinuous quaternion snaps). Must use Dynamic mode.

### 2026-08-28 — Session 2: CRITICAL — position-based solvers missing outer inertial loop + heavy-ball momentum

**User insight:** The point of implicit-Euler / position-based methods (PD, XPBD) is to do fast **linear sub-steps within one global inertial step**. The outer loop has real dynamics (inertia), and the inner loop often uses heavy-ball momentum acceleration. Position-based and force-based should not be confused — the linear sub-steps are local smoothers that cannot by themselves solve nonlinear problems (rotations, dihedral torsions), but they're much cheaper because they don't require costly O(n²) long-range interactions. Typically only 1-2 PD linear sub-steps per global nonlinear inertial step is sufficient.

**Investigation:** Compared our `step_xpbd` (raff.rs:1115-1146) with FireCore's `ProjectiveDynamics_d::run_LinSolve` (FireCore/cpp/common/math/ProjectiveDynamics_d.cpp:686-801).

**Three critical bugs found:**

1. **No outer-loop inertia.** Our `step_xpbd` line 1122-1127:
   ```rust
   if cfg.cdamp > 0.0 {    // cdamp=0 in benchmark → SKIPPED
       x += v*dt;          // no predict, no inertia
   }
   ```
   When `cdamp=0` (used for all position-based solvers in the benchmark), the predict step is **skipped entirely** and velocity is always zero. Our "PD" is just repeated constraint projection with zero momentum — **not Projective Dynamics at all**. It's plain Gauss-Seidel/Jacobi iteration.

   FireCore's reference (line 697-703):
   ```cpp
   ps_pred[i] = points[i] + vel[i]*dt + forces[i]*(dt²/m);  // ALWAYS predict
   ```

2. **No heavy-ball momentum in the inner solver.** FireCore's `updateIterativeMomentum` (line 461-503) applies heavy-ball momentum:
   ```cpp
   double bmix = mixer.get_bmix(i);  // SmartMixer: 0 for first 3 iters, then 0.75
   p_{k+1} = p'_k + bmix * d_k;      // d_k = p_k - p_{k-1}
   ```
   Our `solve_projective_jacobi` does plain Jacobi with no momentum. The `SmartMixer` (line 43-59) ramps bmix from 0 to 0.75 after 3 iterations.

3. **`cdamp` semantics are wrong.** In FireCore, `Cdrag=0` means **no drag** (full inertia). In our code, `cdamp=0` means **no velocity at all** (kill inertia). These are opposite. The velocity update (line 1139) `v = (x_new - x_old) * (cdamp / dt)` with cdamp=0 gives v=0. It should be `v = (x_new - x_old) / dt` always, with damping applied separately.

**What it means:**
- All previous benchmark results for PBD/XPBD/Projective are **without inertia** — they're measuring pure constraint projection, not real PD.
- The "2 steps to convergence" for PBD on CH4 is misleading — it's 2 projection sweeps, not 2 inertial steps. Real PD with inertia would behave differently (possibly better for nonlinear problems, possibly needing more outer steps but fewer inner iters).
- The comparison with FIRE was unfair — FIRE has inertia, our "PD" doesn't.
- The poor scaling of position-based solvers on large molecules may be because without inertia, the inner solver has to do all the work (16 iters), whereas with inertia, 1-2 inner iters per outer step might suffice.

**Reference:** FireCore/cpp/common/math/ProjectiveDynamics_d.h (class def) + .cpp (implementation). Key functions:
- `run_LinSolve` — outer loop: predict → solve → corrector
- `updateIterativeMomentum` — inner loop with heavy-ball momentum + SmartMixer
- `updateJacobi_lin` / `updateGaussSeidel_lin` — linear sub-steps
- `updateJacobi_fly` / `updateGaussSeidel_fly` — nonlinear (recompute RHS each iter)
- `SmartMixer` — bmix scheduling (0 for first 3 iters, 0.75 after)

### Next session — TODO

1. **Fix the outer inertial loop in `step_xpbd`** (CRITICAL)
   - Always do predict: `x_pred = x + v*dt` (even when cdamp=0 for relaxation)
   - Velocity update: `v = (x_new - x_old) / dt` (not multiplied by cdamp)
   - Separate `cdamp` (velocity damping, 0=no drag) from the predict/corrector logic
   - For pure relaxation: use `cdamp=0` (full inertia) + dot(v,F)<0 velocity reset (like the inertial-reset solver)

2. **Add heavy-ball momentum to the inner Jacobi/GS solver**
   - Port FireCore's `SmartMixer`: bmix=0 for first 3 iters, then 0.75
   - `p_{k+1} = p'_k + bmix * (p_k - p_{k-1})`
   - Store momentum `d_k` between iterations

3. **Re-run benchmark with proper PD** — compare:
   - PD with inertia + heavy-ball (proper PD) vs PD without (current)
   - 1-2 inner iters per outer step (proper PD) vs 16 inner iters (current)
   - This should dramatically reduce per-step cost on large molecules

4. **Consider the "fly" variant** — FireCore's `updateJacobi_fly` recomputes the RHS each iteration (nonlinear), which may converge better for the port model's nonlinear rotations. Currently our `solve_projective_jacobi` linearizes r_arm once per outer step.

### 2026-08-28 — Session 3: Attempted proper PD (results invalidated by Session 4 audit)

> **INVALIDATED:** These runs used `cdamp=0.0`, but RAFF defines `cdamp` as a velocity-retention factor (`0` kills velocity, `1` preserves it). Therefore the runs labeled `PD+inertia` still discarded velocity after every outer step. Also, with `iters=4`, `bmix_istart=3`, and mixing disabled on the last iteration, heavy-ball was never active. The numerical tables below remain useful as projection-only baselines, not as measurements of inertial PD or four-step heavy-ball PD.

**What was done:**
- Fixed `step_position_based` (raff.rs): always predict `x += v*dt` (when `pd_inertia=true`), always update velocity `v = (x_new - x_old)/dt` (not multiplied by cdamp), optional `vel_reset` (dot(v,F)<0 → v=0 for relaxation).
- Added heavy-ball momentum to `solve_projective_jacobi`: `p_{k+1} = x_new + bmix * d_k` where `d_k = p_k - p_{k-1}`. Ported FireCore's `SmartMixer` (bmix 0→0.75, start at iter 3, bmix=0 on first/last iter).
- Added `RaffConfig` fields: `pd_inertia`, `vel_reset`, `bmix_start`, `bmix_end`, `bmix_istart`, `bmix_iend`.
- Updated benchmark with new `PosPD` solver configs: Projective + XPBD with inertia + heavy-ball, varying dt (0.05/0.1/0.2) and inner iters (2/4/8).

**What happened (key numbers, steps to T2 = |F|<0.1 eV/Å):**

| Molecule | Distortion | FIRE | PBD legacy | Proj legacy dt=0.2 i16 | **PD-Proj dt=0.2 i4** | PD-Proj dt=0.1 i4 |
|---|---|---|---|---|---|---|
| CH4 | D1 random | 77 | **2** | 4 | 4 | 10 |
| tree20 | D1 random | 117 | **6** | 7 | 12 | 25 |
| tree20 | D2 stretch | 130 | **25** | 23 | 34 | 62 |
| tree100 | D1 random | 179 | 36 | **15** | 32 | 59 |
| tree100 | D2 stretch | 314 | 138 | **50** | 110 | 205 |
| tree100 | D3a dihedral | 284 | 179 | **46** | 119 | 201 |

**Wall time comparison (tree100/D2_stretch, the hardest case):**

| Solver | Steps to T2 | Wall time (µs) | µs/step |
|--------|------------:|----------------:|--------:|
| FIRE | 314 | 4,094 | 13 |
| PBD legacy i16 | 138 | 393,369 | 2,849 |
| Proj legacy i16 dt=0.2 | 50 | 279,108 | 5,582 |
| **PD-Proj i4 dt=0.2** | 110 | 177,910 | 1,617 |
| PD-Proj i4 dt=0.1 | 205 | 327,085 | 1,595 |
| PD-Proj i2 dt=0.1 | 282 | 272,947 | 968 |

**What it means:**
1. **Proper PD with inertia is NOT faster than legacy projection-only in steps.** On CH4, legacy Proj dt=0.2 i16 takes 4 steps; PD-Proj dt=0.2 i4 also takes 4 steps. On tree100/D2, legacy takes 50 steps, PD-Proj takes 110. The outer inertia doesn't help with step count — it actually hurts because the velocity reset kills momentum on the stiff port system.

2. **But proper PD is faster in wall time on large molecules.** PD-Proj i4 dt=0.2 on tree100/D2: 178ms vs 279ms for legacy i16. The 4× fewer inner iters (4 vs 16) more than compensates for the 2× more outer steps (110 vs 50). Per-step cost drops from 5582 µs to 1617 µs.

3. **Heavy-ball momentum helps marginally.** PD-Proj i4 vs i2: 110 vs 282 steps (tree100/D2). The heavy-ball momentum in the inner loop accelerates the Jacobi convergence, so 4 iters with momentum ≈ 8 iters without. But 2 iters is too few.

4. **PBD-compliance or=1.9 legacy is still the step-count champion** on small/medium molecules (2-25 steps). The over-relaxation is very aggressive. But it's unstable at dt≥0.1 and doesn't benefit from inertia.

5. **The velocity reset (dot(v,F)<0 → v=0) may be too aggressive for PD.** The port system is very stiff, so the velocity often points uphill after the inner solve. Killing velocity entirely loses the inertial benefit. FireCore doesn't use velocity reset — it uses damping (Cdrag=0.05) instead. **Next: try PD with damping instead of velocity reset.**

**Dead ends:**
- PD with vel_reset + full inertia (cdamp=0) doesn't beat legacy projection-only in step count. The velocity reset is too aggressive for the stiff port system.
- PD-XPBD with inertia is worse than legacy XPBD (117 vs 42 steps on tree20/D3a). XPBD's Gauss-Seidel already converges well without inertia.

**Next steps:**
1. Try PD with damping (cdamp=0.95) instead of vel_reset — like FireCore's Cdrag=0.05
2. Try PD without vel_reset and without damping (full inertia, no reset) — pure implicit Euler
3. Try the "fly" variant (recompute RHS each inner iter) — may help with nonlinear rotations
4. Investigate why PBD-or1.9 legacy is so fast — is the over-relaxation doing something the inertia should do but doesn't?

### 2026-08-28 — Session 4: GPT-5.6 solver/performance audit

**Correctness findings (must precede further tuning):**

1. **The `PD+inertia` benchmark still killed outer velocity.** `trace_pos_ext` set `cdamp=0.0`; `step_position_based` computes the corrector velocity and then multiplies it by `cdamp`. In RAFF, `cdamp` is a retention factor, so full inertia is `cdamp=1.0`, not zero. Session 3 therefore did not measure inertial PD.
2. **The four-iteration heavy-ball cases had zero heavy-ball steps.** Mixing is disabled on the first and last iteration. With `bmix_istart=3` and `iters=4`, iterations 0–2 have zero mixing and iteration 3 is the last iteration, also zero. Session 3's `i4` results do not measure heavy-ball acceleration.
3. **`n_evals` is not comparable across solver families.** The CSV currently records one evaluation per outer step, while a PD step performs multiple local/global sweeps, repeated Wahba orientation solves, and final force evaluations. `force_vs_evals` therefore understates position-solver work and must be relabeled or replaced by explicit counters (`n_soft`, `n_local`, `n_linear`).
4. **The inner Projective solver does redundant work and allocates in the hot loop.** An owned-port Jacobi pass writes `x_new` and is then completely overwritten by the combined owned+incoming pass. `x_new`, `a_diag`, and `b_incoming` are allocated every inner iteration; `momentum` is allocated every outer step. This explains much of the extreme per-step wall-time gap.
5. **The outer position-based step does not yet implement the intended IMEX split.** It predicts only `x+v·dt`; long-range/nonbonded force is not evaluated once and added as `F_soft·dt²/m`. The separate `step_proximal` path is unfinished and currently treats port forces as both soft and hard. A real comparison with force methods requires one expensive soft-force evaluation per outer step and cheap port/contact inner sweeps.
6. **The code already recomputes port-arm projections every Jacobi iteration.** It is closer to FireCore's nonlinear `*_fly` variant than to a frozen-RHS linear solve. Calling a separate future change “fly” would be misleading; the real missing optimization is to precompute the constant matrix/diagonal and separate local projection from the global solve.

**Priority order:**

- **P0 — validate the algorithm:** run true inertia (`cdamp=1` in the present retention convention) with (a) no reset, (b) `dot(v,F)<0` reset; compare against projection-only. Test active heavy-ball separately with a schedule compatible with the chosen inner-iteration count.
- **P1 — cheap automatic parameters:** choose outer `dt` from a dimensionless stiffness scale, initially `ω_max² ≈ max_i(Σ_j k_ij/m_i + rotational contribution)` and test a small fixed set of multipliers; adapt/restart when energy or residual increases. Stop inner iterations by linear/local residual reduction instead of a fixed count.
- **P2 — remove avoidable cost:** delete the overwritten Jacobi pass; preallocate scratch buffers; precompute constant diagonal/topology contributions. This should improve wall time without changing convergence.
- **P3 — stronger global solve:** because the PD matrix is constant for fixed topology, stiffness, and `dt`, reuse sparse Cholesky/LDLT or preconditioned CG as FireCore does. For molecular trees/bounded-degree graphs this is likely the largest algorithmic speedup over Jacobi.
- **P4 — implement the actual IMEX split:** evaluate Coulomb/PME/dispersion once per outer step, form `y=x+v·dt+M⁻¹F_soft·dt²`, then solve ports and active contacts in the cheap inner loop.
- **P5 — adaptive acceleration:** heavy-ball/Chebyshev parameters should derive from spectral bounds of the preconditioned matrix and restart when the linear residual grows. Blindly fixing `bmix=0.75` is not robust across molecule size and `dt`.

**Focused experiment selected:** P0 only. It is the cheapest experiment and all higher-level parameter conclusions depend on it. Use CH4 plus `tree100/D2_stretch`, compare true inertia with/without reset and active/no heavy-ball. Do not run the full 255-case sweep.

**P0 implementation and focused result:**

- `trace_pos_ext` now uses `cdamp=1.0` for inertial runs (full velocity retention in the present RAFF convention).
- Four-iteration HB cases now use `bmix=0.75` on inner iterations 1–2; first/last remain unmixed.
- Added no-reset and no-HB controls plus `RAFF_BENCH_MOLECULE`, `RAFF_BENCH_DISTORTION`, and `RAFF_BENCH_SOLVER` filters.
- Added diagnostics proving unconstrained inertial motion is preserved (`|Δx|=1.27e-16`, `|Δv|=1.02e-15`) and i4 HB is active (`|x_hb-x_plain|=2.52e-2 Å`).

`tree100/D2_stretch` release-mode results:

| Solver | T2 steps | T1 steps | Wall to T1 | Outcome |
|---|---:|---:|---:|---|
| Legacy Projective dt=.2 i16, projection-only | 50 | 251 | 267 ms | previous best Projective baseline |
| True PD dt=.1 i4, reset, active HB=.75 | **32** | 152 | 46.1 ms | best rough convergence |
| True PD dt=.1 i4, reset, no HB | 38 | **109** | **33.8 ms** | best accurate convergence / wall time |
| True PD dt=.2 i4, active HB, no reset | -- | -- | 3.24 s cap | diverged, final E=40.2 |
| FIRE dtmax=.02 | 314 | 581 | **4.14 ms** | still fastest wall time because its step is much cheaper |

**Interpretation:**

- Correct outer inertia is valuable: no-HB inertial PD reduces Projective wall time from 267 ms to 33.8 ms (**7.9×**) and T1 steps from 251 to 109 (**2.3×**) while using four rather than sixteen inner iterations.
- Velocity reset stabilizes the inertial outer loop; unrestricted dt=.2 inertia diverges.
- Fixed HB=.75 is phase-dependent: it improves T2 from 38→32 steps, but worsens T1 from 109→152. The correct automatic strategy is HB during coarse relaxation followed by residual-triggered restart/disable near the minimum, not a globally fixed mixer.
- Despite much better convergence, current PD remains ~8× slower in wall time than FIRE on tree100 because each inner iteration allocates buffers, repeats Wahba solves, and performs the redundant overwritten pass. P2 hot-loop cleanup is now the immediate wall-time priority; sparse reusable global solves remain the major algorithmic priority.

### 2026-08-28 — Session 5: Projective inner-loop structural optimization

**Baseline and parity gate:** `tree100/D2_stretch`, true PD `dt=.1`, `i4`, reset, no HB: T2=38, T1=109, wall=33.8 ms. The optimization must preserve T2/T1 exactly before its timing is accepted.

**Planned behavior-preserving changes:**
1. Remove the first owned-port Jacobi pass because its entire `x_new` result is overwritten by the subsequent combined pass.
2. Precompute `A_ii = m_i/dt² + Σ_owned k + Σ_incoming k` and its inverse once per outer step.
3. Build the full RHS in one port traversal per inner iteration: add the owner and incoming endpoint contributions together.
4. Allocate `x_new`, RHS, diagonal, and momentum once per outer step; reuse them across inner iterations and swap position buffers.

**Result:** all six convergence diagnostics pass. The focused benchmark preserved T2/T1 exactly at 38/109, while wall time changed from 33.8 to 33.1 ms (~2%). Therefore the removed allocations and traversal were real waste but not the dominant cost. The likely dominant work is repeated adiabatic Wahba solves plus repeated final force diagnostics. Do not claim a major speedup from this cleanup.

### Session 6: automatic heavy-ball restart — failed criterion

Fixed HB=.75 helped T2 but hurt T1. Tested a zero-parameter restart criterion inside each linear solve: use HB only when the plain Jacobi correction aligns with previous inner momentum, `Σ_i (x'_i-x_i)·d_i > 0`.

**Failure:** the CH4 diagnostic produced `|x_hb-x_plain|=0` and failed, proving that the criterion rejected every HB update. In this fixed-point iteration, the Jacobi correction naturally opposes the previous displacement; it is not analogous to a physical force/velocity power test. The change was reverted. A valid automatic restart must compare the true linear residual `||b-Ax||` before/after acceleration (or use spectral bounds), not displacement alignment.

### Session 7: focused profiling and Wahba iteration budget

- Profiled `tree100/D2`, PD `dt=.1`, i4, no HB: 33.53 ms inside `step_position_based` (98.3%), 0.218 ms external force diagnostic, 0.218 ms RMSD, 0.133 ms other. Removing duplicate external diagnostics cannot matter materially.
- A fixed-state probe of 545 Wahba solves took 1.23 ms, but this likely underestimates intermediate-state cost because the warm-start quaternion is already converged.
- Callgrind collected 289M instructions but release symbols are stripped, so function attribution was unavailable; no alternate full rebuild was attempted.
- Inner-count comparison: i2 = 46/226 steps, 39.6 ms; i3 = 45/126, 29.7 ms; i4 = 38/109, 32.5–33.1 ms. **i3 is best for T1 wall time; i4 is best for outer-step count.** `dt=.15,i3` gives 38/146 and 34.9 ms, so `dt=.1,i3` remains the focused accurate optimum.

**Next experiment:** `solve_rotation_wahba` currently allows 64 shifted power iterations to quaternion delta `1e-14`. Test max 16 and `1e-12`; accept only if all convergence diagnostics pass and focused T2/T1 remain physically consistent.

**Outcome and correction:** reducing to 16 iterations produced a 3× wall-time improvement in the focused PD case, but a new bad-start diagnostic exposed a deeper correctness bug present even at 64 iterations: starting from an exactly wrong K-matrix eigenvector (CH4 central quaternion rotated 180°) leaves power iteration trapped at a high-energy stationary orientation (`E=266.7`, `max|τ|=0`). The iteration-budget change was reverted. Warm-start power iteration was replaced by deterministic cyclic Jacobi diagonalization of the full symmetric Davenport 4×4 matrix. The bad-start case now gives `E=0`, and the equilibrium, torque-residual, and seven convergence diagnostics pass.

### Session 8: dynamic versus adiabatic orientation audit

**Meaning of the two approaches:**
- **Adiabatic/Wahba:** orientation is an internal memoryless variable `q*(x)=argmin_q E(x,q)`. Every solve replaces `q`; `ω` and rotational inertia are irrelevant.
- **Dynamic:** orientation is an outer state variable. `ω ← ω + I⁻¹τdt`, `q ← exp(ωdt/2)q`; angular momentum persists between outer steps and damping/reset acts on `ω`.

**Current implementation matrix:**
- ForceMD, inertial-reset, FIRE: true torque/`ω` dynamic rotation is implemented.
- PBD/XPBD with `OrientMode::Dynamic`: direct per-constraint quaternion corrections are implemented, but there is no outer angular-velocity corrector; this is rotational PBD, not torque dynamics with memory.
- Projective with `OrientMode::Adiabatic`: exact memoryless Wahba projection is performed before and after every inner sweep.
- Projective with `OrientMode::Dynamic`: quaternion is currently frozen; no torque integration or quaternion correction occurs. Therefore true dynamic Projective rotation is missing.
- All current Projective benchmark configurations are adiabatic, so no dynamic-vs-memoryless comparison has yet been made.

**Required comparison:** add a Projective outer rotational predictor using port torque and `ω`, hold `q` fixed during the inner translational global solve, include angular power `ω·τ` in reset, and converge on both translational force and torque. Compare outer steps and total inner sweeps; CPU milliseconds are secondary.

**Implementation:** Projective `OrientMode::Dynamic` now evaluates port torque once per outer step, updates `ω` with `I⁻¹τdt`, integrates `q`, and holds that orientation fixed during the translational inner solve. Generalized-power reset uses `v·F+ω·τ` and clears both momenta. The benchmark records `max_t`, requires both force and torque thresholds, and reports total force evaluations, inner sweeps, and orientation operations.

**Focused tree100/D2 result (i4, no HB, reset):**

| Orientation mode | dt | T2 outer | T1 outer | force evals | inner sweeps | orientation ops | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| Adiabatic Davenport/Wahba | .10 | **38** | **113** | 226 | **452** | 566 | converged |
| Dynamic torque/ω | .01 | 523 | 1203 | 3609 | 4812 | 1203 | converged |
| Dynamic torque/ω | .02 | 242 | 593 | 1779 | 2372 | 593 | converged |
| Dynamic torque/ω | .05 | **109** | **251** | **753** | **1004** | **251** | converged |
| Dynamic torque/ω | .10 | -- | -- | 30000 | 40000 | 10000 | diverged, E≈1.53e3 |

**Interpretation:** adiabatic orientation is the better relaxation preconditioner on this case: ~2.2× fewer outer steps and inner sweeps than the best stable dynamic run. Dynamic orientation has physical angular memory but is limited by the explicit rotational stability scale; the stable boundary lies between dt=.05 and .1 for the nominal stiffness/inertia. The automatic dynamic timestep should therefore be based on `dt_rot ≲ c/sqrt(k_rot/I)` with a safety factor, independently of the larger implicit translational PD timestep. A multirate scheme (small rotational substeps, larger translational PD outer step) is more promising than forcing one shared dt.

### Session 9: multirate rotational subcycling — negative result

**Hypothesis:** keep the large implicit translational PD timestep (dt=0.05–0.2) but subcycle the explicit rotational dynamics at `dt_rot = dt/n_rot_substeps`. This would combine adiabatic-level translational convergence with physical angular memory, without the instability that forces single-rate dynamic orientation to use dt≤0.05.

**Implementation:** added `n_rot_substeps` field to `RaffConfig`. In `step_position_based`, the dynamic-Projective orientation block now applies `rot_damp` once per outer step, then subcycles `ω += I⁻¹τ·dt_rot` and `q ← exp(ω·dt_rot/2)⊗q` for `nsub` steps with constant torque (evaluated once per outer step — cheap). The final `ω` is identical to single-rate (sum of substep increments = τ·I⁻¹·dt); only the quaternion path differs (higher-order integration, slightly less total angular displacement: 3/4·τ·I⁻¹·dt² for nsub=2 vs 1·τ·I⁻¹·dt² for nsub=1).

**Diagnostic test:** `test_pd_rotational_subcycling_stable_at_large_dt` — CH4 with dt=0.1/sub8 (dt_rot=0.0125). PASSES (converges to |F|<0.1). Single-rate dt=0.1 diverges on CH4 too, so subcycling does extend the rotational stability limit for small molecules.

**Focused tree100/D2_stretch benchmark:**

| Variant | dt | nsub | dt_rot | T2/T1 | Outcome |
|---|---:|---:|---:|---|---|
| Single-rate dynamic | .05 | 1 | .050 | 109/251 | converged |
| Single-rate dynamic | .10 | 1 | .100 | -- | diverged (E≈1.5e3) |
| Subcycled dynamic | .05 | 2 | .025 | -- | **diverged (E≈69)** |
| Subcycled dynamic | .10 | 2 | .050 | -- | diverged (E≈815) |
| Subcycled dynamic | .10 | 4 | .025 | -- | diverged (E≈362) |
| Subcycled dynamic | .10 | 8 | .0125 | -- | diverged (E≈192) |
| Subcycled dynamic | .20 | 4 | .050 | -- | diverged (E≈3200) |
| Subcycled dynamic | .20 | 8 | .025 | -- | diverged (E≈2700) |

**Root cause analysis (debug prints with `RAFF_DEBUG_SUBCYCLE=1`):**
- Step 1 is identical between single-rate and subcycled (same initial state, same torque).
- Step 2 diverges: single-rate omega=2.031, subcycled omega=2.244 (growing).
- The subcycled quaternion rotates LESS than single-rate (3/4 of torque contribution for nsub=2).
- This causes port tips to be misaligned with the large translational step.
- The translational PD solve can't fix quaternion misalignment (it only moves positions).
- The misalignment creates a feedback loop: bad quaternion → bad port tips → large forces → large torque → larger omega → worse quaternion.

**Fundamental limitation:** the rotational and translational dynamics are coupled through the port constraints. The rotational displacement per step must be proportional to the translational displacement per step. Subcycling breaks this ratio: `Δθ ∝ dt·(3/4)` but `Δx ∝ dt`. At dt=0.01 (single-rate), both Δθ and Δx are 5× smaller than dt=0.05 — the ratio is preserved, and the system converges. At dt=0.05/sub2, Δθ is reduced but Δx is not — the ratio is broken, and the system diverges.

**Conclusion:** multirate rotational subcycling alone does NOT extend the dynamic PD stability limit for relaxation. The rotational and translational timesteps are coupled through the port constraints. The `n_rot_substeps` field is kept for physical simulations where accurate rotational integration is desired, but it does not help with relaxation convergence.

**Next direction:** the promising approach is NOT subcycling but rather a **quasi-adiabatic** scheme: use adiabatic (Wahba) orientation for the inner translational solve (fast convergence) but track a dynamic quaternion separately for the outer-loop prediction (physical angular memory). This decouples the rotational accuracy from the translational stability without breaking the displacement ratio.

### Session 10: inner-coupled rotational Jacobi — breakthrough

**Key insight (user correction):** the inner Jacobi loop IS the substep for both translation and rotation. The port traversal accumulates both translational force and torque for each atom — updating both DOFs is cheap once forces are accumulated. There should NOT be a separate rotational sweep or outer torque integration; the inner loop handles both together, preserving the coupled displacement ratio.

**Implementation:**
1. **Removed outer torque integration** for dynamic Projective. The outer step only predicts: `x += v·dt` (translation) and `q ← exp(ω·dt/2)⊗q` (rotation from carried angular velocity). No torque evaluation in the outer step.
2. **Inner Jacobi loop** now accumulates BOTH translational RHS and torque in ONE port traversal. After the traversal, updates both `x_new[i] = rhs[i] / inv_diag[i]` AND `q[i] ← exp(δθ/2)⊗q[i]` where `δθ = τ / (I/dt² + K_rot)`. K_rot = Σ_s k_s |r_arm|² is the rotational stiffness diagonal.
3. **Corrector** computes both `v = (x_new - x_old)/dt` and `ω = 2·imag(q_new ⊗ q_old⁻¹)/dt` from the total position/quaternion change.
4. **Removed `n_rot_substeps`** — not needed; the inner loop is the substep.

**Focused tree100/D2_stretch benchmark:**

| Variant | dt | i | T2 | T1 | wall |
|---|---:|---:|---:|---:|---:|
| Adiabatic (best T2) | .10 | 4 | 32 | 151 | 23ms |
| Adiabatic (best T1) | .10 | 4 | 38 | 113 | 17ms |
| Old dynamic (outer-only) | .05 | 4 | 109 | 251 | 5.3ms |
| Old dynamic (outer-only) | .10 | 4 | diverged | — | — |
| **New dynamic (inner-coupled)** | **.10** | **4** | **34** | **193** | **5.7ms** |
| **New dynamic + HB** | **.10** | **8** | **30** | **119** | **5.7ms** |
| New dynamic | .15 | 4 | diverged | — | — |
| FIRE | — | — | 299 | 578 | 4.1ms |

**Results:**
- dt=0.1 now CONVERGES (was divergent with outer-only dynamic).
- Dynamic i8+HB beats adiabatic in T2 steps (30 vs 32) at 4× lower wall time (5.7ms vs 23ms).
- T1 is competitive (119 vs 151/113).
- Stability limit moved from dt=.05–.1 to dt=.1–.15.
- FIRE still fastest wall (4.1ms) but 10× more steps (299 vs 30).

**Why it works:** the inner loop's coupled translation+rotation update preserves the displacement ratio. Each inner iteration moves both x and q by amounts proportional to the same dt. The outer step only predicts (carries momentum), the inner loop corrects both DOFs together. This is the same pattern as FireCore's `updateJacobi_fly` — one atom traversal, both DOFs updated.
