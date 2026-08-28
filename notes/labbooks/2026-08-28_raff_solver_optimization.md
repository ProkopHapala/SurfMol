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

### 2026-08-28 — Session 3: Implemented proper PD (outer inertia + heavy-ball momentum)

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
