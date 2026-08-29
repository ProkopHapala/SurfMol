---
type: convention
title: Relaxation convergence measurement
description: How to measure and plot relaxation convergence — use final converged geometry as reference, not the initial geometry.
tags: [convention, relaxation, convergence, plotting, parity]
timestamp: 2026-08-29
---

# Relaxation convergence measurement

## The rule

**NEVER use the initial/pre-perturbation geometry as the reference for convergence.**

The initial geometry is NOT the equilibrium. The perturbed system relaxes to a **new** equilibrium that may differ from the starting geometry (due to `l0` parameters, boundary conditions, nonbonded interactions, etc.).

## Correct procedure

1. **Run the relaxation until forces converge** below a threshold (e.g. max|F| < 1e-6), NOT a fixed number of steps. The solver decides when it's done.
2. **Record positions at every step** along the trajectory.
3. **Take the FINAL converged geometry as the reference** — this is the equilibrium the system is relaxing toward.
4. **Compute displacement backward**: for each trajectory frame `k`, compute `|x_k - x_final|` — how far that frame was from the final equilibrium.
5. **Plot this backward displacement vs step** on a log scale. This shows the actual convergence rate: the curve going to zero means the system is reaching its equilibrium.

## Why not use the initial geometry?

- The initial geometry is the **perturbed** state, not the equilibrium.
- A "plateau" at non-zero displacement from the initial geometry means the system **has converged** to a different equilibrium — it does NOT mean the solver stalled.
- Using the initial geometry as reference produces misleading plots that look like the solver failed when it actually succeeded.

## Force-based convergence criterion

The solver should stop when `max|F| < threshold` (or `|residual|/|b| < threshold` for linear solvers), NOT after a fixed number of steps. This ensures:
- The run is long enough to actually converge (no premature stop).
- The run doesn't waste steps after convergence (no unnecessary computation).
- The final geometry is truly the equilibrium.

## In code

```rust
// WRONG — uses initial geometry as reference
let max_disp = max_i |pos[i] - initial_pos[i]|;

// CORRECT — run until forces converge, then use final as reference
loop {
    step();
    let max_force = compute_max_force();
    if max_force < 1e-6 { break; }
}
let final_pos = download_positions();
// Then for each trajectory frame k:
let disp_k = max_i |traj[k][i] - final_pos[i]|;
```

## In plots

The convergence plot should show:
- X axis: step number
- Y axis: `|x_step - x_final|` (displacement from final equilibrium), log scale
- The curve should go to zero (or machine precision) — that's convergence

NOT:
- `|x_step - x_initial|` — this is meaningless for convergence
