---
type: labbook
title: RARFF-2D debug labbook — dynamics demos, evaporation, GD pathologies
description: Debugging session for the rod-free reactive 2D FF — from isolated energy plots to actual dynamics. What failed, why (with numbers), what it means for the model.
tags: [labbook, debug, rarff2d, dynamics, gradient-descent, basin]
timestamp: 2026-10-03
---

# RARFF-2D debug labbook

Session goal: actual dynamics/relaxation with the gated-Morse 2D FF (`molff/src/rarff2d.rs`) — repair demo + self-assembly + interactive viewer. See [`/doc/topical_audit/rarff2d.md`](/doc/topical_audit/rarff2d.md) for the module map.

## What was tried / what happened

1. **Damped MD (`step_md`, dt=0.02, cdamp=0.92) on corrupted naphthalene** → **evaporation**: spurious atom got F≈71 eV/Å kick, all atoms drifted past rcut, `eatom→0`, E stuck ~−0.6. *Meaning:* inertia + strong repulsion walls turns the system into a hot gas; nothing pulls escapees back (basin problem).
2. **Switched to overdamped GD (`step_gd`, trust-region `dmax`/`dphi_max`)** → first run looked better (E≈−8.5) but `maxF≈12` persisted → diagnosed as frustrated state, then realized E was *worse* than a shorter run.
3. **E increased under capped GD — impossible** → found real bug: demo loop called `step_gd` 10× per `eval` (stale forces catapulted atoms). Eval-every-step fixed it; E now descends monotonically. *Lesson: capped GD monotonicity is a cheap invariant check.*
4. **Orientation limit cycle**: `maxT≈10` frozen → `dphi_max=0.4` let orientations overshoot the gate edge and flip back forever. Fix: `dphi_max (0.05) << gate w (0.3)`, `inv_i=2`. → Clean scene converged E=−32.4, all atoms bonded.
5. **Atom pushed *inward* during corruption gets ejected**: corrupting toward a neighbor lands inside the repulsive wall → kicked out past the short attraction tail → orphan flagged by `eatom`. Corrupting *outward* stays in-basin and repairs. → **Basin problem**: b=1.8 tail can't recapture; motivates weak long-range tail (open issue 4).
6. **Residual maxF≈8–10 at "converged" states**: edge atoms torn between ring-site pull (a=2) and pair bonds — genuine frustration/capped-grind, E stable. Not solver noise.
7. **Self-assembly**: random 14 atoms + arena → seed 7 & 3 form hexagons (ring closure = late sharp E-drop on E(step) curve), seed 2 = branched tree. `eatom` ↔ coordination exactly (−0.48/−0.97/−1.47).
8. **Live viewer** (`apps/editor/src/bin/rarff2d_view.rs`): works first try; `field_at` probe field uses gate≡1 probe (single-sided) — note vs mutual product gate.

## Open issues → topical audit §Open issues

w tuning (ports too narrow ~11°), mutual-gate pairwise verification (fixed+moving atom, no rings), 1 Å grid acceleration (U-Net pixel equivalence), weak long-range tail, probe-gate option in `field_at`, batch refiner API / OpenCL later.
