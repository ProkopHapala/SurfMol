---
type: report
title: "raff_reactive bring-up — verification results + methane assembly finding"
description: FD parity, grid parity, invariances, facing-pair E(r) all pass; point-H methane assembly reached 3/4 bonds (metastable ungated tail); resolved by making H a 1-port atom — docking test now 4/4 green, random-start kept as --ignored diagnostic.
tags: [raff, rarff2d, reactive-forcefield, debug, methane, assembly, port-gate]
timestamp: 2026-10-09
---

# raff_reactive bring-up log

Implementation per `notes/designs/2026-10-09_raff_reactive_3d_inventory.md` §3.1.
Files: `molff/src/raff_reactive.rs` (new), `spacc/src/uniform_grid.rs` (Grid3, new),
`raff.rs` (integrate_md factor-out, fd_check `*_with` closure variants, `quat_rotate` pub(crate)),
`rarff2d.rs` (`support_max` extracted), `tests/test_raff_reactive.rs` (new).

## Verification results (cargo test -p molff --test test_raff_reactive -- --nocapture)

| Test | Result | Numbers |
|---|---|---|
| FD parity (6-atom C/H cloud, eps=1e-5) | PASS | max rel force err **1.10e-9**, torque **4.91e-9** — §3.1 equations confirmed, no sign errors |
| Facing sp3 E(r) | PASS | E(r0=1.54) = **−1.000000000** (Y=1); min located at 1.55±0.02; anti-facing E ≥ 0 over scan; tilt +10° about z → τ_j.z = −2.496 (restoring) |
| Grid parity (150 atoms, domain smaller than cloud → clamping exercised) | PASS | \|ΔE\|=2.8e-13, max\|ΔF\|=2.8e-14, max\|Δτ\|=1.4e-15, max\|Δeatom\|=1.1e-14; npair identical (11175/11175) |
| Invariances | PASS | \|ΣF\|=1.9e-14, \|Σ(x×F+τ)\|=7.7e-13 |
| Methane assembly | **FAIL (3/4 bonds)** | see below |

## Methane assembly — diagnostic record

Setup per spec: C sp3 at origin + 4 point-H random on 2.0 Å sphere,
`step_reactive_md` dt=0.02, cdamp=rot_damp=0.9, flim=20, 3000 steps.
Artifacts: `debug/raff_reactive/methane_traj.xyz`, `methane_energy.tsv`.

- E: −0.397 → −3.002 (monotone decrease overall, still ~−1e-4/step tail).
- Emerged bonds (E<−0.3): **3 of 4**, each E_pair ≈ −1.00, r = 1.136–1.151 Å (spec 1.14±0.05).
- Bound-pair angles: 109.5°, 110.9°, 109.7° — well within 109.47±5°.
- H3: d(C,H3) = 2.00 → 1.75 (attracted) → **monotone ejection to 2.33 Å**, still receding at step 3000.

### Is it kinetics? No — converged metastable minimum

Same seed, 12000 steps: E converges at −3.0033 with max|F|=5.7e-4, max|τ|=3e-4.
H3 never docks. Mechanism: H3 sits ~33° off H1's occupied site, ~76° from the
free port's cone. For a point-H partner the gate is one-sided (`Y=Σci⁴`, ci>0);
at 76°, ci⁴≈3e-3 gives E_gate ≈ −5e-4 while the ungated wall contributes +8e-3 →
net repulsion ejects it; once beyond ~2.5 Å forces → ~0 and it freezes
(cdamp=0.9). This is a **genuine local minimum of the energy landscape**
(greedy sequential docking + narrow one-sided cone), not a bug — FD parity
confirms forces/torques are exact.

### Seed sensitivity (diagnostic, seeds NOT kept)

- seed 0xB5297A4D: 3/4 bonds (kept — most informative outcome)
- seed 0x11111111: 2/4 bonds
- seed 0x77777777: 2/4 bonds

Random-sphere starts produce 2–3 bound H in 3000 damped steps; stragglers
get marooned. The physics itself (gate shape, pair_math) is verified — the
failing part is the *assembly basin width under gradient-descent-only dynamics*.

## Resolution: H is now a 1-port atom (user decision, post-point-H)

`set_1port(i, dir)` added to `RaffTopology` (nport=1, normalized body-frame
dir, fails loud on degenerate). H = 1-port everywhere: FFI type 3, viewer `h`,
demo scenes, test `make_topo`, `set_reactive_inertia` gives it rotational
inertia. `set_point` kept for true point probes. Rationale (user): the bond
distance must live in the H atom's directionality — C–H binds only when BOTH
ports face each other `(ci·cj·cij)^4`, giving emergent valence saturation;
any other approach direction sees only the ungated e² wall and repels.
Analogue of FireCore's cap sites (`apos + h*lcap`). vdW steric wall at a
separate radius was explored then **deferred** — a `max(0,1−Y)`-style gate
mask adds a tuning-sensitive kink; a clean spherical-Morse/C·r⁻⁶ tail can be
added later without touching the bonded term.

### Consequences observed

- `test_methane_assembly` (now a **docking** test: tetrahedral sites + 0.15 Å
  jitter, port-first quats): **4/4 bonds, GREEN**, r=1.1556 Å, all 6 angles
  109.471°, E=−3.949. FFI python self-check mirrors it (conv in 472 steps).
- `test_methane_random_start` (`--ignored` diagnostic): random dirs +
  port-first quats → **2/4 docked**. Out-of-cone H's get ejected by the wall —
  that IS directional bonding. Two new dead zones vs point-H: (a) H outside
  all 4 port cones sees Y=0 regardless of its own orientation; (b) an H whose
  port points fully away feels zero torque and can never reorient. Physically
  a radical's lobe would rehybridize toward the substrate — a future
  "adiabatic/free port" relaxation for nport=1 atoms could model it.
- Cloud demo (random H quats): E +27.2 → −11.6, 12 bonds in 2000 steps —
  random-orientation H docks slower than point-H did (fewer free captures).
- Bug found via pair_Er plot: `quat_align` antiparallel fallback used
  `Quat4d(1,0,0,0)` = 180° about x-axis — **leaves +x invariant**. Fixed in all
  copies (test/demo/viewer/py): 180° about an axis ⊥ a.
