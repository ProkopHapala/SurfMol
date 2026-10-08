---
type: topical-audit
title: RARFF-2D — rod-free orientation-gated reactive pair potential (2D CPU prototype)
tags: [topic, rarff2d, reactive-forcefield, orientation-gated, morse, invAFM, cross-language]
timestamp: 2026-10-03
---

# RARFF-2D — rod-free orientation-gated reactive pair potential

Cross-implementation map for the 2D "rod-free" reactive atom forcefield prototype: pair Morse with mutual angular gating, ring entities, polynomial approximations, CPU reference + demos + interactive viewer. Motivation: invPPAFM inverse-AFM project needs a physics-based consistency score/repair loop for U-Net predicted atom geometries (atoms/bonds/rings guessed per-pixel from experimental images).

## Physics model

Pair energy (per atom pair i,j, distance r, unit bond direction h):

    E_ij = a · ( e² − 2·e·Y ),   Y = g_i·g_j

- `e = pex(-b_ij·(r−r0_ij))` — Morse-like radial factor; **repulsive e² term is ungated**, attraction `-2e·Y` requires **both** atoms oriented port-to-port.
- `r0_ij = r_cov,i + r_cov,j` (0.65 A C_ar → 1.3 A C–C); `b_ij = b_i + b_j` (pair b≈1.8 /A — steeper wall was needed for visible repulsion).
- **Gate** (V4 sine-Lorentzian): `g(Δφ) = 1/(1 + (sin(n·Δφ/2)/w)²)` — n maxima at port dirs, quadratic stiffness `g ≈ 1 − (n·Δφ/2w)²` near alignment. Replaces earlier V2 (1−cos form, flat derivative at optimum) and V1 (`cos(3φ)` literal — produced 6 minima, wrong phase).
- **Radial exp via `pex(x,N=32) = (1+x/32)³²`** — repeated-squaring polynomial exponential, natural compact support (0 for base ≤ 0), derivative `de/dr = −b·e/s`. No `exp()` in the hot loop.
- **Ring entity** `Ring2d` (hexagon/pentagon constraint): `E = −a·B(ρ−R)·(1+cos(n·Δφ))/2 + a_core·e_core²` — site attraction on the circumference annulus with `B(ρ) = (1−(ρ/h)²)⁴` compact bump (C³ at edge, `db/dρ = −8ρ/h²·(1−ρ²/h²)³`), plus poly-Morse repulsive core keeping the hole empty.
- Orientations are unit complex `zc = (cos φ, sin φ)`; torque from `dE/dφ`; analytic forces/torques throughout.

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `crates/libs/molff/src/rarff2d.rs` | active | **Core engine**: `Rarff2d` SoA (pos, zc, vel, om, force, torq, `eatom` per-atom energy = consistency map), `Rarff2dType`/`TYPE_SP2`/`TYPE_SP3`, `pair_ef`, `ring_ef`, `set_rings` (allocates ring scratch once), `eval` (O(N²) pairs + rings + arena), `step_md` (damped symplectic Euler), `step_gd` (overdamped GD w/ trust-region `dmax`,`dphi_max` — repair mode), `relax`, `fd_check`, `field_at` (read-only probe potential for E-map), `arena` confinement. Helpers `pex`, `bump`, `gate` are pub for tests/demos. |
| `crates/libs/molff/tests/test_rarff2d.rs` | active | 8 tests: pair equilibrium, FD parity, anti-node/off-axis repulsion, bend3/hexagon relax, ring+pair hexagon, ring pulls atoms to sites. All passing. |
| `crates/libs/molff/src/bin/rarff2d_demo.rs` | active | Corrupted-naphthalene repair demo (2 fused hexagons + rings, displaced atom ± spurious atom). GD relax, FD checks, CSV/XYZ dumps → `debug/rarff2d_dyn/`. |
| `crates/libs/molff/src/bin/rarff2d_assemble.rs` | active | Self-assembly demo: N random sp2 atoms in soft arena → damped MD → bonded graph; `--seed/--n/--arena/--out`; dumps traj.xyz + detected bonds (r<1.6, mutual gate Y>0.3). |
| `crates/apps/editor/src/bin/rarff2d_view.rs` | active | **Interactive viewer** in editor package (reuses `ImpostorRenderer`, `LineRenderer`, `SurfaceRenderer`, `TrackballCam`, egui): LMB drag atoms, live probe-potential E-map ground plane (`field_at`, dragged/selected atom excluded), `eatom`-colored atoms, port ticks, detected bonds, ring circles, arena. Keys: Space run, A add, N random, R reset, F/B/T/E toggles. Run: `cargo run --release -p editor --bin rarff2d_view`. |
| `scripts/plot_rarff2d_potential.py` | active | Potential survey figures → `debug/rarff2d/`: gate shapes (V1 vs V4), pair E-maps, radial cuts, bonded-pair field, ring E-map, poly-approximation 1D comparisons. |
| `scripts/plot_rarff2d_dyn.py` | active | Repair-dynamics figure → `debug/rarff2d_dyn/dyn_repair.png` (convergence + geometry + per-atom `eatom` before/after, 2 scenes). |
| `scripts/plot_rarff2d_assemble.py` | active | Assembly figure → `debug/rarff2d_assemble/assemble.png` (E(step) + final bond graph colored by `eatom`, multiple seeds). |
| `invPPAFM:export_invAFM/surfmol_forcefields.md` | reference | Private handoff doc: forcefield inventory + design rationale (invAFM side). |

## Verification status

- **FD parity**: forces & torques vs central differences ~1e-8..1e-9 (atom + ring DOFs; harness restores each perturbed DOF before next — earlier contamination bug fixed).
- **Unit tests**: 9/9 in `test_rarff2d.rs` (incl. grid AND groups parity vs O(N²), dE ~1e-13).
- **Spatial acceleration — DONE + measured** (`set_grid` / `set_groups`, mutual-exclusive):
  - `Grid2d` wraps `spacc::Buckets` (count→prefix→scatter); eval uses the **cell-pair** loop: per non-empty cell, intra-cell pairs + forward-stencil halo gathered into contiguous scratch (RRsp3 ghost-list pattern; scratch preallocated in `set_grid`, zero hot-loop allocs), scattered back with zero-skip. dx=rcut → strict 3×3 stencil.
  - `Groups2d` = **topology-free COG collision groups** (nearest group-COG within `r_join`, else spawn) + position-fit AABB + O(G²) overlap prune — the `cluster_aabb_collision.md` pattern minus the fixed-bond assumption. Stale groups never break correctness (AABBs refit each rebuild).
  - `pair_math` is the single physics source for all paths: **transcendental-free** — `r2`-early-exit before `sqrt`, `pex` squaring-approx, gates via complex powers `c`, `c²`, `c·√c` (`phase_half`/`gate_cs`) — no atan2/sin/cos. Inside-eval 68.5→**22.5 ns**; packed baseline **25.5 ns/valid-pair**.
  - **`pex_m` = the cutoff knob** (integer squaring count): `pex(x)=(1+x/2^m)^(2^m)` via `m` repeated squarings (no pow); exact zero at `x=−2^m` → **compact support `r0+2^m/b`, no taper needed** (taper would cost more than the potential). `rcut` is AUTO-DERIVED in `new()`/`set_pex_m()` = max pair support over unique types — never a free parameter (hard cuts create E/F discontinuities). m=3→5.7 Å (default), m=4→10.2 Å, m=5→19.1 Å (r0=1.3, pair b=1.8). Lower m = shorter support AND cheaper pex; caveat: smaller m → more abrupt well → smaller dt. GUI has a `pex m` slider; `debug/rarff2d_cutoff/cutoff.png` shows the N-sweep.
  - Benchmark (`rarff2d_bench`, hexagonal lattice a=1.3, 70% fill, non-commensurate w/ grid; scenario+occupancy plot `debug/rarff2d_bench/scenario.png`): neighbors inside rcut=6 mean=45 max=64; grid dx=6 → **14× at N≈3000, eff 66%** (baseline = ninside·t_packed); groups 1.6× (dense sheet → 3 giant overlapping AABBs, no culling — wins only on fragmented scenes). O(N²) baseline eff 16% (84% of calls early-exit).
  - `dbg_timing`/`dbg_npair`/`dbg_ninside`/`dbg_t` instrumentation (counters+per-pass timers, gated — zero hot cost when off).
- **Repair demo**: clean scene E: −9.3 → −32.4 (all 10 atoms bonded, `eatom` −2.6..−5.2); spurious-atom scene: extra atom steals a site, displaced atom flagged by `eatom` (−0.45 vs −2.7 typical) — **the per-atom consistency map works as intended**.
- **Assembly demos**: 3 seeds; hexagons form spontaneously (seed 7: `{2,10,6,5,3,11}`; seed 3: `{0,3,11,8,2,7}` — ring closure visible as late E-drop in E(step)); seed 2 = branched tree. `eatom` correlates exactly with coordination (−0.48 terminal / −0.97 chain / −1.47 branch).
- **Known physics findings**: (a) *basin problem* — atoms pushed inside repulsive wall get ejected past the short b=1.8 attraction tail and can't recapture (motivates weak long-range tail); (b) capped GD needs `dphi_max << w` else orientation limit-cycles; (c) residual maxF ~8–10 at frozen states = edge atoms frustrated between ring-site pull and pair bonds — genuine frustration, not solver noise; (d) a stale-force/eval-skip bug in the demo loop was caught because capped GD can never *increase* E.

## Open issues (planned next steps)

1. **Angular width `w` tuning (ports too stiff/narrow).** Current `w=0.3` → gate half-width ≈ `2w/n` ≈ 11° for n=3 — capture basin is tight; edge atoms sit in persistent frustration and assembly is finicky. Plan: sweep `w ∈ {0.3,0.45,0.6,0.8}` — plot `E(Δφ)` misalignment curves + re-run assembly seeds; target capture basin ~±20–25° without losing port specificity (hexagon vs chain discrimination). Parameter lives in `Rarff2dType.w`.
2. **Verify the mutual-gate is truly pairwise.** Formula `Y = g_i·g_j` requires BOTH atoms oriented; needs a controlled check WITHOUT ring entities (rings obscure the picture): atom i fixed at origin facing +x, atom j moved on a grid / rotated — verify attraction only when both face each other (Δφ_j ≈ π), torque signs rotate each toward alignment, and a misoriented atom yields rep-only (no partial attraction). Compare product gate vs alternatives `(g_i+g_j)/2`, `min`, `sqrt(g_i·g_j)` if a partially-open bond should still attract — product is current choice by design (dangling port → ~0 attraction); confirm it behaves as intended on the (x_j, φ_j) sweep. `rarff2d_view` already visualizes the single-sided variant (`field_at` uses probe gate ≡ 1); extend test to mutual gate.
3. ~~Grid neighbor acceleration~~ — **done** (see Status): `Grid2d` (dx=rcut → 3×3 stencil, halo-gather cell-pair loop, eff 66%) and COG `Groups2d`; `pex_m=3` gives rcut=5.7 Å auto. Remaining: cap group size so AABBs stay tight on dense sheets; sub-circle rejection inside the stencil (33% of grid time is corner candidates).
4. **Weak long-range tail** — recapture atoms ejected outside the b=1.8 Morse tail (basin problem). Options: LJ-like tail, ring-site pull, or gentle attraction ramp.
5. `field_at` currently uses probe gate ≡ 1 (optimally-oriented probe); add option for actual dragged-atom orientation.
6. Later: batch relaxation API for invPPAFM refiner loop; pentagon sites already supported via `nfold`; 3D port-template version and OpenCL batching only after 2D is tuned.
7. **Ring nfold selection caveat (measured by invPPAFM 2026-10-08):** total E cannot pick ring topology — `eval_oriented` freezes ring zc, so a misaligned candidate ring scores the same mean angular gate for every nfold; coincident wrong-nfold sites can even bind atoms *deeper* than correct sites. Wrong nfold deforms more (drift 0.38 Å) than wrong radius (0.24 Å); softening `a` does not help. Working selector = angular coherence `|mean_k e^{i·n·φ_k}|` + Kasa circle fit on member atoms (invPPAFM `testplot_pic_v3_ffcheck.py`). FFI gap: `add_ring`/`add_ring_p` seed `zc=(1,0)` — a phase arg would seed `φc* = arg(Σe^{inφ_k})/n`.

## See also

- [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) — the *port-based* rigid-atom FF (different mechanism: rigid ports + quaternion bodies; rarff2d replaces rods/ports with an orientation gate on a pair potential)
- [`/doc/topical_audit/spatial_acceleration.md`](/doc/topical_audit/spatial_acceleration.md) — `spacc::Buckets` pattern for the planned grid
- `debug/rarff2d/` — potential survey figures · `debug/rarff2d_dyn/` — repair demo artifacts · `debug/rarff2d_assemble*/` — self-assembly trajectories
