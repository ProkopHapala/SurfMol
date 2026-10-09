---
type: design
title: "Reactive 3D RAFF — inventory of RAFF (3D, fixed topology) + RARFF-2D (reactive) and the cheapest way to combine them"
description: What exists in raff.rs / rarff2d.rs / spacc / FireCore RARFF_SR.h, which pieces are reusable as-is, which must be generalized, and a minimal plan for a reactive 3D port-gated Morse FF with a 3D cell-grid (PIC) pair loop, keeping the fixed-topology RAFF working untouched.
tags: [design, raff, rarff2d, reactive-forcefield, port-based, quaternion, cell-grid, spacc, inventory]
timestamp: 2026-10-09
---

# Reactive 3D RAFF — inventory + combination notes

**Question:** RAFF (3D) has a fixed bond list (ports are springs to *known* neighbors). RARFF-2D is reactive (bonds emerge from a radial Morse × mutual angular gate, no bond list) but 2D with a complex-number orientation. Can we take RARFF-2D's radial function / compact cutoff and put it on RAFF's rigid-port + quaternion bodies, with a 3D cell-grid ("PIC") pair loop?

**Answer:** yes, and almost everything already exists. The only genuinely new code is (1) a 3D port-gate `pair_math` (~60 lines), (2) a generic 3D cell grid in `spacc` (~80 lines), (3) a thin `eval_reactive` cell-pair loop that mirrors `Rarff2d::eval`'s grid branch. Everything else — state, quaternions, ports, integrators, FD checks, pex, Buckets — is reused.

## 1. Inventory

### 1.1 `crates/libs/molff/src/raff.rs` — RAFF 3D, fixed topology (1883 LOC, active, 22+4 tests)

| Piece | Lines | Reusable for reactive? |
|---|---|---|
| `RaffState { pos, quat (xyzw), vel, omega }` | 110 | **as-is** — exactly the rigid-atom DOFs we need (FireCore `apos/qrots/vels/omegas`) |
| `RaffTopology { nport, port_local[natoms*4] (unit body-frame dirs), neighs, neigh_bs, bond_params, mass, inv_inertia, nb_params, excl }` | 149 | **as-is for `nport`/`port_local`/`mass`/`inv_inertia`**. `neighs`/`neigh_bs`/`bond_params`/`excl` are the fixed-topology part — reactive mode simply leaves `neighs = -1` (then `eval_port_forces` contributes 0, see line 592) |
| `set_sp3/set_sp2/set_sp1/set_point` | 306–335 | **as-is** — ideal port templates (= FireCore `sp3_hs/sp2_hs/sp1_hs` minus the pz "electron" slot). Need an `nport=1` cap template for H (currently only via `build_neighs_from_bonds` arm 1 → `(1,0,0)`) |
| `compute_inertia` | 495 | **NOT reusable** — reads `bond_params[l0]` via `neigh_bs`. Reactive needs inertia from `r_cov`: `I ≈ 0.4·(2 r_cov)²` (same formula, different length source) |
| `quat_normalize/quat_mul/quat_conj/quat_rotate/quat_from_omega_dt` | 167–196 | **as-is** (`quat_rotate` is private `fn` — make `pub(crate)`) |
| `eval_port_forces` | 576 | untouched — the fixed-topology spring; coexists (E=0 when `neighs=-1`) |
| `solve_rotation_wahba` / Adiabatic mode | 622 | **not applicable** to reactive (needs a neighbor list). Reactive starts Dynamic-only (torque-driven ω) |
| `eval_nonbonded`/`eval_nonbonded_broad` | 741/821 | optional add-on (LJ/Coulomb); uses `excl` → in reactive mode `excl` is empty so everything interacts — fine, but a reactive Morse wall already provides repulsion; keep `NbConfig.enabled=false` initially |
| `eval_box_forces` | 553 | **as-is** (= RARFF-2D `arena` / FireCore `applyForceBox`) |
| `step_force_md` / `step_inertial_reset` / `step_fire` | 912/957/1047 | integrators are **reusable but hard-wired** to call `eval_port_forces + eval_nonbonded + eval_box_forces` at the top. Cheapest hook: factor the integration loop (lines 921–949) into `integrate_md(state, topo, cfg, fapos, tau) -> (max_f, max_t)` and have `step_force_md` = evals + `integrate_md`. Then a reactive step is `eval_reactive + eval_box + integrate_md`. Same for FIRE if wanted later |
| `fd_check_forces` / `fd_check_torques` | 1607/1653 | **hard-wired to `eval_port_forces`** — generalize to take `eval: impl Fn(&RaffState, &mut [Vec3d], &mut [Vec3d]) -> f64` (surgical: 2 call sites each), then the same harness validates the reactive pair term. Torque FD already perturbs by a small rotation δθ (lines 1667–1669) — exactly what the port gate needs |
| `check_translation_invariance` / `check_rotation_invariance` | 1693/1706 | same generalization |
| `kabsch_rmsd` | 1832 | as-is for geometry parity |

### 1.2 `crates/libs/molff/src/rarff2d.rs` — RARFF-2D reactive (1000+ LOC, active, 9 tests)

| Piece | Lines | Reusable for 3D? |
|---|---|---|
| `Rarff2dType { nfold, r_cov, a, b, w }` + combine rules `r0 = r_i+r_j`, `b = b_i+b_j`, `a = a_i·a_j` | 29 | **reuse the params + combine rules verbatim**; `nfold` is replaced by the port template (`nport`+`port_local`) |
| `pex(x, m)` polynomial exp with compact support `r0 + 2^m/b`, `PEXP_M` | 96–119 | **as-is** (`pub`) — import from `rarff2d` (or move `pex`/`bump` to a tiny `molff::approx` module shared by both; do that only if a 3rd user appears) |
| `pair_support_max(types, m)` → auto-derived `rcut` | 329 | **as-is logic** (private; make `pub` or duplicate the 6-line loop over `(r_cov,b)` pairs) |
| `gate(nfold, w, Δ)` = `1/(1+(sin(nΔ/2)/w)²)`, `phase_half`, `gate_cs` | 127–167 | **2D-only** (angle Δ, n-fold symmetry via complex powers). 3D analogue below (§2.1) uses `cosΔ = h_α·ĥ_ij` directly — no angles at all |
| `pair_math` | 382 | **template** for the 3D version: same structure (r² early exit → sqrt → pex → gates → `E = a(e² − 2eY)` → radial + angular force split). Angular force `dedth·perp/r` becomes the 3D projector `(I − ĥĥᵀ)/r` |
| `Grid2d` (origin, nx, ny, dx, `Buckets`, `cell_of`, halo scratch `h_*`) + `eval` grid branch | 174–202, 644–718 | **pattern to port to 3D**: cell-pair loop = intra-cell pairs + forward-stencil halo gathered into contiguous scratch, scatter-back with zero-skip. Already benchmarked: 14× at N≈3000, 66% eff. The halo scratch is FF-specific (holds `zc`, type) — the geometric part (cell index, rebuild, forward stencil) is generic → `spacc` |
| `Groups2d` COG groups | 212 | skip for now (wins only on fragmented scenes) |
| `step_md` / `step_gd` / `relax` / `eval_oriented` / `emerged_bonds` / `fd_check` | 774– | 2D-specific (complex rotation); 3D equivalents already exist in raff.rs (`step_force_md`, `fd_check_*`). `emerged_bonds` (pairs with `E_pair < thr`) is worth re-implementing in 3D as the bond-detection readout |
| `eatom` per-atom energy (consistency map) | 268 | **keep** — add an `eatom: &mut [f64]` output to the 3D eval (FireCore has per-port `ebonds` instead — finer: tells which *port* is unsatisfied → used by `passivateBonds`) |
| Ring2d / Bond2d entities, FFI | 45–81, `ffi_rarff2d.rs` | 2D/invPPAFM-specific; not part of this task |

### 1.3 `crates/libs/spacc` — spatial acceleration

| Piece | Status | Notes |
|---|---|---|
| `Buckets { ncells, offsets, items }` count→prefix→scatter, `build(cell_of)`, `cell_objects(c)` | active | **as-is** — the CSR core used by `Grid2d`; dimension-agnostic |
| `uniform_grid.rs` | **planned, not implemented** (`spatial_acceleration.md` §Open Issues) | → this task implements it as `Grid3` (origin `[f64;3]`, `n: [usize;3]`, `dx`, `Buckets`, `cell_of`, `cell_at(p)` clamped, `rebuild(pos)`, `forward_cells(c, s, out)` = FireCore `Buckets3D::getForwardNeighbors`) |
| `broad_phase_pairs`, `BroadPhase` | active | cluster-AABB path for fixed molecules — not needed for reactive |

### 1.4 FireCore `cpp/common/molecular/RARFF_SR.h` — the 3D reactive reference (C++)

This is literally "RAFF2D radial + RAFF ports/quaternions" already, just in C++. Key points to mirror / deliberately diverge from:

| Piece | Lines | What it does |
|---|---|---|
| `RigidAtomType { nbond, rbond0, aMorse, bMorse, bh0s (port template) }`, `combine()` | 89–158 | same combine rules as RARFF-2D (`a·a`, `b+b`, `r0+r0`) |
| `projectBonds()` → `hbonds[natom*4] = q.rotate(bh0s)` once per eval | 667–673 | **mirror**: precompute world port dirs `h[natoms*4]` per eval (cheap, 4 quat rotations/atom) so the pair loop is quaternion-free; this is also what the halo gather copies |
| `pairEF` radial: `expr = finiteExp(r − r0, …)`, `E = a·e²`, `Eb = −2a·e` | 453–460 | same Morse split as RARFF-2D; `finiteExp` ≈ our `pex` (we use `pex` — already tuned, compact support derived, no taper) |
| `pairEF` angular: for each port pair (ib, jb): `ci = h_i·ĥ`, `cj = h_j·ĥ`, `cij = h_i·h_j`; skip unless `ci>0, cj<0, cij<0`; `K = (ci·cj·cij)^4`; `E += K·Eb` | 475–543 | the 3D gate. Note the **sum over port pairs** replaces RARFF-2D's n-fold symmetric gate. Force on positions via `SMat3d Dij.fromDeriv(dij, 1/(r·r²))` = projector `(r²I − ddᵀ)/r³`; force on port dirs accumulated into `fbonds` |
| `evalTorques()`: `τ_i = Σ_α h_α × f_α` (after `makeOrthoU`) | 718–732 | generalized force on a unit vector → torque. We can accumulate `τ` directly in the pair loop (`τ_i += h_α × f_α`) — avoids the `fbonds[natoms*4]` buffer |
| `interEF_buckets()` with `Buckets3D::getForwardNeighbors` | 599–636 | exactly the `Grid2d` cell-pair loop in 3D (no halo gather — we keep RARFF-2D's gather, it measured faster) |
| caps (`bondCaps`, `lcap`, `bRepelCaps`), pz alignment `Epz` (sp2 4th slot), `ebonds` per-port energy, `passivateBonds` | 481–508, 547–559, 773 | **phase 2** — implicit H caps and π-alignment. Not needed for a first reactive C/H system where H is an explicit 1-port atom |
| `moveMDdamp`: `v = damp·v + F·dt`, `ω = damp·ω + τ·invRotMass·dt`, `q.dRot_exact(dt, ω)` | 759–771 | identical to `step_force_md` lines 931–946 |

## 2. How to combine (minimal plan)

### 2.1 The 3D pair term — "RARFF-2D radial × port gate"

Per pair (i, j), `d = x_j − x_i`, `r = |d|`, `ĥ = d/r`, world port dirs `h_iα = R(q_i) a_iα`, `h_jβ = R(q_j) a_jβ`:

```
e, dedx = pex(−b·(r − r0), m)            # radial — REUSED from rarff2d (compact support r0 + 2^m/b)
E_rep   = a·e²                            # ungated wall (same as 2D)
Y       = Σ_α Σ_β G(c_iα)·G(−c_jβ)·[·C(c_αβ)]   c_iα = h_iα·ĥ,  c_jβ = h_jβ·ĥ,  c_αβ = h_iα·h_jβ
E       = a·(e² − 2·e·Y)
```

Gate choice — the 3D analogue of the 2D sine-Lorentzian. In 2D `u = sin(nΔ/2)/w`; at a single port `sin²(Δ/2) = (1 − cosΔ)/2`, so the natural port-wise form is

```
G(c) = 1 / (1 + (1 − c)/(2w²))      for c > 0,  else 0  (+ smooth cut, see below)
dG/dc = G² / (2w²)
```

— a *rational function of the cosine*: no `acos`, no `sin`, no `atan2`; same quadratic stiffness near alignment as the tuned 2D gate (`G ≈ 1 − Δ²/(4w²)`), and `w` keeps its 2D meaning (half-width ≈ 2w rad). Summing over ports gives the n-fold structure automatically from the template — for sp2 in-plane this reproduces the 2D 3-fold gate shape up to the overlap of neighbouring lobes (negligible for w ≤ 0.5, ports 120° apart). The `c>0` cut should be made smooth (e.g. multiply by `c²` or use `max(c,0)²`-weighted form) so E is C¹ — the FireCore hard `if(ci<0) continue` on `(c)^4` is C³ because of the 4th power; choose one and FD-verify.

Optional FireCore factor `C(c_αβ)` (ports anti-parallel): at the optimum `h_iα = ĥ = −h_jβ` it is automatically 1, so it only reshapes the basin. Start **without** it (fewer terms, closer to 2D); add behind a flag if 3D assembly shows mis-docking.

Forces/torques (all analytic, FD-verified by the generalized `fd_check_*`):
- radial: `dE/dr = 2ab·dedx·(Y − e)` along `ĥ` (same expression as 2D line 404)
- angular on positions: `∂c_iα/∂x_j = (h_iα − c_iα ĥ)/r` → `F_j −= (−2ae)·Σ (∂Y/∂c_iα)(h_iα − c_iα ĥ)/r` (the 3D projector; 2D's `perp/r`)
- torque: generalized force on port dir `f_α = −∂E/∂h_iα = 2ae·(∂Y/∂c_iα)·ĥ`, then `τ_i += h_iα × f_α` (FireCore `evalTorques` inlined)
- `eatom[i] += E/2`, `eatom[j] += E/2`

Cost: ~`nport_i·nport_j` ≤ 16 dot products per pair inside cutoff — vs 2D's 2 gates. Fine on CPU for now; on GPU the 4×4 loop is unrolled anyway.

### 2.2 Data: what to add to RAFF (not change)

- `pub struct ReactParams { r_cov, a, b, w }` per atom (new `Vec` in `RaffTopology`, like `nb_params`) **or** per-type table + `type_of[i]` — per-atom is simpler and matches `nb_params`.
- `pub struct ReactConfig { enabled, pex_m, rcut (derived, asserted), use_cij: bool }`.
- `RaffTopology::set_reactive(params)`: fills `react_params`, sets `inv_inertia[i] = 1/(0.4·(2 r_cov)² + ε)` (the `compute_inertia` formula with `l0 = 2 r_cov`), derives `rcut = pair_support_max`. Ports come from the existing `set_sp3/set_sp2/set_sp1` + a new `set_cap(i)` (`nport=1`, `(1,0,0)`).
- H as a **1-port rigid atom** (not `set_point`): its single port must aim at the host — this is what makes C–H directional. Rotation about its own axis is a null-space (τ ⊥ h always) — harmless in Dynamic mode.
- Scratch `h_world: Vec<Vec3d>` (`natoms*4`) + `eatom: Vec<f64>` — allocated once in `set_reactive`, never in `eval`.

### 2.3 Pair loop + 3D cell grid (the "PIC" acceleration)

- New `spacc::uniform_grid::Grid3` (generic, no molecular semantics): `cell_at(p)` clamped to domain, `rebuild(pos)`, forward stencil `(dz=0: dx>0 same row; dy>0 all dx) ∪ (dz>0: all dx,dy)` — the FireCore `getForwardNeighbors` set, 13 cells for s=1. `Grid2d` can later be re-expressed on top of a `Grid2` sibling (not now — YAGNI).
- `raff::eval_reactive(state, topo, rcfg, grid: Option<&mut Grid3>, fapos, tau, eatom) -> f64`:
  1. project ports: `h_world[i*4+α] = quat_rotate(q_i, port_local[i*4+α])` (= `projectBonds`)
  2. `None` → O(N²) `i<j` reference loop (parity oracle, like `Rarff2d::eval` fallback)
  3. `Some(grid)` → `Grid2d`-pattern cell-pair loop: intra-cell pairs; forward-halo gather of `(idx, pos, h[4], nport, params)` into contiguous scratch; stream; scatter `(f, τ, e)` back with zero-skip. **Must be bit-identical in E to the O(N²) path** (sum-order differences only, ~1e-13) — same parity test as `test_rarff2d::test_grid_parity`.
- `dx = rcut` → strict 3×3×3 stencil. With `pex_m = 3`, C–C `r0 = 1.54`, pair `b ≈ 1.8` → rcut ≈ 6 Å (same knob/derivation as 2D).

### 2.4 Integrator hook (keeps fixed-topology RAFF untouched)

- Factor lines 921–949 of `step_force_md` into `integrate_md(state, topo, cfg, fapos, tau) -> (max_f, max_t)`; `step_force_md` body becomes `evals; integrate_md`. Behaviour identical — existing 26 RAFF tests are the regression gate.
- `step_reactive_md(state, topo, cfg, rcfg, grid, fapos, tau, eatom) = eval_reactive + eval_box_forces + integrate_md`. Optionally `+ eval_port_forces` so fixed intramolecular bonds and reactive intermolecular contacts can coexist later (currently zero-cost when `neighs = -1`).
- Orientation: **Dynamic only** (ω from τ). Adiabatic/Wahba needs neighbors → not for reactive (a "reactive Wahba" = pick best partner per port is a possible later speed-up).

### 2.5 Verification (define before coding)

L0 (`cargo test`, new `tests/test_raff_reactive.rs`):
1. **FD parity** forces & torques via generalized `fd_check_forces/torques` with `eval_reactive` closure, random 6-atom cloud (sp3/sp2/cap mix), target ~1e-7 rel (2D achieves 1e-8..1e-9).
2. **Two sp3 atoms facing**: E_min at `r = 2·r_cov` equals `−a` (Y=1 → `e²−2e` min = −1 at e=1); anti-facing (ports away) → E ≥ 0 everywhere (pure wall). Torque sign: both rotate toward alignment.
3. **Grid3 parity vs O(N²)**: N≈200 random cloud, |ΔE| < 1e-12, max|ΔF|, |Δτ|, |Δeatom| < 1e-12; also a case with atoms outside the domain (clamped cells).
4. **Translation/rotation invariance**: ΣF = 0, Σ(x×F + τ) = 0 (the latter is the real test of the port-torque bookkeeping — `check_rotation_invariance` generalized).
5. **Methane assembly**: 1 sp3 C + 4 cap H random within 3 Å, `step_reactive_md` damped → 4 bonds emerge (`E_pair < thr`), H–C–H ≈ 109.5°; print per-atom `eatom`.
6. **Fixed-topology regression**: all existing `test_raff*.rs` unchanged and green after the `integrate_md` factor-out.

L1/L2: `debug/raff_reactive/` — E(step), max|F|, trajectory `.xyz`, emerged-bond list; benchmark `N` sweep grid vs O(N²) analogous to `rarff2d_bench`.

### 2.6 Explicitly deferred

- FireCore parity vs `RARFF_SR::pairEF` (different gate form unless `use_cij` + 4th power are implemented as a second gate variant).
- Implicit caps (`bondCaps`), pz/π alignment (sp2 4th slot), per-port `ebonds` + `passivateBonds`.
- Groups/COG culling, OpenCL port (pair loop is already shaped for it: gather, 4×4 unrolled, SoA).
- Editor wiring (`RaffSolverMode` has 6 entries hard-wired to `eval_port_forces` — a 7th "Reactive" mode is a small follow-up once the CLI/test path works).

## 3. Decisions (USER, 2026-10-09)

1. **Gate form: FireCore `(ci·cj·cij)^4`** summed over port pairs (direct `RARFF_SR.h::pairEF` parity; `w` is unused — `ReactParams` = `{r_cov, a, b}`). The cosine-Lorentzian variant of §2.1 is NOT implemented (kept here as a possible later flag).
2. **File placement:** new `molff/src/raff_reactive.rs`; `raff.rs` touched only for the `integrate_md` factor-out, the eval-closure generalization of `fd_check_*`/`check_*_invariance`, and `pub(crate) fn quat_rotate`.
3. **Grid3 in `spacc::uniform_grid`** (generic).
4. **H = point atom** (`nport = 0`): the angular potential of the heteroatom (C/O/N ports) enforces bond angles. Consequence for the product gate: a port-pair sum needs ports on both sides, so for a point partner j the gate is `Y = Σ_α ci_α⁴ (ci_α > 0)` (H-side gate ≡ 1); **point–point pairs (H–H) have Y = 0 → pure Morse wall** (= FireCore caps only repel).
5. Per-atom `ReactParams` (like `nb_params`).

### 3.1 Final pair term (authoritative for implementation; mirrors `RARFF_SR.h::pairEF` 441–564)

`d = x_j − x_i`, `r² = |d|²` (early exit `r² > rcut²`), `r`, `ĥ = d/r`; `r0 = r_i + r_j`, `b = b_i + b_j`, `a = a_i·a_j`; `(e, dedx) = pex(−b(r−r0), m)`.

```
cc_αβ = ci_α · cj_β · cij_αβ         ci_α = h_iα·ĥ,  cj_β = h_jβ·ĥ,  cij_αβ = h_iα·h_jβ
skip pair (α,β) unless  ci_α > 0  &&  cj_β < 0  &&  cij_αβ < 0      (product is then > 0)
K_αβ = cc⁴,   dK = 4cc³
Y = Σ_αβ K_αβ                               (nport_j = 0:  Y = Σ_α ci_α⁴, dK = 4ci_α³ ;  both 0: Y = 0)
E = a·(e² − 2eY)
dE/dr = 2ab·dedx·(Y − e)                   (de/dr = −b·dedx)
∇_j E = (dE/dr)·ĥ + (−2ae)/r · Σ_αβ dK·[ cj·cij·(h_iα − ci·ĥ) + ci·cij·(h_jβ − cj·ĥ) ]      (point j: Σ_α dK·(h_iα − ci·ĥ))
F_j = −∇_j E,  F_i = +∇_j E
f_iα = 2ae · Σ_β dK·[ cj·cij·ĥ + ci·cj·h_jβ ]      τ_i += h_iα × f_iα        (point j: f_iα = 2ae·dK·ĥ)
f_jβ = 2ae · Σ_α dK·[ ci·cij·ĥ + ci·cj·h_iα ]      τ_j += h_jβ × f_jβ
eatom_i += E/2, eatom_j += E/2
```

Sanity: facing sp3 pair at `r = r0` with `h_iα = ĥ = −h_jβ` → `cc = 1·(−1)·(−1) = 1`, Y = 1, `E = a(1 − 2) = −a`.

## See also

- `doc/topical_audit/raff.md`, `doc/topical_audit/rarff2d.md`, `doc/topical_audit/spatial_acceleration.md`
- `notes/designs/raff_theory_equations.md` (§1 port convention, §2.1 dynamic rotation)
- FireCore `cpp/common/molecular/RARFF_SR.h` (`pairEF` 441, `interEF_buckets` 599, `evalTorques` 718)
