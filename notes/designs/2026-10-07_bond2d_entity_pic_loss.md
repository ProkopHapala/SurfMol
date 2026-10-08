---
type: work-notes
title: "Bond2d first-class entity + FFI restore + PIC-loss migration path for invPPAFM img2mol training"
description: Make bonds first-class objects in rarff2d (like Ring2d), restore the lost cdylib FFI, and sketch the migration of the PIC matching loss into Rust. Needed so bond/atom/ring inconsistency becomes a differentiable training signal and a repair-time force.
tags: [design, rarff2d, molff, ffi, invppafm, pic-loss, img2mol]
timestamp: 2026-10-07
---

# Bond2d entity + FFI restore + PIC-loss path

**Requested by invPPAFM project (USER, 2026-10-07).** The img2mol voxel CNN
predicts atoms, **bonds**, and rings as *independent* objects. Bond
consistency ("does each bond endpoint sit on a predicted atom?") and
back-propagation of that inconsistency into the CNN requires bonds to be
first-class citizens in the forcefield — the current rod-free design
("bonds emerge from geometry, no bond list needed", `rarff2d.rs:1-8`)
cannot score a predicted bond object.

## Cross-repo references (READ FIRST)

The semantic contract lives in invPPAFM — same author, sibling repo. The
implementing agent MUST read these before coding:

| File | What it defines |
|------|-----------------|
| `/home/prokop/git/invPPAFM/notes/plans/2026-10-06_pic_match_loss.md` | THE design doc. PIC v3 loss math (fit/miss/seed terms, C1 kernel `relu(1-d²/R²)²`), the "post-processing: reactive-FF repair + cross-class consistency" section (~line 975) and "RARFF-2D Rust audit" section (~line 1080) — the context this task implements |
| `/home/prokop/git/invPPAFM/export_invAFM/scripts/pic_match.py` | Reference implementation of the PIC loss in torch: `pic_atom_loss_v3`, `pic_bond_loss_v3` (two-endpoint kernel), `pic_loss_v3` (11-ch wrapper). If PIC moves to Rust, THIS is the math to port — keep numerics identical (f64, same kernel) |
| `/home/prokop/git/invPPAFM/export_invAFM/scripts/img2mol.py` | Voxel codec. Bond head convention: `bond_w` + `bond_off` (midpoint offset) + `bond_vec` (**half**-vector); endpoints = `mid ± vec`. `unvoxelize_mol_ep` already does the hard KDTree endpoint→atom snap |
| `/home/prokop/git/invPPAFM/export_invAFM/scripts/rarff2d_ffi.py` | The ctypes contract the new FFI must satisfy (11 symbols; extend, don't break) |
| `/home/prokop/git/invPPAFM/export_invAFM/scripts/graph_torch.py` | `_RarffPenalty` — torch.autograd.Function plumbing `-F` as `dE/dpos`. The differentiable-FF precedent; bond consistency gradients will flow the same way |
| `/home/prokop/git/invPPAFM/export_invAFM/surfmol_forcefields.md` | Prior integration doc — §8.2 FFI plan (spec of the lost shim), §9.5 measured caveats (no valence saturation → energy-floor hinge mandatory) |

## Task 1 — Restore the lost FFI (PREREQUISITE, everything else dead without it)

SurfMol was re-initialized (git history starts 2026-06-01; 28 commits). The
2026-10-03 FFI work — `ffi_rarff2d.rs`, `ffi_uff.rs`, cdylib — is NOT in the
current tree and `~/.cargo-target-shared/release/libmolff.so` was wiped.
`rarff2d_ffi.py` asserts on the missing .so today.

1. `crates/libs/molff/Cargo.toml`: `crate-type = ["rlib", "cdylib"]`.
2. `src/ffi_rarff2d.rs` — opaque-handle extern "C" API. Exact symbol list the
   ctypes shim already calls (see `rarff2d_ffi.py:25-35`):
   `rarff2d_new(n, *types u8)`, `rarff2d_free`, `rarff2d_set_state(*pos n,2,
   *zc n,2)`, `rarff2d_eval(*f, *torq, *eatom) -> f64`,
   `rarff2d_eval_oriented(iters, dt, dphi, *f, *eatom) -> f64`,
   `rarff2d_step_gd(dt, dmax, dphi)`, `rarff2d_relax(nsteps, dt, cdamp, tol,
   *out_steps, *out_conv) -> f64`, `rarff2d_get_state(*pos, *zc)`,
   `rarff2d_set_pex_m(m)`, `rarff2d_set_type(i, nfold, r_cov, a, b, w)`,
   `rarff2d_add_ring(x, y, nfold, r) -> i32`.
3. `eval_oriented` does not exist in core `Rarff2d` — it lived in the lost
   shim. Reimplement: set `inv_m=0` (freeze positions), K iterations of
   orientation-only `step_gd`, restore `inv_m`, then `eval()`. ~20 lines in
   `rarff2d.rs` or the shim.
4. Verify: `cargo build --release -p molff` → `libmolff.so`; then
   `python3 /home/prokop/git/invPPAFM/export_invAFM/scripts/rarff2d_ffi.py`
   (its `__main__` sanity check must pass). Default .so path expected:
   `$MOLFF_SO` env or `~/.cargo-target-shared/release/libmolff.so` — with the
   current `.cargo/config.toml` (`target-dir = "target"`) it lands at
   `target/release/libmolff.so`; either set `MOLFF_SO` or symlink.

## Task 2 — `Bond2d` first-class entity (mirror `Ring2d`, rarff2d.rs:43)

The prediction gives a bond as a **rod**: midpoint + axis + half-length
(codec `mid ± vec`). The entity must accept exactly that:

```rust
/// Bond entity — rod with two endpoint site attractors. e_k = pos ± half·zc.
/// Site attraction pulls atoms onto bond ends; interior ridge repels atoms
/// from sitting on the bond's middle. Center/orientation/half-length are
/// DOFs: reaction force/torque/stretch let a bond snap onto an atom pair.
#[derive(Copy, Clone, Debug)]
pub struct Bond2d {
    pub pos: Vec2d,   // midpoint
    pub zc:  Vec2d,   // unit complex axis (cosφ, sinφ) — segment direction
    pub half: f64,    // half-length [A]; endpoints e0 = pos - half·zc, e1 = pos + half·zc
    pub a: f64, pub h: f64,                       // endpoint site well: -a·B(|x_i - e_k|, h)
    pub a_core: f64, pub r0_core: f64, pub b_core: f64,  // interior ridge repulsion (a_core=0 → off)
}
```

`bond_ef(&mut self, ib, i) -> f64` mirroring `ring_ef` (:348):

1. Per endpoint k: `d = x_i - e_k`; site attraction `E += -a·B(|d|, h)`
   (reuse `bump()` :50). Accumulate `force[i]`, plus reaction into
   `bond_f[ib]` (= Σ F on midpoint), `bond_t[ib]` (torque on `zc` from the
   antisymmetric part — endpoints at ±half give ±lever arm), and
   `bond_l[ib]` (**dE/dhalf** — stretch/shrink DOF; symmetric part of
   endpoint forces along the axis). Keep a per-endpoint site-energy readout
   `bond_esite[ib][2]` — this IS the consistency signal: unoccupied site ⇒
   energy ≈ 0 (no atom at that bond end).
2. Interior ridge (optional, `a_core>0`): distance `ρ` from atom to the
   segment's centerline (|s| ≤ half, s = dot(d_mid, zc)); repulsion
   `a_core·pex(-b_core(ρ - r0_core))²` pushes stray atoms off the bond
   interior without affecting the endpoint atoms (r0_core ~ bond radius,
   NOT endpoint distance). If geometrically fiddly, a midpoint-only
   `pex` wall like the ring core is an acceptable v0 — mark TODO.
3. Extend `eval()`'s entity loop (`for ir in rings` at :536) with
   `for ib in bonds { for i in atoms { bond_ef } }` — O(Nb·N) like rings,
   fine at our scale; Grid2d can index entities later if needed.
4. `set_bonds(Vec<Bond2d>)` mirroring `set_rings` (:392) — allocates
   `bond_f`, `bond_t`, `bond_l`, `bond_esite` scratch ONCE (reconfigure
   step, never in eval).
5. `fd_check` (:643): extend to bond DOFs (pos, zc rotate, half stretch) —
   the existing ring-DOF block is the template.
6. Bond DOF stepping: extend `step_gd`/`step_md` or add a small
   `step_entities` — bonds relax pos/φ/half like rings relax pos/φ.
   Ring2d currently also lacks its own DOF stepping — check how
   `rarff2d_view.rs` handles ring DOFs; unify rather than duplicate.

**Consistency readout for the caller:** expose via a new FFI
`rarff2d_bond_report(*site_e /*nb,2*/, *bond_f /*nb,3 pos,torq,len*/)`.
`site_e[k] ≈ 0` ⇒ bond end k has no atom → inconsistency flag straight
into the invPPAFM loss. Do NOT collapse to a per-bond scalar — the loss
needs per-endpoint resolution.

Also add `emerged_bonds(&self, e_thr) -> Vec<(i,j)>` (~30 ln: pairs with
pair energy below threshold) — eval-time set-diff vs predicted bonds
(`B_ff \ B_pred` = missing, `B_pred \ B_ff` = spurious).

## Task 3 — Consistency/probe FFI extras

- `rarff2d_field_at(x, y, skip, type_id) -> f64` — wrap `field_at` (:595).
  The invPPAFM side uses it as "is there an atom / does one want to be
  here" oracle for seed maps and endpoint checks.
- `rarff2d_set_grid(dx, ox, oy, nx, ny)` — expose `set_grid` so Python
  repair on large scenes gets the 15× cell-pair speedup.
- `rarff2d_get_rings(...)` / bond getters — repair loop needs to read
  back relaxed entity DOFs.

## Task 4 (later, separate PR) — PIC loss in Rust / OpenCL

The matching loss (`pic_atom_loss_v3`/`pic_bond_loss_v3` in
`pic_match.py`) is cell↔proposal kernel gathers + reductions — the same
PIC pattern as `Grid2d`'s neighbor search. Porting gives ONE Rust library
doing decode + consistency + repair + loss:

- `molff/src/pic_loss.rs`: inputs = CNN channel maps (w logits, offsets)
  + ref proposal lists; outputs = scalar L + full dL/dmap gradient maps
  (returned via FFI, consumed by a thin `torch.autograd.Function` —
  same envelope-theorem trick as `_RarffPenalty`).
- Cell bucketing: reuse `spacc::Buckets`; refs bucketed once per sample,
  each pred cell gathers refs in its `⌈R/dx⌉` stencil — identical loop
  shape to `eval()`'s grid path.
- **Parity is the contract**: f64, same kernel `relu(1-d²/R²)²`, same
  normalization — match `pic_match.py` to ~1e-13 on the scenario matrix
  in `testplot_pic_v3.py` before anyone trusts it.
- OpenCL (`oclff/opencl/`): only if batch throughput demands; two-pass
  scatter like the Rust halo pattern to avoid atomics. CPU-first keeps
  OpenCL/CUDA contention (Rule F in invPPAFM) a non-issue.

## Hard requirements ( SurfMol AGENTS rules apply )

- Fail-loud: asserts on natom/bond-count changes (like `set_rings`), no
  silent clamps; `fd_check` must cover every new DOF before tests pass.
- Tests (L0 `cargo test -p molff --test test_rarff2d`):
  1. `test_bond_pulls_atoms_to_ends` — Bond2d alone, 2 atoms within site
     wells → relax converges, atoms on endpoints within tol.
  2. `test_bond_empty_site_flag` — bond with one bare endpoint →
     `bond_esite` asymmetric, near-zero on the empty side.
  3. `test_bond_dof_relax` — bond offset/rotated/stretched from an atom
     pair → DOF relax snaps midpoint/axis/half onto the pair.
  4. fd parity for all bond DOFs.
- No allocations in `eval`/`bond_ef` (scratch preallocated in
  `set_bonds`); no `unwrap_or` that swallows.
- Update `crates/libs/molff/src/README.md`, crate README, `CODEMAP.md`;
  artifacts to `debug/rarff2d_bond/`.

## Implementation status (2026-10-07, verified — pending user review)

Tasks 1–3 implemented; Task 4 (PIC loss) deferred per user decision.

- **Task 1 (FFI):** `molff` crate-type `["rlib","cdylib"]` →
  `libmolff.so` at `$CARGO_TARGET_DIR/release/` (this machine:
  `/home/prokop/.cargo/shared_target/release/libmolff.so`).
  `ffi_rarff2d.rs` = full 11-symbol contract + `eval_oriented` (rebuilt in
  core) + extras; `ffi_uff.rs` = explicit-bond UFF pipeline via moltopo.
  Python wrappers updated: `MOLFF_SO` resolver honors `MOLFF_SO` env /
  `CARGO_TARGET_DIR` / legacy path; `uff_ffi.py` imports the resolver.
- **Task 2 (Bond2d):** `Bond2d{pos, zc, half, a, b, w}` + `from_ends`.
  `bond_ef` reuses `pair_math` conventions — bond = nfold=2 pseudo-atom
  of radius `half`; atom port gates on direction atom→midpoint (the j-side
  `cmul(hij,-conj(z))` pattern — first attempt got this backwards, atoms
  gated outward → positive site energy; fixed and FD-verified).
  `bond_ring_ef` = ring site/core field evaluated at the two endpoints,
  forces distributed onto midpoint/axis/half via lever arms. `InterMask`
  four-channel switch. `bond_esite[ib][2]` hemisphere-split occupancy.
  `emerged_bonds(e_thr)` diagnostics. `fd_check` extended to bond DOFs.
- **Entity stepping:** `step_entities` (GD trust region) creeps
  asymptotically — replaced in the fast path by `step_entities_md`
  (damped MD, same symplectic scheme as `step_md`) + `relax_entities`
  loop with entity-residual early stop (`max_e2`). Bond snap-on
  converges in **142 steps** (was: >30000 GD steps and still not there).
- **Tests:** behavior tests moved to Python per user directive —
  `invPPAFM/export_invAFM/scripts/test_rarff2d_bond.py` (5 tests through
  the ctypes ABI: bond-pull, empty-site flag, DOF relax, InterMask
  channel split, emerged_bonds). Rust `test_rarff2d.rs` keeps the
  engine-internal FD parity (now covers bond DOFs) + atom/ring/grid
  tests — 9/9 pass. UFF sanity: naphthalene converges 345 steps,
  bonds 1.08–1.46 A.
- **Caveat found in testing:** a Bond2d placed far OFF the pair axis is
  *repelled* — the gated-Morse flank (`Y < e`) pushes misaligned entities
  away, so bonds snap onto atoms only from near-aligned proposals
  (decoder hypotheses), not from global search. This is correct physics
  but bounds the capture basin; document for invAFM call sites.
- **Not done:** PIC loss port (Task 4) — design unchanged: dense pred
  maps in, (L, dL/dw, dL/doff) out, refs as sparse lists, f64 parity vs
  `pic_match.py` on the `testplot_pic_v3.py` scenario matrix.

## Physics iteration 2 (2026-10-07, later same day — verified, pending review)

Driven by benzene/fan visualization (`debug/testplot_rarff2d_benzene/`,
`debug/testplot_rarff2d_bondfan/`):

- **`bond_gate` mask** (`rarff2d_set_bond_gate`): bit0 = atom-side gate
  g_i, bit1 = bond-side gate g_b. **Default 2 = g_b only.** Measured on
  identical perturbed benzene (4-way sweep): full gates fmax=54 with 2-atom
  ejection + 350-step recapture plateau; g_b-only fmax=40, clean converge;
  pure radial fmax=28 calmest — BUT radial loses endpoint semantics (atom
  docks anywhere on the r=half circle; in the 1-atom+3-bond fan test the
  shared-endpoint star fails to form, gaps stuck [162,40,157] vs [119,122,
  119] for g_b). g_i-off keeps endpoint wells while removing the atom-side
  repulsive flank that produced the "nonphysical" ejection kicks.
- **`bond_bond` channel** (InterMask bit4): quadratic midpoint wall
  `E = a_i·a_j·(d_m − K_BB·(h_i+h_j))²`, K_BB=sin60° → bonds sharing an
  atom fan to 120° (zero force at contact; also prevents predicted-bond
  collapse). Quadratic not Morse — the e² wall's ~50-unit kick ripped rods
  OFF the atom; bounded force ~2a·r_bb fans smoothly. No half-length
  coupling (it shrank rods to dodge overlap instead of rotating).
- **`half0` + `bond_kh` spring** (default 0.3): singly-attached bonds have
  degenerate half (atom at endpoint ⇒ r=half for any half → rods drifted
  to ~4 A). Weak spring to the predicted length pins the drift; kh=1.0
  biased snapped length by 2%, 0.3 keeps T3 within 0.6%.
- **Arena FFI** (`rarff2d_set_arena`) + `field_at`/`step_entities_md`
  single-step exposure for trajectory recording.
- **Tests:** `test_rarff2d_bond.py` now 6/6 (T6 = bond-bond fan to 120°,
  mask-off control stays clumped). Movies `benzene_gb.gif` /
  `benzene_radial.gif` + `.xyz` trajs for external viewers.
- **Updated caveat:** g_b is load-bearing — it is what makes "atom sits on
  the bond END" true; atom-bond attraction alone is a circle. Keep it on.

## Real-molecule verification (2026-10-07 — verified, pending review)

`export_invAFM/scripts/testplot_pic_v3_backprop.py` on a real 30-C PAH
(`pah_db.pkl` entry `rand8p1_s105r`), `debug/testplot_pic_v3_backprop/`:

- **PIC v3 loss+backprop through the real codec**: GT encoded via
  `voxelize_mol_ep(hard=True)` -> `pic_loss_v3` gives L=0, |grad|~1e-7.
  Defected maps (missing/shifted/dup/spurious atom, missing bond, wrong
  endpoint, missing ring) -> all 6 gradient checks pass: pull-up seed
  gradients (dL/dw<0) appear in the seed disk of every missing feature,
  spurious cells get suppressed (dL/dw>0), offset gradients point at the
  true refs including bond endpoint channels p1/p2. So the differentiable
  chain maps->loss->grads is verified on real geometry, all 3 classes.
- **FF at real scale**: 30 C atoms + 37 Bond2d + 8 Ring2d on the true
  skeleton: perturb (sigma=0.12 A pos, 0.4 rad port) -> `relax_entities
  (move_atoms=True)` converged 476 steps, E -73 -> -439, 73/74 ends
  occupied (one atom stranded outside the capture basin — the documented
  local-basin caveat; esite correctly flags it). eval = 0.06 ms.
- **esite diagnostic demo**: dropping one interior C while keeping the
  predicted bond entities flags exactly the 3 bare ends at the missing
  site — the "bond end has no atom" readout works on a real graph.
- **C-skeleton scope**: use C atoms + C-C bonds + rings only; bay-region
  H sit ~0.8 A apart in 2D projection of a bent 3D molecule and fight
  the repulsive wall (out of scope — maps carry no H anyway).

## FIX — `bond_fh` was not −dE/dhalf (2026-10-07, invPPAFM gradient-verification catch)

Found while fd-verifying the invPPAFM `rarff_consistency` autograd
function, which reconstructs per-endpoint gradients from
`bond_report`'s `fdof = (f, τ, fh)` via the DOF chain rule
(pos=(e0+e1)/2, u=half·zc). Endpoint fd residual ~0.05 traced to
`bond_bond_ef`: the overlap wall used `rbb = K_BB·(h_i+h_j)` on
**current** `half` but never accumulated `bond_fh` — so `fdof` was not a
faithful −∇E over DOFs and every DOF-level consumer (endpoint chain
rule, entity stretch stepping) got a corrupted stretch component.

**Fix applied in `crates/libs/molff/src/rarff2d.rs` (`bond_bond_ef`):**
`rbb` now uses **rest lengths `half0`** instead of current `half`. This
honors the documented intent ("fan rods apart, never fight the length
DOF" — the half-coupling shrank rods to dodge overlap) while making
`fdof` exact: the wall genuinely has no `∂E/∂half` term anymore.
Alternative rejected: adding `K_BB·dedr` to `bond_fh` would have kept
the rods-fight-length behavior the comment explicitly warns against.

**Contract (fdof semantics):** `fdof` is −∇E at **fixed `half0`** —
`half0` is a stored rest length, not a DOF. A Python `set_bonds` probe
re-anchors `half0=half` on every call, so endpoint-level finite
differences *through the FFI* show a residual exactly equal to the
wall's rbb-dependence — probe artifact, not a code bug. DOF-level
`fd_check` (Rust `test_fd_parity`, perturbs `half` with `half0` fixed)
is authoritative — 9/9 Rust tests pass after the change.

**invPPAFM-side usage now live** (`export_invAFM/scripts/pic_match.py`):
- `rarff_consistency(apos, ends, rings, w_phys, w_site, e_floor, e_thr,
  fmax)` — autograd.Function; L = w_phys·relu(E−floor·nat) +
  w_site·mean(relu(esite+e_thr)). Atoms get −F (exact); bond endpoints
  get the chain-rule reconstruction above; ring centers energy-only
  (ring DOF forces not exported — gap: needs `rarff2d_ring_report`).
- `rarff_report(...)` — same eval, plain dict (E, eatom, esite, bare).
- `pic_consistency_v3` — the pure-torch soft version (kernel coverage
  of bond/ring sites by predicted atom proposals + softplus creation
  pull on atom logits) — the training-path term wired into
  `pic_loss_v3` via `w_cons`.

## Ring nfold selection caveat (invPPAFM 2026-10-08) — energy can't pick topology

Measured in `invPPAFM/export_invAFM/scripts/testplot_rarff2d_pentagon.py`:

- `eval_oriented` freezes **ring** DOFs (rings move only via
  `step_entities`/`step_entities_md`). A candidate ring evaluated at a
  fixed phase scores the *same* mean angular gate for every nfold —
  pentagon atoms gave identical E for nfold 5/6/7; a hexagon scored
  *lower* for wrong nfolds. **argmin-E over candidate rings is not a
  topology selector** unless the ring phase is aligned first.
- The angular gate itself works: isolated `eatom` on a pentagon under
  nfold=6 = [+0.05, −0.64, −1.75, −1.75, −0.64] vs flat −0.95 at
  nfold=5 — atoms landing on coincident wrong-nfold sites can bind
  *deeper* than correct sites. Total E rewards wrong topology.
- Wrong-nfold breaks geometry (drift 0.38 Å) more than wrong radius
  (0.24 Å); softening `a` (1.0→0.2) does not prevent it (~0.55 Å).
- **Selector that works:** angular coherence `|mean_k e^{i·n·φ_k}|` over
  member atoms — exact 1.0-vs-0, σ=0.12 Å noise 0.91-vs-0.35. Used by
  invPPAFM `testplot_pic_v3_ffcheck.py` to choose nfold (and radius =
  median member distance) per predicted ring before `add_ring`.
- **FFI gap:** `add_ring`/`add_ring_p` always seed `zc=(1,0)` — no way to
  pass the fitted phase `φc* = arg(Σ e^{inφ_k})/n`. Relax self-aligns via
  `ring_t`, but a `rarff2d_set_ring_zc` (or φ arg on add_ring) would seed
  correct phase and speed convergence on rotated molecules.
