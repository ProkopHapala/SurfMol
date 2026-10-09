---
type: folder
title: molff/src
description: Source modules of the molff crate — UFF, RAFF, non-bonded, rigid_sp3, multigrid forcefield engines.
tags: [rust, source, forcefield, uff, raff, nonbonded, multigrid]
timestamp: 2026-08-29
---

# molff/src — Source modules

Source modules of the `molff` crate. See [`../README.md`](../README.md) for the crate overview.

## Module files

- **`lib.rs`** — crate root; declares `pub mod uff; nonbonded; rigid_sp3; raff; raff_reactive; multigrid; rarff2d; ffi_rarff2d; ffi_raff_reactive; ffi_uff;` — crate-type `["rlib","cdylib"]` produces `libmolff.so` for the Python ctypes bindings
- **`ffi_rarff2d.rs`** — extern "C" opaque-handle API for invPPAFM `rarff2d_ffi.py`: state/eval/eval_oriented/step_gd/relax + Bond2d extras (add_bond, bond_report, get_bonds/get_rings, set_inter, set_grid, field_at, emerged_bonds, relax_entities). All f64 C-contiguous.
- **`ffi_raff_reactive.rs`** — extern "C" opaque-handle API (`Raff3dFF`, prefix `raff3d_`) for `SurfMol/scripts/raff3d_ffi.py`: types u8 (0=sp3/1=sp2/2=sp1/3=H point), set_state/get_state (pos (n,3) + quat (n,4) xyzw), eval → (E, F, τ, eatom), step/relax (damped MD via `step_reactive_md`; relax takes separate tol_f/tol_t), get_ports/nports, set_grid/clear_grid, set_traj/clear_traj (multi-frame XYZ via shared `write_xyz_frame`), pairs_within (neighbor query: `rx.grid` or auto-fit `neighbors_within`), field_at (point-probe energy map), emerged_bonds, set_param/set_pex_m/set_cij/set_box. All f64 C-contiguous. Plotter: `scripts/plot_raff3d_potential.py` → `debug/raff3d_potential/`
- **`ffi_uff.rs`** — extern "C" opaque-handle API for invPPAFM `uff_ffi.py`: `uff_new(n, elems_csv, bonds, pos, data_dir)` builds explicit-bond topology via moltopo (neighs→angles/dihedrals/inversions→assign_uff_types→.dat params), `uff_eval`/`uff_relax` (damped MD). Bonded terms only.
- **`uff.rs`** — Universal Force Field: bonds, angles, dihedrals, inversions. SoA aligned arrays, `Buckets` force assembly, complex-number angle powers via `Vec2d::mul_cmplx`
- **`raff.rs`** — RAFF (Rigid-Atom Force Field): port-spring energy, 6 solver modes (ForceMD/InertialReset/FIRE/PBD/XPBD/Projective), Wahba/Horn rotation, non-bonded, harmonic box constraint, `FireState`, `BoxCfg`. See [`/userguide/raff.md`](/userguide/raff.md)
- **`nonbonded.rs`** — `NonBondedFF`: LJ 12-6 + Coulomb + H-bond with 1-2/1-3 exclusion, PBC, force clamping. `BroadPhase` struct + `eval_broad` for AABB-culled eval
- **`rigid_sp3.rs`** — `RigidSp3FF`: **legacy** single-variant rigid body (Dynamic+ForceMD only). Superseded by `raff.rs`
- **`multigrid.rs`** — Multigrid V-cycle solver for linearized molecular elasticity. `LinearOp` trait, `TrussOp` (bond-only), `UffHessianOp` (full UFF Hessian), `GalerkinLevel`, `ModalQuadratic`. See [`/doc/topical_audit/multigrid.md`](/doc/topical_audit/multigrid.md)
- **`raff_reactive.rs`** — **Reactive RAFF**: 3D rigid-atom reactive forcefield on RAFF quaternion state. RARFF-2D radial `pex` Morse × FireCore port-pair gate `(ci·cj·cij)⁴` (`notes/designs/2026-10-09_raff_reactive_3d_inventory.md` §3.1; mirrors `RARFF_SR.h::pairEF`/`projectBonds`/`interEF_buckets`/`evalTorques`). `ReactParams`/`ReactConfig`/`Reactive`: `pair_math` single physics source (point-partner `Y=Σci⁴`, point–point = pure wall), `eval_reactive` (O(N²) + `Grid3` halo-gather cell-pair loop, parity-exact), `step_reactive_md` (`integrate_md` + `eval_box_forces`), `emerged_bonds`, `set_reactive_inertia`, `eatom` consistency map
- **`rarff2d.rs`** — **RARFF-2D**: rod-free orientation-gated reactive pair potential (2D CPU prototype for invAFM geometry repair). Gated Morse `E = a(e² − 2e·g_i·g_j)` with V4 sine-Lorentzian gate and `pex` polynomial-exp; `Ring2d` hexagon/pentagon entities (site bump + repulsive core); **`Bond2d` 2-port entities** (midpoint + axis + half-length; endpoints attract atoms and torque their ports — same gated-Morse math with nfold=2; `bond_esite[2]` per-endpoint occupancy; reaction forces on midpoint/axis/length drive `step_entities_md` snap-on); `InterMask` (atom-atom/ring-atom/atom-bond/ring-bond switches); `step_md`/`step_gd`/`step_entities`/`relax_entities`; `emerged_bonds`; `arena` confinement; per-atom `eatom` consistency map; `fd_check` (covers ring+bond DOFs); `field_at` probe field. See [`/doc/topical_audit/rarff2d.md`](/doc/topical_audit/rarff2d.md) and [`/notes/designs/2026-10-07_bond2d_entity_pic_loss.md`](/notes/designs/2026-10-07_bond2d_entity_pic_loss.md)

## See also

- [`bin/README.md`](bin/README.md) — benchmark binary (`raff_bench`)
- [`../tests/README.md`](../tests/README.md) — integration tests
- [`../README.md`](../README.md) — molff crate overview
- [`../DESIGN.md`](../DESIGN.md) — forcefield data ownership model
