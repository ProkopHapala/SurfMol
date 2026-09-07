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

- **`lib.rs`** — crate root; declares `pub mod uff; nonbonded; rigid_sp3; raff; multigrid;`
- **`uff.rs`** — Universal Force Field: bonds, angles, dihedrals, inversions. SoA aligned arrays, `Buckets` force assembly, complex-number angle powers via `Vec2d::mul_cmplx`
- **`raff.rs`** — RAFF (Rigid-Atom Force Field): port-spring energy, 6 solver modes (ForceMD/InertialReset/FIRE/PBD/XPBD/Projective), Wahba/Horn rotation, non-bonded, harmonic box constraint, `FireState`, `BoxCfg`. See [`/userguide/raff.md`](/userguide/raff.md)
- **`nonbonded.rs`** — `NonBondedFF`: LJ 12-6 + Coulomb + H-bond with 1-2/1-3 exclusion, PBC, force clamping. `BroadPhase` struct + `eval_broad` for AABB-culled eval
- **`rigid_sp3.rs`** — `RigidSp3FF`: **legacy** single-variant rigid body (Dynamic+ForceMD only). Superseded by `raff.rs`
- **`multigrid.rs`** — Multigrid V-cycle solver for linearized molecular elasticity. `LinearOp` trait, `TrussOp` (bond-only), `UffHessianOp` (full UFF Hessian), `GalerkinLevel`, `ModalQuadratic`. See [`/doc/topical_audit/multigrid.md`](/doc/topical_audit/multigrid.md)

## See also

- [`bin/README.md`](bin/README.md) — benchmark binary (`raff_bench`)
- [`../tests/README.md`](../tests/README.md) — integration tests
- [`../README.md`](../README.md) — molff crate overview
- [`../DESIGN.md`](../DESIGN.md) — forcefield data ownership model
