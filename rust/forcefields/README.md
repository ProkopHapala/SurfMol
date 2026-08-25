---
type: rust-crate
title: surfmol-forcefields
description: Forcefield energy/force evaluation, molecular dynamics, relaxation, and the MolWorld coordinator. Initial focus is RigidAtomFF (RAFF).
tags: [rust, crate, forcefield, md, relaxation, raff, uff]
timestamp: 2026-08-25
---

# surfmol-forcefields

The simulation and relaxation engine. Implements forcefield energy/force evaluations, runs molecular dynamics, and performs relaxations.

## Initial focus — RigidAtomFF (RAFF)

The main focus at the start is **RAFF**: a rigid-body forcefield where each atom is a frame with **ports** rotated by quaternion rigid-body dynamics (ARAP). See `DESIGN_GOALS.md` §2 for the full RAFF design.

- **Capping atoms** (H, epairs) are a **rigid appendix fixed to a host-atom port** (no independent DOF) — resolved decision.
- **Two variants:** fixed topology (harmonic, 1-to-1 port↔neighbor) and reactive (dissociative Morse, port↔all-in-proximity).
- **Nonbonding:** AABB bounding boxes, contiguous fragments, 16/32/64/128 atoms per fragment.

## MolWorld — the coordinator

All forcefields are connected via a common `MolWorld` engine, which coordinates callbacks for different interaction flavors (bonding, non-bonding, molecule-surface) within the MD/relaxation loop.

**Data ownership:** `MolWorld` does not own atomic positions or forces — those live in `DynamicAtoms` (`surfmol-common`). Each forcefield module owns only its specialized parameters and borrows shared slices during evaluation. See `DESIGN.md` in this folder.

## Forcefields (planned / in progress)

| Forcefield | Status | Reference |
|------------|--------|-----------|
| RigidAtomFF (RAFF) | **initial focus** | FireCore `RRsp3`, `RARFF_SR.h` |
| UFF | port from FireCore | FireCore `UFF.h`, SPAMMM `UFF.cl` |
| RigidMoleFF | planned | SPAMMM `rigid.cl`, FireCore `RigidBodyFF.h` |
| Projective / PBD | planned | FireCore `ProjectiveDynamics_d.h`, SPAMMM `LFF.cl` |

## See also

- `DESIGN.md` (this folder) — data-ownership and `MolWorld` composability rationale.
- `DESIGN_GOALS.md` (repo root) §2 — RAFF design.
- `Import_other_Repos.md` — reference implementations to port.
