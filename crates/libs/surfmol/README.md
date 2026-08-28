---
type: rust-crate
title: surfmol
description: Integration engine — MolWorld orchestrator coordinates Uff/RigidSp3/Raff/NonBonded/SurfaceFolded forcefields, owns DynamicAtoms, runs MD with convergence detection. Supports broad-phase AABB collision culling.
tags: [rust, crate, integration, md, orchestrator, molworld, broad-phase]
timestamp: 2026-09-29
---

# surfmol

Integration engine that ties together `moltopo` (topology + dynamic atom state), `molff` (intra-molecular forcefields), and `surfff` (surface interaction) into a single `MolWorld` orchestrator. This is the glue layer — no physics implementations live here, only coordination.

## Modules

- **`mol_world.rs`** — `MolWorld` struct: owns `DynamicAtoms` (positions, forces, velocities from `moltopo`) and forcefield engines (`Uff`, `RigidSp3FF`, `Option<NonBondedFF>`, `Option<SurfaceFolded>`). Does NOT own positions/forces directly — those live in `DynamicAtoms` (SSOT principle). `BondedFFMode` enum switches between `Uff`, `RigidSp3`, and `Raff` at runtime. `eval_forces` dispatches bonded forces by mode, then evaluates non-bonded and surface if present; returns `(eb, ea, ed, ei, enb, es)` energy components. **`eval_forces_broad(bp)`** — same as `eval_forces` but uses `NonBondedFF::eval_broad` for AABB-culled non-bonded evaluation (identical results, fewer iterations for multi-molecule systems). Surface evaluation borrows PLQH coefficients from `NonBondedFF` rather than duplicating. `run_md` implements the MD loop: evaluate forces → move each atom → check convergence via `v·f < fconv²` → clean velocities if `ff < 0` (numerical instability). `move_atom_md` is `#[inline(always)]` for the hot loop.

- **`import.rs`** — `load_topology_from_json`: convenience function that calls `moltopo::export::import_json` then constructs `Uff::from_topology`. Returns `(Uff, Vec<String>)` (engine + element names). Planned: `load_topology_from_npz` to load from NPZ files (see `molengine/README.md`).

## Design decisions

- **SSOT ownership** — `MolWorld` orchestrates but doesn't own positions/forces. `DynamicAtoms` owns those; each forcefield owns only its parameters. This avoids the monolithic "super-class" anti-pattern.
- **Optional components** — `NonBondedFF` and `SurfaceFolded` are `Option<T>`, enabling gas-phase simulations without surface or non-bonded.
- **Force accumulation in-place** — forces written directly to `dyn_atoms.fapos` slice, no allocation in hot loop.
- **Energy component breakdown** — `eval_forces` returns all 6 components separately for analysis and debugging.
- **Surface-nonbonded coupling** — surface evaluation borrows PLQH from `NonBondedFF` instead of storing a copy.
- **Broad-phase as opt-in** — `eval_forces_broad` takes a `&BroadPhase` reference; the caller (editor) owns the `BroadPhase` and rebuilds it each step. `eval_forces` (O(N²)) remains the default for single-molecule systems.

## What does NOT belong here

- Forcefield implementations → `molff` / `surfff`
- Topology and atom state → `moltopo`
- Rendering → `molrender`
- CLI scripting → `molengine`

## See also

- `molff/DESIGN.md` — forcefield data ownership model
- `molengine` — CLI that uses `MolWorld` via Rhai scripts
- `editor` — GUI that uses `MolWorld` for interactive MD relaxation (with broad-phase AABB collision)
- [`/doc/topical_audit/spatial_acceleration.md`](/doc/topical_audit/spatial_acceleration.md) — broad-phase collision cross-implementation map
- [`/userguide/editor.md`](/userguide/editor.md) — end-user guide for the editor
