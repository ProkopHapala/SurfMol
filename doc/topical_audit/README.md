---
type: topical-audit
title: Topical Audit
description: Cross-implementation maps per scientific topic — where each algorithm/feature lives across files, crates, and reference repos.
tags: [topical-audit, cross-reference, navigation]
timestamp: 2026-08-25
---

# Topical Audit

Cross-implementation maps per scientific topic. For each topic, this folder documents **where** each algorithm/feature lives — across SurfMol crates, OpenCL kernels, and the reference repos (FireCore, SPAMMM, learn_Rust, blood_of_civilization).

## Purpose

When implementing or debugging a topic (e.g. "UFF bond evaluation", "AABB collision", "rigid body quaternion integration"), the topical audit tells you:
1. Where it currently lives in SurfMol (if at all).
2. Where the reference implementation lives in FireCore / SPAMMM / learn_Rust.
3. Which OpenCL kernels are involved.
4. Parity status (ported / not ported / in progress).

## Suggested topics (populate as needed)

- **`uff.md`** — UFF force field (bonds, angles, dihedrals, inversions). **Populated 2026-09-29.**
- `nonbonded.md` — LJ / Coulomb / H-bond non-bonded interactions
- `rigid_body.md` — 6-DOF quaternion rigid body dynamics
- **`raff.md`** — RigidAtomFF (RAFF): port-based rigid-atom FF, rotation solvers, XPBD/force-MD/projective, split-collision, GPU layouts. **Populated 2026-09-28.**
- ~~`collision.md`~~ — merged into `spatial_acceleration.md` (broad-phase collision is part of spatial acceleration)
- `projective_dynamics.md` — Projective / position-based dynamics for relaxation
- `gridff.md` — B-spline grid interpolation for substrate potentials
- `ewald2d.md` — 2D Ewald summation for periodic surfaces
- `opencl_device.md` — OpenCL device selection and buffer management

## Populated topics

- **`uff.md`** — UFF cross-implementation map: 4-term UFF (bonds, angles, dihedrals, inversions), parameter assignment pipeline (`setup_params` porting FireCore `assignUFFparams`), topology enumeration (dihedral dedup `i4>i1`, 3 inversions per trigonal center), force evaluation (complex-number angle powers), parity status (pentacene tests). **Populated 2026-09-29.**
- **`graph_algorithms.md`** — Positioned graph data contract (`pgraph`) + algorithms (`pgraph_ops`): adjacency (CSR/ELL), components, bridges, reorder. Parity with FireCore `MolecularGraph.h`, `Groups.h`, `CMesh.h`.
- **`spatial_acceleration.md`** — `spacc` crate: AABB fitting, spatial hashing (Buckets), **broad-phase collision** (`broad_phase_pairs`, `BroadPhase` struct, `eval_broad`/`eval_nonbonded_broad`). Parity with FireCore `Buckets.h`, `NBFF::initBBsFromGroups()`, `NBFF::evalSortRange_BBs()`. **Updated 2026-09-29** with broad-phase collision implementation.
- **`raff.md`** — RAFF cross-implementation map: port-based rigid-atom FF, 4 design axes (rotation solver, dynamics, non-bonded, GPU). Parity with FireCore `RRsp3.cl`, SPAMMM. **Populated 2026-09-28.**
- **`multigrid.md`** — Multigrid V-cycle solver for truss-elasticity (bond-stretch Hessian). Parity with NumericalMathPlayground `LinarElasticity/`. Rust CPU implemented + tested; OpenCL kernels copied but not wired; molecule benchmarks underperforming (debugging in progress). **Populated 2026-08-29.**
- **`gridff_faf.md`** — GridFF and FAF OpenCL macro fragment architecture: build/eval split, `//>>>function`/`//>>>macro` conventions, macro-injection contract for sharing NBFF primitives across UFF/SPFF/RAFF/RigidMolFF. Documents the macro-variant principle (N+M fragments instead of N×M kernel files). **Populated 2026-08-29.**

## Status

Six topics populated (`uff`, `graph_algorithms`, `spatial_acceleration`, `raff`, `multigrid`, `gridff_faf`). Remaining topics are placeholders — populate per topic as implementations are ported. Cross-reference `Import_other_Repos.md` for source locations in reference repos.
