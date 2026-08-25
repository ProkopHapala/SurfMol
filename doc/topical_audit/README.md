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

- `uff.md` — UFF force field (bonds, angles, dihedrals, inversions)
- `nonbonded.md` — LJ / Coulomb / H-bond non-bonded interactions
- `rigid_body.md` — 6-DOF quaternion rigid body dynamics
- `raff.md` — RigidAtomFF (RAFF): ARAP ports, fixed vs reactive
- `collision.md` — AABB broad phase, uniform grid, spatial hashing
- `projective_dynamics.md` — Projective / position-based dynamics for relaxation
- `gridff.md` — B-spline grid interpolation for substrate potentials
- `ewald2d.md` — 2D Ewald summation for periodic surfaces
- `opencl_device.md` — OpenCL device selection and buffer management

## Populated topics

- **`graph_algorithms.md`** — Positioned graph data contract (`pgraph`) + algorithms (`pgraph_ops`): adjacency (CSR/ELL), components, bridges, reorder. Parity with FireCore `MolecularGraph.h`, `Groups.h`, `CMesh.h`.
- **`spatial_acceleration.md`** — `spacc` crate: AABB fitting, spatial hashing (Buckets). Parity with FireCore `Buckets.h`, `NBFF::initBBsFromGroups()`.

## Status

Two topics populated (`graph_algorithms`, `spatial_acceleration`). Remaining topics are placeholders — populate per topic as implementations are ported. Cross-reference `Import_other_Repos.md` for source locations in reference repos.
