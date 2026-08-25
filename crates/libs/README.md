---
type: folder
title: crates/libs
description: Library crates — reusable dependencies with no binary targets. Imported by apps and each other.
tags: [rust, workspace, libraries]
timestamp: 2026-08-25
---

# crates/libs

Reusable library crates — no binary targets, only `lib.rs`. Imported by `crates/apps/` and by each other.

## Crates

- **`numtypes`** — low-level data-layout vocabulary: `#[repr(C)]` vectors, matrices, aligned allocator, positioned-graph data contracts, AABB/spatial aliases. No domain knowledge; tiny intrinsic operations only.
- **`numcore`** — numerical algorithms: `fastmath`, `linalg` (analytical 3×3 symmetric eigendecomposition), `math3d`/`math4d` graphics helpers. Does **not** re-export `numtypes`.
- **`pgraph`** — reusable graph algorithms on `numtypes` data: `build_csr_adj`, `build_fixed_adj<K>`, `connected_components`, `find_bridges`, `partition_to_index_groups`, `group_aware_permutation`, geometry helpers.
- **`spacc`** — spatial acceleration on `numtypes` data: `fit_aabb`, `fit_group_aabbs`, `fit_range_aabbs`, `Buckets` (count→prefix→scatter spatial hashing). Uniform grid and Morton codes planned.
- **`moltopo`** — molecular topology SSOT: `Topology`, `Builder` (generational arena + hex grid), `Params` (forcefield parameter files), UFF type assignment, `DynamicAtoms` for MD, XYZ/JSON I/O.
- **`molff`** — intra-molecular forcefields: `Uff` (bonds/angles/dihedrals/inversions with force-piece assembly), `NonBondedFF` (LJ+Coulomb+H-bond with exclusions + PBC), `RigidSp3FF` (quaternion rigid body).
- **`surfff`** — surface interaction forcefield: folded periodic potential with separable tensor-product Fourier basis, complex recurrence for integer harmonics, NaCl surface template.
- **`surfmol`** — integration engine: `MolWorld` orchestrator coordinates all forcefields, owns `DynamicAtoms`, runs MD with convergence detection.
- **`molrender`** — wgpu rendering primitives: sphere impostors (fragment-shader raytracing), line renderer, textured quad surface renderer. No molecular semantics.
- **`molgui`** — GUI support: trackball camera, PCA-aligned GPU thumbnailer, Kekule hex-grid editor, line gizmos.

## Dependency graph

```
                numtypes
       ___________|_____________
      /            |            \
   numcore       pgraph         spacc
      |            |              |
      ↓            ↓              ↘
  moltopo ←── molff ←── surfmol
      ↘       ↗         ↗
       molrender ← molgui
      ↗
   surfff
```

See `ARCHITECTURE.md` for the full crate dependency graph and `notes/designs/topology_builder.md` for the graph/spatial design.
