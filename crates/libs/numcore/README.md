---
type: rust-crate
title: numcore
description: Numerical algorithms acting on numtypes data. Fast Taylor approximations, analytical linalg. No data re-exports.
tags: [rust, crate, math, algorithms]
timestamp: 2026-08-25
---

# numcore

Numerical algorithm crate with zero domain knowledge. It **does not re-export** `numtypes` data — crates that need `Vec3d`, `Quat4i`, `AlignedVec`, `Mat4f`, etc. depend on `numtypes` directly. `numcore` owns the generic numerical algorithms that operate on those types or on slices: fast Taylor approximations and analytical 3×3 symmetric eigendecomposition.

## Modules

- **`math/fastmath.rs`** — Hot-path approximations: `sincos_taylor2` (5th-order Taylor sin/cos), `sincos_r2_taylor` (takes r² as input to avoid sqrt, for small-angle dihedral terms), `dangle` (wraps angle difference to [-π,π] for periodic boundary), `clamp_abs` (branchless absolute-value clamp). All `#[inline(always)]`.
- **`math/linalg.rs`** — `symmetric_eigen_3x3` — analytical (closed-form) 3×3 symmetric eigendecomposition. Ported from FireCore `Mat3.h:Mat3T::eigenvals()` + `eigenvec()`. Replaces nalgebra for PCA in thumbnailer.
- **`math/mod.rs`** — Re-export: `fastmath`, `linalg`.

## Design decisions

- **No `numtypes` re-exports.** The `math::vec2`, `math::vec3`, `math::quat4`, `util`, `math3d`, and `math4d` modules have been removed. If you need `Vec3d`, `Quat4i`, `AlignedVec`, `Aabb3d`, `Mat4f`, use `numtypes` directly.
- **`#[repr(C)]` types come from `numtypes`.** `numcore` functions accept them by value or by slice but never wrap them.

## What does NOT belong here

- Data-layout primitives (vectors, matrices, aligned arrays, graph/spatial contracts) → `numtypes`
- Graph algorithms → `pgraph`
- Spatial acceleration → `spacc`
- Anything chemistry-specific (elements, bonds, atoms) → `moltopo`
- Anything forcefield-related (energy, forces) → `molff` / `surfff`
- Anything rendering-related (shaders, pipelines) → `molrender`

## See also

- `numtypes` — the actual source of `Vec3d`, `Quat4i`, `AlignedVec`, `Aabb3d`, `Mat4f`, `RaggedIndex`, etc.
- `ARCHITECTURE.md` §Component Details
- `molgui` — uses `numcore::math::linalg::symmetric_eigen_3x3`
