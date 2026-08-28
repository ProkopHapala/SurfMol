---
type: rust-crate
title: numtypes
description: Low-level data-layout vocabulary for SurfMol — #[repr(C)] math vectors/matrices, aligned allocators, graph/spatial data contracts. Tiny intrinsic operations only.
tags: [rust, crate, math, data-layout, ffi, aligned-alloc]
timestamp: 2026-08-25
---

# numtypes

The project-wide low-level memory vocabulary. Defines data layouts and the tiny intrinsic `O(1)` operations needed to manipulate one value. No algorithms whose cost scales with dataset size; no approximation/iteration/fitting policies.

## Design rule

> **If it is a common data layout that multiple crates exchange, or a primitive operation intrinsic to a single value, it belongs here.**
>
> **If it is a numerical algorithm (eigen solver, fast approximation, interpolation, spatial sweep), it belongs in `numcore` or `spacc` / `pgraph`.**

## Modules

- **`vec.rs`** — `Vec2f`/`Vec2d`, `Vec3f`/`Vec3d`, `Vec4f`/`Vec4d`, `Vec4i`, `Vec6f`/`Vec6d`.
  - Component-wise `+ - * /`.
  - `array()` / `array_mut()` zero-copy views via `bytemuck`.
  - `v[i]` indexing on vectors.
  - `set_add`, `set_sub`, `set_mul`, `set_add_mul`, `set_lincomb`, `add_lincomb`, `cross` for force-assembly hot paths.
  - `cmul()`/`cconj()` for `Vec2d` as complex storage; `mul_cmplx()`/`udiv_cmplx()` in-place methods.
  - `qmul()`/`qconj()`/`qrotate()` for `Vec4d` as quaternion storage; `f()` alias for the vector part.
  - `Quat4d` and `Quat4i` aliases (with `QUAT4I_MINUS_ONES`) for legacy/migration code.
- **`mat.rs`** — `Mat3d`/`Mat4d`/`Mat4f` as rows of `Vec3`/`Vec4`.
  - `rows()` / `array()` zero-copy views.
  - `dot(v)`, `dot_t(v)`, `mmul3`, `mmul4`, `mmul4f`, `outer`, `det`, `inverse` (3×3 explicit).
  - `Mat4f` graphics helpers: `look_at`, `ortho` (Vulkan NDC), `transpose`/`transposed`, `to_arr4x4()`.
- **`alloc.rs`** — `AlignedVec<T, A>`.
  - 64-byte (or A-byte power-of-two) aligned allocation.
  - `Deref<Target=[T]>` / `DerefMut`.
  - `with_len_fill`, `resize_fill`, `push` for `Copy` data.
- **`graph.rs`** — Graph data contracts.
  - `Index = u32`, `INVALID = -1`.
  - `PGraph`, `PGraphView`.
  - `Elements<N>`, `RaggedIndex` (replaces old `Ragged` + `IndexGroups`).
  - `Permutation`, `Partition`, `RangeGroups`.
  - `CsrAdj`, `FixedRows<K>`, `FixedAdj<K>` (now 64-byte aligned via `AlignedVec`).
- **`spatial.rs`** — `Aabb3d`/`Aabb3f` and `SymMat3d`/`SymMat3f` as `Vec6` aliases.
  - Standalone `aabb_*` and `sym3_*` functions.
  - `aabb_empty`, `aabb_include`, `aabb_contains`, `aabb_merge`, `aabb_overlap`, `aabb_center`, `aabb_size`, `aabb_max_extent`, `aabb_is_valid`.
  - **`aabb_overlap_margin(a, b, margin)`** — AABB overlap test with per-axis margin expansion (mirrors FireCore `RRsp3.cl:123-128`).
  - **`aabb_point_dist2(bb, p)`** — squared distance from point to AABB (0 if inside).
  - **`aabb_sphere_overlap(bb, p, r)`** — sphere-AABB overlap test (mirrors `Grid_dftb.py:240-244`).
  - `sym3_det`, `sym3_dot`, `sym3_outer`, `sym3_quadratic`.

## Unsafe policy

`numtypes` is one of the low-level crates where `unsafe` is expected and audited (allocation, alignment, zero-copy views, `bytemuck` casts). All `unsafe` blocks are small and localized. Algorithm crates should prefer to be safe.

## Relationship to other crates

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

- `numcore` adds algorithms (`fastmath`, `linalg`) and no longer re-exports `numtypes`.
- `pgraph` (formerly `pgraph_ops`) operates on `numtypes` graph layouts.
- `spacc` operates on `numtypes` spatial layouts.
