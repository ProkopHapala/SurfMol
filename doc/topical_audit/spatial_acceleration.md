---
type: topical-audit
title: Spatial Acceleration (AABB, Buckets, broad-phase collision)
tags: [topic, spatial, aabb, buckets, broad-phase, collision, cross-language]
timestamp: 2026-08-25
---

# Spatial Acceleration

Cross-implementation map for spatial acceleration structures: axis-aligned bounding boxes (AABB), spatial hashing / bucketing, and broad-phase collision detection.

## Summary

Spatial acceleration is a separate concern from both graph data (`numtypes::graph`) and graph algorithms (`pgraph`). SurfMol's `spacc` crate provides rebuildable caches — AABB fitting and spatial bucketing — with no molecular semantics. `spacc` depends only on `numtypes` (not `pgraph`); the AABB type is `numtypes::Aabb3d` (`Vec6d` alias), and group fitting takes `numtypes::RaggedIndex`. The key dataflow `positions + RaggedIndex → spacc::fit_group_aabbs → Aabb3d[group]` mirrors FireCore's `NBFF::initBBsFromGroups()`. Spatial hashing via `Buckets` uses the same count→prefix→scatter pattern as FireCore/SSE `Buckets.h`.

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `crates/libs/numtypes/src/spatial.rs` | active | **Intrinsic functions**: `aabb_empty`, `aabb_include`, `aabb_contains`, `aabb_merge`, `aabb_overlap`, `aabb_center`, `aabb_size`, `aabb_max_extent`, `aabb_is_valid`. `Aabb3d`/`Aabb3f` as `Vec6` aliases. Also `sym3_*` for symmetric 3×3 tensors. |
| `crates/libs/spacc/src/aabb.rs` | active | `fit_aabb(pos, ids)`, `fit_group_aabbs(pos, groups, out)`, `fit_range_aabbs(pos, ranges, out)`. Cache-optimal contiguous `RangeGroups` path added. |
| `crates/libs/spacc/src/buckets.rs` | active | `Buckets` — spatial hashing via count→prefix→scatter. One-shot `build(cell_of_obj)`; `cell_objects(c)` returns packed item list. `counts` buffer reused as cursor — no extra allocation during rebuild. |
| `crates/libs/molff/src/uff.rs` | active | Legacy: private `Buckets` struct for force-assembly spatial partition. Not using `spacc`. |
| FireCore `cpp/common/dataStructures/Buckets.h` | reference | C++ reference: count→prefix→scatter spatial hashing. Same pattern as `spacc::Buckets`. |
| FireCore `cpp/common/molecular/NBFF.h` | reference | C++ reference: `initBBsFromGroups()` — group AABB fitting dataflow. |
| FireCore `cpp/common/molecular/MMFFBuilder.h` | reference | C++ reference: fragment/group bounding for collision. |

## Parity Status

| Algorithm | SurfMol | FireCore | Parity |
|-----------|---------|----------|--------|
| Spatial hashing (`Buckets`) | `spacc::buckets` | `Buckets.h` | Same count→prefix→scatter pattern. SurfMol is generic (no molecular semantics); FireCore is templated. |
| Group AABB fitting (`fit_group_aabbs`) | `spacc::aabb` | `NBFF::initBBsFromGroups()` | Same dataflow: positions + group mapping → AABB per group. `RaggedIndex` replaces `IndexGroups`. |
| Contiguous range AABB fitting (`fit_range_aabbs`) | `spacc::aabb` | — | New cache-optimal path for packed fragments; FireCore has no direct equivalent. |
| Force assembly buckets | `molff::uff::Buckets` (private) | `UFF.h` | `molff` has its own `Buckets` not using `spacc`. Should migrate. |

Tests: 7 tests in `spacc` (`aabb` contains/fit/range-fit, buckets basic/empty/skip-unassigned).

## Open Issues

- `molff::uff::Buckets` is a private duplicate of `spacc::Buckets` — should migrate to `numtypes`/`spacc`.
- `NonBondedFF` is still O(N²) — should use `spacc` for broad-phase neighbor finding. See `DESIGN_GOALS.md` §2.3.
- `spacc` P1 modules not yet implemented: `uniform_grid.rs` (uniform spatial grid), `morton.rs` (Morton codes / Z-order curve).
- No BVH (bounding volume hierarchy) yet — needed for rendering culling of large structures.

## Resolved

- `spacc` no longer depends on `pgraph`; it depends only on `numtypes` for `Aabb3d` and `RaggedIndex`.
