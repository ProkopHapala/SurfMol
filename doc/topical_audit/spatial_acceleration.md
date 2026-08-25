---
type: topical-audit
title: Spatial Acceleration (AABB, Buckets, broad-phase collision)
tags: [topic, spatial, aabb, buckets, broad-phase, collision, cross-language]
timestamp: 2026-08-25
---

# Spatial Acceleration

Cross-implementation map for spatial acceleration structures: axis-aligned bounding boxes (AABB), spatial hashing / bucketing, and broad-phase collision detection.

## Summary

Spatial acceleration is a separate concern from both graph data (`pgraph`) and graph algorithms (`pgraph_ops`). SurfMol's `spacc` crate provides rebuildable caches — AABB fitting and spatial bucketing — with no molecular semantics. The key dataflow: `positions + IndexGroups → spacc::fit_aabbs → Aabb[group]`, mirroring FireCore's `NBFF::initBBsFromGroups()`. Spatial hashing via `Buckets` uses the same count→prefix→scatter pattern as FireCore/SSE `Buckets.h`.

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `crates/libs/spacc/src/aabb.rs` | active | `Aabb` (lo/hi Vec3d), `fit_aabb(pos, ids)`, `fit_group_aabbs(pos, groups, out)`. `include_point`/`include_aabb`, `contains`, `center`, `max_extent`. |
| `crates/libs/spacc/src/buckets.rs` | active | `Buckets` — spatial hashing via count→prefix→scatter. Three-step API or one-shot `build(cell_of_obj)`. `cell_objects(c)` returns packed item list. |
| `crates/libs/molff/src/uff.rs` | active | Legacy: `Buckets` struct (private to `Uff`) for force assembly spatial partition. Not using `spacc`. |
| FireCore `cpp/common/dataStructures/Buckets.h` | reference | C++ reference: count→prefix→scatter spatial hashing. Same pattern as `spacc::Buckets`. |
| FireCore `cpp/common/molecular/NBFF.h` | reference | C++ reference: `initBBsFromGroups()` — group AABB fitting dataflow. |
| FireCore `cpp/common/molecular/MMFFBuilder.h` | reference | C++ reference: fragment/group bounding for collision. |

## Parity Status

| Algorithm | SurfMol | FireCore | Parity |
|-----------|---------|----------|--------|
| Spatial hashing (`Buckets`) | `spacc::buckets` | `Buckets.h` | ✅ Same count→prefix→scatter pattern. SurfMol is generic (no molecular semantics); FireCore is templated. |
| Group AABB fitting (`fit_group_aabbs`) | `spacc::aabb` | `NBFF::initBBsFromGroups()` | ✅ Same dataflow: positions + group mapping → AABB per group. |
| Force assembly buckets | `molff::uff::Buckets` (private) | `UFF.h` | ⚠️ `molff` has its own `Buckets` not using `spacc`. Should migrate. |

Tests: 7 tests in `spacc` (AABB include/contains/fit, buckets basic/empty/skip-unassigned).

## Open Issues

- `molff::uff::Buckets` is a private duplicate of `spacc::Buckets` — should migrate to use `spacc`.
- `NonBondedFF` is still O(N²) — should use `spacc` for broad-phase neighbor finding. See `DESIGN_GOALS.md` §2.3.
- `spacc` P1 modules not yet implemented: `uniform_grid.rs` (uniform spatial grid), `morton.rs` (Morton codes / Z-order curve).
- No BVH (bounding volume hierarchy) yet — needed for rendering culling of large structures.
- `spacc` depends on `pgraph` for `IndexGroups` (used by `fit_group_aabbs`). This is the only cross-crate dependency; `spacc` does not depend on `pgraph_ops` or any chemistry crate.
