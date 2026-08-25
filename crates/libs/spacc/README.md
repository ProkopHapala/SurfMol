---
type: rust-crate
title: spacc
description: Spatial acceleration — AABB, Buckets. Rebuildable caches with no molecular semantics. P0 foundation implemented.
tags: [rust, crate, spatial, acceleration, broad-phase]
timestamp: 2026-08-25
---

# spacc

Rebuildable spatial acceleration structures with no molecular semantics. Provides broad-phase collision detection and spatial queries for the `pgraph` / `pgraph_ops` / `spacc` foundation trio.

## What's implemented (P0 foundation)

- **`aabb.rs`** — `Aabb` (lo/hi `Vec3d`): `include_point`, `include_aabb`, `contains`, `center`, `size`, `max_extent`. `fit_aabb(pos, ids)` fits AABB to selected positions. `fit_group_aabbs(pos, groups, out)` fits AABBs for multiple `IndexGroups` — the dataflow `positions + IndexGroups → spacc::fit_aabbs → Aabb[group]` from the design doc.
- **`buckets.rs`** — `Buckets`: spatial hashing via count→prefix→scatter (same pattern as FireCore/SSE `Buckets.h` and `NBFF::initBBsFromGroups`). Three-step API (`count` → `update_offsets` → `scatter`) or one-shot `build(cell_of_obj)`. `cell_objects(c)` returns packed item list per cell. Items with `cell = -1` are skipped.

## Not yet implemented (P1/P2)

- **`uniform_grid.rs`** — uniform spatial grid with `build_uniform_grid(pos, cell_size)`
- **`morton.rs`** — Morton codes (Z-order curve) for locality-preserving indexing

## Planned API

```rust
fit_aabb(pos, ids) -> Aabb
fit_group_aabbs(pos, offsets, items, out)
build_buckets(cell_of_item, ncell) -> Buckets
build_uniform_grid(pos, cell_size) -> UniformGrid
```

## Design principles

- **Depends only on `numcore` + `pgraph`** — no chemistry dependencies
- **Bounds are invalidated by geometry changes** — caller must rebuild after moving atoms; spacc does not track validity
- **May use different grouping than chemistry** — spatial cells are independent of bond topology
- **Can coexist** as AABB / sphere / capsule / OBB — different query types, different structures
- **No molecular semantics** — `Buckets` knows nothing about atoms or bonds, just item→cell mapping

## Architectural role

Spatial acceleration is a separate concern from both the graph data contract (`pgraph`) and graph algorithms (`pgraph_ops`). It enables:
- Broad-phase neighbor finding for non-bonded forcefields (replacing O(N²) in `NonBondedFF`)
- Ray-picking acceleration for large structures
- Bounding volume hierarchies for rendering culling
- Group AABB fitting for fragment-level collision (FireCore `NBFF::initBBsFromGroups` dataflow)

## Tests

7 tests: AABB include/contains, fit_aabb, fit_group_aabbs, buckets basic/empty-cell/skip-unassigned.

## See also

- `notes/designs/topology_builder.md` — full design (§7 for `spacc` scope)
- `pgraph` — data contract (provides `IndexGroups` consumed by `fit_group_aabbs`)
- `pgraph_ops` — algorithms that consume spatial structures
- FireCore/SSE `Buckets.h` — C++ reference for spatial hashing
- FireCore `NBFF::initBBsFromGroups` — reference for group AABB dataflow
