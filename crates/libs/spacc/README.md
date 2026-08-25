---
type: rust-crate
title: spacc
description: Spatial acceleration — AABB fitting and spatial bucketing on numtypes data. No molecular semantics.
tags: [rust, crate, spatial, acceleration, broad-phase, numtypes]
timestamp: 2026-08-25
---

# spacc

Rebuildable spatial acceleration structures with no molecular semantics. Operates on `numtypes` data contracts (`Aabb3d`, `RaggedIndex`) for AABB fitting and spatial hashing.

## What's implemented (P0 foundation)

- **`aabb.rs`** — `fit_aabb(pos, ids)` fits an `Aabb3d` to selected positions. `fit_group_aabbs(pos, groups, out)` fits AABBs for multiple `RaggedIndex` groups. `fit_range_aabbs(pos, ranges, out)` fits AABBs for contiguous `[i0, i1)` ranges — the cache-optimal path for packed fragments. All use `numtypes::aabb_*` intrinsic functions.
- **`buckets.rs`** — `Buckets`: spatial hashing via count→prefix→scatter (FireCore/SSE `Buckets.h` pattern). One-shot `build(cell_of_obj)`; `cell_objects(c)` returns packed item list per cell. Items with `cell = -1` are skipped. A single `counts` buffer doubles as the per-cell cursor, so no extra allocation happens during rebuild.

## Not yet implemented (P1/P2)

- **`uniform_grid.rs`** — uniform spatial grid with `build_uniform_grid(pos, cell_size)`
- **`morton.rs`** — Morton codes (Z-order curve) for locality-preserving indexing

## Planned API

```rust
fit_aabb(pos, ids) -> Aabb3d
fit_group_aabbs(pos, &RaggedIndex, out)
fit_range_aabbs(pos, &[[Index;2]], out)
Buckets::build(cell_of_item)
build_uniform_grid(pos, cell_size) -> UniformGrid
```

## Design principles

- **Depends only on `numtypes`** — no chemistry or graph-algorithm dependencies.
- **Bounds are invalidated by geometry changes** — caller must rebuild after moving atoms; `spacc` does not track validity.
- **May use different grouping than chemistry** — spatial cells are independent of bond topology.
- **Can coexist** as AABB / sphere / capsule / OBB — different query types, different structures.
- **No molecular semantics** — `Buckets` knows nothing about atoms or bonds, just item→cell mapping.

## Architectural role

Spatial acceleration is a separate concern from both the graph data contract (`numtypes::graph`) and graph algorithms (`pgraph`). It enables:
- Broad-phase neighbor finding for non-bonded forcefields (replacing O(N²) in `NonBondedFF`)
- Ray-picking acceleration for large structures
- Bounding volume hierarchies for rendering culling
- Group AABB fitting for fragment-level collision (FireCore `NBFF::initBBsFromGroups` dataflow)

## Tests

7 tests: `Aabb3d` contains, `fit_aabb`, `fit_group_aabbs`, `fit_range_aabbs`, buckets basic/empty-cell/skip-unassigned.

## See also

- `notes/designs/topology_builder.md` — full design (§7 for `spacc` scope)
- `numtypes::spatial` — `Aabb3d`, `Aabb3f`, and the `aabb_*` / `sym3_*` intrinsics
- `pgraph` — algorithms that consume spatial structures
- FireCore/SSE `Buckets.h` — C++ reference for spatial hashing
- FireCore `NBFF::initBBsFromGroups` — reference for group AABB dataflow
