---
type: rust-crate
title: spacc
description: Spatial acceleration — AABB, Buckets, uniform grids, Morton codes. Rebuildable caches with no molecular semantics. Stub — implementation pending.
tags: [rust, crate, spatial, acceleration, broad-phase, stub, planned]
timestamp: 2026-08-25
---

# spacc

**Stub — implementation pending.** See `notes/designs/topology_builder.md` for the full design.

Rebuildable spatial acceleration structures with no molecular semantics. Provides broad-phase collision detection and spatial queries for the `pgraph` / `pgraph_ops` / `spacc` foundation trio.

## Planned modules

- **`aabb.rs`** — axis-aligned bounding boxes
- **`buckets.rs`** — spatial hashing (ported from FireCore/SSE `Buckets.h`)
- **`uniform_grid.rs`** — uniform spatial grid
- **`morton.rs`** — Morton codes (Z-order curve) for locality-preserving indexing

## Planned API

```rust
fit_aabb(pos, ids) -> Aabb
fit_group_aabbs(pos, offsets, items, out)
build_buckets(cell_of_item, ncell) -> Buckets
build_uniform_grid(pos, cell_size) -> UniformGrid
```

## Design principles

- **Depends only on `numcore`** — no graph or chemistry dependencies
- **Bounds are invalidated by geometry changes** — caller must rebuild after moving atoms
- **May use different grouping than chemistry** — spatial cells are independent of bond topology
- **Can coexist** as AABB / sphere / capsule / OBB — different query types, different structures
- **Dataflow**: `positions + IndexGroups → spacc::fit_aabbs → Aabb[group]`

## Architectural role

Spatial acceleration is a separate concern from both the graph data contract (`pgraph`) and graph algorithms (`pgraph_ops`). It enables:
- Broad-phase neighbor finding for non-bonded forcefields
- Ray-picking acceleration for large structures
- Bounding volume hierarchies for rendering culling

## See also

- `notes/designs/topology_builder.md` — full design
- `pgraph` — data contract
- `pgraph_ops` — algorithms that consume spatial structures
- FireCore/SSE `Buckets.h` — C++ reference for spatial hashing
