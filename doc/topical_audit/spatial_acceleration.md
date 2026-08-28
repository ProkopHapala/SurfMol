---
type: topical-audit
title: Spatial Acceleration (AABB, Buckets, broad-phase collision)
tags: [topic, spatial, aabb, buckets, broad-phase, collision, cross-language]
timestamp: 2026-09-29
---

# Spatial Acceleration

Cross-implementation map for spatial acceleration structures: axis-aligned bounding boxes (AABB), spatial hashing / bucketing, and broad-phase collision detection.

## Summary

Spatial acceleration is a separate concern from both graph data (`numtypes::graph`) and graph algorithms (`pgraph`). SurfMol's `spacc` crate provides rebuildable caches — AABB fitting, spatial bucketing, and broad-phase pair finding — with no molecular semantics. `spacc` depends only on `numtypes` (not `pgraph`); the AABB type is `numtypes::Aabb3d` (`Vec6d` alias), and group fitting takes `numtypes::RaggedIndex`. The key dataflow `positions + RaggedIndex → spacc::fit_group_aabbs → Aabb3d[group]` mirrors FireCore's `NBFF::initBBsFromGroups()`. Spatial hashing via `Buckets` uses the same count→prefix→scatter pattern as FireCore/SSE `Buckets.h`. **Broad-phase collision** (`broad_phase_pairs`) finds overlapping cluster-pair indices using margin-expanded AABB overlap tests, mirroring FireCore `NBFF.h:evalSortRange_BBs`. The `molff::nonbonded::BroadPhase` struct wraps this with cluster ranges + a rebuildable AABB cache, and `NonBondedFF::eval_broad` / `raff::eval_nonbonded_broad` use it to cull atom pairs in multi-molecule systems — producing **identical** forces/energy as the O(N²) path (verified by parity tests).

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `crates/libs/numtypes/src/spatial.rs` | active | **Intrinsic functions**: `aabb_empty`, `aabb_include`, `aabb_contains`, `aabb_merge`, `aabb_overlap`, `aabb_overlap_margin`, `aabb_point_dist2`, `aabb_sphere_overlap`, `aabb_center`, `aabb_size`, `aabb_max_extent`, `aabb_is_valid`. `Aabb3d`/`Aabb3f` as `Vec6` aliases. Also `sym3_*` for symmetric 3×3 tensors. |
| `crates/libs/spacc/src/aabb.rs` | active | `fit_aabb(pos, ids)`, `fit_group_aabbs(pos, groups, out)`, `fit_range_aabbs(pos, ranges, out)`. **`broad_phase_pairs(cluster_aabbs, margin)`** — O(N²) over clusters, returns overlapping `(i,j)` pairs. **`aabb_edges(bb)`** — 12 edge segments for line rendering. |
| `crates/libs/spacc/src/buckets.rs` | active | `Buckets` — spatial hashing via count→prefix→scatter. One-shot `build(cell_of_obj)`; `cell_objects(c)` returns packed item list. `counts` buffer reused as cursor — no extra allocation during rebuild. |
| `crates/libs/molff/src/nonbonded.rs` | active | **`BroadPhase`** struct: cluster ranges + rebuildable AABB cache + rcut. `NonBondedFF::eval_broad()` — AABB-culled non-bonded eval (intra-cluster + inter-cluster pairs). Produces identical forces/energy as `eval()`. |
| `crates/libs/molff/src/raff.rs` | active | **`eval_nonbonded_broad()`** — AABB-culled RAFF non-bonded eval. Same physics as `eval_nonbonded()`, fewer iterations. |
| `crates/libs/surfmol/src/mol_world.rs` | active | **`MolWorld::eval_forces_broad(bp)`** — same as `eval_forces()` but uses broad phase for non-bonded. |
| `crates/apps/editor/src/main.rs` | active | Editor integration: `--nmols N`, `--layout lattice\|random`, `--show-aabb` CLI flags. BroadPhase rebuilt each relaxation step. AABB visualization (green tight + red expanded by rcut). |
| `crates/libs/molff/src/uff.rs` | active | Legacy: private `Buckets` struct for force-assembly spatial partition. Not using `spacc`. |
| `crates/libs/molff/tests/test_broad_phase.rs` | active | **3 parity tests**: `test_broad_phase_parity_nonbonded` (2 molecules near), `test_broad_phase_parity_far_molecules` (2 molecules far, 0 BP pairs), `test_broad_phase_parity_raff` (RAFF non-bonded broad vs O(N²)). All passing. |
| FireCore `cpp/common/dataStructures/Buckets.h` | reference | C++ reference: count→prefix→scatter spatial hashing. Same pattern as `spacc::Buckets`. |
| FireCore `cpp/common/molecular/NBFF.h` | reference | C++ reference: `initBBsFromGroups()` — group AABB fitting dataflow. `evalSortRange_BBs()` — bucket-pair broad phase loop (mirrored by `broad_phase_pairs`). |
| FireCore `cpp/common/molecular/MMFFBuilder.h` | reference | C++ reference: fragment/group bounding for collision. |
| FireCore `pyBall/RigidAtomFF/RRsp3.cl` | reference | OpenCL reference: radius-aware AABB per workgroup, AABB-overlap → atom-vs-AABB ghost test → exact sphere-sphere narrow phase. |
| NumericalMathPlayground `BoundingBoxBalancing.md` | reference | Design for AABB workgroup balancing (Morton rebuild, fixed W-particle groups, 2W→W+W retiling). No implementation. |

## Parity Status

| Algorithm | SurfMol | FireCore | Parity |
|-----------|---------|----------|--------|
| Spatial hashing (`Buckets`) | `spacc::buckets` | `Buckets.h` | Same count→prefix→scatter pattern. SurfMol is generic (no molecular semantics); FireCore is templated. |
| Group AABB fitting (`fit_group_aabbs`) | `spacc::aabb` | `NBFF::initBBsFromGroups()` | Same dataflow: positions + group mapping → AABB per group. `RaggedIndex` replaces `IndexGroups`. |
| Contiguous range AABB fitting (`fit_range_aabbs`) | `spacc::aabb` | — | New cache-optimal path for packed fragments; FireCore has no direct equivalent. |
| **Broad-phase pair finding** (`broad_phase_pairs`) | `spacc::aabb` | `NBFF::evalSortRange_BBs()` | Same O(N²) over clusters with margin-expanded AABB overlap. SurfMol returns sorted pairs; FireCore iterates inline. |
| **AABB-overlap margin test** (`aabb_overlap_margin`) | `numtypes::spatial` | `RRsp3.cl:123-128` | Identical 6-comparison test with per-axis margin expansion. |
| **Sphere-AABB overlap** (`aabb_sphere_overlap`) | `numtypes::spatial` | `Grid_dftb.py:240-244` | Point-to-AABB distance² < r². Used for ghost-atom tests. |
| **Broad-phase non-bonded eval** (`eval_broad`) | `molff::nonbonded` | `NBFF.h` | **Parity verified** — `eval_broad` produces identical forces/energy as `eval` (O(N²)). 3 tests in `test_broad_phase.rs`. Energy doubled to match `eval`'s double-counted convention. |
| Force assembly buckets | `molff::uff::Buckets` (private) | `UFF.h` | `molff` has its own `Buckets` not using `spacc`. Should migrate. |

Tests: 8 tests in `spacc` (aabb contains/fit/range-fit/broad-phase-pairs, buckets basic/empty/skip-unassigned). 3 parity tests in `molff/tests/test_broad_phase.rs`. 5 tests in `numtypes` (aabb basic/overlap/overlap-margin/sphere-overlap, sym3).

## Open Issues

- `molff::uff::Buckets` is a private duplicate of `spacc::Buckets` — should migrate to `numtypes`/`spacc`.
- **PBC not supported with broad phase** — `eval_broad` asserts `!b_pbc || npbc == 0`. PBC + broad phase needs periodic image handling (extend AABBs by PBC shifts or use minimum-image in overlap test).
- `spacc` P1 modules not yet implemented: `uniform_grid.rs` (uniform spatial grid), `morton.rs` (Morton codes / Z-order curve).
- No BVH (bounding volume hierarchy) yet — needed for rendering culling of large structures.
- **Buckets acceleration for broad phase** — current `broad_phase_pairs` is O(N²) over clusters. For many clusters (>100), `Buckets` spatial hashing should be used to find overlapping pairs in O(N). See `notes/designs/cluster_aabb_collision.md` §7 step 6.
- **OpenCL port** — `eval_nonbonded_broad` is CPU-only. GPU port should follow FireCore `RRsp3.cl` pattern: radius-aware AABB per workgroup → AABB-overlap → atom-vs-AABB ghost test → exact sphere-sphere narrow phase.

## Resolved

- `spacc` no longer depends on `pgraph`; it depends only on `numtypes` for `Aabb3d` and `RaggedIndex`.
- **Broad-phase collision implemented** (2026-09-29): `broad_phase_pairs` in `spacc`, `BroadPhase` struct + `eval_broad`/`eval_nonbonded_broad` in `molff`, `eval_forces_broad` in `MolWorld`, editor `--nmols`/`--layout`/`--show-aabb` CLI + GUI visualization. Parity tests pass. See [`/notes/designs/cluster_aabb_collision.md`](/notes/designs/cluster_aabb_collision.md).

## See also

- [`/notes/designs/cluster_aabb_collision.md`](/notes/designs/cluster_aabb_collision.md) — full design document for per-cluster AABB broad-phase collision
- [`/userguide/editor.md`](/userguide/editor.md) — end-user guide for the editor (includes `--nmols`/`--show-aabb` examples)
- [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) — RAFF cross-implementation map (broad phase used in `eval_nonbonded_broad`)
