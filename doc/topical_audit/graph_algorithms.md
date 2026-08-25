---
type: topical-audit
title: Graph Algorithms (positioned graph + adjacency + components + bridges + reorder)
tags: [topic, graph, adjacency, csr, bridges, components, cross-language]
timestamp: 2026-08-25
---

# Graph Algorithms

Cross-implementation map for positioned graph data structures and graph algorithms (adjacency building, connected components, bridge finding, reordering/grouping).

## Summary

The positioned graph — vertices with 3D positions + an edge list — is the shared primitive behind molecules, polygon meshes, and trusses. SurfMol separates the **data contract** (`pgraph`: `PGraph`, `CsrAdj`, `FixedAdj<K>`, `Partition`, etc.) from the **algorithm library** (`pgraph_ops`: adjacency builders, BFS components, Tarjan bridges, group-aware permutation). This mirrors FireCore's `MolecularGraph.h` but replaces class ownership with free functions over borrowed slices.

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `crates/libs/pgraph/src/lib.rs` | active | Data contract: `PGraph`, `PGraphView`, `Elements<N>`, `Ragged`, `Permutation`, `FixedAdj<K>`, `CsrAdj`, `Partition`/`IndexGroups`/`RangeGroups`. `Index=u32`, `INVALID=-1`. |
| `crates/libs/pgraph_ops/src/adjacency.rs` | active | `build_csr_adj` (count→prefix→scatter), `build_fixed_adj::<K>` (ELL-like, fails loud on degree > K). Parallel `neigh`+`edge` arrays. |
| `crates/libs/pgraph_ops/src/components.rs` | active | `connected_components` (iterative BFS), `split_by_component`. |
| `crates/libs/pgraph_ops/src/bridges.rs` | active | `find_bridges` (iterative Tarjan DFS, discovery + low-link). No recursion → no stack overflow. |
| `crates/libs/pgraph_ops/src/reorder.rs` | active | `partition_to_index_groups`, `group_aware_permutation` (→ `Permutation` + `RangeGroups`), `apply_permutation`, `permute_edges`. |
| `crates/libs/pgraph_ops/src/geometry.rs` | active | `edge_vec`, `edge_length`, `bounding_box`, `bounding_box_center`, `bounding_box_span`. |
| `crates/libs/moltopo/src/topology.rs` | active | Legacy: `build_bonds_by_cutoff`, `build_angles/dihedrals/inversions_from_bonds`. Will eventually use `pgraph_ops`. |
| `crates/libs/moltopo/src/molecular.rs` | active | Legacy: `Atoms::make_neigh_bs` — hand-rolled FixedAdj<4> builder. Will eventually use `pgraph_ops::build_fixed_adj::<4>`. |
| FireCore `cpp/common/molecular/MolecularGraph.h` | reference | C++ reference: `makeNeighbors()` (CSR), `findBridges()` (Tarjan), `fillSubGraph`, `splitByBond`, `maskCaps`. Stores CSR + masks + BFS fronts + Tarjan scratch in one class. |
| FireCore `cpp/common/molecular/Groups.h` | reference | C++ reference: `setGroupMapping()` — count→prefix→scatter for group packing. |
| FireCore `cpp/common/dataStructures/CMesh.h` | reference | C++ reference: lightweight borrowed mesh arrays (no ownership hierarchy). `PGraphView` is the safe Rust analogue. |

## Parity Status

| Algorithm | SurfMol | FireCore | Parity |
|-----------|---------|----------|--------|
| CSR adjacency (`build_csr_adj`) | `pgraph_ops::adjacency` | `MolecularGraph::makeNeighbors()` | ✅ Same count→prefix→scatter pattern. SurfMol returns `CsrAdj` struct; FireCore stores as member. |
| Fixed adjacency (`build_fixed_adj<4>`) | `pgraph_ops::adjacency` | `Atoms::make_neigh_bs()` | ✅ Same ELL-like layout with `-1` sentinel. SurfMol fails loud on overflow; FireCore silently truncates. |
| Bridge finding (`find_bridges`) | `pgraph_ops::bridges` | `MolecularGraph::findBridges()` | ✅ Same Tarjan algorithm. SurfMol is iterative (no stack overflow); FireCore is recursive. |
| Group packing (`partition_to_index_groups`) | `pgraph_ops::reorder` | `Groups::setGroupMapping()` | ✅ Same count→prefix→scatter. |
| Connected components | `pgraph_ops::components` | `MolecularGraph` (BFS) | ✅ Same BFS approach. |

Tests: 26 tests in `pgraph_ops` covering all design-doc §16 invariants.

## Open Issues

- `moltopo` still uses its own `Topology`/`Builder`/`Atoms::make_neigh_bs` instead of `pgraph`/`pgraph_ops`. Migration planned per `notes/designs/topology_builder.md` P1.
- `pgraph_ops` P2 modules not yet implemented: `loops.rs` (cycle/ring detection), `selection.rs` (SDF selection), `picking.rs` (ray picking), `edit.rs` (editing helpers).
- `FixedAdj<K>` GPU upload: flat `i32[n*K]` layout is ready for OpenCL but not yet wired to any kernel.
