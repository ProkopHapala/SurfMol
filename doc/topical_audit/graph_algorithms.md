---
type: topical-audit
title: Graph Algorithms (positioned graph + adjacency + components + bridges + reorder)
tags: [topic, graph, adjacency, csr, bridges, components, cross-language]
timestamp: 2026-08-25
---

# Graph Algorithms

Cross-implementation map for positioned graph data structures and graph algorithms (adjacency building, connected components, bridge finding, reordering/grouping).

## Summary

The positioned graph — vertices with 3D positions + an edge list — is the shared primitive behind molecules, polygon meshes, and trusses. SurfMol separates the **data contract** (`numtypes::graph`: `PGraph`, `CsrAdj`, `FixedAdj<K>`, `Partition`, `RaggedIndex`, etc.) from the **algorithm library** (`pgraph`: adjacency builders, BFS components, Tarjan bridges, group-aware permutation). This mirrors FireCore's `MolecularGraph.h` but replaces class ownership with free functions over borrowed slices. The `pgraph_ops` crate has been merged into `pgraph`; `pgraph` now depends only on `numtypes`.

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `crates/libs/numtypes/src/graph.rs` | active | **Data contract**: `PGraph`, `PGraphView`, `Elements<N>`, `RaggedIndex` (replaces `Ragged`+`IndexGroups`), `Permutation`, `Partition`, `RangeGroups`, `CsrAdj`, `FixedRows<K>`, `FixedAdj<K>` (using 64-byte aligned `AlignedVec`). `Index=u32`, `INVALID=-1`. |
| `crates/libs/pgraph/src/adjacency.rs` | active | `build_csr_adj` (count→prefix→scatter), `build_fixed_adj::<K>` (ELL-like, fails loud on degree > K). Parallel `neigh`+`edge` arrays. |
| `crates/libs/pgraph/src/components.rs` | active | `connected_components` (iterative BFS), `split_by_component` → `RaggedIndex`. |
| `crates/libs/pgraph/src/bridges.rs` | active | `find_bridges` (iterative Tarjan DFS, discovery + low-link). No recursion → no stack overflow. |
| `crates/libs/pgraph/src/reorder.rs` | active | `partition_to_index_groups` → `RaggedIndex`, `group_aware_permutation` (→ `Permutation` + `RangeGroups`), `apply_permutation`, `permute_edges`. |
| `crates/libs/pgraph/src/geometry.rs` | active | `edge_vec`, `edge_length`, `edge_lengths`, `bounding_box`, `bounding_box_center`, `bounding_box_span`. |
| `crates/libs/moltopo/src/topology.rs` | active | Legacy: `build_bonds_by_cutoff`, `build_angles/dihedrals/inversions_from_bonds`. Will eventually use `pgraph`. |
| `crates/libs/moltopo/src/molecular.rs` | active | Legacy: `Atoms::make_neigh_bs` — hand-rolled FixedAdj<4> builder. Will eventually use `pgraph::build_fixed_adj::<4>`. |
| FireCore `cpp/common/molecular/MolecularGraph.h` | reference | C++ reference: `makeNeighbors()` (CSR), `findBridges()` (Tarjan), `fillSubGraph`, `splitByBond`, `maskCaps`. Stores CSR + masks + BFS fronts + Tarjan scratch in one class. |
| FireCore `cpp/common/molecular/Groups.h` | reference | C++ reference: `setGroupMapping()` — count→prefix→scatter for group packing. |
| FireCore `cpp/common/dataStructures/CMesh.h` | reference | C++ reference: lightweight borrowed mesh arrays (no ownership hierarchy). `PGraphView` is the safe Rust analogue. |

## Parity Status

| Algorithm | SurfMol | FireCore | Parity |
|-----------|---------|----------|--------|
| CSR adjacency (`build_csr_adj`) | `pgraph::adjacency` | `MolecularGraph::makeNeighbors()` | Same count→prefix→scatter pattern. SurfMol returns `CsrAdj` struct; FireCore stores as member. |
| Fixed adjacency (`build_fixed_adj<4>`) | `pgraph::adjacency` | `Atoms::make_neigh_bs()` | Same ELL-like layout with `-1` sentinel. SurfMol fails loud on overflow; FireCore silently truncates. |
| Bridge finding (`find_bridges`) | `pgraph::bridges` | `MolecularGraph::findBridges()` | Same Tarjan algorithm. SurfMol is iterative (no stack overflow); FireCore is recursive. |
| Group packing (`partition_to_index_groups`) | `pgraph::reorder` → `RaggedIndex` | `Groups::setGroupMapping()` | Same count→prefix→scatter. `RaggedIndex` unifies the old `Ragged`/`IndexGroups` split. |
| Connected components | `pgraph::components` | `MolecularGraph` (BFS) | Same BFS approach. |

Tests: 19 tests in `pgraph` covering all design-doc §16 invariants.

## Open Issues

- `moltopo` still uses its own `Topology`/`Builder`/`Atoms::make_neigh_bs` instead of `numtypes::graph`/`pgraph`. Migration planned per `notes/designs/topology_builder.md` P1.
- `pgraph` P2 modules not yet implemented: `loops.rs` (cycle/ring detection), `selection.rs` (SDF selection), `picking.rs` (ray picking), `edit.rs` (editing helpers).
- `FixedAdj<K>` GPU upload: flat `i32[n*K]` layout is ready for OpenCL but not yet wired to any kernel.
- `pgraph_ops` no longer exists; the name and data crate were merged into `numtypes` (data) and `pgraph` (algorithms).
