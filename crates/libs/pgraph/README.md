---
type: rust-crate
title: pgraph
description: Reusable graph algorithms on numtypes data contracts — adjacency, components, bridges, reorder, geometry. Formerly pgraph_ops.
tags: [rust, crate, graph-algorithms, numtypes]
timestamp: 2026-08-25
---

# pgraph

Reusable graph algorithms that operate on the data contracts in `numtypes::graph` (`PGraph`, `CsrAdj`, `FixedAdj<K>`, `Partition`, `Permutation`, `RaggedIndex`, `RangeGroups`). Ported from FireCore `MolecularGraph.h` but without class ownership — free functions over borrowed data.

The original `pgraph` data crate has been merged into `numtypes::graph`; this crate now carries only the algorithm library.

## What's implemented (P0 foundation)

- **`adjacency.rs`** — `build_csr_adj(nverts, edges)` builds CSR via count→prefix→scatter (same pattern as FireCore `MolecularGraph::makeNeighbors`). `build_fixed_adj::<K>(nverts, edges)` builds ELL-like fixed-stride adjacency; fails loud with `DegreeOverflow` on degree > K — never truncates. Both produce parallel `neigh` + `edge` arrays.
- **`components.rs`** — `connected_components(csr)` via iterative BFS; returns `Partition`. `split_by_component(csr)` returns `RaggedIndex` packed by component.
- **`bridges.rs`** — `find_bridges(csr)` via iterative Tarjan DFS with discovery times and low-link values. No recursion → no stack overflow on large graphs. Ported from FireCore `MolecularGraph.h::findBridges` without class ownership.
- **`reorder.rs`** — `partition_to_index_groups(part)` count→prefix→scatter (FireCore `Groups::setGroupMapping` pattern). `group_aware_permutation(part)` packs groups contiguously → `(Permutation, RangeGroups)`. `apply_permutation()`, `permute_edges()` for remapping sidecars.
- **`geometry.rs`** — `edge_vec`, `edge_length`, `edge_lengths`, `bounding_box`, `bounding_box_center`, `bounding_box_span`. Shared by picking, selection, rendering — no molecular semantics.

## Not yet implemented (P1/P2)

- **`loops.rs`** — cycle/ring detection (P2)
- **`selection.rs`** — SDF-based selection (P2)
- **`picking.rs`** — ray-sphere (atoms), ray-cylinder (bonds) picking (P2)
- **`edit.rs`** — editing helpers (P2)

## Design principles

- Algorithms accept slices / `PGraphView` / `CsrAdj` / `FixedAdj` — no ownership of graph data.
- **Depends only on `numtypes`** — the data contracts now live there.
- **Scratch space is local** — never a `PGraph` member; no static DFS time counter.
- **Allocation-free overloads** can take caller buffers for hot paths (not yet needed).
- **Fail loud** on invariant violations (degree overflow, length mismatches).

## Tests

19 tests covering all design-doc §16 invariants: CSR matches brute force, FixedAdj matches CSR, valid entries packed first + remainder `-1`, degree > K errors, permutations preserve associations, Partition→RaggedIndex preserves each item exactly once, bridge finding on triangles/paths/mixed graphs.

## See also

- `numtypes::graph` — the data contracts this crate operates on (`PGraph`, `CsrAdj`, `FixedAdj<K>`, `RaggedIndex`, `Permutation`, `Partition`, `RangeGroups`)
- `notes/designs/topology_builder.md` — full design (§10 for `MolecularGraph.h` → `pgraph` mapping)
- `spacc` — spatial acceleration (used by some algorithms)
- FireCore `MolecularGraph.h` — C++ reference implementation
