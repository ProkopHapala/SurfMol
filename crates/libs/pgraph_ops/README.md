---
type: rust-crate
title: pgraph_ops
description: Reusable graph algorithms — adjacency, components, bridges, reorder, geometry. Operates on pgraph data contract. P0 foundation implemented.
tags: [rust, crate, graph-algorithms]
timestamp: 2026-08-25
---

# pgraph_ops

Reusable geometry/topology/edit algorithms that operate on `pgraph` data structures. Separates the data contract (`pgraph`) from the algorithm library — allows narrower dependencies and independent evolution.

## What's implemented (P0 foundation)

- **`adjacency.rs`** — `build_csr_adj(nverts, edges)` builds CSR via count→prefix→scatter (same pattern as FireCore `MolecularGraph::makeNeighbors`). `build_fixed_adj::<K>(nverts, edges)` builds ELL-like fixed-stride adjacency; fails loud with `DegreeOverflow` on degree > K — never truncates. Both produce parallel `neigh` + `edge` arrays.
- **`components.rs`** — `connected_components(csr)` via iterative BFS; returns `Partition`. `split_by_component(csr)` returns vertex index lists per component.
- **`bridges.rs`** — `find_bridges(csr)` via iterative Tarjan DFS with discovery times and low-link values. No recursion → no stack overflow on large graphs. Ported from FireCore `MolecularGraph.h::findBridges` without class ownership.
- **`reorder.rs`** — `partition_to_index_groups(part)` count→prefix→scatter (FireCore `Groups::setGroupMapping` pattern). `group_aware_permutation(part)` packs groups contiguously → `(Permutation, RangeGroups)`. `apply_permutation()`, `permute_edges()` for remapping sidecars.
- **`geometry.rs`** — `edge_vec`, `edge_length`, `edge_lengths`, `bounding_box`, `bounding_box_center`, `bounding_box_span`. Shared by picking, selection, rendering — no molecular semantics.

## Not yet implemented (P1/P2)

- **`loops.rs`** — cycle/ring detection (P2)
- **`selection.rs`** — SDF-based selection (P2)
- **`picking.rs`** — ray-sphere (atoms), ray-cylinder (bonds) picking (P2)
- **`edit.rs`** — editing helpers (P2)

## Design principles

- Algorithms accept slices / `PGraphView` / `CsrAdj` — no ownership of graph data
- **Scratch space is local** — never a `PGraph` member; no static DFS time counter
- **Allocation-free overloads** can take caller buffers for hot paths (not yet needed)
- Ported from FireCore `MolecularGraph.h` but **without class ownership** — free functions over borrowed data
- **Fail loud** on invariant violations (degree overflow, length mismatches)

## Why separate from `pgraph`?

Keeping algorithms in a separate crate from the data contract allows:
- `pgraph` to have minimal dependencies (just `numcore`)
- `pgraph_ops` to pull in heavier dependencies only when needed
- Consumers that only need the data types (e.g., GPU upload) to depend on `pgraph` alone

## Tests

26 tests covering all design-doc §16 invariants: CSR matches brute force, FixedAdj matches CSR, valid entries packed first + remainder `-1`, degree > K errors, permutations preserve associations, Partition→IndexGroups preserves each item exactly once, bridge finding on triangles/paths/mixed graphs.

## See also

- `notes/designs/topology_builder.md` — full design (§10 for `MolecularGraph.h` → `pgraph_ops` mapping)
- `pgraph` — data contract
- `spacc` — spatial acceleration (used by some algorithms)
- FireCore `MolecularGraph.h` — C++ reference implementation
