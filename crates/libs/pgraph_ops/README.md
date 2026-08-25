---
type: rust-crate
title: pgraph_ops
description: Reusable graph algorithms — adjacency, components, bridges, loops, reorder, geometry, selection, picking. Operates on pgraph data contract. Stub — implementation pending.
tags: [rust, crate, graph-algorithms, stub, planned]
timestamp: 2026-08-25
---

# pgraph_ops

**Stub — implementation pending.** See `notes/designs/topology_builder.md` for the full design.

Reusable geometry/topology/edit algorithms that operate on `pgraph` data structures. Separates the data contract (`pgraph`) from the algorithm library — allows narrower dependencies and independent evolution.

## Planned modules

- **`adjacency.rs`** — build CSR and fixed-stride adjacency from edge lists
- **`components.rs`** — connected components, graph splitting
- **`bridges.rs`** — bridge finding (ported from FireCore `MolecularGraph.h`)
- **`loops.rs`** — cycle/ring detection
- **`reorder.rs`** — permutations, compaction, remapping
- **`geometry.rs`** — edge vectors, lengths, normals, local frames
- **`selection.rs`** — SDF-based selection
- **`picking.rs`** — ray-sphere (atoms), ray-cylinder (bonds) picking
- **`edit.rs`** — editing helpers

## Design principles

- Algorithms accept slices / `PGraphView` / `CsrAdj` — no ownership of graph data
- **Scratch space is local or reusable `Workspace`**, never a `PGraph` member
- **Allocation-free overloads** can take caller buffers for hot paths
- Ported from FireCore `MolecularGraph.h` but **without class ownership** — free functions over borrowed data

## Why separate from `pgraph`?

Keeping algorithms in a separate crate from the data contract allows:
- `pgraph` to have minimal dependencies (just `numcore`)
- `pgraph_ops` to pull in heavier dependencies (spatial structures, geometry) only when needed
- Consumers that only need the data types (e.g., GPU upload) to depend on `pgraph` alone

## See also

- `notes/designs/topology_builder.md` — full design
- `pgraph` — data contract
- `spacc` — spatial acceleration (used by some algorithms)
- FireCore `MolecularGraph.h` — C++ reference implementation
