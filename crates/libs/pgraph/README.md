---
type: rust-crate
title: pgraph
description: Positioned graph data contract — positions + edges + index containers. Domain-agnostic foundation for molecules, meshes, trusses. Stub — implementation pending.
tags: [rust, crate, graph, data-structure, stub, planned]
timestamp: 2026-08-25
---

# pgraph

**Stub — implementation pending.** See `notes/designs/topology_builder.md` for the full design.

Minimal data contract for positioned indexed connectivity: `pos[i]` + `edges[e] = (i, j)`. The defining property is that vertices have positions — this is what distinguishes `pgraph` from a generic graph library. Domain-agnostic: molecules, polygon meshes, and trusses all share this structure.

## Planned contents

- **`PGraph`** — dense vertex/edge storage with `pos: Vec<Vec3d>` and `edges: Vec<[Index; 2]>`
- **`PGraphView<'a>`** — borrowed slices for zero-copy views (safe Rust analogue of FireCore `CMesh`)
- **`Elements<const N>`** — fixed-size element collections (triangles, angle triples, tetrahedra, dihedrals)
- **`Ragged`** — variable-length sets (polygon loops, ring/cycle lists)
- **`Permutation`** — index remapping for compaction/reordering
- **`FixedRows<const K>` / `FixedAdj<const K>`** — ELL-like padded adjacency (constant stride, GPU-friendly, `-1` sentinel for empty slots)
- **`CsrAdj`** — compact CSR adjacency for arbitrary degree
- **`Partition` / `IndexGroups` / `RangeGroups`** — disjoint set / group representations

## Design principles

- **Dense vertex/edge IDs** with valid edge endpoints — no sparse handle indirection
- **Sidecar arrays** for atom types, materials, flags, charges, colors — NOT in PGraph
- **Adjacency is a selectable representation/cache**, not the graph identity
- **Almost no algorithms** — this is a data contract, not an algorithm library. Algorithms live in `pgraph_ops`.

## Architectural role

Part of the `pgraph` / `pgraph_ops` / `spacc` trio — a domain-agnostic foundation that `moltopo` will eventually build upon:

```
        numcore
      /    |    \
   pgraph  spacc  molrender
      \    /
      pgraph_ops
          |
       moltopo
```

This supersedes the earlier `mgraph` proposal (from `rust_workspace_reorg.md`) — `pgraph` ("positioned graph") is unambiguous, while `mgraph` could mean molecule/mesh/material graph.

## See also

- `notes/designs/topology_builder.md` — full design with implementation priority (P0/P1/P2)
- `pgraph_ops` — algorithms on positioned graphs
- `spacc` — spatial acceleration structures
- `moltopo` — chemistry-specific layer that will eventually use `pgraph`
