---
type: rust-crate
title: pgraph
description: Positioned graph data contract — positions + edges + index containers. Domain-agnostic foundation for molecules, meshes, trusses. P0 foundation implemented.
tags: [rust, crate, graph, data-structure]
timestamp: 2026-08-25
---

# pgraph

Minimal data contract for positioned indexed connectivity: `pos[i]` + `edges[e] = (i, j)`. The defining property is that vertices have positions — this is what distinguishes `pgraph` from a generic graph library. Domain-agnostic: molecules, polygon meshes, and trusses all share this structure.

## What's implemented (P0 foundation)

- **`PGraph`** — dense vertex/edge storage with `pos: Vec<Vec3d>` and `edges: Vec<[Index; 2]>`. `validate()` checks edge endpoints in range. `view()` borrows as zero-allocation `PGraphView`.
- **`PGraphView<'a>`** — borrowed slices for zero-copy views (safe Rust analogue of FireCore `CMesh`).
- **`Elements<const N>`** — fixed-size element collections. `Elements<3>` = triangles OR angle triples; `Elements<4>` = tetrahedra OR dihedrals. Share storage, not semantics — the owning layer decides meaning.
- **`Ragged`** — count/offset/packed-items primitive for variable-length sets (polygon loops, ring lists, arbitrary groups). `from_counts()`, `group()`, `group_mut()`.
- **`Permutation`** — bidirectional old2new / new2old remapping for compaction/reordering. `from_new2old()`, `identity()`.
- **`FixedRows<const K>` / `FixedAdj<const K>`** — ELL-like padded adjacency with `-1` sentinel for empty slots. Valid entries packed first; `push()` fails loud on degree > K (never truncates). `FixedAdj` carries parallel `neigh` + `edge` tables so hot kernels don't search for bond ids — matches FireCore `neighs[natom]` + `neighBs[natom]` layout.
- **`CsrAdj`** — compact CSR adjacency for arbitrary degree. `offsets` / `neigh` / `edge` arrays; `neighbors(v)` returns parallel slices.
- **`Partition` / `IndexGroups` / `RangeGroups`** — disjoint group representations: flexible assignment → packed index lists → contiguous ranges after group-aware permutation.

## Design principles

- **Dense vertex/edge IDs** with valid edge endpoints — no sparse handle indirection
- **Sidecar arrays** for atom types, materials, flags, charges, colors — NOT in PGraph
- **Adjacency is a selectable representation/cache**, not the graph identity — build CSR or FixedAdj as needed
- **Almost no algorithms** — this is a data contract, not an algorithm library. Algorithms live in `pgraph_ops`
- **`Index = u32`** — sufficient for all SurfMol use cases (molecules < 1M atoms)
- **`INVALID = -1`** (i32) — ELLPACK convention, GPU-friendly, matches FireCore

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
- `pgraph_ops` — algorithms on positioned graphs (adjacency builders, components, bridges, reorder, geometry)
- `spacc` — spatial acceleration structures (AABB, Buckets)
- `moltopo` — chemistry-specific layer that will eventually use `pgraph`
