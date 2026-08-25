---
type: report
title: numtypes rollout — current pgraph/spacc/numcore issues and refactor plan
description: Issues found in the current pgraph, pgraph_ops, spacc and numcore crates, and the plan for migrating to the new numtypes crate.
tags: [refactor, numtypes, pgraph, spacc, numcore, audit]
timestamp: 2026-08-25
---

# numtypes rollout — current `pgraph` / `spacc` / `numcore` issues

Extracted from `notes/chats/Reorg_and_Topology_mgraph.md` (lines 1720–5004) and verified against the current code.

## 1. Dependency graph is wrong

**Current state:**
```
numcore
   ↖ pgraph
   ↖ spacc → pgraph
   ↖ pgraph_ops → pgraph
```

`spacc/Cargo.toml` depends on `pgraph` only to get `Index` (which is just `u32`) and `IndexGroups`.

**Required state after numtypes:**
```
                         numtypes
                 _________|___________
                /         |           \
             numcore    pgraph        spacc
                \         |           /
                 \        |          /
                  \       |         /
                       moltopo
```

- `spacc` must depend on `numtypes` only, not `pgraph`.
- `pgraph` must depend on `numtypes`.
- `numcore` must depend on `numtypes`.
- `pgraph_ops` should be merged/replaced by `pgraph` (algorithms on `numtypes` data).

## 2. `numcore` mixes data and algorithms

**Current state (`numcore/src/`):**
- `math/vec3.rs`, `math/vec2.rs`, `math/quat4.rs` — data types that now live in `numtypes`.
- `util.rs` — `AlignedVec` now lives in `numtypes::alloc`.
- `math/fastmath.rs`, `math/linalg.rs`, `math/math3d.rs`, `math/math4d.rs` — genuine algorithms.

**Action:**
- Move `Vec2d`, `Vec3d`, `Quat4d`, `Quat4i`, `AlignedVec` to `numtypes`.
- `numcore` keeps `fastmath.rs`, `linalg.rs` (symmetric_eigen_3x3), and the f32 graphics helpers `math3d.rs`/`math4d.rs` (which may later move to `molrender` or `numtypes::mat`).

**Specific issue with `Quat4d`:**
It is a separate type from `Vec4d`. Discussion: remove `Quat4d`; use `Vec4d` with explicit `qmul`/`qconj`/`qrotate`. Same for `Quat4i` (it is just an `i32` pack, better as `Vec4i` or `[i32; 4]`).

**Specific issue with `Vec2d`:**
Currently has `mul_cmplx` as a method. Discussion: `*` must be component-wise; `cmul()` should be a standalone free function.

## 3. `pgraph` data structures need to move to `numtypes`

**Current `pgraph/src/lib.rs`:**
- `PGraph`, `PGraphView`, `Elements<N>`, `Ragged`, `Permutation`, `FixedRows<K>`, `FixedAdj<K>`, `CsrAdj`, `Partition`, `IndexGroups`, `RangeGroups`.

**Action:**
All of these are data contracts, not algorithms. They should live in `numtypes::graph`. The `pgraph` crate should become the *algorithm* crate (`pgraph_ops` content).

## 4. `Ragged` and `IndexGroups` are duplicates

**Current state:**
`Ragged` and `IndexGroups` are byte-for-byte the same concept (`offsets` + `items`).

**Action:**
Collapse to one `RaggedIndex` (already in `numtypes::graph`). If semantic aliases help, `pub type IndexGroups = RaggedIndex;` can be added.

## 5. `FixedRows<K>` is not aligned

**Current state:**
`FixedRows<K> { data: Vec<[i32; K]> }` uses ordinary `Vec`, so no 64-byte alignment.

**Why this matters:**
`FixedAdj<4>` is intended to be GPU/SIMD-friendly (`int4` per atom). For `K=4`, each row is 16 B; a 64 B base alignment guarantees every row is 16 B aligned.

**Action:**
`FixedRows<K>` in `numtypes::graph` already uses `AlignedVec<[i32; K], 64>`.

## 6. `build_fixed_adj` does unnecessary `O(K)` scanning

**Current `pgraph_ops/src/adjacency.rs`:**
After degree counting (O(E)), it calls `adj.neigh.push(...)` four times per edge. `push()` scans the row from slot 0 looking for the first `INVALID` sentinel.

**Why it is wrong / slow:**
- For `K=4` it is tolerable, but for `K=16/64` it is silly.
- The degree counts already computed are the exact write cursors.

**Action:**
Rewrite `build_fixed_adj` in `pgraph` (the algorithm crate) to use a `next[K]`/degree cursor:
```rust
let mut next = vec![0usize; nverts]; // or reuse counts
for (ie, &[a,b]) in edges.iter().enumerate() { ... adj.neigh.data[a][next[a]] = b as i32; next[a] += 1; ... }
```
Keep `FixedRows::push` for incremental small edits, but don't use it for bulk construction.

## 7. CSR and `Buckets` cursor clones allocate on every build

**Current `pgraph_ops/src/adjacency.rs`:**
```rust
let mut cursor = offsets[..nverts].to_vec();
```
This allocates and copies another `O(N)` array every CSR build.

**Current `spacc/src/buckets.rs`:**
```rust
let mut cursor = self.cell_i0s.clone();
```
Same problem: every spatial rebuild allocates.

**Correct pattern:**
Reuse the count buffer as the cursor after computing offsets:
```text
count degrees
prefix -> offsets
counts.fill(0)
scatter at offsets[v] + counts[v]
```
`numtypes::RaggedIndex` builders in `pgraph` should follow this. `spacc::Buckets` should use `cell_ns` (counts) as the cursor.

## 8. `spacc::Buckets` representation is suboptimal

**Current state:**
```rust
pub struct Buckets {
    pub ncells: usize,
    pub cell_ns:  Vec<i32>,
    pub cell_i0s: Vec<i32>,
    pub cell2obj: Vec<i32>,
    pub nobjs: i32,
}
```

**Issues:**
- Duplicated query state: `cell_ns + cell_i0s`.
- `nobjs` is redundant (length of `cell2obj`).
- Allocates cursor clone on every `scatter`.

**Action:**
Simplify to the CSR/Ragged convention:
```rust
pub struct Buckets {
    pub counts:  Vec<u32>,  // rebuild scratch
    pub offsets: Vec<u32>,  // ncells+1
    pub items:   Vec<u32>,  // packed valid items
}
```
`cell(c) = &items[offsets[c]..offsets[c+1]]`.

## 9. `spacc` `Aabb` is not `#[repr(C)]` and allows empty groups silently

**Current state:**
```rust
#[derive(Clone, Debug)]
pub struct Aabb { pub lo: Vec3d, pub hi: Vec3d }
```
- Not `#[repr(C)]` — bad for GPU/FFI.
- `center()` on an empty AABB (lo=+inf, hi=-inf) produces `NaN`.
- `fit_group_aabbs` explicitly writes `Aabb::empty()` for empty groups, silently creating NaN-producing boxes.

**Action:**
`Aabb3d` is now `Vec6d` in `numtypes::spatial`, `#[repr(C)]`. Add `aabb_is_valid()` to reject empty boxes at bake time. `fit_group_aabbs` in `spacc` should either reject empty groups or document the contract and panic/warn.

## 10. `split_by_component` returns `Vec<Vec<Index>>` instead of packed groups

**Current `pgraph_ops/src/components.rs`:**
Returns one `Vec<Index>` per component, each a separate heap allocation.

**Why it is wrong:**
This is exactly the allocation pattern `RaggedIndex`/`IndexGroups` was designed to avoid.

**Action:**
Return `RaggedIndex` (or `IndexGroups` alias) with `offsets`/`items`.

## 11. `permute_edges` only remaps endpoints, does not reorder edges

**Current `pgraph_ops/src/reorder.rs`:**
```rust
edges.iter().map(|&[a, b]| [perm.old2new[a], perm.old2new[b]]).collect()
```

**Issue:**
It remaps vertex IDs but does not sort/reorder the edge stream. After packing atoms by fragment, the edge stream is still in historical insertion order, not grouped by fragment or first endpoint.

**Action:**
Optionally sort edges by `(min(a,b), max(a,b))` or by first endpoint to improve cache locality. Provide a separate `reorder_edges` path.

## 12. Fail-fast validation holes

| Location | Hole | Consequence |
|---|---|---|
| `pgraph::Permutation::from_new2old` | accepts duplicates/missing/invalid indices | silently produces wrong `old2new` |
| `pgraph::Ragged::ngroups`, `IndexGroups::ngroups`, `CsrAdj::nverts` | assume `offsets.len() >= 1` | panic with generic bounds error on empty input |
| `pgraph_ops::adjacency` builders | index edges before checking `a,b < nverts` | generic bounds panic, no context |
| `spacc::Buckets` | no check that `cell_of_obj.len() == nobjs` or `0 <= cell < ncells` | generic panic |
| `pgraph::PGraph::validate` | does not check position finiteness | NaNs can enter force kernels |

**Action:**
Validation should happen at construction/bake boundaries in `pgraph` algorithms with explicit messages and variable values. `numtypes::graph::PGraph::validate` already checks finite positions; `numtypes::graph::Permutation::from_new2old` already validates duplicates/range.

## 13. No contiguous-fragment fast path in `spacc`

**Current `spacc/src/aabb.rs`:**
Only `fit_group_aabbs(pos, IndexGroups, out)` with indexed gathers.

**Action:**
Add `fit_range_aabbs(pos, ranges, out)` for packed fragments (direct sequential `pos[i0..i1]` read). This is the cache-optimal path for compiled fragments of 16/32/64/128 atoms.

## 14. `math3d.rs` / `math4d.rs` do not use the new `Vec`/`Mat` types

**Current state:**
- `math3d.rs` uses raw `[f32; 3]` helpers.
- `math4d.rs` uses `[[f32; 4]; 4]` for look-at/ortho/projection.

**Action:**
`Mat3f`/`Mat4f` should eventually replace these. Graphics functions (`look_at`, `ortho`) depend on handedness/NDC conventions, so they may belong in `molrender` or a `numcore` graphics module, not in the generic math library.

## 15. `AGENTS.md` / `DESIGN_GOALS.md` unsafe rule is contradicted by `AlignedVec`

**Current text:**
"only OpenCL crate uses `unsafe`" (`AGENTS.md` line 82) and "all `unsafe` confined to a single feature-gated OpenCL crate" (`DESIGN_GOALS.md` line 64).

**Reality:**
`numcore/src/util.rs` already uses `std::alloc` and unsafe blocks for aligned allocation.

**Action:**
Replace with a new policy: `unsafe` allowed in low-level foundational modules (`numtypes` alloc, OpenCL FFI) with small documented `// SAFETY:` blocks; expose safe APIs upward. Do not use `unsafe` to bypass borrow checking without measured need.

## 16. Next steps (refactor order)

1. **Migrate `numcore`** to depend on `numtypes`:
   - Remove `math/vec2.rs`, `math/vec3.rs`, `math/quat4.rs`, `util.rs`.
   - Keep `math/fastmath.rs`, `math/linalg.rs`; optionally keep `math3d`/`math4d` until `Mat3f`/`Mat4f` exist.
2. **Replace `pgraph` data + `pgraph_ops` algorithms:**
   - `crates/libs/pgraph` becomes the graph algorithm crate.
   - Import data types from `numtypes::graph`.
3. **Refactor `spacc`:**
   - Remove `pgraph` dependency.
   - Use `numtypes::Aabb3d` and `numtypes::RaggedIndex`.
   - Simplify `Buckets`; add `fit_range_aabbs`.
4. **Update all call sites** in `moltopo`, `molff`, `molrender`, `molgui` to import from `numtypes` or `numcore` as appropriate.
5. **Run the full test suite** (`cargo test --workspace`) and forcefield parity checks.
