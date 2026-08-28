---
type: work-notes
title: Cluster AABB broad-phase collision for non-bonded interactions
description: Evidence-based design for accelerating non-covalent (LJ/Coulomb/Morse/compact-exp) pair evaluation in SurfMol using per-cluster AABBs and the existing spacc crate. Cross-implementation map of FireCore, SPAMMM, and NumericalMathPlayground.
tags: [work-in-progress, design, rust, spacc, aabb, nonbonded, firecore, spammm, collision, broad-phase]
timestamp: 2026-09-28
---

# Cluster AABB broad-phase collision for non-bonded interactions

**Status:** **Implemented** (CPU broad phase + GUI visualization). Steps 1–5 of §7 are done. Steps 6–7 (Buckets acceleration, OpenCL port) remain.

**Goal:** replace the current O(N²) all-pairs non-bonded loop in `molff::nonbonded` with a two-stage **broad-phase AABB cull → narrow-phase pair eval**, so that the editor and later the OpenCL path can scale to multi-molecule and on-surface systems without paying for pairs that are far outside cutoff.

**References:**
- FireCore: `cpp/common/molecular/NBFF.h`, `pyBall/RigidAtomFF/RRsp3/RRsp3.cl`, `pyBall/RigidAtomFF/RRsp3/RRsp3.py`, `pyBall/RigidAtomFF/shared/XPTB_utils.py`, `cpp/common/math/Forces.h`, `doc/DevNotes/ToDo_FastCollision_2.md`, `ToDo_FastCollision_3.md`.
- SPAMMM: `kernels/rigid.cl`, `kernels/nonbonded.cl`, `spammm/forcefields/RigidBodyDynamics.py`, `spammm/forcefields/Assembly.py`, `doc/ARCHITECTURE_ROADMAP.md`, `doc/Tasks/MultiMol_MD_LaunchOverhead.md`.
- NumericalMathPlayground: `topics/ReactiveFF/BoundingBoxBalancing.md`, `py/DFTB/Grid_dftb.py`, `topics/Clustering/Clustering.py`.
- SurfMol: `crates/libs/spacc/src/{aabb,buckets,lib}.rs`, `crates/libs/numtypes/src/spatial.rs`, `crates/libs/molff/src/nonbonded.rs`, `notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md`.

## 1. The problem

`molff::nonbonded::eval_nonbonded` currently iterates every atom against every other atom (with 1-2/1-3 exclusion and PBC), i.e. O(N²). For a single small molecule in the editor this is fine. For the planned multi-molecule / on-surface use cases it is the same cliff SPAMMM already hit:

> SPAMMM `doc/Tasks/MultiMol_MD_LaunchOverhead.md:503-505`: *"Kernel time grows as O(n_mol²)"*, recommends *"Neighbor lists / cutoff — skip far-away molecule pairs."*

FireCore solved this two ways (C++ `NBFF` bucket search and OpenCL `RRsp3` ghost-atom list), both built on the same idea: **fit a tight AABB per cluster/molecule, expand by the interaction cutoff, and only evaluate atom pairs whose cluster AABBs overlap.**

## 2. Cross-implementation evidence

### 2.1 FireCore — two working realizations

#### 2.1a C++ `NBFF` — tight position-only AABBs + bucket-pair search

- **AABB fit** (`NBFF.h:24-41`): `fitAABB` walks `c2o` indices into `apos` and does `setIfLower`/`setIfGreater`. **Position-only** — no radius expansion at fit time; the margin is added at search time.
- **Per-bucket AABBs** (`NBFF.h:300-314`, `updatePointBBs`): one AABB per bucket, buckets come from `atom2group` (`initBBsFromGroups:274-297`). One bucket = one molecule/cluster.
- **Broad phase** (`evalSortRange_BBs:355-409`): for each bucket pair `(ib, jb<ib)`, expand each AABB by `Rcut` (`bb.lo.add(-Rcut); bb.hi.add(Rcut)`) and `selectInBox` to get the candidate atoms.
- **Narrow phase**: `repulsion_R4` (`Forces.h:102-126`) on every surviving pair, with `R_cut = R_i + R_j` (sum of VdW radii) and inner core `R = R_cut - drSR`, `drSR = 0.5 Å`. For C–C (R≈1.7 Å) this gives outer cutoff ≈ 3.4 Å, core ≈ 2.9 Å — the "3–4 Å scale".

#### 2.1b OpenCL `RRsp3.cl` — radius-aware AABB per workgroup + ghost list

- **AABB fit** (`RRsp3.cl:400-439`, `update_bboxes_rigid`): each workgroup thread builds a **radius-aware** per-atom box `p ± r`, then a parallel min/max reduction in local memory stores the group AABB. The stored box already includes VdW radii.
- **Cluster packing** (`XPTB_utils.py:32-170`, `pack_molecules_contiguous`): one workgroup = one cluster of `GROUP_SIZE=64` atoms, nodes-first then caps then dummy `'X'` padding.
- **Broad phase** (`RRsp3.cl:452-558`, `build_local_topology_rigid`): AABB-against-AABB overlap with `bbox_margin` (`bboxes_overlap:123-128`), then a second conservative test — an atom from the other group becomes a "ghost" if its distance to *my group's AABB* is `< margin_sq` where `margin_sq = (2·rmax + bbox_margin)²` (`RRsp3.py:442-444`). Ghosts land in a fixed `MAX_GHOSTS` local list via `atomic_inc`.
- **Narrow phase** (`RRsp3.cl:560-654`, `compute_collision_cluster_rigid`): exact sphere-sphere `d² < (ri+rj)²`, with 1-2/1-3 exclusion (`excl1_local`, `excl2_local`) filtered before the narrow test.
- **Margin values**: `bbox_margin = 0.5 Å` default; hard-sphere collision uses `ri + rj`.

#### 2.1c FireCore split-potential design (relevant to axis 3b of the roadmap)

`ToDo_FastCollision_2.md` / `ToDo_FastCollision_3.md` propose a split potential for fast collisions: linear harmonic part for `r < R_cut` (solved in the PD/Jacobi linear solver) + non-linear smoothing for `R_cut < r < R_cut2` (explicit force) + zero beyond `R_cut2`. Implemented as `getSR_x2_smooth` (`Forces.h:511-542`) with parameter matching `computeMatchingParams` (`Forces.h:555-570`). The docs explicitly say the **same `NBFF` AABB/bucket infrastructure** is the intended broad phase. This is the planned projective-dynamics extension; the working `RRsp3.cl` is the hard-sphere realization.

### 2.2 SPAMMM — broad phase NOT yet in production force code

- **No production per-rigid-body AABB** for molecule-molecule collision. The AABB code that exists is:
  - GUI fragment boxes (`spammm/GUI/FragmentExtension.py:138`, `pad=0.3 Å`).
  - SPFF z-confinement `bboxes` (`kernels/SPFF.cl:1068`) — only z-axis enforced, and **not even uploaded** (`pack_system` has no `toGPU('bboxes', ...)`).
  - Experimental, **broken/inverted** bucket kernels `getShortRangeBuckets` / `getShortRangeBuckets2` (`kernels/nonbonded.cl:315,445`) — comments at `nonbonded.cl:75-94` flag known issues (comparison inverted / AABBs possibly swapped), and there is **no Python call site**.
- **Real collision detection** is atom-atom sphere clash in assembly search (`kernels/assembly.cl:133-141`: `dist_sq < (ri+rj)²`, `overlap = rsum - dist`, score `+= overlap²`, early abort on `max_clash_penalty`), and the mostly-disabled `clash_r2` in `RigidBodyDynamics` (defaults `0.0`).
- **Non-bonded force path is O(N²)**: `RigidBodyPairFF` / `rigid.cl` loops every molecule against every other, every atom against every atom. `doc/nonbonding_forcefields.md:74` mentions "AABB Acceleration" and a `Buckets`/`pointBBs` spatial hash as a *planned* optimization.
- **DFTB grid sphere-AABB** (`kernels/LCAO_grid.cl:537-649`) is a correct `clamp(pos, b_min, b_max)` test, but for quantum projection, not classical FF.
- **Roadmap explicitly lists** "Implement broad-phase collision detection (AABB overlap test)" under *Fast Relaxation with Collision Groups* (`doc/ARCHITECTURE_ROADMAP.md:316-351`), pointing at `BoundingBoxBalancing.md`.

**Takeaway:** SPAMMM is the negative result — the broad phase was designed but never wired in, and the O(N²) cliff is documented. SurfMol should not repeat this; we should wire the broad phase before scaling up.

### 2.3 NumericalMathPlayground — design only, no implementation

- `topics/ReactiveFF/BoundingBoxBalancing.md` is a **design document, not code**. It contains the full strategy but nothing is implemented in NMP.
- **Core strategy** (lines cited in §1 of the NMP report):
  - Per-particle AABB with interaction margin `R_i`: `l_g = min_i(x_i - R_i)`, `u_g = max_i(x_i + R_i)`.
  - Pair-cost objective: `C_pair = Σ_{g<h} I(A_g ∩ A_h ≠ ∅) n_g n_h + Σ_g n_g(n_g-1)/2`. For fixed group size `W`, `C_pair ≈ W² · N_overlap`.
  - **Primary method**: periodic global rebuild with **Morton/Hilbert ordering** + fixed chunks of size `W`.
  - **Maintenance method**: pairwise `2W → W+W` retiling via longest-axis median split.
  - **Quality triggers**: rebuild when `Q(t) > (1+ε)·Q(t_last_rebuild)`, with `Q ∈ {N_overlapping_group_pairs, N_candidate_pairs, Σ S_g (surface area), max_g V_g/V_g0}`.
- **Conclusion of the doc**: opportunistic insertion is fragile, single-particle swaps are weak; the clean architecture is *periodic Morton/grid rebuild + fixed W-particle groups + occasional 2W→W+W local retiling*.
- NMP has only support-level AABB code: `Grid_dftb.py:240-244` (sphere-AABB overlap), `Clustering.py:194-254` (KD-tree with per-node AABB), `Clustering.py:328-387` (ball tree). None of it balances GPU workgroups. Workgroups in NMP are static per-molecule with fixed `WG=32`.

## 3. What SurfMol already has (`spacc` + `numtypes`)

The primitives needed for a CPU broad phase are **already implemented and tested**:

`crates/libs/numtypes/src/spatial.rs` (intrinsics on `Aabb3d = Vec6d` {a=lo, b=hi}):
- `aabb_empty()` — `(+Inf, -Inf)` sentinel.
- `aabb_include(&mut bb, p)` — grow to include point.
- `aabb_include_aabb(&mut bb, other)` — grow to include another AABB.
- `aabb_contains(bb, p)` — point-in-box.
- `aabb_overlap(a, b)` — box-box overlap (no margin).
- `aabb_merge(a, b)` — union.
- `aabb_center(bb)`, `aabb_size(bb)`, `aabb_max_extent(bb)`, `aabb_is_valid(bb)`.

`crates/libs/spacc/src/aabb.rs` (fitting algorithms):
- `fit_aabb(pos, ids)` — tight AABB over selected positions.
- `fit_group_aabbs(pos, groups: &RaggedIndex, out)` — one AABB per ragged group.
- `fit_range_aabbs(pos, ranges: &[[Index;2]], out)` — one AABB per contiguous range (cache-optimal for packed fragments).

`crates/libs/spacc/src/buckets.rs` (spatial hash):
- `Buckets { ncells, offsets, items, counts }` — count → prefix → scatter build, no allocation during rebuild, `cell_of_obj[i] = -1` skips. `cell_objects(c)` returns packed slice.

**What is missing in `spacc`:**
- No `aabb_overlap_margin(a, b, margin)` — trivial to add (expand one box by `margin` then call `aabb_overlap`, or inline the 6 comparisons like `RRsp3.cl:123-128`).
- No `aabb_sphere_overlap(bb, center, r)` — needed if we ever do atom-vs-cluster tests like `RRsp3.cl`'s ghost test; trivial (`d² = Σ max(0, max(lo-c, c-hi))²`, compare to `r²`), see `Grid_dftb.py:240-244` / `LCAO_grid.cl:537-649`.
- No Morton key / radix-sort packing — only needed for the NMP dynamic-rebalance strategy, **not** for the first static-per-molecule broad phase.
- No `Buckets`-driven neighbor pair iterator — the structure exists, the iteration pattern does not.

## 4. Proposed design (CPU first, OpenCL-shaped)

### 4.1 Scope and non-goals

- **In scope:** a CPU broad phase for `molff::nonbonded` that, given per-cluster AABBs and an interaction cutoff, produces a candidate pair list (cluster-pair, then atom-pair) that the narrow phase evaluates. This unblocks multi-molecule editor scenarios and gives the OpenCL port a clear contract.
- **Out of scope (for now):** dynamic Morton rebalancing, `2W→W+W` retiling, projective-dynamics split potential. These are roadmap axis 3b/4 and depend on this broad phase existing first.

### 4.2 Data flow

```
positions (DynamicAtoms) ──┐
                           ├── fit_range_aabbs / fit_group_aabbs ──▶ cluster_aabbs: Vec<Aabb3d>
cluster ranges (topology) ─┘                                              │
                                                                          ▼
                          aabb_overlap_margin(a, b, Rcut) ──▶ overlapping cluster pairs
                                                                          │
                                                                          ▼
                          for each (i,j) cluster pair: atom-atom loop with cutoff ──▶ narrow phase (existing eval_nonbonded pair kernel)
```

- **Cluster = molecule** for now (one AABB per molecule, matching FireCore `NBFF` and SPAMMM's per-body model). The `RaggedIndex`/range form already in `spacc` supports this directly: topology already knows which atoms belong to which molecule.
- **AABBs are a rebuildable cache**, invalidated by geometry changes — exactly the contract `spacc` declares in its module doc. Rebuild once per relaxation step (or less often if clusters barely move).
- **Margin = interaction cutoff `Rcut`**, applied at overlap-test time, not at fit time. This matches `NBFF` (`bb.lo.add(-Rcut)`) and keeps the cached AABB tight. `Rcut` is a single non-bonded parameter (max pair cutoff, e.g. `2·rmax + drSR` or the compact-exp `rc`).

### 4.3 Concrete additions to `spacc`

1. `aabb_overlap_margin(a: Aabb3d, b: Aabb3d, margin: f64) -> bool` — inline 6 comparisons, mirroring `RRsp3.cl:123-128`. Add to `numtypes::spatial` (it's an intrinsic).
2. `aabb_sphere_overlap(bb, c, r) -> bool` — for the optional atom-vs-cluster ghost test. Add to `numtypes::spatial`.
3. `broad_phase_pairs(cluster_aabbs: &[Aabb3d], rcut: f64) -> Vec<(u32,u32)>` — in `spacc` (algorithm, not intrinsic). Returns overlapping cluster-pair indices `i < j`. This is the CPU equivalent of `RRsp3.cl`'s group-pair loop. Optionally takes a `&Buckets` for O(N) instead of O(N²) cluster-pair search when N_clusters is large.
4. *(Optional, later)* `Buckets`-based `cell_of_obj` from cluster COM + a grid cell size `≥ Rcut`, then iterate neighbor cells. This is the `NBFF`/`pointBBs` route and matters once N_clusters > ~100.

### 4.4 Integration into `molff::nonbonded`

- `eval_nonbonded` gains an optional `cluster_aabbs: &[Aabb3d]` (or a small `BroadPhase` struct holding AABBs + `Rcut`). When present, it loops only over `broad_phase_pairs` and within each cluster pair loops atoms (with the existing 1-2/1-3 exclusion). When absent, falls back to current O(N²) — keeps single-molecule editor path unchanged and tests green.
- **Fail-loud invariant:** if `cluster_aabbs.len() != n_clusters` or any AABB is invalid (`!aabb_is_valid`), panic with context (per AGENTS.md). Never silently skip the broad phase.
- **Parity check:** the broad-phase-on path must produce **exactly the same forces/energy** as the O(N²) path (same pairs, just fewer iterations). Add a test that runs both on a 2–3 molecule system and asserts `force_l2_diff < eps` and `|E_diff| < eps`. This is the numerical-parity skill's bread and butter.

### 4.5 OpenCL shape (for axis 4, later)

The CPU design maps cleanly onto the `RRsp3.cl` layout (roadmap variant 4b):
- `update_bboxes_rigid` → per-workgroup radius-aware AABB reduction (already designed in FireCore).
- `build_local_topology_rigid` → AABB-overlap + ghost-atom test → local `MAX_GHOSTS` list.
- `compute_collision_cluster_rigid` → narrow phase with exclusion.
The CPU `broad_phase_pairs` is the host-side reference for the OpenCL kernel's group-pair loop, so the parity target is well-defined from day one.

### 4.6 Visualization (debugging)

FireCore's vispy harness (`pyBall/VispyUtils.py:900-926`, `test_RRsp3_vispy.py:530-545`) draws each AABB as 12 edges with three display modes: `tight`, `overlap` (expand by `bbox_margin`), `full` (expand by `2·rmax + bbox_margin`). SurfMol's editor already renders ports; adding an AABB-edge debug overlay (gated behind a GUI checkbox / `--show-aabb` flag) is the direct analogue and will be essential for verifying the broad phase visually. This is a follow-up, not part of the first cut.

## 5. Decision summary

| Question | Decision | Rationale |
|---|---|---|
| Per-atom or per-cluster AABB? | **Per-cluster (per molecule)** for the broad phase. | Matches FireCore `NBFF` and SPAMMM per-body model; `spacc::fit_range_aabbs` already supports it; cluster count ≪ atom count. |
| Radius-aware or position-only AABB? | **Position-only at fit time, margin at query time.** | Matches `NBFF`; keeps cached AABB tight; one `Rcut` parameter controls the margin. (Radius-aware `p±r` is an OpenCL optimization for the kernel-side reduction.) |
| O(N²) cluster-pair or `Buckets` grid? | **O(N²) cluster-pair first**, `Buckets` optional for N_clusters > ~100. | N_clusters is small in editor; O(N²) over clusters is trivially fast and exact; `Buckets` already exists when needed. |
| Dynamic rebalancing (Morton/retiling)? | **No, not now.** | NMP design is design-only; static per-molecule clusters are correct and sufficient until we hit many-cluster on-surface systems. Defer to roadmap axis 4. |
| Where do the new primitives live? | `aabb_overlap_margin` / `aabb_sphere_overlap` → `numtypes::spatial` (intrinsics); `broad_phase_pairs` → `spacc` (algorithm). | Matches existing crate split (`spacc/lib.rs` doc: "AABB, Buckets, grids"; intrinsics in `numtypes`). |
| Narrow phase? | **Unchanged** — reuse existing `eval_nonbonded` pair kernel. | Surgical edit; broad phase only decides *which* pairs to evaluate. |

## 6. Open questions (for USER)

1. **Cluster definition:** is "one molecule = one cluster" the right granularity, or do we want sub-molecule clusters (e.g. per-residue, or the port-based clusters from `RaffTopology`)? The `RaffTopology` already has port-grouped atoms which could be the cluster unit for port-based non-bonded — but that changes the cutoff semantics. **Default: one molecule = one cluster; revisit when port-based non-bonded lands.**
2. **Cutoff source:** single global `Rcut` (max over all pair types) or per-pair-type cutoffs combined into one effective `Rcut` for the broad phase? FireCore uses `R_i + R_j` per pair; the broad phase needs a single conservative `Rcut = max_ij(R_i+R_j) + drSR`. **Default: single conservative `Rcut`.**
3. **PBC:** the current `eval_nonbonded` handles periodic boundary conditions. The broad phase must respect PBC too (AABBs near a periodic boundary wrap). `spacc` has no PBC-aware AABB yet. **Need to decide: minimum-image AABB expansion, or skip PBC in the first cut (editor is usually non-periodic)?**

## 7. Implementation order (proposed, not started)

1. ✅ Add `aabb_overlap_margin` (+ test) to `numtypes::spatial`. — **Done** (also added `aabb_point_dist2`, `aabb_sphere_overlap`).
2. ✅ Add `broad_phase_pairs(cluster_aabbs, rcut)` (+ test) to `spacc`. — **Done** (also added `aabb_edges` for visualization).
3. ✅ Add a parity test in `molff` that builds a 2–3 molecule system, runs `eval_nonbonded` with and without the broad phase, asserts identical forces/energy. — **Done** (`tests/test_broad_phase.rs`, 3 tests passing).
4. ✅ Wire the broad phase into `eval_nonbonded` behind an optional `BroadPhase` argument (no behavior change when `None`). — **Done** (`NonBondedFF::eval_broad`, `raff::eval_nonbonded_broad`, `MolWorld::eval_forces_broad`).
5. ✅ Wire it into the editor for `--raff` multi-molecule mode; add `--show-aabb` debug overlay. — **Done** (`--nmols N`, `--layout lattice|random`, `--show-aabb`, `A` key toggle, GUI checkbox).
6. *(Later)* `Buckets`-accelerated cluster-pair search for large N_clusters.
7. *(Later, axis 4)* OpenCL `update_bboxes_rigid` + `build_local_topology_rigid` port, with the CPU path as parity reference.

Each step is independently testable and keeps the existing O(N²) path as a fallback/parity check, per the AGENTS.md "Tests Are Diagnostics" rule.

## See also

- [`/doc/topical_audit/spatial_acceleration.md`](/doc/topical_audit/spatial_acceleration.md) — cross-implementation map (updated with broad-phase collision)
- [`/userguide/editor.md`](/userguide/editor.md) — end-user guide with `--nmols`/`--show-aabb` examples
- [`/crates/libs/molff/tests/test_broad_phase.rs`](/crates/libs/molff/tests/test_broad_phase.rs) — parity tests (3 tests, all passing)
- [`/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md`](/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md) — RAFF roadmap (AABB broad-phase marked done)
