---
type: rust-crate
title: moltopo
description: Static molecular topology — bonds/angles/dihedrals, UFF type assignment, forcefield parameter files, dynamic atom state for MD, XYZ I/O. Foundation for forcefield param assignment.
tags: [rust, crate, topology, molecular-graph, uff, ssot, md]
timestamp: 2026-08-25
---

# moltopo

Static molecular topology and forcefield parameter management. This is the **single source of truth (SSOT)** for molecular connectivity — all other representations (rendering, forcefield params, export) derive from it. Mirrors the role of FireCore's `AtomicGraph`.

## Modules

- **`topology.rs`** — `Topology` struct: immutable dense arrays of `apos: Vec<Vec3d>`, `bonds: Vec<[i32;2]>`, `angles: Vec<[i32;3]>`, `dihedrals: Vec<Quat4i>`, `inversions: Vec<Quat4i>`. Provides `build_bonds_by_cutoff` (O(n²) distance bonding), `build_angles_from_bonds` (enumerates neighbor pairs per central atom), `build_dihedrals_from_bonds` (4-atom path enumeration with HashSet dedup using `l > i` convention — matches FireCore `UFFbuilder.h:1159` `i4 > i1` to avoid double-counting `(i,j,k,l)` and `(l,k,j,i)`), `build_inversions_from_bonds` (3 inversions per trigonal center — matches FireCore `UFFbuilder.h:1318-1332`). `hybridization(element, n_neigh)` computes octet-rule hybridization: `4 = nepair + nsigma + npi` where `nsigma = n_neigh`, returns sp3=3, sp2=2, sp=1.

- **`builder.rs`** — `Builder`: dynamic graph editor with generational arena pattern. `AtomH`/`BondH` are `(idx, gen)` handles that detect stale references after slot reuse. Supports soft-remove (marks dead, safe during iteration) and hard-remove (frees slot immediately, invalidates handles). `bake()` converts live atoms/bonds to dense `Topology` with derived angles/dihedrals/inversions. Max 4 neighbors per atom (fixed `[BondH; 4]` array). Includes hexagonal grid editing: `honeycomb_ring_nodes(q, r, a_cc)` returns 6 node positions at axial coordinates (pointy-top orientation), `add_hex_ring`/`remove_hex_ring`/`toggle_hex_ring` for graphene-like structure editing, `snap_to_node` for grid-snapped atom placement. Grid snapping uses 4-decimal precision (×10000) keys to avoid float comparison issues.

- **`params.rs`** — `Params`: loads forcefield parameters from `.dat` files (ElementTypes, AtomTypes, BondTypes, AngleTypes, DihedralTypes). `ElementType` includes covalent/vdW radii, UFF charges, QEq parameters, packed u32 color (0xRRGGBB). `AtomType` includes UFF and MMFF parameters. Lookup via HashMaps with wildcard matching: `get_bond_param(a, b, order)` uses sorted key for order-independence; `get_angle_param(a, b, c)` matches both (a,b,c) and (c,b,a); `get_dihedral_param` supports `*` and element-prefix wildcards. Also implements UFF formulas: `uff_bond_length` (bond order + electronegativity correction), `uff_bond_k`, `uff_angle_sp3/sp2/sp1` (Fourier coefficients), `get_reqh` returns `[r_vdw, sqrt(e_vdw), q_base, h_b]`. These formulas are consumed by `molff::uff::Uff::setup_params` (via `surfmol::MolWorld::setup_uff_params`) to fill all four UFF parameter arrays — see [`/userguide/uff_spff.md`](/userguide/uff_spff.md).

- **`assign_uff.rs`** — `assign_uff_types(elems, neighs)`: assigns UFF atom types from topology neighbor list and octet-rule hybridization. Special cases: H always "H_", nitro groups (N with 2 O neighbors → N_R/O_R with bond order 2), carbonyl O (O_2), alkyne C (C_1 with triple bond orders). General case maps hybridization to suffix: sp3→_3, sp2→_R (C/N/O) or _2, sp→_1. Falls back to unprefixed element name if suffixed type doesn't exist. The returned type strings are consumed by `molff::uff::Uff::setup_params` to select per-term parameters.

- **`molecular.rs`** — `Atoms` (static data: 64-byte-aligned `apos`, `neighs`, `neigh_bs`) and `DynamicAtoms` (adds `fapos`, `vapos` for MD). `make_neigh_bs` builds dual neighbor lists: `neighs[ia]` = atom indices, `neigh_bs[ia]` = bond indices — enables direct bond parameter lookup without searching. Uses `split_at_mut` for safe simultaneous mutable borrowing of two array elements. `move_atom_md` implements velocity Verlet with force clamping (`flim`) and damping; returns `(v·f, v·v, f·f)` for convergence checks. `run_md` takes a force-evaluation closure, decoupling the integrator from any specific forcefield.

- **`export.rs`** — JSON serialization/deserialization of `Topology` via serde. `TopologyData` includes positions, elements, bonds, angles, dihedrals, inversions. Forcefield params are placeholder (TODO). Dihedrals/inversions stored as `[i32; 4]` in JSON, converted to/from `Quat4i` internally.

- **`xyz.rs`** — `read_xyz`: parses XYZ format (atom count, comment, element x y z [charge]). Charge is optional (defaults to 0.0). `write_xyz_frame`: writes with 12.6 decimal precision, supports append mode for trajectories.

## Design decisions

- **Dual neighbor lists** (`neighs` + `neigh_bs`) — atom indices for geometry, bond indices for parameter lookup. Avoids searching bond array during force evaluation.
- **Fixed 4 neighbors per atom** (`Quat4i`, -1 padded) — cache-friendly, matches max valence for organic chemistry. Asserted on add.
- **64-byte alignment** on all arrays — matches cache line size for SIMD.
- **Generational arena in Builder** — `(idx, gen)` handles detect stale references after slot reuse without invalidating the entire arena.
- **Octet-rule hybridization** — chemical theory-based (`4 = nepair + nsigma + npi`), not empirical. Determines UFF type suffix.
- **Wildcard parameter matching** — `*` and element-prefix matching (e.g., "C" matches "C_3", "C_R") for sparse parameter tables.

## What does NOT belong here

- Forcefield energy/force evaluation → `molff` / `surfff`
- Rendering → `molrender`
- Math primitives → `numcore`
- MD orchestration (multi-forcefield coordination) → `surfmol`

## See also

- `ARCHITECTURE.md` §Component Details
- `Import_other_Repos.md` §2 (SPAMMM `AtomicGraph`)
- [`/userguide/uff_spff.md`](/userguide/uff_spff.md) — UFF/SPFF end-user guide (parameter pipeline using `Params` + `assign_uff_types`)
- [`/doc/topical_audit/uff.md`](/doc/topical_audit/uff.md) — UFF cross-implementation map
- `notes/designs/topology_builder.md` — future `pgraph`/`pgraph_ops`/`spacc` refactor
