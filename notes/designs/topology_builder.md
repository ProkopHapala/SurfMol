---
type: work-notes
title: Positioned graph and molecular topology builder design
description: Minimal pgraph/spacc foundation and simplified SurfMol molecular topology builder, derived critically from FireCore and SimpleSimulationEngine.
tags: [work-in-progress, design, rust, topology, pgraph, spacc, firecore]
timestamp: 2026-08-25
---

# Positioned graph and molecular topology builder design

**Status:** proposal; no implementation changes yet.

**References:** FireCore `Atoms.h`, `MMFFBuilder.h` (same lineage called `MMFFBuilderBase.h` in some local notes/branches), `MolecularGraph.h`, `Groups.h`, `NBFF.h`, `UFF.h`, `MMFFsp3_loc.h`; SimpleSimulationEngine `MeshBuilder2.h`, `CMesh.h`, `Buckets.h`, `NeighChunks.h`.

This supersedes the earlier idea that `mgraph` should mean "the dynamic molecular editor". The reusable primitive is **positioned indexed connectivity**; editing, spatial acceleration and chemistry are layers around it.

## 1. Core idea and naming

A molecule, polygon mesh and truss share more than an abstract graph:

```text
pos[i]             geometry of vertex i
edges[e] = (i,j)   connectivity
```

Geometry is essential because shared algorithms need edge vectors/lengths, normals/local frames, bounding boxes, picking, SDF selection, spatial hashing and geometric editing heuristics.

**Recommendation:**
- `pgraph` = *positioned graph* (or physical graph): tiny data contract.
- `pgraph_ops` = reusable geometry/topology/edit algorithms.
- `spacc` = *spatial acceleration*: AABB, Buckets, grids, Morton, broad phase.
- `moltopo` = chemistry-specific topology/builder and compilation.

`pgraph` communicates the defining property better than `mgraph`, which is ambiguous (molecule/mesh/material graph).

## 2. Dependency principle: data contract != algorithm library

The previous workspace proposal mixed a ubiquitous tiny representation with a larger editor/algorithm package. Separate them:

```text
                    numcore
                  /    |    \
             pgraph   spacc  molrender
                \      /
                 pgraph_ops
                     |
                  moltopo
                 /      \
              molff    molgui
                 \      /
                  surfmol
```

Exact Cargo edges can be even narrower because most functions can accept slices.

- **`pgraph`**: boring POD-like index containers/views; almost no algorithms.
- **`pgraph_ops`**: adjacency, components, bridges, loops, reorder/compact, selection, picking, geometry, editing helpers.
- **`spacc`**: rebuildable spatial caches; no molecular semantics.
- **`moltopo`**: atom/bond/valence/type/fragment semantics and molecular builder.

## 3. Fundamental `pgraph` structures

Do not make one giant configurable Graph class. Define a small family of composable arrays.

```rust
pub type Index = u32;

pub struct PGraph {
    pub pos:   Vec<Vec3d>,
    pub edges: Vec<[Index; 2]>,
}

pub struct PGraphView<'a> {
    pub pos:   &'a [Vec3d],
    pub edges: &'a [[Index; 2]],
}

pub struct Elements<const N: usize> {
    pub verts: Vec<[Index; N]>,
}

pub struct Ragged {
    pub offsets: Vec<Index>,
    pub items:   Vec<Index>,
}

pub struct Permutation {
    pub old2new: Vec<Index>,
    pub new2old: Vec<Index>,
}
```

`PGraph` invariants: dense vertex/edge ids; valid edge endpoints. Atom types, materials, flags, charges, colors, etc. are sidecar arrays indexed by the same ids.

`Ragged` is the generic count/offset/packed-items primitive behind polygon loops, arbitrary groups, cycle lists and other variable-length sets. `PGraphView` is the safe Rust analogue of the useful part of SSE `CMesh`: lightweight borrowed arrays, not ownership hierarchy.

## 4. Triangles vs angles; polygons vs rings

Share **storage and algorithms**, not semantics.

- Mesh triangle: explicitly declared 2-cell/face.
- Molecular angle: normally a derived length-2 path `(i-j-k)`.
- Polygon: explicitly declared face boundary.
- Molecular ring: discovered cycle; graphs can have many cycles and no unique face/ring set without extra rules.

Therefore these are **not fields of `PGraph`**. They can use common containers:

```text
Elements<3>   triangles OR molecular angle triples
Elements<4>   tetrahedra OR dihedral/inversion quadruples
Ragged        polygon loops OR ring/cycle sets
```

`pgraph_ops` should provide tuple/loop/cycle machinery; mesh code and `moltopo` decide meaning and generation rules.

## 5. Neighbor representations are first-class data structures

This is performance-critical. We need at least two representations generated from the same edge list.

### 5.1. Fixed-stride padded adjacency: `FixedAdj<K>`

For bounded degree:

```text
v0:  4  8 11 -1
v1:  0  7 -1 -1
v2:  5 -1 -1 -1
```

Valid entries are packed first, unused slots are `-1`, every row has stride `K`. This is **ELLPACK/ELL-like, not CSR**.

```rust
pub struct FixedRows<const K: usize> {
    pub data: Vec<[i32; K]>,
}

pub struct FixedAdj<const K: usize> {
    pub neigh: FixedRows<K>, // neighbor vertex/atom
    pub edge:  FixedRows<K>, // corresponding edge/bond
}
```

Keeping both tables is intentional: FireCore UFF already has `neighs[natom]` + `neighBs[natom]`; a hot kernel should not search again for the bond id.

Properties: constant indexing, easy unrolling, GPU-friendly, predictable cache footprint. Builder must **fail loud** on degree `>K`; never truncate.

Likely choices:
- organic molecules: `K=4`;
- models needing higher coordination: `K=8` if justified;
- mesh/truss kernels: `K=8/16/32/64` only when regularity makes padding worthwhile.

Canonical OpenCL ABI can simply be flat `i32[n*K]`; vector loads (`int4`) are an optimization, not a core Rust type requirement.

### 5.2. True CSR: `CsrAdj`

```rust
pub struct CsrAdj {
    pub offsets: Vec<Index>, // nvert+1
    pub neigh:   Vec<Index>, // 2*nedges for undirected graph
    pub edge:    Vec<Index>, // matching edge ids
}
```

Build by count -> prefix -> scatter, exactly the pattern in FireCore `MolecularGraph::makeNeighbors()` and SSE/FireCore `Buckets`.

CSR is compact and handles arbitrary degree; it is the natural representation for generic topology algorithms and irregular meshes.

### 5.3. Dynamic adjacency

Do not impose one dynamic structure on all domains.

For the **molecular builder**, chemical valence is small: keep four generational bond handles directly in each atom Conf. For a future high-degree mesh editor, a `NeighChunks`-like fixed-prefix + overflow structure may be useful. Add it when that use case exists.

Shared conversions in `pgraph_ops`:

```rust
build_csr_adj(nvert, edges) -> CsrAdj
build_fixed_adj::<K>(nvert, edges) -> Result<FixedAdj<K>, DegreeOverflow>
```

Adjacency is thus a selectable representation/cache, not the identity of the graph.

## 6. Groups, fragments and spatial bounds

Several different concepts are often all called "group":

1. semantic fragment/residue/rigid unit;
2. topological component/ring/patch;
3. computational/GPU/collision block;
4. spatial-grid/BVH group;
5. user selection.

Reuse index containers but do not force one meaning.

### 6.1. Generic disjoint partition

FireCore repeatedly uses atom->group plus a reverse packed mapping. Make this generic:

```rust
pub struct Partition {
    pub item_group: Vec<i32>, // -1 unassigned
}

pub struct IndexGroups {
    pub offsets: Vec<Index>,
    pub items:   Vec<Index>,
}

pub struct RangeGroups {
    pub ranges: Vec<[Index; 2]>, // [i0,n] after contiguous packing
}
```

`Partition -> IndexGroups` is another count->prefix->scatter operation. FireCore `Groups::setGroupMapping()` and `NBFF::initBBsFromGroups()` independently implement this pattern.

### 6.2. Edit form -> packed simulation form

Use flexible assignment while editing, then reorder when compiling:

```text
Partition
   -> group-aware Permutation
   -> reorder pos/edges/all sidecars
   -> RangeGroups
```

This satisfies both easy topology editing and the preference for contiguous fragments in CPU/GPU evaluation.

### 6.3. Bounds belong to `spacc`

Do not put an AABB into `PGraph` and do not make it intrinsic `Fragment` state.

```text
positions + IndexGroups/RangeGroups
             |
             v
       spacc::fit_aabbs
             |
             v
         Aabb[group]
```

Bounds are invalidated by geometry, may use different grouping than chemistry, and may coexist as AABB/sphere/capsule/OBB. FireCore NBFF already demonstrates the useful dataflow: `atom2group -> Buckets -> group AABBs`; keep the dataflow, separate ownership.

## 7. `spacc` initial scope

Keep it small:

```text
spacc/
  aabb.rs
  buckets.rs
  uniform_grid.rs
  morton.rs       # later if useful
```

Possible APIs:

```rust
fit_aabb(pos, ids) -> Aabb
fit_group_aabbs(pos, offsets, items, out)
build_buckets(cell_of_item, ncell) -> Buckets
build_uniform_grid(pos, cell_size) -> UniformGrid
```

Prefer `spacc -> numcore` only. `pgraph_ops` can provide convenience adapters without making spatial acceleration intrinsic to `PGraph`.

## 8. FireCore builder: critical reevaluation

### 8.1. Useful entity model

FireCore `MMFFBuilder.h` has good chemical concepts:

- `Atom`: type, fragment, position, REQ;
- `AtomConf`: local valence topology (`nbond/npi/ne/nH`, bond slots, `pi_dir`);
- `Bond`: endpoints, order, PBC, parameters;
- `Angle`, `Dihedral`, `Inversion`: explicit interaction records;
- `Fragment`: semantic molecule/rigid-group record.

Keep the concepts, but not all FireCore ownership/indirection.

### 8.2. Every atom gets a Conf

FireCore keeps separate `atoms[]` and `confs[]`; cap atoms have `iconf=-1`. This causes repeated `if(iconf>=0)` branches and synchronization/sorting complexity. `sortBonds()` even assumes atoms with Conf precede atoms without Conf.

In SurfMol, every atom gets Conf, including explicit H/e-pair caps:

```rust
pub struct AtomConf {
    pub nbond: u8,
    pub npi:   u8,
    pub ne:    u8,
    pub nh:    u8,
    pub bonds: [BondH; 4],
    pub pi_dir: Vec3d,
}

pub struct MolAtom {
    pub pos:     Vec3d,
    pub element: u8,
    pub atype:   i32,
    pub frag:    i32,
    pub conf:    AtomConf,
}
```

A cap H simply has `nbond=1`; no special no-topology state. The builder is not the hot force loop, so uniform/simple ownership is more valuable than saving a few bytes per cap.

### 8.3. Simplify FireCore's negative neighbor sentinels

FireCore packs `pi=-2`, `epair=-3`, `H=-4` into `AtomConf.neighs[4]`. With explicit caps and a Conf for every atom, prefer clearer semantics:

- `conf.bonds[]`: actual bond handles only;
- `npi/ne/nh`: valence state;
- explicit H/e-pair dummy: ordinary atom + bond;
- future virtual ports: dedicated sidecar if needed.

This removes the earlier draft's need for `BondH + neigh_kind[]` as well.

## 9. Primary molecular topology vs derived interactions

Authoritative mutable state should be small:

```text
atoms + Conf
bonds
fragment/group assignment
manual overrides/annotations
```

Derived by default:

```text
neighbor tables
angles
proper dihedrals
inversions
connected components / bridges / rings
spatial bounds
```

FireCore stores angles/dihedrals in the builder and must remap them when bonds are sorted. For normal UFF-like use, generating them from bonds during compile/bake is simpler and keeps bonds as topology SSOT.

Do not forbid explicit terms. Support overrides when needed:

```rust
pub struct MolOverrides {
    pub angles:     Vec<AngleOverride>,
    pub dihedrals:  Vec<DihedralOverride>,
    pub inversions: Vec<InversionOverride>,
}
```

Compiler generates normal terms then applies overrides/custom exclusions.

## 10. `MolecularGraph.h` -> `pgraph_ops`

Useful FireCore pipeline:

```text
edges
 -> makeNeighbors(): CSR with both neighbor atom + incident bond
 -> fillSubGraph / splitByBond / maskCaps / findBridges
```

Port algorithms, not the class ownership. `MolecularGraph` currently stores primary data, derived CSR, masks, BFS fronts and Tarjan scratch in one object.

Suggested `pgraph_ops` modules:

```text
adjacency.rs
components.rs
bridges.rs
loops.rs
reorder.rs
geometry.rs
selection.rs
picking.rs
edit.rs
```

Improvements:
- scratch is local or reusable `Workspace`, never a `PGraph` member;
- no static DFS time counter;
- algorithms accept slices/`PGraphView`/`CsrAdj`;
- allocation-free overloads can take caller buffers when repeated performance matters.

## 11. Dynamic `MolBuilder`

Keep the current Rust slot/generational-handle strategy as molecular edit state. It need not define the generic `PGraph` API.

```rust
pub struct MolBuilder {
    atoms: Vec<Slot<MolAtom>>,
    bonds: Vec<Slot<MolBond>>,
    free_atoms: Vec<u32>,
    free_bonds: Vec<u32>,
    fragment: Partition,
}
```

A future mesh editor may justify extracting generic `PGraphEdit` slots; avoid designing that abstraction before the second real client exists.

Essential operations:
- add/remove atom/bond; duplicate-edge detection;
- maintain `AtomConf.bonds[4]` exactly on mutation;
- auto-bond (including PBC variant);
- valence/hybridization and `make_conf_geom`;
- add/substitute caps/e-pairs;
- atom/type preparation;
- components/split/bridges via `pgraph_ops`;
- fragment assignment/reorder;
- generic transforms/picking/selection delegated to `pgraph_ops`.

No force-field inheritance, velocities, forces or integrators in the builder.

## 12. Bake is a compiler

The important boundary is not simply dynamic -> static graph. It is editable molecular semantics -> specialized numerical representations:

```text
MolBuilder
  atoms+Conf, bonds, fragments, overrides
       |
       | validate / remove dead slots
       | dense remap + optional fragment reorder
       v
PGraph + molecular sidecars
       |
       +-- CsrAdj                 generic algorithms
       +-- FixedAdj<4>            bounded CPU/GPU kernels
       +-- derive angles/torsions/inversions
       +-- apply overrides
       +-- compile FF parameters
       +-- RangeGroups
       +-- optional spacc caches
       v
MolTopology + force-field-specific compiled data
```

Different force fields may compile different topology layouts; do not make one `Topology` struct contain every possible term merely for API convenience.

A useful split is:

```rust
pub struct MolTopology {
    pub graph: PGraph,
    pub element: Vec<u8>,
    pub atype:   Vec<i32>,
    pub charge:  Vec<f64>,
    pub fragments: RangeGroups,
}

pub struct UffTopo {
    pub adj4:       FixedAdj<4>,
    pub angles:     Elements<3>,
    pub dihedrals:  Elements<4>,
    pub inversions: Elements<4>,
    // FF-specific parameter sidecars
}
```

This resembles FireCore's actual high-performance UFF arrays more closely than its class hierarchy.

## 13. Force fields: keep layouts, reject inheritance

FireCore hierarchy:

```text
Atoms -> ForceField -> NBFF -> UFF / MMFFsp3_loc
```

accumulates position/type state, force/velocity/integrator, nonbond parameters, topology, PBC and acceleration inside the FF object. SurfMol should use composition:

```text
MolWorld
  simulation state (pos/vel/force)
  MolTopology/PGraph
  bonded FF compiled data
  NonBondedFF
  optional SurfaceFF
  optional spacc caches
  integrator
```

The valuable FireCore lesson is **flat aligned arrays, fixed-stride local topology and explicit kernels**, not inheritance.

## 14. GUI/rendering benefit of `PGraph`

`PGraph` is a common interchange representation, not just a topology helper:

```text
positions                 render/pick points
positions + edges         render/pick bonds/trusses
positions + triangles     render mesh faces
```

Domain sidecars supply visual properties (atom radius/color, mesh material/normals, beam radius/stress). Generic picking and geometric selection operate on the same data through `pgraph_ops`.

## 15. Implementation order

### P0 — foundation
1. `PGraph`, `PGraphView`, `Elements<N>`, `Ragged`, `Permutation`.
2. `FixedRows<K>` / `FixedAdj<K>` with `-1` invariant.
3. `CsrAdj` + count/prefix/scatter builder.
4. `build_fixed_adj<K>` with explicit degree-overflow error.
5. `Partition -> IndexGroups`; group-aware permutation -> `RangeGroups`.
6. `spacc::Aabb` + group AABB fitting + `Buckets`.

### P1 — molecular builder
7. Every dynamic atom gets Conf; remove `iconf`/no-Conf paths.
8. Conf bond slots contain only real `BondH` handles.
9. Port/clean FireCore auto-bond, conf geometry, cap/e-pair placement.
10. Port components/split/bridges onto `CsrAdj`.
11. Bake slots -> dense `PGraph` + sidecars + `FixedAdj<4>`.
12. Derive UFF interactions and parity-test against FireCore.

### P2 — shared editor richness
13. Generic compact/remap of sidecars.
14. Loop/ring algorithms.
15. SDF selection + ray picking.
16. High-degree dynamic adjacency only when a mesh editor demands it.

## 16. Tests/parity

Fundamental invariants:
- edge list -> CSR matches brute force;
- edge list -> `FixedAdj<K>` matches CSR;
- valid fixed-row entries packed first; remainder exactly `-1`;
- degree `>K` errors, never truncates;
- permutations preserve all endpoint/sidecar associations;
- `Partition -> IndexGroups` preserves each assigned item exactly once.

FireCore parity:
- same auto-bonds on test molecules;
- same cap/conf geometry within tolerance;
- same baked atom-neighbor + bond-neighbor tables;
- same components/bridges;
- same UFF interaction counts/params where models intentionally match.

Do not require parity for complexity intentionally removed (optional Conf, `confRange`, class ownership).

## 17. Decisions and remaining questions

**Decisions proposed here:**
1. `pgraph` instead of `mgraph` for the foundational positioned graph.
2. `spacc` separate from topology.
3. Data containers (`pgraph`) separate from transferable algorithms (`pgraph_ops`).
4. `PGraph` owns positions + edges only; higher-order elements are sidecars.
5. Triangles/angles and polygons/rings share representations, not semantics.
6. `FixedAdj<K>` (ELL-like) and `CsrAdj` are sibling fundamental adjacency forms.
7. Every molecular atom has Conf; no FireCore `iconf=-1` special class of caps.
8. Fragment/group membership is separate from spatial bounds; bake may pack fragments contiguously.
9. Angles/dihedrals/inversions are derived by default with explicit overrides.
10. Retain FireCore numerical layouts, reject deep FF inheritance.

**Still open:**
- separate Cargo crate `pgraph_ops` vs module/feature of `pgraph` (architecturally separate either way);
- `Partition/IndexGroups/Permutation` in `pgraph` vs even more generic `numcore::index`;
- whether common angle/dihedral lists belong in `MolTopology` or only force-field compiled data;
- explicit e-pair dummy atoms vs virtual ports per force-field model;
- exact generational handle packing in the dynamic builder (not hot until bake).
