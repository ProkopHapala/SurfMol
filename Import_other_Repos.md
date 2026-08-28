# Import from Other Repos

Reference repositories we import algorithms, kernels, and project-organization patterns from. SurfMol is the Rust+OpenCL successor consolidating the jewels of these repos into a clean, compiled, GPU-first codebase.

**Cross-repo rules** (see `AGENTS.md` §Rule 6 — Parity Work):
- When porting/mirroring a feature, cite the reference file + function in a comment, e.g. `// ported from FireCore cpp/common/molecular/UFF.h:UFF::eval`.
- **FireCore is the performance benchmark** — SurfMol (Rust+OpenCL) must be at least as fast as the FireCore C++ reference for any ported algorithm. Measure, do not assume.
- **CPU Rust references are authoritative** for correctness; GPU (OpenCL) must match CPU within tolerance.

---

## 1. FireCore — `/home/prokop/git/FireCore/`

**Role:** Oldest and still most relevant reference for SurfMol's molecular topology, force-field data layout, fragment/group machinery, spatial broad phase, and performance. FireCore is messy because several generations of design accumulated in one inheritance tree, but it contains the closest tested implementations of what SurfMol needs. **Import the data-layout and algorithmic ideas; do not reproduce the class hierarchy.**

> **Filename note:** current GitHub `master` contains the builder in `cpp/common/molecular/MMFFBuilder.h`. Some local/older branches refer to the same lineage as `MMFFBuilderBase.h`; when porting, verify the exact local file/function and cite that.

### Top-level layout
| Dir | Purpose |
|-----|---------|
| `cpp/common/molecular/` | Molecular topology builders, force fields, groups, graph algorithms |
| `cpp/common/dataStructures/` | Buckets, hash maps, sparse/index structures |
| `cpp/common_resources/cl/` | Canonical OpenCL kernels |
| `cpp/apps_OCL/`, `cpp/apps_CUDA/` | GPU-accelerated apps |
| `pyBall/OCL/` | Pure pyOpenCL implementations |
| `pyBall/RigidAtomFF/` | Position-Based Dynamics (XPBD, RRsp3) — ARAP ports |
| `tests/` | Test scripts and validation (START HERE) |
| `doc/` | Technical docs, derivations |

### 1.1. Molecular topology / positioned-graph references — P0

| What | File | What to reuse / reconsider |
|------|------|----------------------------|
| **Minimal positioned particles** | `cpp/common/molecular/Atoms.h` | `atypes[] + apos[]`, aligned flat arrays, plus simple geometry (`getAABB`, transforms). Important precedent for the new **`pgraph` = positions + connectivity** boundary. Do **not** inherit force fields from it. |
| **Dynamic molecular builder** | `cpp/common/molecular/MMFFBuilder.h` (`MM::Atom`, `AtomConf`, `Bond`, `Angle`, `Dihedral`, `Inversion`, `Fragment`, `Builder`) | Main source for topology construction, cap/e-pair geometry, atom typing, bond/angle/torsion generation, fragment insertion and export. Critically simplify it in Rust: **every atom gets a Conf**; no `iconf=-1` branch for caps. |
| **CSR molecular graph algorithms** | `cpp/common/molecular/MolecularGraph.h` | `makeNeighbors()` builds both atom→neighbor and atom→bond CSR; `fillSubGraph`, `splitByBond`, `findBridges`, `maskCaps`. Port algorithms to **`pgraph_ops`**, not the scratch buffers/class ownership. |
| **Group/partition mapping** | `cpp/common/molecular/Groups.h` | `a2g` (atom→group) plus `g2a` + `{i0,n}` group ranges. `setGroupMapping()` is another count→prefix→scatter CSR build. Strong reference for a generic `Partition`/`IndexGroups` primitive. |
| **Group AABB broad phase** | `cpp/common/molecular/NBFF.h::initBBsFromGroups` | Converts `atom2group` into `Buckets`, then maintains one bounding box per group. This is the key reference for **`Partition -> group members -> spacc bounds`**. Keep groups independent of bounds; bounds are derived acceleration caches. |
| **Fixed-stride force-field topology** | `cpp/common/molecular/UFF.h` | `Quat4i neighs[natoms]` + `Quat4i neighBs[natoms]`, both padded with `-1`, plus flat bond/angle/dihedral/inversion arrays. Excellent CPU/GPU compiled representation. |
| **Localized fixed-neighbor FF** | `cpp/common/molecular/MMFFsp3_loc.h` | `nneigh_max=4`, `nnode*4` local arrays, per-node `Quat4d` parameters. Strong evidence for a generic **fixed-row adjacency (`FixedAdj<K>`)** representation for bounded-degree kernels. |

### 1.2. FireCore entity model: keep the chemistry, remove the indirection

`MMFFBuilder.h` separates `MM::Atom` from `MM::AtomConf`; `Atom::iconf==-1` means a cap atom has no configuration. This produces many conditional paths (`getAtomConf`, `tryAddConfToAtom`, bond insertion, type assignment, sorting, cap handling). It also forces fragments to carry both `atomRange` and `confRange`.

**SurfMol redesign:** every atom has the configuration fields, including H caps and explicit e-pair dummies. In the dynamic builder the extra bytes are irrelevant compared with the simplification:

- no `iconf` indirection;
- no separate `confs[]` array or `confRange`;
- no cap-specific topology path;
- one neighbor-maintenance path for all atoms;
- sorting/compaction remaps one atom array rather than atom+conf arrays.

Keep the chemically useful concepts, but distinguish **primary topology** from **derived force-field terms**:

| FireCore entity | SurfMol interpretation |
|-----------------|------------------------|
| `Atom + AtomConf` | one dynamic `MolAtom` record / parallel slot-indexed sidecars |
| `Bond` | primary molecular edge; atom endpoints + order/PBC + optional builder params |
| `Angle` | derived 3-vertex interaction unless explicitly overridden |
| `Dihedral` | derived 4-vertex path/interaction unless explicitly overridden |
| `Inversion` | derived local interaction unless explicitly overridden |
| `Fragment` | molecular metadata referencing a generic group/partition; AABB is **not** intrinsic fragment state |

### 1.3. Neighbor representations: FireCore already uses both builder and hot forms

There are two distinct useful meanings of "neighbors" in FireCore:

1. **Builder-side `AtomConf.neighs[4]`: bond indices.** This makes bond insertion/removal and lookup of bond metadata natural.
2. **Compiled UFF/NBFF arrays:** `neighs` stores neighboring **atom indices**, while `neighBs` stores the corresponding **bond indices**. Both are fixed stride (`K=4`) and padded with `-1`.

This distinction should survive in SurfMol. Do not force one representation to serve both editing and kernels.

For the generic libraries use explicit structures:

```rust
pub struct FixedRows<const K: usize> {
    pub data: Vec<[i32; K]>,   // valid entries packed first, rest = -1
}

pub struct FixedAdj<const K: usize> {
    pub neigh: FixedRows<K>,    // vertex/atom indices
    pub edge:  FixedRows<K>,    // corresponding edge/bond indices
}

pub struct CsrAdj {
    pub offsets: Vec<u32>,      // nvert + 1
    pub neigh:   Vec<u32>,      // 2*nedges for undirected graph
    pub edge:    Vec<u32>,      // matching edge ids
}
```

**Terminology:** `FixedAdj<K>` is an ELLPACK/ELL-like padded row representation, **not CSR**. It is excellent for GPU kernels when degree is bounded. `CsrAdj` is compact and better for arbitrary/high-degree meshes and general graph algorithms. `pgraph_ops` should build either from the same edge list.

Initial policy:
- organic molecular kernels: `FixedAdj<4>`;
- broader chemistry: allow `K=8` if required by a particular model;
- mesh/truss kernels: choose `K=8/16/32/64` only when the workload benefits; otherwise CSR;
- if overflow becomes common, add a hybrid fixed-prefix + overflow representation later rather than complicating v1.

### 1.4. `MolecularGraph.h`: algorithms yes, ownership no

`MolecularGraph` is valuable because it is already almost a pure topology algorithm playground: edge list → CSR → flood fill / split / bridges. But it mixes **primary graph data**, **derived adjacency**, and **algorithm scratch** (`visited`, `disc`, `low`, `parent`, fronts, masks) in one object.

Rust split:

```text
pgraph       primary positions + edge indexes + fundamental index containers
pgraph_ops   build adjacency, components, bridges, loops, selection, edits
spacc        AABB / Buckets / hash-grid / Morton / broad-phase caches
moltopo      chemistry-specific builder + atom/bond/valence/type semantics
```

Port `findBridges` as an algorithm with caller-owned/reusable workspace; do not copy the recursive/static-state implementation literally.

### 1.5. Groups, fragments and spatial acceleration

FireCore contains two nearly identical mappings:

- `Groups.h`: `a2g` plus packed `g2a` ranges;
- `NBFF::initBBsFromGroups`: `atom2group` passed through `Buckets` to obtain group members, then one AABB per group.

This suggests a reusable generic primitive rather than a molecule-specific fragment container:

```rust
pub struct Partition {
    pub item_group: Vec<i32>,   // one group per item, -1 = none
}

pub struct IndexGroups {
    pub offsets: Vec<u32>,      // group -> packed item range
    pub items:   Vec<u32>,
}

pub struct RangeGroups {
    pub ranges: Vec<[u32; 2]>,  // [i0,n], after reordering groups contiguously
}
```

- `Partition` is convenient while editing / assigning fragments.
- `IndexGroups` is the reverse CSR view built by count→prefix→scatter.
- `RangeGroups` is the fastest baked form when fragments have been reordered contiguously.
- `spacc` takes positions + any packed group view and computes `Aabb[]`, bounding spheres, group overlap tables, etc. These are **derived caches**, not members of `PGraph` or `Fragment`.

A molecular `Fragment` then stores semantic metadata (molecule type, rigid pose, color, maybe reference geometry) and a group id/range. A collision broad-phase may use the same grouping or a different computational grouping without changing molecular semantics.

### 1.6. Force-field class hierarchy — performance reference, architectural anti-pattern

FireCore hierarchy:

```text
Atoms
  └── ForceField
        └── NBFF
              ├── UFF
              └── MMFFsp3_loc
```

This makes the force field *be* particle state: positions, velocities, forces, integrator state, non-bonded parameters, topology and spatial caches accumulate through inheritance. SurfMol should preserve the flat hot arrays and kernels but reject the ownership hierarchy.

**Target:** composition and explicit data flow. `MolWorld` in `surfmol` orchestrates state + force fields; each force-field implementation owns only what its kernel needs.

### 1.7. Other FireCore jewels

| What | File | Notes |
|------|------|-------|
| **UFF force field** | `cpp/common/molecular/UFF.h`, `common_resources/cl/UFF.cl` | Performance/data-layout reference; flat aligned arrays, fixed neighbors, explicit interaction lists. |
| **NBFF non-bonded** | `cpp/common/molecular/NBFF.h`, `common_resources/cl/Forces.cl` | LJ + Morse + Coulomb + H-bond; PBC; group AABB broad phase. |
| **GridFF B-spline grid** | `cpp/common/molecular/GridFF.h`, `cl/GridFF.cl` | Tricubic B-spline substrate potential. |
| **Projective Dynamics** | `cpp/common/math/ProjectiveDynamics_d.h` (+ `.cpp`, `_frag.cpp`) | Position-based dynamics for stiff springs. |
| **MolWorld_sp3 MD loop** | `cpp/common/molecular/MolWorld_sp3.h` | Reference MD/relaxation loop and performance benchmark. |
| **Ewald2D** | `common_resources/cl/Surface.cl` | 2D periodic surface electrostatics. |
| **RigidBodyFF** | `cpp/common/molecular/RigidBodyFF.h` | Quaternion rigid-body integration, torque evaluation. |
| **RRsp3 / ARAP ports** | `pyBall/RigidAtomFF/RRsp3/` | Cluster-sorted PBD, multiple rotation solvers. |
| **GOpt / optimizers** | `cpp/common/molecular/GOpt.h`, `GlobalOptimizer.h`, `DynamicOpt.h`, `CG.h`, `lineSearch.h` | Basin hopping + local optimization. |
| **RARFF** | `cpp/common/molecular/RARFF_SR.h`, `FlexibleAtomReactiveFF.h` | Reactive/dissociative reference. |

### 1.8. Perf/parity harness
- `getCPUticks()` cycle counter (used in `MolWorld_sp3.h`, `MolGUI.h`) with `tick2second` calibration.
- `tests/tMMFF/`, `tests/tSiNCs/`, `tests/tEFF/` for parity and timing.
- Fixed-neighbor export should be parity-tested against FireCore `UFF::neighs/neighBs` and builder neighbor output.
- Target: match or beat FireCore hot loops; **do not benchmark builder/editor operations as if they were force-field kernels**.

### 1.9. Porting notes — specific algorithms, constants, and line citations

Concrete implementation details verified by reading the source. Per `AGENTS.md` Rule 6, cite these when porting. Line numbers are from `MMFFBuilderBase.h` (local file); verify against `MMFFBuilder.h` on `master` as noted in the filename note above.

**Key algorithms with line citations** (all in `cpp/common/molecular/MMFFBuilderBase.h` unless noted):

| Algorithm | Line | What it does | Port note |
|-----------|------|--------------|-----------|
| `autoBonds(R)` | L680 | Distance-based bond finding: two atoms bonded if `\|d\| < (Ri+Rj)*Rfac`. `Rfac = -R` (negative R = use params radii). Skips cap-cap bonds via `capping_types`. | Port to `moltopo`. Use covalent radii from `Params`. |
| `autoBondsPBC(R, npbc)` | L721 | PBC variant: loops over lattice images `(ix,iy,iz)`, stamps `bond.ipbc = Vec3i8{ix,iy,iz}`. | Port to `moltopo`. `ipbc` is essential for PBC systems. |
| `makeConfGeom(nb, npi, hs)` | L932 | **The cap geometry engine.** Generates 4 neighbor directions for an atom given existing sigma bonds + hybridization. Hardcoded constants (see below). | Port to `moltopo`. Constants must be exact for parity. |
| `makeSPConf(ia, npi, ne)` | L859 | Assign hybridization (sp3/sp2/sp) + calls `makeConfGeom` + `addCaps`. | Port to `moltopo`. |
| `addCaps(ia, ncap, ne, nb, hs)` | L792 | Insert capping H / epair dummies at computed directions. Uses `Hmask[]` to decide H vs epair placement. | Port to `moltopo`. |
| `addCap(ia, hdir, capAtom, l)` | L882 | Insert one capping atom at `atoms[ia].pos + hdir*l`. | Port to `moltopo`. |
| `addEpair(ia, hdir, l)` | L903 | Insert lone-pair dummy. Type from `params->atypes[host].ePairType`. | Port to `moltopo`. |
| `makeNeighs(&neighs, perAtom)` | L1064 | Export bond topology to flat `neighs[natoms*perAtom]` (atom indices). **Special-case:** `if(jc==-1){ neighs[ja*perAtom]=ia; }` — cap inherits neighbor from host. | Port to `moltopo`/`pgraph_ops`. **This special-case is eliminated** when every atom gets a Conf. |
| `findBridges()` | `MolecularGraph.h:175` | Tarjan's bridge-finding (DFS with `disc[]`/`low[]`/`parent[]`). Identifies bonds whose removal disconnects the graph. | Port to `pgraph_ops`. Use iterative DFS + caller-owned workspace, not recursive/static-state. |
| `fillSubGraph(ia, color)` | `MolecularGraph.h:106` | BFS flood-fill from atom `ia`, coloring reachable atoms. Uses front-buffer swapping (`if0/if1/if2`). | Port to `pgraph_ops`. |
| `splitByBond(ib, color)` | `MolecularGraph.h:118` | Color one side of bond `ib`. | Port to `pgraph_ops`. |
| `makeNeighbors()` | `MolecularGraph.h:49` | Two-pass CSR: count neighbors per atom, then scatter `atom2bond[]` + `atom2neigh[]`. | Port to `pgraph_ops::build_csr_adj`. |

**`NeighType` sentinels** (`MMFFBuilderBase.h:78-82`):
```cpp
enum class NeighType: int { pi = -2, epair = -3, H = -4 };
```
FireCore packs non-bond neighbors (pi orbitals, lone pairs, capping H) into the same `neighs[4]` array using negative sentinels. Positive values = bond indices. When merging Conf into every atom, decide whether to keep this packing (compact, 4 bytes/neighbor) or use a separate `neigh_kinds: [u8; 4]` array (clearer, +4 bytes/atom).

**`makeConfGeom` geometry constants** (`MMFFBuilderBase.h:932-998`) — must be ported exactly for cap placement parity:
- **sp3 (npi=0):** tetrahedral directions. `sqrt(2/3)=0.81649658092`, `sqrt(1/3)=0.57735026919`, `1/3=0.33333333333`. For `nb=2`: `hs = c*cc ± b*cb`. For `nb=1`: three caps at `c*cc + b*cb*2`, `c*cc - b*cb ± a*ca`.
- **sp2 (npi=1):** trigonal planar. `sqrt(3)/2=0.86602540378`, `-0.5`. For `nb=1`: two caps at `c*cc ± a*ca`, one pi at `b`.
- **sp (npi=2):** linear. For `nb=1`: cap at `c*-1`, pi at `b` and `a`.
- For `nb=3` (sp3): cap direction = cross product of bond-edge vectors, normalized, flipped to oppose the sum of existing bonds.

**`Bond.ipbc`** (`MMFFBuilderBase.h:213`): `Vec3i8` (3× `int8_t`) — periodic image index. `(0,0,0)` = no PBC shift. Set by `autoBondsPBC`. Essential for PBC bond evaluation — the forcefield must know which periodic image a bond crosses.

**`Atom.bTypeFixed`** (`MMFFBuilderBase.h:64`): if true, the atom type was explicitly set (e.g., from file) and must not be overridden by topology-based auto-assignment. Port this flag to prevent silent type overwrites.

**`Atom::HcapREQ` / `Atom::defaultREQ`** (`MMFFBuilderBase.h:58-59`):
```cpp
constexpr static Quat4d HcapREQ    = { 1.4870, 0.026095977, 0., 0. }; // sqrt(0.000681)
constexpr static Quat4d defaultREQ = { 1.7,    0.061067605, 0., 0. }; // sqrt(0.0037292524)
```
Default non-covalent parameters for capping H and generic atoms. Port to `moltopo::Params` defaults.

---

## 2. SPAMMM — `/home/prokop/git/SPAMMM/`

**Role:** Full-featured Python + pyOpenCL scanning-probe microscopy and manipulation engine. Scope overlaps heavily with SurfMol: SurfMol is more about manipulation + global optimization, SPAMM focuses more on imaging, but both contain both aspects. **SurfMol is to a large degree rewriting SPAMM into Rust** to eliminate Python overhead and produce a compiled binary.

### Top-level layout
| Dir | Purpose |
|-----|---------|
| `kernels/` | OpenCL `.cl` sources (all GPU compute) |
| `spammm/topology/` | Molecular topology SSOT: `AtomicGraph` |
| `spammm/forcefields/` | UFF, SPFF, LFF, rigid body |
| `spammm/surfaces/` | GridFF, ContactSurface, Ewald |
| `spammm/SPM/` | AFM/STM imaging |
| `spammm/utils/` | `OpenCLBase` (device selection, buffer mgmt) |

### Jewels to port to Rust+OpenCL (high priority)
| What | File | Notes |
|------|------|-------|
| **OpenCLBase (NVIDIA-first device selection)** | `spammm/utils/OpenCLBase.py:133-150` | `select_device(preferred_vendor='nvidia')`. Port to Rust OpenCL crate. |
| **AtomicGraph topology SSOT** | `spammm/topology/AtomicGraph.py` | Stable object identities, `to_arrays()` export. Mirror in `surfmol-topology`. |
| **`getNonBond_ex2`** | `kernels/nonbonded.cl:135-277` | Pairwise LJ/Coulomb with 2nd-neighbor exclusion, local-memory tiling (32 atoms/tile), PBC. |
| **UFF kernels** | `kernels/UFF.cl` (`evalBondsAndHNeigh_UFF`, `evalAngles_UFF`, `evalDihedrals_UFF`, `evalInversions_UFF`, `assembleForces_UFF`) | Harmonic bonds + hneigh vectors reused by angles/dihedrals. |
| **SPFF kernels** | `kernels/SPFF.cl` (`getSPFFf4`, `updateAtomsSPFFf4`, `relax_nsteps_serial`) | Bonds, angles, π-orbital DOFs, FIRE relaxation. |
| **Rigid body 6-DOF** | `kernels/rigid.cl` (`rigid_body_folded_kernel`, `rigid_body_pairff_probe_grid`) + `spammm/forcefields/RigidBodyDynamics.py` | Quaternion integration, gyroscopic term, per-body state, ping-pong multimol MD. |
| **PairFF non-bonded model (legacy + unified)** | `kernels/rigid.cl:2198` (legacy `rigid_body_pairff_kernel`), `kernels/rigid.cl:2452` (unified `rigid_body_pairff_unified_kernel`), `kernels/Forces.cl:260` (`compact_exp_pair_EF`), `kernels/Forces.cl:279` (`pairff_unified_site_EF`) | **Our latest non-covalent interaction model.** See §"PairFF non-bonded model" below for full detail. |
| **LFF projective Jacobi** | `kernels/LFF.cl` + `spammm/forcefields/LFFSolver.py` | Linearized projective Jacobi on K12/K13/K14 springs — fast relaxation surrogate. Closest existing thing to "position-based dynamics" in the repo. |
| **Contact surface (separable B-spline×poly + radial PIC)** | `kernels/contact_surface.cl` (`evalSeparableBsplinePoly`, `relaxStrokesTiltedContactPME*`, `fillContactPMEMeshVL`) | Quasi-2D contact field for static AFM. |
| **GridFF tricubic B-spline + Poisson** | `kernels/gridFF.cl` (`sample3D*`, `poissonW*`) | Tricubic interpolation with PBC; FFT Poisson solver with slab correction. |
| **Rigid-body packing/clash** | `kernels/assembly.cl` (`evaluate_packing_3d`) | Steric clash with early exit, local-memory tiling. |

### Data layouts to mirror
- `float4 apos[natoms]` (w = mass/charge), `float4 aforce[natoms]` (w = energy), `float4 REQs[natoms]` (RvdW, EvdW, Q, H-bond), `int4 neighs[natoms]` (up to 4 neighbors).
- **`float4.w` channel reuse** for energy / secondary results / clash flags — avoid extra buffers.
- Workgroup-sized fragments: 32 atoms/tile (nonbonded), `MAX_ATOMS_PER_BODY=128` (rigid), `LFF_WG_SIZE=64`, `CS_TILE=16`. **Matches the 16/32/64/128 atoms-per-fragment design in `notes/ToDo_user.md`.**

### PairFF non-bonded model (our latest creation — full detail)

**Entry point:** `demos/demo_pairff.py` → `RigidBodyPairFF` (`spammm/forcefields/RigidBodyDynamics.py:1588`).

The demo implements **two switchable non-covalent force models** for rigid-body molecule-molecule interactions. Both operate on a sorted site array `[real_atoms, epairs, sigma_holes]` with per-site `REQ = (R, sqrt(E), Q/pseudo-charge, w_blunt)` and `type ∈ {0=atom, 1=epair, 2=sigma}`.

#### Legacy kernel (`rigid_body_pairff_kernel`, rigid.cl:2198) — 4 separate loops

| Interaction | Formula | Sites involved |
|-------------|---------|----------------|
| **Morse** (atom-atom) | `E = E₀·[exp(2α(r−R₀)) − 2·exp(α(r−R₀))]` | type=0 ↔ type=0 |
| **Coulomb** (atom-atom) | `E = k_e·Q_i·Q_j / √(r² + R2SAFE)` (damped) | type=0 ↔ type=0 |
| **Lorentzian Hbond** (atom-epair) | `E = min(0, Q_atom·Q_epair) · fcut(r/rc) · 1/(w² + r²)` | type=0 ↔ type=1 or type=1 ↔ type=0 |
| **Sigma-hole** (atom-sigma) | `E = min(0, Q_atom·Q_sigma) · fcut(r/rc) · 1/(w² + r²)` | type=0 ↔ type=2 or type=2 ↔ type=0 |

where `fcut = smoothstep(1 − r/rc) = 3x² − 2x³`, `R0 = Ri+Rj`, `E0 = sqrt(Ei)·sqrt(Ej)`, `Q = Qi·Qj`.

**Design:** epairs/sigma-holes are pseudo-atoms with R=0, E=0. They participate ONLY in Hbond/sigma interactions, not Morse/Coulomb. Pseudo-charge stored in REQ.z. The kernel uses 4 separate loops with `if (atom_idx < n_dyn_atoms)` branching — causes warp divergence. Epair-epair and sigma-sigma interactions are skipped.

#### Unified kernel (`rigid_body_pairff_unified_kernel`, rigid.cl:2452) — single branch-free loop

**This is the production model and the one to port to SurfMol.** Uses `pairff_unified_site_EF` (Forces.cl:279) which calls `compact_exp_pair_EF` (Forces.cl:260):

```
V = E₀ · y · (α·y − (1+α))     where y = max(0, 1 − β(ρ−R₀)/8)^8
ρ = r² / (√(r²+w²) + w)         [soft radius, one sqrt]
```

**Branch-free mixing** (all GPU lanes execute same instructions, parameters differ):
```
gij = gi · gj                   [core flag: 1=atom, 0=epair]
R0  = gij · (Ri + Rj)           [epair-atom: R0=0]
E0  = mix(attr, ei·ej, gij)     [attr = -min(0, Qi·Qj)]
α   = gij                       [atom: α=1 (Morse), epair: α=0 (purely attractive)]
w   = wi + wj                   [atom: w=0 (sharp), epair: w>0 (blunt)]
```

Coulomb is added only when `gij > 0.5` (real-real pairs): `E += k_e·Qi·Qj / √(r² + R2SAFE)`.

**Key advantages over legacy:**
- Single loop over all sites (no branching by type → no warp divergence)
- Same instructions for atoms and epairs (just different parameters)
- Compact exponential converges to Morse (not Gaussian like polynomial family)
- Exact `V(R₀)=−E₀`, `V'(R₀)=0`, `V''(R₀)=2E₀β²` for all n
- One sqrt (for soft radius); no `r=sqrt(r²)` for the compact channel
- Cutoff `r_c = R₀ + n/β` (compact support)

**Site types and their roles:**
| Type | Name | R | E | Q | w | α | Role |
|------|------|---|---|---|---|---|------|
| 0 | atom | RvdW | √(EvdW) | charge | 0 | 1 | Morse + Coulomb |
| 1 | epair (lone pair) | 0 | 0 | He (<0) | w>0 | 0 | Hbond acceptor (attracts H+) |
| 2 | sigma-hole | 0 | 0 | Hs (>0) | w>0 | 0 | Hbond donor (attracts O−) |

**Electron pairs** are placed by `AtomicSystem.add_electron_pairs()` at `epair_dist` (default 1.4 Å) from host O/N atoms, along the lone-pair direction. **Sigma holes** are placed on H atoms bonded to O/N at `sigma_dist` (default 1.0 Å) along the O-H bond direction. Both are fixed in the body frame at construction time.

**Additional features in the kernel:**
- Z-harmonic constraint (per-atom, produces both force and torque): `F_z = -k_z·(z - z_target)`
- Anchor springs for mouse dragging: `F = -k_anchor·(p - anchor)`
- FIRE relaxation (adaptive dt, quench on v·F < 0)
- Gyroscopic torque: `α_body = I⁻¹·(τ_body - ω × L)` where `L = I·ω`
- Quaternion update: `q ← normalize(q ⊗ dq_taylor(ω·dt))`

**Kernel variants (all use the same compact-exp model):**
| Kernel | Lines | Use case |
|--------|-------|----------|
| `rigid_body_pairff_unified_kernel` | 2452-2623 | 1 active body + 1 static partner |
| `rigid_body_pairff_unified_env_kernel` | 2643+ | 1 active body + many env molecules (tiled) |
| `rigid_body_pairff_unified_faf_kernel` | 2700+ | + FAF substrate (fused PairFF+FAF) |
| `rigid_body_pairff_unified_env_faf_kernel` | 2734+ | + env + FAF |
| `rigid_body_pairff_unified_allmol[_faf]_kernel` | 2888+ | Multi-body shared buffers (preferred for multi-mol) |

### Notes
- No reactive/dissociative potentials or port-based bonding in SPAMM — those come from FireCore (`RARFF`, `RRsp3`).
- NVIDIA GPU requires unrestricted shell so the ICD is visible; sandbox hides it and falls back to PoCL/CPU (never report PoCL timings as GPU).

---

## 2b. NumericalMathPlayground — `/home/prokop/git/NumericalMathPlayground/`

**Role:** Theoretical derivations and compact-potential fitting playground. Contains the physics justification for the port-based RAFF approach and the fast compact non-bonded potentials used across SPAMMM and SurfMol. **Not a code source to port from — a theory source to cite.**

### Top-level layout
| Dir | Purpose |
|-----|---------|
| `topics/ReactiveFF/` | Rigid-atom rotating-frame FF theory, OpenCL demos |
| `topics/NonBondingFFs/` | Compact pairwise potential design, fitting, demos |

### Jewels to cite (high priority)
| What | File | Notes |
|------|------|-------|
| **Port-based FF literature review + theory** | `topics/ReactiveFF/RigidAtomicRotatingFrameFF.chat.md` | Core theoretical document: port energy formulation, ARAP/Procrustes equivalence to angle FFs, adiabatic vs extended-Lagrangian rotation, novelty vs VALBOND/patchy-particles/AMOEBA. **The physics justification for RAFF.** |
| **Compact pairwise potentials derivation** | `topics/NonBondingFFs/FastPairwisePotentials.chat.md` | Full derivation of compact polynomial Morse (r²-based), compact exponential Morse (recommended), pure-tail analytical solution, f²/f⁴ interpolation, soft-radius for epair blunting, branch-free mixing rules. |
| **Compact potential fitting code** | `topics/NonBondingFFs/fit_radial.py` | Working Python: all compact potential variants, analytical coefficients, mixing rules, comparison plots. |
| **NonBondingFFs overview** | `topics/NonBondingFFs/README.md` | Summary of compact-exp family, GPU branch-free design, PairFF demo, parameter reference. |
| **ReactiveFF OpenCL demos** | `topics/ReactiveFF/reactiveff_ocl_app.py`, `reactiveff_ocl_app3d.py` | OpenCL demo apps for reactive rigid-atom FF. |

### Key equations (centralized in `notes/designs/raff_theory_equations.md`)
- **Compact polynomial Morse:** `V = C_R z² − C_A z`, `z = (1 − r²/r_c²)^q`, force without sqrt, C³ at cutoff.
- **Compact exponential Morse (recommended):** `V = E₀ y [αy − (1+α)]`, `y = max(0, 1 − β(ρ−R₀)/n)^n`, converges to Morse as n→∞. Exact `V(R₀)=−E₀`, `V'(R₀)=0`, `V''(R₀)=2E₀β²` for all n.
- **Soft radius:** `ρ = √(r²+w²) − w = r²/(√(r²+w²)+w)` — blunts epair origin, same GPU instructions as sharp atom core.
- **Branch-free mixing:** `g_ij = g_i·g_j` (core flag), `R₀ = g_ij(R_i+R_j)`, `α = g_ij`, `w = w_i+w_j`. No pair-kind branching on GPU.
- **Pure-tail polynomial:** `r_c² = R₀² + 4n·Δ·R₀`, `Δ = ln(2)/β`. Sets c=0, no repulsive bump, purely attractive tail.

### Porting notes
- The compact-exp kernel is already implemented in SPAMMM `Forces.cl:compact_exp_pair_EF` (l.260-273) — port that to Rust/OpenCL, citing both NumericalMathPlayground (derivation) and SPAMMM (implementation).
- The theory document (`RigidAtomicRotatingFrameFF.chat.md`) should be cited in any publication or design doc justifying the RAFF approach.

### Chronology — how the non-bonded interaction ideas developed

Reconstructed from git logs of NumericalMathPlayground (NMP) and SPAMMM. This shows the evolution from theory → fitting → demo → production kernel.

| Date | Repo | Commit / file | What happened |
|------|------|---------------|---------------|
| **2026-07-15** | NMP | `ff1915b` | First reactive forcefield for 2D/3D sp2/sp3 particles. `reactiveff_ocl_app.py` created. |
| **2026-07-15** | NMP | `17d2626` | Discussion about bond order and Pauli repulsion. `RigidAtomicRotatingFrameFF.chat.md` started. |
| **2026-07-16** | NMP | `RigidAtomicRotatingFrameFF.chat.md` finalized | **Core theory document**: port energy formulation, ARAP/Procrustes equivalence, adiabatic vs extended-Lagrangian rotation, novelty assessment. The physics justification for RAFF. |
| **2026-07-22** | NMP | `9620b52` | `ff_map.py` created — 2D H-bond mapping around molecules. |
| **2026-07-22** | NMP | `1dc80aa` | **Big day**: (1) `ff_map.py` Morse+Coulomb+epairs/sigma-holes 2D maps; (2) `demo_pairff.py` interactive Vispy rigid-body simulator; (3) `fit_radial.py` polynomial fit of unified short-range potential approximating both Morse and electron pairs. `FastPairwisePotentials.chat.md` started (compact polynomial family). |
| **2026-07-22** | NMP | `FastPairwisePotentials.chat.md` (l.1-830) | Compact polynomial Morse derivation: `V = C_R z² − C_A z`, `z = (1−r²/r_c²)^q`, force without sqrt, mixing rules. |
| **2026-07-22** | NMP | `FastPairwisePotentials.chat.md` (l.833-1365) | Pure-tail analytical solution: `r_c² = R₀² + 4n·Δ·R₀`, f²/f⁴ interpolation. |
| **2026-07-22** | NMP | `FastPairwisePotentials.chat.md` (l.1366+) | **Key insight**: polynomial converges to Gaussian, not exponential. Compact exponential family: `y = max(0, 1−β(ρ−R₀)/n)^n` → converges to Morse. Unified `V = E₀y[αy−(1+α)]` for atoms+epairs. Soft radius `ρ = √(r²+w²)−w`. |
| **2026-07-23** | NMP | `84e5e35` | Implemented unified kernel `rigid_body_pairff_unified_kernel` in SPAMMM `kernels/rigid.cl` for atom-atom and atom-epair, tested in `demo_pairff.py`. **The compact-exp kernel was born.** |
| **2026-07-23** | SPAMMM | `0b45550` | SPAMMM `demo_pairff.py` shows rigid body dynamics with H-bond unified kernel. `Forces.cl:compact_exp_pair_EF` (l.260-273) implemented — the production compact-exp kernel (n=8, soft radius, branch-free). |
| **2026-07-24** | SPAMMM | `c1d8a51` | Multi-body picking in demo_pairff.py; FAF substrate restructuring. |
| **2026-07-24** | SPAMMM | `14f5568` | `rigid_body_pairff_unified_faf_kernel` — fused PairFF+FAF for on-surface assembly. |
| **2026-07-28** | SPAMMM | `19059ca` | `PairFF_manual.md` and `PairFF.md` design reports written. Speed improvements, animation. |
| **2026-07-30** | SPAMMM | `4ea1f6a` | Rigid body dynamics for on-surface assembly (production). |
| **2026-08-01** | SPAMMM | `d81ec52` | Two FAF-substrate strategies: per-type combined potential, per-atom mixing rules (Pauli, London, Coulomb, Hb). |
| **2026-08-01** | SPAMMM | `a1fec9e` | Refactored `RigidBodyDynamics.py`; multi-body allmol shared buffers; `rigid_body_pairff_unified_allmol[_faf]_kernel`. |
| **2026-08-03** | SPAMMM | `1c12231` | Proper vmin/vmax in FAF+PairFF dragging; GUI scripts for benzoic acid dimer drag. |
| **2026-08-10** | SPAMMM | (file mtime) | `demo_pairff.py`, `RigidBodyDynamics.py`, `rigid.cl`, `Forces.cl` last modified — current state. |

**Evolution summary:**
1. **Theory** (Jul 15-16, NMP): port-based FF justification, ARAP equivalence, rotation regimes.
2. **Non-bonded potential design** (Jul 22, NMP): compact polynomial → pure-tail → **compact exponential** (the breakthrough — converges to Morse, branch-free for atoms+epairs).
3. **First demo** (Jul 22-23, NMP→SPAMMM): `demo_pairff.py` with Vispy, `fit_radial.py` fitting, unified kernel in `rigid.cl`.
4. **Production** (Jul 24 - Aug 10, SPAMMM): FAF substrate fusion, multi-body allmol, GUI integration, refactoring.

**What superseded what:**
- Compact polynomial Morse (family 1) → **superseded by** compact exponential Morse (family 2) for Morse tail reproduction.
- Pure-tail polynomial → still useful for analytical cutoff, but compact-exp is the production choice.
- Legacy 4-loop kernel (Morse+Coulomb / Lorentzian) → **superseded by** unified compact-exp kernel (single branch-free loop). Legacy kept for comparison.
- Single static partner → **superseded by** multi-body allmol shared buffers.
- `NMP/demo_pairff.py` (legacy default) → **superseded by** `SPAMMM/demos/demo_pairff.py` (unified default, multi-body, FAF).

### External review — `notes/chats/REAFF.chat.md`

**ChatGPT review of the RAFF theory and roadmap (2026-08-28).** Found 12 issues, all corrected in `raff_theory_equations.md`:

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | Port energy inconsistent factors of 2 | Bug | Clean convention: `E = k_p/2 |e|²`, `F = k_p·e`, `k_p = K_bond/2` for reciprocal ports |
| 2 | XPBD constraint `C = |x_j−tip| − l_0 = 0` | Bug | `C = |x_j−tip| = 0` (tip already contains bond length) |
| 3 | Compliance notation `α = 1/(K·dt²)` | Non-standard | `α = 1/K` (physical), `α̃ = 1/(K·dt²)` (timestep-scaled) |
| 4 | Full Procrustes (with centroid subtraction) | Conceptual | Wahba problem (rotation-only, no centering) — `x_i` is a dynamical variable |
| 5 | Center-force projection "required" for analytical | Too strong | Unnecessary at exact adiabatic convergence (envelope theorem); changes the force field |
| 6 | PD characterized as "linear only" | Too narrow | PD supports nonlinear local projections; fixed global step, not fixed constraints |
| 7 | "Four cases" but enum has 3 | Counting | Separate model/solver/schedule axes instead of multiplying enums |
| 8 | "Convex = suitable for XPBD" | Wrong criterion | Replace with "locally projectable"; `½k[R−r]₊²` is not convex in Cartesian |
| 9 | "Split potential" = two different things | Conceptual | Separate exact algebraic decomposition (3b-iii) from approximate replacement (3b-i, 3b-ii) |
| 10 | Compact-exp split recommended for PBD | Numerically poor | Explicit part retains 87.5% of total curvature (`V_attr'' = −1.75 E₀β²` vs `V'' = 2 E₀β²`); verified with SymPy |
| 11 | Concave quadratic tail "unphysical" | Misleading | Morse IS concave/attrative for `r > R_0`; inflection at `R_inf = R_0 + ln(2)/β` is the natural split boundary |
| 12 | erf/erfc mixed with PBD stiffness | Wrong axis | erf/erfc is spatial short/long range decomposition for grids, not PBD stiffness split; leave Coulomb in outer force for CPU prototype |

**Key architectural contributions from the review:**
- **Common proximal problem** (§11): `x^{n+1} = argmin_x [1/(2H²)(x−y)^T M(x−y) + U_h(x,R)]` — XPBD, PD, VBD are all solvers for this same problem.
- **Curvature-based split criterion**: `θ* = argmin_θ max_r |U_ref'' − U_h''|` — directly targets max timestep, not energy fitting.
- **`U''(r)` is the most important plot** for evaluating nonbonded splits.
- **Iterations vs substeps experiment**: compare `1×H, 16 iters` vs `16×h, 1 iter` at equal hard-work budget.
- **Central research question**: "How much of the stiff Hessian can we remove from the explicit outer dynamics using only cheap atom-local implicit solves?"

---

## 3. learn_Rust — `/home/prokop/git/learn_Rust/`

**Role:** Testbed for Rust algorithms and OpenCL interface patterns, including fast collision acceleration. Import tested Rust patterns + OpenCL bindings into SurfMol.

### Top-level layout
| Dir | Purpose |
|-----|---------|
| `examples/` | Progressive demos (11): OpenCL, OpenCL-GL interop, collision, UFF MD |
| `mol_utils/` | `Vec3`, `Quat4`, `AlignedVec` (64-byte aligned, `#[repr(C)]`) |
| `mol_topology/` | Bonds, angles, UFF assignment |
| `mol_engine/` | UFF, nonbonded, MD integration |
| `data/` | Test molecules + UFF params |
| `NOTES/` | Design notes |

### Jewels to import
| What | File | Notes |
|------|------|-------|
| **OpenCL buffer management (ProQue)** | `examples/demo03_opencl/src/main.rs:35-67` | Clean `ocl` 0.19 ProQue + Buffer pattern. |
| **OpenCL-OpenGL zero-copy interop** | `examples/demo06_opencl_opengl_interop/src/main.rs:186-314` | `cl_khr_gl_sharing` platform/device iteration, GLX/EGL handle extraction, `Buffer::from_gl_buffer()`, acquire/release cycle. **Critical for GPU-rendered GUI.** |
| **AlignedVec + Vec3d/Quat4d** | `mol_utils/src/math/vec3.rs`, `mol_utils/src/util.rs` | `#[repr(C)]`, 64-byte aligned, inlined ops. Mirror in `surfmol-common`. |
| **UFF SoA data layout** | `mol_engine/src/uff.rs:43-208` | Neighbor indexing (`neighs`, `neigh_bs`), bucket-based force assembly (`a2f`). |
| **Non-bonded with exclusions + PBC** | `mol_engine/src/nonbonded.rs:54-79` | Sorted exclusion list, PBC shift vectors. |
| **Group-based AABB broad phase** | `examples/demo10_collision_balls/src/main.rs:81-118` + `collision_kernel.cl:224-305` | Per-group AABB reduction (local memory), bit-matrix overlap, degree-based dispatch. WG=32. |
| **Uniform grid + parallel scan** | `examples/demo11_collision_grid/src/main.rs:23-65` + `collision_kernel.cl:45-156` | Blelloch prefix scan, atomic scatter, 3×3 cell stencil neighbor gather. |
| **Morton code spatial sorting** | `examples/demo10_collision_balls/src/main.rs:24-56` | 2D Morton Z-curve for spatial locality. |
| **`bytemuck` zero-cost casts** | `examples/demo05_pointer_reinterpret/src/main.rs` | "Numpy view" pattern: struct slice ↔ flat array slice. |

### Key deps (Cargo.toml)
`ocl = "0.19"`, `eframe/egui = "0.29"`, `wgpu = "24.0"`, `bytemuck = "1.21"`, `nalgebra = "0.33"`, `ndarray = "0.16"`, `rhai = "1.19"`, `clap = "4.5"`, `serde`/`serde_json`.

### Notes
- No dedicated benchmark harness — timing is inline (`std::time::Instant`) in demos. SurfMol should add a real bench harness.
- `ocl` 0.19 is the chosen OpenCL crate (not `opencl3`). blood_of_civilization uses `opencl3` 0.12 — **decision needed** on which to standardize on (see §4).

---

## 4. blood_of_civilization — `/home/prokop/git/blood_of_civilization/`

**Role:** Unrelated game (terrain/economy/combat) but the **most developed Rust project** we have — import Rust project organization and binary/memory optimization settings. Key notes in `doc/AGENTS/notes/Memory_Issues/`.

### Workspace organization (15 crates + xtask)
Pattern: **domain crates (Bevy-free) + app crate (presentation) + xtask (tooling) + feature-gated opencl crate.**
- Domain: `boc_core`, `boc_protocol`, `boc_geo`, `boc_economy`, `boc_tactics`, `boc_chem`, `boc_plot`.
- Integration: `boc_ecs`, `boc_python`, `boc_script`.
- App: `boc_app` (full Bevy, migrating to eframe), `boc_pipedream` (eframe).
- Specialized: `boc_opencl` (feature-gated, **only crate with `unsafe`**), `boc_fluid2d`, `boc_procedural2d`, `vibbug` (HTML debug reports).
- Tooling: `xtask` (`cargo xtask check|test|verify|check-ownership`).

**Naming:** all crates prefixed `boc_`. SurfMol already uses `surfmol-*` prefix — keep that.

### OpenCL integration pattern
- Crate `boc_opencl`, feature-gated (`opencl = ["dep:opencl3"]`), uses `opencl3 = "0.12"`.
- `#![cfg_attr(not(feature = "opencl"), forbid(unsafe_code))]` — **all `unsafe` confined to this one crate**, no raw handle escapes the boundary, every `unsafe` block has a SAFETY comment.
- CPU reference implementations live in domain crates; OpenCL is optional acceleration, not required for correctness. **Mirror this exactly in SurfMol.**

### MUST-IMPORT: Cargo profile overrides
From `Cargo.toml:63-84`. Apply to SurfMol workspace root:
```toml
[profile.dev]
debug = 1                      # line tables only; cuts debug info ~80%, keeps panic file:line
strip = "debuginfo"            # strips DWARF, keeps .eh_frame + symbol table; binary 935MB→343MB

[profile.release]
lto = "thin"
codegen-units = 1
debug = 1                      # keep line tables for release backtraces
strip = "debuginfo"
incremental = true             # fast rebuilds in release
debug-assertions = true        # fail-loud in release
overflow-checks = true         # integer overflow panics in release
```
**Verified effects** (with `debug=1` + `strip="debuginfo"`): panic location KEPT, function names in backtrace KEPT, per-frame file:line LOST (acceptable), `.eh_frame` unwind tables SURVIVE. pipedream release binary 142MB→15MB (9.5×).

### MUST-IMPORT: Shared target directory
`~/.cargo/config.toml`:
```toml
[build]
target-dir = "/path/to/shared/target"
```
Reclaimed 24.7 GB → 2.3 GB (91% reduction) across all Rust projects. **Apply globally.**

### MUST-IMPORT: IDE indexing guard
`.codeiumignore` / `.vscode/settings.json` excluding `target/`, `artifacts/`, `debug/` from language-server indexing (`searchMaxWorkspaceFileCount: 200`). Prevents the LS from indexing multi-GB build artifacts.

### SHOULD-IMPORT
- **xtask** workspace automation (`check`, `test`, `verify`, `check-ownership --base <sha>`).
- **Stale artifact cleanup policy**: `cargo clean` when `target/` exceeds 15 GB; `scripts/target_size.sh` monitor.
- **Dependency audit**: replace heavy crates with light alternatives (e.g. `image`→`png` for PNG-only use; disable `plotters` `ttf` feature).
- **Test binary consolidation**: each integration test file = ~45-50 MB executable; merge where sensible.
- **Unsafe isolation in single feature-gated crate** (see OpenCL pattern above).

### Key Memory_Issues notes (in `doc/AGENTS/notes/Memory_Issues/`)
| File | Key takeaway |
|------|--------------|
| `rust_footprint.md` | 31 GB `target/` from 1,796 crates; `debug=1` is the single biggest lever (5× reduction). |
| `reduce_target_footprint_plan.md` | Shared target dir + `strip="debuginfo"` = 91% disk reduction. Backtrace verification results. |
| `dependency_review.md` | `boc_core` pulls `image` (67 MiB) for 2 calls — replace with `png` (9 MiB). Bevy feature pruning does NOT work (bevy_egui forces features). |
| `migrate_pipedream_to_eframe.md` | Bevy→eframe: deps 568→246 (−57%), binary 142MB→15MB (9.5×). |
| `alternative_gui.md` / `3d_renderer_alternatives.md` | wgpu + egui-wgpu = 14 MiB (vs Bevy 142 MiB). **SurfMol `editor` is the working reference** (299 deps, 14 MiB stripped). |
| `system_memory_optimization.md` / `devin_memory_optimization.md` | 16 GB RAM machine ops: kill junk processes, cap Go LS heap (`GOMEMLIMIT`), disable IDE indexing. |
| `devin_desktop_renderer_leak_bugreport.md` | Electron renderer leaks ~145 MB/min; restart IDE every 20-30 min. |

### GUI decision (already validated for SurfMol)
SurfMol `editor` already uses **wgpu + winit + egui** (14 MiB stripped release, 299 deps) — this is the recommended stack from blood_of_civilization's migration analysis. Keep it; do not adopt Bevy.

---

## 5. SimpleSimulationEngine — `/home/prokop/git/SimpleSimulationEngine/`

**Role:** General simulation/geometry reference. FireCore is the primary source for molecular topology and hot force-field layouts; SimpleSimulationEngine is complementary: it contains the **mesh/editor/topological algorithms and generic geometry utilities** that motivate extracting a reusable positioned-graph layer.

### 5.1. Revised architectural lesson: `pgraph`, `pgraph_ops`, `spacc`

The important common denominator of a molecule, polygon mesh, truss, circuit-like structure or particle network is **not a pure abstract graph**. It is a **positioned graph**:

```text
positions[i]             geometry
edges[e] = (i,j)         1-skeleton / connectivity
sidecar attributes       arbitrary domain meaning
```

This motivates the name **`pgraph`** (positioned/physical graph) more clearly than `mgraph`. `mgraph` is ambiguous (molecule / mesh / material graph) and previously mixed the data contract with the editor implementation.

Keep two levels separate:

- **`pgraph`** — tiny fundamental containers and borrowed views. Intended to be cheap enough that renderer, GUI, molecular topology, mesh/truss code can all depend on it.
- **`pgraph_ops`** — transferable algorithms that operate on those containers: adjacency builds, compaction/remapping, loops, geometry, picking, SDF selection, connected components, etc.
- **`spacc`** — spatial acceleration: AABB, Buckets/CSR binning, hash grids, Morton ordering, broad phase. It should not be embedded in `PGraph`; these structures are derived caches with different invalidation/lifetimes.

### 5.2. Fundamental `pgraph` containers

Suggested minimal set (names provisional):

```rust
pub struct PGraph {
    pub pos:   Vec<Vec3d>,
    pub edges: Vec<[u32; 2]>,
}

pub struct Elements<const N: usize> {
    pub verts: Vec<[u32; N]>,
}

pub struct Ragged {
    pub offsets: Vec<u32>,
    pub items:   Vec<u32>,
}

pub struct FixedRows<const K: usize> {
    pub data: Vec<[i32; K]>,
}

pub struct CsrAdj {
    pub offsets: Vec<u32>,
    pub neigh:   Vec<u32>,
    pub edge:    Vec<u32>,
}
```

`PGraph` does **not** own atom types, mesh materials, force-field parameters, selections, bounding boxes or algorithm scratch. Those are sidecars indexed by vertex/edge/element id.

### 5.3. Higher-order elements: share storage, not semantics

Do **not** hard-code `triangles == angles` or `polygons == rings` into `PGraph`.

- A mesh triangle is an explicitly declared 2-cell/face.
- A molecular angle is normally a derived length-2 path through the bond graph.
- A polygon is an explicit face boundary.
- A molecular ring is a discovered cycle; a graph may contain many cycles and there is no unique set of "faces" without additional embedding/chemical rules.

They can nevertheless reuse the same containers:

```text
Elements<3>   triangle indices / molecular angle triples
Elements<4>   tetrahedra / dihedral atom quadruples
Ragged        polygon loops / rings / arbitrary groups
```

`pgraph_ops` should provide operations on generic tuples/loops, while `moltopo` or a mesh module decides what those tuples *mean* and how they are generated.

### 5.4. Relevant SSE data structures

| File | Reuse | Target | Priority |
|------|-------|--------|----------|
| **`Buckets.h`** | count→prefix→scatter CSR primitive | `spacc` (and generic CSR builder helpers) | P0 |
| **`MeshBuilder2.h`** | flat index topology, soft-remove/compact, edge lookup, loop ordering, extrusion/bridge patterns | `pgraph_ops::edit/topology` | P0 |
| **`CMesh.h`** | tiny non-owning positions/edges/faces view | inspiration for `PGraphView` / slice APIs | P1 |
| **`NeighChunks.h`** | fixed local storage + overflow concept | reference for future dynamic/high-degree adjacency; **not default molecular hot layout** | P1 |
| **`Slots.h`** | small bounded inline association | concept for builder atom bond slots | P1 |
| **`SDfuncs.h`** | inlinable SDF predicates | `pgraph_ops::selection` / `numcore::geom` | P1 |
| **`Selection.h`** | ordered selection + fast membership idea | `pgraph_ops::selection` | P1 |
| **`HashMap.h`** | specialized integer/spatial hashing idea | `spacc` | P1 |
| **`geom3D.h`, `raytrace.h`** | distances, ray-sphere/capsule/AABB/triangle | `numcore::geom` + `pgraph_ops::picking` | P1/P2 |
| **`Table.h`, `BatchBuff.h`** | debug table / sparse batched storage concepts | later, only if demanded | P2 |

**Port policy:** several SSE implementations contain edge-case hazards; port the *idea* and tests, not a mechanical translation.

### 5.5. `MeshBuilder2` patterns worth sharing

- positions and connectivity stay as flat index-based arrays;
- soft removal keeps indices stable during editing;
- compaction uses an old→new permutation and remaps every sidecar;
- edge deduplication uses endpoint-pair lookup;
- edge-loop ordering generalizes directly to cycle/ring ordering;
- selection/picking depend fundamentally on geometry, which is why `PGraph` must contain positions;
- variable-length polygons/groups use flat packed index storage instead of one heap allocation per object.

The C++ `VertT` union itself is **not** a requirement. Rust should keep the common data contract explicit and use sidecars/views rather than overlaying unrelated metadata in the same bytes.

### 5.6. Fixed adjacency vs CSR vs dynamic overflow

Three representations serve different workloads:

1. **`FixedAdj<K>` (ELL-like):** constant stride `K`, packed valid entries, `-1` padding. Best for bounded molecular valence and simple GPU indexing. FireCore UFF/MMFFsp3 is the stronger reference here than SSE.
2. **`CsrAdj`:** compact arbitrary degree. Best for general graph algorithms and irregular meshes.
3. **Dynamic overflow (`NeighChunks`-like):** useful when topology changes frequently and degree is unbounded. Keep as an optional editor structure; do not make every `PGraph` pay for it.

`pgraph_ops::build_fixed_adj::<K>(edges)` and `build_csr_adj(edges)` should be sibling conversions with strict overflow validation.

### 5.7. Geometry, selection, groups and spatial caches

Generic algorithms should mostly accept slices/views rather than require a large smart object:

```rust
edge_vectors(pos, edges, out)
build_csr(nvert, edges)
build_fixed_adj::<4>(nvert, edges)
select_sdf(pos, sdf, out)
pick_edges(pos, edges, ray, radius)
fit_group_aabbs(pos, group_offsets, group_items, out)
```

Group membership is generic indexing/topology data; AABBs/Buckets/hash grids are `spacc` caches. This keeps the primary representation transferable to rendering/editing while spatial acceleration can be rebuilt independently after geometry changes.

---

## Cross-repo import priority summary

| Priority | Source | What | Target in SurfMol |
|----------|--------|------|-------------------|
| P0 | FireCore | `Atoms.h` positioned particle arrays + geometry | `pgraph` design reference |
| P0 | FireCore | `MMFFBuilder.h` Atom/Conf/Bond builder algorithms | `moltopo` builder (every atom gets Conf) |
| P0 | FireCore | `UFF.h` / `MMFFsp3_loc.h` fixed `K=4` adjacency layouts | `pgraph::FixedAdj<K>` + compiled `molff` layouts |
| P0 | FireCore | `MolecularGraph.h` CSR + components/bridges | `pgraph_ops` |
| P0 | FireCore | `Groups.h` + `NBFF::initBBsFromGroups` | `Partition/IndexGroups` + `spacc` bounds |
| P0 | FireCore | UFF + NBFF kernels / `MolWorld_sp3::MDloop()` | `molff` / performance benchmark |
| P0 | SimpleSimulationEngine | `Buckets` count→prefix→scatter | `spacc` / generic packed-index helpers |
| P0 | SimpleSimulationEngine | `MeshBuilder2` edit/compact/loop algorithms | `pgraph_ops` |
| P0 | blood_of_civilization | Cargo footprint/profile settings | workspace tooling |
| P0 | learn_Rust | OpenCL-GL zero-copy interop + aligned Rust math | GPU/runtime layers |
| P0 | SPAMMM | NVIDIA-first OpenCL device setup | Rust OpenCL layer |
| P1 | SimpleSimulationEngine | SDF selection, picking, geometry | `pgraph_ops` + `numcore::geom` |
| P1 | SimpleSimulationEngine | `NeighChunks` overflow idea | optional dynamic adjacency, later |
| P1 | learn_Rust | group AABB broad phase / uniform grid / Morton | `spacc` + OpenCL kernels |
| P1 | FireCore | RRsp3 / Projective Dynamics / rigid-body code | `molff` / RAFF |
| P1 | SPAMMM | `rigid.cl`, nonbonded, AtomicGraph lessons | relevant runtime/topology layers |
| P2 | FireCore | GridFF, Ewald2D, RARFF, GOpt | later force-field features |
| P2 | SSE | `Table`, `BatchBuff` | only if concrete need appears |

## Resolved / current design decisions
1. **OpenCL crate:** **`ocl` 0.19** (from learn_Rust). Higher-level ProQue/Buffer API; keep unsafe isolated in the OpenCL boundary.
2. **Common geometry/topology core:** use a tiny **`pgraph`** concept (positions + edges + fundamental index containers), not a pure abstract graph and not a molecule-specific editor object.
3. **Transferable algorithms:** keep in **`pgraph_ops`** so consumers that only need the data contract do not pull editing/selection/topology machinery.
4. **Spatial acceleration:** **`spacc`** is separate (`Aabb`, Buckets, grids, Morton, broad phase). Spatial structures are rebuildable caches, not intrinsic `PGraph` fields.
5. **Higher-order elements:** triangles/angles and polygons/rings may reuse `Elements<N>`/`Ragged` storage, but are **not semantically identified** in the core.
6. **Neighbor hot layout:** bounded kernels use `FixedAdj<K>` (ELL-like padded rows, `-1` invalid); arbitrary-degree algorithms can use `CsrAdj`.
7. **Dynamic molecular Conf:** every atom, including caps, gets the Conf/valence fields; eliminate FireCore's optional `iconf` indirection.
8. **Fragment memory layout:** bake/reorder to **contiguous fragments** when performance benefits; retain generic group/partition representation while editing.
9. **Capping atoms (H, epairs):** the topology builder may represent explicit caps uniformly; whether they carry independent dynamics or are rigid appendices is a force-field/integrator choice, not a `pgraph` rule.
