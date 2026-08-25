# Ownership & Architecture Design — SurfMol Forcefield System

## 1. Overview

This document describes the ownership model, data flow, and composability strategy of the Rust molecular forcefield engine (`surfmol-forcefields`). The system is built around the principle **"borrow, don't own"** — each forcefield module owns only its specialized data and borrows shared atomic state during evaluation. This avoids the monolithic "super-class" anti-pattern and keeps the design modular, testable, and zero-overhead.

## 2. Design Goals

1. **Separation of Concerns**: Static topology, dynamic state, and forcefield parameters must live in distinct structs.
2. **Zero-Copy Evaluation**: Forcefields must operate on borrowed slices, never copy atom positions or forces.
3. **Dynamic Composability**: Any subset of forcefields (bonded + nonbonded + surface + rigid-body) must be runnable without structural changes.
4. **No Monolithic Owner**: `MolWorld` orchestrates but does not accumulate unrelated data fields.
5. **Testability**: Each module can be unit-tested in isolation with mock slices.
6. **Parity with C++ Reference**: Keep naming and layout close to the C++ `UFF.h` / `NBFF.h` / `Atoms.h` stack for cross-language reasoning.

## 3. Core Data Hierarchy

```
DynamicAtoms                    (MD state: geometry + forces + velocities)
├── Atoms                       (static geometry + types + neighbors)
│   ├── natoms: i32
│   ├── atypes: Vec<usize>      (atom type indices, shared by all FF modules)
│   ├── apos: AlignedVec        (positions, 64-byte aligned)
│   ├── neighs: AlignedVec      (neighbor atom indices per atom)
│   └── neigh_bs: AlignedVec    (neighbor bond indices per atom)
├── fapos: AlignedVec           (forces on positions)
└── vapos: AlignedVec           (velocities)
```

`DynamicAtoms` is the **single owner** of all per-atom dynamic arrays. It also provides the MD integrators (`move_atom_md`, `run_md`, `clean_force`, `clean_velocity`) because these operate on `apos/fapos/vapos` and belong with the state they mutate.

## 4. Forcefield Modules — What Each Owns

### 4.1 `Uff` (Bonded Forcefield)

**Owns:**
- `bon_atoms`, `ang_atoms`, `dih_atoms`, `inv_atoms` — topology indices
- `ang_ngs`, `dih_ngs`, `inv_ngs` — precomputed neighbor-slot offsets for fast force assembly
- `hneigh` — cached normalized bond vectors (updated each frame)
- `fint`, `fbon`, `fang`, `fdih`, `finv` — internal force buffers for terms before assembly
- `a2f` — bucket map for assembling `fint` into per-atom forces
- `bon_params`, `ang_params`, `dih_params`, `inv_params` — forcefield constants

**Does NOT own:**
- `apos`, `fapos`, `vapos` (moved to `DynamicAtoms`)
- `neighs`, `neigh_bs` (moved to `Atoms`)

**Borrow pattern:**
```rust
pub fn eval_forces(&mut self, apos: &[Vec3d], fapos: &mut [Vec3d], neighs: &[Quat4i], neigh_bs: &[Quat4i]) -> (f64, f64, f64, f64)
```

**Rationale:** Bonded topology is immutable after setup, but positions and forces change every timestep. By borrowing slices, `Uff` stays lightweight and multiple `Uff`-like modules could theoretically coexist with the same geometry.

### 4.2 `NonBondedFF` (Non-Bonded Forcefield)

**Owns:**
- `reqs: AlignedVec<[f64; 4]>` — per-atom van-der-Waals radius, epsilon, charge, hydrogen-bond parameter
- `plqs: AlignedVec<[f64; 4]>` — precomputed Pauli/London/Q/Hb coefficients for surface coupling
- `excl` — 1-2 and 1-3 exclusion lists
- `lvec`, `npbc`, `shifts` — periodic boundary condition state
- Cutoff constants

**Does NOT own:**
- Any atomic positions or forces (borrows from `DynamicAtoms`)

**Borrow pattern:**
```rust
pub fn eval(&mut self, fapos: &mut [Vec3d], apos: &[Vec3d]) -> f64
```

**Rationale:** Non-bonded parameters (`REQs`) are distinct from bonded topology and change independently (e.g., when charges are reassigned). Keeping them separate allows swapping non-bonded models without touching `Uff`.

### 4.3 `SurfaceFolded` (Substrate Interaction)

**Owns:**
- FFT grid, harmonics, coefficient buffers for the corrugated surface potential
- `ntypes`, `coefs`, `qls`, `E0qs` — type-dependent surface parameters

**Does NOT own:**
- Any per-atom arrays

**Borrow pattern:**
```rust
pub fn eval_all_scratch(&self, apos: &[Vec3d], plqs: &[[f64; 4]], fapos: &mut [Vec3d], scratch: &mut SurfaceScratch) -> f64
```

**Key design decision:** `SurfaceFolded` receives **precomputed PLQ coefficients** from `NonBondedFF` rather than raw atom types + REQ parameters. This eliminates:
- Per-atom type lookups inside the hot loop
- On-the-fly `req2plq` conversion during surface evaluation

**Rationale:** The surface potential is a mixing function of the substrate geometry and the probe atom's non-bonded coefficients. Precomputing PLQs once (in `NonBondedFF::make_plqs`) amortizes the cost across all surface evaluations.

### 4.4 `RigidSp3FF` (Rigid-Body Constraint)

**Owns:**
- `quat`, `omega`, `tau` — rotational state per atom
- `nport`, `port_local` — rigid-body port geometry (sp3/sp2/sp1)

**Does NOT own:**
- Positions, forces, velocities, neighbors, bond params

**Borrow pattern:**
```rust
pub fn eval_forces(&mut self, apos: &[Vec3d], fapos: &mut [Vec3d], uff: &Uff, neighs: &[Quat4i], neigh_bs: &[Quat4i]) -> f64
pub fn move_atom_md(&mut self, i: usize, apos: &mut [Vec3d], fapos: &[Vec3d], vapos: &mut [Vec3d], uff: &Uff, neigh_bs: &[Quat4i], dt: f64, flim: f64, cdamp: f64) -> (f64, f64, f64)
```

**Rationale:** `RigidSp3FF` is an optional bonded-force alternative to `Uff`. It needs `Uff` for bond parameters and neighbor topology, but not for positions/forces. This shows how a new bonded model can be dropped in without changing `MolWorld`'s data layout — only the dispatch logic in `eval_forces` and `move_atom_md` needs a new match arm.

## 5. `MolWorld` — Orchestrator, Not Owner

`MolWorld` is deliberately **not** a data container. It owns handles to the modules and dispatches evaluation/MD in the correct order:

```rust
pub struct MolWorld {
    pub dyn_atoms: DynamicAtoms,        // THE shared atomic state
    pub uff: Uff,                        // bonded topology + params
    pub rigid_sp3: RigidSp3FF,           // optional rigid-body constraint
    pub bonded_mode: BondedFFMode,       // Uff | RigidSp3
    pub nonbonded: Option<NonBondedFF>, // van-der-Waals + Coulomb
    pub surface: Option<SurfaceFolded>,  // substrate interaction
}
```

### 5.1 Evaluation Flow (`eval_forces`)

```
1. Borrow slices from dyn_atoms:
   apos  = dyn_atoms.atoms.apos.as_slice()
   fapos = dyn_atoms.fapos.as_mut_slice()
   neighs = dyn_atoms.atoms.neighs.as_slice()
   neigh_bs = dyn_atoms.atoms.neigh_bs.as_slice()

2. Bonded forces:
   Uff:       uff.eval_forces(apos, fapos, neighs, neigh_bs)
   RigidSp3:  rigid_sp3.eval_forces(apos, fapos, &uff, neighs, neigh_bs)

3. Non-bonded forces (accumulate into same fapos):
   nonbonded.eval(fapos, apos)

4. Surface forces (accumulate into same fapos):
   surface.eval_all_scratch(apos, plqs_from_nonbonded, fapos, scratch)
```

All forcefields write into the **same** `fapos` array. There is no merging step — forces accumulate directly.

### 5.2 MD Flow (`run_md`)

```
for each step:
    eval_forces()          // all modules write to dyn_atoms.fapos
    for each atom i:
        move_atom_md(i)    // dispatches to basic MD or rigid-body MD
```

`DynamicAtoms::move_atom_md` handles standard translation; `RigidSp3FF::move_atom_md` adds quaternion rotation. `MolWorld` dispatches based on `bonded_mode`.

## 6. Key Design Decisions & Justifications

### Decision 1: Neighbor lists in `Atoms`, not `Uff`

**Why:** Neighbor lists (`neighs`, `neigh_bs`) are a geometric property of the topology, not a forcefield-specific optimization. Both `Uff` and `RigidSp3FF` need them. Moving them to `Atoms` means:
- `Uff` becomes purely "bonded parameter + topology index" storage
- New forcefields that need neighbors don't depend on `Uff`
- `Atoms::make_neigh_bs` can be called once during setup, before any forcefield exists

**Trade-off:** Slightly more arguments passed to `Uff` methods. This is negligible compared to the architectural clarity gained.

### Decision 2: Precompute PLQs in `NonBondedFF`, not `SurfaceFolded`

**Why:** `req2plq(alpha)` converts `[RvdW, sqrt(EvdW), Q, Hb]` → `[Pauli, London, Q, Hb]`. This conversion is expensive (involves power functions). If done inside `SurfaceFolded::eval_atom`, it would be repeated every timestep for every atom. By precomputing in `NonBondedFF::make_plqs(alpha)`, we:
- Amortize the cost to O(natoms) once per parameter change
- Allow `SurfaceFolded` to receive a flat `[f64; 4]` per atom with no type lookup
- Make the surface evaluation kernel simpler and faster

**Trade-off:** `NonBondedFF` must expose `plqs` as a public slice. This is acceptable because `plqs` is derived data with no mutable interior.

### Decision 3: `DynamicAtoms` owns MD integrators

**Why:** `move_atom_md` mutates `apos`, `vapos`, and reads `fapos`. These are all fields of `DynamicAtoms`. Co-locating the integrator with the state it mutates:
- Makes the borrow checker happy (all accessed fields are `self` fields)
- Keeps MD logic in one place
- Allows `DynamicAtoms::run_md` to take a generic `eval_forces` closure, making it reusable across different forcefield compositions

**Trade-off:** `MolWorld` must dispatch `move_atom_md` to either `DynamicAtoms` (Uff mode) or `RigidSp3FF` (rigid mode) based on `bonded_mode`. This adds a thin dispatch layer, but keeps the door open for more integrator variants.

### Decision 4: No `surface_atom_types` in `MolWorld`

**Why (removed):** Previously, `MolWorld` had a `surface_atom_types: Vec<usize>` for surface type indices. This was redundant with `atypes` and created a synchronization burden. The surface now receives PLQ coefficients directly, which are already atom-specific and type-agnostic.

## 7. Pros and Cons

### Pros
- **Modularity:** Each forcefield can be tested in isolation with mock slices.
- **Composability:** Adding a new forcefield only requires a new `Option<Module>` in `MolWorld` and a call in `eval_forces`.
- **Zero-overhead:** All forcefield methods use borrowed slices; no copying or heap allocation during evaluation.
- **Cache-friendly:** Flat `AlignedVec` arrays for positions, forces, and parameters.
- **Clear ownership:** Rust borrow checker enforces that only one module writes to `fapos` at a time (sequentially), while multiple can read `apos`.
- **Parity with C++:** The split between `Atoms`, `ForceField` (dynamics), and `NBFF` in the C++ codebase is mirrored here.

### Cons
- **Verbosity:** Methods now take more arguments (apos, fapos, neighs, neigh_bs). This is the price of explicit borrowing.
- **Dispatch overhead:** `MolWorld` has a small runtime dispatch (`match bonded_mode`) in `eval_forces` and `move_atom_md`. This is negligible compared to the force evaluation cost.
- **Lifetime complexity:** Borrowing slices from `dyn_atoms` while also borrowing `self.uff` or `self.rigid_sp3` requires careful ordering or raw field access to satisfy the borrow checker.
- **No "single struct" convenience:** You cannot pass `&mut MolWorld` to a function and let it reach into `apos` and `fapos` freely without borrowing conflicts. The explicit slice extraction in `eval_forces` is a deliberate discipline.

## 8. Future Adaptations

### Adding a new forcefield (e.g. external field, QM/MM coupling)

1. Create a new struct that owns its specialized data (e.g. grid, coefficients, QM region mask).
2. Provide an `eval(&mut self, fapos: &mut [Vec3d], apos: &[Vec3d], ...) -> f64` method.
3. Add `Option<NewFF>` to `MolWorld`.
4. Insert a call in `MolWorld::eval_forces` after the existing forcefields.

No changes to `DynamicAtoms`, `Atoms`, or existing forcefields are needed.

### Adding a new MD integrator (e.g. Langevin, FIRE)

1. Add a new method to `DynamicAtoms` that mutates `apos`/`vapos`/`fapos`.
2. Or create a standalone integrator struct that borrows `&mut DynamicAtoms`.

The `run_md` closure-based API in `DynamicAtoms` already supports this.

### Parallel evaluation (future)

The current design intentionally avoids mutable aliasing of `fapos` within a single forcefield. To parallelize across forcefields (e.g. bonded and non-bonded concurrently), we would:
1. Evaluate each forcefield into a **private** `fapos` buffer
2. Sum buffers into `dyn_atoms.fapos`

This fits naturally because each forcefield already operates on borrowed slices — we just change the output buffer.

### GPU offload (future)

The flat array layout (`AlignedVec<Vec3d, 64>`) is already GPU-friendly. To offload:
1. Upload `apos`, `neighs`, `neigh_bs`, `bon_params`, etc. to device once.
2. Upload updated `apos` each timestep.
3. Download `fapos` after evaluation.

The separation between `Atoms` (upload once) and `DynamicAtoms` (upload/download each step) makes this explicit.

## 9. Glossary

| Term | Meaning |
|------|---------|
| `REQ` | van-der-Waals Radius, sqrt(Epsilon), charge, H-bond param per atom |
| `PLQ` | Pauli repulsion, London dispersion, charge, H-bond coefficient |
| `hneigh` | Cached normalized bond vectors (Quat4d: dx, dy, dz, 1/r) |
| `a2f` | Atom-to-force bucket map for assembling per-term forces |
| `AlignedVec<T, A>` | Heap array with alignment `A` (64 bytes for SIMD) |

## 10. Related Files

| File | Role |
|------|------|
| `common/src/molecular.rs` | `Atoms`, `DynamicAtoms` |
| `forcefields/src/uff.rs` | Bonded topology & force kernels |
| `forcefields/src/nonbonded.rs` | LJ + Coulomb non-bonded evaluation |
| `forcefields/src/surface.rs` | Corrugated surface potential |
| `forcefields/src/rigid_sp3.rs` | Rigid-body rotational constraints |
| `forcefields/src/mol_world.rs` | Orchestrator (`MolWorld`) |
| `apps/src/main.rs` | Application wiring |
