---
type: userguide
title: "UFF & SPFF Forcefields — Setup, Parameters, and Relaxation"
description: End-user guide for the Universal Force Field (UFF) and Surface-Pi Force Field (SPFF) in SurfMol — how to assign atom types, load parameters, run relaxation, and interpret results for aromatic molecules.
tags: [user-guide, forcefield, uff, spff, aromatic, relaxation, fire, params]
timestamp: 2026-09-29
---

# UFF & SPFF Forcefields

This guide covers how to use the **Universal Force Field (UFF)** and **Surface-Pi Force Field (SPFF)** in SurfMol: assigning atom types, loading parameters from `.dat` files, running physical relaxation, and interpreting the results — especially for aromatic molecules like pentacene.

## What is UFF?

UFF (Rappé et al., 1992) is a general-purpose molecular forcefield that covers all elements of the periodic table. It models four bonded interaction types:

| Term | Formula | What it controls |
|------|---------|------------------|
| **Bond** | `E = ½·k·(l−l₀)²` | Bond stretching |
| **Angle** | `E = k·(c₀+c₁cosθ+c₂cos2θ+c₃cos3θ)` | Angle bending (Fourier form) |
| **Dihedral** | `E = ½·V·(1+d·cos(n·φ))` | Torsion around bonds |
| **Inversion** | `E = K·(C₀+C₁cosω+C₂cos2ω)` | Out-of-plane bending at sp² centers |

The **inversion** term is critical for aromatic molecules — it penalizes pyramidalization at trigonal (sp²) centers, keeping planar molecules like benzene and pentacene flat.

## What is SPFF?

SPFF (Surface-Pi Force Field) extends UFF with an explicit **pi-orbital bending** term that couples the pi-orbital direction to the sigma-bond plane. This provides stiffness against **global bending** of aromatic sheets — something UFF inversions alone cannot do (UFF inversions only penalize *local* pyramidalization, not smooth sheet curvature).

SPFF adds a `Ksp` term: `E_sp = ½·Ksp·(1−cos²α)` where `α` is the angle between the pi-orbital and the normal to the sigma-bond plane. This is the key term for modeling pentacene on surfaces, where the molecule must remain flat against the substrate.

> **Status:** SPFF is implemented in the OpenCL kernels (`opencl/SPFF.cl`) and the `oclff` crate, but not yet wired into the Rust CPU `molff` path. UFF is fully functional on CPU.

## Prerequisites

- Rust toolchain (stable, 2021 edition)
- UFF parameter files in `data/`:
  - `ElementTypes.dat` — element properties (radii, charges, colors)
  - `AtomTypes.dat` — per-atom-type UFF parameters
  - `BondTypes.dat` — bond parameters (k, l0)
  - `AngleTypes.dat` — angle parameters (k, θ₀)
  - `DihedralTypes.dat` — dihedral parameters (V, n, phase)

## The UFF parameter pipeline

Running UFF requires three steps: **(1) build topology → (2) assign UFF types → (3) load parameters**. Each step feeds the next.

### Step 1: Build topology

Topology = bonds, angles, dihedrals, inversions derived from atom positions. The `moltopo` crate provides two paths:

```rust
use moltopo::topology::{Topology, build_bonds_by_cutoff, build_angles_from_bonds, build_dihedrals_from_bonds, build_inversions_from_bonds};

// Path A: simple cutoff-based bonding
let bonds = build_bonds_by_cutoff(&apos, 1.8);  // 1.8 Å cutoff
let angles = build_angles_from_bonds(natoms, &bonds);
let dihedrals = build_dihedrals_from_bonds(&bonds);
let inversions = build_inversions_from_bonds(natoms, &bonds);
let top = Topology { apos, bonds, angles, dihedrals, inversions };

// Path B: Builder with covalent radii (more accurate)
use moltopo::builder::Builder;
let top = Builder::from_positions_and_radii(&apos, &elems, &radii, 0.4).bake();
```

**Dihedral deduplication:** `build_dihedrals_from_bonds` keeps only one of `(i,j,k,l)` and `(l,k,j,i)` — the same dihedral viewed from opposite ends. This matches FireCore's `i4 > i1` convention. Without dedup, duplicate dihedrals produce opposite forces that cancel, causing zero-energy non-planar states.

**Inversions:** `build_inversions_from_bonds` generates 3 inversions per trigonal (sp²) center — one per neighbor being the "out-of-plane" atom. This matches FireCore `UFFbuilder.h:1318-1332`.

### Step 2: Assign UFF atom types

UFF atom types encode hybridization: `C_3` (sp³ carbon), `C_R` (aromatic/sp² carbon), `C_2` (sp² non-aromatic), `C_1` (sp carbon), `H_` (hydrogen), etc.

```rust
use moltopo::assign_uff::assign_uff_types;

// neighs: [[i32; 4]; natoms] — up to 4 neighbor atom indices per atom (-1 padded)
let uff_types = assign_uff_types(&elems, &neighs);
// Returns: ["C_R", "C_R", "C_R", ..., "H_", "H_", ...]
```

The assignment uses octet-rule hybridization: `4 = n_epair + n_sigma + n_pi` where `n_sigma = n_neighbors`. Special cases:
- **Aromatic C/N/O** → `_R` suffix (resonant)
- **Carbonyl O** → `O_2`
- **Nitro N** → `N_R` with bond order 2
- **Alkyne C** → `C_1` with triple bond

### Step 3: Load parameters and set up the forcefield

This is the critical step that fills the `Uff` struct's parameter arrays with real physical values.

```rust
use moltopo::params::Params;
use surfmol::mol_world::{MolWorld, BondedFFMode};

// Load .dat parameter files
let mut params = Params::new();
params.load_element_types("data/ElementTypes.dat");
params.load_atom_types("data/AtomTypes.dat");
params.load_bond_types("data/BondTypes.dat");
params.load_angle_types("data/AngleTypes.dat");
params.load_dihedral_types("data/DihedralTypes.dat");

// Build MolWorld from topology
let mut mw = MolWorld::from_topology(&top);
mw.bonded_mode = BondedFFMode::Uff;
mw.make_neigh_bs();

// Assign UFF types
let neighs_arr: Vec<[i32; 4]> = mw.dyn_atoms.neighs().iter().map(|q| q.as_array()).collect();
let types = assign_uff_types(&elems, &neighs_arr);

// *** Fill UFF parameter arrays from .dat files + UFF formulas ***
mw.setup_uff_params(&params, &types);

// Bake neighbor caches for angle/dihedral/inversion force evaluation
mw.bake_angle_neighs();
mw.bake_dihedral_neighs();
mw.bake_inversion_neighs();
mw.map_atom_interactions();
```

### What `setup_uff_params` does

`MolWorld::setup_uff_params` calls `Uff::setup_params`, which is a port of FireCore's `assignUFFparams`. It fills four parameter arrays:

| Array | Size | Formula / Source |
|-------|------|------------------|
| `bon_params[ib]` | `[k, l0]` | `l0 = uff_bond_length(ri, rj, bo, χi, χj)`, `k = 0.5·28.8·Qi·Qj/l0³` |
| `ang_params[ia]` | `[k, c0, c1, c2, c3]` | Fourier coefficients from hybridization (sp1/sp2/sp3), `κ` from Coulomb-like UFF formula |
| `dih_params[id]` | `[V, d, n]` | `V·(1+d·cos(n·φ))` — V, d, n from central atom hybridizations |
| `inv_params[ii]` | `[K, C0, C1, C2]` | `K=6` kcal/mol for sp² C/N, `K=50` for carbonyls; `C=[1,−1,0]` |

**Bond order for aromatic bonds:** Aromatic `C_R–C_R` bonds use bond order **1.0** (not 1.5 or 2.0), matching FireCore. Using bond order 2.0 causes ~3.8 eV of compressive strain in pentacene, which exceeds the inversion barrier and causes buckling.

### `set_dummy_params` — for testing only

```rust
mw.set_dummy_params();  // bond-only: k=100, l0=current length; all other params zero
```

`set_dummy_params` fills bond parameters from the current geometry (k=100 eV/Å², l0 = current bond length) and zeros all angle/dihedral/inversion parameters. This is useful for **kernel testing** (verifying force evaluation code paths) but produces **non-physical** results — no angle bending, no torsion, no planarity enforcement.

## Running relaxation

### Using FIRE (recommended for tests)

FIRE (Fast Inertial Relaxation Engine, Bitzek et al., 2006) is the most robust relaxation method for UFF. It adaptively adjusts the timestep and zeroes velocities when the system overshoots.

```rust
use molff::raff::FireState;

let mut fire = FireState::new(0.001, 0.05);  // dt=0.001, alpha=0.05

for itr in 0..5000 {
    let (eb, ea, ed, ei, _, _) = mw.eval_forces();
    let e = eb + ea + ed + ei;
    
    // FIRE update: mix velocity with force direction, adapt dt
    // (see molff::raff::step_fire for the full implementation)
    
    if fmax < 1e-3 { break; }  // converged
}
```

### Using the editor GUI

The editor uses **damped MD with inertial velocity reset** (not FIRE) for UFF/RigidSp3 modes:

1. Load a molecule: `cargo run -p editor -- data/xyz/pentacene.xyz`
2. Press `F` to cycle to **UFF** mode (auto-selected if sp² atoms detected)
3. Press `SPACE` to start relaxation
4. Watch the energy decrease in the top-left panel
5. Press `SPACE` again to stop

The editor's `do_relax_step` zeros all velocities when `Σ v·f < 0` (energy overshoot), which acts as a poor-man's FIRE. For production relaxation, use the test harnesses or `molengine` with FIRE.

### Using molengine (CLI)

```rhai
// relax.rhai — Rhai script for molengine
let sim = load_topology("pentacene.json");
let natoms = get_natoms(sim);
print(`Loaded ${natoms} atoms`);

let e0 = eval_forces(sim);
print(`Initial energy: ${e0}`);

let niter = relax(sim, 5000, 0.02, 0.001, 1000.0, 0.1);
print(`Relaxed in ${niter} steps`);

let e1 = eval_forces(sim);
print(`Final energy: ${e1}`);
```

> **Note:** `molengine` currently loads topology via `load_topology_from_json`, which creates a `Uff` with **zero parameters**. You must call `setup_uff_params` before `eval_forces` will produce physical results. This wiring is planned but not yet implemented in the Rhai API.

## Interpreting results

### Energy components

`eval_forces` returns 6 energy components:

| Component | Symbol | What it measures |
|-----------|--------|------------------|
| `E_bond` | Eb | Bond stretching energy |
| `E_angle` | Ea | Angle bending energy |
| `E_dih` | Ed | Dihedral torsion energy |
| `E_inv` | Ei | Inversion (out-of-plane) energy |
| `E_nb` | Enb | Non-bonded (LJ + Coulomb) energy |
| `E_surf` | Es | Surface interaction energy |

For a **planar aromatic molecule at equilibrium**: `E_bond ≈ 0`, `E_angle ≈ 0`, `E_dih = 0`, `E_inv = 0`. Non-zero values indicate strain.

### Pentacene example

Pentacene (C₂₂H₁₄) is a good test case — 36 atoms, 40 bonds, 66 angles, 104 dihedrals, 66 inversions.

**Planar geometry (UFF equilibrium):**
```
E_bond=3.834 eV  E_angle=0.010  E_dih=0.000  E_inv=0.000
```
The 3.8 eV bond energy comes from UFF bond lengths (1.458 Å for aromatic C–C) being slightly longer than DFT-optimized geometry (1.40 Å). This is a known UFF limitation.

**Out-of-plane displacement (atom 0, z=0.5 Å):**
```
E_bond=3.979  E_angle=0.398  E_dih=2.634  E_inv=0.158
Fz = -11.72 eV/Å  (restoring force toward plane)
```
All four terms contribute to the restoring force. Dihedrals dominate for large displacements; inversions dominate for local pyramidalization.

### Local pyramidalization vs global bending

A key physical insight from the pentacene tests:

| Distortion type | E_inv | E_dih | Relaxes to planar? |
|----------------|-------|-------|---------------------|
| **Local pyramidalization** (one atom pushed out) | 0.060 eV | 0.805 eV | **Yes** — z_rms 0.05→0.012 |
| **Global parabolic bend** (smooth bow shape) | 0.0004 eV | 0.012 eV | **Partially** — UFF dihedral multi-well (φ=0 and φ=π) allows non-planar minima |

**Why?** UFF inversions penalize *local* pyramidalization (one atom out of plane), not *global* bending (smooth sheet curvature). For a smooth bend, neighboring atoms move together, so the local out-of-plane angle at each center is nearly zero → negligible inversion energy.

**SPFF solves this:** The `Ksp` pi-sigma orthogonality term directly couples the pi-orbital direction to the sigma-bond plane, penalizing any out-of-plane deformation regardless of smoothness. This is the physically correct term for aromatic sheet stiffness.

## Test harnesses

SurfMol includes two UFF relaxation tests that serve as usage examples:

### `relax_pentacene_uff.rs`

Tests full UFF stiffness: in-plane relaxation → out-of-plane distortion → relaxation back to planar.

```bash
cargo test -p surfmol --test relax_pentacene_uff -- --nocapture
```

Artifacts saved to `debug/relax_pentacene_uff/`:
- `trajectory_fire.tsv` — energy, fmax, z_rms per step

### `relax_pentacene_bend.rs`

Tests bend relaxation: parabolic bend + local pyramidalization, with .xyz trajectory output and plotting.

```bash
cargo test -p surfmol --test relax_pentacene_bend -- --nocapture
```

Artifacts saved to `debug/relax_pentacene_bend/`:
- `pentacene_planar.xyz` — planar baseline
- `pentacene_bent.xyz` — bent state (before)
- `pentacene_bend_relaxed.xyz` — after global bend relaxation
- `pentacene_pyr_bent.xyz` — pyramidalized state (before)
- `pentacene_pyr_relaxed.xyz` — after pyramidalization relaxation
- `pentacene_pyr_traj.xyz` — multi-frame .xyz trajectory (viewable in VMD/Ovito)
- `before_after_pyr.png` — 3D scatter: pyramidalized vs relaxed vs planar
- `convergence_pyr.png` — E, fmax, z_rms vs FIRE step
- `trajectory_pyr_overlay.png` — z(x) profile at selected steps

### `relax_xylitol_uff.rs`

Tests UFF on a flexible sp³ molecule (xylitol) with `set_dummy_params` (bond-only).

```bash
cargo test -p surfmol --test relax_xylitol_uff -- --nocapture
```

## Using `buildff` to inspect topology

The `buildff` CLI tool builds topology and assigns UFF types, exporting to JSON or binary:

```bash
# JSON output (human-readable)
cargo run -p buildff -- data/xyz/pentacene.xyz --json pentacene.json

# Binary output (for MD ingestion)
cargo run -p buildff -- data/xyz/pentacene.xyz --bin pentacene.ufftopo

# Custom bonding tolerance
cargo run -p buildff -- data/xyz/pentacene.xyz --tol 0.45 --json pentacene.json
```

The JSON output includes per-atom: element, position, UFF type, hybridization, neighbors. This is useful for verifying that UFF types are assigned correctly before running relaxation.

> **Note:** `buildff` only assigns UFF *type strings* — it does not compute or export UFF *parameters* (k, l0, etc.). The consumer (editor, molengine, test harness) must load `.dat` files and call `setup_uff_params`.

## Theory: UFF inversion term

The inversion (improper torsion) term at a trigonal center atom `I` with three bonded neighbors `J`, `K`, `L` is:

```
E_inv = K · (C₀ + C₁·cos(ω) + C₂·cos(2ω))
```

where `ω` is the angle between the bond `I→J` and the plane defined by `I, K, L` (the "out-of-plane" angle). For a planar sp² center, `ω = 0` for all three neighbors, and `E_inv = K·(C₀+C₁+C₂) = 0` (the coefficients are chosen so the planar state is the minimum).

Three inversions are generated per trigonal center (one per neighbor being the "out-of-plane" atom), matching FireCore's `UFFbuilder.h:1318-1332`.

**K values:**
- sp² C, N: `K = 6` kcal/mol = 0.26 eV
- Carbonyl C (=O): `K = 50` kcal/mol = 2.17 eV
- sp³ atoms: `K = 0` (no inversion penalty — tetrahedral geometry)

## See also

- [`/userguide/editor.md`](/userguide/editor.md) — editor GUI guide (forcefield mode cycling, relaxation)
- [`/doc/topical_audit/uff.md`](/doc/topical_audit/uff.md) — UFF cross-implementation map (SurfMol Rust vs FireCore C++ vs SPAMMM)
- [`/crates/libs/molff/README.md`](/crates/libs/molff/README.md) — `molff` crate docs (UFF force kernels)
- [`/crates/libs/moltopo/README.md`](/crates/libs/moltopo/README.md) — `moltopo` crate docs (topology, UFF type assignment, params)
- `FireCore` — C++ reference implementation (`src/forcefields/uff.cpp`, `UFFbuilder.h`)
- `SPAMMM` — Python/OpenCL reference (UFF + SPFF)
- Rappé, A. K., et al. "UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations." *J. Am. Chem. Soc.* 114.25 (1992): 10024-10035.
