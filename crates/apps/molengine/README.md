---
type: rust-app
title: molengine
description: CLI MD/relaxation engine — Rhai-scripted. Loads topology (TopologyData JSON), runs UFF and RAFF forcefield evaluation and relaxation. Supports all 6 RAFF solver modes (ForceMD, InertialReset, FIRE, PBD, XPBD, Projective), harmonic box constraint, non-bonded LJ+Coulomb.
tags: [rust, crate, cli, md, simulation, rhai, raff, uff, relaxation, projective-dynamics, xpbd, fire, box-constraint]
timestamp: 2026-08-29
---

# molengine

CLI molecular simulation engine. Loads a molecular topology (with forcefield types already assigned by `buildff`), constructs a `MolWorld` from the `surfmol` integration crate, and exposes MD/relaxation operations to Rhai scripts. Supports both UFF and RAFF forcefields with all 6 RAFF solver modes.

## Role in the pipeline

`molengine` is the **runner** in SurfMol's builder/runner architecture (see [`crates/apps/README.md`](../README.md)). It consumes `TopologyData` JSON produced by `buildff` and runs forcefield evaluation / relaxation controlled by a Rhai script. It has no GUI — it is a pure compute engine for batch processing and pipelines.

```
buildff (builder)                molengine (runner)
  XYZ → topo.json  ─────────────►  topo.json + script.rhai → relaxed.xyz
```

## Usage

```
molengine --script <rhai_file>
```

## Rhai API

### Topology & params

| Function | Description |
|----------|-------------|
| `load_topology(path) → sim` | Load `TopologyData` JSON (from `buildff --json`), returns `SimulationEngine` |
| `setup_uff_params(sim, data_dir)` | Load `.dat` files from `data_dir` and fill UFF parameter arrays (bonds, angles, dihedrals, inversions). **Must be called before `eval_forces`/`relax`** — JSON load creates UFF with zero params. |
| `get_natoms(sim) → int` | Number of atoms |

### UFF relaxation

| Function | Description |
|----------|-------------|
| `eval_forces(sim) → float` | Evaluate UFF forces (bonds + angles + dihedrals + inversions + non-bonded + surface), return total energy |
| `step_md(sim, dt, flim, damping)` | One UFF MD step (velocity Verlet with force clamping + damping) |
| `relax(sim, niter, dt, fconv, flim, damping) → int` | UFF relaxation loop, returns number of steps performed |

### RAFF setup

| Function | Description |
|----------|-------------|
| `setup_raff(sim)` | Build `RaffTopology` + `RaffState` from the loaded UFF bond topology + positions. Uses per-atom ARAP port geometry (`set_port_geometry_from_reference` — identity rotation = E_port=0 at input geometry). |
| `set_raff_solver(sim, mode)` | Select solver: `"forcemd"` / `"inertial"` / `"fire"` / `"pbd"` / `"xpbd"` / `"projective"` |
| `set_raff_orient(sim, mode)` | Orientation: `"adiabatic"` (memoryless Wahba/Horn, default) / `"dynamic"` (physical inertia, inner-coupled for Projective) |
| `set_raff_dt(sim, dt)` | Timestep (default 0.01; 0.1 for Projective with inner-coupled rotation) |
| `set_raff_damping(sim, damping)` | Damping factor (0=kill velocity, 1=no damping; 0.9 typical for relaxation) |
| `set_raff_iters(sim, iters)` | Inner iterations for position-based solvers (default 4; 8+ with heavy-ball for stiff systems) |
| `set_raff_hb(sim, momentum)` | Heavy-ball momentum for inner Jacobi (0=disabled, 0.5–0.75 typical for stiff systems) |
| `set_raff_pd_inertia(sim, bool)` | Enable/disable PD outer-loop inertia (predict `x += v·dt` before inner solve; true = proper Projective Dynamics) |
| `set_raff_vel_reset(sim, bool)` | Enable/disable velocity reset when `v·F < 0` (uphill — like FIRE/inertial-reset) |
| `set_raff_box(sim, min_x, min_y, min_z, max_x, max_y, max_z, k)` | Enable harmonic box constraint (soft AABB confinement, `k` in eV/Å²) |
| `set_raff_nb(sim, enabled, rcut, k_coll, f_max)` | Configure non-bonded (LJ+Coulomb+collision): cutoff, collision stiffness, force clamp |
| `set_raff_charges(sim, [q0, q1, ...])` | Set per-atom Coulomb charges from a Rhai array |

### RAFF relaxation

| Function | Description |
|----------|-------------|
| `raff_step(sim) → float` | One RAFF relaxation step (dispatches to selected solver), returns total energy |
| `raff_relax(sim, max_steps, e_tol) → [energy, steps, converged]` | RAFF relaxation loop with progress printing. Returns array: `[final_energy, n_steps, converged(0/1)]` |
| `get_raff_energy(sim) → float` | Evaluate port + non-bonded energy without stepping (diagnostic) |
| `get_raff_pos(sim) → [x0,y0,z0, ...]` | Flat array of current RAFF positions |
| `save_raff_xyz(sim, path)` | Save current RAFF positions to an XYZ file |

## Example scripts

### UFF relaxation

```rhai
let sim = load_topology("topo.json");
println(`Loaded ${get_natoms(sim)} atoms`);
setup_uff_params(sim, "data/");
let e = eval_forces(sim);
println(`Initial E = ${e}`);
let n = relax(sim, 1000, 0.02, 0.001, 1000.0, 0.1);
println(`Relaxed in ${n} steps`);
```

### RAFF Projective Dynamics with box constraint

```rhai
let sim = load_topology("topo.json");
setup_uff_params(sim, "data/");
setup_raff(sim);
set_raff_solver(sim, "projective");
set_raff_orient(sim, "dynamic");
set_raff_dt(sim, 0.1);
set_raff_iters(sim, 8);
set_raff_hb(sim, 0.5);
set_raff_nb(sim, true, 8.0, 100.0, 50.0);
set_raff_box(sim, -10.0, -10.0, 0.0, 10.0, 10.0, 5.0, 100.0);
let result = raff_relax(sim, 500, 1e-6);
println(`Final E = ${result[0]}, steps = ${result[1]}, converged = ${result[2]}`);
save_raff_xyz(sim, "relaxed.xyz");
```

### RAFF FIRE optimization

```rhai
let sim = load_topology("topo.json");
setup_uff_params(sim, "data/");
setup_raff(sim);
set_raff_solver(sim, "fire");
set_raff_orient(sim, "adiabatic");
set_raff_dt(sim, 0.01);
let result = raff_relax(sim, 1000, 1e-6);
println(`Final E = ${result[0]}, steps = ${result[1]}, converged = ${result[2]}`);
save_raff_xyz(sim, "relaxed.xyz");
```

## Input format

### TopologyData JSON (current)

Loads topology from `TopologyData` JSON (produced by `buildff --json`). Uses `moltopo::export::import_json` which deserializes the flat-array format: `natoms`, `elements`, `positions`, `bonds`, `angles`, `dihedrals`, `inversions`, `bond_params`, `angle_params`, `dihedral_params`, `inversion_params`, `atom_params`.

> **Important:** `load_topology` creates a `Uff` with **zero parameter arrays** — all bond/angle/dihedral/inversion parameters are unset. Before calling `eval_forces` or `relax`, you must call `setup_uff_params(sim, "data/")` to load `.dat` files and fill the parameter arrays. Without this step, all UFF forces and energies will be zero. For RAFF, `setup_uff_params` is also needed — it provides the bond stiffness `k` and rest length `l0` that `setup_raff` uses to build port spring parameters.

### NPZ (planned)

**TODO:** Add `load_topology_from_npz` to load topology from NPZ files (numpy's `.npz` archive format) produced by `buildff --npz` or by Python scripts directly. See [`buildff/README.md`](../buildff/README.md) for the planned format.

## Dependencies

- `surfmol` — `MolWorld` orchestrator (DynamicAtoms + Uff + NonBonded + Surface)
- `molff` — forcefield types (`Uff`, `NonBondedFF`, **`raff`**: `RaffTopology`/`RaffState`/`RaffConfig`/`BoxCfg`/`FireState`/`PosSolver`)
- `moltopo` — `Topology` (used by `load_topology_from_json`; `Params` for `setup_uff_params`)
- `numtypes` — `Vec3d`, `Quat4d` (RAFF state)
- `rhai` — scripting engine
- `clap` — CLI argument parsing

## See also

- [`crates/apps/README.md`](../README.md) — builder/runner architecture overview + Rhai API reference
- [`buildff`](../buildff/README.md) — produces the topology JSON consumed by this engine
- [`/userguide/raff.md`](/userguide/raff.md) — RAFF solver modes end-user guide (GUI + CLI usage, theory)
- [`/userguide/uff_spff.md`](/userguide/uff_spff.md) — UFF/SPFF forcefield setup, parameters, and relaxation guide
- [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) — RAFF cross-implementation map
- `surfmol` — provides `MolWorld` and topology loading functions
