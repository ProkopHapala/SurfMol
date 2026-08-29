---
type: folder
title: crates/apps
description: Binary crates — standalone executables that import library crates. Each has a main.rs and produces a CLI/GUI tool. Two CLI tools form a Unix-style builder/runner pipeline; the editor is the interactive GUI.
tags: [rust, workspace, applications, binaries, cli, gui, builder, runner]
timestamp: 2026-08-29
---

# crates/apps

Binary crates — standalone executables that import library crates from `crates/libs/`. Each has a `src/main.rs` and produces a CLI or GUI tool.

## Architecture: builder / runner / editor

SurfMol's app layer follows a **Unix-style separation of concerns**: two lightweight, independent CLI programs (builder + runner) that share a common data format, plus one interactive GUI editor that integrates everything.

```
         XYZ file                TopologyData JSON              Relaxed XYZ
         (atoms)        buildff  (bonds, angles, params)  molengine  (positions)
  data/xyz/*.xyz  ──────────►  topo.json  ──────────────────────────►  out.xyz
                         (builder)                    (runner, Rhai-scripted)
                              │                                           │
                              │         editor (GUI: all-in-one)          │
                              └──────────►  wgpu + egui  ◄────────────────┘
                                   (interactive editing + relaxation)
```

### Builder (`buildff`)

**Input:** XYZ file (atom positions + elements).
**Output:** `TopologyData` JSON (canonical format via `moltopo::export::TopologyData`).

The builder is a one-shot CLI that:
1. Reads an XYZ file
2. Builds molecular topology (bonds, angles, dihedrals, inversions) via covalent-radius heuristics
3. Assigns UFF atom types from hybridization
4. Exports the result as JSON (consumed by the runner) or binary (for direct ingestion)

```bash
cargo run -p buildff -- data/xyz/CH4.xyz --json topo.json
```

The builder has **no forcefield evaluation** — it only prepares the topology. It is stateless and fast.

### Runner (`molengine`)

**Input:** `TopologyData` JSON (from `buildff`) + Rhai script.
**Output:** Relaxed structures, energies, trajectories (via script-controlled I/O).

The runner is a **Rhai-scripted** simulation engine. The script controls everything: which forcefield to use (UFF / RAFF), which solver mode (ForceMD / InertialReset / FIRE / PBD / XPBD / Projective), what parameters to set, how many steps to run, and what to save. The runner exposes a set of registered Rhai functions (see below).

```bash
cargo run -p molengine -- --script relax.rhai
```

The runner has **no GUI** — it is a pure compute engine. It can be used in batch mode, in pipelines, or on a headless server.

### Editor (`editor`)

The editor is the **interactive GUI** that integrates everything: structure editing (Kekule hex-grid), real-time relaxation (UFF / RigidSp3 / RAFF with all 6 solver modes), visualization (impostor spheres, bonds, ports, surface potential, AABB collision boxes), and atom picking/dragging/pinning. It is a wgpu + egui + winit application.

```bash
cargo run -p editor -- --raff --raff-solver projective --box --box-min -10,-10,0 --box-max 10,10,5
```

The editor is for interactive exploration and visualization. For batch processing, use the builder + runner pipeline.

### Data format: `TopologyData`

The builder and runner share a common JSON format: `moltopo::export::TopologyData`. This is a flat-array format (no nested per-atom objects) designed for zero-copy ingestion:

```json
{
  "natoms": 5,
  "elements": ["C", "H", "H", "H", "H"],
  "positions": [[0,0,0], [0.629,0.629,0.629], ...],
  "bonds": [[0,1], [0,2], [0,3], [0,4]],
  "angles": [[1,0,2], [1,0,3], ...],
  "dihedrals": [],
  "inversions": [],
  "bond_params": [],
  "angle_params": [],
  "dihedral_params": [],
  "inversion_params": [],
  "atom_params": []
}
```

The `*_params` arrays are optional (can be empty) — the runner fills them at runtime via `setup_uff_params` (loads `.dat` files from `data/`).

**NPZ format** (planned): will replace JSON for large systems. See `buildff/README.md`.

## Apps

- **`buildff`** — CLI builder: XYZ → topology → UFF type assignment → `TopologyData` JSON or binary export. Stateless, no forcefield evaluation. Consumed by `molengine`. See [`buildff/README.md`](buildff/README.md).

- **`molengine`** — CLI runner: Rhai-scripted MD/relaxation engine. Loads `TopologyData` JSON, runs UFF and/or RAFF forcefield evaluation and relaxation. Supports all 6 RAFF solver modes (ForceMD, InertialReset, FIRE, PBD, XPBD, Projective), harmonic box constraint, non-bonded LJ+Coulomb, and per-atom charges. See [`molengine/README.md`](molengine/README.md) and [`/userguide/raff.md`](/userguide/raff.md) §CLI usage.

- **`editor`** — interactive molecular editor and on-surface MD simulator. wgpu + egui + winit. Kekule hex-grid editing, real-time relaxation with 6 RAFF solver modes (ForceMD, InertialReset, FIRE, PBD, XPBD, Projective) + UFF/RigidSp3, NaCl surface potential, harmonic box constraint, atom picking/dragging/pinning. See [`/userguide/editor.md`](/userguide/editor.md) and [`/userguide/raff.md`](/userguide/raff.md).

- **`molbrowser`** — gallery browser for XYZ molecule files. eframe (egui). Batched GPU thumbnail generation with PCA alignment, responsive grid layout, incremental loading.

## Rhai API (molengine)

The runner exposes these functions to Rhai scripts:

### Topology & params
| Function | Description |
|----------|-------------|
| `load_topology(path) → sim` | Load `TopologyData` JSON, returns `SimulationEngine` |
| `setup_uff_params(sim, data_dir)` | Load `.dat` files and fill UFF parameter arrays (bonds, angles, dihedrals, inversions) |
| `get_natoms(sim) → int` | Number of atoms |

### UFF relaxation
| Function | Description |
|----------|-------------|
| `eval_forces(sim) → float` | Evaluate UFF forces, return total energy |
| `step_md(sim, dt, flim, damping)` | One UFF MD step (velocity Verlet) |
| `relax(sim, niter, dt, fconv, flim, damping) → int` | UFF relaxation loop, returns n_steps |

### RAFF setup
| Function | Description |
|----------|-------------|
| `setup_raff(sim)` | Build `RaffTopology` + `RaffState` from UFF bonds + positions (per-atom ARAP port geometry) |
| `set_raff_solver(sim, mode)` | Select solver: `"forcemd"` / `"inertial"` / `"fire"` / `"pbd"` / `"xpbd"` / `"projective"` |
| `set_raff_orient(sim, mode)` | Orientation: `"adiabatic"` / `"dynamic"` |
| `set_raff_dt(sim, dt)` | Timestep |
| `set_raff_damping(sim, damping)` | Damping factor (0=kill, 1=no damping) |
| `set_raff_iters(sim, iters)` | Inner iterations for position-based solvers |
| `set_raff_hb(sim, momentum)` | Heavy-ball momentum (0=disabled, 0.5–0.75 typical) |
| `set_raff_pd_inertia(sim, bool)` | Enable/disable PD outer-loop inertia |
| `set_raff_vel_reset(sim, bool)` | Enable/disable velocity reset on v·F<0 |
| `set_raff_box(sim, min_x, min_y, min_z, max_x, max_y, max_z, k)` | Enable harmonic box constraint |
| `set_raff_nb(sim, enabled, rcut, k_coll, f_max)` | Configure non-bonded (LJ+Coulomb+collision) |
| `set_raff_charges(sim, [q0, q1, ...])` | Set per-atom Coulomb charges |

### RAFF relaxation
| Function | Description |
|----------|-------------|
| `raff_step(sim) → float` | One RAFF step (returns total energy) |
| `raff_relax(sim, max_steps, e_tol) → [energy, steps, converged]` | RAFF relaxation loop |
| `get_raff_energy(sim) → float` | Evaluate port + non-bonded energy without stepping |
| `get_raff_pos(sim) → [x0,y0,z0, ...]` | Flat array of current positions |
| `save_raff_xyz(sim, path)` | Save current RAFF positions to XYZ file |

### Example Rhai script

```rhai
// Load topology + setup UFF params
let sim = load_topology("topo.json");
println(`Loaded ${get_natoms(sim)} atoms`);
setup_uff_params(sim, "data/");

// Build RAFF + configure Projective Dynamics
setup_raff(sim);
set_raff_solver(sim, "projective");
set_raff_orient(sim, "dynamic");
set_raff_dt(sim, 0.1);
set_raff_iters(sim, 8);
set_raff_hb(sim, 0.5);

// Non-bonded + box constraint (surface simulation)
set_raff_nb(sim, true, 8.0, 100.0, 50.0);
set_raff_box(sim, -10.0, -10.0, 0.0, 10.0, 10.0, 5.0, 100.0);

// Relax
let e0 = get_raff_energy(sim);
println(`Initial E = ${e0}`);
let result = raff_relax(sim, 500, 1e-6);
println(`Final E = ${result[0]}, steps = ${result[1]}, converged = ${result[2]}`);

// Save
save_raff_xyz(sim, "relaxed.xyz");
```

## Shared dependencies

All apps depend on `moltopo` (topology/XYZ/params) and `numcore` (math). GUI apps (`editor`, `molbrowser`) additionally depend on `molrender` + `molgui`. Simulation apps (`editor`, `molengine`) additionally depend on `surfmol` + `molff` + `surfff`.

## See also

- `crates/libs/README.md` — library crate index
- `ARCHITECTURE.md` — full crate dependency graph
- [`buildff/README.md`](buildff/README.md) — builder CLI (XYZ → topology JSON)
- [`molengine/README.md`](molengine/README.md) — runner CLI (Rhai-scripted MD/relaxation)
- [`/userguide/editor.md`](/userguide/editor.md) — editor end-user guide
- [`/userguide/raff.md`](/userguide/raff.md) — RAFF solver modes end-user guide (GUI + CLI)
