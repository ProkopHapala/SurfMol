---
type: userguide
title: "RAFF — Rigid-Atom Force Field Solver Modes & Relaxation"
description: End-user guide for RAFF solver modes (ForceMD, InertialReset, FIRE, PBD, XPBD, Projective), orientation strategies (Adiabatic vs Dynamic), harmonic box constraint, CLI flags, GUI controls, and didactic theory of position-based dynamics.
tags: [user-guide, raff, solver, projective-dynamics, xpbd, pbd, fire, relaxation, box-constraint, rigid-atom, port-based]
timestamp: 2026-08-29
---

# RAFF — Rigid-Atom Force Field Solver Modes & Relaxation

RAFF (Rigid-Atom Force Field) is SurfMol's port-based rigid-atom forcefield. Each atom is a rigid body with 1–4 body-frame "ports" — attachment points for bonded neighbors, rotated by the atom's quaternion. A harmonic spring between the rotated port tip and the neighbor position drives both translation and rotation. This replaces explicit angle/dihedral terms with a single port-spring energy, making it naturally suited to rigid-body dynamics and GPU parallelization.

This guide covers the **six solver modes**, **two orientation strategies**, the **harmonic box constraint**, and how to control all of them from the **GUI** and **CLI**.

## Prerequisites

- Rust toolchain (stable, 2021 edition)
- Linux with X11 or Wayland (for wgpu rendering)
- GPU drivers (wgpu supports Vulkan, Metal, DX12 — on Linux typically Vulkan)

## Quick start

```bash
# Default: ForceMD + Adiabatic orientation
cargo run --release -p editor -- --raff

# Projective Dynamics + Dynamic orientation + heavy-ball momentum
cargo run --release -p editor -- --raff --raff-solver projective --raff-orient dynamic --raff-iters 8 --raff-hb 0.5

# FIRE relaxation with box constraint simulating a surface
cargo run --release -p editor -- --raff --raff-solver fire --box --box-min -8,-8,0 --box-max 8,8,5 --box-k 100

# XPBD with 16 inner iterations, 2D plane
cargo run --release -p editor -- --raff --raff-solver xpbd --raff-iters 16 --2d
```

Press `SPACE` to start/stop relaxation. The `RAFF Settings` panel (right side) shows live energy and all controls.

## Solver modes

RAFF supports six solver modes, selectable from the GUI dropdown or the `--raff-solver` CLI flag. They fall into two families: **force-based** (explicit integration of forces) and **position-based** (implicit constraint solving).

### Force-based solvers

| Mode | CLI value | Description |
|------|-----------|-------------|
| **ForceMD** | `forcemd` | Damped symplectic Euler. The simplest integrator: `v += F/m·dt`, `x += v·dt`. Velocity damping `cdamp` after each step. Default mode. |
| **InertialReset** | `inertial` | Full inertia with velocity reset. Same as ForceMD but zeros all velocities when `Σ v·F < 0` (force opposing motion). A simple FIRE variant — converges faster than plain damped MD. |
| **FIRE** | `fire` | Fast Inertial Relaxation Engine (Bitzek 2006). Adaptive timestep + velocity mixing. The standard algorithm used by real optimizers — quasi-Newton behavior near the minimum. Much faster than damped Euler for geometry relaxation. |

### Position-based solvers

| Mode | CLI value | PosSolver | Description |
|------|-----------|-----------|-------------|
| **PBD** | `pbd` | `PbdCompliance` | PBD with compliance: `λ = C/w_total` each iteration (no lagged multiplier). Simplest; can over-correct/oscillate on stiff bonds. |
| **XPBD** | `xpbd` | `Xpbd` | True XPBD (Macklin 2016): lagged `λ_acc` carried between iterations, `dλ = -(C + α̃·λ_acc)/w_total`. Stiffness-independent, no over-correction. |
| **Projective** | `projective` | `Projective` | Projective Dynamics: nonlinear local projection + fixed global quadratic step solved by Jacobi. Best for stiff linear(ized) spring networks. Inner-coupled rotation (both translation and rotation updated in each inner Jacobi iteration). |

### Performance comparison (tree100/D2_stretch benchmark)

| Variant | dt | iters | T2 steps | T1 steps | wall time |
|---------|-----|-------|----------|----------|-----------|
| Adiabatic (best T2) | 0.10 | 4 | 32 | 151 | 23 ms |
| Adiabatic (best T1) | 0.10 | 4 | 38 | 113 | 17 ms |
| Dynamic + Projective i8 + HB | 0.10 | 8 | **30** | **119** | **5.7 ms** |
| Dynamic + Projective i4 | 0.10 | 4 | 34 | 193 | 5.7 ms |
| FIRE | — | — | 299 | 578 | 4.1 ms |

**Key result:** Dynamic Projective with inner-coupled rotation at dt=0.1 converges in 30 steps (T2) at 4× lower wall time than the best adiabatic variant. The inner Jacobi loop acts as a natural substep for both translational and rotational DOFs.

Run the benchmark yourself:
```bash
cargo run --release -p molff --bin raff_bench
```

## Orientation strategies

Each atom has a quaternion `q_i` that rotates its body-frame ports into world space. Two strategies for updating `q_i`:

| Strategy | CLI value | Description |
|----------|-----------|-------------|
| **Adiabatic** | `adiabatic` | Memoryless: re-solve `q_i` every step via Wahba/Horn quaternion eigen decomposition of the port covariance matrix. No `omega`, no torque integration. More stable, but does not conserve angular momentum. **Default.** |
| **Dynamic** | `dynamic` | Physical inertia: integrate `omega` from torque, update `q` via `q ← exp(ω·dt/2) ⊗ q`. Conserves angular momentum. For Projective mode, rotation is coupled into the inner Jacobi loop (both `x` and `q` updated each inner iteration). |

**Recommendation:** Use **Adiabatic** for simple relaxation (more stable, no tuning needed). Use **Dynamic** with **Projective** for fast convergence on stiff systems — the inner-coupled rotation is the breakthrough that made dt=0.1 stable.

## Position-based sub-options

When a position-based solver (PBD / XPBD / Projective) is selected, additional options appear in the GUI:

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| **Inner iters** | `--raff-iters N` | 4 | Number of inner Jacobi/XPBD iterations per outer step. More = better convergence per step but slower per step. 4–8 is typical; 8+ with heavy-ball for stiff systems. |
| **PD inertia** | `--raff-pd-inertia` / `--raff-no-pd-inertia` | enabled | Outer-loop inertia: predict `x += v·dt` before the inner solve. True = proper Projective Dynamics (carries momentum between outer steps). False = legacy projection-only mode. |
| **Vel reset** | `--raff-vel-reset` / `--raff-no-vel-reset` | enabled | Reset velocity to zero when `v·F < 0` (uphill). Like FIRE/inertial-reset. Prevents energy buildup with full inertia (`cdamp=1`). |
| **Heavy-ball momentum** | `--raff-hb M` | 0.0 (off) | Momentum mixing for the inner Jacobi solver. Ramps from 0 to `M` over iterations `bmix_istart`–`bmix_iend`. 0 = disabled. 0.5–0.75 is typical for stiff systems. Accelerates inner convergence 2–3×. |
| **HB start iter** | (GUI only) | 3 | Iteration to start ramping momentum. |
| **HB end iter** | (GUI only) | 10 | Iteration to end ramping (after this, full `M`). |

## Harmonic box constraint

The box constraint confines atoms within an axis-aligned bounding box (AABB) with a harmonic restoring force. This simulates a soft surface boundary — atoms are free inside the box but pushed back when they cross the boundary.

### Physics

For each atom at position `p` outside the box `[min, max]`:

```
F = k · (limit - p)     on each violated axis
E = ½ · k · δ²          per violated axis
```

where `limit` is the nearest box face and `δ = |p - limit|` is the penetration depth. Atoms inside the box feel no force.

- **Force-based solvers** (ForceMD, InertialReset, FIRE): box force added to the force array before integration — physically correct.
- **Position-based solvers** (PBD, XPBD, Projective): box applied as a post-solve explicit position correction `δx = k·(limit-x)·dt²/m` after the inner Jacobi loop — equivalent to one explicit Euler step of the harmonic potential.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--box` | off | Enable harmonic box constraint |
| `--box-min x,y,z` | `-10,-10,0` | Min corner of the free region (comma-separated) |
| `--box-max x,y,z` | `10,10,10` | Max corner of the free region (comma-separated) |
| `--box-k K` | `50.0` | Spring constant in eV/Å² |

### GUI controls

In the `RAFF Settings` panel, under **Box Constraint**:
- `Harmonic box (AABB)` checkbox — enable/disable
- `k_box (eV/Å²)` slider (1–500)
- Min corner: x/y/z drag values
- Max corner: x/y/z drag values

### Example: simulating a surface

```bash
# Atoms confined to z ∈ [0, 5], free in x,y ∈ [-8, 8], stiff spring
cargo run --release -p editor -- --raff --raff-solver projective --raff-orient dynamic \
  --box --box-min -8,-8,0 --box-max 8,8,5 --box-k 100
```

The z=0 plane acts as the "surface" — atoms falling below z=0 are pushed back up. The z=5 ceiling prevents atoms from drifting away. Adjust `--box-k` to control stiffness: higher = harder wall, lower = softer confinement.

## CLI reference (RAFF-specific)

All RAFF flags require `--raff` to enter RAFF mode.

| Flag | Default | Description |
|------|---------|-------------|
| `--raff` | off | Start in RAFF mode (simulation, show ports, enable non-bonded, disable surface) |
| `--raff-solver MODE` | `forcemd` | Solver: `forcemd` / `inertial` / `fire` / `pbd` / `xpbd` / `projective` |
| `--raff-orient MODE` | `adiabatic` | Orientation: `adiabatic` / `dynamic` |
| `--raff-iters N` | 4 | Inner iterations for position-based solvers (1–32) |
| `--raff-pd-inertia` | on | Enable PD outer-loop inertia (predict `x += v·dt`) |
| `--raff-no-pd-inertia` | off | Disable PD outer-loop inertia (legacy projection-only) |
| `--raff-vel-reset` | on | Enable velocity reset when `v·F < 0` |
| `--raff-no-vel-reset` | off | Disable velocity reset |
| `--raff-hb M` | 0.0 | Heavy-ball momentum (0 = disabled, 0.5–0.75 typical) |
| `--box` | off | Enable harmonic box constraint |
| `--box-min x,y,z` | `-10,-10,0` | Box min corner |
| `--box-max x,y,z` | `10,10,10` | Box max corner |
| `--box-k K` | 50.0 | Box spring constant (eV/Å²) |
| `--2d` | off | Constrain atoms to z=0 plane (2D simulation) |
| `--show-aabb` | off | Visualize broad-phase AABBs (green = tight, red = expanded by rcut) |

## GUI reference (RAFF Settings panel)

The `RAFF Settings` panel appears on the right side when RAFF mode is active (`--raff` or cycle FF with `F` key).

### Solver section
- **Mode** dropdown: ForceMD / InertialReset / FIRE / PBD / XPBD / Projective
- **Orient** dropdown: Adiabatic / Dynamic
- **Non-bonded (LJ+Coul+Coll)** checkbox
- **2D plane (z=0 constraint)** checkbox

### PD Options section (position-based solvers only)
- **inner iters** slider (1–32)
- **PD inertia (outer loop)** checkbox
- **vel reset (v·F<0 → v=0)** checkbox
- **heavy-ball momentum** slider (0.0–0.95)
- **HB start iter** / **HB end iter** sliders (visible when HB > 0)

### Non-bonded Params section
- **rcut (Å)** slider (3–20)
- **k_coll** slider (0–500)
- **f_max** slider (0–200)
- **Exclude 1-2** / **Exclude 1-3** checkboxes

### Box Constraint section
- **Harmonic box (AABB)** checkbox
- **k_box (eV/Å²)** slider (1–500)
- **Min corner** x/y/z drag values
- **Max corner** x/y/z drag values

### Energy display
- `E_port` — port spring energy (eV)
- `E_nb` — non-bonded energy (eV)
- `E_tot` — total energy (eV)

## CLI usage (molengine)

The CLI runner `molengine` supports the same RAFF solver modes as the GUI editor, controlled via Rhai scripts. The workflow is a two-step pipeline:

### Step 1: Build topology (buildff)

```bash
cargo run -p buildff -- data/xyz/CH4.xyz --json ch4_topo.json
```

This produces a `TopologyData` JSON file with bonds, angles, and UFF type assignments. No forcefield parameters — those are loaded at runtime.

### Step 2: Run RAFF relaxation (molengine)

Write a Rhai script (e.g. `relax.rhai`) and run it:

```bash
cargo run -p molengine -- --script relax.rhai
```

### Rhai API reference

| Function | Description |
|----------|-------------|
| `load_topology(path) → sim` | Load `TopologyData` JSON from `buildff` |
| `setup_uff_params(sim, data_dir)` | Load `.dat` files and fill UFF parameter arrays (**required** before `eval_forces`/`setup_raff`) |
| `setup_raff(sim)` | Build `RaffTopology` + `RaffState` from UFF bonds + positions (ARAP port geometry) |
| `set_raff_solver(sim, mode)` | `"forcemd"` / `"inertial"` / `"fire"` / `"pbd"` / `"xpbd"` / `"projective"` |
| `set_raff_orient(sim, mode)` | `"adiabatic"` / `"dynamic"` |
| `set_raff_dt(sim, dt)` | Timestep |
| `set_raff_damping(sim, damping)` | Damping factor (0=kill, 1=no damping) |
| `set_raff_iters(sim, iters)` | Inner iterations for position-based solvers |
| `set_raff_hb(sim, momentum)` | Heavy-ball momentum (0=disabled) |
| `set_raff_pd_inertia(sim, bool)` | PD outer-loop inertia |
| `set_raff_vel_reset(sim, bool)` | Velocity reset on v·F<0 |
| `set_raff_box(sim, min_x, min_y, min_z, max_x, max_y, max_z, k)` | Harmonic box constraint |
| `set_raff_nb(sim, enabled, rcut, k_coll, f_max)` | Non-bonded (LJ+Coulomb+collision) |
| `set_raff_charges(sim, [q0, q1, ...])` | Per-atom Coulomb charges |
| `raff_step(sim) → float` | One RAFF step (returns energy) |
| `raff_relax(sim, max_steps, e_tol) → [energy, steps, converged]` | Relaxation loop |
| `get_raff_energy(sim) → float` | Energy without stepping |
| `get_raff_pos(sim) → [x0,y0,z0,...]` | Current positions |
| `save_raff_xyz(sim, path)` | Save positions to XYZ |
| `get_natoms(sim) → int` | Number of atoms |

### CLI Example 1: Projective Dynamics with box constraint

```rhai
// relax_projective.rhai
let sim = load_topology("ch4_topo.json");
println(`Loaded ${get_natoms(sim)} atoms`);
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
save_raff_xyz(sim, "ch4_relaxed.xyz");
```

```bash
cargo run -p buildff -- data/xyz/CH4.xyz --json ch4_topo.json
cargo run -p molengine -- --script relax_projective.rhai
```

### CLI Example 2: FIRE optimization

```rhai
// relax_fire.rhai
let sim = load_topology("ch4_topo.json");
setup_uff_params(sim, "data/");
setup_raff(sim);
set_raff_solver(sim, "fire");
set_raff_orient(sim, "adiabatic");
set_raff_dt(sim, 0.01);
let result = raff_relax(sim, 1000, 1e-6);
println(`Final E = ${result[0]}, steps = ${result[1]}, converged = ${result[2]}`);
save_raff_xyz(sim, "ch4_fire.xyz");
```

### CLI Example 3: UFF relaxation (no RAFF)

```rhai
// relax_uff.rhai
let sim = load_topology("ch4_topo.json");
setup_uff_params(sim, "data/");
let e = eval_forces(sim);
println(`Initial E = ${e}`);
let n = relax(sim, 1000, 0.02, 0.001, 1000.0, 0.1);
println(`Relaxed in ${n} steps`);
```

See [`/crates/apps/molengine/README.md`](/crates/apps/molengine/README.md) for the full Rhai API and more examples.

## Usage examples (GUI editor)

### Example 1: Fast relaxation with Projective Dynamics

```bash
cargo run --release -p editor -- --raff --raff-solver projective --raff-orient dynamic \
  --raff-iters 8 --raff-hb 0.5 --dt 0.1
```

The fastest configuration for geometry relaxation: 8 inner Jacobi iterations with heavy-ball momentum (0.5), dynamic orientation, and a large timestep (dt=0.1). Converges in ~30 steps for typical molecules.

### Example 2: Stable relaxation with Adiabatic + ForceMD

```bash
cargo run --release -p editor -- --raff --raff-solver forcemd --raff-orient adiabatic --dt 0.05
```

The most stable configuration: adiabatic rotation (memoryless re-solve) with damped force-MD. Slower than Projective but never diverges. Good for initial testing.

### Example 3: FIRE optimization

```bash
cargo run --release -p editor -- --raff --raff-solver fire
```

FIRE adapts its timestep automatically — no need to tune `dt`. Good for reaching the exact minimum (quasi-Newton behavior near convergence). Slower than Projective in the initial phase but very precise at the end.

### Example 4: Surface simulation with box constraint

```bash
cargo run --release -p editor -- --raff --raff-solver projective --raff-orient dynamic \
  --box --box-min -10,-10,0 --box-max 10,10,8 --box-k 200 --nmols 4 --layout random --show-aabb
```

4 molecules randomly placed, confined to a box (z=0 surface, z=8 ceiling, ±10 Å in x/y), with AABB visualization. The box simulates a surface boundary — molecules settle on the z=0 "surface" and interact via non-bonded forces.

### Example 5: 2D collision with XPBD

```bash
cargo run --release -p editor -- --raff --raff-solver xpbd --raff-iters 16 --2d \
  --nmols 3 --layout lattice --show-aabb
```

3 molecules in 2D (z=0 plane) with XPBD constraints (16 inner iterations). The 2D constraint is a hard plane — not a soft potential like the box. Useful for testing collision dynamics in a controlled 2D setting.

## Theory: What is a port-spring forcefield?

In a traditional forcefield (UFF), the molecular energy is a sum of bonded terms:

```
E = Σ_bonds ½k(r-r₀)² + Σ_angles ½k(θ-θ₀)² + Σ_dihedrals V_n(1+cos(nφ-δ)) + Σ_inversions ½k(ψ-ψ₀)²
```

Each term requires a separate evaluation path (distance, angle, dihedral, inversion) with different formulas and parameter lookups.

**RAFF replaces all of this with a single port-spring energy.** Each atom has 1–4 "ports" — body-frame attachment points for its bonded neighbors. The port tip position is:

```
tip_i = x_i + R(q_i) · (l0 · a_α)
```

where `R(q_i)` is the rotation matrix from the atom's quaternion, `l0` is the equilibrium bond length, and `a_α` is the port direction in the body frame. A harmonic spring connects the port tip to the neighbor:

```
E_port = ½ k_p |x_j - tip_i|²
F = k_p · (x_j - tip_i)
τ = r_arm × F        (torque on atom i)
```

where `r_arm = R(q_i) · (l0 · a_α)` is the port arm vector. The port tip encodes both bond length and angle geometry via the quaternion rotation — so a single spring per port replaces the 3-term (angle+dihedral+inversion) UFF machinery.

**Per-port stiffness:** `k_p = K_bond / 2` because each bond contributes two ports (one per atom). The total stiffness of the bond = `k_p_i + k_p_j = K_bond`.

## Theory: What is position-based dynamics?

**Force-based MD** (ForceMD, InertialReset, FIRE) integrates Newton's equations explicitly:

```
v_new = v + F/m · dt
x_new = x + v_new · dt
```

This is simple but has a fundamental limitation: the timestep `dt` must be small enough that the fastest vibration (stiffest bond) is resolved. For a stiff spring with `k=500 eV/Å²` and mass `m=1`, the vibrational period is `T = 2π√(m/k) ≈ 0.28`, so `dt < 0.01` is needed for stability.

**Position-based dynamics** (PBD, XPBD, Projective) reformulates the problem as a constraint projection:

1. **Predict**: `x_pred = x + v·dt` (ignoring forces)
2. **Solve constraints**: iteratively adjust `x_pred` to satisfy all bond constraints (target lengths)
3. **Correct**: `v = (x_new - x_old) / dt`

The constraint solve is implicit — it finds positions that satisfy all constraints simultaneously, regardless of stiffness. This allows much larger timesteps (dt=0.1 vs dt=0.01 for force-based).

### PBD vs XPBD vs Projective

- **PBD (with compliance)**: each iteration solves `λ = C/w_total` independently. Simple but can over-correct on stiff bonds — the correction from one bond can violate another, causing oscillation.

- **XPBD (Macklin 2016)**: carries a lagged multiplier `λ_acc` between iterations: `dλ = -(C + α̃·λ_acc)/w_total`. The accumulated correction prevents over-correction — stiffness-independent convergence. The gold standard for constraint-based simulation.

- **Projective Dynamics**: splits the problem into a **local projection** (each spring projects to its rest length independently) and a **global solve** (find positions that best satisfy all projections simultaneously, via Jacobi iteration on a linearized system). The global solve is a fixed quadratic problem — the system matrix doesn't change between iterations (only the RHS does), so it can be precomputed. Best for stiff linear(ized) spring networks.

### Inner-coupled rotation

The key innovation in SurfMol's Projective Dynamics: instead of solving rotation in an outer loop (which limits stability), the inner Jacobi loop updates **both** position and quaternion in each iteration. One port traversal accumulates both translational RHS and torque; updating both `x` and `q` is cheap. This makes the inner Jacobi loop the natural substep for both DOFs — no separate rotational subcycling needed.

## Theory: What is FIRE?

**FIRE (Fast Inertial Relaxation Engine)** is a momentum-based relaxation algorithm (Bitzek et al. 2006) that adapts the timestep and mixes velocity with force direction:

```
v_new = (1-α)·v + α·|v|·F̂
x_new = x + v_new·dt
```

where `α` is a mixing parameter that starts at 0.1 and decreases toward 0 (more velocity, less force). When `v·F < 0` (moving uphill), velocity is reset and `dt` is reduced. When `v·F > 0` for consecutive steps, `dt` is increased (up to `dt_max`).

This gives FIRE a **quasi-Newton** behavior near the minimum — it follows the force direction but uses momentum to accelerate through flat regions and avoid zig-zagging. It's the standard algorithm used by real optimizers (ASE, VASP, LAMMPS).

## See also

- [`editor.md`](editor.md) — editor end-user guide (all CLI flags, GUI controls, keyboard shortcuts)
- [`/crates/apps/README.md`](/crates/apps/README.md) — builder/runner architecture overview + Rhai API reference
- [`/crates/apps/molengine/README.md`](/crates/apps/molengine/README.md) — CLI runner (Rhai-scripted MD/relaxation, full Rhai API)
- [`/crates/apps/buildff/README.md`](/crates/apps/buildff/README.md) — CLI builder (XYZ → topology JSON)
- [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) — RAFF cross-implementation map (SurfMol vs FireCore vs SPAMMM)
- [`/notes/designs/raff_theory_equations.md`](/notes/designs/raff_theory_equations.md) — all RAFF mathematical formulations
- [`/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md`](/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md) — RAFF phased implementation plan
- [`/notes/reports/2026-08-28_raff_solver_benchmark_report.md`](/notes/reports/2026-08-28_raff_solver_benchmark_report.md) — benchmark report with performance tables
- [`/notes/labbooks/2026-08-28_raff_solver_optimization.md`](/notes/labbooks/2026-08-28_raff_solver_optimization.md) — labbook of solver optimization sessions
- [`/crates/libs/molff/README.md`](/crates/libs/molff/README.md) — molff crate overview (raff.rs module)
- [`/crates/apps/editor/README.md`](/crates/apps/editor/README.md) — editor app README (RAFF integration)
- [`/CODEMAP.md`](/CODEMAP.md) — repo structure and crate dependency graph
