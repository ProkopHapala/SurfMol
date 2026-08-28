---
type: rust-app
title: editor
description: Interactive molecular editor and on-surface MD simulator — wgpu rendering, egui UI, Kekule hex-grid editing, real-time relaxation with UFF / RigidSp3 / RAFF + nonbonded + NaCl surface, mouse-drag atom picking.
tags: [rust, app, gui, wgpu, egui, editor, md, surface, raff, rigid-atom, port-based]
timestamp: 2026-09-28
---

# editor

Interactive molecular editor and physics simulator for molecules on surfaces. Combines structure editing (Kekule hex-grid), real-time MD relaxation (UFF / RigidSp3 / RAFF + nonbonded + NaCl surface), and visualization (impostor spheres, bond lines, port lines, surface potential texture) in a single wgpu + egui application.

## What it does

- **Load molecules** from XYZ files (CLI arg or default `data/xyz/benzene.xyz`), build topology via `Builder::from_positions_and_radii`, assign UFF types, set up nonbonded FF and NaCl surface.
- **Edit structures** via Kekule hex-grid editor: paint/toggle hex rings, draw atoms, generate graphene nanoribbons with passivation, auto H-cap, then "Bake to Sim" to convert builder state to a simulation world.
- **Relax structures** with real-time MD: three forcefield modes (`F` key cycles Uff → RigidSp3 → Raff). UFF/RigidSp3 uses velocity Verlet with force clamping, damping, spring-based atom dragging, "zero velocity on opposition" heuristic. RAFF uses symplectic Euler with adiabatic rotation solving, port-spring forces, non-bonded LJ+Coulomb, and the same stopping criterion.
- **RAFF mode** (`--raff` flag): starts in simulation mode (not Kekule editor), shows ports, enables non-bonded, disables surface. Builds `RaffTopology` from world topology on init. `do_raff_step()` runs per-frame: eval port forces → eval non-bonded → apply spring drag → 2D constraint → stopping criterion → integrate → adiabatic rotation re-solve → sync positions back to world.
- **2D constraint** (`--2d` flag): flattens all atoms to z=0 plane, zeros z-component of forces/torques, clamps z-position and z-velocity. Only rotation around z-axis allowed. For testing RAFF in a 2D plane.
- **Visualize** atoms as impostor spheres (adjustable size via `--atom-scale` or GUI slider), bonds as multi-segment lines, ports as orange lines from atom center to rotated port tip (synced with RAFF quaternions), surface potential as a colored textured quad (blue-white-red diverging colormap), with axes, crosshairs, and selection rings.
- **Interact** via trackball camera (arc-ball rotation, shift+drag pan, scroll zoom), ray-sphere atom picking (`PICK_RAY_R = 0.5` Å), atom pinning, spring dragging (LMB click to pick, LMB drag to pull).

## Key algorithms

- **Ray-sphere picking** (`ray_sphere`): quadratic discriminant `b² - c` where `b = oc·rd`, `c = |oc|² - sr²`. Returns closest intersection distance.
- **Spring dragging** (`get_force_spring_ray`): projects atom position onto mouse ray, applies perpendicular spring force `-dp_perp * k` to drag atom along ray. Ported from FireCore `MolWorld_sp3.h:1505` (`getForceSpringRay`). Force is perpendicular to the ray (screen-plane constraint), free along depth.
- **MD relaxation loop** (`do_relax_step`): runs `per_frame` iterations, evaluates all forcefields, applies spring force to selected atom, zeros velocities if `Σ dot(v,f) < 0` (energy overshoot / stopping criterion), applies damping, moves unpinned atoms.
- **RAFF relaxation loop** (`do_raff_step`): runs `per_frame` iterations — eval port forces (`eval_port_forces`) → eval non-bonded (`eval_nonbonded`) → apply spring drag → 2D constraint (zero z-force/torque) → stopping criterion (`Σ dot(v,f) < 0` → zero all velocities) → symplectic Euler integration (damped) → adiabatic rotation re-solve (`solve_all_rotations`) → 2D clamp (z=0, z-vel=0) → sync positions to world for rendering.
- **Port rendering sync**: in RAFF mode, ports are drawn from `topo.port_tip(state, i, s)` which applies `state.quat[i]` — the actual rotated port arm from the physics solver. In RigidSp3 mode, ports use fixed geometry from `world.rigid_sp3.get_port_tip()`.
- **Surface potential texture** (`rebuild_surface_cache`): samples NaCl surface on 257×257 grid aligned to lattice parallelogram, evaluates `surf.eval_atom` with unit charge probe, maps to blue-white-red colormap, uploads as wgpu texture.
- **Builder-to-Sim pipeline** ("Bake to Sim"): `cleanup_dead` → `bake` → `MolWorld::from_topology` → rebuild neighbor lists → reassign UFF types → rebuild nonbonded FF with REQ → reload .dat params → rebuild surface.

## CLI arguments

- `--raff` — start in RAFF mode (simulation, not Kekule editor; show ports; enable non-bonded; disable surface; damping=0.1, per_frame=20)
- `--2d` — flatten atoms to z=0 plane, constrain forces/velocities/positions to 2D
- `--atom-scale S` — atom render size multiplier (default 0.25, range 0.05–0.5; also adjustable via GUI slider)
- `--copies-x N`, `--copies-y N`, `--spacing S` — replicate input molecule in x/y
- `--group-size N` — group size for replicated copies
- `--perFrame N` — MD iterations per render frame (default 100, or 20 with `--raff`)
- `--dt T` — MD timestep (default 0.02)
- Positional: path to XYZ file (default `data/xyz/benzene.xyz`)

## Keyboard shortcuts

`H` help · `SPACE` run/stop relaxation · `B` bonds · `S` surface · `G` groups · `T` ports · `K` labels · `D` debug cursor · `P` pin · `C` reset camera · `L` cycle label mode · `E` toggle editor · `F` cycle bonded FF (Uff→RigidSp3→Raff) · `1-4` edit modes · `N` nonbonded · `M` surface · `[` `]` adjust pick_k · `-` `=` adjust per_frame · `ESC` deselect

## GUI panels

- **Settings** (right): Physics (iters/frame, dt, damping, zero-V-on-opposition), Display (atom size slider, label mode, bonded FF mode), RAFF Settings (non-bonded toggle, orient mode, dt/damping/iters, exclusion checkboxes, 2D plane checkbox, live energy display).
- **Inspector** (left): selected atom info (element, UFF type, charge, position, RvdW, pin status).
- **Kekule Editor** (left, edit mode): edit mode selector, ribbon generator, bake/export buttons.
- **Help** (bottom): keyboard shortcuts + CLI flags.

## Caveats (gotchas)

- **Camera orthographic projection**: the `view_proj` matrix in `trackball.rs` must be stored in **column-major** order so that `clip.w = 1.0` for orthographic projection. A transposed (row-major) layout causes `clip.w` to vary with position, triggering the GPU's perspective divide and producing a "fisheye" distortion. Fixed 2026-09-28.
- **Port rendering in RAFF mode**: ports must be drawn from `topo.port_tip(state, i, s)` (applies `state.quat[i]`), NOT from `world.rigid_sp3.get_port_tip()` (fixed geometry). The fixed-geometry path is only correct for `RigidSp3` mode.
- **Stopping criterion**: `zero_v_on_opposition` zeros all velocities when `Σ dot(v_i, f_i) < 0`. Without this, the molecule oscillates forever at zero damping. Applied in both `do_relax_step` (UFF/RigidSp3) and `do_raff_step` (RAFF).
- **RAFF default damping**: `--raff` sets damping=0.1 (cdamp=0.9) and per_frame=20 for stable relaxation. Without damping, the molecule wiggles violently.

## Dependencies

- `molgui` (trackball, kekule editor, gizmos)
- `molrender` (impostor, line, surface renderers)
- `surfmol` (MolWorld orchestrator)
- `molff` (UFF, NonBondedFF, RigidSp3FF, **raff**)
- `surfff` (NaCl surface)
- `moltopo` (Builder, Params, XYZ)
- `numcore` (math)
- `wgpu`, `winit`, `egui`, `glam`

## See also

- [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) — RAFF cross-implementation map
- [`/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md`](/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md) — RAFF phased plan with checkboxes
- `molbrowser` — gallery browser using the same rendering stack
- `molgui` — reusable GUI components (trackball camera, kekule editor)
- `surfmol` — MolWorld orchestration
