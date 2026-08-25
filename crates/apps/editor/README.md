---
type: rust-app
title: editor
description: Interactive molecular editor and on-surface MD simulator — wgpu rendering, egui UI, Kekule hex-grid editing, real-time relaxation with UFF + nonbonded + NaCl surface.
tags: [rust, app, gui, wgpu, egui, editor, md, surface]
timestamp: 2026-08-25
---

# editor

Interactive molecular editor and physics simulator for molecules on surfaces. Combines structure editing (Kekule hex-grid), real-time MD relaxation (UFF + nonbonded + NaCl surface), and visualization (impostor spheres, bond lines, surface potential texture) in a single wgpu + egui application.

## What it does

- **Load molecules** from XYZ files (CLI arg or default `data/xyz/pentacene.xyz`), build topology via `Builder::from_positions_and_radii`, assign UFF types, set up nonbonded FF and NaCl surface.
- **Edit structures** via Kekule hex-grid editor: paint/toggle hex rings, draw atoms, generate graphene nanoribbons with passivation, auto H-cap, then "Bake to Sim" to convert builder state to a simulation world.
- **Relax structures** with real-time MD: velocity Verlet integration with force clamping, damping, spring-based atom dragging, "zero velocity on opposition" heuristic. Runs `per_frame` iterations (default 100) per render frame.
- **Visualize** atoms as impostor spheres, bonds as multi-segment lines, surface potential as a colored textured quad (blue-white-red diverging colormap), with axes, crosshairs, and selection rings.
- **Interact** via trackball camera (arc-ball rotation, shift+drag pan, scroll zoom), ray-sphere atom picking (`PICK_RAY_R = 0.5` Å), atom pinning, spring dragging.

## Key algorithms

- **Ray-sphere picking** (`ray_sphere`): quadratic discriminant `b² - c` where `b = oc·rd`, `c = |oc|² - sr²`. Returns closest intersection distance.
- **Spring dragging** (`get_force_spring_ray`): projects atom position onto mouse ray, applies perpendicular spring force `-dp_perp * k` to drag atom along ray.
- **MD relaxation loop** (`do_relax_step`): runs `per_frame` iterations, evaluates all forcefields, applies spring force to selected atom, zeros velocities if `F·V < 0` (energy overshoot), applies damping, moves unpinned atoms.
- **Surface potential texture** (`rebuild_surface_cache`): samples NaCl surface on 257×257 grid aligned to lattice parallelogram, evaluates `surf.eval_atom` with unit charge probe, maps to blue-white-red colormap, uploads as wgpu texture.
- **Builder-to-Sim pipeline** ("Bake to Sim"): `cleanup_dead` → `bake` → `MolWorld::from_topology` → rebuild neighbor lists → reassign UFF types → rebuild nonbonded FF with REQ → reload .dat params → rebuild surface.

## CLI arguments

- `--copies-x N`, `--copies-y N`, `--spacing S` — replicate input molecule in x/y
- `--group-size N` — group size for replicated copies
- `--perFrame N` — MD iterations per render frame (default 100)
- `--dt T` — MD timestep (default 0.05)
- Positional: path to XYZ file

## Keyboard shortcuts

`H` help · `B` bonds · `S` surface · `G` groups · `T` ports · `K` labels · `D` debug cursor · `P` pin · `C` reset camera · `L` cycle label mode · `E` toggle editor · `F` toggle bonded FF · `1-4` edit modes · `N` nonbonded · `M` surface · `[` `]` adjust pick_k · `-` `=` adjust per_frame

## Dependencies

- `molgui` (trackball, kekule editor, gizmos)
- `molrender` (impostor, line, surface renderers)
- `surfmol` (MolWorld orchestrator)
- `molff` (UFF, NonBondedFF, RigidSp3FF)
- `surfff` (NaCl surface)
- `moltopo` (Builder, Params, XYZ)
- `numcore` (math)
- `wgpu`, `winit`, `egui`, `glam`

## See also

- `molbrowser` — gallery browser using the same rendering stack
- `molgui` — reusable GUI components
- `surfmol` — MolWorld orchestration
