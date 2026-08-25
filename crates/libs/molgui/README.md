---
type: rust-crate
title: molgui
description: GUI support library — trackball camera, PCA-aligned GPU thumbnailer, Kekule hex-grid editor, line gizmos. Reusable across editor and browser apps.
tags: [rust, crate, gui, wgpu, trackball, thumbnail, hex-grid, kekule]
timestamp: 2026-08-25
---

# molgui

GUI support library — reusable components shared by `editor` and `molbrowser`. No application logic, no main loop, no window management. Provides camera math, offscreen rendering, hex-grid editing, and line geometry generation.

## Modules

- **`gui/trackball.rs`** — `TrackballCam`: arc-ball camera with smooth quaternion interpolation. `mouse_to_sphere` maps 2D mouse to 3D point on unit sphere (classic trackball mapping: inside unit circle → `z = sqrt(1-r²)`, outside → project to rim with `z=0`). `rotate` uses `Quat::from_rotation_arc(b, a)` — note the (b, a) order rotates from new to old position. Separate `rotation` (current) and `target_rotation` (target) with `slerp` interpolation at `lerp_speed=20.0` for smooth motion. `screen_ray` computes world-space ray from mouse for picking. `camera_data` manually constructs the orthographic projection matrix (same math as thumbnailer) and returns `CameraData` for molrender. Zoom clamped to [0.2, 200.0].

- **`gui/thumbnailer.rs`** — `MolThumbnailer`: GPU-accelerated offscreen molecule rendering. `align_to_principal_axes` performs PCA via inertia tensor eigendecomposition (`nalgebra::Matrix3::symmetric_eigen`): smallest eigenvalue → largest spatial extent → X axis, largest eigenvalue → smallest extent → Z (view direction). This ensures consistent thumbnail orientation regardless of input orientation. `render` pipeline: align → build `AtomInstance` array (radii = `r_vdw * 0.3`) → compute bounding box → fit orthographic camera → render impostor spheres → render bond lines → GPU readback via `copy_texture_to_buffer` + `map_async`. Uses `Rgba8UnormSrgb` format, dark background `[0.08, 0.08, 0.12, 1.0]`.

- **`gui/kekule_editor.rs`** — `KekuleEditor`: hex-grid molecular editor for graphene-like structures. `EditMode` enum: `Select`, `HexPaint`, `HexToggle`, `AtomDraw`, `BondDraw`. Hex editing uses axial coordinates (q, r) with pointy-top orientation, delegates to `moltopo::Builder` for topology management. `build_zigzag_ribbon` generates graphene nanoribbons: `dy = a_cc * 1.5`, `dx = a_cc * sqrt(3)`, alternating row offset. `parse_passivation_string` encodes edge functionalization: `n→NH`, `N→N`, `o→C=O`, `O→O`, `H→CH`, `h→C-OH`. `adjust_h_caps` adds H atoms to under-coordinated carbons (radial placement in xy plane — TODO: proper geometry from existing neighbors). `collect_*` functions (`collect_hex_grid_points`, `collect_hex_lines`, `collect_builder_bonds`, `collect_builder_atoms`, `collect_ghost_hexes`) extract visualization data from builder state.

- **`gui/gizmos.rs`** — Pure geometry generators returning `Vec<LineVertex>`: `make_bond_segments` (multi-segment lines between two points), `make_ring` (circle in xy-plane via polar coordinates), `make_axes` (RGB XYZ axes), `make_crosshair` (3D crosshair with ±X/±Y/±Z arms). All pre-allocate with `Vec::with_capacity`.

## Design decisions

- **PCA alignment for thumbnails** — inertia tensor eigendecomposition gives consistent orientation regardless of input coordinate frame. Longest dimension → X (horizontal), shortest → Z (view direction).
- **Arc-ball with smooth interpolation** — separate target/current rotation with slerp avoids jerky camera motion during interaction.
- **Hex-grid editing delegates to Builder** — `KekuleEditor` holds no topology state; it calls `Builder::add_hex_ring`/`toggle_hex_ring`/`snap_to_node`. This keeps the SSOT in `moltopo`.
- **Visualization separation** — `collect_*` functions extract data from Builder, keeping rendering decisions in the app layer.

## What does NOT belong here

- Application logic (event loop, window, egui) → `crates/apps/`
- Rendering primitives (shaders, pipelines) → `molrender`
- Topology state → `moltopo`

## See also

- `molrender` — `ImpostorRenderer`, `LineRenderer` used by thumbnailer
- `moltopo` — `Builder` for hex-grid topology management
- `editor` — uses `TrackballCam`, `KekuleEditor`, gizmos
- `molbrowser` — uses `MolThumbnailer`
