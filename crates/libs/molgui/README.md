---
type: rust-crate
title: molgui
description: GUI support library — trackball camera (orthographic, column-major projection), PCA-aligned GPU thumbnailer, Kekule hex-grid editor, line gizmos. Reusable across editor and browser apps.
tags: [rust, crate, gui, wgpu, trackball, thumbnail, hex-grid, kekule, orthographic]
timestamp: 2026-09-28
---

# molgui

GUI support library — reusable components shared by `editor` and `molbrowser`. No application logic, no main loop, no window management. Provides camera math, offscreen rendering, hex-grid editing, and line geometry generation.

## Modules

- **`gui/trackball.rs`** — `TrackballCam`: arc-ball camera with smooth quaternion interpolation. `mouse_to_sphere` maps 2D mouse to 3D point on unit sphere (classic trackball mapping: inside unit circle → `z = sqrt(1-r²)`, outside → project to rim with `z=0`). `rotate` uses `Quat::from_rotation_arc(b, a)` — note the (b, a) order rotates from new to old position. Separate `rotation` (current) and `target_rotation` (target) with `slerp` interpolation at `lerp_speed=20.0` for smooth motion. `screen_ray` computes world-space ray from mouse for picking — ray origin is `cam_pos + right*mbx + up*mby`, direction is `fwd` (orthographic, all rays parallel). `camera_data` manually constructs the orthographic projection matrix and returns `CameraData` for molrender. Zoom clamped to [0.2, 200.0]. **Critical**: the `view_proj` matrix is stored in **column-major** order (`[[col0], [col1], [col2], [col3]]`) so that `clip.w = 1.0` for orthographic projection — a transposed (row-major) layout causes `clip.w` to vary with world position, triggering the GPU's perspective divide and producing a "fisheye" distortion. The projection scales by `sx = 1/(zoom*aspect)`, `sy = 1/zoom`, `sz = -1/(far-near)`, with translation in the last row.

- **`gui/thumbnailer.rs`** — `MolThumbnailer`: GPU-accelerated offscreen molecule rendering. `align_to_principal_axes` performs PCA via inertia tensor eigendecomposition (`nalgebra::Matrix3::symmetric_eigen`): smallest eigenvalue → largest spatial extent → X axis, largest eigenvalue → smallest extent → Z (view direction). This ensures consistent thumbnail orientation regardless of input orientation. `render` pipeline: align → build `AtomInstance` array (radii = `r_vdw * 0.3`) → compute bounding box → fit orthographic camera → render impostor spheres → render bond lines → GPU readback via `copy_texture_to_buffer` + `map_async`. Uses `Rgba8UnormSrgb` format, dark background `[0.08, 0.08, 0.12, 1.0]`.

- **`gui/kekule_editor.rs`** — `KekuleEditor`: hex-grid molecular editor for graphene-like structures. `EditMode` enum: `Select`, `HexPaint`, `HexToggle`, `AtomDraw`, `BondDraw`. Hex editing uses axial coordinates (q, r) with pointy-top orientation, delegates to `moltopo::Builder` for topology management. `build_zigzag_ribbon` generates graphene nanoribbons: `dy = a_cc * 1.5`, `dx = a_cc * sqrt(3)`, alternating row offset. `parse_passivation_string` encodes edge functionalization: `n→NH`, `N→N`, `o→C=O`, `O→O`, `H→CH`, `h→C-OH`. `adjust_h_caps` adds H atoms to under-coordinated carbons (radial placement in xy plane — TODO: proper geometry from existing neighbors). `collect_*` functions (`collect_hex_grid_points`, `collect_hex_lines`, `collect_builder_bonds`, `collect_builder_atoms`, `collect_ghost_hexes`) extract visualization data from builder state.

- **`gui/gizmos.rs`** — Pure geometry generators returning `Vec<LineVertex>`: `make_bond_segments` (multi-segment lines between two points), `make_ring` (circle in xy-plane via polar coordinates), `make_axes` (RGB XYZ axes), `make_crosshair` (3D crosshair with ±X/±Y/±Z arms). All pre-allocate with `Vec::with_capacity`.

## Design decisions

- **PCA alignment for thumbnails** — inertia tensor eigendecomposition gives consistent orientation regardless of input coordinate frame. Longest dimension → X (horizontal), shortest → Z (view direction).
- **Arc-ball with smooth interpolation** — separate target/current rotation with slerp avoids jerky camera motion during interaction.
- **Column-major orthographic projection** — the `view_proj` matrix is stored as `[[sx*r.x, sy*u.x, sz*f.x, 0], [sx*r.y, ...], [sx*r.z, ...], [sx*tx, sy*ty, sz*tz+tz, 1]]` where each inner array is a **column** of the matrix. This ensures `clip.w = 1.0` (the last element of the last column) for all world positions, which is required for orthographic projection. Row-major storage would put the translation in `clip.w`, causing perspective divide and fisheye distortion. Fixed 2026-09-28 after debugging "fisheye" distortion in the editor.
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
