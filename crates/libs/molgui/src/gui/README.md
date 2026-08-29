---
type: folder
title: molgui/src/gui
description: GUI support modules — trackball camera, Kekule hex-grid editor, GPU thumbnailer, bond gizmos, clipboard bridge.
tags: [rust, gui, trackball, kekule, thumbnailer, gizmos, clipboard, egui, wgpu]
timestamp: 2026-08-29
---

# molgui/src/gui — GUI support modules

Reusable GUI components for the `editor` and `molbrowser` apps. No simulation logic — pure presentation and interaction.

## Module files

- **`mod.rs`** — module organizer; declares `pub mod gizmos; kekule_editor; thumbnailer; trackball; clipboard;`
- **`trackball.rs`** — `TrackballCam`: orbit camera (target, rotation Quat, zoom, lerp). Column-major orthographic projection (fixed fisheye bug 2026-09-28)
- **`kekule_editor.rs`** — `KekuleEditor`: hex-grid molecular editor. `EditMode`, `collect_hex_grid_points()`, `collect_builder_bonds/atoms()`, `export_xyz()`, `element_color()`
- **`thumbnailer.rs`** — `MolThumbnailer`: wraps `ImpostorRenderer` + `LineRenderer` for egui thumbnail textures. PCA alignment via `numcore::math::linalg::symmetric_eigen_3x3`
- **`gizmos.rs`** — `make_bond_segments()`: multi-segment bond line generation for rendering
- **`clipboard.rs`** — `Clipboard`: arboard wrapper (text-only, `default-features=false`). `inject_cut_copy_if_needed`, `inject_paste_if_needed`, `handle_output_commands`. Replaces egui-winit's clipboard feature (avoids pulling `image` crate)

## See also

- [`../README.md`](../README.md) — molgui crate overview
- [`../../README.md`](../../README.md) — molgui crate README
- [`/crates/libs/molrender/README.md`](/crates/libs/molrender/README.md) — wgpu rendering primitives (ImpostorRenderer, LineRenderer)
