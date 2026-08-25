---
type: rust-app
title: molbrowser
description: Gallery browser for XYZ molecule files — batched GPU thumbnail generation with PCA alignment, responsive grid layout, incremental loading.
tags: [rust, app, gui, eframe, browser, thumbnails]
timestamp: 2026-08-25
---

# molbrowser

Gallery-style browser for molecule XYZ files. Scans a folder for `.xyz` files, generates GPU-accelerated thumbnails with PCA-based orientation alignment, and displays them in a responsive grid.

## What it does

- **Scans folder** for `.xyz` files, sorted alphabetically
- **Generates thumbnails** via `MolThumbnailer` (from `molgui`): PCA alignment → impostor sphere rendering → bond lines → GPU readback → egui texture
- **Batched rendering**: 4 thumbnails per frame to maintain UI responsiveness (incremental `next_to_render` index)
- **Responsive grid**: column count computed from window width: `cols = floor((available_w + spacing) / (thumb_w + spacing))`
- **Bond detection**: covalent radii sum + 0.4 Å tolerance, at most 4 closest neighbors per atom (prevents over-bonding in dense systems)

## CLI

```
molbrowser [folder]
```

Defaults to `data/xyz` relative to executable or workspace root. Thumbnail size: 128px.

## Key algorithms

- **Covalent radii bond detection**: for each atom i, find neighbors j where `|r_i - r_j| < radii[i] + radii[j] + 0.4` Å, sort by distance, keep 4 closest. Radii from `Params::get_element_type(el).r_cov`.
- **Incremental rendering**: `next_to_render` index advances by 1 per `render_next` call; `update` calls `render_next` up to 4 times per frame, requests repaint after 16ms (~60 FPS) until all thumbnails done.

## Dependencies

- `molgui` (`MolThumbnailer`)
- `moltopo` (`read_xyz`, `Params`)
- `eframe` (egui framework)
- `numcore` (Vec3d)

## See also

- `molgui` — `MolThumbnailer` with PCA alignment
- `editor` — full editor using the same rendering stack
