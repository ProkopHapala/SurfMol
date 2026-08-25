---
type: folder
title: crates/apps
description: Binary crates — standalone executables that import library crates. Each has a main.rs and produces a CLI/GUI tool.
tags: [rust, workspace, applications, binaries]
timestamp: 2026-08-25
---

# crates/apps

Binary crates — standalone executables that import library crates from `crates/libs/`. Each has a `src/main.rs` and produces a CLI or GUI tool.

## Apps

- **`editor`** — interactive molecular editor and on-surface MD simulator. wgpu + egui + winit. Kekule hex-grid editing, real-time UFF relaxation, NaCl surface potential visualization, atom picking/dragging/pinning.
- **`molbrowser`** — gallery browser for XYZ molecule files. eframe (egui). Batched GPU thumbnail generation with PCA alignment, responsive grid layout, incremental loading.
- **`buildff`** — CLI tool: XYZ → topology → UFF type assignment → JSON or binary export. Consumed by `molengine`.
- **`molengine`** — CLI MD/relaxation engine. Loads topology (JSON now, NPZ planned), runs forcefield evaluation and MD via Rhai scripts.

## Shared dependencies

All apps depend on `moltopo` (topology/XYZ/params) and `numcore` (math). GUI apps (`editor`, `molbrowser`) additionally depend on `molrender` + `molgui`. Simulation apps (`editor`, `molengine`) additionally depend on `surfmol` + `molff` + `surfff`.

## See also

- `crates/libs/README.md` — library crate index
- `ARCHITECTURE.md` — full crate dependency graph
