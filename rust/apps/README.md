---
type: rust-crate
title: surfmol-apps
description: High-level GUI applications (editor, mol_browser) and shared GUI utilities. No simulation logic — wires together backend crates.
tags: [rust, crate, gui, apps, egui, wgpu]
timestamp: 2026-08-25
---

# surfmol-apps

High-level composite applications crate. **No simulation logic lives here.** All backends (forcefields, topology, rendering, math) are imported from sibling crates. This crate wires together multiple backend modules into user-facing GUI applications.

## Layout

- `src/` — executable binaries, one per GUI application. Each binary is a thin frontend that wires together backend crates.
- `src/lib.rs` — crate root (module organizer, contains only `pub mod gui;`).
- `src/gui/` — shared GUI utilities reused by multiple frontends (e.g. `MolThumbnailer`, `TrackballCam`).
- `tests/` — integration tests for GUI/composite functionality (tests that require multiple backend crates or GUI context).

## Current binaries

| Binary | Path | Description |
|--------|------|-------------|
| `mol_browser` | `src/mol_browser.rs` | XYZ directory browser with GPU thumbnail grid |
| `editor` | `src/editor.rs` | 3D molecular editor / viewer |

## Planned

1. **MolBrowser** — fast `.xyz`/`.pdb`/`.mol`/`.cif` browser with GPU thumbnail grid.
2. **MolEdit2D** — efficient 2D molecule drawing (ChemSketch-like).
3. **MolWorld App** — 3D environment to move/relax molecules on surfaces (God view + Fly view).

## Test location rule

- Backend module tests (no GUI, single-module) → their crate's `tests/` dir.
- GUI/composite app tests (require GUI or multiple backends) → `apps/tests/`.

## See also

- `ARCHITECTURE.md` §Component Details and §File Naming.
- `DESIGN_GOALS.md` §4 (GUI optimized for forcefield debugging) and §8 (planned applications).
