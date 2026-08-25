---
type: rust-crate
title: surfmol-molrender
description: wgpu rendering primitives — molecular meshes, bonds, surfaces, gizmos. No simulation logic.
tags: [rust, crate, rendering, wgpu, gpu]
timestamp: 2026-08-25
---

# surfmol-molrender

wgpu-based rendering primitives for molecules: atom meshes, bond cylinders, surfaces, line gizmos. **No simulation logic lives here.** Consumed by `surfmol-apps` for the GUI.

## Stack

- **wgpu** + **winit** + **egui** — validated by the existing `editor` binary (14 MiB stripped release, 299 deps). See `Import_other_Repos.md` §4 GUI decision. **Do not adopt Bevy.**
- **OpenCL-GL zero-copy interop** for rendering GPU-resident atom arrays directly (from learn_Rust `demo06`).

## What does NOT belong here

- Simulation / forcefields → `surfmol-forcefields`.
- GUI application wiring → `surfmol-apps`.
- Math primitives → `surfmol-common`.

## See also

- `ARCHITECTURE.md` §Component Details.
- `Import_other_Repos.md` §3 (learn_Rust OpenCL-GL interop) and §4 (GUI decision).
