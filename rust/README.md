---
type: rust-workspace
title: Rust Workspace
description: Primary Rust workspace — common, topology, forcefields, molrender, apps crates.
tags: [rust, workspace, primary-code]
timestamp: 2026-08-25
---

# Rust Workspace

Primary Rust workspace. This is where the bulk of SurfMol's logic lives. Build from here with `cargo build` / `cargo test`.

## Crates

| Crate | Path | Role |
|-------|------|------|
| `surfmol-common` | `common/` | Core math, data structures, `DynamicAtoms`, `AlignedVec`. Bedrock, no chemistry/physics. |
| `surfmol-topology` | `topology/` | Molecular graph SSOT (atoms, bonds, angles, atom types). Foundation for forcefield param assignment. |
| `surfmol-forcefields` | `forcefields/` | Forcefield energy/force eval, MD, relaxation, `MolWorld` coordinator. See `forcefields/DESIGN.md`. |
| `surfmol-molrender` | `molrender/` | wgpu rendering primitives (meshes, gizmos, surfaces). No simulation logic. |
| `surfmol-apps` | `apps/` | GUI applications (`editor`, `mol_browser`) + shared GUI utils. No simulation logic. |

## Build & run

```bash
cargo build                              # build all crates
cargo test                               # run all tests
cargo run -p surfmol-apps --bin editor   # launch the 3D editor
cargo run -p surfmol-apps --bin mol_browser  # launch the XYZ browser
```

## Conventions

- **File naming:** see `ARCHITECTURE.md` §File Naming (unique descriptive names, no generic `utils.rs`).
- **Test location:** backend module tests → crate `tests/`; GUI/composite tests → `apps/tests/`.
- **Binary location:** GUI apps → `apps/src/`; CLI tools → their backend crate's `src/bin/`.
- **Crate naming:** `surfmol-*` prefix for crates.io compatibility.

## See also

- `ARCHITECTURE.md` (repo root) — full crate layout and design.
- `DESIGN_GOALS.md` (repo root) — goals and design decisions.
- `forcefields/DESIGN.md` — forcefield data-ownership and `MolWorld` composability.
