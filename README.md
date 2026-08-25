---
type: repository
title: SurfMol
description: Integrated editor and simulator of molecules on surfaces — Rust/OpenCL core with Python bindings.
tags: [molecular-dynamics, opencl, rust, gpu, surface-science, forcefield]
timestamp: 2026-08-25
---

# SurfMol

Integrated editor and simulator of molecules on surfaces, with a Rust/OpenCL core and Python bindings.

## What this is

A compiled, GPU-first platform for on-surface molecular **manipulation** and **global optimization**, and scanning-probe microscopy. Successor to [FireCore](https://github.com/...) (C++) and [SPAMMM](https://github.com/...) (Python+pyOpenCL), rewritten in Rust to eliminate Python overhead and produce a clean, debuggable, GPU-accelerated binary.

**Primary languages:** Rust (logic, GUI, orchestration) + OpenCL (GPU acceleration). Python is a minor layer only (support scripts, quick illustrations).

## Repository layout

| Path | Role | See |
|------|------|-----|
| `rust/` | Primary Rust workspace (common, topology, forcefields, molrender, apps) | `ARCHITECTURE.md` |
| `opencl/` | OpenCL `.cl` kernel sources | `opencl/README.md` |
| `data/` | Molecular input files + FF parameter files | `data/README.md` |
| `userguide/` | End-user docs for finished modules (how to run, GUI controls, theory) | `userguide/README.md` |
| `doc/` | Permanent developer docs + topical audits | `doc/README.md` |
| `notes/` | Temporary work-in-progress (chats, designs, labbooks, reports, tasks, TODOs) | `notes/README.md` |
| `debug/` | Diagnostic plots (gitignored except README) | `debug/README.md` |

## Key documents

| Document | Purpose |
|----------|---------|
| `AGENTS.md` | Binding rules of conduct for agents. Read first. |
| `ARCHITECTURE.md` | Crate layout, file-naming rules, directory structure. |
| `DESIGN_GOALS.md` | Scientific and engineering goals, design decisions. |
| `Import_other_Repos.md` | Reference repos (FireCore, SPAMMM, learn_Rust, blood_of_civilization) and what to import. |
| `notes/ToDo_user.md` | User-facing TODO list and design decisions. |

## Getting started

1. Read `AGENTS.md` (binding rules) and `ARCHITECTURE.md` (structure).
2. Build the Rust workspace: `cargo build` from `rust/`.
3. Run the editor: `cargo run -p surfmol-apps --bin editor` from `rust/`.
4. For what to import from other repos, see `Import_other_Repos.md`.
