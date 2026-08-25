---
type: rust-crate
title: surfmol-common
description: Core math and fundamental data structures — Vec3, Quat4, AlignedVec, DynamicAtoms. Bedrock, no chemistry or physics.
tags: [rust, crate, math, data-structures, bedrock]
timestamp: 2026-08-25
---

# surfmol-common

Core math and fundamental data structures. The bedrock of SurfMol — completely agnostic to chemistry or physics specifics. Every other crate depends on this; it depends on nothing domain-specific.

## Responsibilities

- `Vec3` / `Quat4` math (`#[repr(C)]`, 64-byte aligned, inlined ops). Ported from learn_Rust `mol_utils`.
- `AlignedVec` — cache-aligned container for SIMD-friendly forcefield evaluation.
- `DynamicAtoms` — the shared atomic state (positions, velocities, forces) that `MolWorld` borrows but does not own. See `forcefields/DESIGN.md`.

## What does NOT belong here

- Chemistry / physics (bonds, forcefields) → `surfmol-topology`, `surfmol-forcefields`.
- Rendering → `surfmol-molrender`.
- GUI → `surfmol-apps`.

## See also

- `ARCHITECTURE.md` §Component Details.
- `Import_other_Repos.md` §3 (learn_Rust AlignedVec/Vec3d).
