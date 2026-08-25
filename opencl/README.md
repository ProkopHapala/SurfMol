---
type: opencl-kernels
title: OpenCL Kernels
description: OpenCL .cl kernel sources for GPU acceleration of forcefields, nonbonded interactions, rigid body dynamics, surfaces, and grids.
tags: [opencl, gpu, kernels, forcefield, acceleration]
timestamp: 2026-08-25
---

# OpenCL Kernels

OpenCL `.cl` kernel sources for GPU acceleration. These are the GPU compute kernels; the Rust host-side orchestration lives in the Rust crates (see `Import_other_Repos.md` for the OpenCL crate decision — `ocl` 0.19).

## Current kernels (ported from FireCore / SPAMMM)

| File | Purpose | Origin |
|------|---------|--------|
| `UFF.cl` | UFF force evaluation (bonds, angles, dihedrals, inversions). | FireCore `common_resources/cl/UFF.cl` |
| `relax_multi.cl` | Unified multi-system force evaluation + bucket neighbor search. | FireCore |
| `relax_multi_mini.cl` | Minimal variant of `relax_multi.cl`. | FireCore |
| `Rigid.cl` | Rigid body dynamics kernels. | FireCore / SPAMMM |
| `GridFF.cl` | B-spline grid interpolation for substrate surface potentials. | FireCore / SPAMMM |
| `Surface.cl` | Surface interactions (Morse/LJ/Coulomb), Ewald2D. | FireCore / SPAMMM |
| `Assembly.cl` | Rigid-body molecular assembly / packing / clash evaluation. | SPAMMM |

## Rules

- **CPU Rust references are authoritative** for correctness; GPU must match CPU within tolerance (see `AGENTS.md` Rule 5).
- **NVIDIA GPU preferred** for all timings; never report PoCL/CPU timings as GPU.
- When porting/mirroring a kernel, cite the reference file in a comment.
- Kernels here are the GPU source of truth; the Rust OpenCL crate loads and dispatches them.

## See also

- `Import_other_Repos.md` — which kernels to port and from where.
- `doc/topical_audit/` — per-topic cross-implementation maps (populate as ports progress).
