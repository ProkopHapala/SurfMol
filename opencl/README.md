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

|| File | Purpose | Origin |
||------|---------|--------|
|| `UFF.cl` | UFF force evaluation (bonds, angles, dihedrals, inversions) + MMFFsp3 integrator + pi-orbital nonbonded. **Self-contained** (defines own `cl_Mat3`/`R2safe`/`EXCL_MAX`). | FireCore `common_resources/cl/UFF.cl` |
|| `SPFF.cl` | SPFFsp3 force field = FireCore **MMFFsp3** GPU forcefield (bonds, angles, pi-pi, pi-sigma, MD integrator, recoil gather). **Requires** `common.cl`+`Forces.cl` concatenated first. | SPAMMM `kernels/SPFF.cl` |
|| `common.cl` | Shared types/macros/helpers (`cl_Mat3`/`R2safe`/`EXCL_MAX`, `mixREQ_arithmetic`, `clampForce`, `udiv_cmplx`). Concatenated FIRST before SPAMMM modular kernels. | SPAMMM `kernels/common.cl` |
|| `Forces.cl` | Inline pairwise potentials (`getLJQH`, `getMorseQH`, `getCoulomb`). Not `__kernel`; concatenated after `common.cl`. | SPAMMM `kernels/Forces.cl` |
|| `relax_multi.cl` | Unified multi-system force evaluation + bucket neighbor search (FireCore `getMMFFf4` = MMFFsp3 force eval). | FireCore |
|| `relax_multi_mini.cl` | Minimal variant of `relax_multi.cl`. | FireCore |
|| `Rigid.cl` | Rigid body dynamics kernels. | FireCore / SPAMMM |
|| `RRsp3.cl` | **RAFF/RRsp3 rigid-atom port forcefield** — 14 kernels: predict_dynamics, update_velocities_dynamics, update_bboxes_rigid, build_local_topology_rigid, compute_collision_cluster_rigid, 5 port variants (current/orig/substep/shapematch/eigen), compute_tips, compute_optimal_rotation_eigen, apply_corrections_rigid_ports. **Self-contained**. Compile-time macros: `GROUP_SIZE`, `MAX_GHOSTS`, `ENABLE_COLL`, `ENABLE_PORT`. Wired via `oclff::rrsp3::RRsp3`. CPU↔GPU parity verified. | FireCore `pyBall/RigidAtomFF/RRsp3/RRsp3.cl` |
|| `GridFF.cl` | B-spline grid interpolation for substrate surface potentials. Canonical FireCore tricubic B-spline. | FireCore `common_resources/cl/GridFF.cl` |
|| `gridff_spammm.cl` | SPAMMM GridFF: B-spline, Poisson solver, `make_GridFF` texture build, `sampleGridFF_Bspline_points`, charge projection. **Requires** `common.cl`+`Forces.cl` first. **Deprecated** — extracted into `gridff_build.cl`/`gridff_eval.cl`; not loaded by Rust. | SPAMMM `kernels/gridFF.cl` |
|| `Surface.cl` | Surface interactions (Morse/LJ/Coulomb), Ewald2D. Canonical FireCore; missing trailing hard-core hybrid section. | FireCore `common_resources/cl/Surface.cl` |
|| `surface_spammm.cl` | SPAMMM surface: folded atomic forcefield (FAF), Ewald2D, brute Morse, isosurfaces, tensor-exp/poly FAF. **Requires** `common.cl`+`Forces.cl` first. **Deprecated** — extracted into `faf_build.cl`/`faf_eval.cl`; not loaded by Rust. | SPAMMM `kernels/surface.cl` |
|| `grids.cl` | Grid utilities (lattice helpers, index math). **Deprecated** — not loaded by Rust. | SPAMMM `kernels/grids.cl` |
|| `PME.cl` / `PME8.cl` | Particle-mesh Ewald solvers. | SPAMMM `kernels/PME.cl` / `PME8.cl` |
|| `contact_surface.cl` | Quasi-2D contact surface: separable B-spline × polynomial + radial PIC. | SPAMMM `kernels/contact_surface.cl` |
|| `gridff_build.cl` | **Macro/fragment library** for GridFF construction (build kernels + helpers). | SurfMol |
|| `gridff_eval.cl` | **Macro library** for GridFF sampling + `SURF_INJECT_GRIDFF_BSPLINE`/`SURF_INJECT_NONE` surface injection macros for `getNonBond_generic.cl`. | SurfMol |
|| `faf_build.cl` | **Fragment library** for FAF construction (surface/isosurface/Ewald build kernels). | SurfMol |
|| `faf_eval.cl` | **Macro library** for FAF evaluation + `SURF_INJECT_FAF` surface injection macro for `getNonBond_generic.cl`. | SurfMol |
|| `nb_common.cl` | **NB loop macro fragments** — Axis 1 (`NB_PAIR_LJQH`) + Axis 2 (`NB_EXCL_*_NEIGHS4`) for `getNonBond_generic.cl`. | SurfMol |
|| `getNonBond_generic.cl` | **3-axis NB kernel template** — `NB_PAIR_FORCE` × `NB_EXCL_*` × `SURF_INJECT` macro slots. Assembled by `ClAssembler`. | SurfMol |
|| `getNonBond_reference.cl` | **Verbatim reference** — FireCore `getNonBond` + `getNonBond_GridFF_Bspline` for diffing. Not compiled. | FireCore |
|| `Assembly.cl` | Rigid-body molecular assembly / packing / clash evaluation. | SPAMMM |
|| `multigrid.cl` | Multigrid restriction (R=P^T tree-reduce), prolongation (tiled P), coarse Cholesky solve. | NumericalMathPlayground `LinarElasticity/kernels_multigrid.cl` |
|| `block_jacobi.cl` | Block Jacobi smoother (local-memory patches), residual, Dinv for truss/bond stiffness. | NumericalMathPlayground `LinarElasticity/kernels_block_jacobi.cl` |

**Note:** `multigrid.cl` + `block_jacobi.cl` are copied but **not yet wired** into any Rust crate. The Rust CPU implementation in `molff::multigrid` is the current authoritative reference. See `/doc/topical_audit/multigrid.md`.

## Rules

- **CPU Rust references are authoritative** for correctness; GPU must match CPU within tolerance (see `AGENTS.md` Rule 5).
- **NVIDIA GPU preferred** for all timings; never report PoCL/CPU timings as GPU.
- When porting/mirroring a kernel, cite the reference file in a comment.
- Kernels here are the GPU source of truth; the Rust OpenCL crate loads and dispatches them.
- **Two kernel conventions — do NOT mix in one OpenCL program:**
  - **FireCore self-contained** (`UFF.cl`, `relax_multi.cl`, …): each defines its own `cl_Mat3`/`R2safe`/`EXCL_MAX`. Load standalone.
  - **SPAMMM modular** (`SPFF.cl`): load as `common.cl` + `Forces.cl` + `SPFF.cl` (+ `nonbonded.cl`/`gridFF.cl`/`surface.cl` if non-bonded needed). `common.cl` is always FIRST.

## Macro assembler / fragment composition

Rust-side implementation: `crates/libs/oclff/src/assemble.rs` (mirror of SPAMMM `OpenCLBase.preprocess_opencl_source` / `parse_cl_lib`).

Conventions:

- `//>>>function NAME (signature)` and `//>>>macro NAME` mark library blocks in a `.cl` file. `ClLibrary::parse()` extracts them.
- `//<<<file FRAGMENT` (exact stripped line) injects a registered fragment by name.
- `//<<<macro MARKER` (exact stripped line) injects a macro body from `Substitutions`.
- `//<<<function MARKER(...)` replaces the exact string `//<<<function MARKER` with a function name from `Substitutions`.

**Architecture:**

- `gridff_build.cl` / `faf_build.cl` assemble **OpenCL programs** for constructing grids / surface potentials (`make_*`, `project_*`, `poisson*`, `eval_potential*`).
- `gridff_eval.cl` / `faf_eval.cl` are **macro libraries**; they do **not** compile as standalone programs. Their `//>>>macro` blocks are injected into the `getNonBonded` loop of each forcefield (`UFF.cl`, `SPFF.cl`, `RAFF.cl`, `RigidMolFF.cl`) so all forcefields share the same grid/surface NBFF primitive.
- `gridff_spammm.cl` / `surface_spammm.cl` are the canonical whole-file SPAMMM sources. The `//>>>`-marked build/eval blocks have been extracted from them into the four SurfMol fragment files. **Deprecated** — not loaded by Rust code; safe to delete.

**3-axis NB kernel template** (`getNonBond_generic.cl`):

The macro-variant principle is concretely realized as a single template kernel with three orthogonal macro axes:
1. **Pairwise potential** (`NB_PAIR_FORCE`): `NB_PAIR_LJQH` (UFF+SPFF shared) — in `nb_common.cl`
2. **Exclusion strategy** (`NB_EXCL_*`): `NB_EXCL_*_NEIGHS4` (4-neighbor int4) — in `nb_common.cl`
3. **Surface injection** (`SURF_INJECT`/`SURF_ARGS`): `SURF_NONE`, `SURF_GRIDFF_BSPLINE`, `SURF_FAF` — in `gridff_eval.cl`/`faf_eval.cl`

The `ClAssembler` injects `#define` aliases (via `NB_VARIANT_DEFINES` macro) mapping generic names to chosen variants. Currently: 1×1×3 = 3 kernel variants from 8 fragments. See `doc/topical_audit/gridff_faf.md` §3-axis NB template.

**Macro-variant principle (avoiding combinatoric explosion):**

We have N forcefields (UFF, SPFF, RAFF, RigidMolFF) and M surface/grid interaction variants (GridFF B-spline, FAF folded, FAF tensor-exp/poly, FAF harmonics, brute Morse, Ewald2D, ...). Without macro composition, each combination needs its own kernel file → N×M files that drift apart.

The macro assembler separates the forcefield loop from the interaction primitive:
1. **Forcefield kernels** contain the `getNonBonded` loop (iteration, exclusion, PBC, accumulation) and declare `//<<<macro NBFF_EVAL` at the interaction point.
2. **Interaction macros** (`gridff_eval.cl`, `faf_eval.cl`) contain `//>>>macro SAMPLE_3D`, `//>>>macro GET_SURF_FOLDED`, etc. — self-contained potential/force evaluation blocks.
3. **Composition at compile time** — `ClAssembler` injects the chosen macro into the chosen forcefield. UFF + GridFF? Assemble UFF.cl with `SAMPLE_3D`. RAFF + FAF tensor-poly? Assemble RAFF.cl with `GET_SURF_FOLDED_TENSOR_POLY`.

Result: N + M fragments instead of N×M kernel files. Adding a forcefield or surface variant is additive, not multiplicative. See `doc/topical_audit/gridff_faf.md` for the full contract.

## See also

- `Import_other_Repos.md` — which kernels to port and from where.
- `doc/topical_audit/gridff_faf.md` — build/eval split, macro-injection contract, and kernel inventory.
