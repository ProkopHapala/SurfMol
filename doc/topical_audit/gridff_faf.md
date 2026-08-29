---
type: TopicalAudit
title: GridFF and FAF — OpenCL Macro Fragment Architecture
description: Cross-implementation map for GridFF/FAF B-spline grids and folded-atomic forcefield. Defines the build/eval split, macro-fragment conventions, and how eval macros are injected into the non-bonded force kernels of UFF/SPFF/RAFF/RigidMolFF.
tags: [gridff, faf, opencl, macros, b-spline, nonbonded]
timestamp: 2026-08-29
---

# GridFF and FAF — OpenCL Macro Fragment Architecture

GridFF (substrate grid forcefield) and FAF (folded atomic forcefield) are not standalone GPU programs in SurfMol. They are **source-fragment libraries** consumed by the `oclff` assembler. The build fragments run once to construct grids/potentials; the eval fragments are injected as `//>>>macro` blocks into the hot non-bonded force loops of `UFF.cl` / `SPFF.cl` / `RAFF.cl` / `RigidMolFF.cl` so all forcefields share the same NBFF primitive.

## Core idea

- **Do not compile** `gridff_spammm.cl` or `surface_spammm.cl` as full programs. Instead, annotate every `__kernel` and its dependent helpers with `//>>>function` / `//>>>macro` markers.
- **Build templates** (`gridff_build.cl`, `faf_build.cl`) assemble the construction kernels (`make_*`, `project_*`, `poisson*`, `eval_potential*`, `getSurfaceIso*`) into an OpenCL program that writes `float4` grid buffers.
- **Eval templates** (`gridff_eval.cl`, `faf_eval.cl`) are **macro libraries** that expose sampling functions (`sample*`, `getSurfFolded*`) via `//>>>macro` blocks. These blocks are then injected, one per target forcefield, using `//<<<macro NAME` sentinels inside that forcefield's non-bonded loop.
- **Target host kernels** (future) are `getNonBonded` / `getNBFF` in `UFF.cl`, `SPFF.cl`, `RAFF.cl`, `RigidMolFF.cl`. Each host declares the macro slot once; the same macro body can be shared by all forcefields.

## Why this direction

- A single NBFF primitive avoids four independent grid/surface sampling implementations.
- `//>>>macro` keeps the hot eval code textually together in `gridff_eval.cl` / `faf_eval.cl` while letting each forcefield's `getNonBonded` specialize argument names.
- Build and eval can be unit-tested as separate OpenCL programs; eval parity can be tested by compiling a tiny `test_eval.cl` that only calls the macro.

## The macro-variant principle (avoiding combinatoric explosion)

The fundamental problem: we have N forcefields (UFF, SPFF, RAFF, RigidMolFF) and M surface/grid interaction variants (GridFF B-spline, FAF folded, FAF tensor-exp, FAF tensor-poly, FAF harmonics, brute Morse, Ewald2D, ...). Without macro composition, each combination needs its own hand-written kernel file → N×M kernel files, most of them near-duplicates that drift apart over time.

The macro assembler solves this by **separating the forcefield loop from the interaction primitive**:

1. **Forcefield kernels** (UFF.cl, SPFF.cl, RAFF.cl, RigidMolFF.cl) contain the `getNonBonded` loop — the per-atom or per-pair iteration, exclusion logic, PBC handling, force accumulation. They declare `//<<<macro NBFF_EVAL` at the point where the non-bonded interaction is evaluated.

2. **Interaction primitive macros** (gridff_eval.cl, faf_eval.cl) contain `//>>>macro SAMPLE_3D`, `//>>>macro GET_SURF_FOLDED`, etc. — the actual potential/force evaluation code, written as self-contained blocks.

3. **Composition at compile time** — `ClAssembler` injects the chosen macro body into the chosen forcefield kernel. Want UFF + GridFF B-spline? Assemble UFF.cl with `SAMPLE_3D`. Want RAFF + FAF tensor-poly? Assemble RAFF.cl with `GET_SURF_FOLDED_TENSOR_POLY`. No new kernel files needed.

This gives N + M fragments instead of N×M kernel files. Adding a new forcefield or a new surface variant is additive, not multiplicative. The same principle applies to build kernels: `gridff_build.cl` and `faf_build.cl` are composed with `common.cl` + `Forces.cl` to produce construction programs without duplicating the shared infrastructure.

## Proposed `gridff_spammm.cl` split

| `__kernel` | Block type | Destination fragment | Purpose |
|------------|-----------|---------------------|---------|
| `sample3D`, `sample3D_grid`, `sample3D_comb2`, `sample3D_comb`, `sample1D_pbc`, `sampleGridFF_Bspline_points`, `sampleGridFF` | eval macro | `gridff_eval.cl` | Sample B-spline grids at points or grid locations. |
| `make_MorseFF`, `make_MorseFF_f4`, `make_Coulomb_points`, `make_GridFF` | build | `gridff_build.cl` | Build Pauli/London/Coulomb grids from atoms. |
| `project_atom_on_grid_cubic_pbc`, `project_atoms_on_grid_quintic_pbc` | build | `gridff_build.cl` | Project atom density onto grid with PBC. |
| `poissonW_old`, `poissonW`, `laplace_real_pbc`, `slabPotential`, `slabPotential_zyx` | build | `gridff_build.cl` | Spectral/real-space Poisson / slab correction. |
| `BsplineConv3D`, `BsplineConv3D_tex`, `Convolution3D_General`, `addMul`, `dot_wg`, `setLinear`, `move`, `setMul`, `setCMul`, `set` | utility | `grids.cl` / `common.cl` | Array operations and convolutions used by both build and eval. |
| `basis`, `dbasis`, `fe1D`, `fe2d`, `fe3d_pbc`, `make_inds_pbc`, `choose_inds_pbc*` | utility | `gridff_eval.cl` (eval) or `grids.cl` | Spline basis/helpers; eval macros need them at call site. |

## Proposed `surface_spammm.cl` split

| `__kernel` | Block type | Destination fragment | Purpose |
|------------|-----------|---------------------|---------|
| `getSurfFolded`, `getSurfFolded_workgroup`, `getSurfFolded_harmonics`, `getSurfFolded_tensor_exp`, `getSurfFolded_tensor_poly` | eval macro | `faf_eval.cl` | Folded-atomic forcefield evaluation (force + energy at a point). |
| `getSurfMorse`, `eval_potential_vacuum`, `eval_potential_full`, `eval_potential_brute`, `eval_potential_cluster`, `getSurfFlat`, `getSurfaceIsoSurfMorse`, `getSurfaceIsoGridFF`, `addDipoleField`, `compute_ewald_coefficients` | build | `faf_build.cl` | Construct surface potentials / isosurfaces / Ewald2D coefficients. |

## OpenCL assembler contract

- `//>>>macro SAMPLE_GRIDFF_BSPLINE( g0, dg, ng, Eg, n, ps, fes ) { ... }` — defines a macro named `SAMPLE_GRIDFF_BSPLINE`.
- `//>>>function sampleGridFF_Bspline_points(...)` — defines a function; can be emitted as-is.
- `//<<<macro SAMPLE_GRIDFF_BSPLINE` — injects the macro body into the surrounding source (usually inside `getNonBonded`).
- `//<<<file gridff_eval.cl` is **not** used for the eval library; instead the assembler is told to load `gridff_eval.cl` as a `ClLibrary`, then target kernels use `//<<<macro NAME`.

## Verification plan

1. Build templates compile and their `__kernel` names are discoverable by `oclff::KernelHeader`.
2. Eval macro bodies compile when injected into a minimal `test_eval.cl` host that calls the macro once.
3. GPU vs CPU parity for `sample3D` and `getSurfFolded` against `surfff::SurfaceFolded` (authoritative CPU reference).
4. Injected eval in `UFF.cl` / `SPFF.cl` reproduces the same grid energy/force as a standalone `gridff_spammm.cl` build+sample run.

## Open issues

- Exact helper-function dependency graph for each `__kernel` (some helpers are shared between build and eval; need a `grids.cl` utility fragment or duplicated minimal copies).
- Target host kernel names and argument convention for `getNonBonded` in `UFF.cl` / `SPFF.cl` / `RAFF.cl` / `RigidMolFF.cl`.
- Whether `gridff_spammm.cl` and `surface_spammm.cl` are kept as canonical whole files or deleted after the fragments are extracted.

## See also

- `opencl/README.md` — kernel listing and assembly rules
- `opencl/gridff_build.cl`, `opencl/gridff_eval.cl`, `opencl/faf_build.cl`, `opencl/faf_eval.cl` — the fragment templates (after refactor)
- `crates/libs/oclff/src/assemble.rs` — Rust `ClAssembler` / `ClLibrary` / `Substitutions`
- `notes/designs/2026-08-29_gridff_faf_porting_notes.md` — original porting notes
