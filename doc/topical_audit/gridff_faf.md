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

## 3-axis NB kernel template (`getNonBond_generic.cl`)

The macro-variant principle is concretely realized in `getNonBond_generic.cl` — a single template kernel with three orthogonal macro axes. The `ClAssembler` injects `#define` aliases (via the `NB_VARIANT_DEFINES` macro substitution) that map generic names to specific variants before compilation.

### The three axes

| Axis | Generic macro | Variants | Fragment file |
|------|--------------|----------|---------------|
| 1. Pairwise potential | `NB_PAIR_FORCE(dp, REQK, R2damp)` | `NB_PAIR_LJQH` (UFF+SPFF shared) | `nb_common.cl` |
| 2. Exclusion strategy | `NB_EXCL_ARGS`, `NB_EXCL_SETUP`, `NB_EXCL_TEST`, `NB_EXCL_PBC_TEST` | `NB_EXCL_*_NEIGHS4` (4-neighbor int4) | `nb_common.cl` |
| 3. Surface injection | `SURF_ARGS`, `SURF_INJECT(posi, REQKi, fe)` | `SURF_NONE`, `SURF_GRIDFF_BSPLINE`, `SURF_FAF` | `gridff_eval.cl`, `faf_eval.cl` |

### How it works

1. The template (`getNonBond_generic.cl`) uses generic macro names (`NB_PAIR_FORCE`, `NB_EXCL_*`, `SURF_INJECT`, `SURF_ARGS`).
2. The Rust harness builds a `#define` alias block (the `NB_VARIANT_DEFINES` macro body) mapping each generic name to the chosen variant, e.g. `#define NB_PAIR_FORCE(dp,REQK,R2damp) NB_PAIR_LJQH(dp,REQK,R2damp)`.
3. `ClAssembler::assemble()` injects the alias block via `//<<<macro NB_VARIANT_DEFINES`.
4. The C preprocessor resolves the aliases at OpenCL compile time.

### Current variants

1 pairwise × 1 exclusion × 3 surface = **3 kernel variants** from **8 fragments** (1 template + 1 pairwise + 1 exclusion + 2 surface-args + 2 surface-inject + 1 none). FireCore has 6+ hand-written `getNonBond*` kernels for the same space; adding FAF injection in the FireCore style would require 2 more hand-written kernels. With the template, it's one more `SURF_INJECT_FAF` macro.

### Surface injection contract

Each `SURF_INJECT_*` macro expands to a self-contained block that:
- Declares any needed `__local` memory (e.g. `xqs[4]`, `yqs[4]` for GridFF PBC indices)
- Cooperatively loads shared data with `barrier(CLK_LOCAL_MEM_FENCE)`
- Samples the surface potential at `posi` using atom parameters `REQKi`
- Accumulates force+energy into `fe` (float4: xyz=force, w=energy)

Available variables in scope at injection point: `iG`, `iS`, `iL`, `nL`, `natoms`, `i0a`, `iaa`, `posi` (float3), `REQKi` (float4), `fe` (float4), `GFFParams` (float4).

Each `SURF_ARGS_*` macro declares the extra kernel arguments needed by the corresponding `SURF_INJECT_*` (appended after `GFFParams`). `SURF_ARGS_NONE` is empty.

### Reference vs assembled

`getNonBond_reference.cl` contains verbatim copies of FireCore `getNonBond` (UFF.cl:1023-1204) and `getNonBond_GridFF_Bspline` (UFF.cl:1523-1717). These are not compiled — they exist for diffing against the assembled output to verify the template reproduces the reference behavior.

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

- ~~Target host kernel names and argument convention for `getNonBonded` in `UFF.cl` / `SPFF.cl` / `RAFF.cl` / `RigidMolFF.cl`.~~ **Resolved**: `getNonBond_generic.cl` template with 3-axis macro slots. UFF and SPFF share the same NB kernel (both call `getLJQH`); the forcefield difference is only in the bonded kernels.
- ~~Whether `gridff_spammm.cl` and `surface_spammm.cl` are kept as canonical whole files or deleted after the fragments are extracted.~~ **Resolved**: content extracted into `gridff_build.cl`/`gridff_eval.cl`/`faf_build.cl`/`faf_eval.cl`. The whole files are deprecated and no longer loaded by any Rust code. Safe to delete.
- Exact helper-function dependency graph for each `__kernel` (some helpers are shared between build and eval; need a `grids.cl` utility fragment or duplicated minimal copies).
- **GPU compilation parity**: assembled `getNonBond_generic` variants compile on NVIDIA GPU (not yet tested — only source assembly is tested in `test_assemble_nb_generic.rs`).
- **Numerical parity**: assembled `getNonBond_uff_gridff` vs FireCore `getNonBond_GridFF_Bspline` — not yet tested.
- **Exclusion list variant** (`NB_EXCL_*_EXCL_LIST`): stub only; needs implementation when packed sorted exclusion lists are needed (ex2 style).

## See also

- `opencl/README.md` — kernel listing and assembly rules
- `opencl/getNonBond_generic.cl` — the 3-axis NB kernel template
- `opencl/getNonBond_reference.cl` — verbatim FireCore reference kernels
- `opencl/nb_common.cl` — Axis 1 + Axis 2 macro fragments
- `opencl/gridff_build.cl`, `opencl/gridff_eval.cl`, `opencl/faf_build.cl`, `opencl/faf_eval.cl` — the fragment templates
- `crates/libs/oclff/src/assemble.rs` — Rust `ClAssembler` / `ClLibrary` / `Substitutions`
- `crates/libs/oclff/tests/test_assemble_nb_generic.rs` — assembly tests for 3-axis NB variants
- `notes/designs/2026-08-29_nbff_surface_injection_design.md` — design spec for the 3-axis approach
- `notes/designs/2026-08-29_gridff_faf_porting_notes.md` — original porting notes
