---
type: folder
title: oclff/tests
description: Integration tests for the oclff crate — UFF OpenCL parity, SPFF compile smoke, and ClAssembler fragment parsing.
tags: [opencl, gpu, test, parity, uff, spff, assemble, fragments]
timestamp: 2026-08-29
---

# oclff/tests — Integration tests

Integration tests for the `oclff` crate. Run with `cargo test -p oclff`.

## Test files

### `test_uff_cl.rs` — UFF OpenCL bond parity

**What it tests:** `UffOcl::eval_bonds` (GPU) vs `molff::uff::Uff::eval_atom_bonds` (CPU) for a simple H₂ molecule.

**Test function:** `uff_bond_parity_h2`
- Sets up H₂ (2 atoms, 1 bond, k=100 eV/Å², l0=0.9 Å)
- Evaluates bond forces on CPU (`Uff::eval_atom_bonds`) and GPU (`UffOcl::eval_bonds`)
- Compares per-atom forces: asserts `|F_gpu - F_cpu| < 1e-4` for each component

**Status:** PASS. GPU bond evaluation matches CPU reference within f32 tolerance.

**Reference:** FireCore C++ `UFF.h::evalBonds` (parity target for both CPU and GPU).

---

### `test_spff_cl.rs` — SPFFsp3 OpenCL compile smoke

**What it tests:** `SpffOcl::new()` — verifies that the SPFFsp3 OpenCL program compiles successfully (concatenation of `common.cl` + `Forces.cl` + `SPFF.cl`).

**Test function:** `spff_cl_compiles`
- Calls `SpffOcl::new()` which builds the `ProQue` with the concatenated kernel source
- Asserts no OpenCL compile error

**Status:** PASS (compile only). No force evaluation or parity check — a full CPU Rust SPFF reference does not yet exist in `molff`. Once ported (from SPAMMM `SPFF_cl.py` / FireCore `MMFFsp3_loc.h`), this test should be extended to dispatch `getSPFFf4` and compare per-atom forces.

---

### `test_assemble_fragments.rs` — ClAssembler fragment parsing

**What it tests:** `ClLibrary::parse` and `ClAssembler::assemble` — verifies that the `//>>>function` / `//>>>macro` fragment libraries parse correctly and that macro injection works.

**Test functions (5):**

| Test | What it verifies |
|------|------------------|
| `parse_gridff_build_fragments` | `gridff_build.cl` parses to 21 `//>>>function` blocks (BsplineConv3D, make_MorseFF, make_GridFF, poissonW, project_*, etc.) and 0 macros |
| `parse_gridff_eval_macros` | `gridff_eval.cl` parses to 7 `//>>>macro` blocks (SAMPLE_3D, SAMPLE_3D_GRID, SAMPLE_GRIDFF_BSPLINE_POINTS, etc.) |
| `parse_faf_build_fragments` | `faf_build.cl` parses to 10 `//>>>function` blocks (getSurfMorse, eval_potential_*, compute_ewald_coefficients, getSurfaceIso*) |
| `parse_faf_eval_macros` | `faf_eval.cl` parses to 5 `//>>>macro` blocks (GET_SURF_FOLDED, GET_SURF_FOLDED_WORKGROUP, GET_SURF_FOLDED_HARMONICS, GET_SURF_FOLDED_TENSOR_EXP, GET_SURF_FOLDED_TENSOR_POLY) |
| `assemble_inject_sample_3d` | `ClAssembler` injects the `SAMPLE_3D` macro into a test kernel template via `//<<<macro SAMPLE_3D` sentinel — verifies the assembled source contains the macro body |

**Status:** All 5 tests PASS.

**Reference:** SPAMMM `OpenCLBase.preprocess_opencl_source` / `parse_cl_lib` (parity target for the assembler).

## Running

```bash
# All tests
cargo test -p oclff

# Specific test
cargo test -p oclff --test test_uff_cl
cargo test -p oclff --test test_assemble_fragments
```

> **Note:** `test_uff_cl` and `test_spff_cl` require an NVIDIA GPU (uses `nvidia_proque`). `test_assemble_fragments` is CPU-only (no GPU needed).

## See also

- [`../README.md`](../README.md) — oclff crate README (full API, open issues)
- [`../src/bin/README.md`](../src/bin/README.md) — CLI smoke-test binaries
- [`/doc/topical_audit/gridff_faf.md`](/doc/topical_audit/gridff_faf.md) — GridFF/FAF macro-fragment architecture
- [`/opencl/README.md`](/opencl/README.md) — OpenCL kernel inventory and conventions
