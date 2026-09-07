---
type: task-list
title: OCL harness consolidation — oclff crate + OpenCL base utilities
description: Tasks to merge molff-ocl + surfff-ocl into one OpenCL harness crate (oclff) and port kernel-header/buffer utilities from SPAMMM/FireCore.
tags: [opencl, oclff, consolidation, molff-ocl, surfff-ocl]
timestamp: 2026-08-29
---

# OCL harness consolidation — `oclff` crate + OpenCL base utilities

## Context

We now have two OpenCL harness crates in `crates/libs/`:

- `molff-ocl` — UFF, SPFFsp3, RAFF (RRsp3) GPU harnesses.
- `surfff-ocl` — GridFF + FAF GPU harness + the new `ClAssembler` macro system.

The user wants one unified OpenCL package `oclff` holding all OpenCL harnesses. Also, we need to copy/extend the OpenCL base utilities from SPAMMM/FireCore (`OpenCLBase.py`, `ocl_utils.py`, `OCL.h`...) that handle device selection, kernel header parsing, and automatic buffer allocation.

## Done already

- [x] Found SPAMMM/FireCore macro-assembly system (`OpenCLBase.preprocess_opencl_source` / `parse_cl_lib` / `//>>>` / `//<<<`).
- [x] Ported it to Rust: `surfff-ocl/src/assemble.rs`.
- [x] Created `gridff_build.cl`, `gridff_eval.cl`, `faf_build.cl`, `faf_eval.cl` templates.
- [x] Created `GridFFBuildOcl`, `GridFFEvalOcl`, `FafBuildOcl`, `FafEvalOcl` in `surfff-ocl/src/lib.rs`.
- [x] `cargo check` and `cargo test -p surfff-ocl` pass.
- [x] User added `nvidia_proque()` to `molff-ocl/src/lib.rs` and wired `uff.rs` / `spff.rs` to use it.

## Open questions / decisions

1. **Crate name**: `oclff` is preferred (OpenCL forcefields). Does it live at `crates/libs/oclff`?
2. **Scope of `oclff`**:
   - All GPU harnesses: UFF, SPFF, RAFF/RRsp3, GridFF, FAF, nonbonded, PME, contact surface, multigrid?
   - Or just the forcefield + surface harnesses and keep app-specific GPU crates separate?
3. **OpenCL base utilities** to port:
   - Device selection (NVIDIA-first): `nvidia_proque()` is already in `molff-ocl/src/lib.rs`.
   - Kernel header parser: extract `__kernel` signatures, argument names/types.
   - Buffer allocator: allocate `ocl::Buffer` with right dtype/length based on kernel header.
   - Kernel builder with named arguments / default values.
   - Macro assembler (`ClAssembler`) already in `surfff-ocl`.

## Task list

### 1. Investigate OpenCL base utilities (this session)

- [ ] Read `SPAMMM/spammm/utils/OpenCLBase.py` for `extract_kernel_headers`, `parse_kernel_header`, `generate_kernel_args`, buffer management.
- [ ] Read `FireCore/pyBall/OCL/oclUtils.py` and `FireCore/pyBall/OCL/OpenCLBase.py` for buffer/kernel helpers.
- [ ] Read `FireCore/cpp/common/OpenCL/OCL.h` for C++ equivalent (`newTask`, argument types).
- [ ] Report what utilities exist and which ones to port to Rust.

### 2. Merge `molff-ocl` + `surfff-ocl` → `oclff`

- [ ] Decide whether to rename one crate or create new `oclff` and delete the old two.
- [ ] Update workspace `Cargo.toml` members + dependencies.
- [ ] Move `molff-ocl/src/{pack,uff,spff,rrsp3}` and `surfff-ocl/src/{assemble,lib}` into `oclff/src/`.
- [ ] Move shared `nvidia_proque()` into `oclff::ocl_base`.
- [ ] Update `CODEMAP.md` and `opencl/README.md`.
- [ ] Update tests to use `oclff`.
- [ ] Run `cargo check` / `cargo test`.

### 3. Port OpenCL base utilities to `oclff`

- [ ] `oclff::ocl_base::nvidia_proque(src) -> ProQue` (already exists; move here).
- [ ] `oclff::ocl_base::KernelHeader` — parsed `__kernel` signature.
- [ ] `oclff::ocl_base::parse_kernel_header(src: &str) -> Vec<KernelHeader>`.
- [ ] `oclff::ocl_base::BufferSpec` + allocator based on header type + host array length.
- [ ] `oclff::ocl_base::KernelBuilder` that maps argument names to buffers/scalars.
- [ ] `oclff::ocl_base::ClAssembler` (currently `surfff-ocl::assemble`).

### 4. Wire the GPU harnesses to use the new utilities

- [ ] `UffOcl`, `SpffOcl`, `GridFF*Ocl`, `Faf*Ocl` use `nvidia_proque()` from `oclff`.
- [ ] Optionally use `KernelBuilder` for buffer/scalar argument mapping.
- [ ] Add compile-smoke tests that use the assembler + `nvidia_proque()` (requires NVIDIA GPU).

### 5. Consolidate / split the OpenCL `.cl` files with the macro system

- [ ] Add `//>>>` build/eval blocks to `opencl/gridff_spammm.cl` and `opencl/surface_spammm.cl`.
- [ ] Replace the full `//<<<file gridff_spammm.cl` in `gridff_build.cl` / `gridff_eval.cl` with targeted fragments.
- [ ] Replace the full `//<<<file surface_spammm.cl` in `faf_build.cl` / `faf_eval.cl` with targeted fragments.
- [ ] Embed eval fragments into `NBFF.cl` / `nonbonded.cl` templates as needed.
- [ ] Run OpenCL compile checks on NVIDIA.

## Files to read

- `SPAMMM/spammm/utils/OpenCLBase.py` — header parsing, `generate_kernel_args`, buffer dict.
- `SPAMMM/spammm/utils/ocl_init_old.py` — older OpenCL init helpers.
- `FireCore/pyBall/OCL/oclUtils.py` — C++-style OCL utilities.
- `FireCore/pyBall/OCL/OpenCLBase.py` — copied base class.
- `FireCore/cpp/common/OpenCL/OCL.h` — C++ `newTask`, `getKernelSource`, `buildProgram`.

## Key utilities found

### `OpenCLBase.py` (SPAMMM/FireCore)

- `extract_kernel_headers(source) -> {name: header_string}`
  - Regex `__kernel\s+void\s+([A-Za-z0-9_]+)\s*\(`.
  - Finds matching `)` (paren-level aware) and captures the full header.
- `parse_kernel_header(header) -> Vec<(name, kind)>`
  - Returns `(arg_name, type_tag)` where `0 = buffer/image`, `1 = constant`.
  - Detects `__global` as buffer, `__read_only image*_t` as image, else scalar.
- `generate_kernel_args(kname, overrides) -> Vec<arg>`
  - Looks up `kernel_params` and `buffer_dict` by argument name.
  - Builds the ordered argument list for a kernel call from the parsed header.
- Buffer helpers:
  - `create_buffer(name, size, flags)` — create + store in `buffer_dict`.
  - `check_buf(name, required_size, flags)` — create or resize if too small / none.
  - `try_make_buff(name, sz)` — reuse if same size, else reallocate.
  - `try_make_buffers(buffs, suffix="_buff")` — batch `try_make_buff`.
  - `toGPU_(buf, host_data, offset)` / `fromGPU_(buf, host_data, ...)` — copy helpers.
  - `download_buf(name_or_buf, dtype)` — device → flat numpy array, dtype inferred.
  - `roundUpGlobalSize(global_size)` — round to local work group multiple.

### `oclUtils.py` (FireCore)

- `OCLEnvironment` — platform index → context + queue.
- `loadProgram(fname)` — load file from `CL_PATH` and build.
- `updateBuffer(buff, cl_buff, access)` — create-or-update a buffer from host array.
- `printInfo()` / `printPlatformInfo()` — device diagnostics.

### `OCL.h` (FireCore C++)

- `getKernelSource(filename)` — read whole `.cl` file.
- `buildProgram(fname, program, build_options)` — compile and exit on failure.
- `newTask(name, program, dim, global, local)` — create a kernel object and store it.
- `upload(i, cpu_data, n, i0)` / `download(...)` — buffer copy helpers.

## Expected final crate layout

```
crates/libs/oclff/
  src/lib.rs          # re-exports
  src/ocl_base.rs     # nvidia_proque, kernel header parser, buffer allocator, KernelBuilder
  src/assemble.rs     # ClAssembler, ClLibrary, Substitutions
  src/uff.rs          # UffOcl
  src/spff.rs         # SpffOcl
  src/rrsp3.rs        # Rigid/RAFF OCL harness (when present)
  src/gridff.rs       # GridFFBuildOcl, GridFFEvalOcl
  src/faf.rs          # FafBuildOcl, FafEvalOcl
  tests/test_uff_cl.rs
  tests/test_spff_cl.rs
  tests/test_assemble.rs
```

## Notes

- `assemble.py` in SPAMMM may refer to **rigid-body self-assembly** (different from `ClAssembler`/`OpenCLBase` macro assembly). Do not confuse the two.
- NVIDIA device selection is required; PoCL/CPU must not be reported as GPU.
- The CPU Rust references (`molff::uff`, `surfff::SurfaceFolded`) remain authoritative for correctness.
