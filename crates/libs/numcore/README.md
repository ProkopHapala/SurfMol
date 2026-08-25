---
type: rust-crate
title: numcore
description: Low-level numerical primitives — #[repr(C)] math vectors, aligned allocators, fast Taylor approximations. Zero domain knowledge, reusable in any project.
tags: [rust, crate, math, simd, aligned-alloc, ffi]
timestamp: 2026-08-25
---

# numcore

Numerical foundation crate with zero domain knowledge — used by every other SurfMol crate but reusable in any project that needs aligned arrays, f64 physics vectors, or f32 graphics math. The split between f64 (physics) and f32 (graphics) is intentional: forcefields need double precision, GPU rendering needs single.

## Modules

- **`math/vec3.rs`** — `Vec3d` (f64, `#[repr(C)]`): the workhorse 3D vector for all physics. Dual API: in-place methods (`add`, `sub`, `add_mul`) for hot loops that avoid allocation, and static methods (`set_add`, `set_sub`, `set_lincomb`) for one-shot expressions. Fused multiply-add operations (`add_mul`, `set_add_mul`, `add_lincomb`) target the common force-accumulation pattern `F += a * scalar`. `normalize` guards against zero-norm at 1e-14 threshold.
- **`math/vec2.rs`** — `Vec2d` (f64, `#[repr(C)]`): used as complex numbers (`mul_cmplx`, `udiv_cmplx`) for Fourier series in angle/dihedral forcefields. No operator overloads — avoids confusion between vector and complex semantics.
- **`math/quat4.rs`** — `Quat4d` (f64) and `Quat4i` (i32), both `#[repr(C)]`. `Quat4i` is not a rotation quaternion — it's a 4-int pack used for neighbor indices and 4-atom interaction tuples (dihedrals, inversions). `QUAT4I_MINUS_ONES` is the sentinel for empty neighbor slots.
- **`math/fastmath.rs`** — Hot-path approximations: `sincos_taylor2` (5th-order Taylor sin/cos), `sincos_r2_taylor` (takes r² as input to avoid sqrt, for small-angle dihedral terms), `dangle` (wraps angle difference to [-π,π] for periodic boundary), `clamp_abs` (branchless absolute-value clamp). All `#[inline(always)]`.
- **`math/math3d.rs`** — f32 array-based 3D helpers (`normalize3`, `cross3`, `dot3`, `sub3`, `add3`, `mul3s`) for GPU/graphics code. Uses `[f32; 3]` not a struct — zero-overhead for WGSL interop.
- **`math/math4d.rs`** — f32 4×4 matrix ops: `look_at` (right-handed view matrix), `ortho` (Vulkan/DirectX NDC with Z∈[0,1], not OpenGL's [-1,1]), `mul4x4`, `transpose4x4`. Row-major storage; transpose before GPU upload since WGSL expects column-major.
- **`util.rs`** — `AlignedVec<T, const A: usize>`: manual aligned allocator using `alloc_zeroed` with `Layout::from_size_align`. Guarantees A-byte alignment (power-of-two, asserted). Used with A=64 throughout SurfMol for cache-line-aligned force/position arrays. Zero-initialized to avoid UB; assumes `T: Copy` (no per-element Drop).

## Design decisions

- **`#[repr(C)]` on all vector types** — enables zero-copy FFI to OpenCL/CUDA and direct GPU buffer upload via `bytemuck`.
- **No SIMD intrinsics** — relies on compiler auto-vectorization of the component-wise loops. The aligned arrays ensure SIMD loads are safe.
- **Dual in-place/static API on Vec3d** — forcefield hot paths use in-place methods to avoid stack copies; one-shot expressions use static constructors.

## What does NOT belong here

- Anything chemistry-specific (elements, bonds, atoms) → `moltopo`
- Anything forcefield-related (energy, forces) → `molff` / `surfff`
- Anything rendering-related (shaders, pipelines) → `molrender`

## See also

- `ARCHITECTURE.md` §Component Details
- `moltopo` — uses `Vec3d`, `Quat4i`, `AlignedVec` for atom positions and neighbor lists
- `molff` — uses `Vec3d`, `Quat4d`, `Vec2d` (as complex) for force evaluation
