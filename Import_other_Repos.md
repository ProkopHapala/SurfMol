# Import from Other Repos

Reference repositories we import algorithms, kernels, and project-organization patterns from. SurfMol is the Rust+OpenCL successor consolidating the jewels of these repos into a clean, compiled, GPU-first codebase.

**Cross-repo rules** (see `AGENTS.md` §Rule 6 — Parity Work):
- When porting/mirroring a feature, cite the reference file + function in a comment, e.g. `// ported from FireCore cpp/common/molecular/UFF.h:UFF::eval`.
- **FireCore is the performance benchmark** — SurfMol (Rust+OpenCL) must be at least as fast as the FireCore C++ reference for any ported algorithm. Measure, do not assume.
- **CPU Rust references are authoritative** for correctness; GPU (OpenCL) must match CPU within tolerance.

---

## 1. FireCore — `/home/prokop/git/FireCore/`

**Role:** Oldest repo for on-surface molecular dynamics and global optimization. Messy/disorganized C++ + Python(pyOpenCL) but contains many jewels. **SurfMol is its successor** — we import only the most useful forcefields and features, with a more organized GUI and OpenCL interface. **FireCore is the performance benchmark**: SurfMol must be at least as fast as this C++ code.

### Top-level layout
| Dir | Purpose |
|-----|---------|
| `cpp/common/` | High-performance C++ core: math, forcefields, data structures |
| `cpp/common_resources/cl/` | Canonical OpenCL kernels |
| `cpp/apps_OCL/`, `cpp/apps_CUDA/` | GPU-accelerated apps |
| `pyBall/OCL/` | Pure pyOpenCL implementations |
| `pyBall/RigidAtomFF/` | Position-Based Dynamics (XPBD, RRsp3) — ARAP ports |
| `tests/` | Test scripts and validation (START HERE) |
| `doc/` | Technical docs, derivations |

### Jewels to import (high priority)
| What | File | Notes |
|------|------|-------|
| **UFF force field** | `cpp/common/molecular/UFF.h`, `UFFbuilder.h`, `common_resources/cl/UFF.cl` | SoA layout, 64-byte aligned, OpenMP parallel. Core MM foundation. |
| **NBFF non-bonded** | `cpp/common/molecular/NBFF.h`, `common_resources/cl/Forces.cl` | LJ + Morse + Coulomb + H-bond with damping; AABB short-range; PBC. |
| **GridFF B-spline grid** | `cpp/common/molecular/GridFF.h`, `cl/GridFF.cl` | Tricubic B-spline interpolation; substrate surface potential. |
| **Buckets spatial partition** | `cpp/common/dataStructures/Buckets.h` (+ `Buckets2D/3D.h`, `HashMap2D.h`) | Spatial hashing, bi-directional object↔bucket. Core for neighbor search / collision. |
| **Projective Dynamics** | `cpp/common/math/ProjectiveDynamics_d.h` (+ `.cpp`, `_frag.cpp`) | Position-based dynamics for stiff springs; implicit, stable. |
| **MolWorld_sp3 MD loop** | `cpp/common/molecular/MolWorld_sp3.h` (`MDloop()` ~L2124-2169) | Reference MD/relaxation loop with `getCPUticks()` timing — **perf benchmark target**. |
| **Ewald2D surface electrostatics** | `common_resources/cl/Surface.cl` | 2D Ewald summation for periodic surfaces. |
| **RigidBodyFF quaternion dynamics** | `cpp/common/molecular/RigidBodyFF.h` | Quaternion rigid body integration, torque eval. |
| **RRsp3 rigid PBD + ARAP ports** | `pyBall/RigidAtomFF/RRsp3/` (`RRsp3.cl` 1311 lines, `RRsp3.py` 624) | Cluster-sorted PBD with ARAP ports, multiple rotation solvers. **Directly relevant to RAFF design** (see `notes/ToDo_user.md`). |

### Jewels (medium priority)
| What | File |
|------|------|
| MMFFsp3_loc force field | `cpp/common/molecular/MMFFsp3_loc.h` |
| GOpt global optimization (basin-hopping) | `cpp/common/molecular/GOpt.h`, `GlobalOptimizer.h` |
| `relax_multi.cl` unified multi-system kernel | `common_resources/cl/relax_multi.cl` |
| RARFF reactive force field (Morse, bond making/breaking) | `cpp/common/molecular/RARFF_SR.h`, `FlexibleAtomReactiveFF.h` |
| DynamicOpt / CG / lineSearch optimizers | `cpp/common/math/DynamicOpt.h`, `CG.h`, `lineSearch.h` |

### Perf benchmark harness
- `getCPUticks()` cycle counter (used in `MolWorld_sp3.h:2130`, `MolGUI.h:1197`) with `tick2second` calibration.
- `tests/tMMFF/`, `tests/tSiNCs/` (timing reports in `OUT_nc_ensemble_v2/out/timing_report.md`), `tests/tEFF/` (CPU vs GPU parity).
- Target: match or beat `MolWorld_sp3::MDloop()` (~1–10 μs/iter for small systems).

---

## 2. SPAMMM — `/home/prokop/git/SPAMMM/`

**Role:** Full-featured Python + pyOpenCL scanning-probe microscopy and manipulation engine. Scope overlaps heavily with SurfMol: SurfMol is more about manipulation + global optimization, SPAMM focuses more on imaging, but both contain both aspects. **SurfMol is to a large degree rewriting SPAMM into Rust** to eliminate Python overhead and produce a compiled binary.

### Top-level layout
| Dir | Purpose |
|-----|---------|
| `kernels/` | OpenCL `.cl` sources (all GPU compute) |
| `spammm/topology/` | Molecular topology SSOT: `AtomicGraph` |
| `spammm/forcefields/` | UFF, SPFF, LFF, rigid body |
| `spammm/surfaces/` | GridFF, ContactSurface, Ewald |
| `spammm/SPM/` | AFM/STM imaging |
| `spammm/utils/` | `OpenCLBase` (device selection, buffer mgmt) |

### Jewels to port to Rust+OpenCL (high priority)
| What | File | Notes |
|------|------|-------|
| **OpenCLBase (NVIDIA-first device selection)** | `spammm/utils/OpenCLBase.py:133-150` | `select_device(preferred_vendor='nvidia')`. Port to Rust OpenCL crate. |
| **AtomicGraph topology SSOT** | `spammm/topology/AtomicGraph.py` | Stable object identities, `to_arrays()` export. Mirror in `surfmol-topology`. |
| **`getNonBond_ex2`** | `kernels/nonbonded.cl:135-277` | Pairwise LJ/Coulomb with 2nd-neighbor exclusion, local-memory tiling (32 atoms/tile), PBC. |
| **UFF kernels** | `kernels/UFF.cl` (`evalBondsAndHNeigh_UFF`, `evalAngles_UFF`, `evalDihedrals_UFF`, `evalInversions_UFF`, `assembleForces_UFF`) | Harmonic bonds + hneigh vectors reused by angles/dihedrals. |
| **SPFF kernels** | `kernels/SPFF.cl` (`getSPFFf4`, `updateAtomsSPFFf4`, `relax_nsteps_serial`) | Bonds, angles, π-orbital DOFs, FIRE relaxation. |
| **Rigid body 6-DOF** | `kernels/rigid.cl` (`rigid_body_folded_kernel`, `rigid_body_pairff_probe_grid`) + `spammm/forcefields/RigidBodyDynamics.py` | Quaternion integration, gyroscopic term, per-body state, ping-pong multimol MD. |
| **LFF projective Jacobi** | `kernels/LFF.cl` + `spammm/forcefields/LFFSolver.py` | Linearized projective Jacobi on K12/K13/K14 springs — fast relaxation surrogate. Closest existing thing to "position-based dynamics" in the repo. |
| **Contact surface (separable B-spline×poly + radial PIC)** | `kernels/contact_surface.cl` (`evalSeparableBsplinePoly`, `relaxStrokesTiltedContactPME*`, `fillContactPMEMeshVL`) | Quasi-2D contact field for static AFM. |
| **GridFF tricubic B-spline + Poisson** | `kernels/gridFF.cl` (`sample3D*`, `poissonW*`) | Tricubic interpolation with PBC; FFT Poisson solver with slab correction. |
| **Rigid-body packing/clash** | `kernels/assembly.cl` (`evaluate_packing_3d`) | Steric clash with early exit, local-memory tiling. |

### Data layouts to mirror
- `float4 apos[natoms]` (w = mass/charge), `float4 aforce[natoms]` (w = energy), `float4 REQs[natoms]` (RvdW, EvdW, Q, H-bond), `int4 neighs[natoms]` (up to 4 neighbors).
- **`float4.w` channel reuse** for energy / secondary results / clash flags — avoid extra buffers.
- Workgroup-sized fragments: 32 atoms/tile (nonbonded), `MAX_ATOMS_PER_BODY=128` (rigid), `LFF_WG_SIZE=64`, `CS_TILE=16`. **Matches the 16/32/64/128 atoms-per-fragment design in `notes/ToDo_user.md`.**

### Notes
- No reactive/dissociative potentials or port-based bonding in SPAMM — those come from FireCore (`RARFF`, `RRsp3`).
- NVIDIA GPU requires unrestricted shell so the ICD is visible; sandbox hides it and falls back to PoCL/CPU (never report PoCL timings as GPU).

---

## 3. learn_Rust — `/home/prokop/git/learn_Rust/`

**Role:** Testbed for Rust algorithms and OpenCL interface patterns, including fast collision acceleration. Import tested Rust patterns + OpenCL bindings into SurfMol.

### Top-level layout
| Dir | Purpose |
|-----|---------|
| `examples/` | Progressive demos (11): OpenCL, OpenCL-GL interop, collision, UFF MD |
| `mol_utils/` | `Vec3`, `Quat4`, `AlignedVec` (64-byte aligned, `#[repr(C)]`) |
| `mol_topology/` | Bonds, angles, UFF assignment |
| `mol_engine/` | UFF, nonbonded, MD integration |
| `data/` | Test molecules + UFF params |
| `NOTES/` | Design notes |

### Jewels to import
| What | File | Notes |
|------|------|-------|
| **OpenCL buffer management (ProQue)** | `examples/demo03_opencl/src/main.rs:35-67` | Clean `ocl` 0.19 ProQue + Buffer pattern. |
| **OpenCL-OpenGL zero-copy interop** | `examples/demo06_opencl_opengl_interop/src/main.rs:186-314` | `cl_khr_gl_sharing` platform/device iteration, GLX/EGL handle extraction, `Buffer::from_gl_buffer()`, acquire/release cycle. **Critical for GPU-rendered GUI.** |
| **AlignedVec + Vec3d/Quat4d** | `mol_utils/src/math/vec3.rs`, `mol_utils/src/util.rs` | `#[repr(C)]`, 64-byte aligned, inlined ops. Mirror in `surfmol-common`. |
| **UFF SoA data layout** | `mol_engine/src/uff.rs:43-208` | Neighbor indexing (`neighs`, `neigh_bs`), bucket-based force assembly (`a2f`). |
| **Non-bonded with exclusions + PBC** | `mol_engine/src/nonbonded.rs:54-79` | Sorted exclusion list, PBC shift vectors. |
| **Group-based AABB broad phase** | `examples/demo10_collision_balls/src/main.rs:81-118` + `collision_kernel.cl:224-305` | Per-group AABB reduction (local memory), bit-matrix overlap, degree-based dispatch. WG=32. |
| **Uniform grid + parallel scan** | `examples/demo11_collision_grid/src/main.rs:23-65` + `collision_kernel.cl:45-156` | Blelloch prefix scan, atomic scatter, 3×3 cell stencil neighbor gather. |
| **Morton code spatial sorting** | `examples/demo10_collision_balls/src/main.rs:24-56` | 2D Morton Z-curve for spatial locality. |
| **`bytemuck` zero-cost casts** | `examples/demo05_pointer_reinterpret/src/main.rs` | "Numpy view" pattern: struct slice ↔ flat array slice. |

### Key deps (Cargo.toml)
`ocl = "0.19"`, `eframe/egui = "0.29"`, `wgpu = "24.0"`, `bytemuck = "1.21"`, `nalgebra = "0.33"`, `ndarray = "0.16"`, `rhai = "1.19"`, `clap = "4.5"`, `serde`/`serde_json`.

### Notes
- No dedicated benchmark harness — timing is inline (`std::time::Instant`) in demos. SurfMol should add a real bench harness.
- `ocl` 0.19 is the chosen OpenCL crate (not `opencl3`). blood_of_civilization uses `opencl3` 0.12 — **decision needed** on which to standardize on (see §4).

---

## 4. blood_of_civilization — `/home/prokop/git/blood_of_civilization/`

**Role:** Unrelated game (terrain/economy/combat) but the **most developed Rust project** we have — import Rust project organization and binary/memory optimization settings. Key notes in `doc/AGENTS/notes/Memory_Issues/`.

### Workspace organization (15 crates + xtask)
Pattern: **domain crates (Bevy-free) + app crate (presentation) + xtask (tooling) + feature-gated opencl crate.**
- Domain: `boc_core`, `boc_protocol`, `boc_geo`, `boc_economy`, `boc_tactics`, `boc_chem`, `boc_plot`.
- Integration: `boc_ecs`, `boc_python`, `boc_script`.
- App: `boc_app` (full Bevy, migrating to eframe), `boc_pipedream` (eframe).
- Specialized: `boc_opencl` (feature-gated, **only crate with `unsafe`**), `boc_fluid2d`, `boc_procedural2d`, `vibbug` (HTML debug reports).
- Tooling: `xtask` (`cargo xtask check|test|verify|check-ownership`).

**Naming:** all crates prefixed `boc_`. SurfMol already uses `surfmol-*` prefix — keep that.

### OpenCL integration pattern
- Crate `boc_opencl`, feature-gated (`opencl = ["dep:opencl3"]`), uses `opencl3 = "0.12"`.
- `#![cfg_attr(not(feature = "opencl"), forbid(unsafe_code))]` — **all `unsafe` confined to this one crate**, no raw handle escapes the boundary, every `unsafe` block has a SAFETY comment.
- CPU reference implementations live in domain crates; OpenCL is optional acceleration, not required for correctness. **Mirror this exactly in SurfMol.**

### MUST-IMPORT: Cargo profile overrides
From `Cargo.toml:63-84`. Apply to SurfMol workspace root:
```toml
[profile.dev]
debug = 1                      # line tables only; cuts debug info ~80%, keeps panic file:line
strip = "debuginfo"            # strips DWARF, keeps .eh_frame + symbol table; binary 935MB→343MB

[profile.release]
lto = "thin"
codegen-units = 1
debug = 1                      # keep line tables for release backtraces
strip = "debuginfo"
incremental = true             # fast rebuilds in release
debug-assertions = true        # fail-loud in release
overflow-checks = true         # integer overflow panics in release
```
**Verified effects** (with `debug=1` + `strip="debuginfo"`): panic location KEPT, function names in backtrace KEPT, per-frame file:line LOST (acceptable), `.eh_frame` unwind tables SURVIVE. pipedream release binary 142MB→15MB (9.5×).

### MUST-IMPORT: Shared target directory
`~/.cargo/config.toml`:
```toml
[build]
target-dir = "/path/to/shared/target"
```
Reclaimed 24.7 GB → 2.3 GB (91% reduction) across all Rust projects. **Apply globally.**

### MUST-IMPORT: IDE indexing guard
`.codeiumignore` / `.vscode/settings.json` excluding `target/`, `artifacts/`, `debug/` from language-server indexing (`searchMaxWorkspaceFileCount: 200`). Prevents the LS from indexing multi-GB build artifacts.

### SHOULD-IMPORT
- **xtask** workspace automation (`check`, `test`, `verify`, `check-ownership --base <sha>`).
- **Stale artifact cleanup policy**: `cargo clean` when `target/` exceeds 15 GB; `scripts/target_size.sh` monitor.
- **Dependency audit**: replace heavy crates with light alternatives (e.g. `image`→`png` for PNG-only use; disable `plotters` `ttf` feature).
- **Test binary consolidation**: each integration test file = ~45-50 MB executable; merge where sensible.
- **Unsafe isolation in single feature-gated crate** (see OpenCL pattern above).

### Key Memory_Issues notes (in `doc/AGENTS/notes/Memory_Issues/`)
| File | Key takeaway |
|------|--------------|
| `rust_footprint.md` | 31 GB `target/` from 1,796 crates; `debug=1` is the single biggest lever (5× reduction). |
| `reduce_target_footprint_plan.md` | Shared target dir + `strip="debuginfo"` = 91% disk reduction. Backtrace verification results. |
| `dependency_review.md` | `boc_core` pulls `image` (67 MiB) for 2 calls — replace with `png` (9 MiB). Bevy feature pruning does NOT work (bevy_egui forces features). |
| `migrate_pipedream_to_eframe.md` | Bevy→eframe: deps 568→246 (−57%), binary 142MB→15MB (9.5×). |
| `alternative_gui.md` / `3d_renderer_alternatives.md` | wgpu + egui-wgpu = 14 MiB (vs Bevy 142 MiB). **SurfMol `editor` is the working reference** (299 deps, 14 MiB stripped). |
| `system_memory_optimization.md` / `devin_memory_optimization.md` | 16 GB RAM machine ops: kill junk processes, cap Go LS heap (`GOMEMLIMIT`), disable IDE indexing. |
| `devin_desktop_renderer_leak_bugreport.md` | Electron renderer leaks ~145 MB/min; restart IDE every 20-30 min. |

### GUI decision (already validated for SurfMol)
SurfMol `editor` already uses **wgpu + winit + egui** (14 MiB stripped release, 299 deps) — this is the recommended stack from blood_of_civilization's migration analysis. Keep it; do not adopt Bevy.

---

## Cross-repo import priority summary

| Priority | Source | What | Target in SurfMol |
|----------|--------|------|-------------------|
| P0 | blood_of_civilization | Cargo profile overrides (`debug=1`, `strip`, LTO) | workspace `Cargo.toml` |
| P0 | blood_of_civilization | Shared `target-dir` in `~/.cargo/config.toml` | global cargo config |
| P0 | blood_of_civilization | `.codeiumignore` / IDE indexing guard | repo root |
| P0 | learn_Rust | OpenCL-GL zero-copy interop (`demo06`) | `surfmol-apps` GUI rendering |
| P0 | learn_Rust | AlignedVec + Vec3d/Quat4d | `surfmol-common` |
| P0 | FireCore | UFF + NBFF + Buckets | `surfmol-forcefields` + `surfmol-common` |
| P0 | FireCore | `MolWorld_sp3::MDloop()` | **perf benchmark target** |
| P0 | SPAMMM | `OpenCLBase` NVIDIA-first device selection | Rust OpenCL crate |
| P1 | FireCore | RRsp3 rigid PBD + ARAP ports | `surfmol-forcefields` (RAFF, see `notes/ToDo_user.md`) |
| P1 | FireCore | Projective Dynamics | `surfmol-forcefields` |
| P1 | SPAMMM | `rigid.cl` 6-DOF rigid body | `surfmol-forcefields` |
| P1 | SPAMMM | `nonbonded.cl` `getNonBond_ex2` | `opencl/` kernels |
| P1 | learn_Rust | Group AABB broad phase + uniform grid scan | `opencl/` collision kernels |
| P1 | SPAMMM | AtomicGraph topology SSOT | `surfmol-topology` |
| P2 | FireCore | GridFF, Ewald2D, RARFF, GOpt | `surfmol-forcefields` (later) |
| P2 | SPAMMM | LFF projective Jacobi, contact_surface, gridFF | `opencl/` (later) |
| P2 | blood_of_civilization | xtask automation, unsafe-isolation crate pattern | workspace tooling |

## Resolved design decisions
1. **OpenCL crate:** **`ocl` 0.19** (from learn_Rust). Higher-level ProQue/Buffer API; the OpenCL-GL interop demo already works with it. Adopt the unsafe-isolation-in-a-single-feature-gated-crate pattern from blood_of_civilization regardless of crate choice.
2. **Fragment memory layout:** **contiguous fragments** — each fragment's atoms (node + capping) contiguous in memory for best cache/workgroup locality. Trade-off: harder to update topology, accepted.
3. **Capping atoms (H, epairs):** **rigid appendix fixed to a host-atom port** (no independent DOF). Simpler, fewer DOF, faster. Revisit if H-relaxation fidelity becomes an issue.
