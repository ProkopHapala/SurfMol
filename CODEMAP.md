---
type: developer-docs
title: CODEMAP
description: Repo structure, file inventory, crate dependency graph, pinned dependencies, and build/test commands.
tags: [codemap, navigation, structure, dependencies]
timestamp: 2026-08-29
---

# CODEMAP

Repository structure, file inventory, crate dependency graph, and build/test commands. Update this when structure changes (see `AGENTS.md` §Folder Roles).

## 1. Top-level layout

```
SurfMol/
├── AGENTS.md                 # Binding agent rules of conduct (read first)
├── ARCHITECTURE.md           # Crate layout, file-naming rules, directory structure
├── DESIGN_GOALS.md           # Scientific/engineering goals + design decisions
├── Import_other_Repos.md     # Reference repos (FireCore, SPAMMM, learn_Rust, blood_of_civilization)
├── CODEMAP.md                # This file
├── README.md                 # Repo README (OKF)
├── Cargo.toml                # Workspace root (14 members, resolver 2)
├── .cargo/config.toml        # Shared target dir fallback (CARGO_TARGET_DIR overrides)
├── data/                     # Molecular inputs (.xyz, .mol, .mol2) + FF params (.dat)
├── debug/                    # Diagnostic plots (gitignored except README)
├── doc/                      # Permanent developer docs + topical_audit/
├── notes/                    # Ephemeral work-in-progress (chats, designs, labbooks, reports, tasks, TODOs)
├── opencl/                   # OpenCL .cl kernel sources
├── crates/                   # Rust workspace (10 libs + 4 apps)
│   ├── libs/                 # Library crates (no binary targets)
│   └── apps/                 # Binary crates (CLI/GUI tools)
└── userguide/                # End-user docs for finished modules (editor.md populated)
```

## 2. Rust workspace

**Workspace root:** `Cargo.toml` (15 members, resolver 2).
**Shared target dir:** `CARGO_TARGET_DIR` env var (machine-local, typically `~/.cargo/shared_target`); repo `.cargo/config.toml` falls back to local `target/`.

### Members

```
crates/libs/   (12 library crates, no binary targets)
  numtypes, numcore, moltopo, pgraph, spacc,
  molff, oclff, surfff, surfmol,
  molrender, molgui

crates/apps/   (4 binary crates)
  buildff, molengine, editor, molbrowser
```

### Crate dependency graph

```
                numtypes
       ___________|_____________
      /            |            \
   numcore       pgraph         spacc
      |            |              |
      ↓            ↓              ↘
  moltopo ←── molff ←── surfmol
      ↘       ↗         ↗
       molrender ← molgui
      ↗
   surfff
   ↓
   oclff (← molff, surfff, moltopo)
```

Apps depend on libs:
- `buildff` → moltopo, numcore
- `molengine` → surfmol, molff, moltopo, numcore
- `editor` → surfmol, molff, surfff, moltopo, numcore, molgui, molrender, **spacc** (for `aabb_edges` visualization)
- `molbrowser` → moltopo, numcore, molgui, molrender

**Note:** `molff` now depends on `spacc` (for `BroadPhase` struct using `broad_phase_pairs` + `fit_range_aabbs`).

### Workspace dependencies (`Cargo.toml`)

| Crate | Version | Notes |
|-------|---------|-------|
| `eframe` | 0.34 | `default-features=false`, `features=["default_fonts","wgpu","wayland","x11"]` |
| `egui` | 0.34 | Immediate-mode GUI |
| `egui-winit` | 0.34 | `default-features=false`, `features=["wayland","x11","links"]` |
| `egui-wgpu` | 0.34 | egui ↔ wgpu backend |
| `ndarray` | 0.16 | **Currently unused in source** — listed in `molff` and `molengine` Cargo.toml but no `use ndarray` in any `src/`. Candidate for removal. |
| `rand` | 0.8 | RNG |
| `ocl` | 0.19 | **OpenCL bindings — chosen crate** (see `DESIGN_GOALS.md` §10) |
| `wgpu` | 29 | GPU compute/graphics |
| `pollster` | 0.4 | Async runtime (block_on) |
| `bytemuck` | 1.21 | `features=["derive"]` — zero-cost casts for GPU buffers |
| `winit` | 0.30 | Windowing |
| `glam` | 0.29 | **Under review** — used for `Quat`/`Vec2`/`Vec3` in `editor`, `molgui` (`trackball`, `gizmos`), and `molbrowser` Cargo.toml. Can be replaced by `numtypes::Vec2f`/`Vec3f`/`Vec4f` + `qmul` once `numtypes` quaternion helpers mature. |
| `serde` | 1.0 | `features=["derive"]` — serialization |
| `serde_json` | 1.0 | JSON I/O |
| `rhai` | 1.19 | Scripting (molengine) |
| `clap` | 4.5 | `features=["derive"]` — CLI parsing |
| `image` | 0.25 | `default-features=false`, `features=["png"]` — PNG I/O (dev-dep in molrender, dep in molgui) |
| `arboard` | 3.6 | `default-features=false` — clipboard (molgui, editor) |

### Build profiles

Ported from `blood_of_civilization/doc/AGENTS/notes/Memory_Issues/reduce_target_footprint_plan.md` A1:
- `[profile.dev]`: `debug=1` (line tables only), `strip="debuginfo"` (~16× smaller debug binaries, `.eh_frame` survives for backtraces)
- `[profile.release]`: `lto="thin"`, `codegen-units=1`, `debug=1`, `incremental=true`, `debug-assertions=true`, `overflow-checks=true`, `strip="debuginfo"`

## 3. File inventory by crate

### `numtypes` (`crates/libs/numtypes/`, ~380 LOC)
*Low-level memory/data-layout vocabulary — `#[repr(C)]` math vectors/matrices, aligned allocators, graph/spatial data contracts. Tiny intrinsic operations only.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 34 | Crate root; re-exports `vec`, `mat`, `alloc`, `graph`, `spatial` public items. |
| `src/vec.rs` | ~170 | `Vec2f/d`, `Vec3f/d`, `Vec4f/d`, `Vec4i`, `Vec6f/d`; component-wise ops, `array()` zero-copy views, `Index<usize>`, complex helpers (`cmul`, `cconj`), quaternion helpers (`qmul`, `qconj`, `qrotate`), `Quat4d`/`Quat4i` aliases, `QUAT4I_MINUS_ONES`. |
| `src/mat.rs` | ~150 | `Mat3d`/`Mat4d` and `Mat4f`; `rows()`/`array()` views, `dot`, `mmul3`/`mmul4`/`mmul4f`, `outer`, `det`, `inverse`. `Mat4f` adds `look_at`, `ortho`, `to_arr4x4()` for graphics. |
| `src/alloc.rs` | ~95 | `AlignedVec<T, A>` — 64-byte aligned allocator, `Deref`/`DerefMut` to `[T]`, `with_len_fill`, `resize_fill`, `push`. |
| `src/graph.rs` | ~150 | `Index`, `INVALID`, `PGraph`, `PGraphView`, `Elements<N>`, `RaggedIndex` (replaces `Ragged`+`IndexGroups`), `Permutation`, `Partition`, `RangeGroups`, `CsrAdj`, `FixedRows<K>`, `FixedAdj<K>` (using aligned `AlignedVec`). |
| `src/spatial.rs` | ~90 | `Aabb3d`/`Aabb3f` and `SymMat3d`/`SymMat3f` as `Vec6` aliases; standalone `aabb_*` and `sym3_*` intrinsic functions. `aabb_overlap_margin`, `aabb_point_dist2`, `aabb_sphere_overlap` added for broad-phase collision. |

### `numcore` (`crates/libs/numcore/`, ~110 LOC)
*Numerical algorithms. Does **not** re-export `numtypes` data; owns `fastmath` and `linalg` only.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 1 | Crate root: `pub mod math;` |
| `src/math/mod.rs` | 2 | Module declarations: `fastmath`, `linalg`. |
| `src/math/fastmath.rs` | 36 | `sq`, `dangle`, `clamp_abs`, `sincos_taylor2`, `sincos_r2_taylor` (Taylor approximations). |
| `src/math/linalg.rs` | 92 | `symmetric_eigen_3x3` — analytical (closed-form) 3×3 symmetric eigendecomposition. Ported from FireCore `Mat3.h:Mat3T::eigenvals()` + `eigenvec()`. Replaces nalgebra for PCA in thumbnailer. **Added:** `cholesky_factor_f64`, `cholesky_solve_f64`, `dense_matvec_f64` — f64 dense Cholesky for the multigrid coarse solve. See `/doc/topical_audit/multigrid.md`. |

(The `util`, `vec2`, `vec3`, `quat4` re-export modules and the f32 `math3d`/`math4d` array helpers have been removed. Data primitives and f32/f64 matrix constructors now live in `numtypes`.) |

### `moltopo` (`crates/libs/moltopo/`, 1855 LOC)
*Molecular topology SSOT. Bonds/angles/dihedrals/inversions. UFF type assignment.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 5 | Crate root: `pub mod topology; molecular; builder; params; assign_uff; export; xyz;` |
| `src/topology.rs` | 173 | `Topology` (apos, bonds, angles, dihedrals, inversions), `ne_pairs()`, `hybridization()`, `build_bonds_by_cutoff()`, `build_angles/dihedrals/inversions_from_bonds()`. Diagnostic prints (parity with C++ `MMFFBuilderBase.h`). |
| `src/molecular.rs` | 165 | `Atoms` (static: natoms, atypes, apos, neighs, neigh_bs; `make_neigh_bs()`), `DynamicAtoms` (Atoms + fapos + vapos; `move_atom_md()`, `run_md()`, `clean_force/velocity`). **SSOT for atomic state.** |
| `src/builder.rs` | 599 | `Builder` — slot-based molecular graph with generational handles (`AtomH`, `BondH`), soft/hard remove, `cleanup_dead()`, `bake()` → `Topology`. Hex grid editing (`honeycomb_ring_nodes`, `snap_to_node`, `add_hex_ring`). `from_positions_cutoff()`, `from_positions_and_radii()`. |
| `src/params.rs` | 664 | `Params` — loads `ElementTypes.dat`, `AtomTypes.dat`, `BondTypes.dat`, `AngleTypes.dat`, `DihedralTypes.dat`. Structs: `ElementType`, `AtomType`, `BondParam`, `AngleParam`, `DihedralParam`. Wildcard matching for angle/dihedral params. |
| `src/assign_uff.rs` | 125 | `assign_uff_types()` — octet-rule hybridization → UFF suffix (_3/_R/_2/_1). Special cases: H_, nitro N_R/O_R, C=O O_2, alkyne C_1. |
| `src/export.rs` | 74 | `TopologyData` (serde), `Topology::export_json()`, `import_json()`. TODO: .npy export. |
| `src/xyz.rs` | 48 | `XyzSystem`, `read_xyz()`, `write_xyz_frame()` — XYZ file I/O. |

### `pgraph` (`crates/libs/pgraph/`, ~620 LOC)
*Reusable graph algorithms on `numtypes` data. Ported from FireCore `MolecularGraph.h` without class ownership.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 11 | Module declarations: adjacency, components, bridges, reorder, geometry. |
| `src/adjacency.rs` | 155 | `build_csr_adj` (count→prefix→scatter), `build_fixed_adj::<K>` (fails loud on degree > K via `DegreeOverflow`). Both produce parallel `neigh` + `edge` arrays. |
| `src/components.rs` | 85 | `connected_components(csr)` via iterative BFS → `Partition`. `split_by_component(csr)` → `RaggedIndex`. |
| `src/bridges.rs` | 130 | `find_bridges(csr)` via iterative Tarjan DFS (discovery times + low-link). No recursion → no stack overflow. |
| `src/reorder.rs` | 154 | `partition_to_index_groups`, `group_aware_permutation` (→ `Permutation` + `RangeGroups`), `apply_permutation`, `permute_edges`. |
| `src/geometry.rs` | 87 | `edge_vec`, `edge_length`, `edge_lengths`, `bounding_box`, `bounding_box_center`, `bounding_box_span`. |

### `spacc` (`crates/libs/spacc/`, ~280 LOC)
*Spatial acceleration — rebuildable caches, no molecular semantics. Operates on `numtypes` layouts. See [`/doc/topical_audit/spatial_acceleration.md`](/doc/topical_audit/spatial_acceleration.md).*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 6 | Module declarations: aabb, buckets. |
| `src/aabb.rs` | 155 | `fit_aabb(pos, ids)`, `fit_group_aabbs(pos, RaggedIndex, out)`, `fit_range_aabbs(pos, ranges, out)`. **`broad_phase_pairs(cluster_aabbs, margin)`** — O(N²) over clusters, returns overlapping `(i,j)` pairs. **`aabb_edges(bb)`** — 12 edge segments for line rendering. Uses `numtypes::Aabb3d` and `numtypes::RaggedIndex`. |
| `src/buckets.rs` | 95 | `Buckets` — spatial hashing via count→prefix→scatter (FireCore `Buckets.h` pattern). `build(cell_of_obj)`; `cell_objects(c)`. Single `counts` buffer doubles as cursor; no extra allocation during rebuild. |

### `molff` (`crates/libs/molff/`, 2900 LOC)
*Intra-molecular forcefields. See `notes/designs/` for forcefield data ownership. See `/doc/topical_audit/raff.md` for RAFF cross-implementation map and `/doc/topical_audit/spatial_acceleration.md` for broad-phase collision.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 6 | Crate root: `pub mod uff; nonbonded; rigid_sp3; raff; multigrid;` |
| `src/uff.rs` | 665 | `Uff` — bonded forcefield. `Buckets` (spatial partition for force assembly). SoA `AlignedVec` arrays. `eval_atom_bonds()`, `eval_angle_prokop()`, `eval_dihedral_prokop()` (Fourier series via `Vec2d::mul_cmplx`). |
| `src/nonbonded.rs` | 400 | `NonBondedFF` — LJ + Coulomb + H-bond. `reqs`, `plqs`, `make_second_neighs()` (1-2 + 1-3 exclusion), `make_pbc_shifts()`. `eval()` / `eval_pbc()` — O(N²) with exclusion skip. **`BroadPhase`** struct (cluster ranges + AABB cache + rcut). **`eval_broad()`** — AABB-culled eval, identical results to `eval()`. |
| `src/rigid_sp3.rs` | 237 | `RigidSp3FF` — **legacy** port-based rigid body FF (single variant: Dynamic+ForceMD). Superseded by `raff.rs`. |
| `src/raff.rs` | 1550 | **RAFF** — multi-variant port-based rigid-atom FF. `RaffTopology`/`RaffState`/`RaffConfig`/`NbConfig`/`BoxCfg`/`PosSolver`/`FireState`. Port forces, Wahba/Horn rotation solver, `step_force_md`, `step_inertial_reset`, `step_fire`, `step_position_based` (dispatches to `PbdCompliance`/`Xpbd`/`Projective`), `step_proximal`, `solve_collisions`, `eval_nonbonded`, **`eval_nonbonded_broad`**, **`eval_box_forces`** (harmonic AABB constraint), `kabsch_rmsd`, FD checks. See `/doc/topical_audit/raff.md` and `/userguide/raff.md`. |
| `src/multigrid.rs` | ~500 | **Multigrid V-cycle solver** for truss-elasticity (bond-stretch Hessian). `TrussOp` (matrix-free matvec, diagonal blocks), `jacobi_smooth` (damped block Jacobi), `select_pivots_maximin` + `build_pivot_prolongation` (geometric coarse basis), `galerkin_coarse` + `solve_two_grid`/`solve_multigrid` (V-cycle), `dense_solve` (test reference). Parity with NumericalMathPlayground `LinarElasticity/`. See `/doc/topical_audit/multigrid.md`. |
| `src/bin/raff_bench.rs` | 185 | **Benchmark binary** — parameter sweep of all 3 position-based solvers + force-MD. Reports n_steps, n_port_evals, t_wall_us. Run: `cargo run --release -p molff --bin raff_bench`. |
| `tests/test_rigid_sp3.rs` | 110 | Tetrahedral sp3 center (CH4-like) + water test. |
| `tests/test_raff.rs` | 607 | 22 tests: port forces, rotation convergence, energy/momentum conservation, XPBD constraints, collisions, adiabatic torque residual. All passing. |
| `tests/test_raff_convergence.rs` | 216 | 4 tests: force-MD + all 3 position-based solvers converge to same geometry (Kabsch RMSD < 1e-3). Kabsch invariants. chain4 dihedral null space. All passing. |
| `tests/test_broad_phase.rs` | 177 | **3 parity tests**: broad-phase vs O(N²) for `NonBondedFF::eval_broad` and `raff::eval_nonbonded_broad`. Near/far molecule configurations. All passing. |
| `tests/test_multigrid.rs` | ~200 | **4 tests**: T1 matvec parity (vs dense), T2 diagonal-block parity, T3 direct-solve parity (MG vs Gaussian elimination), T4 convergence vs Jacobi (8×8 grid, 4.7× speedup). All passing. See `/doc/topical_audit/multigrid.md`. |
| `tests/test_multigrid_molecules.rs` | ~165 | **3 cantilever benchmarks**: pentacene (rigid stick), n-hexadecane (flexible rope), DiTriptyceno-helicene (branching I-beam). Compares direct vs Jacobi vs MG (manual + automatic pivots). All passing. Linear V-cycle retained as diagnostic; modal approach is the primary strategy — see `/notes/reports/2026-08-29_multigrid_consolidated_report.md`. |

### `surfff` (`crates/libs/surfff/`, 512 LOC)
*Surface interaction forcefield — CPU reference for FAF (folded atomic forcefield).*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 512 | `SurfaceFolded` — separable tensor-product basis (Fourier in x/y, exp decay in z). Complex recurrence for harmonics. `eval_atom_scratch()`, `eval_all_clamped()`. `setup_nacl_surface()`. `SurfaceScratch`. Unit tests: harmonics recurrence, constant/cos/z-decay basis, req2plq. |
| `tests/test_surface.rs` | 188 | Surface eval tests + SVG plot generator (pure Rust, no deps). |

### `oclff` (`crates/libs/oclff/`, ~3600 LOC)
*OpenCL GPU harness for all forcefields: UFF, SPFF, RAFF/RRsp3, GridFF, FAF. Consolidated from `molff-ocl` + `surfff-ocl`. Uses the macro assembler to compose kernel variants from fragment libraries — see `opencl/README.md`, `doc/topical_audit/gridff_faf.md`, and [`/crates/libs/oclff/README.md`](/crates/libs/oclff/README.md) for full API docs and open issues.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | ~40 | Crate root. `nvidia_proque()` — builds `ProQue` on NVIDIA platform. Re-exports `assemble`, `pack`, `uff`, `spff`, `rrsp3`, `surfff` modules. |
| `src/assemble.rs` | ~240 | **`ClAssembler`** / **`ClLibrary`** / **`Substitutions`** — Rust port of SPAMMM `OpenCLBase.preprocess_opencl_source` / `parse_cl_lib`. Parses `//>>>function` / `//>>>macro` blocks from `.cl` libraries. Injects fragments via `//<<<file`, `//<<<macro`, `//<<<function` sentinels. Unit tests for parse + assemble + function-name substitution. |
| `src/pack.rs` | ~600 | `PackedSystem`, `pack_molecules`, `MolInput`, `build_neighs_from_bonds`, `make_exclusions_1st_2nd`, `make_bk_slots_clustered`, `make_ports_from_neighs`, `make_h2o_geometry`, `masses_from_elems`. Cluster-sorted GPU layout (Axis 4b). |
| `src/uff.rs` | ~150 | `UffOcl` — UFF OpenCL harness. Loads `UFF.cl`, dispatches `eval_bonds`. Parity tested vs CPU. |
| `src/spff.rs` | ~50 | `SpffOcl` — SPFFsp3 OpenCL harness. Loads `common.cl`+`Forces.cl`+`SPFF.cl`. Compile smoke only. |
| `src/rrsp3.rs` | ~1000 | **`RRsp3`** — RAFF/RRsp3 OpenCL harness. `PortKernel` (Current/Orig/Substep/Shapematch/Eigen), `StepConfig`, `step_cluster` (relaxation), `step_dynamics` (MD), upload/download for state/neighs/excl/ports/bk_slots. 14 kernels from `opencl/RRsp3.cl`. **CPU↔GPU parity verified** (memoryless + massfull). |
| `src/surfff.rs` | ~110 | `GridFFBuildOcl`, `GridFFEvalOcl`, `FafBuildOcl`, `FafEvalOcl` — OpenCL harnesses for GridFF/FAF. Uses `ClAssembler` to compose programs from fragment libraries. |
| `src/bin/raff_ocl_smoke.rs` | ~340 | CLI smoke test: 2×H₂O, single step, perturbed relaxation, optional CPU↔GPU parity (Kabsch RMSD). Saves to `debug/raff_ocl_smoke/`. |
| `src/bin/raff_ocl_xylitol.rs` | ~220 | CLI smoke test: 4×xylitol (21 atoms, 10 nodes), force-based convergence, trajectory + TSV. Saves to `debug/raff_ocl_xylitol/`. |
| `tests/test_uff_cl.rs` | ~56 | UFF OpenCL parity test vs `molff::uff::Uff` CPU reference. |
| `tests/test_spff_cl.rs` | ~12 | SPFFsp3 OpenCL compile smoke test. |
| `tests/test_assemble_fragments.rs` | ~90 | **5 tests**: parse `gridff_build.cl` (21 functions), parse `gridff_eval.cl` (7 macros), parse `faf_build.cl` (10 functions), parse `faf_eval.cl` (5 macros), assemble + inject `SAMPLE_3D` macro into test kernel. All passing. |

### `surfmol` (`crates/libs/surfmol/`, 155 LOC)
*Integration engine: `MolWorld` orchestrator.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 6 | Crate root: `pub mod mol_world; import;` |
| `src/mol_world.rs` | 165 | `MolWorld` — orchestrator. `BondedFFMode::{Uff, RigidSp3, Raff}`. Owns `DynamicAtoms`, `Uff`, `RigidSp3FF`, optional `NonBondedFF`, optional `SurfaceFolded`. `eval_forces()`, **`eval_forces_broad(bp)`** (AABB-culled), `run_md()`, `move_atom_md()`. **`setup_uff_params(params, types)`** — fills all 4 UFF param arrays from `.dat` files + UFF formulas (ports FireCore `assignUFFparams`). **`set_dummy_params()`** — bond-only testing setup. |
| `src/import.rs` | 13 | `load_topology_from_json()` → `(Uff, Vec<String>)`. **Note:** returns `Uff` with zero params — caller must call `setup_uff_params` before `eval_forces`. |

### `molrender` (`crates/libs/molrender/`, 939 LOC)
*wgpu rendering primitives. No simulation logic. WGSL shaders inline.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 154 | Crate root + `ThumbnailRenderer` — offscreen wgpu renderer. Auto-fitting ortho camera. RGBA readback. |
| `src/impostor.rs` | 363 | `ImpostorRenderer` — raytraced sphere impostors (WGSL inline). `AtomInstance`, `CameraData`. Uses `numtypes::Vec3f`/`Mat4f` for camera math. |
| `src/line_renderer.rs` | 191 | `LineRenderer` — line segments (WGSL inline). `LineVertex`. |
| `src/surface_renderer.rs` | 229 | `SurfaceRenderer` — textured quad for surface potential (WGSL inline). |
| `tests/debug_simple.rs` | — | Single-atom render debug. |
| `tests/debug_single.rs` | — | Single-atom render with full params. |
| `tests/debug_eico.rs` | 69 | Eicosanediol thumbnail debug. |
| `tests/impostor_single.rs` | 112 | Direct ImpostorRenderer test (known GPU headless failure). |
| `tests/render_all.rs` | 74 | Render all XYZ thumbnails. |
| `tests/render_thumbs.rs` | 69 | Sample thumbnail render. |

### `molgui` (`crates/libs/molgui/`, 795 LOC)
*GUI support: trackball camera, thumbnailer, Kekule editor, clipboard.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 1 | Crate root: `pub mod gui;` |
| `src/gui/mod.rs` | 4 | `pub mod gizmos; kekule_editor; thumbnailer; trackball; clipboard;` |
| `src/gui/gizmos.rs` | — | `make_bond_segments()` — multi-segment bond line generation. |
| `src/gui/kekule_editor.rs` | 338 | `KekuleEditor` — hex-grid molecular editor. `EditMode`, `collect_hex_grid_points()`, `collect_builder_bonds/atoms()`, `export_xyz()`, `element_color()`. |
| `src/gui/thumbnailer.rs` | 234 | `MolThumbnailer` — wraps `ImpostorRenderer` + `LineRenderer` for egui thumbnail textures. PCA alignment via `numcore::math::linalg::symmetric_eigen_3x3`. |
| `src/gui/trackball.rs` | 61 | `TrackballCam` — orbit camera (target, rotation Quat, zoom, lerp). **Column-major orthographic projection** (fixed fisheye bug 2026-09-28). |
| `src/gui/clipboard.rs` | — | `Clipboard` — arboard wrapper. `inject_cut_copy_if_needed`, `inject_paste_if_needed`, `handle_output_commands`. Replaces egui-winit's clipboard feature. |
| `tests/test_thumb.rs` | — | MolThumbnailer integration test (saves PNG via `image` crate). |

### `buildff` (`crates/apps/buildff/`, 240 LOC)
*CLI builder: XYZ → topology → UFF type assignment → `TopologyData` JSON or binary export. Stateless, no forcefield evaluation. Consumed by `molengine`.*

| File | LOC | Contents |
|------|-----|----------|
| `src/main.rs` | 240 | Reads XYZ → builds topology → assigns UFF types → outputs `TopologyData` JSON (via `Topology::export_json` — flat arrays, canonical format shared with `molengine`) and/or binary (`UFFTOPO` magic header + flat arrays). Flags: `--json`, `--bin`, `--tol`, `--rcut`. UFF type histogram printed to stdout. |

### `molengine` (`crates/apps/molengine/`, 410 LOC)
*CLI runner: Rhai-scripted MD/relaxation engine. Loads `TopologyData` JSON, runs UFF and RAFF forcefield evaluation. Supports all 6 RAFF solver modes (ForceMD, InertialReset, FIRE, PBD, XPBD, Projective), harmonic box constraint, non-bonded LJ+Coulomb.*

| File | LOC | Contents |
|------|-----|----------|
| `src/main.rs` | 410 | Rhai-scripted MD/relaxation. `SimulationEngine` wraps `MolWorld` + optional RAFF state (`RaffState`/`RaffTopology`/`RaffConfig`/`NbConfig`/`FireState`) in `Arc<Mutex>`. **UFF API:** `load_topology`, `setup_uff_params`, `eval_forces`, `step_md`, `relax`, `get_natoms`. **RAFF API:** `setup_raff` (builds RaffTopology from UFF bonds + ARAP port geometry), `set_raff_solver`/`set_raff_orient`/`set_raff_dt`/`set_raff_damping`/`set_raff_iters`/`set_raff_hb`/`set_raff_pd_inertia`/`set_raff_vel_reset`/`set_raff_box`/`set_raff_nb`/`set_raff_charges`, `raff_step`/`raff_relax`/`get_raff_energy`/`get_raff_pos`/`save_raff_xyz`. `build_raff_from_world` helper mirrors editor's pipeline. See [`/userguide/raff.md`](/userguide/raff.md) §CLI usage. |

### `editor` (`crates/apps/editor/`, 1700 LOC)
*Interactive molecular editor and on-surface MD simulator. wgpu + egui + winit. Supports UFF / RigidSp3 / RAFF forcefield modes with 6 RAFF solver modes (ForceMD, InertialReset, FIRE, PBD, XPBD, Projective). See [`/userguide/editor.md`](/userguide/editor.md) and [`/userguide/raff.md`](/userguide/raff.md) for end-user guides.*

| File | LOC | Contents |
|------|-----|----------|
| `src/main.rs` | 1700 | 3D molecular editor. TrackballCam, atom picking (ray-sphere), bond/port drawing, hex-grid Kekule editing, MD relaxation (Uff/RigidSp3/Raff + NonBonded + NaCl surface), RAFF integration with all 6 solver modes (`RaffSolverMode` enum, `do_raff_step` dispatches to `step_force_md`/`step_inertial_reset`/`step_fire`/`step_position_based`), harmonic box constraint (`BoxCfg`, `--box` CLI), spring drag, 2D constraint, pinning, port sync, surface potential visualization, atom-scale CLI/GUI, clipboard support. **Multi-molecule loading** (`--nmols N`, `--layout lattice\|random`), **AABB broad-phase collision** (`BroadPhase` struct, `eval_forces_broad`/`eval_nonbonded_broad`), **AABB visualization** (`--show-aabb`). CLI: `--raff`, `--raff-solver`, `--raff-orient`, `--raff-iters`, `--raff-pd-inertia`, `--raff-vel-reset`, `--raff-hb`, `--box`, `--box-min`, `--box-max`, `--box-k`, `--2d`, `--atom-scale`, `--nmols`, `--layout`, `--show-aabb`, `--perFrame`, `--dt`. |

### `molbrowser` (`crates/apps/molbrowser/`, 250 LOC)
*Gallery browser for XYZ molecule files. eframe (egui).*

| File | LOC | Contents |
|------|-----|----------|
| `src/main.rs` | 250 | `MolBrowserApp` — XYZ directory browser with egui thumbnail grid. Batched GPU thumbnail generation with PCA alignment, responsive grid layout, incremental loading. |

## 4. OpenCL kernels (`opencl/`)

Ported from FireCore / SPAMMM. See `opencl/README.md` and `doc/topical_audit/gridff_faf.md` for the macro-fragment architecture.

**Macro-fragment principle:** GridFF and FAF are not standalone programs. They are `//>>>function` / `//>>>macro` fragment libraries. Build fragments assemble into construction programs; eval fragments are injected via `//<<<macro NAME` into the `getNonBonded` loop of UFF/SPFF/RAFF/RigidMolFF so all forcefields share one NBFF primitive. This avoids combinatoric explosion: instead of N forcefields × M surface variants = N×M kernel files, we have N + M fragments composed at compile time by `oclff::ClAssembler`.

| File | Size | Purpose | Origin |
|------|------|---------|--------|
| `UFF.cl` | 108 KB | UFF force evaluation (bonds, angles, dihedrals, inversions) | FireCore |
| `SPFF.cl` | — | SPFFsp3 force field = FireCore MMFFsp3 | SPAMMM |
| `common.cl` | — | Shared types/macros/helpers (concatenated FIRST before SPAMMM modular kernels) | SPAMMM |
| `Forces.cl` | — | Inline pairwise potentials (`getLJQH`, `getMorseQH`, `getCoulomb`) | SPAMMM |
| `RRsp3.cl` | — | RAFF/RRsp3 rigid-atom forcefield kernels | FireCore |
| `relax_multi.cl` | 284 KB | Unified multi-system force eval + bucket neighbor search | FireCore |
| `relax_multi_mini.cl` | 186 KB | Minimal variant of relax_multi | FireCore |
| `Rigid.cl` | 17 KB | Rigid body dynamics kernels | FireCore/SPAMMM |
| `gridff_spammm.cl` | 2106 lines | SPAMMM GridFF — canonical whole-file reference (B-spline, Poisson, make_GridFF, sampleGridFF). **Requires** `common.cl`+`Forces.cl` first. | SPAMMM |
| `surface_spammm.cl` | 1867 lines | SPAMMM surface — canonical whole-file reference (FAF, Ewald2D, isosurfaces). **Requires** `common.cl`+`Forces.cl` first. | SPAMMM |
| `gridff_build.cl` | 1063 lines | **Fragment library** — 21 `//>>>function` blocks: utility + build kernels (make_MorseFF, poissonW, project_*, make_GridFF). Extracted from `gridff_spammm.cl`. | SurfMol |
| `gridff_eval.cl` | 626 lines | **Macro library** — 7 `//>>>macro` blocks (SAMPLE_3D, SAMPLE_3D_GRID, SAMPLE_GRIDFF_BSPLINE_POINTS, etc.) + helper inline functions. Injected into `getNonBonded` via `//<<<macro`. Extracted from `gridff_spammm.cl`. | SurfMol |
| `faf_build.cl` | 715 lines | **Fragment library** — 10 `//>>>function` blocks (getSurfMorse, eval_potential_*, compute_ewald_coefficients, getSurfaceIso*). Extracted from `surface_spammm.cl`. | SurfMol |
| `faf_eval.cl` | 762 lines | **Macro library** — 5 `//>>>macro` blocks (GET_SURF_FOLDED, GET_SURF_FOLDED_WORKGROUP, GET_SURF_FOLDED_HARMONICS, GET_SURF_FOLDED_TENSOR_EXP, GET_SURF_FOLDED_TENSOR_POLY) + helper inline functions. Injected into `getNonBonded` via `//<<<macro`. Extracted from `surface_spammm.cl`. | SurfMol |
| `grids.cl` | — | Grid utilities (lattice helpers, index math) | SPAMMM |
| `PME.cl` / `PME8.cl` | — | Particle-mesh Ewald solvers | SPAMMM |
| `contact_surface.cl` | — | Quasi-2D contact surface | SPAMMM |
| `Assembly.cl` | 7 KB | Rigid-body assembly / packing / clash | SPAMMM |
| `multigrid.cl` | — | Multigrid restriction/prolongation/coarse solve | NumericalMathPlayground |
| `block_jacobi.cl` | — | Block Jacobi smoother for truss/bond stiffness | NumericalMathPlayground |

**Rust OpenCL harness:** `oclff` crate (`crates/libs/oclff/`). `ClAssembler` parses fragment libraries and composes kernel variants at runtime. See `doc/topical_audit/gridff_faf.md` for the build/eval split and macro-injection contract.

## 5. Data files (`data/`)

| File | Contents |
|------|----------|
| `ElementTypes.dat` | Element params: Z, valence, color, r_cov, r_vdw, e_vdw, q_uff, QEq params. |
| `AtomTypes.dat` | Atom type params: UFF types with r_uff, r_vdw, e_vdw, q_base, h_b, MMFF params. |
| `BondTypes.dat` | Bond params: atom type pair, order, l0, k. |
| `AngleTypes.dat` | Angle params: atom type triple, a0, k. |
| `DihedralTypes.dat` | Dihedral params: atom type quad, order, k, a0, n. |
| `mol/` | Molecules in `.mol` / `.mol2` (benzene, cubane, adamantane, ...). |
| `xyz/` | Molecules in `.xyz`. |

## 6. Build & test commands

All commands run from repo root (`/home/prokop/git/SurfMol`):

```bash
# Build
cargo build                              # build all crates
cargo build --release                    # release build

# Run binaries
cargo run -p editor                      # 3D molecular editor
cargo run -p molbrowser                  # XYZ directory browser
cargo run -p buildff -- <xyz> --json out.json --bin out.bin  # topology CLI
cargo run -p molengine -- --script examples/relax.rhai       # Rhai MD engine

# Test
cargo test                               # all tests
cargo test -p molff                      # forcefield tests
cargo test -p molrender                  # render tests
cargo test -p numtypes -p pgraph -p spacc  # data-layout + graph/spatial tests

# Check
cargo check --workspace                  # fast type check
cargo clippy --workspace                 # lints
```

**Test data paths:** tests use `CARGO_MANIFEST_DIR` + `../../..` to reach `data/` from crate dirs.

## 7. Key data structures (cross-crate)

| Struct | Crate | Role |
|--------|-------|------|
| `Vec2d`, `Vec3d`, `Vec4d`, `Quat4d`, `Quat4i`, `Vec6d` | numtypes | `#[repr(C)]` math primitives; `Vec2d` as complex, `Vec4d` as quaternion, `Vec6d` for AABB/symmetric 3×3 |
| `Mat3d`, `Mat4d` | numtypes | `#[repr(C)]` rows-of-vectors matrices; `rows()`/`array()` zero-copy views |
| `AlignedVec<T, 64>` | numtypes | 64-byte aligned allocator for SIMD-friendly arrays |
| `PGraph`, `PGraphView` | numtypes | Positioned graph: pos + edges (domain-agnostic) |
| `FixedAdj<K>`, `CsrAdj` | numtypes | Adjacency representations (ELL-like / CSR) |
| `RaggedIndex`, `Partition`, `RangeGroups` | numtypes | Group/partition representations |
| `Permutation` | numtypes | Bidirectional index remapping |
| `Aabb3d`, `Aabb3f` | numtypes | `Vec6` aliases for AABB; `aabb_*` intrinsic functions |
| `symmetric_eigen_3x3` | numcore | Analytical 3×3 symmetric eigendecomposition (replaces nalgebra) |
| `cholesky_factor_f64`, `cholesky_solve_f64`, `dense_matvec_f64` | numcore | f64 dense Cholesky factor + solve + matvec (for multigrid coarse solve) |
| `Atoms` | moltopo | Static atomic data (apos, atypes, neighs, neigh_bs) |
| `DynamicAtoms` | moltopo | Atoms + fapos + vapos + MD integrators. **Single owner of per-atom state.** |
| `Topology` | moltopo | Flat arrays: apos, bonds, angles, dihedrals, inversions |
| `Builder` | moltopo | Slot-based graph with generational handles (AtomH, BondH), hex-grid editing |
| `Params` | moltopo | Loaded FF parameter tables |
| `Uff` | molff | Bonded FF: SoA arrays, Buckets force assembly, hneigh |
| `RigidSp3FF` | molff | **Legacy** port-based rigid body: quat, omega, tau. Single variant (Dynamic+ForceMD). |
| `RAFF` | molff | **Multi-variant** port-based rigid-atom FF: `RaffTopology`/`RaffState`/`RaffConfig`/`BoxCfg`/`PosSolver`/`FireState`, port forces, Wahba/Horn rotation, `step_force_md`/`step_inertial_reset`/`step_fire`/`step_position_based` (PBD/XPBD/Projective)/`step_proximal`, `eval_nonbonded`/`eval_nonbonded_broad`, `eval_box_forces` (harmonic AABB constraint), `solve_collisions`, `kabsch_rmsd`. 34 tests + benchmark binary. See `/doc/topical_audit/raff.md` and `/userguide/raff.md`. |
| `NonBondedFF` | molff | LJ+Coulomb+Hbond: reqs, plqs, excl, PBC shifts. `eval_broad()` for AABB-culled eval. |
| `BroadPhase` | molff | Per-cluster AABB broad-phase collision: cluster ranges + rebuildable AABB cache + rcut. Used by `eval_broad` / `eval_nonbonded_broad`. |
| `TrussOp` | molff | Matrix-free truss operator for multigrid: bonds (ei, ej, k_eff, n_dirs) + mass_dt2. `matvec`, `diagonal_blocks`, `assemble_dense`. See `/doc/topical_audit/multigrid.md`. |
| `SurfaceFolded` | surfff | Separable Fourier basis surface potential |
| `MolWorld` | surfmol | Coordinator: DynamicAtoms + Uff + RigidSp3FF + optional NonBondedFF/SurfaceFolded |
| `Buckets` | spacc | Spatial hashing (count→prefix→scatter); replaces `molff::uff::Buckets` long-term |
| `AtomInstance`, `CameraData` | molrender | GPU vertex/uniform layouts (match WGSL structs) |
| `ImpostorRenderer`, `LineRenderer`, `SurfaceRenderer` | molrender | wgpu render pipelines |
| `TrackballCam`, `KekuleEditor`, `MolThumbnailer`, `Clipboard` | molgui | GUI utilities |

## 8. Parity references (C++ / Python)

Each ported module cites its reference (see `AGENTS.md` §Rule 6 — Parity Work):

| SurfMol module | Reference | File |
|----------------|-----------|------|
| `Uff` | FireCore C++ | `cpp/common/molecular/UFF.h` |
| `NonBondedFF` | FireCore C++ | `cpp/common/molecular/NBFF.h` |
| `RigidSp3FF` | FireCore Python+OpenCL | `pyBall/RigidAtomFF/RRsp3/` |
| `SurfaceFolded` | SPAMMM Python | `kernels/surface.cl`, `spammm/surfaces/` |
| `Builder` hex grid | SPAMM Python | `KekuleBackend.py` |
| `Topology` diagnostic prints | FireCore C++ | `MMFFBuilderBase.h` |
| `Params` diagnostic prints | FireCore C++ | `MMFFparams.h` |
| `symmetric_eigen_3x3` | FireCore C++ | `Mat3.h:Mat3T::eigenvals()` + `eigenvec()` |
| `pgraph_ops::build_csr_adj` | FireCore C++ | `MolecularGraph.h::makeNeighbors()` |
| `pgraph_ops::find_bridges` | FireCore C++ | `MolecularGraph.h::findBridges()` |
| `pgraph_ops::partition_to_index_groups` | FireCore C++ | `Groups::setGroupMapping()` |
| `spacc::Buckets` | FireCore/SSE C++ | `Buckets.h` |
| `spacc::fit_group_aabbs` | FireCore C++ | `NBFF::initBBsFromGroups()` |
| `spacc::broad_phase_pairs` | FireCore C++ | `NBFF::evalSortRange_BBs()` |
| `numtypes::aabb_overlap_margin` | FireCore OpenCL | `RRsp3.cl:123-128` (`bboxes_overlap` with margin) |
| `numtypes::aabb_sphere_overlap` | FireCore Python | `Grid_dftb.py:240-244` (point-to-AABB distance) |
| `molff::BroadPhase` + `eval_broad` | FireCore C++ | `NBFF.h` bucket-pair broad phase + narrow phase |
| `molff::multigrid` (TrussOp, V-cycle, prolongation) | NumericalMathPlayground Python+OpenCL | `topics/LinarElasticity/MultiGrid.py`, `TrussSolver.py`, `kernels_multigrid.cl`, `kernels_block_jacobi.cl` |
| `numcore::cholesky_factor_f64` / `cholesky_solve_f64` | NumericalMathPlayground Python | `topics/LinarElasticity/MultiGrid.py` (numpy `np.linalg.cholesky`) |

See `Import_other_Repos.md` for the full cross-repo import map.

## 9. What's NOT yet implemented

- **OpenCL integration:** `oclff` crate is the consolidated OpenCL harness (UFF, SPFF, RAFF/RRsp3, GridFF, FAF). `ClAssembler` parses `//>>>function`/`//>>>macro` fragment libraries and composes kernel variants at runtime. GridFF/FAF build/eval fragments extracted from SPAMMM sources. CPU Rust references in `molff`/`surfff` remain authoritative for correctness. **RAFF/RRsp3 GPU harness implemented**: `RRsp3` struct with 5 port kernel variants (Current/Orig/Substep/Shapematch/Eigen), `step_cluster` (relaxation) + `step_dynamics` (MD), 14 OpenCL kernels from `RRsp3.cl`. CPU↔GPU parity verified (memoryless: Adiabatic vs Shapematch RMSD=0.0096Å; massfull: Dynamic vs Current RMSD=0.000000Å). Smoke tests: 2×water + 4×xylitol. See [`/crates/libs/oclff/README.md`](/crates/libs/oclff/README.md) for open issues.
- **RAFF (RigidAtomFF):** `RigidSp3FF` is the precursor; full RAFF implemented with 6 solver modes (ForceMD, InertialReset, FIRE, PBD, XPBD, Projective), 2 orientation strategies (Adiabatic, Dynamic), harmonic box constraint, heavy-ball momentum, and inner-coupled rotation. All wired to editor via GUI + CLI. See [`/userguide/raff.md`](/userguide/raff.md) and [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md).
- **Projective / Position-Based Dynamics:** **Implemented** — `PosSolver::{PbdCompliance, Xpbd, Projective}` via `step_position_based`. Projective uses inner-coupled rotation (both x and q updated each inner Jacobi iteration). All 3 variants converge to same geometry (Kabsch RMSD < 1e-3). Wired to editor. See [`/userguide/raff.md`](/userguide/raff.md).
- **AABB collision acceleration in NonBondedFF:** **Implemented** (2026-09-29). `BroadPhase` struct + `eval_broad` / `eval_nonbonded_broad` + `MolWorld::eval_forces_broad`. Parity tests pass. See [`/doc/topical_audit/spatial_acceleration.md`](/doc/topical_audit/spatial_acceleration.md) and [`/notes/designs/cluster_aabb_collision.md`](/notes/designs/cluster_aabb_collision.md). PBC + broad phase not yet supported.
- **Global optimization (GOpt):** not yet implemented.
- **NPZ format:** `buildff` output and `molengine` input planned to support NPZ (currently JSON only).
- **`pgraph_ops` P2 modules:** `loops.rs` (cycle/ring detection), `selection.rs` (SDF selection), `picking.rs` (ray picking), `edit.rs` (editing helpers) not yet implemented.
- **Multigrid / modal relaxation:** Rust CPU V-cycle implemented in `molff::multigrid` + tested (parity + convergence). Linear V-cycle underperforms end-to-end (dominated by fine-level work). **Modal coarse-graining achieves 53× speedup** on pentacene via fitted Newton + timestep scaling. Two approaches designed: (A) fitted modal [verified], (B) force-projection Galerkin V-shape [not yet implemented]. OpenCL kernels copied but not wired. See `/doc/topical_audit/multigrid.md` and `/notes/reports/2026-08-29_multigrid_consolidated_report.md`.
- **`spacc` P1 modules:** `uniform_grid.rs`, `morton.rs` not yet implemented.
- **`moltopo` migration to `pgraph`:** `moltopo` still uses its own `Topology`/`Builder` structs; planned to build on `pgraph`/`pgraph_ops` per `notes/designs/topology_builder.md`.
- **`CODEMAP.md` status:** this file is current as of 2026-08-29.
