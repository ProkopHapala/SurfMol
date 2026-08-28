---
type: developer-docs
title: CODEMAP
description: Repo structure, file inventory, crate dependency graph, pinned dependencies, and build/test commands.
tags: [codemap, navigation, structure, dependencies]
timestamp: 2026-08-25
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
└── userguide/                # End-user docs for finished modules
```

## 2. Rust workspace

**Workspace root:** `Cargo.toml` (14 members, resolver 2).
**Shared target dir:** `CARGO_TARGET_DIR` env var (machine-local, typically `~/.cargo/shared_target`); repo `.cargo/config.toml` falls back to local `target/`.

### Members

```
crates/libs/   (10 library crates, no binary targets)
  numtypes, numcore, moltopo, pgraph, spacc,
  molff, surfff, surfmol, molrender, molgui

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
```

Apps depend on libs:
- `buildff` → moltopo, numcore
- `molengine` → surfmol, molff, moltopo, numcore
- `editor` → surfmol, molff, surfff, moltopo, numcore, molgui, molrender
- `molbrowser` → moltopo, numcore, molgui, molrender

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
| `src/spatial.rs` | ~70 | `Aabb3d`/`Aabb3f` and `SymMat3d`/`SymMat3f` as `Vec6` aliases; standalone `aabb_*` and `sym3_*` intrinsic functions. |

### `numcore` (`crates/libs/numcore/`, ~110 LOC)
*Numerical algorithms. Does **not** re-export `numtypes` data; owns `fastmath` and `linalg` only.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 1 | Crate root: `pub mod math;` |
| `src/math/mod.rs` | 2 | Module declarations: `fastmath`, `linalg`. |
| `src/math/fastmath.rs` | 36 | `sq`, `dangle`, `clamp_abs`, `sincos_taylor2`, `sincos_r2_taylor` (Taylor approximations). |
| `src/math/linalg.rs` | 92 | `symmetric_eigen_3x3` — analytical (closed-form) 3×3 symmetric eigendecomposition. Ported from FireCore `Mat3.h:Mat3T::eigenvals()` + `eigenvec()`. Replaces nalgebra for PCA in thumbnailer. |

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

### `spacc` (`crates/libs/spacc/`, ~220 LOC)
*Spatial acceleration — rebuildable caches, no molecular semantics. Operates on `numtypes` layouts.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 6 | Module declarations: aabb, buckets. |
| `src/aabb.rs` | 96 | `fit_aabb(pos, ids)`, `fit_group_aabbs(pos, RaggedIndex, out)`, `fit_range_aabbs(pos, ranges, out)`. Uses `numtypes::Aabb3d` and `numtypes::RaggedIndex`. |
| `src/buckets.rs` | 95 | `Buckets` — spatial hashing via count→prefix→scatter (FireCore `Buckets.h` pattern). `build(cell_of_obj)`; `cell_objects(c)`. Single `counts` buffer doubles as cursor; no extra allocation during rebuild. |

### `molff` (`crates/libs/molff/`, 2289 LOC)
*Intra-molecular forcefields. See `notes/designs/` for forcefield data ownership. See `/doc/topical_audit/raff.md` for RAFF cross-implementation map.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 6 | Crate root: `pub mod uff; nonbonded; rigid_sp3; raff;` |
| `src/uff.rs` | 665 | `Uff` — bonded forcefield. `Buckets` (spatial partition for force assembly). SoA `AlignedVec` arrays. `eval_atom_bonds()`, `eval_angle_prokop()`, `eval_dihedral_prokop()` (Fourier series via `Vec2d::mul_cmplx`). |
| `src/nonbonded.rs` | 300 | `NonBondedFF` — LJ + Coulomb + H-bond. `reqs`, `plqs`, `make_second_neighs()` (1-2 + 1-3 exclusion), `make_pbc_shifts()`. `eval()` / `eval_pbc()` — O(N²) with exclusion skip. |
| `src/rigid_sp3.rs` | 237 | `RigidSp3FF` — **legacy** port-based rigid body FF (single variant: Dynamic+ForceMD). Superseded by `raff.rs`. |
| `src/raff.rs` | 1085 | **RAFF** — multi-variant port-based rigid-atom FF. `RaffTopology`/`RaffState`/`RaffConfig`/`NbConfig`. Port forces, Wahba/Horn rotation solver, `step_force_md`, `step_xpbd`, `step_proximal`, `solve_collisions`, `eval_nonbonded`, FD checks. See `/doc/topical_audit/raff.md`. |
| `tests/test_rigid_sp3.rs` | 110 | Tetrahedral sp3 center (CH4-like) + water test. |
| `tests/test_raff.rs` | 607 | 22 tests: port forces, rotation convergence, energy/momentum conservation, XPBD constraints, collisions, adiabatic torque residual. All passing. |

### `surfff` (`crates/libs/surfff/`, 512 LOC)
*Surface interaction forcefield.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 512 | `SurfaceFolded` — separable tensor-product basis (Fourier in x/y, exp decay in z). Complex recurrence for harmonics. `eval_atom_scratch()`, `eval_all_clamped()`. `setup_nacl_surface()`. `SurfaceScratch`. Unit tests: harmonics recurrence, constant/cos/z-decay basis, req2plq. |
| `tests/test_surface.rs` | 188 | Surface eval tests + SVG plot generator (pure Rust, no deps). |

### `surfmol` (`crates/libs/surfmol/`, 155 LOC)
*Integration engine: `MolWorld` orchestrator.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 6 | Crate root: `pub mod mol_world; import;` |
| `src/mol_world.rs` | 140 | `MolWorld` — orchestrator. `BondedFFMode::{Uff, RigidSp3}`. Owns `DynamicAtoms`, `Uff`, `RigidSp3FF`, optional `NonBondedFF`, optional `SurfaceFolded`. `eval_forces()`, `run_md()`, `move_atom_md()`. |
| `src/import.rs` | 13 | `load_topology_from_json()` → `(Uff, Vec<String>)`. |

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

### `buildff` (`crates/apps/buildff/`, 290 LOC)
*CLI tool: XYZ → topology → UFF type assignment → JSON or binary export.*

| File | LOC | Contents |
|------|-----|----------|
| `src/main.rs` | 290 | Reads XYZ → builds topology → assigns UFF types → outputs JSON and/or binary (`UFFTOPO` magic header + flat arrays). Flags: `--json`, `--bin`, `--tol`, `--rcut`. |

### `molengine` (`crates/apps/molengine/`, 90 LOC)
*CLI MD/relaxation engine. Rhai-scripted.*

| File | LOC | Contents |
|------|-----|----------|
| `src/main.rs` | 90 | Rhai-scripted MD/relaxation. Registers `load_topology`, `eval_forces`, `step_md`, `relax`, `get_natoms`. `SimulationEngine` wraps `MolWorld` in `Arc<Mutex>`. |

### `editor` (`crates/apps/editor/`, 1433 LOC)
*Interactive molecular editor and on-surface MD simulator. wgpu + egui + winit. Supports UFF / RigidSp3 / RAFF forcefield modes.*

| File | LOC | Contents |
|------|-----|----------|
| `src/main.rs` | 1433 | 3D molecular editor. TrackballCam, atom picking (ray-sphere), bond/port drawing, hex-grid Kekule editing, MD relaxation (Uff/RigidSp3/Raff + NonBonded + NaCl surface), RAFF integration (`do_raff_step`, spring drag, 2D constraint, stopping criterion, port sync), surface potential visualization, atom-scale CLI/GUI, clipboard support. CLI: `--raff`, `--2d`, `--atom-scale`, `--perFrame`, `--dt`. |

### `molbrowser` (`crates/apps/molbrowser/`, 250 LOC)
*Gallery browser for XYZ molecule files. eframe (egui).*

| File | LOC | Contents |
|------|-----|----------|
| `src/main.rs` | 250 | `MolBrowserApp` — XYZ directory browser with egui thumbnail grid. Batched GPU thumbnail generation with PCA alignment, responsive grid layout, incremental loading. |

## 4. OpenCL kernels (`opencl/`)

Ported from FireCore / SPAMMM. See `opencl/README.md` and `Import_other_Repos.md`.

| File | Size | Purpose | Origin |
|------|------|---------|--------|
| `UFF.cl` | 108 KB | UFF force evaluation (bonds, angles, dihedrals, inversions) | FireCore |
| `relax_multi.cl` | 284 KB | Unified multi-system force eval + bucket neighbor search | FireCore |
| `relax_multi_mini.cl` | 186 KB | Minimal variant of relax_multi | FireCore |
| `Rigid.cl` | 17 KB | Rigid body dynamics kernels | FireCore/SPAMMM |
| `GridFF.cl` | 64 KB | B-spline grid interpolation for substrate potentials | FireCore/SPAMMM |
| `Surface.cl` | 33 KB | Surface interactions (Morse/LJ/Coulomb), Ewald2D | FireCore/SPAMMM |
| `Assembly.cl` | 7 KB | Rigid-body assembly / packing / clash | SPAMMM |

**Note:** These kernels are not yet wired into the Rust crates (no `ocl` usage in any crate yet). The Rust CPU implementations in `molff` are the current authoritative references.

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
| `Atoms` | moltopo | Static atomic data (apos, atypes, neighs, neigh_bs) |
| `DynamicAtoms` | moltopo | Atoms + fapos + vapos + MD integrators. **Single owner of per-atom state.** |
| `Topology` | moltopo | Flat arrays: apos, bonds, angles, dihedrals, inversions |
| `Builder` | moltopo | Slot-based graph with generational handles (AtomH, BondH), hex-grid editing |
| `Params` | moltopo | Loaded FF parameter tables |
| `Uff` | molff | Bonded FF: SoA arrays, Buckets force assembly, hneigh |
| `RigidSp3FF` | molff | **Legacy** port-based rigid body: quat, omega, tau. Single variant (Dynamic+ForceMD). |
| `RAFF` | molff | **Multi-variant** port-based rigid-atom FF: `RaffTopology`/`RaffState`, port forces, Wahba/Horn rotation, `step_force_md`/`step_xpbd`/`step_proximal`, `eval_nonbonded`, `solve_collisions`. 22 tests. See `/doc/topical_audit/raff.md`. |
| `NonBondedFF` | molff | LJ+Coulomb+Hbond: reqs, plqs, excl, PBC shifts |
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

See `Import_other_Repos.md` for the full cross-repo import map.

## 9. What's NOT yet implemented

- **OpenCL integration:** `ocl` 0.19 is declared in workspace deps but no crate uses it yet. CPU Rust is authoritative.
- **RAFF (RigidAtomFF):** `RigidSp3FF` is the precursor; full RAFF not yet implemented. See `DESIGN_GOALS.md` §2.
- **Projective / Position-Based Dynamics:** not yet implemented. See `DESIGN_GOALS.md` §3.
- **AABB collision acceleration in NonBondedFF:** `spacc` provides the structures but `NonBondedFF` is still O(N²). See `DESIGN_GOALS.md` §2.3.
- **Global optimization (GOpt):** not yet implemented.
- **NPZ format:** `buildff` output and `molengine` input planned to support NPZ (currently JSON only).
- **`pgraph_ops` P2 modules:** `loops.rs` (cycle/ring detection), `selection.rs` (SDF selection), `picking.rs` (ray picking), `edit.rs` (editing helpers) not yet implemented.
- **`spacc` P1 modules:** `uniform_grid.rs`, `morton.rs` not yet implemented.
- **`moltopo` migration to `pgraph`:** `moltopo` still uses its own `Topology`/`Builder` structs; planned to build on `pgraph`/`pgraph_ops` per `notes/designs/topology_builder.md`.
- **`CODEMAP.md` status:** this file is current as of 2026-08-25.
