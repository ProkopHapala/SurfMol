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
├── .gitignore
├── data/                     # Molecular inputs (.xyz, .mol, .mol2) + FF params (.dat)
├── debug/                    # Diagnostic plots (gitignored except README)
├── doc/                      # Permanent developer docs + topical_audit/
├── notes/                    # Ephemeral work-in-progress (chats, designs, labbooks, reports, tasks, TODOs)
├── opencl/                   # OpenCL .cl kernel sources
├── rust/                     # Primary Rust workspace (5 crates)
└── userguide/                # End-user docs for finished modules
```

## 2. Rust workspace

**Workspace root:** `rust/Cargo.toml` (5 members, resolver 2).
**Shared target dir:** `rust/.cargo/config.toml` → `target-dir = "../../target"` (i.e. `/home/prokop/git/SurfMol/target`).

### Crate dependency graph

```
surfmol-common  (no internal deps; bytemuck)
      ↑
surfmol-topology  (→ common; serde, serde_json)
      ↑
surfmol-forcefields  (→ common, topology; rhai, clap, serde, serde_json, ndarray)
      ↑
surfmol-molrender  (→ common, topology; wgpu, bytemuck, pollster, nalgebra)
      ↑
surfmol-apps  (→ forcefields, common, topology, molrender; wgpu, winit, eframe, egui, egui-winit, egui-wgpu, glam, nalgebra, image)
```

### Workspace dependencies (`rust/Cargo.toml`)

| Crate | Version | Notes |
|-------|---------|-------|
| `eframe` | 0.29 | `default-features=false`, `features=["default_fonts","glow"]` |
| `egui` | 0.34 | Immediate-mode GUI |
| `egui-winit` | 0.34 | egui ↔ winit bridge |
| `egui-wgpu` | 0.34 | egui ↔ wgpu backend |
| `egui_plot` | 0.34 | Plotting (declared, not yet used) |
| `ndarray` | 0.16 | N-d arrays (forcefields) |
| `nalgebra` | 0.33 | Linear algebra (molrender, apps) |
| `rand` | 0.8 | RNG |
| `ocl` | 0.19 | **OpenCL bindings — chosen crate** (see `DESIGN_GOALS.md` §10) |
| `wgpu` | 29 | GPU compute/graphics |
| `pollster` | 0.4 | Async runtime (block_on) |
| `bytemuck` | 1.21 | `features=["derive"]` — zero-cost casts for GPU buffers |
| `winit` | 0.30 | Windowing |
| `glam` | 0.29 | Math (apps: Quat, Vec3) |

### Per-crate extra deps

| Crate | Extra deps |
|-------|------------|
| `surfmol-topology` | `serde 1.0` (derive), `serde_json 1.0` |
| `surfmol-forcefields` | `rhai 1.19` (scripting), `clap 4.5` (derive), `serde 1.0`, `serde_json 1.0`, `ndarray` (workspace) |
| `surfmol-molrender` | `image 0.25` (dev-dep, `default-features=false`, `features=["png"]`) |
| `surfmol-apps` | `egui_extras 0.29`, `glam` (workspace), `nalgebra` (workspace), `image 0.25` |

## 3. File inventory by crate

### `surfmol-common` (`rust/common/`, 8128 LOC total workspace)
*Bedrock: math + data structures. No chemistry/physics. `#[repr(C)]`, 64-byte aligned.*

| File | LOC | Contents |
|------|-----|----------|
| `src/common.rs` | 4 | Crate root: `pub mod math; pub mod util; pub mod xyz; pub mod molecular;` |
| `src/util.rs` | 60 | `AlignedVec<T, A>` — cache-aligned allocator (64-byte), `as_slice`/`as_mut_slice`/`resize_fill`. `unsafe` for raw alloc. |
| `src/xyz.rs` | 48 | `XyzSystem`, `read_xyz()`, `write_xyz_frame()` — XYZ file I/O. |
| `src/molecular.rs` | 165 | `Atoms` (static: natoms, atypes, apos, neighs, neigh_bs; `make_neigh_bs()`), `DynamicAtoms` (Atoms + fapos + vapos; `move_atom_md()`, `run_md()`, `clean_force/velocity`). **SSOT for atomic state.** |
| `src/math/mod.rs` | 6 | Module declarations. |
| `src/math/vec3.rs` | 43 | `Vec3d` (`#[repr(C)]`, f64), ops (add/sub/mul/dot/cross/norm/normalize), `VEC3_ZERO`, `VEC3_NAN`. |
| `src/math/vec2.rs` | — | `Vec2d` with `mul_cmplx` (complex multiply for angle/dihedral Fourier series). |
| `src/math/quat4.rs` | 31 | `Quat4d` (f64 xyzw), `Quat4i` (i32 xyzw), `QUAT4I_MINUS_ONES`. |
| `src/math/math3d.rs` | 26 | f32 helpers: `normalize3`, `cross3`, `dot3`, `sub3`, `add3`, `mul3s` (for GPU rendering). |
| `src/math/math4d.rs` | 37 | f32 4×4 matrices: `look_at`, `ortho` (Vulkan NDC), `mul4x4`, `transpose4x4`. |
| `src/math/fastmath.rs` | 36 | `sq`, `dangle`, `clamp_abs`, `sincos_taylor2`, `sincos_r2_taylor` (Taylor approximations). |

### `surfmol-topology` (`rust/topology/`)
*Molecular graph SSOT. Bonds/angles/dihedrals/inversions. UFF type assignment.*

| File | LOC | Contents |
|------|-----|----------|
| `src/topology_lib.rs` | 5 | Crate root: `pub mod topology; builder; params; export; assign_uff;` |
| `src/topology.rs` | 173 | `Topology` (apos, bonds, angles, dihedrals, inversions), `ne_pairs()`, `hybridization()`, `build_bonds_by_cutoff()`, `build_angles/dihedrals/inversions_from_bonds()`. Diagnostic prints (parity with C++ `MMFFBuilderBase.h`). |
| `src/builder.rs` | 599 | `Builder` — slot-based molecular graph with generational handles (`AtomH`, `BondH`), soft/hard remove, `cleanup_dead()`, `bake()` → `Topology`. Hex grid editing (`honeycomb_ring_nodes`, `snap_to_node`, `add_hex_ring`). `from_positions_cutoff()`, `from_positions_and_radii()`. |
| `src/params.rs` | 664 | `Params` — loads `ElementTypes.dat`, `AtomTypes.dat`, `BondTypes.dat`, `AngleTypes.dat`, `DihedralTypes.dat`. Structs: `ElementType`, `AtomType`, `BondParam`, `AngleParam`, `DihedralParam`. Wildcard matching for angle/dihedral params. `assign_uff_types()` (duplicate, also in `assign_uff.rs`). |
| `src/assign_uff.rs` | 125 | `assign_uff_types()` — octet-rule hybridization → UFF suffix (_3/_R/_2/_1). Special cases: H_, nitro N_R/O_R, C=O O_2, alkyne C_1. |
| `src/export.rs` | 74 | `TopologyData` (serde), `Topology::export_json()`, `import_json()`. TODO: .npy export. |
| `src/bin/assign_uff.rs` | 290 | **CLI binary `assign-uff`**: reads XYZ → builds topology → assigns UFF types → outputs JSON and/or binary (`UFFTOPO` magic header + flat arrays). Flags: `--json`, `--bin`, `--tol`, `--rcut`. |

### `surfmol-forcefields` (`rust/forcefields/`)
*Forcefield eval, MD, relaxation, `MolWorld` coordinator. See `rust/forcefields/DESIGN.md`.*

| File | LOC | Contents |
|------|-----|----------|
| `src/forcefields.rs` | 6 | Crate root: `pub mod uff; nonbonded; surface; mol_world; import; rigid_sp3;` |
| `src/mol_world.rs` | 140 | `MolWorld` — orchestrator. `BondedFFMode::{Uff, RigidSp3}`. Owns `DynamicAtoms`, `Uff`, `RigidSp3FF`, optional `NonBondedFF`, optional `SurfaceFolded`. `eval_forces()` → (eb,ea,ed,ei,enb,es). `run_md()`, `move_atom_md()`. Setup wrappers (`make_neigh_bs`, `bake_*_neighs`, `update_hneigh`, `setup_nacl_surface`). |
| `src/uff.rs` | 665 | `Uff` — bonded forcefield. `Buckets` (spatial partition for force assembly). SoA `AlignedVec` arrays: `bon_atoms`, `ang_atoms`, `dih_atoms`, `inv_atoms`, `hneigh`, `fint/fbon/fang/fdih/finv`, params. `eval_atom_bonds()`, `eval_angle_prokop()`, `eval_dihedral_prokop()` (Fourier series via `Vec2d::mul_cmplx`). `map_atom_interactions()`, `assemble_forces()`, `update_hneigh()`. Diagnostic prints (parity with C++ `UFF.h`). |
| `src/rigid_sp3.rs` | 237 | `RigidSp3FF` — **port-based rigid body FF (RAFF precursor)**. Per-atom quaternion (`quat`), angular velocity (`omega`), torque (`tau`), port geometry (`port_local`, `nport`). `set_sp3/sp2/sp1/point()`, `set_port_geometry_from_types()`. `eval_forces()` — port tip ↔ neighbor atom harmonic spring + torque. `move_atom_md()` — translational + quaternion rotation integration. `get_port_tip()`. |
| `src/nonbonded.rs` | 300 | `NonBondedFF` — LJ + Coulomb + H-bond. `reqs` (RvdW, sqrt(EvdW), Q, Hb), `plqs` (Pauli, London, Q, Hb). `make_plqs()`, `make_second_neighs()` (1-2 + 1-3 exclusion, EXCL_MAX=16), `make_pbc_shifts()`. `eval()` / `eval_pbc()` — O(N²) with exclusion skip + force clamping. `check_req_limits()`. |
| `src/surface.rs` | 512 | `SurfaceFolded` — separable tensor-product basis (Fourier in x/y, exp decay in z). Complex recurrence for harmonics (1 cos/sin + nmax complex muls). `eval_atom_scratch()` (no per-atom alloc), `eval_all_clamped()`. `setup_nacl_surface()` — NaCl checkerboard. `SurfaceScratch` (reusable buffers). Unit tests: harmonics recurrence, constant/cos/z-decay basis, req2plq. |
| `src/import.rs` | 13 | `load_topology_from_json()` → `(Uff, Vec<String>)`. |
| `src/mol_engine.rs` | 90 | **CLI binary `mol_engine`**: Rhai-scripted MD/relaxation. Registers `load_topology`, `eval_forces`, `step_md`, `relax`, `get_natoms`. `SimulationEngine` wraps `MolWorld` in `Arc<Mutex>`. |
| `DESIGN.md` | — | Ownership model: "borrow, don't own". `DynamicAtoms` is single owner; FFs borrow slices. Data hierarchy diagram. |
| `examples/md.rhai` | — | Rhai MD script example. |
| `examples/relax.rhai` | — | Rhai relaxation script example. |
| `tests/test_rigid_sp3.rs` | 110 | Tetrahedral sp3 center (CH4-like) test. |
| `tests/test_surface.rs` | 188 | Surface eval tests + SVG plot generator (pure Rust, no deps). |

### `surfmol-molrender` (`rust/molrender/`)
*wgpu rendering primitives. No simulation logic. WGSL shaders inline.*

| File | LOC | Contents |
|------|-----|----------|
| `src/molrender.rs` | 154 | Crate root + `ThumbnailRenderer` — offscreen wgpu renderer (ImpostorRenderer wrapper). Auto-fitting ortho camera. RGBA readback. |
| `src/impostor.rs` | 365 | `ImpostorRenderer` — raytraced sphere impostors (WGSL shader inline). `AtomInstance` (pos, radius, color), `CameraData` (view_proj, eye, right, up, forward). Re-exports `math3d`/`math4d` helpers. |
| `src/line_renderer.rs` | 191 | `LineRenderer` — line segments (WGSL inline). `LineVertex` (pos, col). |
| `src/surface_renderer.rs` | 229 | `SurfaceRenderer` — textured quad for surface potential visualization (WGSL inline). |
| `tests/debug_simple.rs` | — | Single-atom render debug. |
| `tests/debug_single.rs` | — | Single-atom render with full params. |
| `tests/debug_eico.rs` | 69 | Eicosanediol thumbnail debug. |
| `tests/impostor_single.rs` | 112 | Direct ImpostorRenderer test. |
| `tests/render_all.rs` | 74 | Render all XYZ thumbnails. |
| `tests/render_thumbs.rs` | 69 | Sample thumbnail render. |

### `surfmol-apps` (`rust/apps/`)
*GUI applications. No simulation logic — wires backend crates together.*

| File | LOC | Contents |
|------|-----|----------|
| `src/lib.rs` | 1 | Crate root: `pub mod gui;` |
| `src/editor.rs` | 1153 | **Binary `editor`** — 3D molecular editor. winit + wgpu + egui. TrackballCam, atom picking (ray-sphere), bond drawing, hex-grid Kekule editing, MD relaxation (RigidSp3 + NonBonded + NaCl surface), surface potential visualization. Constants: `LATTICE_A=5.66` (NaCl), `BETA_CHARGE=0.3`, etc. |
| `src/mol_browser.rs` | 249 | **Binary `mol_browser`** — XYZ directory browser with egui thumbnail grid. `MolEntry`, `MolBrowserApp`. |
| `src/gui/mod.rs` | 4 | `pub mod gizmos; kekule_editor; thumbnailer; trackball;` |
| `src/gui/gizmos.rs` | — | `make_bond_segments()` — multi-segment bond line generation. |
| `src/gui/kekule_editor.rs` | 338 | `KekuleEditor` — hex-grid molecular editor. `EditMode::{Select,HexPaint,HexToggle,AtomDraw,BondDraw}`. `collect_hex_grid_points()`, `collect_builder_bonds/atoms()`, `export_xyz()`, `builder_summary()`, `element_color()`. |
| `src/gui/thumbnailer.rs` | 234 | `MolThumbnailer` — wraps `ImpostorRenderer` + `LineRenderer` for egui thumbnail textures. |
| `src/gui/trackball.rs` | 63 | `TrackballCam` — orbit camera (target, rotation Quat, zoom, lerp). |
| `tests/test_thumb.rs` | — | MolThumbnailer integration test (saves PNG via `image` crate). |

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

**Note:** These kernels are not yet wired into the Rust crates (no `ocl` usage in any crate yet). The Rust CPU implementations in `surfmol-forcefields` are the current authoritative references.

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

All commands run from `rust/`:

```bash
# Build
cargo build                              # build all crates
cargo build --release                    # release build

# Run binaries
cargo run -p surfmol-apps --bin editor          # 3D molecular editor
cargo run -p surfmol-apps --bin mol_browser     # XYZ directory browser
cargo run -p surfmol-topology --bin assign-uff -- <xyz> --json out.json --bin out.bin  # topology CLI
cargo run -p surfmol-forcefields --bin mol_engine -- --script examples/relax.rhai     # Rhai MD engine

# Test
cargo test                               # all tests
cargo test -p surfmol-forcefields        # forcefield tests only
cargo test -p surfmol-molrender          # render tests only
cargo test -p surfmol-apps               # GUI/composite tests

# Check
cargo check --workspace                  # fast type check
cargo clippy --workspace                 # lints
```

**Test data paths:** tests use relative paths like `../../data/ElementTypes.dat` — run from `rust/` or the crate dir.

## 7. Key data structures (cross-crate)

| Struct | Crate | Role |
|--------|-------|------|
| `Vec3d`, `Quat4d`, `Quat4i` | common | `#[repr(C)]` math primitives, f64 |
| `AlignedVec<T, 64>` | common | 64-byte aligned allocator for SIMD-friendly arrays |
| `Atoms` | common | Static atomic data (apos, atypes, neighs, neigh_bs) |
| `DynamicAtoms` | common | Atoms + fapos + vapos + MD integrators. **Single owner of per-atom state.** |
| `Topology` | topology | Flat arrays: apos, bonds, angles, dihedrals, inversions |
| `Builder` | topology | Slot-based graph with generational handles (AtomH, BondH), hex-grid editing |
| `Params` | topology | Loaded FF parameter tables (elements, atom types, bonds, angles, dihedrals) |
| `Uff` | forcefields | Bonded FF: SoA arrays, Buckets force assembly, hneigh, eval_*_prokop |
| `RigidSp3FF` | forcefields | Port-based rigid body: quat, omega, tau, port_local. **RAFF precursor.** |
| `NonBondedFF` | forcefields | LJ+Coulomb+Hbond: reqs, plqs, excl, PBC shifts |
| `SurfaceFolded` | forcefields | Separable Fourier basis surface potential |
| `MolWorld` | forcefields | Coordinator: DynamicAtoms + Uff + RigidSp3FF + optional NonBondedFF/SurfaceFolded |
| `AtomInstance`, `CameraData` | molrender | GPU vertex/uniform layouts (match WGSL structs) |
| `ImpostorRenderer`, `LineRenderer`, `SurfaceRenderer` | molrender | wgpu render pipelines |
| `TrackballCam`, `KekuleEditor`, `MolThumbnailer` | apps | GUI utilities |

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

See `Import_other_Repos.md` for the full cross-repo import map.

## 9. What's NOT yet implemented

- **OpenCL integration:** `ocl` 0.19 is declared in workspace deps but no crate uses it yet. CPU Rust is authoritative.
- **RAFF (RigidAtomFF):** `RigidSp3FF` is the precursor; full RAFF (ARAP ports, reactive/dissociative Morse, fixed vs reactive variants) not yet implemented. See `DESIGN_GOALS.md` §2.
- **Projective / Position-Based Dynamics:** not yet implemented. See `DESIGN_GOALS.md` §3.
- **AABB collision acceleration:** not yet implemented (nonbonded is O(N²)). See `DESIGN_GOALS.md` §2.3.
- **Global optimization (GOpt):** not yet implemented.
- **`.npy` export:** `export.rs` has a TODO.
- **Cargo profile overrides** (`debug=1`, `strip`, LTO): not yet applied to workspace root. See `Import_other_Repos.md` §4.
- **xtask automation:** not yet created. See `Import_other_Repos.md` §4.
- **`CODEMAP.md` status:** this file is current as of 2026-08-25.
