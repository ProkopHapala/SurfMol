---
type: work-notes
title: Rust workspace reorganization proposal (v3 — revised with user feedback)
description: Crate naming (no hyphens), surfmol integration crate, molrender fully generic, dependency graph analysis with questionable edges called out.
tags: [work-in-progress, design, rust, cargo, workspace]
timestamp: 2026-08-25
---

# Rust workspace reorganization proposal (v3)

**Status:** proposal — no changes made yet. Revised after user feedback (v3).

**Key changes from v2:**
- **No hyphens in crate names** — `moltopo`, `molff`, `surfff`, `molrender`, `molgui`, `molbrowser`, `molengine`, `assignuff`.
- **New `surfmol` integration crate** — contains `MolWorld` orchestrator. Ties `molff` + `surfff` + `moltopo` together. `molff` no longer depends on `surfff`.
- **`molrender` is fully generic** — exposes only generic arrays (sphere radius, color, line vertices). No type/element knowledge. GUI layer populates render instances from molecular types.
- **All names checked on crates.io — all available.**
- **Dependency graph explicitly analyzed** with questionable edges called out.

## 1. Crate names and roles

| Name | Was | Role | crates.io | In code |
|------|-----|------|-----------|---------|
| **`numcore`** | `surfmol-common` (math + util) | `#[repr(C)]` math (Vec3d, Quat4d, Vec2d, fastmath, math3d, math4d) + `AlignedVec` (64-byte aligned allocator). Zero domain knowledge. | ✅ available | `use numcore::...` |
| **`moltopo`** | `surfmol-topology` + common (xyz, molecular) | **Static** molecular topology: `Topology` (bonds/angles/dihedrals as SoA arrays), `Params`, `assign_uff`, `export`, `xyz` I/O, `Atoms`/`DynamicAtoms` (particle state + MD integrator). Read-optimized, immutable after construction. SoA at field level: `pos: Vec<Vec3d>`, `element: Vec<u8>`, `charge: Vec<f64>`. | ✅ available | `use moltopo::...` |
| **`mgraph`** | (new, from `surfmol-topology`'s `Builder`) | **Dynamic** graph editor: slot-based graph with generational handles, NeighChunks neighbor lists, soft-remove + compact, rich per-node metadata (flags, uid, selection). Name abbreviates both **mol-graph** and **mesh-graph** — generic graph structure usable for both molecules and meshes. Bakes to `moltopo` for simulation. See §10.7. | ✅ available | `use mgraph::...` |
| **`molff`** | `surfmol-forcefields` (intra-mol part) | Intra-molecular forcefields: `Uff` (bonded), `RigidSp3FF` (rigid body), `NonBondedFF` (LJ+Coulomb+Hbond). **No surface dependency.** | ✅ available | `use molff::...` |
| **`surfff`** | `surfmol-forcefields` (surface part) | Surface–molecule interaction: `SurfaceFolded` (separable Fourier basis), future `GridFF` (B-spline grid). | ✅ available | `use surfff::...` |
| **`surfmol`** | (new) | **Integration engine**: `MolWorld` orchestrator (DynamicAtoms + Uff + RigidSp3 + NonBonded + SurfaceFolded), `load_topology_from_json`. Ties molff + surfff + moltopo together. | ✅ available | `use surfmol::...` |
| **`molrender`** | `surfmol-molrender` | wgpu rendering primitives: `ImpostorRenderer` (raytraced spheres), `LineRenderer`, `SurfaceRenderer`, `ThumbnailRenderer`. **Fully generic — no type/element knowledge.** Takes `AtomInstance { pos, radius, color }`. | ✅ available | `use molrender::...` |
| **`molgui`** | `surfmol-apps` (lib part) | GUI toolkit: `TrackballCam`, `KekuleEditor`, `MolThumbnailer`, `gizmos`. Populates molrender instances from moltopo types. | ✅ available | `use molgui::...` |
| **`assignuff`** | `surfmol-topology` (binary) | CLI: build topology from XYZ, assign UFF types, export JSON/binary. | ✅ available | `cargo install assignuff` |
| **`molengine`** | `surfmol-forcefields` (binary) | CLI: Rhai-scripted MD/relaxation engine. | ✅ available | `cargo install molengine` |
| **`editor`** | `surfmol-apps` (binary) | GUI: 3D molecular editor with hex-grid Kekule editing, MD relaxation, surface visualization. | ✅ available | `cargo run -p editor` |
| **`molbrowser`** | `surfmol-apps` (binary) | GUI: XYZ directory browser with thumbnail grid. | ✅ available | `cargo run -p molbrowser` |

## 2. Dependency graph (explicit, with analysis)

### 2.1. Full graph

```
                    numcore  (bytemuck only — ZERO domain knowledge)
                   /   |    \
                  /    |     \
          moltopo   surfff   molrender
          /  |        |          |
         /   |        |          |
    molff   |        |        molgui ---- mgraph  (dynamic graph editor; bakes to moltopo)
       \    |        |        /    \
        \   |        |       /      \
         \  |        |     editor  molbrowser
          \ |        |      |
        surfmol      |      |
       (MolWorld)    |      |
            \       /      /
             \     /      /
            molengine    /
             (CLI)      /
                        /
                   (editor also depends on surfmol + mgraph for interactive editing)
```

**Note on `mgraph`:** `mgraph` (dynamic graph editor) depends on `numcore` and bakes to `moltopo` (produces a static topology snapshot). `molgui` uses `mgraph` for interactive Kekule editing and `moltopo` for reading baked topology. `mgraph` is named to abbreviate both "mol-graph" and "mesh-graph" — it's a generic graph structure. See §10.7–10.8.

### 2.2. Clean dependency edges (no concerns)

| Edge | Why | Status |
|------|-----|--------|
| `moltopo → numcore` | Uses `Vec3d`, `Quat4i`, `AlignedVec` for atom positions, neighbor lists, aligned arrays. | ✅ Clean — math is fundamental |
| `surfff → numcore` | Uses `Vec3d` only. SurfaceFolded is pure math on positions. | ✅ Clean — no molecular knowledge |
| `molrender → numcore` | Uses `Vec3d`, `math3d` (f32 helpers), `math4d` (matrices) for camera/instance data. | ✅ Clean — rendering needs math |
| `molff → moltopo` | `Uff::from_topology(&Topology)` constructor. `NonBondedFF` uses `Quat4i` neighs. | ✅ Clean — intra-mol FF needs molecular topology |
| `surfmol → molff` | `MolWorld` owns `Uff`, `RigidSp3FF`, `NonBondedFF`. | ✅ Clean — integration engine uses intra-mol FF |
| `surfmol → surfff` | `MolWorld` owns `Option<SurfaceFolded>`, calls `surf.eval_all_clamped()`. | ✅ Clean — integration engine uses surface FF |
| `surfmol → moltopo` | `MolWorld` owns `DynamicAtoms`, uses `Topology` for construction. | ✅ Clean — integration engine needs particle state |
| `molgui → molrender` | Uses `ImpostorRenderer`, `LineRenderer`, `CameraData` for rendering. | ✅ Clean — GUI uses rendering primitives |
| `molgui → moltopo` | `KekuleEditor` uses `Builder`, `AtomH`, `BondH`. `MolThumbnailer` uses `Params` for element colors/radii to populate `AtomInstance`. | ✅ Clean — GUI binds molecular types to render instances |
| `molengine → surfmol` | Rhai scripts call `MolWorld` methods. | ✅ Clean — CLI uses integration engine |
| `assignuff → moltopo` | CLI builds topology and assigns UFF types. | ✅ Clean — CLI uses topology |
| `assignuff → numcore` | Uses `xyz::read_xyz` (wait — xyz moves to moltopo). Actually only needs moltopo. | ✅ Clean |

### 2.3. Questionable dependency edges (need user decision)

#### ⚑ Q1: `editor → surfmol` — does the editor need the full integration engine?

**Current:** `editor.rs` (1153 LOC) uses `MolWorld` for MD relaxation, `NonBondedFF` for non-bonded setup, `SurfaceFolded` for NaCl surface, `Builder` for hex-grid editing, `Params` for atom types, `ImpostorRenderer`/`LineRenderer`/`SurfaceRenderer` for rendering.

**Question:** Should `editor` depend on `surfmol` (gets `MolWorld`), or should it depend on individual crates (`molff`, `surfff`, `moltopo`) and assemble `MolWorld` itself?

**My recommendation:** `editor → surfmol`. The editor is a SurfMol application — it should use the integration engine. If someone wants a non-SurfMol editor, they'd write their own app.

**Alternative:** If `editor` should be more flexible, it could depend on `molff` + `surfff` + `moltopo` directly and construct `MolWorld` from `surfmol` as a convenience. But this is the same thing — it still needs `surfmol` for `MolWorld`.

**Verdict:** `editor → surfmol + molgui + molrender`. No concern.

#### ⚑ Q2: `molbrowser → moltopo` — does the browser need full topology, or just xyz + params?

**Current:** `mol_browser.rs` (249 LOC) uses `read_xyz` (from moltopo), `Params` (from moltopo), `MolThumbnailer` (from molgui).

**Question:** `molbrowser` only reads XYZ files and renders thumbnails. It doesn't do MD or forcefield evaluation. Does it need `moltopo` (which includes `Builder`, `assign_uff`, etc.) or could it use a lighter dependency?

**My recommendation:** Keep `molbrowser → moltopo`. The browser needs `Params` for element colors/radii (to populate render instances) and `read_xyz` for loading molecules. These are in `moltopo`. Splitting `moltopo` further (e.g., extracting `xyz` + `params` into a lighter crate) would be over-engineering at this point.

**Verdict:** `molbrowser → molgui + moltopo + molrender`. No concern, but note that `molbrowser` pulls in `moltopo` (which includes `Builder`, `assign_uff`) even though it only uses `read_xyz` + `Params`. This is acceptable — Cargo features could gate this later if needed.

#### ⚑ Q3: `molff → moltopo` — could `molff` be topology-agnostic?

**Current:** `Uff::from_topology(&Topology)` takes a `Topology` struct. `Uff::new(natoms, bonds, angles, dihedrals, inversions)` takes raw arrays.

**Question:** If we removed `Uff::from_topology()` from `molff` and moved it to `surfmol` (as a convenience constructor), would `molff` still need `moltopo`?

**Analysis:**
- `uff.rs` imports `Topology` only for `from_topology()`. The `new()` constructor takes raw `&[[i32;2]]`, `&[[i32;3]]`, `&[Quat4i]` — no `Topology` needed.
- `rigid_sp3.rs` imports `Uff` from `crate::uff` — no topology.
- `nonbonded.rs` imports `Quat4i` from `numcore` — no topology.
- `import.rs` uses `import_json` from `moltopo` — but this moves to `surfmol`.

**If we move `from_topology()` and `import.rs` to `surfmol`:** `molff` would depend only on `numcore`. This would make `molff` a pure intra-molecular forcefield library with zero topology dependency — users pass raw arrays.

**Trade-off:**
- **Pro:** `molff` becomes more independent. Users who have their own topology representation can use `molff` directly with raw arrays.
- **Con:** `Uff::from_topology()` is a natural constructor that belongs with `Uff`. Moving it to `surfmol` splits the API.

**My recommendation:** **Keep `molff → moltopo`.** The dependency is natural (intra-mol FF needs molecular topology). The convenience of `from_topology()` outweighs the independence gain. If someone wants `molff` without `moltopo`, they can use `Uff::new()` with raw arrays — the `Topology` import is just for the convenience constructor.

**Verdict:** `molff → moltopo + numcore`. Acceptable, but flagged for user decision.

#### ⚑ Q4: `molgui → moltopo` — should the GUI toolkit depend on topology?

**Current:** `molgui` contains:
- `gizmos.rs` — uses `LineVertex` from molrender. No moltopo.
- `kekule_editor.rs` — uses `Builder`, `AtomH`, `BondH` from moltopo. **Needs moltopo.**
- `thumbnailer.rs` — uses `Params` from moltopo for element colors/radii. **Needs moltopo.**
- `trackball.rs` — uses `CameraData` from molrender. No moltopo.

**Question:** Should `molgui` depend on `moltopo`, or should the moltopo-dependent parts (`KekuleEditor`, `MolThumbnailer`) move to a separate crate?

**My recommendation:** Keep `molgui → moltopo`. The GUI toolkit is SurfMol-specific by nature — it's a molecular GUI. `KekuleEditor` edits molecular topology, `MolThumbnailer` renders molecules. Splitting these out would create a crate with just `TrackballCam` + `gizmos`, which is too small to justify.

**Verdict:** `molgui → molrender + moltopo`. Acceptable.

#### ⚑ Q5: `molrender` has NO dependency on any molecular crate — confirmed?

**Yes.** After the API change (§4 below), `molrender` depends only on:
- `numcore` (for `Vec3d`, `math3d`, `math4d`)
- `wgpu`, `bytemuck`, `pollster`

`molrender` has zero knowledge of:
- Elements, atom types, UFF types
- Bonds, angles, dihedrals, topology
- Forcefields, MD, energy
- `Params`, `Builder`, `Topology`

The GUI layer (`molgui`) is responsible for converting molecular data → render instances:
```rust
// In molgui/thumbnailer.rs — the GUI does the type→render binding:
let instances: Vec<AtomInstance> = apos.iter().zip(elems).map(|(p, el)| {
    let col = params.element_color_f32(el);
    let r = params.element_radius_vdw(el) * 0.3;
    AtomInstance { pos: [p.x as f32, p.y as f32, p.z as f32], radius: r, color: col, _pad: 0.0 }
}).collect();
renderer.render(size, &instances);  // molrender knows nothing about elements
```

**Verdict:** ✅ Clean. `molrender` is fully transferable to non-molecular projects.

#### ⚑ Q6: Should `surfmol` depend on `molrender`?

**No.** `surfmol` is a simulation engine, not a rendering layer. It depends on:
- `molff` (intra-molecular FF)
- `surfff` (surface FF)
- `moltopo` (topology, DynamicAtoms)

Rendering is a separate concern handled by `molrender` + `molgui`. The `editor` app depends on both `surfmol` (for simulation) and `molgui`/`molrender` (for rendering).

**Verdict:** ✅ `surfmol` does NOT depend on `molrender`. Clean separation: simulation vs rendering.

### 2.4. Independence summary

| Crate | Can be used without... | Reusable in |
|-------|----------------------|-------------|
| `numcore` | everything | ANY project (spacecraft, fluids, games, molecular) |
| `surfff` | moltopo, molff, surfmol, molrender, molgui | any particle-on-surface simulation |
| `molrender` | moltopo, molff, surfff, surfmol, molgui | any 3D particle/bond rendering project |
| `moltopo` | molff, surfff, surfmol, molrender, molgui | standalone molecular topology library |
| `molff` | surfff, surfmol, molrender, molgui | standalone intra-molecular FF library (needs moltopo) |
| `surfmol` | molrender, molgui, all binaries | full SurfMol simulation engine |
| `molgui` | surfmol, molff, surfff, all binaries | molecular GUI toolkit (needs molrender + moltopo) |

## 3. Registering crate names on crates.io

### 3.1. All names are available

Checked 2026-08-25 via `crates.io/api/v1/crates/<name>`:

| Name | Status |
|------|--------|
| `numcore` | ✅ available |
| `simcore` | ✅ available (fallback) |
| `moltopo` | ✅ available |
| `molff` | ✅ available |
| `surfff` | ✅ available |
| `surfmol` | ✅ available |
| `molsurf` | ✅ available (fallback) |
| `molrender` | ✅ available |
| `molgui` | ✅ available |
| `molbrowser` | ✅ available |
| `molengine` | ✅ available |
| `assignuff` | ✅ available |
| `mgraph` | ✅ available |

### 3.2. How to register (claim) a crate name

A crate name is claimed on crates.io **the first time you publish it**. There is no separate "name reservation" step. To claim a name:

1. **Create a crates.io account** at https://crates.io/ (login via GitHub).
2. **Get an API token:** Settings → API Settings → Create New Token.
3. **Configure cargo with the token:**
   ```bash
   cargo login <your-api-token>
   ```
4. **Add required metadata** to the crate's `Cargo.toml`:
   ```toml
   [package]
   name = "numcore"
   version = "0.1.0"
   edition = "2021"
   description = "Core numerical primitives: #[repr(C)] math vectors, aligned allocators"
   license = "MIT"              # required
   repository = "https://github.com/ProkopHapala/SurfMol"
   # readme = "README.md"       # recommended
   # keywords = ["math", "simd", "simulation"]   # optional, max 5
   # categories = ["mathematics", "science"]      # optional, must be from official list
   ```
5. **Publish:**
   ```bash
   cargo publish -p numcore
   ```
   This uploads the crate to crates.io and **permanently claims the name**. Once claimed, no one else can publish a crate with that name.

### 3.3. Strategy for claiming names early

If you want to claim names before the code is ready to publish:

1. Create a minimal crate (just `Cargo.toml` + `src/lib.rs` with a comment).
2. Add required metadata (`description`, `license`).
3. `cargo publish` — this claims the name with a `0.1.0` that has minimal content.
4. Later, publish `0.2.0`, `0.3.0`, etc. with real code.

**Caveat:** crates.io has a "yank" feature but no "delete" — once published, a version exists forever (though yanked versions can't be used by new projects). Don't publish garbage just to claim a name; publish a minimal but sensible stub.

**Recommendation:** Claim `numcore` and `surfmol` first (the two most important names). The others (`moltopo`, `molff`, `surfff`, `molrender`) are less likely to be taken by someone else.

### 3.4. Workspace vs independent publishing

In a Cargo workspace, each member crate can be published independently:
```bash
cargo publish -p numcore       # publishes only numcore
cargo publish -p moltopo       # publishes only moltopo
cargo publish -p surfmol       # publishes only surfmol
```

The `[workspace.dependencies]` section uses `path = "crates/numcore"` for local development. When publishing, you also add `version = "0.1.0"` so that downstream users get the published version from crates.io:

```toml
# In moltopo's Cargo.toml:
[dependencies]
numcore = { path = "../numcore", version = "0.1.0" }
```

This means: "use the local path when building from source, but require version 0.1.0+ from crates.io when published."

## 4. `molrender` — fully generic rendering (no type knowledge)

### 4.1. Current coupling

`ThumbnailRenderer::render()` currently takes `&[Vec3d]` (positions), `&[String]` (elements), `&Params` (for colors/radii), and internally maps elements → colors/radii:

```rust
// CURRENT (coupled):
pub fn render(&mut self, size: u32, apos: &[Vec3d], elems: &[String], bonds: &[[usize;2]], params: &Params) -> Vec<u8>
```

This forces `molrender` to depend on `moltopo` (for `Params`).

### 4.2. Proposed: generic API

`molrender` exposes only generic render primitives. The caller (GUI layer) populates instances:

```rust
// PROPOSED (decoupled):
// molrender — no type knowledge, just render what you're given:
pub struct AtomInstance {
    pub pos: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub _pad: f32,
}

impl ThumbnailRenderer {
    pub fn render(&mut self, size: u32, instances: &[AtomInstance]) -> Vec<u8> { ... }
}

impl ImpostorRenderer {
    pub fn set_atoms(&mut self, instances: &[AtomInstance]) { ... }
    pub fn render(&mut self, target: &wgpu::TextureView, clear: wgpu::Color, camera: &CameraData) { ... }
}

impl LineRenderer {
    pub fn set_lines(&mut self, vertices: &[LineVertex]) { ... }
    pub fn render(&mut self, target: &wgpu::TextureView, camera: &CameraData) { ... }
}
```

### 4.3. Type → render binding lives in `molgui`

The GUI layer converts molecular data to render instances:

```rust
// In molgui/thumbnailer.rs — GUI does the binding:
pub fn render_molecule(&mut self, apos: &[Vec3d], elems: &[String], bonds: &[[usize;2]], params: &Params) -> Vec<u8> {
    let instances: Vec<AtomInstance> = apos.iter().zip(elems.iter()).map(|(p, el)| {
        let col = params.element_color_f32(el);
        let r = params.element_radius_vdw(el) * 0.3;
        AtomInstance { pos: [p.x as f32, p.y as f32, p.z as f32], radius: r, color: col, _pad: 0.0 }
    }).collect();
    self.renderer.render(self.thumb_size, &instances)
}
```

This means:
- `molrender` knows about spheres, lines, cameras, WGSL shaders. **Nothing else.**
- `molgui` knows about elements, atom types, bonds, Params. **It binds molecular types to render instances.**
- A non-molecular project (e.g., fluid simulation) can use `molrender` directly: create `AtomInstance`s from particle data, no `Params` needed.

### 4.4. What `molrender` `Cargo.toml` looks like

```toml
[package]
name = "molrender"
version = "0.1.0"
edition = "2021"
description = "Generic wgpu rendering primitives: sphere impostors, lines, surface quads"

[dependencies]
numcore = { workspace = true }
wgpu = { workspace = true }
bytemuck = { workspace = true }
pollster = { workspace = true }

[dev-dependencies]
image = { workspace = true }
```

**No `moltopo`. No `molff`. No `surfff`. No `surfmol`.** Just math + GPU.

## 5. The `surfmol` integration crate

### 5.1. Why it exists

`molff` (intra-molecular FF) and `surfff` (surface FF) are independent. But a real simulation needs both — plus topology, plus particle state, plus an MD integrator. That orchestration is `MolWorld`.

Without `surfmol`, `MolWorld` would have to live in either `molff` (forcing `molff → surfff` dependency) or `surfff` (forcing `surfff → molff` dependency). Both break the independence of the respective crate.

`surfmol` is the **integration layer** that depends on both and contains the orchestrator:

### 5.2. Contents

| File | Source | Contents |
|------|--------|----------|
| `mol_world.rs` | was `forcefields/src/mol_world.rs` | `MolWorld`, `BondedFFMode`, `eval_forces()`, `run_md()`, `move_atom_md()`, setup wrappers |
| `import.rs` | was `forcefields/src/import.rs` | `load_topology_from_json()` → `(MolWorld, Vec<String>)` |

### 5.3. `Cargo.toml`

```toml
[package]
name = "surfmol"
version = "0.1.0"
edition = "2021"
description = "SurfMol molecular simulation engine: integrates intra-molecular and surface forcefields"

[dependencies]
numcore = { workspace = true }
moltopo = { workspace = true }
molff = { workspace = true }
surfff = { workspace = true }
```

**No rendering, no GUI, no CLI deps.** Pure simulation engine.

### 5.4. Dependency direction

```
molff     surfff     moltopo
  \         |          /
   \        |         /
    \       |        /
     \      |       /
      \     |      /
       \    |     /
        surfmol
      (MolWorld)
```

`surfmol` depends on all three. None of the three depend on `surfmol` or on each other (except `molff → moltopo`, see ⚑ Q3).

## 6. Target directory layout

```
SurfMol/
├── Cargo.toml                    # workspace root (at repo root)
├── Cargo.lock
├── crates/
│   ├── numcore/                  # math + AlignedVec (domain-agnostic)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs            # was common.rs (only math/ + util.rs)
│   │   │   ├── util.rs
│   │   │   └── math/
│   │   └── README.md
│   ├── moltopo/                  # static molecular topology (read-optimized, SoA)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs            # was topology_lib.rs
│   │   │   ├── topology.rs
│   │   │   ├── params.rs
│   │   │   ├── assign_uff.rs
│   │   │   ├── export.rs
│   │   │   ├── xyz.rs            # moved from common
│   │   │   └── molecular.rs      # moved from common (Atoms, DynamicAtoms)
│   │   └── README.md
│   ├── mgraph/                   # dynamic graph editor (mol-graph + mesh-graph)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── graph.rs          # slot-based graph, generational handles (was builder.rs)
│   │   │   ├── neigh_chunks.rs   # cache-line neighbor lists (ported from SSE NeighChunks.h)
│   │   │   ├── selection.rs      # SDF-based selection (ported from SSE Selection.h)
│   │   │   └── bake.rs           # mgraph -> moltopo conversion (export to static SoA)
│   │   └── README.md
│   ├── molff/                    # intra-molecular forcefields
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs            # was forcefields.rs (without surface.rs, mol_world.rs, import.rs)
│   │   │   ├── uff.rs
│   │   │   ├── rigid_sp3.rs
│   │   │   └── nonbonded.rs
│   │   ├── DESIGN.md             # was forcefields/DESIGN.md
│   │   ├── tests/
│   │   │   └── test_rigid_sp3.rs
│   │   └── README.md
│   ├── surfff/                   # surface–molecule interaction
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs            # was surface.rs (standalone)
│   │   ├── tests/
│   │   │   └── test_surface.rs
│   │   └── README.md
│   ├── surfmol/                  # integration engine (MolWorld)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── mol_world.rs      # was forcefields/src/mol_world.rs
│   │   │   └── import.rs         # was forcefields/src/import.rs
│   │   └── README.md
│   ├── molrender/                # wgpu rendering primitives (generic, no type knowledge)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs            # was molrender.rs (ThumbnailRenderer API changed)
│   │   │   ├── impostor.rs
│   │   │   ├── line_renderer.rs
│   │   │   └── surface_renderer.rs
│   │   ├── tests/
│   │   └── README.md
│   ├── molgui/                   # GUI toolkit (binds molecular types to render instances)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   └── gui/
│   │   │       ├── mod.rs
│   │   │       ├── gizmos.rs
│   │   │       ├── kekule_editor.rs
│   │   │       ├── thumbnailer.rs
│   │   │       └── trackball.rs
│   │   ├── tests/
│   │   │   └── test_thumb.rs
│   │   └── README.md
│   ├── assignuff/                # CLI binary
│   │   ├── Cargo.toml
│   │   └── src/main.rs           # was topology/src/bin/assign_uff.rs
│   ├── molengine/                # CLI binary
│   │   ├── Cargo.toml
│   │   ├── src/main.rs           # was forcefields/src/mol_engine.rs
│   │   └── examples/
│   │       ├── md.rhai
│   │       └── relax.rhai
│   ├── editor/                   # GUI binary
│   │   ├── Cargo.toml
│   │   └── src/main.rs           # was apps/src/editor.rs
│   └── molbrowser/               # GUI binary
│       ├── Cargo.toml
│       └── src/main.rs           # was apps/src/mol_browser.rs
├── data/                         # stays at root
├── opencl/                       # stays at root
├── doc/
├── notes/
├── ...
```

**Workspace `Cargo.toml` at root:**
```toml
[workspace]
members = [
    "crates/numcore",
    "crates/moltopo",
    "crates/mgraph",
    "crates/molff",
    "crates/surfff",
    "crates/surfmol",
    "crates/molrender",
    "crates/molgui",
    "crates/assignuff",
    "crates/molengine",
    "crates/editor",
    "crates/molbrowser",
]
resolver = "2"

[workspace.package]
license = "MIT"                          # see open question §9.1
repository = "https://github.com/ProkopHapala/SurfMol"
edition = "2021"

[workspace.dependencies]
# Internal crates
numcore = { path = "crates/numcore" }
moltopo = { path = "crates/moltopo" }
mgraph = { path = "crates/mgraph" }
molff = { path = "crates/molff" }
surfff = { path = "crates/surfff" }
surfmol = { path = "crates/surfmol" }
molrender = { path = "crates/molrender" }
molgui = { path = "crates/molgui" }

# External crates
bytemuck = { version = "1.21", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
wgpu = "29"
pollster = "0.4"
winit = "0.30"
eframe = { version = "0.29", default-features = false, features = ["default_fonts", "glow"] }
egui = "0.34"
egui-winit = "0.34"
egui-wgpu = "0.34"
egui_extras = "0.29"
egui_plot = "0.34"
glam = "0.29"
nalgebra = "0.33"
ndarray = "0.16"
rand = "0.8"
ocl = "0.19"
rhai = "1.19"
clap = { version = "4.5", features = ["derive"] }
image = { version = "0.25", default-features = false, features = ["png"] }
```

## 7. File-level mapping (what moves where)

### `surfmol-common` → split between `numcore` and `moltopo`

| File | Current | New crate | New path |
|------|---------|-----------|----------|
| `math/*` (6 files) | `common/src/math/` | `numcore` | `numcore/src/math/` |
| `util.rs` | `common/src/util.rs` | `numcore` | `numcore/src/util.rs` |
| `xyz.rs` | `common/src/xyz.rs` | `moltopo` | `moltopo/src/xyz.rs` |
| `molecular.rs` | `common/src/molecular.rs` | `moltopo` | `moltopo/src/molecular.rs` |
| `common.rs` | `common/src/common.rs` | `numcore` | `numcore/src/lib.rs` (rewrite: only `pub mod math; pub mod util;`) |

### `surfmol-topology` → `moltopo` (minus binary)

| File | Current | New path |
|------|---------|----------|
| `topology_lib.rs` | `topology/src/topology_lib.rs` | `moltopo/src/lib.rs` (add `pub mod xyz; pub mod molecular;`) |
| `topology.rs` | `topology/src/topology.rs` | `moltopo/src/topology.rs` |
| `builder.rs` | `topology/src/builder.rs` | `mgraph/src/graph.rs` (moves to `mgraph` — dynamic graph editor) |
| `params.rs` | `topology/src/params.rs` | `moltopo/src/params.rs` |
| `assign_uff.rs` | `topology/src/assign_uff.rs` | `moltopo/src/assign_uff.rs` |
| `export.rs` | `topology/src/export.rs` | `moltopo/src/export.rs` |
| `bin/assign_uff.rs` | `topology/src/bin/assign_uff.rs` | `assignuff/src/main.rs` (separate crate) |

### `surfmol-forcefields` → split between `molff`, `surfff`, `surfmol`, `molengine`

| File | Current | New crate | New path |
|------|---------|-----------|----------|
| `uff.rs` | `forcefields/src/uff.rs` | `molff` | `molff/src/uff.rs` |
| `rigid_sp3.rs` | `forcefields/src/rigid_sp3.rs` | `molff` | `molff/src/rigid_sp3.rs` |
| `nonbonded.rs` | `forcefields/src/nonbonded.rs` | `molff` | `molff/src/nonbonded.rs` |
| `surface.rs` | `forcefields/src/surface.rs` | `surfff` | `surfff/src/lib.rs` |
| `mol_world.rs` | `forcefields/src/mol_world.rs` | `surfmol` | `surfmol/src/mol_world.rs` |
| `import.rs` | `forcefields/src/import.rs` | `surfmol` | `surfmol/src/import.rs` |
| `mol_engine.rs` | `forcefields/src/mol_engine.rs` | `molengine` | `molengine/src/main.rs` |
| `DESIGN.md` | `forcefields/DESIGN.md` | `molff` | `molff/DESIGN.md` |
| `examples/*.rhai` | `forcefields/examples/` | `molengine` | `molengine/examples/` |
| `tests/test_rigid_sp3.rs` | `forcefields/tests/` | `molff` | `molff/tests/` |
| `tests/test_surface.rs` | `forcefields/tests/` | `surfff` | `surfff/tests/` |
| `forcefields.rs` | `forcefields/src/forcefields.rs` | `molff` | `molff/src/lib.rs` (rewrite: only `pub mod uff; pub mod rigid_sp3; pub mod nonbonded;`) |

### `surfmol-molrender` → `molrender` (with API change)

| File | Current | New path | Change |
|------|---------|----------|--------|
| `molrender.rs` | `molrender/src/molrender.rs` | `molrender/src/lib.rs` | `ThumbnailRenderer::render()` API: take `&[AtomInstance]` instead of `(&[Vec3d], &[String], &Params)` |
| `impostor.rs` | `molrender/src/impostor.rs` | `molrender/src/impostor.rs` | No change |
| `line_renderer.rs` | `molrender/src/line_renderer.rs` | `molrender/src/line_renderer.rs` | No change |
| `surface_renderer.rs` | `molrender/src/surface_renderer.rs` | `molrender/src/surface_renderer.rs` | No change |
| `tests/*.rs` | `molrender/tests/` | `molrender/tests/` | Update: callers do Params→AtomInstance mapping themselves |

**`molrender` `Cargo.toml`:** Remove `surfmol-topology` dependency. Only `numcore` + `wgpu` + `bytemuck` + `pollster`.

### `surfmol-apps` → split into `molgui` + `editor` + `molbrowser`

| File | Current | New crate | New path |
|------|---------|-----------|----------|
| `lib.rs` | `apps/src/lib.rs` | `molgui` | `molgui/src/lib.rs` |
| `gui/*` (5 files) | `apps/src/gui/` | `molgui` | `molgui/src/gui/` |
| `editor.rs` | `apps/src/editor.rs` | `editor` | `editor/src/main.rs` |
| `mol_browser.rs` | `apps/src/mol_browser.rs` | `molbrowser` | `molbrowser/src/main.rs` |
| `tests/test_thumb.rs` | `apps/tests/` | `molgui` | `molgui/tests/` |

## 8. `use` statement changes

Every `use` in the codebase needs updating:

| Old | New |
|-----|-----|
| `use surfmol_common::math::vec3::Vec3d` | `use numcore::math::vec3::Vec3d` |
| `use surfmol_common::util::AlignedVec` | `use numcore::util::AlignedVec` |
| `use surfmol_common::xyz` | `use moltopo::xyz` |
| `use surfmol_common::molecular::DynamicAtoms` | `use moltopo::molecular::DynamicAtoms` |
| `use surfmol_topology::topology::Topology` | `use moltopo::topology::Topology` |
| `use surfmol_topology::builder::Builder` | `use moltopo::builder::Builder` |
| `use surfmol_topology::params::Params` | `use moltopo::params::Params` |
| `use surfmol_forcefields::uff::Uff` | `use molff::uff::Uff` |
| `use surfmol_forcefields::rigid_sp3::RigidSp3FF` | `use molff::rigid_sp3::RigidSp3FF` |
| `use surfmol_forcefields::nonbonded::NonBondedFF` | `use molff::nonbonded::NonBondedFF` |
| `use surfmol_forcefields::surface::SurfaceFolded` | `use surfff::SurfaceFolded` |
| `use surfmol_forcefields::mol_world::MolWorld` | `use surfmol::mol_world::MolWorld` |
| `use surfmol_molrender::impostor::*` | `use molrender::impostor::*` |
| `use surfmol_apps::gui::*` | `use molgui::gui::*` |

## 9. Migration steps (when ready to execute)

Single atomic reorganization (one branch, one commit):

1. `git checkout -b reorg/crate-layout`
2. Create root `Cargo.toml` with workspace members.
3. Create `crates/` directory.
4. Move and rename crates per §7.
5. Rename all lib files to `src/lib.rs`. Remove `[lib] path = ...` from all `Cargo.toml`.
6. Extract binaries to their own crates.
7. Decouple `molrender`: change `ThumbnailRenderer::render()` API to take `&[AtomInstance]`. Move Params→instance mapping to `molgui/thumbnailer.rs`.
8. Update all `use` statements per §8.
9. Move `rust/Cargo.lock` → `Cargo.lock` (root).
10. Delete `rust/.cargo/config.toml`.
11. Delete `rust/` directory if empty.
12. Update test data paths: `../../data/` → `../data/`.
13. Update `CODEMAP.md`, `AGENTS.md`, `ARCHITECTURE.md`, `Import_other_Repos.md`.
14. `cargo check --workspace`.
15. `cargo test --workspace`.

### Optional (separate PR):

16. Add crate metadata to each `[package]` (description, license, repository, keywords, categories).
17. Claim names on crates.io (§3).
18. Consolidate all shared deps to `[workspace.dependencies]`.
19. Add `benches/` with criterion benchmarks.

## 10. Open questions for user decision

### 10.1. License

What license? MIT, Apache-2.0, dual MIT/Apache-2.0, or GPL? Needed for crates.io publishing. **My recommendation: dual MIT/Apache-2.0** (most permissive, standard in Rust ecosystem).

### 10.2. `molff → moltopo` dependency (⚑ Q3)

Keep `Uff::from_topology()` in `molff` (needs `moltopo`), or move it to `surfmol` (makes `molff` depend only on `numcore`)? **My recommendation: keep in `molff`** — the convenience constructor is natural there.

### 10.3. `editor → surfmol` dependency (⚑ Q1)

Should the editor depend on `surfmol` (integration engine) or on individual crates? **My recommendation: `editor → surfmol`** — the editor is a SurfMol application.

### 10.4. Versioning

All crates at `0.1.0` initially. Independent semver after that? **My recommendation: yes.** `numcore` might stabilize to `1.0` before `molff`.

### 10.5. `DynamicAtoms` location

`DynamicAtoms` (particle state + MD integrator) currently lives in `moltopo` (moved from `common`). Should it eventually move to `numcore` as a generic `Particles` struct for non-molecular sims? **My recommendation: leave in `moltopo` for now**, extract when a non-molecular use case arises.

### 10.6. `surfmol` vs `molsurf` for integration crate name

User suggested `surfmol` is probably better. Both are available on crates.io. **Confirmed: `surfmol`.**

### 10.7. Split `moltopo` into static + dynamic? (user-raised, important)

**User's question:** Should `moltopo` be split into:
1. A **static, read-optimized topology** (minimal graph: what is bonded to what, fragment/group splits) — SoA layout, cache-friendly for forcefield evaluation.
2. A **dynamic, edit-optimized molecular editor** — slot-based, add/remove atoms and bonds interactively, rich metadata.

**Analysis:** This is exactly the `CMesh` (static) vs. `MeshBuilder2` (dynamic) duality in SimpleSimulationEngine (`cpp/common/geometry/MeshBuilder2.h`). See `Import_other_Repos.md` §5.3 for the full analysis.

**The two representations have conflicting optimization goals:**

| Property | Static (forcefield eval) | Dynamic (interactive editing) |
|----------|------------------------|------------------------------|
| Layout | SoA at field level: `pos: Vec<Vec3d>` (positions stay as Vec3d — x/y/z are always processed together), `element: Vec<u8>`, `charge: Vec<f64>` — separate arrays per field, each contiguous. GPU: `pos: Vec<[f32;4]>` (float4, w=mass/charge). | AoS: `Atom { pos: Vec3d, element: u8, flags: u32, uid: u32, i0ngBonds: i16, ... }` — one struct array with all fields interleaved |
| Neighbor lists | CSR / Buckets (counting sort, O(1) range access) | NeighChunks (cache-line per atom, overflow chaining) |
| Removal | None (immutable) | Soft-remove (flag dead, compact later) |
| Allocation | One-time, exact size | Incremental, with headroom |
| Access pattern | Sequential (loop over all atoms/bonds) | Random (pick one atom, edit it) |
| Metadata | Minimal (element, charge) | Rich (flags, uid, selection, chunk membership) |

**Proposed split:**

- **`moltopo`** (static) — the current `Topology` struct: SoA arrays of bonds/angles/dihedrals, `Atoms` with CSR neighbor lists (`neighs`, `neigh_bs` via Buckets). Immutable after construction. This is what `molff` and `surfmol` consume for forcefield evaluation. Optimized for fast sequential reads.

- **`mgraph`** (dynamic) — a new crate: slot-based graph with generational handles (the current `Builder`), NeighChunks for variable-degree neighbor lists, soft-remove + compact, rich metadata (flags, uid, selection state). This is what `molgui`'s `KekuleEditor` uses for interactive editing. Optimized for random insert/delete. **Name `mgraph` abbreviates both "mol-graph" and "mesh-graph"** — it's a generic graph structure usable for both molecules and meshes (see §10.8).

**Conversion:** `mgraph → moltopo` is a "bake" operation: take the dynamic graph, compact dead entries, export to SoA arrays, build CSR neighbor lists. This is the `Builder2::export_pos/edges/tris()` pattern — the dynamic editor exports to a static snapshot for simulation.

```
mgraph (dynamic editor)  --bake()-->  moltopo (static topology)
                                           |
                                     molff / surfff / surfmol consume
```

**Dependency:**
- `moltopo` — depends on `numcore` only. No dependency on `mgraph`.
- `mgraph` — depends on `numcore` + `moltopo` (for the bake/export target). Or: `mgraph` produces raw arrays and `moltopo` constructs from them — no direct dependency.
- `molgui` — depends on `mgraph` (for editing) + `moltopo` (for reading baked topology).

**Alternative: keep as one crate with two modes.** Like `Builder2`'s dual-mode `VertT` union, `moltopo` could have both representations in one crate, switching modes as needed. This avoids the crate split but couples the two concerns.

**My recommendation:** **Split into `moltopo` (static) + `mgraph` (dynamic).** The optimization goals are genuinely conflicting, and the user explicitly raised this. The bake operation is a natural boundary. But this is a **user decision** — see open question below.

**Open question for user:** Split `moltopo` into `moltopo` (static) + `mgraph` (dynamic), or keep as one crate with dual modes?

### 10.8. Share graph algorithms between mesh and molecule? (user-raised)

**User's question:** Can we base molecular topology on SimpleSimulationEngine's `MeshBuilder2`/`CMesh`, or derive from it, to avoid reimplementing mesh algorithms for molecules (and vice versa)?

**Analysis:** Molecular topology (atoms + bonds + angles) is structurally a graph, just like a mesh (vertices + edges + faces). The algorithms are identical:
- Neighbor list construction (Buckets for static, NeighChunks for dynamic)
- Edge/bond deduplication (hash map on vertex/atom pairs)
- Soft-remove + compaction (mark dead, rebuild indices)
- Edge-loop / bond-ring sorting (sort bonds into a ring)
- Selection by SDF (select atoms within a region)
- Picking (ray-sphere for atoms, ray-cylinder for bonds)

**Three approaches** (detailed in `Import_other_Repos.md` §5.4):

1. **Generic graph crate** (`graphcore` or in `numcore`): `Graph` struct with generic verts/edges/neigh. Both `moltopo`/`mgraph` and a future Rust mesh crate build on it.
2. **Trait-based sharing**: `GraphLike` trait, implemented separately for mesh and molecule.
3. **Direct port**: Port Builder2 algorithms to Rust in `mgraph`, extract to shared crate when a second use case appears.

**My recommendation:** **Approach 3 (direct port) for now.** Port Builder2's algorithms into `mgraph` directly. If/when a Rust mesh project is started, extract the shared algorithms into `graphcore` at that point. This avoids premature abstraction (YAGNI) while keeping the door open. The algorithms are simple enough that re-extraction later is cheap.

**Open question for user:** Generic graph crate now (Approach 1), or direct port + extract later (Approach 3)?
