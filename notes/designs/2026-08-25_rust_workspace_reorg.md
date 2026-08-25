---
type: work-notes
title: Rust workspace reorganization proposal
description: Notes on making the Rust workspace conform to standard Cargo conventions, with a path to publishing individual crates.
tags: [work-in-progress, design, rust, cargo, workspace]
timestamp: 2026-08-25
---

# Rust workspace reorganization proposal

**Status:** proposal — no changes made yet. This note documents the current non-standard aspects and proposes a target layout.

## 1. Current issues

### 1.1. `rust/` wrapper directory is non-standard

The Cargo workspace lives at `rust/Cargo.toml`, not at the repo root. This forces a `.cargo/config.toml` hack to redirect `target-dir` to `../../target`:

```toml
# rust/.cargo/config.toml
[build]
target-dir = "../../target"
```

Standard Rust projects either:
- Put `Cargo.toml` at the **repo root** (single-crate projects), or
- Put `Cargo.toml` at the repo root with members in a `crates/` subdirectory (multi-crate workspaces).

The `rust/` wrapper adds a nesting level with no benefit and complicates relative paths in tests (`../../data/ElementTypes.dat`).

### 1.2. Non-standard library file names

Every crate uses a custom `[lib] path = "src/xxx.rs"` instead of the default `src/lib.rs`:

| Crate | Current lib path | Standard |
|-------|-----------------|----------|
| `surfmol-common` | `src/common.rs` | `src/lib.rs` |
| `surfmol-topology` | `src/topology_lib.rs` | `src/lib.rs` |
| `surfmol-forcefields` | `src/forcefields.rs` | `src/lib.rs` |
| `surfmol-molrender` | `src/molrender.rs` | `src/lib.rs` |
| `surfmol-apps` | `src/lib.rs` | `src/lib.rs` (already standard) |

This means every `Cargo.toml` has an unnecessary `[lib]` section. It also confuses IDEs and tooling that expect `src/lib.rs`.

### 1.3. Binaries mixed into library crates

Two binaries live inside library crates:

| Binary | Currently in | Issue |
|--------|-------------|-------|
| `assign-uff` | `surfmol-topology` (`src/bin/assign_uff.rs`) | Couples a CLI tool to the library crate |
| `mol_engine` | `surfmol-forcefields` (`src/mol_engine.rs` with `[[bin]]`) | Same; also at `src/` root, not `src/bin/` |

If these crates are ever published to crates.io, the binaries get compiled whether the user wants them or not (unless `required-features` gating is added). For independent installability, each CLI tool should be its own crate.

### 1.4. `surfmol-apps` mixes lib + binaries at `src/` root

`surfmol-apps` has `src/lib.rs` (just `pub mod gui;`) plus two binaries at the `src/` root:
- `src/editor.rs` → `[[bin]] name = "editor"`
- `src/mol_browser.rs` → `[[bin]] name = "mol_browser"`

Standard convention: binaries that belong to a lib crate go in `src/bin/`. Separate application crates go in their own crate directory.

### 1.5. No crate metadata for publishing

No `[package]` section has `description`, `license`, `repository`, `readme`, `keywords`, or `categories`. These are required (or strongly recommended) for crates.io publishing.

### 1.6. Workspace dependencies not consistently used

Some dependencies are declared directly in member crates rather than via `[workspace.dependencies]`:

- `serde = "1.0"` in topology and forcefields (should be workspace dep)
- `serde_json = "1.0"` in topology and forcefields
- `image = "0.25"` in molrender (dev-dep) and apps
- `egui_extras = "0.29"` in apps

This risks version skew between crates.

### 1.7. No `benches/` directories

No criterion benchmarks exist yet. Standard location would be `crates/<name>/benches/`.

## 2. Proposed target layout

### Option A — Workspace at repo root, `crates/` subdirectory (recommended)

This is the most common pattern for multi-crate Rust projects (e.g., `tokio`, `bevy`, `wgpu`):

```
SurfMol/
├── Cargo.toml              # workspace root (NEW — at repo root)
├── Cargo.lock              # (moved from rust/)
├── .cargo/config.toml      # optional, only if custom target-dir needed
├── crates/
│   ├── surfmol-common/     # or just "common/" — see naming note below
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs      # was common.rs
│   │   └── README.md
│   ├── surfmol-topology/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs      # was topology_lib.rs
│   │   └── README.md
│   ├── surfmol-forcefields/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs      # was forcefields.rs
│   │   ├── examples/
│   │   └── README.md
│   ├── surfmol-molrender/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs      # was molrender.rs
│   │   └── README.md
│   ├── surfmol-apps/       # GUI library crate (gui module)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   └── gui/
│   │   └── README.md
│   ├── assign-uff/         # CLI tool (extracted from topology)
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   ├── mol-engine/         # CLI tool (extracted from forcefields)
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   ├── editor/             # GUI app (extracted from apps)
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   └── mol-browser/        # GUI app (extracted from apps)
│       ├── Cargo.toml
│       └── src/main.rs
├── data/                   # stays at root
├── opencl/                 # stays at root
├── doc/
├── notes/
├── ...
```

**Workspace `Cargo.toml` at root:**
```toml
[workspace]
members = [
    "crates/surfmol-common",
    "crates/surfmol-topology",
    "crates/surfmol-forcefields",
    "crates/surfmol-molrender",
    "crates/surfmol-apps",
    "crates/assign-uff",
    "crates/mol-engine",
    "crates/editor",
    "crates/mol-browser",
]
resolver = "2"

[workspace.dependencies]
# ... (same as current, plus serde, serde_json, image, egui_extras)
```

### Option B — Workspace at repo root, flat member directories

Simpler but less conventional for large projects:

```
SurfMol/
├── Cargo.toml
├── common/
├── topology/
├── forcefields/
├── molrender/
├── apps/
├── ...
```

**Not recommended** — flat layout at repo root mixes Rust crates with non-Rust directories (`data/`, `opencl/`, `doc/`, `notes/`) and gets messy as the project grows.

### Option C — Keep `rust/` but fix everything else

Minimal change: keep `rust/Cargo.toml` as workspace root, but fix lib names, extract binaries, add metadata. This addresses issues 1.2–1.6 but not 1.1.

**Only choose this** if there's a strong reason to keep Rust separate from the repo root (e.g., the root will eventually host Python/other-language workspaces side by side).

## 3. Directory naming: `crates/surfmol-foo/` vs `crates/foo/`

Two conventions exist in the Rust ecosystem:

| Convention | Example projects | Pros | Cons |
|-----------|-----------------|------|------|
| `crates/surfmol-foo/` (matches crate name) | `wgpu`, `tracing` | `use surfmol_common::...` matches path | Verbose directory names |
| `crates/foo/` (short dir, crate name in Cargo.toml) | `tokio`, `bevy` | Shorter paths | Directory ≠ crate name |

**Recommendation:** Use `crates/surfmol-foo/` (matches crate name). This makes `cargo add surfmol-common` and the filesystem path unambiguous, and is the convention used by `wgpu` (which this project already depends on).

## 4. Splitting `surfmol-common` for independent publishability

`surfmol-common` currently bundles four concerns:

| Module | Dependencies | Independent value |
|--------|-------------|-------------------|
| `math/` (Vec3d, Quat4d, fastmath, math3d, math4d) | `bytemuck` only | **High** — pure math, useful beyond SurfMol |
| `util.rs` (AlignedVec) | std only | **Medium** — useful but small |
| `xyz.rs` (XYZ I/O) | std only | **Medium** — useful but small |
| `molecular.rs` (Atoms, DynamicAtoms, MD integrator) | `math` + `util` | **Low** — SurfMol-specific |

**Proposal:** Split into two crates:

- **`surfmol-math`** — `math/` modules + `util.rs` (AlignedVec). Zero SurfMol-specific code. Publishable as a standalone `#[repr(C)]` math library with alignment-aware allocators. This is the most independently useful crate.
- **`surfmol-common`** — `xyz.rs` + `molecular.rs`. Depends on `surfmol-math`. Contains SurfMol-specific data structures.

This split is **optional** — it only matters if you want to publish `surfmol-math` independently. If you keep it as one crate, that's fine too.

## 5. Extracting binaries into separate crates

### 5.1. `assign-uff` (currently in `surfmol-topology`)

**Current:** `rust/topology/src/bin/assign_uff.rs` — 290 LOC, depends on `surfmol-common::xyz`, `surfmol-topology::{builder, assign_uff, topology}`.

**Proposed:** `crates/assign-uff/` — its own crate with:
```toml
[package]
name = "assign-uff"
version = "0.1.0"
edition = "2021"
description = "CLI tool: build molecular topology from XYZ and assign UFF atom types"
# ...metadata

[dependencies]
surfmol-common = { path = "../surfmol-common", version = "0.1.0" }
surfmol-topology = { path = "../surfmol-topology", version = "0.1.0" }
serde = { workspace = true }
serde_json = { workspace = true }
```

**Benefit:** `cargo install assign-uff` works without pulling in the rest of the workspace. Users who only need topology assignment don't need forcefields/render apps.

### 5.2. `mol-engine` (currently in `surfmol-forcefields`)

**Current:** `rust/forcefields/src/mol_engine.rs` — 90 LOC, depends on `rhai`, `clap`, `surfmol-forcefields`.

**Proposed:** `crates/mol-engine/` — its own crate with `src/main.rs`.

**Benefit:** `cargo install mol-engine` gives users the Rhai-scripted MD engine without the GUI deps.

### 5.3. `editor` and `mol-browser` (currently in `surfmol-apps`)

**Current:** `src/editor.rs` (1153 LOC) and `src/mol_browser.rs` (249 LOC) are `[[bin]]` targets in `surfmol-apps`, which is also a lib (`src/lib.rs` = `pub mod gui;`).

**Proposed:**
- `crates/surfmol-apps/` — library crate containing only `gui/` modules (gizmos, kekule_editor, thumbnailer, trackball). This is the reusable GUI toolkit.
- `crates/editor/` — binary crate, `src/main.rs` (was `editor.rs`). Depends on `surfmol-apps` + forcefields + molrender.
- `crates/mol-browser/` — binary crate, `src/main.rs` (was `mol_browser.rs`). Depends on `surfmol-apps` + molrender.

**Benefit:** The GUI toolkit (`surfmol-apps` lib) can be reused by future apps without pulling in the `editor` binary's heavy deps.

## 6. Crate metadata to add

For each crate that might be published, add to `[package]`:

```toml
[package]
name = "surfmol-common"
version = "0.1.0"
edition = "2021"
description = "Common math, data structures, and I/O for SurfMol molecular simulation"
license = "MIT"                          # or Apache-2.0, or dual
repository = "https://github.com/<user>/SurfMol"
readme = "README.md"
keywords = ["molecular-dynamics", "forcefield", "chemistry", "scientific"]
categories = ["science", "simulation"]
```

Also add to root `Cargo.toml`:
```toml
[workspace.package]
license = "MIT"
repository = "https://github.com/<user>/SurfMol"
edition = "2021"
```

Then each member can inherit: `license.workspace = true`, etc.

## 7. Workspace dependency consolidation

Move all shared deps to `[workspace.dependencies]` in the root `Cargo.toml`:

```toml
[workspace.dependencies]
# Existing
eframe = { version = "0.29", ... }
egui = "0.34"
# ...

# Add these (currently declared per-crate):
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
image = { version = "0.25", default-features = false, features = ["png"] }
egui_extras = "0.29"
clap = { version = "4.5", features = ["derive"] }
rhai = "1.19"
```

Then in member crates: `serde = { workspace = true }` instead of `serde = { version = "1.0", ... }`.

## 8. Migration steps (when ready to execute)

These should be done as a single atomic reorganization (one PR, one commit). Order matters:

1. **Backup:** `git stash` or create a branch `reorg/cargo-layout`.
2. **Create root `Cargo.toml`** with workspace members pointing to `crates/`.
3. **Move `rust/<crate>/` → `crates/surfmol-<crate>/`** for each member.
4. **Rename lib files:** `src/common.rs` → `src/lib.rs`, `src/topology_lib.rs` → `src/lib.rs`, etc. Remove `[lib] path = ...` from each `Cargo.toml`.
5. **Extract binaries:**
   - `rust/topology/src/bin/assign_uff.rs` → `crates/assign-uff/src/main.rs`
   - `rust/forcefields/src/mol_engine.rs` → `crates/mol-engine/src/main.rs`
   - `rust/apps/src/editor.rs` → `crates/editor/src/main.rs`
   - `rust/apps/src/mol_browser.rs` → `crates/mol-browser/src/main.rs`
   - Keep `rust/apps/src/gui/` → `crates/surfmol-apps/src/gui/`
6. **Move `rust/Cargo.lock` → `Cargo.lock`** (root).
7. **Delete `rust/.cargo/config.toml`** (no longer needed — target goes to `./target` at root by default).
8. **Delete `rust/` directory** if empty.
9. **Update test data paths:** `../../data/` → `../data/` (one level less nesting).
10. **Update `CODEMAP.md`, `AGENTS.md`, `ARCHITECTURE.md`** to reflect new paths.
11. **`cargo check --workspace`** to verify.
12. **`cargo test --workspace`** to verify all tests pass with new paths.

### Optional (separate PR, lower priority):

13. Split `surfmol-common` into `surfmol-math` + `surfmol-common` (see §4).
14. Add crate metadata (§6).
15. Consolidate workspace deps (§7).
16. Add `benches/` with criterion benchmarks.

## 9. Impact on existing files

Files that reference paths and would need updating:

| File | Current reference | New reference |
|------|------------------|---------------|
| `CODEMAP.md` §2 | `rust/Cargo.toml`, `rust/.cargo/config.toml` | `Cargo.toml` (root) |
| `CODEMAP.md` §3 | `rust/common/src/common.rs` etc. | `crates/surfmol-common/src/lib.rs` etc. |
| `CODEMAP.md` §6 | "run from `rust/`" | "run from repo root" |
| `AGENTS.md` | `rust/forcefields/DESIGN.md` | `crates/surfmol-forcefields/DESIGN.md` |
| `ARCHITECTURE.md` | all `rust/` paths | `crates/` paths |
| `Import_other_Repos.md` | `rust/` paths | `crates/` paths |
| Test files | `../../data/ElementTypes.dat` | `../data/ElementTypes.dat` |
| `rust/README.md` | (would be deleted or merged into root README) | — |

## 10. Open questions

1. **License:** What license will the crates use? (MIT, Apache-2.0, dual, GPL?)
2. **Repository URL:** What is the canonical GitHub URL?
3. **Should `surfmol-common` be split?** (§4 — only needed if publishing math independently)
4. **Keep `rust/` or not?** (Option A vs Option C — depends on whether non-Rust code will live at root)
5. **Versioning:** All crates at `0.1.0` currently. Should they be versioned independently after split?
