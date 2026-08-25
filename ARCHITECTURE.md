# SurfMol Architecture and Design Goals

## Design Philosophy
The SurfMol repository is designed with a strict modularity and language-based organization. The primary components are implemented in Rust and OpenCL for high performance, with Python bindings planned as a future layer for scripting and accessibility.

## File Naming and Organization Rules

### 1. Unique, Descriptive Names
- **No generic names** like `shared.rs`, `frontend.rs`, `utils.rs` that could exist in multiple crates
- **Use crate-specific prefixes** for library roots and shared modules to ensure uniqueness across the repository
- **Exception:** `lib.rs` is acceptable as a crate root module organizer (it is standard Rust and only appears once per crate)
- **Examples:**
  - `common/src/common.rs` (not `lib.rs`)
  - `forcefields/src/forcefields.rs` (not `lib.rs`)
  - `topology/src/topology_lib.rs` (distinguishes from `topology.rs` module)
  - `apps/src/lib.rs` — standard crate root, only `pub mod gui;`

### 2. No Tiny Stub Files
- **Avoid files with only module declarations** (e.g., `pub mod foo;` in a 2-line file) unless serving as a crate root or module organizer
- **Inline small modules** into their parent or the crate root if they don't warrant separate files
- **Exception:** Crate root files (`lib.rs`) and module organizers (`mod.rs`) may contain only `pub mod` declarations
- **Example:** `apps/src/lib.rs` containing only `pub mod gui;` is acceptable — it is a standard Rust crate root

### 3. Test Location Rules
- **Backend module tests** (no GUI, single-module focus) → `crate/tests/` directory
  - `forcefields/tests/test_rigid_sp3.rs` — tests rigid_sp3 logic
  - `forcefields/tests/test_surface.rs` — tests surface potential
  - `molrender/tests/debug_simple.rs` — tests rendering primitives
- **GUI/composite app tests** (require GUI or multiple backends) → `apps/tests/` directory
  - `apps/tests/test_thumb.rs` — tests MolThumbnailer (uses rendering + topology)

### 4. Binary Location Rules
- **GUI applications** → `apps/src/` (not `src/bin/`)
  - `apps/src/editor.rs` — 3D molecular editor
  - `apps/src/mol_browser.rs` — XYZ directory browser
- **CLI tools** → place in the backend crate they belong to, not in `apps/`
  - `topology/src/bin/assign_uff.rs` — UFF type assignment CLI
  - `forcefields/src/mol_engine.rs` — MD engine CLI with Rhai scripting

## Directory Structure
The repository is organized by language and capability:

```text
SurfMol/
├── README.md
├── docs/                # Documentation and design notes
├── opencl/              # OpenCL kernels for GPU acceleration
├── rust/                # Primary Rust workspace
│   ├── common/          # Core math, data structures, and utilities
│   ├── topology/        # Molecular graph representation, bond/angle definitions
│   ├── forcefields/     # Forcefield implementations, relaxation, and dynamics
│   └── apps/            # Applications (GUI and CLI tools)
└── libs/                # (Future) Python bindings and FFI wrappers
```

## Folder Roles & Metadata (OKF)

Every folder **MUST** have a `README.md` in [OKF format](https://okf.md/) — YAML frontmatter (required `type`, recommended `title`/`description`/`tags`/`timestamp`) + markdown body. The README is the folder's self-description for both humans and agents. Update it when the folder's role or contents change.

### Folder roles (binding)
| Folder | Role | Permanence |
|--------|------|------------|
| `userguide/` | End-user docs for **finished, polished** modules: how to run the program, features, GUI controls (mouse/keys), CLI options, usage examples, didactic theory accessible to a student. | Permanent |
| `doc/` | Permanent polished info **for developers** that helps navigation and understanding of the code. | Permanent |
| `doc/topical_audit/` | Cross-implementation maps per scientific topic (where each algorithm/feature lives across files/crates/repos). | Permanent |
| `notes/` | **Temporary** work-in-progress info about what we are doing right now. Not polished. | Ephemeral |
| `notes/chats/` | Chat logs / transcripts relevant to current work. | Ephemeral |
| `notes/designs/` | Work-in-progress design sketches and proposals. | Ephemeral |
| `notes/labbooks/` | Per-task debugging labbooks (updated continuously — see `AGENTS.md` §Testing & Validation). | Ephemeral |
| `notes/reports/` | Per-task debug/investigation reports. | Ephemeral |
| `notes/tasks/` | Active task definitions and checklists. | Ephemeral |
| `notes/ToDo_user.md` | User-facing TODO list and design decisions. | Ephemeral |
| `notes/ToDo_agents.md` | Agent-facing TODO list. | Ephemeral |
| `data/` | Molecular input files (`.xyz`, `.mol`, `.mol2`), FF parameter files (`.dat`). Read-only inputs. | Permanent |
| `opencl/` | OpenCL `.cl` kernel sources for GPU acceleration. | Permanent |
| `rust/` | Primary Rust workspace root. | Permanent |
| `rust/common/` | `surfmol-common`: core math, data structures, `DynamicAtoms`, `AlignedVec`. | Permanent |
| `rust/topology/` | `surfmol-topology`: molecular graph SSOT (bonds, angles, atom types). | Permanent |
| `rust/forcefields/` | `surfmol-forcefields`: forcefield energy/force eval, MD, relaxation, `MolWorld`. See `rust/forcefields/DESIGN.md`. | Permanent |
| `rust/molrender/` | `surfmol-molrender`: wgpu rendering primitives (meshes, gizmos, surfaces). | Permanent |
| `rust/apps/` | `surfmol-apps`: GUI applications (`editor`, `mol_browser`) + shared GUI utils. No simulation logic. | Permanent |
| `debug/` | Diagnostic plots (`.png`/`.svg`) and visual artifacts. **Gitignored** except `debug/README.md`. Cleaned regularly. | Ephemeral |

### OKF frontmatter conventions for this repo
- `type`: use one of `repository`, `userguide`, `developer-docs`, `topical-audit`, `work-notes`, `data`, `opencl-kernels`, `rust-workspace`, `rust-crate`.
- `title`: human-readable folder name.
- `description`: one sentence on the folder's role.
- `tags`: lowercase, e.g. `[forcefield, gpu, rust]`.
- `timestamp`: ISO 8601 of last significant update.


## Relation To other repos

See `Import_other_Repos.md` for the full list of reference repos (FireCore, SPAMMM, learn_Rust, blood_of_civilization) and what to import from each.

- **FireCore is the perf benchmark** — SurfMol (Rust+OpenCL) must be ≥ as fast as FireCore C++ for any ported algorithm. Measure, don't assume.

## Component Details

### 1. `surfmol-common` (rust/common)
Contains core math and fundamental data structures. This is the bedrock of the application, completely agnostic to chemistry or physics specifics.

### 2. `surfmol-topology` (rust/topology)
A lightweight, forcefield-agnostic library for creating and managing molecular graphs.
- **Responsibilities:** Define atoms, create bonds and angles (as vertices, edges, polygons).
- **Usage:** Heavily utilized by the molecular editor to represent structures. It serves as the foundation from which forcefield definitions (assigning atom-types, bond parameters, angle parameters) are derived.

### 3. `surfmol-forcefields` (rust/forcefields)
The simulation and relaxation engine.
- **Responsibilities:** Implement various forcefield energy/force evaluations, run molecular dynamics, and perform relaxations.
- **Modularity (`MolWorld`):** All forcefields are connected via a common `MolWorld` engine. This engine acts as a coordinator, combining callbacks for different flavors of interactions (bonding, non-bonding, molecule-surface interactions) within the MD-loop. This ensures minimal overhead during fine-grained parallelization (e.g., OpenMP parallelization over atoms, where each atom computes its specific interactions).
- **Data Ownership:** `MolWorld` does not own atomic positions or forces directly; these live in `DynamicAtoms` (`surfmol-common`). Each forcefield module owns only its specialized parameters and borrows shared slices during evaluation. For the full ownership model, design decisions, and future extension patterns, see `rust/forcefields/DESIGN.md`.

### 4. `surfmol-apps` (rust/apps)
High-level composite applications crate. **No simulation logic lives here.** All backends (forcefields, topology, rendering, math) are imported from sibling crates. This crate wires together multiple backend modules into user-facing GUI applications.

**Layout:**
- `src/` — executable binaries, one per GUI application. Each binary is a thin frontend that wires together backend crates.
- `src/lib.rs` — crate root (module organizer, contains only `pub mod gui;`).
- `src/gui/` — shared GUI utilities reused by multiple frontends (e.g. `MolThumbnailer`, `TrackballCam`).
  - `gui/mod.rs` — `pub mod thumbnailer;`
  - `gui/thumbnailer.rs` — `MolThumbnailer` widget.
- `tests/` — integration tests for GUI/composite functionality (tests that require multiple backend crates or GUI context).

**Current binaries:**
| Binary | Path | Description |
|--------|------|-------------|
| `mol_browser` | `src/mol_browser.rs` | XYZ directory browser with GPU thumbnail grid |
| `editor` | `src/editor.rs` | 3D molecular editor / viewer |

**Test location rule:**
- **Backend module tests** (no GUI, single-module focus) → live in their respective crate's `tests/` directory (e.g. `forcefields/tests/`, `molrender/tests/`)
- **GUI/composite app tests** (require GUI or multiple backends) → live in `apps/tests/`

**Planned:**
1. **MolBrowser:** A fast tool for searching and visualizing molecular files (`.xyz`, `.pdb`, `.mol`, `.cif`) across directories. It generates pre-rendered small texture tiles for quick thumbnail views (similar to image browsers like ACDSee or XnView).
2. **MolEdit2D:** An efficient 2D molecule drawing tool (similar to ChemSketch).
3. **MolWorld App:** A rich 3D environment allowing users to move and relax molecules on surfaces. It features:
   - **3rd-Person (God) View:** For editing and layout (like a city-builder).
   - **1st-Person (Fly) View:** For immersive navigation and interaction within complex molecular structures (like a flight simulator).

### CLI Tooling Plan (Topology → Forcefield Assignment Pipeline)
To separate topology construction and forcefield parameter assignment from the MD runtime, a lightweight CLI tool will generate pre-processed forcefield inputs:
- **Input:** `.xyz` files (geometry + elements)
- **Pipeline:** Read XYZ → Build bonds by cutoff → Enumerate angles/dihedrals/inversions → Assign UFF atom types (using octet-rule hybridization) → Output
- **Output formats:**
  - **JSON:** Human-readable topology + UFF types + parameters for debugging and version control.
  - **Binary flat arrays (`.npy` or custom):** Dense, data-oriented arrays (apos, atypes, bond indices, params) for zero-copy ingestion by the MD engine. Matches the `AlignedVec` layout used by `Uff` and `MolWorld`.
- **Rationale:** Decouples expensive topology analysis from the hot MD loop; enables batch processing of molecule libraries; makes forcefield assignments inspectable and reproducible.

### 5. `libs` (Future)
Will provide wrappers and Python bindings mapping Python calls to the underlying Rust modules. This is treated as a subsequent layer built atop a stable Rust foundation.

## Crate Naming Strategy
To integrate cleanly with the global Rust ecosystem (crates.io) and enable code sharing across repositories, crates should drop generic `mol_*` prefixes and use globally recognizable names (e.g., `surfmol-common`, `surfmol-topology`, `surfmol-forcefields`).

