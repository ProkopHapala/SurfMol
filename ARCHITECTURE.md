# SurfMol Architecture and Design Goals

## Design Philosophy
The SurfMol repository is designed with a strict modularity and language-based organization. The primary components are implemented in Rust and OpenCL for high performance, with Python bindings planned as a future layer for scripting and accessibility.

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
Contains the executable targets, including CLI bindings and Rust-based GUI applications.
Planned applications include:
1. **MolBrowser:** A fast tool for searching and visualizing molecular files (`.xyz`, `.pdb`, `.mol`, `.cif`) across directories. It generates pre-rendered small texture tiles for quick thumbnail views (similar to image browsers like ACDSee or XnView).
2. **MolEdit2D:** An efficient 2D molecule drawing tool (similar to ChemSketch).
3. **MolWorld App:** A rich 3D environment allowing users to move and relax molecules on surfaces. It features:
   - **3rd-Person (God) View:** For editing and layout (like a city-builder).
   - **1st-Person (Fly) View:** For immersive navigation and interaction within complex molecular structures (like a flight simulator).

### 5. `libs` (Future)
Will provide wrappers and Python bindings mapping Python calls to the underlying Rust modules. This is treated as a subsequent layer built atop a stable Rust foundation.

## Crate Naming Strategy
To integrate cleanly with the global Rust ecosystem (crates.io) and enable code sharing across repositories, crates should drop generic `mol_*` prefixes and use globally recognizable names (e.g., `surfmol-common`, `surfmol-topology`, `surfmol-forcefields`).
