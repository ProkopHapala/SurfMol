# SurfMol Design Goals

This document captures the **scientific and engineering goals** that drive SurfMol's design. For crate layout, file-naming rules, and directory structure see `ARCHITECTURE.md`. For what we import from other repos see `Import_other_Repos.md`. For the user-facing TODO list see `notes/ToDo_user.md`.

## 1. Mission

Build a **compiled, GPU-first, debuggable** on-surface molecular manipulation and global-optimization platform that:
- Replaces the Python orchestration overhead of SPAMMM with Rust.
- Matches or **exceeds the performance of the FireCore C++ reference** (the benchmark — see `Import_other_Repos.md` §1).
- Keeps a clean, modular GUI optimized for **debugging different forcefields** side-by-side.

**Primary languages:** Rust (logic, GUI, orchestration) + OpenCL (GPU). Python is a minor layer only (support scripts, quick illustrations).

## 2. Initial Focus — RigidAtomFF (RAFF)

The main focus at the start is **RigidAtomFF (RAFF)**: a rigid-body forcefield where each atom is a frame with **ports** rotated by quaternion rigid-body dynamics (as-rigid-as-possible, ARAP — see FireCore `pyBall/RigidAtomFF/RRsp3/`). Ports interact with the position of a fixed neighbor atom.

### 2.1 Bonding via atom-frame ports
- Ports are rotated by quaternion rigid-body dynamics (ARAP paper).
- Each port interacts with the position of a fixed neighbor atom.
- **Capping atoms** (e.g. hydrogen, epairs) have **no ports** and are a **rigid appendix fixed to a given port of the host atom** (no independent DOF). Simpler, fewer DOF, faster. Revisit if H-relaxation fidelity becomes an issue.

### 2.2 Two RAFF variants
1. **Fixed topology** — each port interacts with exactly one neighbor atom (1-to-1 bijective map: port *k* of atom *i* connects to atom *j*) via a harmonic potential.
2. **Reactive forcefield** — each port can interact with all atoms in proximity via a **dissociative potential** (Morse, or its fast polynomial approximation). Reference: FireCore `RARFF_SR.h`.

### 2.3 Nonbonding acceleration
- **AABB bounding boxes** for fragments, optimized to GPU workgroup/local memory with **16 / 32 / 64 / 128 atoms per fragment** (matches SPAMMM kernel tiling — see `Import_other_Repos.md` §2).
- **Fragment memory layout: contiguous fragments** — each fragment's atoms (node + capping) are contiguous in memory for best cache/workgroup locality. Trade-off: harder to update topology, accepted.
- **Fast collision** uses a spatially-linearized potential: harmonic spring-like `E = k(|r_i − r_j| − R0_ij)^2` at short distance, transitioning to a polynomial Morse approximation at far distance.

## 3. Relaxation over Dynamics — Projective / Position-Based

Projective Dynamics and Position-Based Dynamics are first-class, **optimized for relaxation rather than molecular dynamics**. References to port:
- FireCore `cpp/common/math/ProjectiveDynamics_d.h` (position-based dynamics for stiff springs, implicit, stable).
- FireCore `pyBall/RigidAtomFF/RRsp3/` (cluster-sorted rigid PBD with ARAP ports).
- SPAMMM `kernels/LFF.cl` + `LFFSolver.py` (linearized projective Jacobi — fast relaxation surrogate).

## 4. GUI Optimized for Forcefield Debugging

The GUI must efficiently **bind and visualize arrays of atoms and bonds** where these arrays can come from **different forcefields**. We need a **uniform representation shared between forcefields** so they can communicate, even though the internal representations of UFF, RigidAtomFF, and RigidMoleFF are very different.

- **SSOT for topology** lives in `surfmol-topology` (the `AtomicGraph` equivalent — see SPAMMM `spammm/topology/AtomicGraph.py`); all forcefield param arrays derive from it.
- **`MolWorld`** does not own positions/forces — those live in `DynamicAtoms` (`surfmol-common`); each forcefield owns only its specialized params and borrows shared slices. See `rust/forcefields/DESIGN.md`.
- **GUI stack:** wgpu + winit + egui (14 MiB stripped release — validated by the existing `editor` binary; see `Import_other_Repos.md` §4 GUI decision). **Do not adopt Bevy.**
- **OpenCL-GL zero-copy interop** for rendering GPU-resident atom arrays directly (from learn_Rust `demo06`).

## 5. Performance Goals

- **Match or beat FireCore C++** for every ported algorithm. `MolWorld_sp3::MDloop()` (~1–10 μs/iter for small systems) is the reference. Measure with a real benchmark harness (FireCore uses `getCPUticks()`; SurfMol should add `std::time::Instant` + a `cargo xtask bench` target).
- **Rust is the engine, OpenCL is the accelerator.** CPU Rust references are authoritative for correctness; GPU must match CPU within tolerance.
- **NVIDIA GPU preferred** for all OpenCL timings (port SPAMMM `OpenCLBase.select_device(preferred_vendor='nvidia')`). Never report PoCL/CPU timings as GPU.
- **Data-oriented layouts:** flat `float4` arrays with `.w` channel reuse (energy, clash flags, secondary results); workgroup-sized fragments (16/32/64/128 atoms); SoA with 64-byte alignment (mirror FireCore `UFF.h` / learn_Rust `AlignedVec`).
- **Fuse secondary checks into existing kernels** — if a kernel already computes a distance/overlap, add clash/collision flags in the same loop; never recompute on host.
- **Minimal orchestration:** no hot loops in Python; push compute to OpenCL kernels.

## 6. Build & Binary Footprint Goals

Adopted from blood_of_civilization's `Memory_Issues` analysis (see `Import_other_Repos.md` §4 for full detail):

- **Cargo profiles** (`debug=1`, `strip="debuginfo"`, release `lto="thin"` + `codegen-units=1` + `incremental=true` + `debug-assertions=true` + `overflow-checks=true`) — keeps panic backtraces while cutting binary size ~10×.
- **Shared `target-dir`** in `~/.cargo/config.toml` (~91% disk reduction across projects).
- **IDE indexing guard** (`.codeiumignore` / `.vscode/settings.json`) so the language server does not index multi-GB build artifacts.
- **Unsafe isolation:** all `unsafe` confined to a single feature-gated OpenCL crate (`#![cfg_attr(not(feature = "opencl"), forbid(unsafe_code))]`); no raw handle escapes the crate; every `unsafe` block has a SAFETY comment.
- **Dependency discipline:** audit heavy crates; prefer light alternatives (e.g. `png` over `image` for PNG-only use); avoid Bevy.

## 7. Reusable Architecture Goals

- **Inventory first** — before writing anything, check sibling crates, `CODEMAP.md`, and the reference repos in `Import_other_Repos.md`.
- **Composability over bloat** — `MolWorld` coordinates forcefield callbacks (bonding, non-bonding, surface) within the MD/relaxation loop; each forcefield owns only its params.
- **Separation of concerns** — no plotting/rendering in core libraries; GUI in `surfmol-apps` only; CLI tools in their backend crate's `src/bin/`.
- **Generalization over duplication** — generalize an existing function if it almost fits; if a risky major change threatens backward compatibility, stop and ask for approval.
- **Parity citations** — when porting from FireCore (C++), SPAMMM (Python), or learn_Rust, cite the reference file + function in a comment.

## 8. Planned Applications

1. **MolBrowser** — fast `.xyz`/`.pdb`/`.mol`/`.cif` directory browser with GPU thumbnail grid.
2. **MolEdit2D** — efficient 2D molecule drawing (ChemSketch-like).
3. **MolWorld App** — rich 3D environment to move and relax molecules on surfaces:
   - **3rd-person (God) view** for editing and layout (city-builder-like).
   - **1st-person (Fly) view** for immersive navigation inside complex structures (flight-simulator-like).

## 9. CLI Tooling — Topology → Forcefield Assignment Pipeline

Decouple topology construction and forcefield parameter assignment from the MD runtime:
- **Input:** `.xyz` (geometry + elements).
- **Pipeline:** Read XYZ → build bonds by cutoff → enumerate angles/dihedrals/inversions → assign UFF atom types (octet-rule hybridization) → output.
- **Outputs:** JSON (human-readable, debuggable, VCS-friendly) + binary flat arrays (`.npy` or custom) for zero-copy ingestion by the MD engine, matching the `AlignedVec` layout used by `Uff` and `MolWorld`.

## 10. Resolved Design Decisions

1. **OpenCL crate:** **`ocl` 0.19** (from learn_Rust). Higher-level ProQueue/Buffer API; OpenCL-GL interop demo already works with it. Adopt the unsafe-isolation-in-a-single-feature-gated-crate pattern from blood_of_civilization regardless.
2. **Fragment memory layout:** **contiguous fragments** — node + capping atoms contiguous per fragment for best cache/workgroup locality.
3. **Capping atoms (H, epairs):** **rigid appendix fixed to a host-atom port** (no independent DOF).
