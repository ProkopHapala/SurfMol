---
type: userguide
title: "Editor — Interactive Molecular Editor & On-Surface MD Simulator"
description: End-user guide for the SurfMol editor — how to run it, CLI options, GUI controls, keyboard shortcuts, and usage examples with different molecules.
tags: [user-guide, editor, gui, cli, molecular-dynamics, raff, aabb, broad-phase]
timestamp: 2026-09-29
---

# Editor — Interactive Molecular Editor & On-Surface MD Simulator

The SurfMol editor is an interactive 3D molecular editor and on-surface molecular dynamics simulator. It supports three forcefield modes (UFF, RigidSp3, RAFF), non-bonded interactions (LJ + Coulomb), NaCl surface potentials, multi-molecule systems with AABB broad-phase collision, and a hex-grid Kekule editor for building molecules from scratch.

## Prerequisites

- Rust toolchain (stable, 2021 edition)
- Linux with X11 or Wayland (for wgpu rendering)
- GPU drivers (wgpu supports Vulkan, Metal, DX12 — on Linux typically Vulkan)

## Building

```bash
cd /home/prokop/git/SurfMol
cargo build -p editor
```

The binary is at `target/debug/editor` (or `~/.cargo/shared_target/debug/editor` if `CARGO_TARGET_DIR` is set).

## Running

```bash
cargo run -p editor                              # default: benzene.xyz, 2 molecules
cargo run -p editor -- data/xyz/benzoic_acid.xyz # specific molecule
cargo run -p editor -- [file.xyz] [CLI flags]    # full control
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `[file.xyz]` (positional) | `data/xyz/benzene.xyz` | Input molecule in XYZ format |
| `--nmols N` | `2` | Number of molecule copies to spawn. Each copy is a separate cluster for broad-phase AABB collision. |
| `--layout L` | `lattice` | Molecule placement strategy: `lattice` (grid with tight touching AABBs + collision margin) or `random` (non-overlapping random placement with reproducible LCG seed) |
| `--spacing S` | `12.0` | Extra spacing between molecules in lattice layout (Å) |
| `--show-aabb` | off | Render cluster bounding boxes: green = tight AABB, red = expanded by rcut (overlap test region) |
| `--raff` | off | Start in RAFF mode (simulation, not Kekule editor; show ports; enable non-bonded; disable surface; damping=0.1, per_frame=20) |
| `--2d` | off | Flatten atoms to z=0 plane, constrain forces/velocities/positions to 2D |
| `--atom-scale S` | `0.25` | Atom render size multiplier (range 0.05–0.5; also adjustable via GUI slider) |
| `--perFrame N` | `100` (or `20` with `--raff`) | MD iterations per render frame |
| `--dt T` | `0.02` | MD timestep |
| `--group-size N` | `32` | Group size for replicated copies |

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `SPACE` | Start/stop relaxation |
| `H` | Toggle help panel |
| `B` | Toggle bonds |
| `S` | Toggle surface |
| `G` | Toggle group AABBs |
| `T` | Toggle ports |
| `K` | Toggle labels |
| `L` | Cycle label mode (None → AtomNumber → AtomType → Charge → ElementName) |
| `D` | Toggle debug cursor |
| `A` | Toggle AABB bounding boxes (broad-phase visualization) |
| `P` | Pin/unpin picked atom |
| `C` | Reset camera |
| `E` | Toggle Kekule editor panel |
| `F` | Cycle bonded FF: Uff → RigidSp3 → RAFF |
| `N` | Toggle non-bonded (LJ+Coulomb) |
| `M` | Toggle surface FF (NaCl) |
| `1`–`4` | Edit modes: Select / HexPaint / HexToggle / AtomDraw |
| `[` / `]` | Decrease/increase pick_k (spring dragging stiffness) |
| `-` / `=` | Decrease/increase per_frame |
| `ESC` | Deselect atom |

## Mouse controls

| Action | Effect |
|--------|--------|
| LMB click atom | Pick/unpick atom (Select mode) |
| LMB drag atom | Pull atom with spring force (sim mode) |
| LMB click | Hex paint/toggle/atom draw (Edit mode) |
| RMB click | Unpick (Select) / remove hex (Edit) |
| Shift+LMB drag | Pan camera |
| RMB drag | Rotate camera (trackball) |
| Scroll | Zoom |

## GUI panels

- **SurfMol** (top-left): Title + live energy display (Etotal, bond, angle, dihed, inv, nb, surf).
- **Atom Info** (top-right): Selected atom info (element, UFF type, charge, position, RvdW, pin status).
- **Settings** (right): Physics (iters/frame, dt, damping, zero-V-on-opposition), Display (atom size slider, label mode, bonded FF mode), non-bonded/surface status, **AABB checkbox**, cluster count + BP pair count.
- **RAFF Settings** (right, RAFF mode only): Non-bonded toggle, 2D plane, orient mode (Adiabatic/Dynamic), rcut/k_coll/f_max sliders, exclusion checkboxes, live energy display.
- **Kekule Editor** (left, edit mode): Edit mode selector, atom type, ribbon generator, bake/export buttons.
- **Help** (bottom-left): Keyboard shortcuts + CLI flags.
- **Status** (bottom-center): Relaxation ON/OFF indicator.

## Usage examples

### Example 1: Single molecule relaxation (UFF mode)

```bash
cargo run -p editor -- data/xyz/benzene.xyz --nmols 1
```

Loads a single benzene molecule in UFF mode with NaCl surface. Press `SPACE` to start relaxation. The molecule will settle onto the surface.

### Example 2: Benzoic acid dimer with AABB visualization (RAFF mode)

```bash
cargo run -p editor -- data/xyz/benzoic_acid.xyz --nmols 2 --show-aabb --raff --2d
```

Loads 2 benzoic acid molecules in RAFF mode, 2D plane, with AABB bounding boxes visible:
- **Green boxes**: tight per-molecule AABBs (fitted to atom positions)
- **Red boxes**: AABBs expanded by rcut (8 Å) — the overlap test region
- The editor reports `[BroadPhase] 2 clusters, rcut=8 Å, 1 overlapping pairs` at startup

Press `SPACE` to relax. The two molecules will interact via LJ + Coulomb + collision forces, but only atom pairs whose cluster AABBs overlap are evaluated — producing identical results to O(N²) but faster.

### Example 3: 4 molecules with random placement

```bash
cargo run -p editor -- data/xyz/benzoic_acid.xyz --nmols 4 --layout random --show-aabb --raff --2d
```

Places 4 benzoic acid molecules at random non-overlapping positions (reproducible LCG seed). The editor reports `[BroadPhase] 4 clusters, rcut=8 Å, 2 overlapping pairs` — only 2 of the 6 possible cluster pairs are close enough to interact.

### Example 4: Pyrrol cluster (3 molecules, lattice layout)

```bash
cargo run -p editor -- data/xyz/pyrrol.xyz --nmols 3 --layout lattice --show-aabb --raff --2d
```

Loads 3 pyrrol molecules in a grid layout. Reports `[BroadPhase] 3 clusters, rcut=8 Å, 3 overlapping pairs`.

### Example 5: Build a molecule from scratch (Kekule editor)

```bash
cargo run -p editor
```

Starts in Kekule editor mode (default, unless `--raff`). Use the hex grid to:
1. Press `2` for HexPaint mode, click hexes to paint carbon atoms
2. Press `4` for AtomDraw mode to place specific atom types (C/N/O/H)
3. Toggle "Auto H" and "Auto Bonds" for automatic hydrogen capping and bond detection
4. Click "Bake to Sim" to convert the builder graph into a simulation topology
5. Press `SPACE` to relax the baked molecule

### Example 6: Adjusting atom render size

```bash
cargo run -p editor -- data/xyz/benzoic_acid.xyz --atom-scale 0.15
```

Smaller atoms (0.15 vs default 0.25) — useful for seeing bonds and AABBs more clearly. Also adjustable via the GUI slider in the Settings panel.

## Available molecule files

Molecules are in `data/xyz/` (XYZ format) and `data/mol/` (MOL/MOL2 format — not yet supported by the editor, only XYZ):

| File | Atoms | Description |
|------|-------|-------------|
| `data/xyz/benzene.xyz` | 6 | Benzene ring (C6H6, planar) |
| `data/xyz/benzoic_acid.xyz` | 15 | Benzoic acid (C7H6O2, flat z=0) |
| `data/xyz/pyrrol.xyz` | 10 | Pyrrole (C4H5N, 5-membered ring) |
| `data/xyz/water.xyz` | 3 | Water (H2O) |
| `data/xyz/eicosanediol.xyz` | 62 | Eicosane-1,2-diol (long chain) |

To use your own molecule, create an XYZ file:
```
<natoms>
<comment line>
<element> <x> <y> <z>
...
```

## Forcefield modes

The editor supports three bonded forcefield modes, cycled with the `F` key:

| Mode | Description | When to use |
|------|-------------|-------------|
| **UFF** | Universal Force Field — explicit bonds, angles, dihedrals, inversions | General molecules, aromatic systems (sp2) |
| **RigidSp3** | Legacy port-based rigid body (sp3 only) | Tetrahedral centers (CH4-like) |
| **RAFF** | Rigid-Atom Force Field — port springs replace angles/dihedrals | Rigid-body dynamics, multi-molecule collision, GPU-ready |

RAFF mode is recommended for multi-molecule collision simulations. Start with `--raff` to enter RAFF mode directly.

## Broad-phase AABB collision

When `--nmols N` with N > 1, the editor creates a `BroadPhase` struct that holds per-molecule AABBs. Each relaxation step:

1. **Rebuild**: AABBs are refitted from current atom positions (`BroadPhase::rebuild`)
2. **Pair finding**: overlapping cluster pairs are found via `broad_phase_pairs` (O(N²) over clusters, margin-expanded AABB overlap test)
3. **Force eval**: only atom pairs in overlapping clusters are evaluated (`eval_broad` / `eval_nonbonded_broad`)

This produces **identical** forces and energy as the O(N²) all-pairs method — it's purely a culling optimization. Parity is verified by tests in `crates/libs/molff/tests/test_broad_phase.rs`.

The AABB margin is the non-bonded cutoff (8 Å by default). Two molecules whose AABBs don't overlap even after expanding by 8 Å have zero non-bonded interaction and are skipped entirely.

### Visualization

Press `A` or use `--show-aabb` to see:
- **Green wireframe boxes**: tight per-molecule AABBs (fitted to atom positions)
- **Red wireframe boxes**: AABBs expanded by rcut — the overlap test region. If two red boxes overlap, the molecule pair is evaluated.

The Settings panel shows live cluster count and BP pair count.

## Theory: What is a forcefield?

A **forcefield** is a set of mathematical functions that compute the potential energy and forces on each atom based on their positions. The editor uses:

- **Bonded terms** (UFF/RAFF): bond stretching, angle bending, dihedral torsion, inversion — keep the molecule's shape
- **Non-bonded terms** (LJ + Coulomb): Lennard-Jones van der Waals + Coulomb electrostatics — intermolecular interactions
- **Surface terms** (NaCl): electrostatic + Pauli + London dispersion from the substrate

The **relaxation** loop (SPACE) iteratively evaluates forces and moves atoms along them (damped molecular dynamics) until the system reaches a local energy minimum.

## Theory: What is AABB broad-phase collision?

**Axis-Aligned Bounding Boxes (AABBs)** are the simplest 3D bounding volume — a box aligned with the x/y/z axes, defined by a minimum and maximum corner. Each molecule gets one AABB enclosing all its atoms.

**Broad-phase collision** is a two-stage filtering strategy:
1. **Broad phase** (cheap): find which pairs of molecules *might* collide by testing AABB overlap
2. **Narrow phase** (expensive): for each candidate pair, evaluate exact atom-atom interactions

This avoids the O(N²) atom-atom distance check when molecules are far apart. The margin expansion (rcut) ensures we don't miss any interacting pairs — if two expanded AABBs don't overlap, no atom pair can be within the cutoff.

## See also

- [`/crates/apps/editor/README.md`](/crates/apps/editor/README.md) — developer README for the editor crate
- [`/doc/topical_audit/spatial_acceleration.md`](/doc/topical_audit/spatial_acceleration.md) — cross-implementation map for AABB/broad-phase
- [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) — RAFF forcefield cross-implementation map
- [`/notes/designs/cluster_aabb_collision.md`](/notes/designs/cluster_aabb_collision.md) — design document for broad-phase collision
- [`/notes/designs/raff_theory_equations.md`](/notes/designs/raff_theory_equations.md) — RAFF mathematical formulations
- [`/CODEMAP.md`](/CODEMAP.md) — repo structure and crate dependency graph
