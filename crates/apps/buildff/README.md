---
type: rust-app
title: buildff
description: CLI builder — XYZ → topology → UFF type assignment → TopologyData JSON or binary export. Stateless, no forcefield evaluation. Consumed by molengine (the runner).
tags: [rust, crate, cli, topology, forcefield, uff, builder]
timestamp: 2026-08-29
---

# buildff

CLI builder that takes an XYZ file, builds molecular topology (bonds, angles, dihedrals, inversions) via covalent-radius heuristics, assigns forcefield atom types (currently UFF), and exports the result for consumption by `molengine` (the runner) or other tools.

## Role in the pipeline

`buildff` is the **builder** in SurfMol's builder/runner architecture (see [`crates/apps/README.md`](../README.md)). It is a stateless one-shot CLI that prepares topology — it performs **no forcefield evaluation**. The runner (`molengine`) loads the exported topology and runs simulation.

```
buildff (builder)                molengine (runner)
  XYZ → topo.json  ─────────────►  topo.json + script.rhai → relaxed.xyz
```

## Usage

```
buildff <xyz_file> [options]

Options:
  --json <path>       Write TopologyData JSON (canonical format, consumed by molengine)
  --bin  <path>       Write flat binary arrays for MD ingestion
  --tol  <f>          Covalent radius tolerance (default 0.4 Å)
  --rcut <f>          Override with global cutoff (ignores radii)
```

## Output formats

### JSON (`--json`)

Canonical `TopologyData` JSON format (via `moltopo::export::Topology::export_json`). Flat arrays — no nested per-atom objects. Consumed by `molengine` via `load_topology_from_json` → `moltopo::export::import_json`.

```json
{
  "natoms": 5,
  "elements": ["C", "H", "H", "H", "H"],
  "positions": [[0,0,0], [0.629,0.629,0.629], ...],
  "bonds": [[0,1], [0,2], [0,3], [0,4]],
  "angles": [[1,0,2], [1,0,3], ...],
  "dihedrals": [],
  "inversions": [],
  "bond_params": [],
  "angle_params": [],
  "dihedral_params": [],
  "inversion_params": [],
  "atom_params": []
}
```

The `*_params` arrays are left empty by `buildff` — the runner fills them at runtime via `setup_uff_params(sim, "data/")` which loads `.dat` parameter files. This separation keeps the builder stateless (no dependency on parameter files) and allows the runner to switch parameter sets without rebuilding the topology.

**Format change (2026-08-29):** Previously `buildff` wrote a custom `TopologyJson` format with per-atom objects (`atoms: [{index, element, position, uff_type, hybridization, neighbors}, ...]`). This has been replaced with the canonical `TopologyData` format (flat arrays) to match what `molengine`'s `load_topology_from_json` expects. The UFF type histogram is still printed to stdout for diagnostic purposes.

### Binary (`--bin`)

Custom flat binary format (`UFFTOPO` magic header + sequential f64/i32 arrays). Designed for zero-copy ingestion by the MD engine.

## Planned: NPZ format

**TODO:** Replace the custom `UFFTOPO` binary format with **NPZ** (numpy's zip-based archive format, `.npz`). NPZ is a well-recognized standard for packed numerical data that Python can create and inspect directly via `numpy.savez()` / `numpy.load()`.

Planned named arrays inside the `.npz` archive:
- `apos` — `(N, 3) f64` atom positions
- `atypes` — `(N,) i32` atom type indices
- `bonds` — `(Nb, 2) i32` bond atom pairs
- `angles` — `(Na, 3) i32` angle atom triples
- `dihedrals` — `(Nd, 4) i32` dihedral atom quads
- `inversions` — `(Ni, 4) i32` inversion atom quads
- `type_table` — `(Nt, 8) u8` fixed-width type name strings

Benefits:
- Python interop: `numpy.load("topo.npz")` gives direct access to all arrays
- Inspectable: `unzip -l topo.npz` shows array names
- Rust support: `ndarray-npy` crate reads/writes npz
- `molengine` would gain a `load_topology_from_npz` function alongside the existing `load_topology_from_json`

## Dependencies

- `moltopo` — topology building (`Builder`), UFF type assignment (`assign_uff`), XYZ I/O, `Topology::export_json`
- `numcore` — math primitives

## See also

- [`crates/apps/README.md`](../README.md) — builder/runner architecture overview
- [`molengine`](../molengine/README.md) — consumes the topology output and runs MD/relaxation
- `moltopo` — provides the topology building and type assignment logic
