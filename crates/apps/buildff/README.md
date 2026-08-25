---
type: rust-app
title: buildff
description: CLI tool that builds molecular topology from XYZ, assigns forcefield types (UFF now, more later), and exports JSON or binary for the MD engine.
tags: [rust, crate, cli, topology, forcefield, uff]
timestamp: 2026-08-25
---

# buildff

CLI tool that takes an XYZ file, builds molecular topology (bonds, angles, dihedrals, inversions) via covalent-radius heuristics, assigns forcefield atom types (currently UFF), and exports the result for consumption by `molengine` or other tools.

## Usage

```
buildff <xyz_file> [options]

Options:
  --json <path>       Write human-readable topology JSON
  --bin  <path>       Write flat binary arrays for MD ingestion
  --tol  <f>          Covalent radius tolerance (default 0.4 Å)
  --rcut <f>          Override with global cutoff (ignores radii)
```

## Output formats

### JSON (`--json`)
Human-readable topology with per-atom info (element, position, UFF type, hybridization, neighbors) and bond/angle/dihedral/inversion lists. Consumed by `molengine` via `load_topology_from_json`.

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

- `moltopo` — topology building (`Builder`), UFF type assignment (`assign_uff`), XYZ I/O
- `numcore` — math primitives
- `serde` / `serde_json` — JSON export

## See also

- `molengine` — consumes the topology output and runs MD/relaxation
- `moltopo` — provides the topology building and type assignment logic
