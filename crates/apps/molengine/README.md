---
type: rust-app
title: molengine
description: CLI MD/relaxation engine — loads topology (JSON now, NPZ planned), runs forcefield evaluation and molecular dynamics via Rhai scripts.
tags: [rust, crate, cli, md, simulation, rhai]
timestamp: 2026-08-25
---

# molengine

CLI molecular simulation engine. Loads a molecular topology (with forcefield types already assigned by `buildff`), constructs a `MolWorld` from the `surfmol` integration crate, and exposes MD/relaxation operations to Rhai scripts.

## Usage

```
molengine --script <rhai_file>
```

Rhai scripts call registered functions:

```rhai
let sim = load_topology("molecule.json");
let natoms = get_natoms(sim);
let e = eval_forces(sim);
step_md(sim, 0.02, 1000.0, 0.1);
let niter = relax(sim, 1000, 0.02, 0.001, 1000.0, 0.1);
print(`Relaxed in ${niter} steps, final energy = ${e}`);
```

## Input formats

### JSON (current)
Loads topology from human-readable JSON (produced by `buildff --json`). Uses `load_topology_from_json` which calls `moltopo::export::import_json` then constructs `Uff::from_topology`.

### NPZ (planned)

**TODO:** Add `load_topology_from_npz` to load topology from NPZ files (numpy's `.npz` archive format) produced by `buildff --npz` or by Python scripts directly. NPZ is a well-recognized standard for packed numerical data that Python can create and inspect via `numpy.savez()` / `numpy.load()`.

Planned API:
```rust
// In surfmol::import (alongside existing load_topology_from_json):
pub fn load_topology_from_npz<P: AsRef<Path>>(path: P) -> Result<(Uff, Vec<String>), Box<dyn std::error::Error>>;
```

This would read the named arrays (`apos`, `atypes`, `bonds`, `angles`, `dihedrals`, `inversions`, `type_table`) directly into `Uff::new()` (which already accepts raw arrays), bypassing the `Topology` intermediate struct. This would let `molengine` drop its direct `moltopo` dependency — it would only need `surfmol` (for `MolWorld`) + `molff` (for `Uff`/`NonBondedFF`).

Rust support: `ndarray-npy` crate for reading/writing npz.

## Dependencies

- `surfmol` — `MolWorld` orchestrator (DynamicAtoms + Uff + NonBonded + Surface)
- `molff` — forcefield types (`Uff`, `NonBondedFF`)
- `moltopo` — `Topology` (used by `load_topology_from_json`; can be dropped once NPZ path is added)
- `rhai` — scripting engine
- `clap` — CLI argument parsing

## See also

- `buildff` — produces the topology JSON/NPZ that this engine consumes
- `surfmol` — provides `MolWorld` and topology loading functions
