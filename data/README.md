---
type: data
title: Data
description: Molecular input files (.xyz, .mol, .mol2) and forcefield parameter files (.dat). Read-only inputs.
tags: [data, molecules, forcefield-params, inputs]
timestamp: 2026-08-25
---

# Data

Molecular input files and forcefield parameter files. **Read-only inputs** — do not write generated outputs here (use `debug/` or `tmp/`).

## Contents

| Path | Contents |
|------|----------|
| `AtomTypes.dat`, `BondTypes.dat`, `AngleTypes.dat`, `DihedralTypes.dat`, `ElementTypes.dat` | UFF / element parameter tables. |
| `mol/` | Molecules in `.mol` / `.mol2` format (benzene, cubane, adamantane, ...). |
| `xyz/` | Molecules in `.xyz` format. |

## Usage

These files are consumed by the topology builder (`surfmol-topology`) and forcefield assignment CLI. See `ARCHITECTURE.md` §CLI Tooling Plan.
