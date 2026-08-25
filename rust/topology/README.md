---
type: rust-crate
title: surfmol-topology
description: Lightweight forcefield-agnostic molecular graph SSOT — atoms, bonds, angles, atom types. Foundation for forcefield param assignment.
tags: [rust, crate, topology, molecular-graph, ssot]
timestamp: 2026-08-25
---

# surfmol-topology

A lightweight, forcefield-agnostic library for creating and managing molecular graphs. This is the **single source of truth (SSOT)** for molecular topology — all other representations (rendering, export, forcefield params) derive from it. Mirrors the role of SPAMMM's `AtomicGraph`.

## Responsibilities

- Define atoms, create bonds and angles (as vertices, edges, polygons).
- Atom type assignment (e.g. UFF types via octet-rule hybridization).
- Export to flat arrays for zero-copy ingestion by the forcefield engine.

## Usage

Heavily utilized by the molecular editor to represent structures. Serves as the foundation from which forcefield definitions (atom-types, bond params, angle params) are derived.

## What does NOT belong here

- Forcefield energy/force evaluation → `surfmol-forcefields`.
- Rendering → `surfmol-molrender`.
- Math primitives → `surfmol-common`.

## See also

- `ARCHITECTURE.md` §Component Details.
- `Import_other_Repos.md` §2 (SPAMMM `AtomicGraph`).
