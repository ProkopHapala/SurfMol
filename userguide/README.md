---
type: userguide
title: User Guide
description: End-user documentation for finished, polished modules — how to run the program, features, GUI controls, CLI options, examples, and didactic theory.
tags: [user-guide, end-user, theory]
timestamp: 2026-09-29
---

# User Guide

End-user documentation for **finished, polished** modules of SurfMol. This folder is written for the end user (a student or researcher using SurfMol), not for developers of the codebase.

## What belongs here

- **How to run** the program (build steps, launch commands, prerequisites).
- **Feature list** of finished modules (what SurfMol can do right now).
- **GUI controls** — mouse, keyboard shortcuts, camera, selection, editing.
- **CLI options** — flags, arguments, input/output formats.
- **Usage examples** — concrete walkthroughs with sample inputs from `data/`.
- **Theory** behind the operation, in didactic form accessible to a student (e.g. what a forcefield is, what ARAP ports are, what projective dynamics does).

## What does NOT belong here

- Work-in-progress notes → `notes/`.
- Developer-facing architecture/code navigation → `doc/`.
- Internal design decisions → `DESIGN_GOALS.md`.

## Guides

- **[`editor.md`](editor.md)** — Interactive molecular editor & on-surface MD simulator. CLI options (`--nmols`, `--layout`, `--show-aabb`, `--raff`, `--2d`), keyboard shortcuts, GUI panels, usage examples with benzoic acid / pyrrol / benzene, broad-phase AABB collision visualization, and didactic theory (forcefields, AABB collision).

## Status

One guide populated (`editor`). Add more as modules become polished and user-facing.
