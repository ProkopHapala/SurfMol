---
type: work-notes
title: Labbooks
description: Per-task debugging labbooks, updated continuously during a debugging session. Ephemeral.
tags: [work-in-progress, labbook, debugging, ephemeral]
timestamp: 2026-08-25
---

# Labbooks

Per-task debugging labbooks. **Every debugging session must have a task-specific labbook here**, updated **continuously** — after every todo item, every failed test run, every dead-end (see `AGENTS.md` Rule 4).

## What to write

- What was tried.
- What happened (with numbers — energies, residuals, timings, per-atom values).
- What it means.
- Dead-ends and why they were dead-ends.

**Do not wait until the end to write the labbook.** If the debugging loop is long, the labbook will never be written and all knowledge about dead-ends is lost.

## Naming

`<date>_<task>_debug.md` — e.g. `2026-08-25_raff_port_energy_debug.md`.
