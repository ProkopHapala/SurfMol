---
type: developer-docs
title: Developer Documentation
description: Permanent polished documentation for developers — navigation, understanding, and cross-implementation maps of the codebase.
tags: [developer-docs, navigation, architecture]
timestamp: 2026-08-25
---

# Developer Documentation

Permanent, polished documentation **for developers** that helps navigation and understanding of the code. Unlike `notes/` (ephemeral work-in-progress), this folder holds information that stays valid and useful over time.

## What belongs here

- `CODEMAP.md` — repo structure, file inventory, crate dependency graph.
- `topical_audit/` — cross-implementation maps per scientific topic (where each algorithm/feature lives across files, crates, and reference repos).
- Permanent design rationale that is not already in `ARCHITECTURE.md` or `DESIGN_GOALS.md`.
- Developer onboarding guides.

## What does NOT belong here

- End-user how-to-run guides → `userguide/`.
- Work-in-progress notes, labbooks, reports → `notes/`.
- Binding agent rules → `AGENTS.md`.
- Reference-repo import plans → `Import_other_Repos.md` (repo root).

## Subfolders

| Path | Role |
|------|------|
| `topical_audit/` | Cross-implementation maps per scientific topic. See `topical_audit/README.md`. |

## Status

`topical_audit/` exists but is empty. Populate `CODEMAP.md` and topical audits as the codebase matures.
