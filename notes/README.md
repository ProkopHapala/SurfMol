---
type: work-notes
title: Work Notes
description: Temporary work-in-progress information about what we are doing right now — chats, designs, labbooks, reports, tasks, and TODOs. Not polished.
tags: [work-in-progress, ephemeral, notes]
timestamp: 2026-08-25
---

# Work Notes

**Temporary** work-in-progress information about what we are doing right now. This folder is **not polished** and is **not** permanent documentation. Content here may be deleted or migrated to `doc/` or `userguide/` once it matures.

## Subfolders

| Path | Role |
|------|------|
| `chats/` | Chat logs / transcripts relevant to current work. |
| `designs/` | Work-in-progress design sketches and proposals. |
| `labbooks/` | Per-task debugging labbooks, updated continuously (see `AGENTS.md` Rule 4). |
| `reports/` | Per-task debug / investigation reports. |
| `tasks/` | Active task definitions and checklists. |

## Files

| File | Role |
|------|------|
| `ToDo_user.md` | User-facing TODO list and design decisions. |
| `ToDo_agents.md` | Agent-facing TODO list. |

## Rules

- **Ephemeral:** anything here can be reorganized or deleted once superseded.
- **Labbooks must be updated continuously** — after every todo item, every failed test, every dead-end. Do not wait until the end (see `AGENTS.md` Rule 4).
- When a note matures into permanent documentation, **migrate** it to `doc/` or `userguide/` and leave a pointer here.
- Do not put finished user-facing docs here — use `userguide/`.
- Do not put permanent developer docs here — use `doc/`.
