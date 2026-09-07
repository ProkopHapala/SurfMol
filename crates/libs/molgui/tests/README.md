---
type: folder
title: molgui/tests
description: Integration tests for the molgui crate — thumbnailer rendering test.
tags: [rust, test, gui, thumbnailer, rendering]
timestamp: 2026-08-29
---

# molgui/tests — Integration tests

Integration tests for the `molgui` crate. Run with `cargo test -p molgui`.

## Test files

- **`test_thumb.rs`** — `MolThumbnailer` integration test: renders a molecule thumbnail to PNG via the `image` crate. Tests the full pipeline: PCA alignment → `ImpostorRenderer` + `LineRenderer` → RGBA readback → PNG save. Requires GPU (wgpu)

## Running

```bash
cargo test -p molgui                  # all tests
cargo test -p molgui --test test_thumb  # thumbnailer only
```

> **Note:** `test_thumb` requires a GPU (wgpu context). May fail on headless servers without a GPU or software renderer.

## See also

- [`../README.md`](../README.md) — molgui crate overview
- [`../src/gui/README.md`](../src/gui/README.md) — gui submodule files (thumbnailer.rs)
- [`/crates/libs/molrender/README.md`](/crates/libs/molrender/README.md) — wgpu rendering primitives
