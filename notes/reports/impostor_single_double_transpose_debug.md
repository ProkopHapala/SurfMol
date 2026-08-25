---
type: report
title: "impostor_single_atom 0 visible pixels — double-transpose of view_proj"
tags: [bug, wgpu, math, molrender, matrix-convention]
timestamp: 2026-08-25
---

# Bug: `impostor_single_atom` produced 0 visible pixels

## Symptom
`cargo test -p molrender --test impostor_single` failed:
`impostor single atom non-bg pixels: 0 / 16384` → panic "should produce visible pixels".

Task description claimed this was a known pre-existing failure caused by
"headless GPU produces 0 visible pixels". **That diagnosis was wrong.**

## Investigation

1. Reproduced: 0 non-bg pixels on every run.
2. Probed adapter via `adapter.get_info()`:
   `name="NVIDIA GeForce RTX 3090" backend=Vulkan driver="NVIDIA"` —
   this is the **real discrete GPU**, not a headless/software fallback.
   The "headless GPU" explanation was incorrect.
3. Inspected `render_all` thumbnails (a "passing" test that only asserts
   `ok > 0`, i.e. PNGs were saved, not that they contain atoms):
   - Every pixel was `(80, 80, 97, 255)` — exactly the sRGB-encoded clear
     color `(0.08, 0.08, 0.12)` — uniform, 1 unique color.
   - So rendering was broken **everywhere**, not just in `impostor_single`.
     `render_all` was a false green: it asserts file creation, not content.

## Root cause

`numcore::math::math4d::look_at` / `ortho` build matrices in **row-major,
row-vector** convention: `clip = point * M`. The Rust arrays are stored
row-major in memory.

WGSL `mat4x4` uniforms are read **column-major** from the byte buffer, so
the same bytes are interpreted as `M^T`. WGSL then computes `M_wgsl * v`
(column-vector) = `M^T * v` = `v * M` — which is exactly the intended
row-vector product. **No host-side transpose is needed.**

The code was additionally calling `transpose4x4` before upload, producing
a **double transpose**: WGSL received `(M^T)^T = M` and computed `M * v`
— wrong convention. For the test's camera the billboard NDC.z came out
≈ -0.001 (behind the near plane), so every fragment was clipped → 0 visible.

The misleading doc comment on `look_at` ("Upload via transpose4x4 for
WGSL column-major consumption") actively prescribed the bug.

## Fix

Removed the spurious `transpose4x4` call at all 4 upload sites and
corrected the doc comment:

- `crates/libs/molrender/tests/impostor_single.rs` — `vp = mul4x4(view, proj)` (was `transpose4x4(mul4x4(...))`)
- `crates/libs/molrender/src/lib.rs` (ThumbnailRenderer) — `view_proj: vp` (was `transpose4x4(vp)`)
- `crates/libs/molgui/src/gui/trackball.rs` — dropped `transpose4x4(vp)` line
- `crates/libs/molgui/src/gui/thumbnailer.rs` — `view_proj: vp` (was `transpose4x4(vp)`)
- `crates/libs/numcore/src/math/math4d.rs` — doc comment now states the
  row-vector convention and explicitly warns against double-transposing.

`transpose4x4` is left in `numcore` as a public utility (no callers now);
not deleted per YAGNI/preservation rules.

## Verification

- `cargo test -p molrender`: **6/6 pass** (was 5/6).
  - `impostor_single_atom`: 1412 / 16384 non-bg pixels (filled disc, correct).
  - `debug_single_atom`: 16384 / 16384 (whole framebuffer covered — expected for that test's setup).
- `render_all` thumbnails now contain real content:
  - Before: 1 unique color (= clear color), 16384 "non-bg" (false positive — clear color is non-black).
  - After: e.g. benzene 26 unique colors, 1213 non-bg pixels — actual atoms rendered.
  - `debug/benzene_thumb_after_fix.png` saved for L2 visual review.
- `cargo build -p molgui -p editor`: clean (only pre-existing warnings).
- `cargo test -p molgui`: clean.

## Why molbrowser atoms were visible despite the bug (2026-08-25 follow-up)

User reported molecules were visible in molbrowser even before the fix.
Verified empirically: temporarily restored `transpose4x4` in `MolThumbnailer`,
ran thumbnail rendering:

```
WITH BUG: benzene    non_clear=2728/16384   (atoms visible!)
WITH BUG: H2O        non_clear=5082/16384
WITH BUG: NaCl_1x1_L3 non_clear=1830/16384
```

After fix:
```
FIXED:   benzene    non_clear=1460/16384
FIXED:   H2O        non_clear=5186/16384
FIXED:   NaCl_1x1_L3 non_clear=832/16384
```

**Root cause of the discrepancy: two different camera construction methods.**

### `MolThumbnailer` (molbrowser, trackball) — direct basis construction

Builds `vp` directly from camera basis vectors (right/up/forward), placing
translations in the **4th column** of the row-major matrix:
```
vp = [
    [sx*r.x, sx*r.y, sx*r.z, sx*tx],   // row 0
    [sy*u.x, sy*u.y, sy*u.z, sy*ty],   // row 1
    [sz*f.x, sz*f.y, sz*f.z, sz*tz+tz], // row 2
    [0,     0,     0,     1],           // row 3
]
```
After `transpose4x4`, translations move to the **4th row** of `M_wgsl`,
affecting `clip.w`, NOT `clip.z`. For an atom at the origin (z=0):
- `clip.z = sz * 0 = 0` → `ndc.z = 0` → exactly at near plane, NOT clipped.
- `clip.w = 1` (translation terms multiply z=0) → no perspective distortion.
- Atoms are visible, just slightly distorted for off-center atoms (clip.w ≠ 1
  when z ≠ 0, causing a mild non-uniform scaling).

### `impostor_single` test & `ThumbnailRenderer` (render_all) — `look_at × ortho`

Uses `mul4x4(look_at(...), ortho(...))`. The `look_at` view matrix has a
non-zero **4th row** translation: `[-dot(x,eye), -dot(y,eye), -dot(z,eye), 1]`.
After `mul4x4(view, proj)` and `transpose4x4`, this z-translation ends up in
`M_wgsl[2][3]` (the z-translation slot), making:
- `clip.z = sz * world_z + M_wgsl[2][3] * 1` → a **constant negative offset**
- For the test (eye at z=5, near=0.1): `clip.z ≈ -0.001` → `ndc.z ≈ -0.001 < 0`
  → **outside Vulkan NDC [0,1]** → rasterizer clips the entire triangle → 0 pixels.

For `ThumbnailRenderer` with an angled eye (not axis-aligned), the double-
transpose also transposes the **rotation** part (inverting it), causing
severe projection distortion on top of the clipping.

### Summary table

| Code path | Camera method | Bug effect | Result |
|-----------|--------------|------------|--------|
| `MolThumbnailer` (molbrowser) | direct basis, axis-aligned | translation → clip.w, ndc.z=0 | atoms visible (mild distortion) |
| `TrackballCam` (editor) | direct basis, rotated | rotation transposed + translation → clip.w | atoms visible (distorted) |
| `impostor_single` test | `look_at × ortho`, axis-aligned | z-translation → clip.z, ndc.z<0 | **0 pixels (clipped)** |
| `ThumbnailRenderer` (render_all) | `look_at × ortho`, angled | rotation transposed + z-translation → clip.z | **0 pixels (clipped)** |

The bug was **real and present everywhere**, but only caused visible failure
(total clipping) in the `look_at × ortho` paths. The direct-basis paths were
resilient because their translation layout survived the double-transpose
without pushing fragments outside the near plane.

## Notes

- `render_all` is a weak test (asserts `ok > 0` = "files were saved", not
  "atoms are visible"). It was passing while rendering was completely
  broken. Worth strengthening separately — not touched here per surgical-edit rule.
- The task description's "headless GPU" diagnosis was a guess that
  didn't match reality (real RTX 3090 in use). Verified via
  `adapter.get_info()` before theorizing about the math.
