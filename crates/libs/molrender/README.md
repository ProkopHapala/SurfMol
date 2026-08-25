---
type: rust-crate
title: molrender
description: wgpu rendering primitives — sphere impostors via fragment-shader raytracing, line renderer, textured quad surface renderer. Generic, no molecular semantics.
tags: [rust, crate, wgpu, rendering, impostor, gpu, webgpu]
timestamp: 2026-08-25
---

# molrender

Low-level wgpu rendering primitives for molecular visualization. Three composable renderers that share a depth buffer and camera uniform. **No molecular semantics** — the crate operates on `AtomInstance` (pos + radius + color) and `LineVertex` (pos + color) arrays. Type-to-render-instance conversion is the GUI layer's job.

## Modules

- **`impostor.rs`** — Sphere rendering via **billboard quads + fragment-shader raytracing**. Each atom is a screen-aligned quad (4 corners, instanced across all atoms). The fragment shader solves the ray-sphere quadratic: `disc = b² - c` where `b = oc·ray_dir`, `c = |oc|² - r²`. Hit point → normal → Lambertian shading from directional light `[0.3, 0.5, 0.8]` with ambient 0.3. Writes `@builtin(frag_depth)` from `clip.z/clip.w` for correct depth sorting (essential — the billboard quad's depth would be wrong without this). Orthographic mode: parallel rays along `-cam.forward`, ray origin shifted behind the billboard by `radius + 1.0`. `AtomInstance` is 32 bytes (vec4-aligned), `CameraData` is 144 bytes (view_proj + eye + right + up + forward + ortho flag + ray_shift). Storage buffer for atoms allows dynamic count up to `max_atoms` without reallocation. `cull_mode: None` (billboards must always render). Re-exports `numcore` math helpers (`normalize3`, `cross3`, `look_at`, `ortho`, `mul4x4`, `transpose4x4`).

- **`line_renderer.rs`** — Colored line rendering with alpha blending. `LineVertex` is 28 bytes (pos + RGBA). `LineList` topology. Takes an external `encoder` to composite over the impostor pass (uses `LoadOp::Load`, not `Clear`). Transient vertex buffer recreated each call — acceptable for dynamic line data (bonds, gizmos). Shared `CameraData` from impostor module.

- **`surface_renderer.rs`** — Textured parallelogram quad for scalar field visualization (surface potential maps). Bindings: camera uniform + texture + linear sampler. 6 indices (2 triangles). Arbitrary orientation via `u_edge`/`v_edge` vectors (not axis-aligned). Linear filtering for smooth interpolation. Uses `LoadOp::Load` to composite over existing depth/color. Expects depth view from impostor renderer for correct occlusion.

- **`lib.rs`** — `ThumbnailRenderer`: wraps `ImpostorRenderer` for offscreen thumbnail generation. Auto-computes camera framing from molecular bounding box with `max_span * 0.5 + rmax` padding. Orthographic projection. GPU readback via `copy_texture_to_buffer` + `map_async`. Uses `pollster::block_on` for synchronous init (acceptable for thumbnails).

## Design decisions

- **Impostor rendering over tessellated spheres** — billboard quad + fragment raytracing is much faster for thousands of atoms. The fragment shader does the sphere intersection per-pixel, so the visual result is a correct sphere with proper depth and shading.
- **Custom `frag_depth`** — without writing `@builtin(frag_depth)`, the billboard quad's flat depth would cause incorrect occlusion between atoms. The shader computes the actual sphere surface depth.
- **Composable render passes** — all renderers take an external encoder and use `LoadOp::Load`, enabling the sequence: impostor (clear) → lines (composite) → surface (composite). They share the depth buffer from `impostor.depth_view()`.
- **Storage buffer for atoms** — `set_atoms` / `set_instances` upload via `queue.write_buffer`. Dynamic atom count up to `max_atoms` without GPU memory reallocation.
- **`#[repr(C)]` + `bytemuck`** — all GPU data structures are C-layout for zero-copy upload.
- **Vulkan/DirectX NDC** (Z∈[0,1]) — not OpenGL's [-1,1]. Set in `numcore::math::math4d::ortho`.

## What does NOT belong here

- Molecular semantics (elements, bonds, topology) → `moltopo`
- Camera interaction (trackball, picking) → `molgui`
- Application logic (editor, browser) → `crates/apps/`

## See also

- `molgui` — `TrackballCam` produces `CameraData`, `MolThumbnailer` wraps `ImpostorRenderer`
- `numcore::math::math4d` — `look_at`, `ortho`, `mul4x4`, `transpose4x4`
- `editor` — uses all three renderers in sequence
