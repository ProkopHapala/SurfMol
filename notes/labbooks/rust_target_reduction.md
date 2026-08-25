cargo-bloat needs a release build. Let me start that and meanwhile gather other data in parallel:

The shared target has bevy .rlibs from other projects (boc). I need molbrowser-specific data. Let me wait for the release build and use cargo-bloat:

Release build done. Now let me run cargo-bloat and gather per-crate data:

cargo-bloat rebuilt without strip (64 MB) to get symbols. The actual stripped release is 14 MB. Let me get the section breakdown and dep tree:

This is very revealing. The duplicate analysis shows a **major issue**: egui 0.29 AND 0.34 are both pulled in. Let me dig deeper:

Now I have the complete picture. Let me gather the final piece — .rlib sizes for SurfMol's actual deps:

Here's the full analysis.

---

# Binary Breakdown — why is `molbrowser` 14 MB stripped (45 MB debug)?

## Section sizes

| Section | Unstripped (64 MB) | Stripped (14 MB) | What it is |
|---------|-------------------:|-----------------:|-----------|
| `.debug_str` | 20.5 MB | — | Debug string table (names of all symbols) |
| `.debug_info` | 16.3 MB | — | Type/variable debug info |
| `.debug_ranges` | 7.3 MB | — | Debug address ranges |
| `.debug_line` | 6.5 MB | — | Line number tables |
| **`.text`** | **9.0 MB** | **9.0 MB** | **Actual machine code** |
| `.rodata` | 2.5 MB | 2.5 MB | Constants, string literals |
| `.eh_frame` | 0.8 MB | 0.8 MB | Unwind tables (backtraces) |
| `.rela.dyn` | 0.5 MB | 0.5 MB | Relocations |
| `.data.rel.ro` | 0.4 MB | 0.4 MB | Read-only relocated data |
| `.debug_aranges` | 0.2 MB | — | Debug address ranges |
| `.debug_abbrev` | 0.07 MB | — | Debug abbrev table |

**The actual code is 9 MB.** The other 50 MB of the unstripped binary is DWARF debug info (stripped by our `strip = "debuginfo"` profile). The 14 MB stripped binary = 9 MB code + 2.5 MB rodata + 0.8 MB eh_frame + 1.7 MB other.

## .text breakdown by crate (cargo-bloat, release)

| % of .text | Size | Crate | Role |
|-----------:|-----:|-------|------|
| 25.8% | 2.2 MB | `std` | Rust standard library |
| 15.1% | 1.3 MB | `naga` | WGSL shader compiler (pulled by wgpu) |
| 9.1% | 800 KB | `wgpu_core` | GPU abstraction core |
| 7.3% | 637 KB | `winit` | Windowing, event loop |
| 5.8% | 511 KB | `wgpu_hal` | GPU hardware abstraction (Vulkan/OpenGL) |
| 4.6% | 407 KB | `egui` **(0.34)** | Immediate-mode GUI |
| 2.9% | 254 KB | `wayland_client` | Wayland protocol |
| 2.6% | 228 KB | `tiny_skia` | 2D rasterizer (pulled by sctk-adwaita for window decorations) |
| 1.8% | 157 KB | `epaint` **(0.34)** | egui paint backend |
| 1.5% | 131 KB | `eframe` | egui app framework |
| 1.4% | 123 KB | `hashbrown` | Hash maps |
| 1.1% | 97 KB | `smithay_clipboard` | Clipboard |
| 1.1% | 97 KB | `x11rb_protocol` | X11 protocol |
| 1.1% | 96 KB | `ttf_parser` | Font parsing |
| 1.0% | 85 KB | `wayland_protocols` | Wayland protocols |
| 0.9% | 81 KB | `x11_dl` | X11 dynamic loading |
| 0.9% | 79 KB | `egui_winit` **(0.34)** | egui-winit bridge |
| 0.8% | 72 KB | `x11rb` | X11 Rust bindings |
| 0.7% | 64 KB | `webbrowser` | Open URL in browser |
| 0.7% | 64 KB | `glow` **(0.14)** | OpenGL wrapper (from eframe 0.29) |
| 0.6% | 57 KB | `glutin` | OpenGL context (from eframe 0.29) |
| 0.5% | 48 KB | `url` | URL parsing |
| 0.4% | 39 KB | `image` | Image I/O (pulled by arboard + eframe 0.29) |
| 0.4% | 33 KB | `molbrowser` | **Our actual code** |
| 0.2% | 17 KB | `moltopo` | Our topology crate |
| 0.2% | 15 KB | `molgui` | Our GUI crate |
| 0.1% | 8 KB | `molrender` | Our render crate |

**Our code (molbrowser + moltopo + molgui + molrender + numcore) = ~80 KB, less than 1% of the binary.** The other 99% is dependencies.

## .rlib build-cache sizes (top offenders, release)

| Size | Crate | Notes |
|-----:|-------|-------|
| 68 MB | `ash` | Vulkan bindings (pulled by wgpu-hal) |
| 30 MB | `vello_cpu` | CPU vector rasterizer (pulled by egui 0.34's epaint for text) |
| 27 MB | `naga` | WGSL shader compiler |
| 15 MB | `zerocopy` | Zero-copy traits |
| 15 MB | `glam` | Math library |
| 13 MB | `simba` | nalgebra's SIMD abstraction |
| 13 MB | `rustix` | Safe Unix syscalls (×3 copies!) |
| 12 MB | `moxcms` | Color management (pulled by image) |
| 11 MB | `nalgebra` | Math library (used for 2 lines of code!) |
| 11 MB | `egui` | ×4 copies (2 versions × 2 feature sets) |
| 8.6 MB | `fearless_simd` | SIMD (pulled by moxcms) |
| 8.1 MB | `winit` | ×2 copies |
| 7.6 MB | `image` | ×2 copies (pulled by arboard + eframe 0.29) |
| 6.7 MB | `khronos_api` | OpenGL registry XML (pulled by glutin) |
| 6.6 MB | `skrifa` | Font rasterizer (pulled by egui 0.34's vello stack) |

---

# Dependency Analysis — 306 unique crates, 14 duplicate versions

## The #1 problem: egui version split (0.29 + 0.34)

The workspace `Cargo.toml` declares:
```toml
eframe = { version = "0.29", features = ["default_fonts", "glow"] }   # pulls egui 0.29
egui = "0.34"                                                          # pulls egui 0.34
egui-winit = "0.34"                                                    # pulls egui-winit 0.34
egui-wgpu = "0.34"                                                     # pulls egui-wgpu 0.34
egui_extras = "0.29"                                                   # pulls egui_extras 0.29
```

**`eframe 0.29` internally depends on `egui 0.29`**, so BOTH full egui stacks are compiled and linked:

| Crate | v0.29 (from eframe) | v0.34 (direct dep) |
|-------|:---:|:---:|
| `egui` | ✅ | ✅ |
| `epaint` | ✅ | ✅ |
| `emath` | ✅ | ✅ |
| `ecolor` | ✅ | ✅ |
| `egui-winit` | ✅ | ✅ |
| `epaint_default_fonts` | ✅ | ✅ |
| `egui_glow` | ✅ | — |
| `glow` | ✅ (0.14) | — (0.17 from wgpu) |

This cascades into **14 duplicate crate versions** total:

| Crate | Versions | Root cause |
|-------|----------|------------|
| `egui` / `epaint` / `emath` / `ecolor` / `egui-winit` / `epaint_default_fonts` | 0.29 + 0.34 | eframe 0.29 vs direct egui 0.34 |
| `glow` | 0.14 + 0.17 | eframe 0.29 (glow backend) vs wgpu-hal (OpenGL fallback) |
| `calloop` | 0.13 + 0.14 | smithay-client-toolkit 0.19 vs 0.20 |
| `calloop-wayland-source` | 0.3 + 0.4 | same |
| `smithay-client-toolkit` | 0.19 + 0.20 | egui-winit 0.29 vs 0.34 |
| `hashbrown` | 0.15 + 0.16 + 0.17 | 3 versions from different deps |
| `foldhash` | 0.1 + 0.2 | hashbrown version split |
| `rustix` | 0.38 + 1.1 | calloop 0.13 vs 0.14 |
| `linux-raw-sys` | 0.4 + 0.12 | rustix version split |
| `thiserror` | 1.0 + 2.0 | old vs new deps |
| `rustc-hash` | 1.1 + 2.1 | naga vs egui-wgpu |

## Unused dependencies found

| Dep | Declared in | Actually used in `src/`? | Action |
|-----|-------------|--------------------------|--------|
| `eframe` | editor | **NO** — editor manually creates winit+wgpu+egui-winit+egui-wgpu | Remove from editor |
| `egui_extras` | editor, workspace | **NO** — zero `use egui_extras` anywhere | Remove everywhere |
| `egui-wgpu` | molbrowser | **NO** — molbrowser uses eframe's built-in renderer | Remove from molbrowser |
| `egui_plot` | workspace | **NO** — not in any crate's Cargo.toml | Remove from workspace |
| `nalgebra` | molrender, molbrowser | **NO** — only used in molgui's thumbnailer.rs (2 lines) | Remove from molrender + molbrowser |

## nalgebra vs glam — two math libraries for 2 lines of code

`nalgebra` (11 MB .rlib + pulls `simba` 13 MB + `num-traits` + `num-complex` + `num-rational` + `num-integer` + `typenum` + `rawpointer` + `matrixmultiply` + `approx` + `nalgebra-macros` + `paste` + `static_assertions` = ~30 MB of .rlib) is used for exactly **2 lines** in `thumbnailer.rs:204-214`:

```rust
let mat = nalgebra::Matrix3::new(...);
let eig = mat.symmetric_eigen();  // 3×3 symmetric eigendecomposition
```

`glam` (15 MB .rlib, already a dep) doesn't have eigendecomposition, but a 3×3 symmetric Jacobi eigenvalue solver is ~30 lines of code.

---

# Recommendations (ranked by impact)

## R1. Upgrade eframe 0.29 → 0.34 (biggest lever) ⭐

**Saves:** ~8 crates eliminated (entire egui 0.29 stack), ~14 duplicate versions collapse to 1, eliminates glow 0.14 + glutin + glutin_egl_sys + glutin_glx_sys + glutin-winit + gl_generator + khronos_api (6.7 MB .rlib) + khronos-egl. Estimated **~4-5 MB from .text**, ~30-40 MB from .rlib build cache.

**How:**
```toml
# Before:
eframe = { version = "0.29", default-features = false, features = ["default_fonts", "glow"] }
egui = "0.34"
egui-winit = "0.34"
egui-wgpu = "0.34"
egui_extras = "0.29"

# After:
eframe = { version = "0.34", default-features = false, features = ["default_fonts", "wgpu", "wayland", "x11"] }
egui = "0.34"
egui-winit = "0.34"
egui-wgpu = "0.34"
```

Switch from `glow` (OpenGL) to `wgpu` backend — we already use wgpu everywhere else. Eliminates the entire OpenGL/glutin stack.

**Risk:** Medium — eframe 0.29 → 0.34 has API changes. molbrowser uses `eframe::App`, `eframe::NativeOptions`, `eframe::run_native` — these are stable across versions. Editor doesn't use eframe at all.

## R2. Remove unused deps (quick wins)

| Action | Files | Saves |
|--------|-------|-------|
| Remove `eframe` from editor | `crates/apps/editor/Cargo.toml` | ~6.5 MB .rlib (eframe 0.29) |
| Remove `egui_extras` from editor + workspace | `crates/apps/editor/Cargo.toml`, `Cargo.toml` | small |
| Remove `egui-wgpu` from molbrowser | `crates/apps/molbrowser/Cargo.toml` | small (eframe 0.34 includes it) |
| Remove `egui_plot` from workspace | `Cargo.toml` | 0 (not pulled) |
| Remove `nalgebra` from molrender + molbrowser | 2× `Cargo.toml` | 0 (molgui still pulls it) |

## R3. Replace nalgebra with glam + 30-line Jacobi solver ⭐

**Saves:** ~30 MB .rlib (nalgebra 11 MB + simba 13 MB + num-traits + num-complex + num-rational + num-integer + typenum + rawpointer + matrixmultiply + approx + nalgebra-macros + paste + static_assertions). ~500 KB from .text.

**How:** Write a `symmetric_eigen_3x3` function in `numcore` (or molgui) using the Jacobi method (~30 lines). Replace the 2 nalgebra lines in `thumbnailer.rs:204-214`. Remove `nalgebra` from all Cargo.toml files.

**Risk:** Low — 3×3 symmetric eigendecomposition is well-understood. The Jacobi algorithm is ~30 lines and numerically stable.

## R4. Consider dropping `arboard` (clipboard) if not needed

`arboard` pulls `image` (7.6 MB .rlib + moxcms 12 MB + fearless_simd 8.6 MB = ~28 MB .rlib chain). It's pulled by `egui-winit` for clipboard support. If we don't need clipboard in our apps, we could fork egui-winit or disable the feature (if possible).

**Risk:** Medium — would need to check if egui-winit allows disabling clipboard.

## R5. `vello_cpu` (30 MB .rlib) — egui 0.34's text rasterizer

egui 0.34 switched to vello for text rasterization. This is the same issue documented in the boc `dependency_review.md` §5. LTO strips most of it in release binaries (only ~2 MB in the stripped binary), but the **build-cache cost is 30 MB .rlib**. No easy fix without pinning to an older egui version (which we just upgraded). Accept it.

## Summary table

| # | Action | .text savings | .rlib savings | Effort | Risk |
|---|--------|-------------:|--------------:|--------|------|
| R1 | eframe 0.29 → 0.34 (wgpu backend) | ~4-5 MB | ~30-40 MB | Medium | Medium |
| R2 | Remove unused deps | ~1 MB | ~7 MB | Trivial | None |
| R3 | Replace nalgebra with glam + Jacobi | ~0.5 MB | ~30 MB | Low | Low |
| R4 | Drop arboard (if clipboard unused) | ~0.3 MB | ~28 MB | Medium | Medium |
| R5 | vello_cpu (accept) | — | — | — | — |
| **Total** | | **~6 MB** | **~95 MB** | | |

**Expected result if R1+R2+R3 applied:** molbrowser stripped binary ~8 MB (from 14 MB), dep count ~250 (from 306), .rlib build cache ~50 MB smaller.

Want me to start implementing any of these? R2 (unused deps) is trivial and safe. R1 (eframe upgrade) is the biggest win but needs testing. R3 (nalgebra → glam) is self-contained.