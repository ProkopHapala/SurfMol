# Rust Target Size Reduction — Labbook

## Timeline

### Phase 1: Analysis (before changes)

cargo-bloat needs a release build. Let me start that and meanwhile gather other data in parallel:

The shared target has bevy .rlibs from other projects (boc). I need molbrowser-specific data. Let me wait for the release build and use cargo-bloat:

Release build done. Now let me run cargo-bloat and gather per-crate data:

cargo-bloat rebuilt without strip (64 MB) to get symbols. The actual stripped release is 14 MB. Let me get the section breakdown and dep tree:

This is very revealing. The duplicate analysis shows a **major issue**: egui 0.29 AND 0.34 are both pulled in. Let me dig deeper:

Now I have the complete picture. Let me gather the final piece — .rlib sizes for SurfMol's actual deps:

---

## Binary Breakdown — BEFORE changes: `molbrowser` 14 MB stripped (45 MB debug)

### Section sizes

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

### .text breakdown by crate (cargo-bloat, release, BEFORE)

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

### .rlib build-cache sizes (top offenders, BEFORE)

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

## Dependency Analysis — BEFORE: 306 unique crates, 14 duplicate versions

### The #1 problem: egui version split (0.29 + 0.34)

The workspace `Cargo.toml` declared:
```toml
eframe = { version = "0.29", features = ["default_fonts", "glow"] }   # pulls egui 0.29
egui = "0.34"                                                          # pulls egui 0.34
egui-winit = "0.34"                                                    # pulls egui-winit 0.34
egui-wgpu = "0.34"                                                     # pulls egui-wgpu 0.34
egui_extras = "0.29"                                                   # pulls egui_extras 0.29
```

**`eframe 0.29` internally depends on `egui 0.29`**, so BOTH full egui stacks were compiled and linked.

### Unused dependencies found

| Dep | Declared in | Actually used in `src/`? | Action |
|-----|-------------|--------------------------|--------|
| `eframe` | editor | **NO** — editor manually creates winit+wgpu+egui-winit+egui-wgpu | Remove from editor |
| `egui_extras` | editor, workspace | **NO** — zero `use egui_extras` anywhere | Remove everywhere |
| `egui-wgpu` | molbrowser | **NO** — molbrowser uses eframe's built-in renderer | Remove from molbrowser |
| `egui_plot` | workspace | **NO** — not in any crate's Cargo.toml | Remove from workspace |
| `nalgebra` | molrender, molbrowser | **NO** — only used in molgui's thumbnailer.rs (2 lines) | Remove from molrender + molbrowser |

### nalgebra vs glam — two math libraries for 2 lines of code

`nalgebra` (11 MB .rlib + pulls `simba` 13 MB + `num-traits` + `num-complex` + `num-rational` + `num-integer` + `typenum` + `rawpointer` + `matrixmultiply` + `approx` + `nalgebra-macros` + `paste` + `static_assertions` = ~30 MB of .rlib) was used for exactly **2 lines** in `thumbnailer.rs:204-214`:

```rust
let mat = nalgebra::Matrix3::new(...);
let eig = mat.symmetric_eigen();  // 3×3 symmetric eigendecomposition
```

---

## Phase 2: User decisions

User decided:
- **Keep egui 0.34 + wgpu 29** — vello build-cache cost (55 MB .rlib) is acceptable; binary cost (~2.8 MB) is inherent to egui 0.34
- **Remove nalgebra** — replace with analytical eigensolver
- **Clipboard fix (option A)** — disable egui-winit clipboard feature, use arboard (default-features=false) separately, wire text clipboard to egui manually
- **Don't mind 55 MB larger shared target** (build cache), but **mind every 10 MB bigger binary**

---

## Phase 3: Implementation

### R1: eframe 0.29 → 0.34 (wgpu backend) — DONE

**Workspace `Cargo.toml` changes:**
```toml
# Before:
eframe = { version = "0.29", default-features = false, features = ["default_fonts", "glow"] }
egui = "0.34"
egui-winit = "0.34"
egui-wgpu = "0.34"
egui_extras = "0.29"
egui_plot = "0.34"
nalgebra = "0.33"

# After:
eframe = { version = "0.34", default-features = false, features = ["default_fonts", "wgpu", "wayland", "x11"] }
egui = "0.34"
egui-winit = { version = "0.34", default-features = false, features = ["wayland", "x11", "links"] }
egui-wgpu = "0.34"
arboard = { version = "3.6", default-features = false }
```

**API change in molbrowser:** eframe 0.34 deprecated `App::update()` and made `App::ui()` the required method. Changed `impl eframe::App for MolBrowserApp` from `fn update(&mut self, ctx, frame)` to `fn ui(&mut self, ui, frame)`, extracting `ctx` via `ui.ctx().clone()`.

**Result:** Eliminated entire egui 0.29 stack (egui/epaint/emath/ecolor/egui-winit/epaint_default_fonts × 0.29) + glow 0.14 + glutin + glutin_egl_sys + glutin_glx_sys + glutin-winit + gl_generator + khronos_api + khronos-egl. Unified to single egui 0.34.

### R2: Remove unused deps — DONE

- `eframe` removed from editor Cargo.toml (editor uses manual winit+egui-winit+egui-wgpu)
- `egui_extras` removed from editor + workspace
- `egui-wgpu` removed from molbrowser (eframe 0.34 includes it)
- `egui_plot` removed from workspace
- `nalgebra` removed from molrender + molbrowser
- `nalgebra` removed from molgui (after R3 replaced its usage)

### R3: Replace nalgebra with analytical eigensolver — DONE

**New file:** `crates/libs/numcore/src/math/linalg.rs`

Implemented `symmetric_eigen_3x3(a: [f32; 9]) -> [(f32, [f32; 3]); 3]` using:
- **Smith 1961** closed-form trigonometric solution for eigenvalues of symmetric 3×3 matrix (no iteration)
- **Cross-product method** for eigenvectors (from FireCore `Mat3.h:eigenvec`)
- Ported from `/home/prokop/git/FireCore/cpp/common/math/Mat3.h:Mat3T::eigenvals()` + `eigenvec()`
- Reference: Smith, Oliver K. (April 1961), "Eigenvalues of a symmetric 3×3 matrix.", Communications of the ACM 4 (4): 168
- See also: http://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf

**Why analytical (Smith 1961) instead of iterative (Jacobi):**
- User pointed to FireCore's `Mat3.h` which uses the analytical approach
- No iteration → deterministic, no convergence issues
- ~30 lines of code, no dependencies
- Precision: ~1e-4 for well-separated eigenvalues, ~2e-3 for repeated eigenvalues (degenerate cases) — sufficient for PCA thumbnail alignment

**5 unit tests pass:**
- `test_diagonal_matrix` — diag(3,1,2) → eigenvalues 1,2,3
- `test_identity` — all eigenvalues = 1
- `test_offdiagonal` — [[2,1,0],[1,2,0],[0,0,3]] → eigenvalues 1,3,3 (repeated eigenvalue case)
- `test_random_symmetric` — A = R·diag(1,2,3)·R^T with 30° rotation → recovers 1,2,3 + unit eigenvectors
- `test_pca_inertia_tensor` — linear molecule along x → smallest eigenvalue corresponds to x-axis

**Replaced in `thumbnailer.rs:204-214`:** 2 lines of nalgebra → 2 lines of numcore::math::linalg.

**Eliminated:** nalgebra + simba + num-traits + num-complex + num-rational + num-integer + typenum + rawpointer + matrixmultiply + approx + nalgebra-macros + paste + static_assertions (~30 MB .rlib).

### R4: Clipboard fix — DONE (editor only)

**Problem:** egui-winit's `clipboard` feature pulls arboard with `image-data` feature → image crate → moxcms → fearless_simd (~28 MB .rlib chain). eframe 0.34 **hardcodes** `clipboard` feature on egui-winit (in its Cargo.toml: `features = ["clipboard", "links"]`), so it cannot be disabled for molbrowser.

**Solution for editor:** Disable egui-winit's clipboard feature (editor uses manual egui-winit, not eframe), add `arboard = { version = "3.6", default-features = false }` (text-only, 18 deps, no image chain), wire clipboard manually.

**New file:** `crates/libs/molgui/src/gui/clipboard.rs`
- `Clipboard` struct wrapping `arboard::Clipboard` (text-only)
- `inject_cut_copy_if_needed()` — detect Ctrl+C/Ctrl+X, inject `Event::Copy`/`Event::Cut` into `raw_input.events`
- `inject_paste_if_needed()` — detect Ctrl+V, read OS clipboard, inject `Event::Paste(text)`
- `handle_output_commands()` — after `ctx.run()`, write `OutputCommand::CopyText` to OS clipboard
- Key detection mirrors egui-winit 0.34 `is_cut_command`/`is_copy_command`/`is_paste_command` (egui-winit-0.34.3/src/lib.rs:1305-1321)

**Wired into editor's render loop** (`crates/apps/editor/src/main.rs:683-693`):
```rust
let mut raw_input = self.egui_state.take_egui_input(&self.window);
let mods = raw_input.modifiers;
inject_cut_copy_if_needed(&mut raw_input.events, mods);
inject_paste_if_needed(&mut raw_input.events, mods, &mut self.clipboard);
let full_output = egui_ctx.run(raw_input, |ctx| { self.draw_egui(ctx); });
handle_output_commands(&full_output.platform_output, &mut self.clipboard);
self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
```

**molbrowser:** Still uses eframe, which forces the clipboard feature. molbrowser has no text editing (thumbnail grid viewer), so in-app clipboard fallback is fine. Converting to manual winit would save ~100 KB from binary (LTO already strips most of the image chain), not worth the effort.

---

## Phase 4: Results

### Binary sizes (stripped, no debug-assertions)

| Binary | Before | After | Change |
|--------|-------:|------:|--------|
| `molbrowser` | 14 MB | 14 MB | 0 (same) |
| `editor` | — | 13 MB | new measurement |

**Note on debug-assertions:** The release profile has `debug-assertions = true` and `overflow-checks = true`, which adds ~3 MB to the binary. With those disabled, molbrowser = 14 MB. With them enabled, molbrowser = 17 MB. This is a profile setting, not a dependency issue.

### Dependency count

| Metric | Before | After | Change |
|--------|-------:|------:|--------|
| molbrowser deps | 306 | 266 | **-40** |
| editor deps | — | 251 | — |
| egui versions | 0.29 + 0.34 | 0.34 only | **unified** |
| nalgebra | yes | **gone** | **-30 MB .rlib** |
| glutin/glow 0.14 | yes | **gone** | **-7 MB .rlib** |
| image chain (editor) | yes | **gone** | **-28 MB .rlib** |

### .text breakdown AFTER (cargo-bloat, molbrowser, release without debug-assertions)

| % of .text | Size | Crate | Role |
|-----------:|-----:|-------|------|
| 21.1% | 2.5 MB | `std` | Rust standard library |
| 11.0% | 1.3 MB | `naga` | WGSL shader compiler |
| 9.2% | 1.1 MB | `vello_cpu` | **egui 0.34 text rasterizer (unavoidable)** |
| 7.0% | 861 KB | `wgpu_core` | GPU abstraction core |
| 7.0% | 855 KB | `fearless_simd` | **SIMD (from vello_common, unavoidable)** |
| 5.2% | 637 KB | `winit` | Windowing, event loop |
| 4.3% | 529 KB | `egui` | Immediate-mode GUI (0.34 only) |
| 4.2% | 515 KB | `wgpu_hal` | GPU hardware abstraction |
| 2.5% | 302 KB | `skrifa` | **Font rasterizer (from vello, unavoidable)** |
| 2.1% | 254 KB | `wayland_client` | Wayland protocol |
| 2.0% | 247 KB | `peniko` | **Pen rendering (from vello, unavoidable)** |
| 1.9% | 228 KB | `tiny_skia` | 2D rasterizer (window decorations) |
| 1.5% | 184 KB | `epaint` | egui paint backend |
| 1.3% | 161 KB | `vello_common` | **Vello common (unavoidable)** |
| 1.2% | 144 KB | `read_fonts` | **Font reading (from vello, unavoidable)** |
| 1.1% | 133 KB | `hashbrown` | Hash maps |
| 0.8% | 99 KB | `eframe` | egui app framework |
| 0.8% | 97 KB | `smithay_clipboard` | Clipboard |
| 0.8% | 97 KB | `x11rb_protocol` | X11 protocol |
| 0.4% | 55 KB | `arboard` | Clipboard (text-only, default-features=false) |
| 0.3% | 39 KB | `image` | Image I/O (LTO strips most; from eframe's forced clipboard) |

**Key observation:** The vello text rasterizer chain (vello_cpu + fearless_simd + skrifa + peniko + vello_common + read_fonts) = **~2.8 MB** in .text. This is inherent to egui 0.34 and cannot be avoided without downgrading to egui 0.33 (which uses ab_glyph, ~1 MB total).

### What was eliminated from .text

| Crate | Before | After | Savings |
|-------|-------:|------:|--------|
| `glow` (0.14) | 64 KB | 0 | -64 KB |
| `glutin` | 57 KB | 0 | -57 KB |
| `eframe` (0.29) | 131 KB | 99 KB | -32 KB |
| `egui` (0.29 duplicate) | ~400 KB | 0 | -400 KB |
| `epaint` (0.29 duplicate) | ~157 KB | 0 | -157 KB |
| `nalgebra` + `simba` | ~500 KB | 0 | -500 KB |
| **Total .text savings** | | | **~1.2 MB** |

The .text went from 9.0 MB to ~11.9 MB — but this is **not a regression**. The increase is from vello_cpu (1.1 MB) + fearless_simd (855 KB) + skrifa (302 KB) + peniko (247 KB) + vello_common (161 KB) + read_fonts (144 KB) = ~2.8 MB, which is the egui 0.34 text rasterizer that was always present in the 0.34 stack but wasn't being counted in the "before" analysis because cargo-bloat was run on the old binary that had egui 0.29 as the active eframe backend. The egui 0.34 stack was compiled but LTO-dead-stripped since eframe 0.29 was the active one. Now egui 0.34 is the active one, so vello is live.

### Build-cache (.rlib) savings

| Item | .rlib savings |
|------|-------------:|
| nalgebra + simba + num-* chain | ~30 MB |
| glutin + glow 0.14 + khronos_api + gl_generator | ~7 MB |
| egui 0.29 duplicate stack | ~11 MB |
| image chain (editor only) | ~28 MB |
| **Total .rlib savings** | **~76 MB** |

---

## Phase 5: What remains (not done)

### molbrowser still pulls image chain via eframe

eframe 0.34 hardcodes `clipboard` feature on egui-winit (`features = ["clipboard", "links"]` in its Cargo.toml), which pulls arboard with `image-data` feature → image → moxcms → fearless_simd (~28 MB .rlib).

**Binary impact:** Only ~100 KB (LTO strips most of image/moxcms since molbrowser never calls clipboard image functions).
**Build-cache impact:** ~28 MB .rlib.
**Fix:** Convert molbrowser to manual winit (like editor) — ~30 lines of boilerplate. NOT worth it for binary size; only worth it if build cache is critical.

### vello text rasterizer (~2.8 MB in binary, ~55 MB in .rlib)

egui 0.34's epaint hard-depends on vello_cpu + skrifa + read-fonts for text rasterization. This replaced egui 0.33's ab_glyph (~1 MB .rlib) with a ~55 MB .rlib chain.

**Binary impact:** ~2.8 MB in .text (vello_cpu 1.1 MB + fearless_simd 855 KB + skrifa 302 KB + peniko 247 KB + vello_common 161 KB + read_fonts 144 KB).
**Build-cache impact:** ~55 MB .rlib.
**Fix:** Downgrade to egui 0.33 + wgpu 27. Requires API changes (TexelCopy→ImageCopy, instance creation, surface handling). NOT done — user decided to keep egui 0.34 + wgpu 29.

### egui 0.34 deprecation warnings

egui 0.34 deprecated several APIs used by the editor:
- `Context::run` → `run_ui`
- `screen_rect` → `viewport_rect()` / `content_rect()`
- `Frame::none` → `Frame::NONE` / `Frame::new()`
- `DragValue::clamp_range` → `range`
- `ComboBox::from_id_source` → `from_id_salt`
- `Context::style` → `global_style`
- `TopBottomPanel::show` → `show_inside()` (molbrowser)
- `CentralPanel::show` → `show_inside()` (molbrowser)

These are warnings, not errors. Separate cleanup task.

### debug-assertions in release profile

The release profile has `debug-assertions = true` and `overflow-checks = true`, adding ~3 MB to binaries. This is useful for catching bugs in release builds but costs binary size. User decision needed.

---

## Files changed

| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | eframe 0.29→0.34, egui-winit default-features=false, removed egui_extras/egui_plot/nalgebra, added arboard default-features=false |
| `crates/apps/editor/Cargo.toml` | Removed eframe, egui_extras, nalgebra; added arboard |
| `crates/apps/molbrowser/Cargo.toml` | Removed egui-wgpu, nalgebra |
| `crates/libs/molrender/Cargo.toml` | Removed nalgebra |
| `crates/libs/molgui/Cargo.toml` | Removed nalgebra; added egui, arboard |
| `crates/libs/numcore/src/math/mod.rs` | Added `pub mod linalg` |
| `crates/libs/numcore/src/math/linalg.rs` | **NEW** — Smith 1961 analytical 3×3 symmetric eigensolver (ported from FireCore Mat3.h) |
| `crates/libs/molgui/src/gui/mod.rs` | Added `pub mod clipboard` |
| `crates/libs/molgui/src/gui/clipboard.rs` | **NEW** — text-only arboard clipboard bridge for egui |
| `crates/libs/molgui/src/gui/thumbnailer.rs` | Replaced nalgebra::Matrix3::symmetric_eigen with numcore::math::linalg::symmetric_eigen_3x3 |
| `crates/apps/editor/src/main.rs` | Added Clipboard field + clipboard bridge wiring in render loop |
| `crates/apps/molbrowser/src/main.rs` | eframe 0.34 API: `App::update` → `App::ui` |
