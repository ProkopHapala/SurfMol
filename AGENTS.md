# SurfMol — Agent Rules of Conduct

SurfMol: on-surface molecular manipulation, global optimization, scanning-probe microscopy. Numerical correctness, physical consistency, debuggability, and performance are paramount.

**Languages:** Rust (simulation, GUI, orchestration) + OpenCL (GPU). Python is a minor glue layer only — never a hot path. C++ (FireCore) is legacy reference for parity/perf comparison.

## Core Principles

- **KISS** — simplest solution that works; one-liner > ten-liner.
- **AHA** — avoid hasty abstractions and boilerplate.
- **YAGNI** — surgical edits; touch only what's needed; no unrelated cleanup; comment out, don't delete; ask if ambiguous.
- **DRY** — inventory existing code before writing new; generalize rather than duplicate.
- **SoC** — separate compute from presentation. Domain crates (`surfmol-common`, `surfmol-topology`, `surfmol-forcefields`) have no GUI dependency; `surfmol-apps` is the only GUI crate; no plotting/rendering in core libs.
- **SSOT** — one authoritative source of truth. Topology SSOT lives in `surfmol-topology` (`AtomicGraph`); all other representations derive from it. `MolWorld` does not own positions/forces — those live in `DynamicAtoms` (`surfmol-common`); each forcefield owns only its params. See `rust/forcefields/DESIGN.md`.
- **TDD** — define verification before coding; parity vs reference/analytical/physical invariants; run tests after every change.
- **Fail Fast, Fix the Physics** — **silent fallbacks are strictly prohibited.** Anything unexpected (NaN, Inf, out-of-range value, missing file, shape mismatch, failed convergence, violated invariant) must **fail loud and early** with a full stack trace — panics > silent `Ok`. Never mask a bug by clamping divergent values, returning a default, retuning until `Ok`, or broad error-swallowing. **Rust:** no `.unwrap_or(default)`/`.ok()` that drops an `Err` you didn't diagnose; no `let _ = result;`; propagate with `?` or `panic!`/`assert!`/`expect("context")` instead. **Python:** no bare `except:` or `except Exception:` that passes/logs-and-continues; catch only the specific error you handle, else let it raise. If a solver fails, **fix the solver, not the scenario**. Everything emerges from fundamental interactions, not scripts.
- Compact code, unlimited line length; short names for math/physics symbols (`E_tot`, `T_ij`, `m_i`, `F_ij`).

## ⚑ Tests Are Diagnostics, Not the Goal (most-violated rule)

**Passing tests is NOT the endgoal.** A green test suite says nothing about physical correctness. Tests are a tool to **locate where the program is wrong**, in order to **understand and fix the physics**. The goal is root cause, not green-at-any-cost.

- **Do NOT make tests green at all cost.** A red test is a diagnostic: it tells you *where the physics is wrong*. Keep it red if necessary to document broken physics — a known problem is better than a masked one.
- **Never cheat a test green** by violating physics, suppressing errors, loosening tolerances, or picking cautious parameters.
- When a test fails, the response is to **investigate the physics**, not to silence the test. If the test itself is wrong, fix the test — but say so explicitly and justify it.

## Never Do This

- **NEVER use `rm`, `sed -i`, `cat >`, `echo >>`, heredocs, or shell redirects to delete/modify files.** Use the Devin `edit`/`write`/`read` tools so changes appear in the diff viewer. If an edit tool fails, do smaller targeted edits — never fall back to shell.
- Never delete/rearrange existing code, or make unrelated aesthetic edits, without explicit permission.
- Never apply quick-fixes that hide root causes (hard-coded outputs, clamping to hide divergence).
- Never reinvent existing functionality — inventory first (`CODEMAP.md`, `Import_other_Repos.md`, sibling crates, reference repos: FireCore, SPAMMM, learn_Rust, blood_of_civilization).
- Never copy-paste between crates — extract to a shared lib.
- Never cheat a test green (see ⚑ Tests Are Diagnostics above).
- **Ask, don't Guess** — when unsure, ask the user.

## Surgical Edits & Checkpointing

- **Minimum intervention:** write only what the task needs.
- **Strict checkpointing:** after every significant step, summarize what changed, what was verified, what remains.
- **Preservation:** back up before major module changes; comment out (`//`/`#`) deprecated/experimental code instead of deleting; mark unfinished with `// TODO` / `// DEBUG`.
- **Never mark "fixed"/"done" without explicit USER confirmation.** A code change is not proof. You must: (1) run a test demonstrating the fix, (2) show the result, (3) wait for confirmation. When in doubt, leave status as "investigating"/"unverified".

## Reusable Architecture

- **Inventory first** — review reference sources before writing anything new (see Never Do This).
- **Composability over bloat** — build integrated systems, not isolated scripts; refactor into shared-crate functions.
- **Generalize over duplicate** — if a function almost fits, generalize it; if generalization risks backward compatibility, stop and report for approval.
- Separate GUI (`surfmol-apps`), CLI tools (in their backend crate's `src/bin/`), and backend modules. Test scripts are thin wrappers calling shared-crate functions; consolidate related scripts into one with CLI routing.

## Testing & Validation

- **Numerical sanity:** place checks ensuring values are finite, in-range, not unexpected zeros.
- **Diagnostic tests, not pass/fail:** print actual numbers (per-atom residuals, per-cell energies, worst contributor, sign of deviation). Assert on physical invariants (energy conservation, sign convention, monotonicity, symmetry). On failure, output should locate the bug without re-running. `assert_eq!` is a smoke check, not a scientific test.
- **Parity before coding:** define correctness checks vs reference code (FireCore C++ is the perf+correctness benchmark), analytical solutions, conservation laws, symmetry, physical limits. See skill:`numerical-parity`.
- **Foreground execution:** run tests synchronously with full output — never background, `| tail`, `| head`, `| grep`, or `&`.
- **Three review levels:** L0 `cargo test` (automated regression) · L1 agent reads `.out`/`.log` artifacts unfiltered · L2 human reviews `.png`/`.svg` plots in `debug/` or `artifacts/`.
- **Labbook:** every debugging session gets `notes/reports/<task>_debug.md`, updated continuously (after each todo, each failed run, each dead-end) — what was tried, what happened (with numbers), what it means. See skill:`debug`.
- **Refactoring discipline:** before refactoring, run each old test and show results to USER; delete old files only after explicit approval; never delete plots.
- **Visual review:** use shared plotting utilities, not ad-hoc code. See skill:`visual-debugging`.
- **Images in chat:** use `<ref_file file="/abs/path/to/image.png" />` — the only format that renders as a clickable image. Markdown `![]()`/`[]()` do NOT render. Save PNGs to `debug/`/`artifacts/`.
- **Long-running scripts MUST print unbuffered progress** (`eprintln!`/`println!` with flush, or `PYTHONUNBUFFERED=1`) — print starts, accepted steps with energy decrease, and finish. Never run silently for minutes.
- **Debug prints are gated, not deleted.** Use verbosity-gated logging (Rust: `log` crate macros `error!`/`warn!`/`info!`/`debug!`/`trace!` filtered via `RUST_LOG`; or `eprintln!` behind a `const VERBOSE: bool`/`--verbose` flag) so output is controlled by level, not by removing lines. **Do NOT remove debug prints until the program is functioning correctly.** If output is too noisy, **lower the debug level** (e.g. `RUST_LOG=warn`) — do not delete the print statements. Silent code is undebuggable; gated prints let you re-raise verbosity the moment something breaks again.
- **Informative messages, not "it broke".** Every error, panic, assertion, and debug print must carry enough context to locate the bug without re-running: **where** (function/module/file:line — Rust's `panic!`/`expect` and `#[track_caller]` give this; Python's `logging` with `%(funcName)s:%(lineno)d`), **what** happened (the violated invariant / unexpected state, in plain words), and the **values of all relevant variables** (inputs, indices, shapes, energies, residuals — print the numbers, not just names). Bad: `"failed"`, `"NaN error"`, `unwrap()`. Good: `expect(&format!("bond {i}-{j} stretched: |r|={r:.3} > cutoff {c:.3}"))`, `panic!("energy non-finite at step {step}: E={E}, max|F|={fmax}");`. A message that doesn't let you reconstruct the failure is a bug in the message.
- **Invoke relevant skills** when a task matches: `numerical-parity`, `visual-debugging`, `gpu-debugging`, `forcefield-validation`, `port-to-opencl`.

## Performance

- **Rust is the engine** — all simulation logic in Rust. Flat arrays, cache-aware, preallocate; prefer `&[T]`/`&mut [T]` over `Vec<T>` in hot paths; SoA/data-oriented layouts; be explicit about `f32` vs `f64`.
- **OpenCL is the accelerator** — GPU must match CPU within tolerance. Prefer **NVIDIA GPU**; never report PoCL/CPU timings as GPU timings. GPU is single precision f32 by default (f64 much workse preformance), packed float4 arrays prefered, workgroupsize ~32 prefered.
- **Minimal orchestration** — push heavy compute into OpenCL kernels; Python only orchestrates, never hot loops.
- **GPU kernels:** design for memory latency; gather > scatter; minimize branching/atomics/sync; maximize shared/local memory; avoid host-device transfers; **fuse secondary checks into existing kernels** (add clash flags in a loop that already computes distance — never recompute on host). See skill:`port-to-opencl`.

## Style

- **No micro-abstractions** — no 1-line stubs/wrappers; inline if simple.
- **Clean interfaces** — group related state into structs; use builder/default named args to avoid long call strings.
- **Compact layout** — long lines, minimal blank lines; no wrapping that disrupts readability.
- **Naming & comments** — short math/physics symbol names; comments for intent/rationale/derivations only, placed inline behind the code line.
- **Rust:** gated debug logging (see Testing & Validation §Debug prints); `&[f32]`/`&mut [f32]` in hot paths; `bytemuck` for zero-copy OpenCL casts; `///` rustdoc (not `/* */`).
- **OpenCL:** kernels in `.cl` under `opencl/`; CPU reference authoritative; only OpenCL crate uses `unsafe`.
- **Python:** support scripts/illustrations only; NumPy for array glue; `plt.show()` only in CLI/main, never in libs.
- **Parity work:** when porting from FireCore/SPAMMM/learn_Rust, cite the reference file+function in a comment (e.g. `// ported from FireCore/src/forcefields/uff.cpp:Uff::eval`).

## Navigation & Folder Roles

- **Key docs:** `CODEMAP.md` (structure, file inventory, crate graph) · `ARCHITECTURE.md` (crate layout, file naming, **folder roles & OKF metadata**) · `Import_other_Repos.md` (reference repos) · `DESIGN_GOALS.md` · `rust/forcefields/DESIGN.md` (forcefield data ownership) · `notes/ToDo_user.md`.
- **Test location:** backend module tests → crate `tests/`; GUI/composite app tests → `apps/tests/`. See `ARCHITECTURE.md` §File Naming.
- **Folder metadata (OKF):** every folder must have a `README.md` in [OKF format](https://okf.md/) (YAML frontmatter: required `type`; recommended `title`/`description`/`tags`/`timestamp` + markdown body). The binding folder-role table and frontmatter conventions live in `ARCHITECTURE.md` §Folder Roles.
- **Docs hygiene:** before writing, search existing implementations; after implementing, update the folder's `README.md` and `CODEMAP.md` if structure changed.
