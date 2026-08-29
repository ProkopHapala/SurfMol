---
type: folder
title: oclff/src/bin
description: CLI smoke-test binaries for the RRsp3 OpenCL harness. Each binary sets up a molecule system, uploads to GPU, runs relaxation, and saves diagnostics to debug/.
tags: [opencl, gpu, smoke-test, cli, raff, rrsp3, relaxation, parity]
timestamp: 2026-08-29
---

# oclff/src/bin — CLI smoke-test binaries

Standalone CLI binaries for testing the RRsp3 OpenCL harness end-to-end. Each binary packs molecules into the cluster-sorted GPU layout, uploads to the GPU, runs relaxation, and saves trajectory + convergence data to `debug/`.

## Binaries

### `raff_ocl_smoke.rs` — 2× water smoke test

Port of FireCore `pyBall/RigidAtomFF/RRsp3/test_RRsp3_smoke.py`.

**What it does:**
1. Sets up 2 water molecules (6 atoms, 2 nodes, 4 caps)
2. Packs into cluster-sorted layout (group_size=64, 1 workgroup)
3. Uploads state, neighs, exclusions, ports to GPU
4. Runs one `step_cluster` — checks local index ranges are valid
5. Perturbs positions (0.3 Å random) and runs GPU relaxation with force-based convergence
6. Saves trajectory (`traj.xyz`) and convergence (`convergence.tsv`) to `debug/raff_ocl_smoke/`
7. Optional `--parity`: runs CPU RAFF relaxation with matching solver, compares final geometry via Kabsch RMSD

**Usage:**
```bash
cargo run -p oclff --bin raff_ocl_smoke [--port current|orig|substep|shapematch|eigen] [--parity] [--traj PATH] [--tsv PATH]
```

**Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--port KERNEL` | `current` | Port kernel variant: `current`, `orig`, `substep`, `shapematch`, `eigen` |
| `--parity` | off | Run CPU↔GPU parity check (Kabsch RMSD on final geometry) |
| `--traj PATH` | `debug/raff_ocl_smoke/traj.xyz` | Trajectory output path |
| `--tsv PATH` | `debug/raff_ocl_smoke/convergence.tsv` | Convergence data output path |

**Parity results (with `--parity`):**
| Variant | CPU | GPU | Kabsch RMSD | Status |
|---------|-----|-----|-------------|--------|
| Memoryless | Adiabatic (Wahba) | Shapematch (Kabsch) | 0.0096 Å | PASS |
| Massfull | Dynamic (inertia) | Current (inertia) | 0.000000 Å | PASS |

---

### `raff_ocl_xylitol.rs` — 4× xylitol relaxation

Tests the RRsp3 harness on a larger, more complex molecule (21 atoms, 10 nodes, 12 caps per copy).

**What it does:**
1. Loads xylitol from `data/xyz/xylitol.xyz`
2. Detects bonds via covalent radii (using `moltopo::Builder`)
3. Classifies atoms as nodes (bond degree > 1) vs caps (H atoms)
4. Packs `n_copies` (default 4) into separate workgroups (256 total atoms with padding)
5. Perturbs positions (0.2 Å random)
6. Runs GPU relaxation with force-based convergence (max 5000 steps)
7. Saves trajectory, convergence TSV, initial perturbed XYZ, and final equilibrium XYZ to `debug/raff_ocl_xylitol/`

**Usage:**
```bash
cargo run -p oclff --bin raff_ocl_xylitol [--port current|shapematch] [--n_copies N]
```

**Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--port KERNEL` | `current` | Port kernel: `current` (massfull) or `shapematch` (massless) |
| `--n_copies N` | `4` | Number of xylitol copies (each in its own workgroup) |

**Known issue:** Collisions are disabled (`radius=0`) because xylitol has 1-4 atom pairs within collision radius that are not excluded by `make_exclusions_1st_2nd` (only 1-2 and 1-3 excluded). See [`/crates/libs/oclff/README.md`](../../README.md) §Open issues.

**Convergence behavior:** Xylitol shows slower convergence (~5000 steps to reach ~2e-5 max force) compared to water, due to 10 coupled nodes. The convergence plot shows fast initial drop followed by slow linear convergence — expected for coupled rigid-body systems.

## Debug artifacts

Both binaries save to `debug/<binary_name>/`:
- `traj.xyz` — XYZ trajectory (one frame per N steps)
- `convergence.tsv` — TSV with columns: `step`, `max_disp`, `max_force`, `rms_disp`, `rms_force`
- `initial_perturbed.xyz` — perturbed starting geometry (xylitol only)
- `final_equilibrium.xyz` — final converged geometry (xylitol only)

Plot with: `python3 debug/raff_ocl_smoke/plot_convergence.py` (or `raff_ocl_xylitol`).

## See also

- [`../README.md`](../README.md) — oclff crate README (full API, open issues)
- [`/notes/conventions/relaxation_convergence.md`](/notes/conventions/relaxation_convergence.md) — relaxation convergence methodology
- [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) — RAFF cross-implementation map
- FireCore `pyBall/RigidAtomFF/RRsp3/test_RRsp3_smoke.py` — original Python smoke test (parity reference)
