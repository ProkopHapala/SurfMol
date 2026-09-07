---
type: topical-audit
title: "RAFF — Rigid-Atom Force Field (port-based, non-reactive)"
description: Cross-implementation map for the port-based rigid-atom forcefield. Covers rotation solvers (dynamic vs analytical), dynamics strategies (force-MD vs XPBD vs projective), non-bonded models (full Morse vs compact-exp vs split-collision), and GPU layout variants. Tracks SurfMol implementation status vs FireCore/SPAMMM references.
tags: [topical-audit, raff, rigid-atom, port-based, forcefield, rotation-solver, xpbd, projective-dynamics, split-collision, opencl, gpu]
timestamp: 2026-09-28
---

# RAFF — Rigid-Atom Force Field (port-based, non-reactive)

## Summary

The port-based rigid-atom forcefield (RAFF) models each atom as a rigid body with 1–4 "ports" — body-frame attachment points for bonded neighbors. The port tip position `tip_i = x_i + R_i · (l0 · a_α)` is rotated by the atom's quaternion `q_i`. A harmonic spring between `tip_i` and the neighbor position `x_j` drives both translation and rotation. This replaces explicit angle/dihedral terms with a single port-spring energy, making the forcefield naturally suited to rigid-body dynamics and GPU parallelization.

## Design axes (4 independent branches)

See [`/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md`](/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md) §1 for the full design matrix.

| Axis | Variants | SurfMol status |
|---|---|---|
| **1. Rotation solver** | Dynamic DOF (omega+tau+I) · Analytical (polar/eigen/newton) | Dynamic: **done** (`raff.rs` step_force_md). Analytical: **done** (`solve_rotation_wahba` — Horn quaternion eigen, used in Adiabatic mode). Polar + Newton: **not ported**. |
| **2. Dynamics strategy** | Force-MD (symplectic Euler) · XPBD · Projective (Jacobi) | Force-MD: **done** (`step_force_md`). InertialReset: **done** (`step_inertial_reset` — velocity reset on v·F<0). FIRE: **done** (`step_fire` — adaptive dt + `FireState`). All 3 position-based variants **done**: `PosSolver::{PbdCompliance, Xpbd, Projective}` via `step_position_based` dispatcher. PBD-compliance = original `step_xpbd` (no lagged λ). True XPBD = lagged `λ_acc` (Macklin 2016). Projective = Jacobi global solve, linearized `r_arm`, **inner-coupled rotation** (both x and q updated each inner iteration). **Convergence test passes** (all reach same geometry, Kabsch RMSD < 1e-3). **Benchmark binary** `raff_bench` measures steps/evals/wall-time. **All 6 solver modes wired to editor** via GUI dropdown + `--raff-solver` CLI flag. |
| **3. Non-bonded model** | Full Morse+Coulomb · Compact-exp · Split-collision (3 sub-variants) | LJ+Coulomb: **done** (`eval_nonbonded`). Morse: **not ported**. Compact-exp: **not ported**. Split-collision: **not ported**. |
| **4. GPU layout** | One-WG-per-body · Cluster-sorted · LFF-Jacobi · per-bond · single-WG | **Cluster-sorted: implemented** (`oclff::rrsp3::RRsp3`, 5 port kernel variants, 14 OpenCL kernels in `RRsp3.cl`). CPU↔GPU parity verified (memoryless: Adiabatic vs Shapematch RMSD=0.0096Å; massfull: Dynamic vs Current RMSD=0.000000Å). step_cluster (relaxation) + step_dynamics (MD). Smoke tests: 2×water + 4×xylitol. See [`/crates/libs/oclff/README.md`](/crates/libs/oclff/README.md). Other layouts: not started. |

## Implementations

### SurfMol (Rust — CPU reference, authoritative)

| Component | File | LOC | Status | Notes |
|---|---|---|---|---|
| **RAFF core engine** | `crates/libs/molff/src/raff.rs` | 1550 | **active** | `RaffTopology` (port geometry, neighbors, bond params, mass, inertia), `RaffState` (pos, vel, quat, omega), `RaffConfig` (dt, damping, orient mode, `pos_solver`, `pd_inertia`, `vel_reset`, heavy-ball `bmix_*`, `box_cfg`), `BoxCfg` (harmonic AABB constraint), `NbConfig` (non-bonded), `FireState` (adaptive dt/alpha, `Default` impl). All 6 solver modes: `step_force_md`, `step_inertial_reset`, `step_fire`, `step_position_based` (dispatches to PBD/XPBD/Projective), `step_proximal` (stub), `solve_collisions`. **`eval_box_forces`**: harmonic AABB restoring force. **Port geometry:** `set_port_geometry_from_reference` (per-atom ARAP, default) and `set_port_geometry_from_types` (idealized sp2/sp3, needs neighbor reindexing — see §1.4 of theory doc). **Mass:** defaults to 1.0 (relaxation mode); physical masses needed for dynamics. |
| **Port force evaluation** | `raff.rs:393` | — | **active** | `eval_port_forces`: harmonic spring `F = k_p · (x_j − tip_i)`, torque `τ = r_arm × F`. Energy `E = ½ k_p |e|²`. |
| **Analytical rotation (Wahba/Horn)** | `raff.rs:440` | — | **active** | `solve_rotation_wahba`: builds 4×4 symmetric K-matrix from port covariance, dominant eigenvector via power iteration (warm-started). Used in `OrientMode::Adiabatic`. Single-port special case: `quat_align_vectors` (half-vector method). |
| **Non-bonded (LJ+Coulomb)** | `raff.rs:585` | — | **active** | `eval_nonbonded`: O(N²) pairwise LJ + Coulomb with 1-2/1-3 exclusion. Force clamping (`f_max`). **`eval_nonbonded_broad`**: AABB-culled variant (identical results). No Morse, no compact-exp, no split-collision yet. |
| **Collision solver (XPBD)** | `raff.rs:723` | — | **active** | `solve_collisions`: Jacobi sphere-sphere XPBD constraint, positional correction. |
| **Finite-difference checks** | `raff.rs:907–1042` | — | **active** | `fd_check_forces`, `fd_check_torques`, `check_translation_invariance`, `check_rotation_invariance`, `check_adiabatic_torque_residual` — diagnostic tests, not pass/fail. |
| **Tests** | `crates/libs/molff/tests/test_raff.rs` | 607 | **active** | 22 tests: port force parity, rotation solver convergence, energy conservation, momentum conservation, XPBD constraint satisfaction, collision resolution, adiabatic torque residual. All passing. |
| **Convergence + Kabsch tests** | `crates/libs/molff/tests/test_raff_convergence.rs` | 216 | **active** | 4 tests: force-MD + all 3 position-based solvers converge to same geometry (Kabsch RMSD < 1e-3 vs exact input reference). Kabsch invariants (identity/translate/rotate → 0). chain4 dihedral null space documented. All passing. (2026-08-28) |
| **Benchmark binary** | `crates/libs/molff/src/bin/raff_bench.rs` | 185 | **active** | Parameter sweep: `{dt, iters, over_relax}` × `{PBD, XPBD, Projective, ForceMD}` on CH4/water/tree-20/tree-100. Reports `n_steps`, `n_port_evals`, `t_wall_us` (single-thread). Run: `cargo run --release -p molff --bin raff_bench`. (2026-08-28) |
| **Broad-phase parity tests** | `crates/libs/molff/tests/test_broad_phase.rs` | 177 | **active** | 3 tests: `eval_broad` vs `eval` (NonBondedFF), far molecules (0 BP pairs), `eval_nonbonded_broad` vs `eval_nonbonded` (RAFF). All passing. |
| **Benzene diagnostic** | `crates/libs/molff/tests/test_benzene_diag.rs` | 178 | **active** | Regression test: per-atom ARAP port geometry gives E_port=0 and stable benzene structure. Documents the bug where idealized sp2 ports caused geometrically inconsistent port-to-neighbor assignment. |
| **Editor integration** | `crates/apps/editor/src/main.rs` | 1700 | **active** | `BondedFFMode::Raff` in `MolWorld`. `do_raff_step()` dispatches to all 6 solver modes (ForceMD/InertialReset/FIRE/PBD/XPBD/Projective) via `RaffSolverMode` enum. GUI dropdown for solver mode + orient mode + PD sub-options (iters, pd_inertia, vel_reset, heavy-ball) + box constraint (enable, k, min/max). CLI flags: `--raff`, `--raff-solver`, `--raff-orient`, `--raff-iters`, `--raff-pd-inertia`, `--raff-vel-reset`, `--raff-hb`, `--box`, `--box-min`, `--box-max`, `--box-k`, `--2d`, `--nmols`, `--layout`, `--show-aabb`. Uses `set_port_geometry_from_reference` (per-atom ARAP). See [`/userguide/raff.md`](/userguide/raff.md). |
| **Legacy RigidSp3 (force-MD only)** | `crates/libs/molff/src/rigid_sp3.rs` | 237 | **deprecated** | Original single-variant implementation. Kept as reference for `Dynamic+ForceMD` baseline. Superseded by `raff.rs`. |

### SurfMol (Rust/OpenCL — GPU harness)

| Component | File | LOC | Status | Notes |
|---|---|---|---|---|
| **RRsp3 GPU harness** | `crates/libs/oclff/src/rrsp3.rs` | ~1000 | **active** | `RRsp3` struct wraps 14 OpenCL kernels from `opencl/RRsp3.cl`. `PortKernel` enum (Current/Orig/Substep/Shapematch/Eigen), `StepConfig`, `step_cluster` (relaxation), `step_dynamics` (MD). Upload/download for state/neighs/excl/ports/bk_slots. NVIDIA GPU selection via `nvidia_proque()`. Kernel embedded at compile time via `include_str!`. Port of FireCore `RRsp3.py`. See [`/crates/libs/oclff/README.md`](/crates/libs/oclff/README.md). |
| **Host-side data prep** | `crates/libs/oclff/src/pack.rs` | ~600 | **active** | `pack_molecules` (cluster-sorted layout), `build_neighs_from_bonds`, `make_exclusions_1st_2nd`, `make_bk_slots_clustered`, `make_ports_from_neighs`, `masses_from_elems`. Ported from `RRsp3.py` host helpers. |
| **Smoke test (water)** | `crates/libs/oclff/src/bin/raff_ocl_smoke.rs` | ~340 | **active** | 2×H₂O: single step, perturbed relaxation with force-based convergence, optional CPU↔GPU parity (Kabsch RMSD). Saves trajectory + convergence TSV to `debug/raff_ocl_smoke/`. |
| **Smoke test (xylitol)** | `crates/libs/oclff/src/bin/raff_ocl_xylitol.rs` | ~220 | **active** | 4×xylitol (21 atoms each, 10 nodes): force-based convergence, trajectory + convergence TSV. Saves to `debug/raff_ocl_xylitol/`. Collisions disabled (1-4 exclusion issue). |
| **GPU kernel source** | `opencl/RRsp3.cl` | 1748 | **active** | 14 kernels: predict_dynamics, update_velocities_dynamics, update_bboxes_rigid, build_local_topology_rigid, compute_collision_cluster_rigid, 5 port variants (current/orig/substep/shapematch/eigen), compute_tips, compute_optimal_rotation_eigen, apply_corrections_rigid_ports. Compile-time macros: GROUP_SIZE, MAX_GHOSTS, ENABLE_COLL, ENABLE_PORT. |

### FireCore (C++ — parity/perf reference)

| Component | File | Lines | Status | Notes |
|---|---|---|---|---|
| Force-based reactive RAFF | `cpp/common/molecular/RARFF_SR.h` | `pairEF` 441 | reference | Parity target for force-based path. Morse + angular gating. |
| XPBD massfull kernel | `pyBall/RigidAtomFF/RRsp3/RRsp3.cl` | 659–911 | reference | `compute_ports_cluster_rigid` — port for `step_xpbd`. |
| XPBD massless shapematch (polar) | `RRsp3.cl` | 1089 | reference | Newton–Schulz polar decomposition. **Not ported to SurfMol.** |
| XPBD massless eigen (Davenport q) | `RRsp3.cl` | 1260, 1516 | reference | Horn quaternion eigen. **Ported** as `solve_rotation_wahba`. |
| XPBD massless Newton in ω | `RRsp3.cl` | 916 | reference | Newton–Raphson in angular-velocity space. **Not ported.** |
| XPBD collision (Jacobi) | `RRsp3.cl` | 562 | reference | `compute_collision_cluster_rigid`. **Ported** as `solve_collisions`. |
| Piecewise quadratic split | `cpp/common/math/Forces.h` | 511–539 | reference | `getSR_x2_smooth`. **Not ported.** |
| Projective Dynamics (C++) | `ProjectiveDynamics_d.h/.cpp` | 124, 198, 505 | reference | Jacobi / Cholesky solvers. **Not ported.** |
| Spring drag (mouse picking) | `MolWorld_sp3.h` | 1505–1508 | reference | `getForceSpringRay(p, hray, ray0, Kpick)`. **Ported** to editor `main.rs:45`. |

### SPAMMM (Python/OpenCL — production non-bonded reference)

| Component | File | Lines | Status | Notes |
|---|---|---|---|---|
| Compact-exp Morse kernel | `kernels/Forces.cl` | 260–273 | reference | `compact_exp_pair_EF` — branch-free, n=8. **Not ported.** Primary target for Phase 2a. |
| Unified site-pair primitive | `kernels/Forces.cl` | 279–309 | reference | `pairff_unified_site_EF` — atoms + epairs + sigma-holes, same instructions. **Not ported.** |
| Rigid-body pairff kernel | `kernels/rigid.cl` | 2452–2623 | reference | 5 kernel variants, all compact-exp. **Not ported.** |
| LFF projective Jacobi | `spammm/forcefields/LFFSolver.py` + `kernels/LFF.cl` | 91, 61 | reference | Linearized FF from UFF spring network. **Not ported.** Phase 2c target. |
| Interactive drag demo | `demos/demo_pairff.py` | full | reference | Working interactive rigid-body FF with mouse drag. **Concept ported** to editor. |

## Parity status

| Pair | Verified | Tolerance | Test |
|---|---|---|---|
| SurfMol `eval_port_forces` vs analytical | **yes** | 1e-10 | `test_raff.rs::test_port_force_*` |
| SurfMol `solve_rotation_wahba` vs FD gradient | **yes** | 1e-6 | `test_raff.rs::test_adiabatic_torque_residual` |
| SurfMol `step_force_md` energy conservation | **yes** | <1e-4 drift / 1000 steps | `test_raff.rs::test_force_md_energy_conservation` |
| SurfMol `step_xpbd` constraint satisfaction | **yes** | <1e-8 | `test_raff.rs::test_xpbd_converges_ch4` |
| SurfMol all position-based solvers → same geometry | **yes** | Kabsch RMSD < 1e-3 | `test_raff_convergence.rs::test_same_geometry_{ch4,water}` |
| SurfMol position-based vs force-MD geometry parity | **yes** | Kabsch RMSD < 1e-3 | `test_raff_convergence.rs` (all 3 PosSolver variants vs ForceMD, CH4+water) |
| SurfMol `get_force_spring_ray` vs FireCore | **yes** (by construction) | exact | Same formula: `-dp_perp * k` |
| SurfMol vs FireCore `RARFF_SR.h::pairEF` | **no** | 1e-3 rel | **TODO** — needs Morse port first |
| SurfMol vs SPAMMM `compact_exp_pair_EF` | **no** | 1e-3 | **TODO** — needs compact-exp port |
| CPU↔GPU parity (memoryless: Adiabatic vs Shapematch) | **yes** | Kabsch RMSD 0.0096 Å | `raff_ocl_smoke --parity` (2×H₂O, 500 steps) |
| CPU↔GPU parity (massfull: Dynamic vs Current) | **yes** | Kabsch RMSD 0.000000 Å | `raff_ocl_smoke --parity` (2×H₂O, 500 steps) |
| CPU↔GPU parity (Orig/Substep/Eigen kernels) | **no** | — | **TODO** — compile and run but no parity comparison |
| CPU↔GPU parity (step_dynamics MD path) | **no** | — | **TODO** — only step_cluster (relaxation) tested |

## Open issues / TODO

- [ ] **Polar decomposition rotation solver** (Axis 1b-i): port Newton–Schulz `R ← ½R(3I−RᵀR)` from `RRsp3.cl:1089`. Currently only Horn eigen is implemented.
- [ ] **Newton-in-ω rotation solver** (Axis 1b-iii): port from `RRsp3.cl:916`. Local 3×3 Hessian solve.
- [x] **Projective Dynamics wiring** (Axis 2c): `step_proximal` (the IMEX stub) exists but is superseded. The real Projective Dynamics solver is `PosSolver::Projective` in `solve_projective_jacobi` (tested + benchmarked + wired to editor). `step_proximal` should either be removed or refactored to call `step_position_based` with `PosSolver::Projective`.
- [x] **Position-based solvers wired to editor**: `do_raff_step` dispatches to all 6 solver modes via `RaffSolverMode` enum. GUI dropdown + `--raff-solver` CLI flag. All modes tested and functional.
- [ ] **Morse non-bonded** (Axis 3a): add `Morse(D_e, a, r0)` to `eval_nonbonded` as physics reference.
- [ ] **Compact-exp non-bonded** (Axis 3a'): port `compact_exp_pair_EF` from SPAMMM — the production model.
- [ ] **Split-collision** (Axis 3b): implement 3 sub-variants (piecewise quadratic, hard contact + erf/erfc, compact-exp split). Required for stable XPBD with non-bonded.
- [ ] **Central-force recoil** (Axis 1b conservation): analytical rotation solvers must project port force onto center–center line to conserve `L_trans`. Currently the analytical solver runs *after* force eval — the force still uses tip→atom, which violates angular momentum for memoryless rotation. See roadmap §1 conservation consequence.
- [ ] **Capping atoms as rigid appendix** (roadmap §4.2): H atoms should be `pos = host_pos + host_quat · port_local[slot] · l_H` (no independent DOF). Currently H is a free-translation point atom.
- [x] **GPU implementation (Axis 4 — cluster-sorted)**: `oclff::rrsp3::RRsp3` implemented. 5 port kernel variants, 14 OpenCL kernels, step_cluster + step_dynamics. CPU↔GPU parity verified (memoryless + massfull). Smoke tests: 2×water + 4×xylitol. See [`/crates/libs/oclff/README.md`](/crates/libs/oclff/README.md).
- [ ] **GPU 1-4 exclusions**: `make_exclusions_1st_2nd` only excludes 1st/2nd neighbors. Molecules with 1-4 pairs within collision radius (e.g. xylitol) need 1-4 exclusion list or collisions must be disabled (radius=0).
- [ ] **GPU `k_coll` unused**: collision kernel uses geometric overlap, not `k_coll`. Setting `k_coll=0` has no effect. Workaround: `radius=0`.
- [ ] **GPU non-bonded (LJ/Coulomb)**: no GPU kernel for inter-molecular non-bonded forces. CPU `eval_nonbonded` is the only option.
- [ ] **GPU `step_dynamics` parity**: MD path (predict → velocities) has no parity test. Only relaxation tested.
- [ ] **GPU ghost atom test**: smoke tests use molecules far apart. Cross-workgroup ghost atom path untested with overlapping bboxes.
- [ ] **Electron-pair / sigma-hole sites** (Axis 3a'): port `add_electron_pairs_via_atomic_system` from SPAMMM for Hbond directionality.

## Key caveats (gotchas)

- **Port force sign convention**: `F = k_p · (x_j − tip_i)` — force on atom `i` is *toward* the neighbor. Torque `τ = r_arm × F` where `r_arm = quat_rotate(q, port_local · l0)`. Energy `E = ½ k_p |e|²` with `e = x_j − tip`. Each bond counted twice (once per port), so per-port stiffness `k_p = K_bond / 2`.
- **Adiabatic vs Dynamic orientation**: `OrientMode::Adiabatic` re-solves `q_i` every step via `solve_all_rotations` (memoryless, no `omega`). `OrientMode::Dynamic` integrates `omega` from torque. Adiabatic is the default in the editor — it is more stable but does not conserve angular momentum (no rotational inertia).
- **2D constraint** (`--2d` flag): zeros z-component of forces/torques, clamps z-position to 0, zeros z-velocity. Only rotation around z-axis allowed (`tau.x = tau.y = 0`). This is a hard constraint, not a soft potential.
- **Harmonic box constraint** (`--box` flag, `BoxCfg`): confines atoms within an AABB with restoring spring force `F = k·(limit - x)`. Force-based solvers add it to the force array before integration; position-based solvers apply it as a post-solve position correction `δx = k·(limit-x)·dt²/m`. Simulates a soft surface boundary. See [`/userguide/raff.md`](/userguide/raff.md) §Harmonic box constraint.
- **Stopping criterion** (`zero_v_on_opposition`): when `Σ dot(v_i, f_i) < 0` (total force opposing total motion), zero all velocities. This is the FireCore `MolWorld_sp3` relaxation heuristic — converges to equilibrium without oscillation. Without it, the molecule oscillates forever at zero damping.
- **Camera orthographic projection** (trackball.rs): the `view_proj` matrix must be stored in **column-major** order (`[[col0], [col1], [col2], [col3]]`) so that `clip.w = 1.0` for orthographic projection. A transposed (row-major) layout causes `clip.w` to vary with position, triggering the GPU's perspective divide and producing a "fisheye" distortion. See `trackball.rs:53–58`.
- **Port rendering sync**: in RAFF mode, ports must be drawn from `topo.port_tip(state, i, s)` (which applies `state.quat[i]`), NOT from `world.rigid_sp3.get_port_tip()` (which uses fixed geometry). The fixed-geometry path is only correct for `RigidSp3` mode where quaternions are not tracked separately.

## See also

- [`/userguide/raff.md`](/userguide/raff.md) — RAFF solver modes end-user guide (ForceMD/FIRE/PBD/XPBD/Projective, box constraint, theory)
- [`/userguide/editor.md`](/userguide/editor.md) — editor end-user guide (all CLI flags, GUI controls)
- [`/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md`](/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md) — phased implementation plan with checkboxes
- [`/notes/designs/raff_theory_equations.md`](/notes/designs/raff_theory_equations.md) — all mathematical formulations centralized
- [`/notes/reports/2026-08-28_raff_solver_benchmark_report.md`](/notes/reports/2026-08-28_raff_solver_benchmark_report.md) — benchmark report with performance tables
- [`/crates/libs/molff/README.md`](/crates/libs/molff/README.md) — molff crate overview (includes raff.rs module)
- [`/crates/apps/editor/README.md`](/crates/apps/editor/README.md) — editor app (RAFF integration, CLI flags)
- FireCore `Analytic_Procrustes_doc.md` — ARAP/Procrustes rotation math
- FireCore `RRsp3_momentum_design.md` — XPBD vs PD conservation analysis
- SPAMMM `doc/Topics/ForceFields/PairFF.md` — production compact-exp non-bonded design
