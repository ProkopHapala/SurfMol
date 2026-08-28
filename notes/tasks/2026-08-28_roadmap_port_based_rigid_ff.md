---
type: work-notes
title: "Roadmap — Port-based rigid-atom forcefield (non-reactive RAFF) on CPU, then GPU"
description: Phased plan to implement the non-reactive port-based rigid-atom forcefield in SurfMol, covering rotation solvers, force-based vs position-based dynamics, split-collision non-bonded, and GPU kernel layouts. Consolidates design docs from FireCore and SPAMMM.
tags: [roadmap, forcefield, rigid-atom, port-based, RAFF, XPBD, position-based-dynamics, split-collision, OpenCL, GPU]
timestamp: 2026-08-28
---

# Roadmap — Port-based rigid-atom forcefield (non-reactive RAFF)

This roadmap consolidates the design decisions scattered across FireCore, SPAMMM, NumericalMathPlayground, and the existing SurfMol codebase into one phased plan. The goal: make SurfMol a convenient environment for **testing different forcefield variants** of the port-based rigid-atom model, first on CPU (physics-correct), then on GPU (performance).

**Companion document:** [`raff_theory_equations.md`](../designs/raff_theory_equations.md) — all mathematical formulations (port energy, rotation solvers, XPBD, PD, compact non-bonded, split-collision, quaternion updates) centralized in one place for comparing pros/cons.

## 0. Current state of SurfMol (baseline)

What already exists and is usable:

| Component | File | Status |
|---|---|---|
| `RigidSp3FF` — port-based rigid FF, **force-based + dynamic rotational DOF** only | `crates/libs/molff/src/rigid_sp3.rs` (237 LOC) | Working: sp3/sp2/sp1 port geometry, quaternion integration via `omega`, torque → angular velocity → `dq`. One solver variant only. **Superseded by `raff.rs`.** |
| **`raff.rs` — multi-variant RAFF engine** | `crates/libs/molff/src/raff.rs` (1085 LOC) | **Working**: `RaffTopology`/`RaffState`/`RaffConfig`, port forces, Wahba/Horn rotation solver, LJ+Coulomb non-bonded, `step_force_md`, `step_xpbd`, `step_proximal`, collision solver, FD checks. 22 tests passing. |
| `MolWorld` orchestrator | `crates/libs/surfmol/src/mol_world.rs` (140 LOC) | Dispatches `BondedFFMode::{Uff, RigidSp3, Raff}`. Already composes bonded + nonbonded + surface. |
| `NonBondedFF` — LJ + Coulomb + H-bond, O(N²) | `crates/libs/molff/src/nonbonded.rs` (300 LOC) | Working, with 1-2/1-3 exclusion + PBC. **No Morse, no split-collision yet.** |
| `Uff` — harmonic bonded FF (reference for bond params) | `crates/libs/molff/src/uff.rs` (665 LOC) | Working. Owns `bon_params` used by `RigidSp3FF` and `raff.rs`. |
| **`editor` app — RAFF integration** | `crates/apps/editor/src/main.rs` (1433 LOC) | **Working**: `--raff`/`--2d`/`--atom-scale` CLI flags, RAFF settings GUI panel, `do_raff_step()` with spring drag + 2D constraint + stopping criterion, port rendering synced with RAFF quaternions, camera orthographic fix. |
| Data layout primitives | `numtypes`, `pgraph`, `spacc` | `Vec3d/Quat4d`, `AlignedVec<T,64>`, `FixedAdj<4>`, `Buckets`, `Aabb3d` all ready. |

**What is missing (the work):**
1. ~~Alternative rotation solvers (analytical/memoryless vs dynamic DOF).~~ **Partly done** — Horn eigen (Wahba) implemented. Polar + Newton-in-ω still TODO.
2. ~~Position-based dynamics (XPBD / Projective) as an alternative to force-based MD.~~ **Partly done** — `step_xpbd` + `step_proximal` implemented. Projective not wired to editor.
3. Split-collision non-bonded (linearized short-range + non-linear dissociative). **Not started.**
4. GPU (OpenCL) implementation with layout variants. **Not started — CPU first.**
5. ~~GUI improvements for side-by-side forcefield comparison + interactive dragging feedback.~~ **Partly done** — RAFF GUI panel, spring drag, port visualization, atom scale slider. Side-by-side mode still TODO.

### Progress tracker (2026-09-28 snapshot)

See [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) for the full cross-implementation map.

**Completed:**
- [x] `raff.rs` module: `RaffTopology`, `RaffState`, `RaffConfig`, `NbConfig` data structures
- [x] Port force evaluation (`eval_port_forces`) — harmonic spring, torque, energy
- [x] Analytical rotation solver — Wahba/Horn quaternion eigen (`solve_rotation_wahba`)
- [x] Adiabatic orientation mode (`OrientMode::Adiabatic` → `solve_all_rotations`)
- [x] Dynamic orientation mode (`OrientMode::Dynamic` → omega integration)
- [x] Force-MD integration (`step_force_md`) — symplectic Euler with damping
- [x] XPBD integration (`step_xpbd`) — port constraints + collision
- [x] Projective dynamics stub (`step_proximal`) — Jacobi, not wired to editor
- [x] Collision solver (`solve_collisions`) — Jacobi sphere-sphere XPBD
- [x] Non-bonded LJ+Coulomb (`eval_nonbonded`) with 1-2/1-3 exclusion
- [x] Finite-difference diagnostic checks (forces, torques, invariances)
- [x] 22 unit tests — all passing
- [x] `BondedFFMode::Raff` in `MolWorld`
- [x] Editor `--raff` CLI flag (start in RAFF mode, show ports, enable non-bonded)
- [x] Editor `--2d` CLI flag (flatten to z=0, constrain forces/velocities to 2D)
- [x] Editor `--atom-scale` CLI flag + GUI slider (adjust atom render size)
- [x] Editor `do_raff_step()` — per-frame relaxation with spring drag, 2D constraint
- [x] Editor stopping criterion (`zero_v_on_opposition` for RAFF path)
- [x] Editor RAFF settings GUI panel (non-bonded toggle, orient mode, sliders, energy display)
- [x] Editor port rendering synced with RAFF quaternions (`topo.port_tip(state, i, s)`)
- [x] Camera orthographic projection fix (column-major `view_proj` matrix in `trackball.rs`)
- [x] Default damping (0.1) and per_frame (20) for `--raff` mode
- [x] **All 3 position-based solver variants** (Axis 2) — `PosSolver::{PbdCompliance, Xpbd, Projective}` selectable via `RaffConfig.pos_solver`. `step_position_based` dispatcher; `step_xpbd` kept as PbdCompliance wrapper. True XPBD has lagged `λ_acc`; Projective Dynamics uses Jacobi global solve with linearized `r_arm`. (2026-08-28)
- [x] **Convergence-to-same-geometry test** (Q2a) — `tests/test_raff_convergence.rs`: force-MD + all 3 position-based solvers relax to the same geometry (Kabsch RMSD < 1e-3 vs exact input reference). 4 tests, all passing. Documents the chain4 dihedral null space. (2026-08-28)
- [x] **Kabsch rigid-body alignment** (`kabsch_rmsd`) — optimal rotation RMSD for comparing solver outputs invariant to translation/rotation drift. (2026-08-28)
- [x] **Parameter-sweep benchmark binary** (Q2b) — `src/bin/raff_bench.rs`: sweeps `{dt, iters, over_relax}` × `{PBD, XPBD, Projective}` × `{ForceMD}` on CH4/water/tree-20/tree-100. Reports `n_steps`, `n_port_evals`, `t_wall_us` (single-thread). First results: PBD-compliance fastest (7-31 macrosteps with over-relax), XPBD/Projective ~3-10x more iterations but stable everywhere; Force-MD 100-1000x slower. PBD diverges at dt=0.1+or=1.9. (2026-08-28)

**TODO (remaining):**
- [ ] **Wire position-based solvers to editor** — `do_raff_step` currently inlines force-MD only; add `PosSolver` selector to GUI + dispatch to `step_position_based`. (Q2b follow-up)
- [ ] **Benchmark: non-bonded split + position-based** — once split-collision (Phase 2b) is implemented, re-run `raff_bench` with non-bonded enabled to measure the IMEX proximal split (theory doc §11).
- [ ] **Benchmark: `H_max` stability map** — theory doc §11.5/§11.7 calls for `H_max` (max stable macrostep) per solver, not just steps-to-tolerance. Add to `raff_bench`.
- [ ] **Benchmark: `U''(r)` plot for non-bonded splits** — theory doc §11.7 "most important plot". Add a plotting utility (Phase 2).
- [ ] **Benchmark: distortion types (D1/D2/D3)** — theory doc §11.8. Three distortion classes: (D1) random displacement [done], (D2) uniaxial stretch along PCA long axis [done] — the long-narrow-valley / Rosenbrock pathology that motivates multi-grid, (D3a) dihedral soft DOF (H2O2) [done], (D3b) non-covalent assembly (benzoic acid dimer) [blocked on H-bond/electron-pair system, Phase 2c]. The benchmark must cover D1+D2+D3a; D3b is staged.
- [ ] **Benchmark: two convergence targets (T1/T2)** — theory doc §11.8. T1 = accurate (RMSD < 0.0001 Å, max|F| < 0.001 meV/Å). T2 = rough/interactive (RMSD < 0.05 Å, max|F| < 0.1 eV/Å). Report steps-to-T2 and steps-to-T1 separately. **T2 is the higher-priority target** (interactive pre-opt + global relaxation). [done]
- [ ] **Benchmark: convergence + force curves (log-scale plots)** — residual RMSD and max|F| vs macrostep/soft-eval count, log-scale, per solver × distortion × molecule. PNG plots in `debug/raff_bench/`. [done]
- [ ] Polar decomposition rotation solver (Newton–Schulz, `RRsp3.cl:1089`)
- [ ] Newton-in-ω rotation solver (`RRsp3.cl:916`)
- [ ] Central-force recoil for analytical rotation (conserve `L_trans`)
- [ ] Morse non-bonded (physics reference, `Forces.h` / `Forces.cl:getMorsePLQH`)
- [ ] Compact-exp non-bonded (production model, `Forces.cl:compact_exp_pair_EF`)
- [ ] Split-collision: piecewise quadratic (`getSR_x2_smooth`)
- [ ] Split-collision: hard contact + erf/erfc (`SoftSplineHardAtomCore.chat.md`)
- [ ] Split-collision: compact-exp split (reuses compact-exp kernel)
- [x] **AABB broad-phase for non-bonded** (per-cluster AABB cull → narrow phase). Design: [`cluster_aabb_collision.md`](../designs/cluster_aabb_collision.md). Implemented: `aabb_overlap_margin` + `broad_phase_pairs` in `spacc`/`numtypes`, `BroadPhase` struct + `eval_broad`/`eval_nonbonded_broad` in `molff`, `eval_forces_broad` in `MolWorld`, editor `--nmols`/`--layout`/`--show-aabb` CLI + GUI visualization. Parity tests pass (`tests/test_broad_phase.rs`).
- [ ] **Per-type port reindexing** — `set_port_geometry_from_types` uses idealized sp2/sp3 directions, but `build_neighs_from_bonds` assigns ports in bond-list order, causing geometrically inconsistent port-to-neighbor pairing. Fix: add `reindex_ports_by_direction` that permutes neighbor slots to match idealized port directions. Per-atom ARAP (`set_port_geometry_from_reference`) is the current workaround/default. See `raff_theory_equations.md` §1.4.
- [ ] Capping atoms as rigid appendix (H = host_pos + host_quat · port · l_H)
- [ ] Electron-pair / sigma-hole site system
- [ ] Side-by-side forcefield comparison mode in editor
- [ ] Energy/momentum HUD in editor (E_pot, E_kin, |P|, |L| real-time)
- [x] **AABB broad-phase collision** (per-cluster AABB cull → narrow phase) — `spacc::broad_phase_pairs`, `molff::BroadPhase`, `eval_broad`/`eval_nonbonded_broad`, editor `--nmols`/`--layout`/`--show-aabb`. Parity tests pass.
- [ ] GPU: port CPU force-eval to OpenCL (`molff-ocl` crate)
- [ ] GPU: layout variants 4a–4e + benchmark
- [ ] GPU: CPU↔GPU parity test

## 1. Design axes (the 4 branches the user identified)

These are the **independent axes** we want to test combinations of. The architecture must let us swap each axis without rewriting the others.

### Axis 1 — Rotation solver: analytical (memoryless) vs dynamic DOF

| Variant | Description | Reference |
|---|---|---|
| **1a. Dynamic DOF** (current) | Quaternion is a mechanical DOF with angular velocity `omega`, inertia `I`, torque `tau`. Symplectic Euler: `omega += I⁻¹·tau·dt`, `q ← q · dq(omega·dt)`. | SurfMol `rigid_sp3.rs:move_atom_md`; FireCore `Rigid.cl:make_qrot`; SPAMMM `RigidBodyDynamics.py`. |
| **1b. Analytical / memoryless** | Quaternion is **not** a DOF. Each step, compute the optimal rotation that best aligns ports to neighbors (ARAP/Procrustes local step). No `omega`, no `tau` integration. Three sub-variants: | FireCore `Analytic_Procrustes_doc.md`; `RRsp3.cl` massless kernels. |
| 1b-i. **Newton–Schulz polar** | Build covariance `H = Σ k_j (p_j−c_p)(r_j−c_r)ᵀ`, iterate `R ← ½R(3I−RᵀR)` (3–5 steps), `q ← mat3_to_quat(R)`. | `RRsp3.cl:compute_ports_cluster_rigid_shapematch` l.1089; `Analytic_Procrustes_doc.md` §A. |
| 1b-ii. **Horn quaternion (K-matrix power iteration)** | Build 4×4 symmetric K from `H`, dominant eigenvector via 4 power iterations (warm-started from prev frame). Inherently det=+1. | `RRsp3.cl:compute_optimal_rotation_eigen` l.1260; `Analytic_Procrustes_doc.md` §B. |
| 1b-iii. **Newton–Raphson in ω-space** | Local 3×3 Hessian from port lever arms, `ω = H⁻¹τ` substeps. | `RRsp3.cl:compute_ports_cluster_rigid_substep_optimized` l.916. |

**Conservation consequence (critical, from `RRsp3_momentum_design.md` §6):**
- **Dynamic DOF (1a):** off-center port forces → torque absorbed by rotational inertia → conserves `L_total = L_trans + L_spin`.
- **Analytical (1b):** no rotational inertia → must project impulse onto **center–center line** (central force) to conserve `L_trans = Σ r×p`. Otherwise unbalanced torque violates angular momentum.

### Axis 2 — Dynamics: force-based (impulse/MD) vs position-based (XPBD/Projective)

| Variant | Description | Reference |
|---|---|---|
| **2a. Force-based MD** (current) | Eval forces → integrate velocity → integrate position. Symplectic Euler / velocity Verlet. Good for dynamics, energy conservation tracking. | SurfMol `rigid_sp3.rs`; FireCore `MolWorld_sp3.h`. |
| **2b. XPBD** | Constraints `C(x)=0` with compliance `α=1/(K·dt²)`. Position update `Δx = −C/(α+Σw‖∇C‖²)·w∇C`. Explicit, GPU-friendly, handles collisions naturally. Velocity derived from position delta. | `RRsp3.cl`; `RRsp3_momentum_design.md` §2.2. |
| **2c. Projective Dynamics (PD)** | Energy min: `E = ½dt⁻²(x−y)ᵀM(x−y) + ΣW_c(x)`. Solve `(M/dt²+L)Δx = forces`. Jacobi / Cholesky / momentum variants. Better for stiff systems / relaxation. | FireCore `ProjectiveDynamics_d.h/.cpp/_frag.cpp`; SPAMMM `LFF.cl` + `LFFSolver.py`. |

**Decision (from `RRsp3_momentum_design.md` §2.2):** XPBD is preferred over PD for the rigid-atom case because it is explicit, handles non-linear constraints (collisions) naturally, and needs no global matrix solve. PD is the better choice when the spring network is **linear** and **fixed topology** (LFF surrogate). We implement **both 2b and 2c** to compare; 2c reuses the existing `Uff` spring network as its linearization target.

**Combinations to test (the "4 cases"):** Axis 1 × Axis 2 gives 2×2 primary cases, but axis 1 has 3 analytical sub-variants. The 4 headline cases:
1. **Dynamic DOF + Force MD** (1a + 2a) — *current implementation, baseline.*
2. **Dynamic DOF + XPBD** (1a + 2b) — physical inertia, XPBD port constraints.
3. **Analytical (polar/eigen/newton) + Force MD** (1b + 2a) — memoryless rotation, central-force recoil, MD integration.
4. **Analytical + XPBD** (1b + 2b) — memoryless rotation, XPBD constraints, central-force recoil. *This is the `RRsp3` massless design.*

### Axis 3 — Non-bonded interaction: full Morse+Coulomb vs compact-exp vs split-collision

| Variant | Description | Reference |
|---|---|---|
| **3a. Full Morse + Coulomb** | Standard pairwise: `E = Morse(rij) + Coulomb(rij)` with 1-2/1-3 exclusion, cutoff. O(N²) or AABB-bucketed. | SurfMol `nonbonded.rs` (currently LJ, needs Morse option); FireCore `NBFF.h`; SPAMMM `Forces.cl:getMorsePLQH`. |
| **3a'. Compact exponential Morse** (recommended production model) | `V = E₀·y·(α·y − (1+α))`, `y = max(0, 1−β(ρ−R₀)/8)^8`, `ρ = √(r²+w²)−w`. Branch-free for atoms+epairs+sigma-holes (same instructions, different params). Converges to Morse as n→∞. **This is what SPAMMM `demo_pairff.py` uses in unified mode.** | NMP `FastPairwisePotentials.chat.md` (l.1392+); SPAMMM `Forces.cl:compact_exp_pair_EF` 260-273, `pairff_unified_site_EF` 279-309; `rigid.cl:rigid_body_pairff_unified_kernel` 2452. See `Import_other_Repos.md` §"PairFF non-bonded model" for full detail. |
| **3a''. Legacy Lorentzian Hbond** (superseded, kept for comparison) | 4-loop kernel: Morse+Coulomb (atom-atom) + Lorentzian `fcut·1/(w²+r²)` (atom-epair/sigma). Type branching → warp divergence. | SPAMMM `rigid.cl:rigid_body_pairff_kernel` 2198. See `raff_theory_equations.md` §4.3b. |
| **3b. Split-collision (linearized short-range + non-linear dissociative)** | Split the potential so the **short-range repulsive** part becomes a **linear/harmonic constraint** solvable by XPBD/PD, and the **long-range dissociative** part is an explicit external force. Two sub-approaches: | FireCore `doc/DevNotes/ToDo_FastCollision_2.md`, `ToDo_FastCollision_3.md`; `SoftSplineHardAtomCore.chat.md`. |
| 3b-i. **Piecewise quadratic** (`getSR_x2_smooth`) | `U₁=½k₁(r−R_min)²+E_min` for `r<R_cut` (convex, PD-solvable); `U₂=½k₂(r−R_cut2)²` for `R_cut≤r<R_cut2` (concave, explicit). C¹ continuous. | FireCore `Forces.h:511-539`. |
| 3b-ii. **One-sided hard contact + erf/erfc Coulomb split** (recommended by GPT review) | `U_hard = ½k_h[R_h−r]₊²` (one-sided contact, XPBD constraint); `1/r = erfc(r/σ)/r + erf(r/σ)/r` (short-range core + long-range smooth grid). Avoids the concave quadratic's unphysical attractive region. | `SoftSplineHardAtomCore.chat.md`; `pyBall/OCL/Surface_utils.py:2700-2830`. |
| 3b-iii. **Compact-exp split** (new, from theory doc §5.3) | `V_rep = E₀αy²` (convex for α>0, XPBD constraint) + `V_attr = −E₀(1+α)y` (explicit force). Uses the **same** compact-exp kernel for both parts. | `raff_theory_equations.md` §5.3. |

**Why 3b matters for position-based dynamics (axis 2b/2c):** XPBD constraints must be **convex** (or at least have a well-defined projection). The Morse attractive tail is concave → cannot be a PBD constraint directly. Splitting isolates the convex repulsive core as a constraint and treats the tail as an explicit force. This is the key enabler for stable PBD with non-bonded interactions.

**Production recommendation:** Use **3a' (compact-exp)** as the primary non-bonded model — it's the latest creation, already tested in SPAMMM `demo_pairff.py`, branch-free for atoms+epairs, and converges to Morse. For PBD variants, use **3b-iii (compact-exp split)** since it reuses the same kernel. Keep **3a (full Morse)** as the physics reference and **3a'' (legacy Lorentzian)** for comparison only.

### Axis 4 — GPU kernel layout variants

| Variant | Description | Reference |
|---|---|---|
| **4a. One workgroup per molecule/body** | `WG=32`, 4 atoms/thread → max 128 atoms/body. Force/torque reduced in local mem. Best for rigid molecules. | FireCore `Rigid.cl:191,196`; SPAMMM `rigid.cl`. |
| **4b. Cluster-sorted (one WG per cluster)** | `WG=64`, nodes first, ghost atoms in `MAX_GHOSTS=128` local slots. `bkSlots`/`revSlot` gather to avoid atomics. Best for port-based rigid atoms with local neighbor lists. | `RRsp3.cl:97-105,454`; `RRsp3.py:23-55`. |
| **4c. One WG per molecule (LFF/Jacobi)** | `lff_jacobi` — threads beyond `natoms` gated but hit barriers. Diagonal Jacobi, no global solve. | SPAMMM `LFF.cl:20,61`. |
| **4d. Per-bond / per-atom (UFF style)** | One thread per bond or per atom, global reduction. Simple, good for large systems. | FireCore `UFF.cl`. |
| **4e. Small-system single-WG** | For small systems (≤ WG size), run the entire force eval in one workgroup — no global memory round-trip. | Discussed in `DESIGN_GOALS.md` §2.3; not yet implemented in references. |

## 2. Phased implementation plan

### Phase 0 — Refactor `RigidSp3FF` for multi-variant support (CPU)

**Goal:** make the rotation-solver and dynamics-strategy axes swappable without duplicating the port geometry / force-eval code.

**Tasks:**
- [x] 0.1. Extract a `PortModel` struct (owns `nport`, `port_local`, bond-param lookup) from `RigidSp3FF`. This is shared by all variants. → **Done as `RaffTopology` in `raff.rs`.**
- [x] 0.2. Define a `RotationSolver` trait/enum with variants: `Dynamic { quat, omega, tau, invI }`, `AnalyticalPolar`, `AnalyticalEigen`, `AnalyticalNewton`. → **Done as `OrientMode::{Dynamic, Adiabatic}` + `solve_rotation_wahba` (Horn eigen). Polar + Newton still TODO.**
- [x] 0.3. Define a `DynamicsStrategy` enum: `ForceMD`, `Xpbd`, `Projective`. → **Done as `DynMode::{ForceMD, Xpbd}` + `step_proximal` stub.**
- [x] 0.4. Rename `RigidSp3FF` → `RaffFF` (rigid-atom FF) or keep `RigidSp3FF` as the dynamic+force variant and add `RaffFF` as the configurable one. **Ask user** which naming they prefer. → **Resolved: new `raff.rs` module, `rigid_sp3.rs` kept as legacy reference.**
- [x] 0.5. Keep the current `rigid_sp3.rs` as the `Dynamic + ForceMD` reference; do not delete. → **Done.**

**Verification:** existing `test_rigid_sp3.rs` must still pass on the `Dynamic+ForceMD` path. → **Passing.**

### Phase 1 — Implement all 4 CPU cases (Axis 1 × Axis 2)

**Goal:** physics-correct CPU implementations of all 4 headline combinations, each independently testable.

**1a. Dynamic DOF + Force MD** — *already exists* (`rigid_sp3.rs`). Use as baseline.
- [x] Port force eval + torque + symplectic Euler integration. → **Done in `raff.rs:step_force_md`.**

**1b. Dynamic DOF + XPBD** — port the massfull XPBD kernel from `RRsp3.cl:compute_ports_cluster_rigid` (l.659-911):
- [x] Port constraint: `C = |tip_i − x_j| − l0 = 0`, compliance `α = 1/(K·dt²)`.
- [x] XPBD impulse: `λ = −C / (α + invM_i + invM_j + w_ang)`, where `w_ang = |r×n|²·invI`.
- [x] Apply `dpos_i += λ·invM_i·n`, `dpos_j -= λ·invM_j·n`, `dθ_i += (r×(λ·n))·invI`.
- [x] Velocity update from position delta (XPBD standard).
- [x] Collision solver (`solve_collisions`). → **Done in `raff.rs:step_xpbd` (PbdCompliance variant).**
- [x] **True XPBD with lagged `λ_acc`** (Macklin 2016) — `PosSolver::Xpbd` in `solve_xpbd_lagged`. Stiffness-independent convergence. (2026-08-28)
- [x] **Projective Dynamics (Jacobi)** — `PosSolver::Projective` in `solve_projective_jacobi`. Nonlinear local projection + fixed global quadratic step. (2026-08-28)
- [x] **Convergence-to-same-geometry test** — all 3 position-based variants + force-MD reach the same geometry (Kabsch RMSD < 1e-3). `tests/test_raff_convergence.rs`. (2026-08-28)
- [x] **Parameter-sweep benchmark** — `src/bin/raff_bench.rs`: steps/evals/wall-time per solver × params. First results show PBD-compliance fastest (with over-relax), XPBD/Projective more stable, Force-MD 100-1000x slower. (2026-08-28)

**1c. Analytical + Force MD** — implement central-force recoil + memoryless rotation:
- [ ] Eval port spring force along **center–center line** `n = (x_j − x_i)/|...|` (not tip→atom), to conserve `L_trans`. **TODO — current force uses tip→atom, violates L_trans for analytical.**
- [x] Rotation: each step, solve optimal `q` from ARAP/Procrustes (pick one sub-variant first — recommend **Horn eigen** for robustness). → **Done: `solve_rotation_wahba` (Horn eigen).**
- [x] Integrate position with standard MD; rotation is algebraic (no `omega`). → **Done: `OrientMode::Adiabatic` in `step_force_md`.**

**1d. Analytical + XPBD** — the `RRsp3` massless design:
- [ ] XPBD port constraint with **central-line projection**. **TODO.**
- [x] Rotation: algebraic solve (polar / eigen / newton — implement all 3 as selectable sub-variants). → **Horn eigen done. Polar + Newton TODO.**
- [ ] Two-pass for eigen variant: (1) `compute_optimal_rotation_eigen`, (2) linear recoil from precomputed `tips`. **TODO.**

**Verification (per case):**
- [x] **Energy conservation** (force MD cases): track `E_kin + E_pot` over 1000 steps, report drift. → **Test passing.**
- [x] **Momentum conservation** (all cases): `|Σ m_i v_i| < tol` (linear), `|Σ r_i×p_i + Σ I_i ω_i| < tol` (angular, dynamic cases) or `|Σ r_i×p_i| < tol` (analytical cases). → **Translation/rotation invariance tests passing.**
- [ ] **Parity vs `RARFF_SR.h::pairEF`** on a 2-atom / 4-atom rigid motif (CH4-like): match energy + force to 1e-3 relative. **TODO — needs Morse port.**
- [x] **Symmetry:** tetrahedral sp3 center with 4 identical neighbors → force on center = 0, torque = 0 at equilibrium. → **Test passing.**
- [x] **Diagnostic test:** print per-atom residuals, worst contributor, sign of deviation (per AGENTS.md §Testing). → **FD checks implemented.**

### Phase 2 — Non-bonded interactions (Axis 3)

**2a. Compact exponential Morse (3a' — primary, port from SPAMMM):**
- [ ] Port `compact_exp_pair_EF` from SPAMMM `Forces.cl:260-273` to Rust. This is the **production model** from `demo_pairff.py` unified mode.
- [ ] Port `pairff_unified_site_EF` from SPAMMM `Forces.cl:279-309` — the branch-free site-pair primitive that handles atoms + epairs + sigma-holes with the same instructions.
- [ ] Implement the site type system: `type ∈ {0=atom, 1=epair, 2=sigma}`, with `REQ = (R, √E, Q/pseudo-charge, w_blunt)`.
- [ ] Implement branch-free mixing: `gij = gi·gj`, `R0 = gij·(Ri+Rj)`, `α = gij`, `w = wi+wj`, `E0 = mix(attr, ei·ej, gij)`.
- [ ] Add electron-pair / sigma-hole placement (port `add_electron_pairs_via_atomic_system` from `RigidBodyDynamics.py:1488`).
- [ ] Verify: compact-exp energy + force matches Morse at R₀ (exact by construction: `V(R₀)=−E₀`, `V'(R₀)=0`, `V''(R₀)=2E₀β²`).

**2a-ref. Full Morse + Coulomb (3a — physics reference):**
- [ ] Add `Morse` pair potential to `NonBondedFF` (alongside existing LJ). Params: `D_e, a, r0` per atom-type pair, derived from `REQs`.
- [ ] Port from FireCore `Forces.h` / SPAMMM `Forces.cl:getMorsePLQH` (l.235).
- [ ] Verify: Morse energy + force matches analytical at 3 sample distances.
- [ ] Use this as the reference to validate compact-exp convergence (compact-exp → Morse as n→∞).

**2a-legacy. Legacy Lorentzian Hbond (3a'' — comparison only):**
- [ ] Port the 4-loop kernel from SPAMMM `rigid.cl:2198` (`rigid_body_pairff_kernel`) for comparison.
- [ ] Lorentzian: `V = min(0, Q_atom·Q_dummy) · fcut(r/rc) · 1/(w²+r²)`.
- [ ] Keep as a fallback/comparison — the unified compact-exp supersedes it.

**2b. Split-collision (3b — for PBD variants):**
- [ ] Implement **3b-iii** (compact-exp split) first — `V_rep = E₀αy²` (convex, XPBD constraint) + `V_attr = −E₀(1+α)y` (explicit force). Reuses the same compact-exp kernel from 2a.
- [ ] Implement **3b-i** (piecewise quadratic, `getSR_x2_smooth`) — simplest, directly referenced in FireCore `Forces.h:511-539`.
- [ ] Implement **3b-ii** (one-sided hard contact + erf/erfc split) as the physically-correct alternative per `SoftSplineHardAtomCore.chat.md`.
- [ ] Wire the short-range part as an **XPBD constraint** (when dynamics = XPBD) or as a **clamped harmonic force** (when dynamics = ForceMD).
- [ ] Wire the long-range part as an **explicit external force** in both cases.
- [ ] **Consolidate the design docs:** create `notes/designs/split_collision.md` summarizing 3b-i/ii/iii with the GPT review's critique and the final recommendation.

**Verification:**
- [ ] Compact-exp → Morse convergence: energy + force match to 1e-3 at sample distances for n=8.
- [ ] Split potential + explicit residual = full Morse+Coulomb to 1e-3 at sample distances.
- [ ] XPBD with split-collision: stable for 1000 steps on a 10-atom cluster, no penetration.
- [ ] Compare relaxation convergence: split+XPBD vs full-Morse+ForceMD vs compact-exp+ForceMD.

### Phase 3 — GUI: interactive forcefield debugging + mouse dragging

**Goal:** the user can pull a molecule around by mouse and see how each forcefield variant reacts, side-by-side.

**Tasks:**
- [x] 3.1. Add a **forcefield-variant selector** to the editor GUI (dropdown or hotkeys): the 4 CPU cases from Phase 1, plus non-bonded mode (full vs split). → **Done: `F` key cycles Uff/RigidSp3/Raff. RAFF settings panel with orient mode, non-bonded toggle.**
- [x] 3.2. The existing drag-spring (`get_force_spring_ray` in `editor/main.rs:45`) works for ForceMD. For XPBD, add a **position-target constraint**: the dragged atom's position is pinned to the mouse ray intersection (XPBD constraint with very stiff compliance). → **Spring drag works in both `do_relax_step` and `do_raff_step`. XPBD position-target not yet needed (spring force sufficient).**
- [x] 3.3. **Visualize ports** (already `show_ports` flag exists) — draw port tips as small spheres + lines from atom center to tip, colored by constraint violation (green=satisfied, red=stretched). → **Done: ports drawn as orange lines from atom center to tip. In RAFF mode, synced with `state.quat[i]` via `topo.port_tip()`. Color-by-violation not yet implemented.**
- [ ] 3.4. **Energy / momentum HUD** — display `E_pot, E_kin, |P|, |L|` in real time so the user can see conservation drift while dragging. **TODO — E_port + E_nb displayed, but not E_kin/|P|/|L|.**
- [ ] 3.5. **Side-by-side mode** (stretch goal): run two forcefield variants on the same molecule in two viewports, same drag input → direct visual comparison. **TODO.**

**Verification (L2 visual):**
- [x] Drag an atom in a tetrahedral molecule → neighbors follow via port springs, molecule rotates if dragged off-center. → **Verified with `--raff --2d data/xyz/C2H4.xyz`.**
- [ ] Switch from Dynamic+ForceMD to Analytical+XPBD → behavior changes (no rotational inertia, instant alignment). **TODO — XPBD not wired to editor.**
- [ ] Save screenshots/GIFs to `debug/` for review. **TODO.**

### Phase 4 — GPU (OpenCL) implementation (Axis 4)

**Only after Phase 1-2 are physics-correct on CPU.** CPU Rust is the authoritative reference for GPU parity.

**4.1. Port the CPU force-eval to OpenCL kernels** in `opencl/` (new file `RAFF.cl` or extend `Rigid.cl`).
- [ ] Start with **4a (one WG per body)** for rigid-molecule mode.
- [ ] Port **4b (cluster-sorted)** for the port-based rigid-atom mode (the primary target).
- [ ] Use `float4` packed arrays (apos.w = mass/charge, fapos.w = energy), `int4 neighs`, workgroup 32.

**4.2. Wire OpenCL into Rust** via the `ocl` crate (already in workspace deps, not yet used).
- [ ] Create a new crate `crates/libs/molff-ocl` (or feature-gate in `molff`).
- [ ] Upload `apos, quat, neighs, neigh_bs, bon_params` once; upload `apos` each step; download `fapos`.
- [ ] **CPU↔GPU parity test:** same molecule, same params, same dt → `max|F_cpu − F_gpu| < 1e-4` per atom.

**4.3. Implement layout variants 4c, 4d, 4e** and benchmark:
- [ ] Measure `μs/step` for N = 10, 100, 1000, 10000 atoms.
- [ ] Compare to FireCore C++ `MolWorld_sp3::MDloop()` (the benchmark per `DESIGN_GOALS.md` §5).
- [ ] Report in `notes/reports/raff_gpu_bench.md`.

**4.4. Fuse collision flags into existing kernels** (per AGENTS.md §Performance): if a distance kernel already runs, add clash flags in the same loop — never recompute on host.
- [ ] Fuse collision flags into distance kernels.

### NumericalMathPlayground (`/home/prokop/git/NumericalMathPlayground/`)

**Added to source list 2026-08-28.** Contains the theoretical derivations and compact-potential fitting that underpin the non-bonded interaction design.

| Topic | File | What it contains |
|---|---|---|
| **Port-based FF literature review + theory** | `topics/ReactiveFF/RigidAtomicRotatingFrameFF.chat.md` | The core theoretical document: port energy formulation, relation to conventional angle FFs (ARAP/Procrustes equivalence), adiabatic vs extended-Lagrangian vs relaxation rotation, novelty assessment vs VALBOND/patchy-particles/AMOEBA. **The physics justification for the whole RAFF approach.** |
| **Compact pairwise potentials (polynomial + exponential)** | `topics/NonBondingFFs/FastPairwisePotentials.chat.md` | Full derivation of compact polynomial Morse (family 1, r²-based), compact exponential Morse (family 2, recommended), pure-tail analytical solution, f²/f⁴ interpolation, soft-radius for epair blunting, branch-free mixing rules. |
| **Compact potential fitting code** | `topics/NonBondingFFs/fit_radial.py` | Working Python implementation of all compact potential variants, analytical coefficient derivation, mixing rules, comparison plots. |
| **NonBondingFFs overview** | `topics/NonBondingFFs/README.md` | Summary of compact-exp family, GPU branch-free design, PairFF demo, parameter reference. |
| **ReactiveFF OpenCL demos** | `topics/ReactiveFF/reactiveff_ocl_app.py`, `reactiveff_ocl_app3d.py` | OpenCL demo apps for the reactive rigid-atom FF. |
| **Bounding box balancing** | `topics/ReactiveFF/BoundingBoxBalancing.md` | Spatial acceleration design notes. |

**Key equations from NumericalMathPlayground** (centralized in `notes/designs/raff_theory_equations.md`):
- Compact polynomial Morse: `V = C_R z² − C_A z`, `z = (1 − r²/r_c²)^q`, force without sqrt.
- Compact exponential Morse: `V = E₀ y [αy − (1+α)]`, `y = max(0, 1 − β(ρ−R₀)/n)^n`, converges to Morse as n→∞.
- Soft radius: `ρ = √(r²+w²) − w = r²/(√(r²+w²)+w)` — blunts epair origin, same instructions as atom core.
- Branch-free mixing: `g_ij = g_i·g_j`, `R₀ = g_ij(R_i+R_j)`, `α = g_ij`, `w = w_i+w_j`.

### SPAMMM PairFF demo + kernels (additional detail)

| Topic | File | Lines | What it contains |
|---|---|---|---|
| **PairFF rigid-body demo** | `SPAMMM/demos/demo_pairff.py` | full | Interactive Vispy demo: rigid molecules, Morse+Coulomb+Hbond, FIRE relaxation, mouse drag. **The working reference for interactive rigid-body FF.** |
| **PairFF user manual** | `SPAMMM/demos/PairFF_manual.md` | full | Controls, concepts, CLI reference, didactic tour. |
| **PairFF design report** | `SPAMMM/doc/Topics/ForceFields/PairFF.md` | full | Architecture, kernel table, Python API, status. |
| **LFF projective relaxation** | `SPAMMM/doc/Topics/ForceFields/LFF_ProjectiveRelax.md` | full | LFF design: spring classes (K12/K13/K14), Jacobi iteration, fit knobs, parity benchmarks. |
| **compact_exp_pair_EF kernel** | `SPAMMM/kernels/Forces.cl` | 260-273 | The actual GPU implementation of compact exponential Morse (n=8) + soft radius. **Port this to Rust/OpenCL.** |
| **pairff_unified_site_EF** | `SPAMMM/kernels/Forces.cl` | 279-309 | Unified branch-free site-pair primitive (atoms + epairs + sigma holes). |
| **Rigid body kernels** | `SPAMMM/kernels/rigid.cl` | 245, 465, 2452 | Quaternion integration, rigid_body_dynamics_kernel, unified pairff kernels. |
| **Nonbonded kernels** | `SPAMMM/kernels/nonbonded.cl` | 135-277 | `getNonBond_ex2` — pairwise LJ/Coulomb with 2nd-neighbor exclusion, local-memory tiling. |

## 3. Reference source map (where to port what from)

### FireCore (`/home/prokop/git/FireCore/`)

| Topic | File | Lines | What to port |
|---|---|---|---|
| Force-based reactive RAFF (Morse + angular gating) | `cpp/common/molecular/RARFF_SR.h` | `pairEF` 441, `evalTorques` 718, `move` 747 | Parity reference for force-based path. |
| Flexible atom sp-hybridization FF | `cpp/common/molecular/FlexibleAtomReactiveFF.h` | `FlexibleAtomType` 45, `combine` 89 | Onsite/bond split concept. |
| XPBD massfull port kernel | `pyBall/RigidAtomFF/RRsp3/RRsp3.cl` | `compute_ports_cluster_rigid` 659-911 | Phase 1b (Dynamic+XPBD). |
| XPBD massless shapematch (polar) | `RRsp3.cl` | `compute_ports_cluster_rigid_shapematch` 1089 | Phase 1d sub-variant 1b-i. |
| XPBD massless eigen (Davenport q) | `RRsp3.cl` | `compute_optimal_rotation_eigen` 1260, `compute_ports_cluster_rigid_eigen_tips` 1516 | Phase 1d sub-variant 1b-ii. |
| XPBD massless Newton in ω | `RRsp3.cl` | `compute_ports_cluster_rigid_substep_optimized` 916 | Phase 1d sub-variant 1b-iii. |
| XPBD collision (Jacobi sphere-sphere) | `RRsp3.cl` | `compute_collision_cluster_rigid` 562 | Phase 2b collision. |
| Heavy-ball momentum + corrections | `RRsp3.cl` | `apply_corrections_rigid_ports` 1640 | XPBD convergence acceleration. |
| ARAP/Procrustes derivation | `pyBall/RigidAtomFF/shared/Analytic_Procrustes_doc.md` | full | Phase 1c/1d rotation math. |
| XPBD vs PD derivation + conservation | `pyBall/RigidAtomFF/RRsp3/RRsp3_momentum_design.md` | full | Phase 1 design rationale. |
| RRsp3 host dispatch | `pyBall/RigidAtomFF/RRsp3/RRsp3.py` | `step_cluster` 471, `step_dynamics` 638 | GPU orchestration pattern. |
| RRsp3 code map | `pyBall/RigidAtomFF/RRsp3/RRsp3.audit.md` | full | Kernel breakdown. |
| Projective Dynamics (C++) | `cpp/common/math/ProjectiveDynamics_d.h/.cpp/_frag.cpp` | `LinSolveMethod` 124, `run_LinSolve` 198, `updateIterativeJacobi` 505 | Phase 2c (PD). |
| Piecewise quadratic split | `cpp/common/math/Forces.h` | `getSR_x2_smooth` 511-539 | Phase 2b-i. |
| Split-collision design (quadratic) | `doc/DevNotes/ToDo_FastCollision_2.md`, `ToDo_FastCollision_3.md` | full | Phase 2b design. |
| Split-collision critique (hard contact + erf) | `doc/Topics/FastCollisionSplitNonbond/SoftSplineHardAtomCore.chat.md` | full | Phase 2b-ii design. |
| GPU: one WG per body | `cpp/common_resources/cl/Rigid.cl` | `WORKGROUP_SIZE` 191, `rigid_body_dynamics_kernel` 196 | Phase 4a. |
| GPU: cluster-sorted | `pyBall/RigidAtomFF/RRsp3/RRsp3.cl` | `GROUP_SIZE` 97, `build_local_topology_rigid` 454 | Phase 4b. |
| GPU: LFF Jacobi (one WG per mol) | `cpp/common_resources/cl/LFF.cl` | `lff_jacobi` 30, `lff_nb_jacobi` 219 | Phase 4c. |
| GPU: rigid body + GridFF | `cpp/common_resources/cl/Rigid.cl` | `rigid_body_gridff_kernel` 313 | Phase 4 (surface coupling). |
| GPU: erf/erfc split + hybrid grid | `cpp/common_resources/cl/Surface.cl` | `eval_hardcore_grid` 983, `eval_hybrid_grid` 1028 | Phase 2b-ii GPU. |

### SPAMMM (`/home/prokop/git/SPAMMM/`)

| Topic | File | Lines | What to port |
|---|---|---|---|
| Rigid-body 6-DOF engine | `spammm/forcefields/RigidBodyDynamics.py` | `from_molecules` 3158, `run_multimol_md` 2707 | Rigid-molecule (not rigid-atom) reference. |
| **PairFF non-bonded model (production)** | `kernels/Forces.cl` | `compact_exp_pair_EF` 260-273, `pairff_unified_site_EF` 279-309 | **Phase 2a (primary).** Branch-free compact-exp for atoms+epairs+sigma. See `Import_other_Repos.md` §"PairFF non-bonded model". |
| **PairFF unified kernel** | `kernels/rigid.cl` | `rigid_body_pairff_unified_kernel` 2452-2623, `_env_` 2643, `_faf_` 2700, `_allmol_` 2888 | **Phase 2a + Phase 4.** 5 kernel variants, all same compact-exp model. |
| **PairFF legacy kernel** (comparison) | `kernels/rigid.cl` | `rigid_body_pairff_kernel` 2198-2425 | Phase 2a-legacy. 4-loop Morse+Coulomb/Lorentzian, superseded by unified. |
| **Electron-pair / sigma-hole placement** | `spammm/forcefields/RigidBodyDynamics.py` | `add_electron_pairs_via_atomic_system` 1488, `_extend_reqs_with_epairs` 3265 | Phase 2a. Site type system for Hbond directionality. |
| **PairFF demo** (interactive reference) | `demos/demo_pairff.py` | full | **The working reference for interactive rigid-body FF.** Unified mode default, multi-body, FAF. |
| Rigid-body kernels | `kernels/rigid.cl` | `make_qrot` 245, `rigid_body_dynamics_kernel` 465 | GPU quaternion integration. |
| LFF projective Jacobi | `spammm/forcefields/LFFSolver.py` + `kernels/LFF.cl` | `build_linearized_from_uff` 91, `lff_jacobi` 61 | Phase 2c (PD surrogate). |
| LFF design doc | `doc/Topics/ForceFields/LFF_ProjectiveRelax.md` | full | Phase 2c rationale. |
| Morse decomposed (PLQH) | `kernels/Forces.cl` | `getMorsePLQH` 235 | Phase 2a-ref (Morse reference). |
| PME split / soft-core | `spammm/surfaces/PMESplit.py` | `SplitParams` 46, `soft_core_split` 393 | Phase 2b split reference. |
| AtomicGraph topology SSOT | `spammm/topology/AtomicGraph.py` | `to_arrays` 444, `npi` 58 | Already mirrored in `moltopo::Topology`. |
| Rigid pose representation | `spammm/forcefields/RigidEnsemble.py` | `pos + qrot` (xyzw) | Convention reference (SurfMol uses `Quat4d` = xyzw). |

## 4. Resolved decisions (user-confirmed 2026-08-28)

1. **Naming:** Rename `RigidSp3FF` → `RaffFF` and make it configurable (RotationSolver + DynamicsStrategy enums). One struct, all variants. Touches existing code but cleanest long-term.
2. **Capping atoms:** `DESIGN_GOALS.md` §2.1 says capping atoms (H, epairs) have **no ports** and are a **rigid appendix fixed to a port of the host atom** (no independent DOF). The current `rigid_sp3.rs` gives H `nport=0` (point atom, free translation). **TODO:** implement the "rigid appendix" model (H position = host_pos + host_quat·port_local[slot]·l_H) as the primary; keep free-translation H as a fallback. *(Not yet confirmed by user — flagged for first implementation step.)*
3. **Analytical rotation solvers:** implement **all 3** (polar, eigen, Newton-in-ω) together as selectable sub-variants in Phase 1.
4. **Split-collision:** implement **all 3** sub-variants: 3b-i (piecewise quadratic, ~20 lines), 3b-ii (one-sided hard contact + erf/erfc), and 3b-iii (compact-exp split, reuses the same kernel as 2a). **3b-iii is the primary** since it reuses the compact-exp model; 3b-i and 3b-ii are for comparison.
5. **GPU crate:** new separate crate `crates/libs/molff-ocl` with all OpenCL kernels + unsafe. Matches `DESIGN_GOALS.md` §6.

## 5. Suggested execution order

```
Phase 0 (refactor for swappability)  ──┐
                                        ↓
Phase 1 (4 CPU cases)  ─────────────── test each case ──→ Phase 3 (GUI)
                                        ↓
Phase 2 (non-bonded: full Morse, then split) ──→ test with each Phase 1 case
                                        ↓
Phase 4 (GPU: port CPU→OpenCL, layout variants, parity, bench)
```

Phases 0-1 are the immediate priority. Phase 2 can start in parallel with Phase 3 (GUI) once Phase 1 is done. Phase 4 is last.

## 6. Verification summary (TDD — define before coding)

| Check | Method | Tolerance |
|---|---|---|
| Force/energy parity vs FireCore `RARFF_SR.h::pairEF` | 2-atom + 4-atom motif | 1e-3 rel |
| Linear momentum conservation | `|Σ m_i v_i|` over 1000 steps | < 1e-6·|initial| |
| Angular momentum (dynamic) | `|Σ r×p + Σ Iω|` | < 1e-5·|initial| |
| Angular momentum (analytical) | `|Σ r×p|` (central force) | < 1e-5·|initial| |
| Energy conservation (MD) | `|E_tot(t) − E_tot(0)| / E_tot(0)` | < 1e-4 over 1000 steps |
| Tetrahedral equilibrium | force=0, torque=0 at rest | < 1e-10 |
| Morse analytical | `E(r), F(r)` at r = 0.9r0, r0, 1.1r0 | exact to 1e-12 |
| Split = full | `U_split + U_residual = U_morse+coulomb` | 1e-3 |
| CPU↔GPU parity | `max|F_cpu − F_gpu|` per atom | 1e-4 (f32) |
| GPU perf vs FireCore | `μs/step` at N=100,1000 | ≤ FireCore |

All tests print per-atom residuals, worst contributor, and sign of deviation (diagnostic, not pass/fail — per AGENTS.md §Tests Are Diagnostics).
