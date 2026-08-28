---
type: rust-crate
title: molff
description: Intra-molecular forcefields — UFF (bonds/angles/dihedrals/inversions), RAFF (port-based rigid-atom FF with XPBD/force-MD/analytical rotation), legacy rigid sp3, non-bonded LJ+Coulomb+H-bond with exclusion lists and PBC.
tags: [rust, crate, forcefield, uff, raff, rigid-atom, port-based, molecular-dynamics, xpbd, nonbonded]
timestamp: 2026-09-28
---

# molff

Intra-molecular forcefield library: four engines that share the same data-oriented design (64-byte aligned arrays, parameter packing, cached intermediates) but have no interdependency. `surfmol` orchestrates them together; each can be used standalone with raw arrays.

## Modules

- **`uff.rs`** — Universal Force Field implementation. `Uff` owns bonded topology (atom indices, parameters) and force buffers. Key algorithm: **force piece assembly** — forces are computed per-term (angles, dihedrals, inversions) into separate buffers (`fang`, `fdih`, `finv`), then assembled per-atom via a `Buckets` mapping structure (`a2f`). Bonds are NOT mapped to buckets — evaluated directly in the per-atom loop. `hneigh` caches normalized bond vectors + inverse distance (`Quat4d.xyz = direction, .w = 1/|r|`) to avoid repeated normalization in angle/dihedral/inversion kernels. Angle/dihedral/inversion forces use `Vec2d` as complex numbers for efficient cos(nθ)/sin(nθ) via repeated complex multiplication — no trig calls in the inner loop. Force layout: `[dih0..dih3][inv0..inv3][ang0..ang2][bon0..]` with precomputed offsets (`i0dih`, `i0inv`, `i0ang`, `i0bon`). `eval_forces` returns `(Eb, Ea, Ed, Ei)` energy components.

- **`raff.rs`** — **RAFF (Rigid-Atom Force Field)**: the multi-variant port-based rigid-atom engine. Replaces explicit angle/dihedral terms with a single port-spring energy: each atom has 1–4 body-frame "ports" rotated by its quaternion, and a harmonic spring between the rotated port tip and the neighbor position drives both translation and rotation. Supports 4 design axes: (1) rotation solver — `OrientMode::Dynamic` (omega+tau+inertia integration) or `OrientMode::Adiabatic` (memoryless Wahba/Horn quaternion eigen solve via `solve_rotation_wahba`); (2) dynamics — `step_force_md` (symplectic Euler), `step_xpbd` (XPBD port constraints + `solve_collisions` Jacobi sphere-sphere), `step_proximal` (projective Jacobi stub); (3) non-bonded — `eval_nonbonded` (LJ+Coulomb, O(N²), 1-2/1-3 exclusion); (4) GPU — not yet started. Key structs: `RaffTopology` (port geometry, neighbors, bond params, mass, inertia), `RaffState` (pos, vel, quat, omega), `RaffConfig` (dt, damping, orient mode), `NbConfig` (non-bonded params). Per-port stiffness `k_p = K_bond / 2` (each bond counted twice). Port tip: `tip = x_i + quat_rotate(q_i, port_local · l0)`. 22 tests in `tests/test_raff.rs` covering port forces, rotation convergence, energy/momentum conservation, XPBD constraints, collisions, adiabatic torque residual. See [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) for the full cross-implementation map and [`/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md`](/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md) for the phased plan.

- **`nonbonded.rs`** — `NonBondedFF`: Lennard-Jones 12-6 + Coulomb + hydrogen bond. Parameters packed as REQH `[R_vdw, sqrt(E_vdw), Q, Hb]` — stores `sqrt(E)` not `E` directly for the geometric-mean combination rule. `make_plqs` precomputes Pauli/London coefficients: `c_l = exp(-α·R)·sqrt(E)`, `c_p = exp(-α·R)·c_l` (exponential damping). Exclusion list: sorted `i32` array of 1-2 and 1-3 neighbor indices (max 16 per atom), traversed in O(n) per atom to skip bonded pairs. PBC support: precomputes all periodic image shift vectors once, iterates over images in `eval_pbc`. Force clamping (`fmax=10.0` eV/Å, enabled by default) prevents numerical explosion at close range. `COULOMB_CONST = 14.3996` eV·Å/e². H-bond term active only when `Hb < 0` (negative flags donor/acceptor).

- **`rigid_sp3.rs`** — `RigidSp3FF`: **legacy** single-variant rigid body dynamics for sp³ atoms. Superseded by `raff.rs` which supports all variants. Kept as the `Dynamic+ForceMD` baseline reference. Each atom has 4 local "ports" (bond attachment points) rotated by its quaternion. `set_sp3` sets tetrahedral ports (±1/√3), `set_sp2` trigonal (120° in xy), `set_sp1` linear (±x), `set_point` none. `eval_forces` computes harmonic spring between quaternion-rotated port tip and bonded atom position; accumulates torque `τ = r × F`. Energy factor 0.25 (each bond counted twice). `move_atom_md` integrates both translation (velocity Verlet) and rotation: angular velocity update `ω_new = ω·(1-rot_damp) + τ·I⁻¹·dt`, quaternion update `q_new = normalize(q_ωdt · q_old)`. Moment of inertia approximated as `I = 0.4·⟨l²⟩` (sphere approximation from average bond length).

## Design decisions

- **Force piece assembly pattern** — term-level forces computed independently, then assembled per-atom via bucket mapping. Enables parallel evaluation of different term types and clean separation of concerns.
- **Cached bond vectors** (`hneigh`) — normalized direction + inverse distance stored per neighbor slot. Angle/dihedral/inversion kernels read from cache instead of recomputing.
- **Complex number angle powers** — `Vec2d` as `(cos θ, sin θ)`, powers via repeated `mul_cmplx`. Avoids `cos(nθ)` trig calls in inner loops.
- **REQH stores sqrt(E)** — geometric mean combination rule `sqrt(Ei·Ej)` becomes `sqrt(Ei)·sqrt(Ej)`, a simple multiply.
- **Sorted exclusion list** — pre-sorted i32 indices enable O(n) skip traversal without hash lookups.
- **64-byte alignment** throughout — SIMD-compatible, cache-line-aligned.
- **RAFF port-spring replaces angle/dihedral** — a single harmonic spring per port replaces the 3-term (angle+dihedral+inversion) UFF machinery. The port tip position encodes both bond length and angle geometry via the quaternion rotation. This is naturally suited to rigid-body dynamics and GPU parallelization (one spring per port, no angle/dihedral traversal).
- **Per-port stiffness halving** — `k_p = K_bond / 2` because each bond contributes two ports (one per atom). Total stiffness of the bond = `k_p_i + k_p_j = K_bond`.
- **Adiabatic rotation = memoryless** — `OrientMode::Adiabatic` re-solves `q_i` every step via Wahba/Horn eigen decomposition of the port covariance matrix. No `omega`, no `tau` integration. More stable than dynamic but does not conserve angular momentum (no rotational inertia).

## What does NOT belong here

- Surface interactions → `surfff`
- MD orchestration (combining multiple forcefields) → `surfmol`
- Topology building and parameter files → `moltopo`
- Rendering → `molrender`
- GPU kernels → `molff-ocl` (planned, not yet created)

## See also

- `DESIGN.md` — forcefield data ownership model ("borrow, don't own")
- `surfmol` — `MolWorld` orchestrator that combines Uff + NonBondedFF + RaffFF + SurfaceFolded
- [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) — RAFF cross-implementation map (SurfMol vs FireCore vs SPAMMM)
- [`/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md`](/notes/tasks/2026-08-28_roadmap_port_based_rigid_ff.md) — RAFF phased implementation plan with checkboxes
- [`/notes/designs/raff_theory_equations.md`](/notes/designs/raff_theory_equations.md) — all RAFF mathematical formulations
- `FireCore` — C++ reference implementation for parity testing
- `SPAMMM` — Python/OpenCL production non-bonded reference (compact-exp Morse)
