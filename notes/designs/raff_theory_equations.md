---
type: design-doc
title: "RAFF Theory — Centralized equations for all forcefield variants"
description: All mathematical formulations for the port-based rigid-atom forcefield (RAFF) in one place: port energy, rotation solvers (dynamic DOF, analytical polar/eigen/newton), force-based MD, XPBD, Projective Dynamics, compact non-bonded potentials (polynomial + compact-exp), split-collision for PBD, and quaternion update rules. Consolidated from FireCore, SPAMMM, and NumericalMathPlayground.
tags: [theory, equations, RAFF, XPBD, projective-dynamics, quaternion, compact-morse, split-collision, non-bonded]
timestamp: 2026-08-28
---

# RAFF Theory — Centralized equations for all forcefield variants

This document collects **every equation** needed to implement and compare the port-based rigid-atom forcefield variants. Sources are cited inline. The goal is one place to compare pros/cons.

> **⚠ Corrections applied 2026-08-28** based on ChatGPT review (`notes/chats/REAFF.chat.md`). 12 issues fixed: port energy convention (§1), XPBD constraint bug (§3.2), compliance notation (§3.2), wrong Procrustes variant → Wahba (§2.2), center-force projection unnecessary at adiabatic convergence (§2.3), PD characterization (§3.3), "convex" → "locally projectable" (§5), exact vs approximate split separation (§5), compact-exp split curvature analysis (§5.3), concave quadratic reframed around inflection point (§5.1), erf/erfc reclassified as spatial decomposition (§5.2), and common proximal problem + curvature-based split criterion added (§11). See `REAFF.chat.md` for the full review.

**Notation:**
- Atom *i*: position `x_i ∈ ℝ³`, orientation `q_i ∈ S³` (unit quaternion), rotation matrix `R_i = R(q_i) ∈ SO(3)`.
- Ports: fixed body-frame vectors `a_{iα}` (α = 1..z_i), e.g. tetrahedral for sp3.
- Port tip in world frame: `tip_{iα} = x_i + R_i · (l_{iα} · a_{iα})` where `l_{iα}` is the bond length.
- Neighbor assigned to port α: atom `j = j(i,α)`.
- `d_α = x_j − x_i` (neighbor direction from center).
- `r_α = R_i · (l_{iα} · a_{iα})` (rotated port arm, world frame).
- `e_α = d_α − r_α = x_j − tip_{iα}` (port error vector: neighbor minus tip).

---

## 1. Port energy (the valence model)

The core idea (from `RigidAtomicRotatingFrameFF.chat.md` §1): each port is a stiff spring connecting the rotated port tip to the neighbor atom position.

> **Convention (corrected 2026-08-28):** Use a single clean convention with `k_p` as the **per-port** stiffness. If a physical bond is represented by **two reciprocal directed ports** (one on each atom), use `k_p = K_bond / 2` so that the pair sums to the physical bond stiffness. Do not hide this factor inside the force formula.

$$
\boxed{E_{i\alpha} = \frac{k_p}{2} |e_{i\alpha}|^2, \qquad F_{i\alpha} = k_p \, e_{i\alpha}}
$$

where `e_α = d_α − r_α = x_j − tip_{iα}` is the port error vector (neighbor minus tip).

**Force on atom *i* (from port α):**
$$
F_{i\alpha} = k_p \, e_{i\alpha} = k_p \, (x_j - \text{tip}_{i\alpha})
$$

**Force on neighbor *j* (Newton's 3rd law):**
$$
F_{j \leftarrow i\alpha} = -F_{i\alpha}
$$

**Torque on atom *i* (from port α):**
$$
\tau_{i\alpha} = r_\alpha \times F_{i\alpha}
$$

**Total port energy:**
$$
E_{\text{port}} = \sum_i \sum_{\alpha \in P_i} \frac{k_p}{2} |e_{i\alpha}|^2
$$

> **Note on the existing code convention:** SurfMol `rigid_sp3.rs:169,176` uses `F = 0.5k·e`, `E = 0.25k·|e|²` **per port**, with double-counting (two reciprocal ports per bond) giving `F_total = k·e`, `E_total = k/2·|e|²`. This is numerically equivalent to the clean convention with `k_p = k/2`, but the factor is hidden inside the force formula. The clean convention above makes it explicit.

*Source: SurfMol `rigid_sp3.rs:eval_forces` l.132-181; FireCore `RARFF_SR.h:pairEF` l.441.*

### 1.1 Relation to conventional angle force fields

If the orientation is **adiabatically eliminated** (minimized over R_i for fixed neighbor positions), the port energy becomes an effective many-body angular potential (`RigidAtomicRotatingFrameFF.chat.md` §7):

$$
E_i^{\text{eff}}(\{d_\alpha\}) = \min_{R_i \in SO(3)} E_i(\{d_\alpha\}, R_i)
$$

Near equilibrium, expanding `R_i ≈ R_i^0 · exp([δθ_i]_×)` and `d_α = d_α^0 + δd_α`:

$$
E_i \approx \frac{1}{2} \sum_\alpha k_\alpha \left| \delta d_\alpha - \delta\theta_i \times r_\alpha^0 \right|^2
$$

Eliminating `δθ_i` gives a quadratic energy in all neighbor displacements:

$$
E_i^{\text{eff}} = \frac{1}{2} \delta d^T \left[ K - KB(B^T K B)^{-1} B^T K \right] \delta d
$$

This shows the port model is **not** just z independent springs — the shared rotation creates **correlated angular stiffness** between all ports. It is a low-rank factorization of the angle-angle Hessian. For z neighbors: conventional FF has O(z²) angle terms; port model has O(z) port terms + 1 orientation.

### 1.2 Physical interpretation of the auxiliary rotation

The quaternion `q_i` is **not** a literal nuclear rotation — it represents a local valence-electron/hybridization frame (`RigidAtomicRotatingFrameFF.chat.md` §9). Three regimes:

| Regime | Description | Use case |
|--------|-------------|----------|
| **Adiabatic** | `R_i* = argmin_{R_i} E` each step; no inertia | Pure relaxation, cleanest for classical FF |
| **Extended-Lagrangian** | Small fictitious inertia, cold/damped thermostat | MD with thermal correctness (like Drude polarization) |
| **Structural relaxation** | Fictitious inertia = numerical preconditioning | Fast minimization (no thermodynamic meaning) |

### 1.3 Mass convention — relaxation vs dynamics

**SurfMol convention:** the mass `m_i` in `RaffTopology::mass[]` is a **numerical parameter**, not necessarily the physical atomic mass.

| Mode | Mass | Why |
|------|------|-----|
| **Relaxation** (default in editor) | `m_i = 1.0` for all atoms | Uniform mass gives uniform convergence rate per atom — fastest minimization. Heavy atoms (C=12, O=16) don't lag behind H=1, so the structure settles in fewer steps. No thermodynamic meaning. |
| **Dynamics** (MD with thermal correctness) | `m_i =` physical atomic mass | Needed for correct momentum conservation, thermal equilibrium, phonon spectra. Set via `set_masses_from_elements()` (planned). |

The same logic applies to the rotational inertia `I_i`: for relaxation, `I_i = 0.4 · l²` (no mass factor) gives uniform rotational convergence; for dynamics, `I_i = 0.4 · m_i · l²` is physical.

> **Current SurfMol state:** `mass[]` defaults to 1.0 (relaxation mode). The editor's `build_raff_from_world` does not override it. To switch to dynamics mode, add a `set_masses_from_elements` call and multiply inertia by mass. This is a future task — not needed for relaxation-only use.

### 1.4 Port geometry — per-type vs per-atom (two ARAP variants)

There are **two formulations** of the port geometry `a_{iα}`:

| Variant | Port directions | Source | Status |
|---------|----------------|--------|--------|
| **Per-atom ARAP** (current default) | `a_{iα}` = normalized direction to neighbor α in the **initial/reference configuration** | `RaffTopology::set_port_geometry_from_reference` | **Implemented** — editor default |
| **Per-type** (idealized) | `a_{iα}` from UFF type suffix: sp2→120° trigonal, sp3→tetrahedral, sp1→linear, H_→point | `RaffTopology::set_port_geometry_from_types` | **Implemented** — alternative, not used by editor |

**Per-atom ARAP (current default):** each atom's ports are the actual neighbor directions in the initial configuration, stored per-atom. The Wahba solver then finds the rotation that best preserves the **original** local geometry. This is the classic ARAP (As-Rigid-As-Possible) approach from Sorkine & Alexa 2007 — the reference frame is the initial shape, not an idealized hybridization. At initialization, identity rotation perfectly aligns all ports (E_port = 0).

**Per-type (idealized):** all atoms of the same hybridization get the same idealized port geometry (e.g. all sp2 carbons get 120° trigonal). The Wahba solver finds the rotation that best aligns these idealized ports to the actual neighbor directions. This is the UFF-style approach — the "ideal" geometry is encoded in the port directions.

**Why per-atom is the default:** `build_neighs_from_bonds` assigns ports to neighbors in **bond-list order**, not geometric order. With idealized sp2 ports at 0°/120°/240°, the port-to-neighbor assignment may be geometrically inconsistent (port 0 at 0° assigned to a neighbor at 120°). No rotation can align all 3 ports simultaneously in this case — the Wahba residual is large, producing huge spurious forces. Per-atom ARAP avoids this entirely: ports ARE the initial directions, so identity rotation works regardless of assignment order.

**Trade-offs:**
- Per-atom: preserves the original shape exactly (ARAP), works for any geometry, E_port=0 at init. But can't correct a bad initial geometry toward ideal angles.
- Per-type: forces idealized angles (120°, 109.5°). Could correct a distorted structure, but only works if port-to-neighbor assignment is geometrically consistent (needs port permutation or geometric sorting of neighbors). **Currently broken for most molecules due to bond-list-order assignment.**

**To use per-type:** call `topo.set_port_geometry_from_types(&uff_types)` instead of `set_port_geometry_from_reference(&pos)`. To make per-type work correctly, also need to permute neighbor slots so port 0 goes to the neighbor closest to the port 0 direction (not yet implemented).

**Future task — per-type port reindexing:** `build_neighs_from_bonds` assigns ports in bond-list order. To make `set_port_geometry_from_types` work, add a `reindex_ports_by_direction` method that, after setting idealized port directions, permutes the `neighs`/`neigh_bs` slots so each idealized port direction is paired with the neighbor closest to that direction. This would enable per-type mode (correcting distorted geometry toward ideal 120°/109.5° angles) as an alternative to per-atom ARAP. Not urgent — per-atom ARAP works well for relaxation.

---

## 2. Rotation solvers (Axis 1)

### 2.1 Dynamic DOF — physical rotational inertia

The quaternion is a mechanical DOF with angular velocity `ω_i`, inertia tensor `I_i`, torque `τ_i`.

**Angular velocity update (symplectic Euler):**
$$
\omega_i \leftarrow (1 - \gamma_{\text{rot}}) \cdot \omega_i + I_i^{-1} \cdot \tau_i \cdot dt
$$

where `γ_rot` is rotational damping.

**Quaternion from angular velocity:**
$$
\Delta q = \text{quat\_from\_axis\_angle}(\hat{\omega}, |\omega| \cdot dt) = \left[ \hat{\omega} \sin\frac{|\omega| dt}{2},\; \cos\frac{|\omega| dt}{2} \right]
$$

**Quaternion update:**
$$
q_i \leftarrow \text{normalize}(\Delta q \otimes q_i)
$$

**Inertia estimate** (SurfMol `rigid_sp3.rs:202-218`): from bond lengths,
$$
I_i \approx 0.4 \cdot \bar{l}^2, \quad \bar{l}^2 = \frac{1}{n} \sum_{\alpha} l_\alpha^2
$$

**Conservation:** off-center port forces → torque absorbed by rotational inertia → conserves `L_total = L_trans + L_spin`.

*Source: SurfMol `rigid_sp3.rs:move_atom_md` l.183-236; FireCore `Rigid.cl:make_qrot` l.245; SPAMMM `rigid.cl:qrot_omega` l.255.*

### 2.2 Analytical / memoryless — ARAP / Procrustes local step

The quaternion is **not** a DOF. Each step, compute the optimal rotation aligning ports to neighbors. This is the **ARAP local step** (`Analytic_Procrustes_doc.md`).

> **Corrected 2026-08-28:** The atom center `x_i` is a dynamical variable, not a free parameter. The correct problem is **Wahba** (rotation-only), **not** full Procrustes (which also solves translation `t`). **No centroid subtraction.** Centered Procrustes is only valid if you intentionally solve the entire rigid pose `(x_i, R_i)` analytically — which we do not.

**Goal (Wahba problem):** minimize over rotation only, at fixed `x_i`:
$$
E_i(R) = \sum_\alpha k_\alpha \| d_\alpha - R \, r_\alpha \|^2, \qquad d_\alpha = x_{j_\alpha} - x_i
$$

Since `|d_α|` and `|r_α|` do not depend on `R`, this is equivalent to maximizing `Σ k_α d_α^T R r_α`.

**Step 1 — Cross-covariance matrix (3×3), no centering:**
$$
\boxed{H_i = \sum_\alpha k_\alpha \, d_\alpha \, r_\alpha^T}
$$

**Step 2 — Optimal rotation** is the polar factor (closest orthogonal matrix) of `H`:
$$
R^* = \text{polar}(H)
$$

Three methods to compute `R* = polar(H)`:

#### 2.2a Newton–Schulz polar decomposition

Initialize `R_0 = H`. Iterate (3–5 steps):
$$
\boxed{R_{k+1} = \frac{1}{2} R_k (3I - R_k^T R_k)}
$$

Converges to the orthogonal factor when `H` is nonsingular and close to rotation. Pure matmul/add — GPU friendly. **Risk:** near-singular `H` (degenerate/coplanar geometry) → divergence or reflection; may need det fix.

*Source: `RRsp3.cl:compute_ports_cluster_rigid_shapematch` l.1089; `Analytic_Procrustes_doc.md` §A.*

#### 2.2b Horn quaternion via K-matrix power iteration

Build scalar covariance components `s_{ab}` of `H`, then the symmetric 4×4 **Horn K-matrix**:

$$
\text{tr} = s_{xx} + s_{yy} + s_{zz}
$$

$$
K = \begin{pmatrix} \text{tr} & s_{yz}{-}s_{zy} & s_{zx}{-}s_{xz} & s_{xy}{-}s_{yx} \\ \cdot & s_{xx}{-}s_{yy}{-}s_{zz} & s_{xy}{+}s_{yx} & s_{zx}{+}s_{xz} \\ \cdot & \cdot & s_{yy}{-}s_{xx}{-}s_{zz} & s_{yz}{+}s_{zy} \\ \cdot & \cdot & \cdot & s_{zz}{-}s_{xx}{-}s_{yy} \end{pmatrix}
$$

**Dominant eigenvector of K** = optimal unit quaternion `q*`. Solve via 4 power iterations with normalization, **warm-started** from previous frame's quaternion:

$$
q_{k+1} = \text{normalize}(K \cdot q_k)
$$

**Pros:** inherently det=+1, numerically robust for small neighbor counts, warm-start exploits temporal coherence. **Cons:** sensitive if all weights ~0 or points colinear; eigenvector can flip sign.

*Source: `RRsp3.cl:compute_optimal_rotation_eigen` l.1260; `Analytic_Procrustes_doc.md` §B.*

#### 2.2c Newton–Raphson in ω-space

Local 3×3 Hessian from port lever arms:
$$
H_{\text{rot}} = \sum_\alpha k_\alpha [r_\alpha]_\times^T [r_\alpha]_\times
$$

Solve `ω = H_rot⁻¹ τ` in a few Newton substeps, then update `q ← quat_from_axis_angle(ω̂, |ω|) ⊗ q`.

*Source: `RRsp3.cl:compute_ports_cluster_rigid_substep_optimized` l.916.*

### 2.3 Conservation consequence (critical)

| Solver class | Rotational inertia | Recoil direction | Conserved |
|--------------|-------------------|------------------|-----------|
| **Dynamic DOF** (2.1) | Yes (physical) | Off-center (tip→atom) | `P`, `L_total = L_trans + L_spin` |
| **Analytical** (2.2) | No (geometric) | Off-center (ordinary port forces) | `P`, `L_trans` (if R_i exactly converged) |

> **Corrected 2026-08-28:** The previous version claimed that analytical rotation requires **center-center force projection** to conserve angular momentum. This is **too strong**. If `R_i` is genuinely adiabatically minimized, the envelope theorem gives:
>
> $$\frac{\partial E}{\partial R_i} = 0 \quad \Rightarrow \quad \sum_\alpha r_{i\alpha} \times F_{i\alpha} = 0$$
>
> The ordinary off-center port forces then have zero total orbital torque when summed over the system, by rotational invariance. **No center-line projection is needed at exact convergence.**
>
> The center-force projection is useful as a **hack for approximate convergence** — if the orientation is only approximately solved, a residual torque remains, and projecting onto the center-center line restores conservation despite the error. But this **changes the force field** and is no longer the gradient of the original port energy `E_eff(x) = min_R E(x,R)`.
>
> **Recommended conservation test:** (1) converge each local `R_i*`; (2) calculate ordinary off-center port forces; (3) measure `τ_i = Σ r_α × F_α` (should be ~0 per atom); (4) measure total orbital torque `Σ x_i × F_i` (should be ~0 globally). If this works, it demonstrates the center-line projection is unnecessary and lets you retain the true gradient of `E_eff`.

*Source: `RRsp3_momentum_design.md` §6.3; corrected based on `REAFF.chat.md` point 5.*

---

## 3. Dynamics strategies (Axis 2)

### 3.1 Force-based MD (impulse/MD)

Standard symplectic Euler / velocity Verlet. Forces from §1, torques from §1, rotation from §2.1 or §2.2.

**Velocity update:**
$$
v_i \leftarrow (1 - \gamma) v_i + \frac{F_i}{m_i} dt
$$

**Position update:**
$$
x_i \leftarrow x_i + v_i \, dt
$$

**Force clamping** (SurfMol `rigid_sp3.rs:190-193`):
$$
F_{\text{clamped}} = F \cdot \min\left(1, \frac{F_{\text{lim}}}{|F|}\right)
$$

**Pros:** energy conservation trackable, physical dynamics, standard MD. **Cons:** stiff bonds → small dt, slow relaxation.

### 3.2 XPBD (Extended Position-Based Dynamics)

Constraints `C(x) = 0` with compliance. Explicit, GPU-friendly, handles non-linear constraints (collisions) naturally.

> **Corrected 2026-08-28 (compliance notation):** Physical compliance is `α = 1/K` (inverse stiffness). The timestep-scaled quantity entering XPBD is `α̃ = α/dt² = 1/(K·dt²)`. The previous version called `1/(K·dt²)` simply `α`, which invites implementation errors. Here we use `α̃` for the timestep-scaled compliance.

**Port constraint** (distance from tip to neighbor):
$$
C = |x_j - \text{tip}_i| = 0
$$

> **Corrected 2026-08-28 (constraint bug):** The port tip already contains the bond length: `tip_i = x_i + R_i·(l_0·a_α)`. Therefore the constraint is `C = |x_j - tip_i| = 0`, **not** `|x_j - tip_i| - l_0 = 0`. The latter would try to place the neighbor another `l_0` beyond the port tip. The actual code (`RRsp3.cl:1054`) uses `diff = xj - (xi + r_arm)` with `r_arm = R·(l_0·a)` — correct.

**XPBD position update:**
$$
\boxed{\Delta x = \frac{-C}{\tilde\alpha + \sum_i w_i \|\nabla_i C\|^2} \cdot w_i \nabla_i C}
$$

where `w_i = 1/m_i` (inverse mass), `α̃ = 1/(K·dt²)` (timestep-scaled compliance).

**For the port constraint** (massfull, dynamic DOF — `RRsp3.cl:659-911`):
$$
n = \frac{x_j - \text{tip}_i}{|x_j - \text{tip}_i|}, \qquad r_\text{arm} = R_i \cdot (l_0 \cdot a_\alpha)
$$

$$
w_{\text{ang}} = |r_\text{arm} \times n|^2 \cdot I_i^{-1}
$$

$$
w_{\text{total}} = \frac{1}{m_i} + \frac{1}{m_j} + w_{\text{ang}} + \tilde\alpha
$$

$$
\lambda = \frac{C}{w_{\text{total}}}
$$

**Corrections:**
$$
\Delta x_i = \lambda \cdot \frac{1}{m_i} \cdot n, \qquad \Delta x_j = -\lambda \cdot \frac{1}{m_j} \cdot n
$$

$$
\Delta \theta_i = \lambda \cdot I_i^{-1} \cdot (r_\text{arm} \times n)
$$

**Velocity update** (from position delta):
$$
v_i \leftarrow \frac{x_i^{\text{new}} - x_i^{\text{old}}}{dt}
$$

**Heavy-ball momentum** (solver acceleration, not physical — `RRsp3_momentum_design.md` §3.1):
$$
\Delta x_{\text{total}} = \Delta x_{\text{constraint}} \cdot \text{relaxation} + \Delta x_{\text{prev}} \cdot \beta
$$

Reset `Δx_prev` at the start of each physics time step.

**Pros:** explicit (no global solve), handles collisions, stable for stiff constraints. **Cons:** iterative convergence, compliance tuning.

### 3.3 Projective Dynamics (PD)

> **⚠ Critical architectural note (2026-08-28):** PD is NOT just iterative constraint projection. It has a **two-loop structure**:
> - **Outer loop** (nonlinear, inertial): `x_pred = x + v·dt + F_ext·dt²/m` → solve → `v = (x_new - x)/dt`. This carries real dynamics between steps. The linear sub-steps alone cannot solve nonlinear problems (rotations, dihedral torsions) — the outer loop with real dynamics is necessary for global convergence.
> - **Inner loop** (linear, fast): Jacobi/Gauss-Seidel on the prefactored PD matrix `A = M/dt² + L`. Typically only **1-2 inner iterations** per outer step. The inner loop is a local smoother — much cheaper than force-based because it doesn't evaluate costly O(n²) long-range interactions (Coulomb, PME).
> - **Heavy-ball momentum** in the inner loop: `p_{k+1} = p'_k + bmix·d_k` where `d_k = p_k - p_{k-1}`. FireCore's `SmartMixer` ramps bmix from 0 (first 3 iters) to 0.75 (after). This dramatically accelerates the linear solver convergence.
>
> **SurfMol implementation status:** `step_position_based` has the translational outer predictor/corrector and `solve_projective_jacobi` has configurable inner heavy-ball momentum. RAFF's `cdamp` is a retention multiplier, so undamped inertia requires `cdamp=1`. Adiabatic Projective solves `q*(x)` memorylessly by full Davenport 4×4 Jacobi diagonalization before and after inner sweeps. Dynamic Projective: the outer step only predicts (`x += v·dt`, `q ← exp(ω·dt/2)⊗q` — no torque evaluation); the inner Jacobi loop accumulates BOTH translational RHS and torque in ONE port traversal, then updates both `x` and `q` together (`δθ = τ/(I/dt² + K_rot)`, `K_rot = Σ_s k_s|r_arm|²`). The corrector computes both `v = Δx/dt` and `ω = 2·imag(Δq)/dt`. Generalized-power reset uses `v·F+ω·τ`. PBD/XPBD's direct quaternion constraint corrections remain rotational PBD rather than torque dynamics with angular memory. The tree100/D2 comparison: dynamic PD with inner-coupled rotation at dt=0.1/i8+HB beats adiabatic in T2 steps (30 vs 32) at 4× lower wall time (5.7ms vs 23ms). Stability limit extended from dt=.05–.1 to dt=.1–.15. **Multirate rotational subcycling (Session 9) does NOT work** — port constraints couple rotational and translational displacement, so independent rotational substeps break the coupled ratio. See labbook Sessions 4, 8, 9, and 10. Reference: FireCore `ProjectiveDynamics_d::run_LinSolve` (line 686-801) + `updateIterativeMomentum` (line 461-503).

Energy minimization formulation (`ProjectiveDynamics_d.h`; `RRsp3_momentum_design.md` §2.2):

$$
E(x) = \frac{1}{2 \, dt^2} (x - y)^T M (x - y) + \sum_c W_c(x)
$$

where `y = x^n + v^n·dt + F_ext·dt²/m` is the **inertial prediction** (NOT just `x^n`). This is the key — the `M/dt²` anchor term provides implicit-Euler inertia.

**Linearized solve:**
$$
\boxed{\left(\frac{M}{dt^2} + L\right) \Delta x = \text{forces}}
$$

where `L` is the Laplacian of the constraint stiffness matrix. Solved via:
- **Cholesky** (exact, expensive)
- **Jacobi** (diagonal, cheap, iterative)
- **Jacobi + heavy-ball momentum** (accelerated: `p_{k+1} = p'_k + bmix·d_k`, bmix 0→0.75)

**Outer loop (run_LinSolve):**
$$
y_i = x_i^n + v_i^n \cdot dt + F_i^{ext} \cdot \frac{dt^2}{m_i} \quad \text{(predict)}
$$
$$
x^{n+1} = \text{solve}(A, b(y)) \quad \text{(inner linear solve, 1-2 iters)}
$$
$$
v^{n+1} = \frac{x^{n+1} - x^n}{dt} \quad \text{(corrector — always, not multiplied by cdamp)}
$$

> **Corrected 2026-08-28 (PD characterization):** The previous version characterized PD as essentially "linear constraints only." This is too narrow. PD supports **nonlinear geometric constraints** through nonlinear **local projections** — what is fixed is the global step's quadratic structure, not the constraints. The original Bouaziz et al. 2014 paper explicitly describes nonlinear constraint manifolds and local projection followed by a global quadratic compromise. Dynamic collisions are awkward mainly because the active contact set changes, destroying the pre-factored global system — not because PD fundamentally requires linear constraints.

**LFF (Linearized Force-Field)** — SPAMMM's working PD surrogate (`LFF_ProjectiveRelax.md`):

Springs replace bonds/angles/dihedrals:
| Class | Graph | `l_0` | `K` |
|-------|-------|-------|-----|
| K₁₂ | 1-2 (bonds) | UFF `r_0` | UFF bond `k` |
| K₁₃ | 1-3 (angle ends) | law of cosines | Fourier `k` × ~8, clipped |
| K₁₄ | 1-4 (dihedral ends) | current `|a-d|` | `clip(40V, 5, 80)` |

**Jacobi iteration:**
$$
b_i = \frac{M_i}{dt^2} p_i + \sum_j K_{ij} p_{ij}^{\text{rest}}, \qquad A_{ii} = \frac{M_i}{dt^2} + \sum_j K_{ij}
$$

$$
\boxed{p_i \leftarrow \frac{b_i}{A_{ii}}}
$$

**Pros:** best for stiff linear spring networks, large dt, fast relaxation. **Cons:** linearization loses non-linear physics; not energy-parity with UFF/SPFF; needs capped K for dihedrals.

### 3.4 XPBD vs PD — when to use which

> **Corrected 2026-08-28:** PD and XPBD need not define a different force field. They are alternative **algorithms** applied to the same port model. The key distinction is the solver strategy, not the physics. See §11 for the unifying proximal problem formulation.

| Criterion | XPBD | Projective Dynamics |
|-----------|------|-------------------|
| Constraint type | Non-linear (collisions, contacts) | Nonlinear local projections + fixed global quadratic step |
| Solve | Explicit per-constraint (Gauss-Seidel) | Global/diagonal linear (pre-factorable) |
| GPU | Natural (per-constraint parallel) | One WG per molecule (Jacobi) |
| Collisions | Native | Awkward (active set changes destroy pre-factor) |
| Stiffness | Compliance `α̃ = 1/(K dt²)` | `M/dt² + L` |
| Best for | Rigid-atom ports + collisions | Fixed-topology bonded solver, LFF spring surrogate |

**Decision (from `RRsp3_momentum_design.md`):** XPBD for the rigid-atom port model (non-linear constraints, collisions). PD/LFF for fast relaxation surrogate of linearized UFF. Both solve the same proximal problem (§11) with different algorithms.

---

## 4. Non-bonded potentials (Axis 3)

### 4.1 Full Morse + Coulomb (reference)

**Morse:**
$$
V_{\text{Morse}}(r) = E_0 \left[ e^{-2\beta(r - R_0)} - 2 e^{-\beta(r - R_0)} \right]
$$

**Coulomb:**
$$
V_{\text{Coulomb}}(r) = \frac{k_e Q_i Q_j}{r}
$$

**Mixing rules:**
$$
R_{0,ij} = R_i + R_j, \qquad E_{0,ij} = e_i \cdot e_j, \qquad e_i = \sqrt{E_{ii}}
$$

**Force:**
$$
F = -\frac{dV}{dr} = 2 E_0 \beta \left[ e^{-2\beta(r-R_0)} - e^{-\beta(r-R_0)} \right] \cdot \hat{r}
$$

*Source: FireCore `NBFF.h`; SPAMMM `Forces.cl:getMorsePLQH` l.235.*

### 4.2 Compact polynomial Morse (family 1)

From `FastPairwisePotentials.chat.md` §1-2. Uses `r²` only (no sqrt), compact support, repeated squaring.

**Overlap variable:**
$$
z(r) = \left(1 - \frac{r^2}{r_c^2}\right)^q, \qquad z = 0 \text{ for } r \geq r_c
$$

**Energy:**
$$
\boxed{V(r) = C_R \, z^2 - C_A \, z}
$$

**Force (no sqrt needed):**
$$
\boxed{\mathbf{F} = 2q \, r_c^{-2} \left(1 - \frac{r^2}{r_c^2}\right)^{q-1} (2 C_R z - C_A) \cdot \mathbf{r}}
$$

**Parametrization from R₀, E₀:**
$$
z_0 = z(R_0) = \left(1 - \frac{R_0^2}{r_c^2}\right)^q
$$

$$
C_R = \frac{E_0}{z_0^2}, \qquad C_A = \frac{2 E_0}{z_0}
$$

Then `V(R₀) = −E₀`, `V'(R₀) = 0` exactly.

**Effective Morse slope at minimum:**
$$
a_{\text{eff}} = \frac{2q R_0}{r_c^2 - R_0^2} = \frac{2q}{R_0(\lambda^2 - 1)}, \qquad \lambda = r_c / R_0
$$

**Harmonic curvature:**
$$
k = V''(R_0) = 2 E_0 a_{\text{eff}}^2 = \frac{8 E_0 q^2}{R_0^2 (\lambda^2 - 1)^2}
$$

**Choose q and λ to match Morse:** `λ = √(1 + 2q/(a R₀))`.

**Recommended:** q=4 for soft/bonded, q=8 for excluded-volume atom pairs. q=2^m for repeated squaring.

**Pros:** no sqrt, no exp, compact support, C³ smooth at cutoff (q=4). **Cons:** converges to Gaussian-like radial dependence, not exponential — poor Morse tail reproduction even at high q.

### 4.3 Compact exponential Morse (family 2 — recommended)

From `FastPairwisePotentials.chat.md` (line 1392+); implemented in SPAMMM `Forces.cl:compact_exp_pair_EF` l.260.

**Compact exponential overlap:**
$$
u(r) = \max\left(0, \; 1 - \frac{\beta}{n}(r - R_0)\right), \qquad y(r) = u(r)^n
$$

As `n → ∞`, `y → exp(−β(r − R₀))`, so this converges directly to Morse.

**Cutoff:** `r_c = R₀ + n/β`.

**Unified energy (atoms + epairs, same instructions):**
$$
\boxed{V(r) = E_0 \, y \, [\alpha \, y - (1 + \alpha)]}
$$

where:
- `α = 1, w = 0`: compact Morse (atom-atom), `V = E₀(y² − 2y)`
- `α = 0, w > 0`: purely attractive blob (atom-epair), `V = −E₀ y`
- `V(y=1) = −E₀` for every α (exact depth)
- `V'(R₀) = 0` for every α, n (exact equilibrium)
- `V''(R₀) = 2 E₀ β²` for every n (exact Morse curvature)

**Force (branch-free, one sqrt for soft radius):**
$$
\mathbf{F} = E_0 \, \beta \, [2\alpha \, y - (1+\alpha)] \cdot u^{n-1} \cdot \frac{\mathbf{r}}{\sqrt{r^2 + w^2}}
$$

**Soft radius** (blunts epair origin without branch):
$$
\boxed{\rho(r, w) = \sqrt{r^2 + w^2} - w = \frac{r^2}{\sqrt{r^2 + w^2} + w}}
$$

- `w = 0`: `ρ = r` (sharp atom core)
- `w > 0`: `ρ ≈ r²/(2w)` near origin (smooth parabolic center)

Replace `r` by `ρ` in the compact exponential. Physical cutoff: `r_c² = ρ_c (ρ_c + 2w)` where `ρ_c = R₀ + n/β`.

**Branch-free mixing rules** (precomputed in type-pair table):
$$
g_{ij} = g_i \cdot g_j \quad (\text{core flag: 1=atom, 0=epair})
$$

$$
R_0 = g_{ij} (R_i + R_j) \quad (\text{epair-atom: } R_0 = 0)
$$

$$
E_0 = e_i \cdot e_j \quad (\text{geometric energy mixing})
$$

$$
\alpha = g_{ij} \quad (\text{atom: } \alpha=1, \text{epair: } \alpha=0)
$$

$$
w = w_i + w_j \quad (\text{atom: } w=0, \text{epair: } w>0)
$$

**GPU implementation** (SPAMMM `Forces.cl:260-273`, n=8):
```c
float r2  = dot(dr, dr);
float rw  = sqrt(r2 + w*w);
float rho = r2 / fmax(rw + w, eps);
float u   = fmax(0.0f, 1.0f - (beta*0.125f)*(rho - R0)); // /8
float u2  = u*u;  float u4 = u2*u2;
float y   = u4*u4;          // u^8
float u7  = u4*u2*u;        // u^7
float E   = E0 * y * (alpha*y - (1.0f + alpha));
float f_over_r = E0 * beta * (2.0f*alpha*y - (1.0f+alpha)) * u7 / fmax(rw, eps);
// F_vec = f_over_r * dr;
```

**Recommended n=8:** three squarings for u^8, truncation error ~2e⁻⁸ vs Morse.

**Pros:** converges to Morse (not Gaussian), branch-free (same instructions for atoms and epairs), exact R₀/E₀/curvature, one sqrt. **Cons:** still has a hard cutoff at `r_c = R₀ + n/β` (C⁰, not C¹ — but the energy is already very small there).

**Site types and their parameters** (from `demo_pairff.py` / `RigidBodyDynamics.py:3265`):

| Type | Name | R | E | Q | w | α | Role |
|------|------|---|---|---|---|---|------|
| 0 | atom | RvdW | √(EvdW) | charge | 0 | 1 | Morse + Coulomb |
| 1 | epair (lone pair) | 0 | 0 | He (<0) | w>0 | 0 | Hbond acceptor (attracts H+) |
| 2 | sigma-hole | 0 | 0 | Hs (>0) | w>0 | 0 | Hbond donor (attracts O−) |

Electron pairs are placed by `AtomicSystem.add_electron_pairs()` at `epair_dist` (default 1.4 Å) from host O/N atoms along the lone-pair direction. Sigma holes are placed on H atoms bonded to O/N at `sigma_dist` (default 1.0 Å) along the O-H bond direction. Both are fixed in the body frame at construction time.

**SPAMMM kernel variants** (all use the same compact-exp model, different tiling/buffer strategies):

| Kernel | rigid.cl lines | Use case |
|--------|---------------|----------|
| `rigid_body_pairff_unified_kernel` | 2452-2623 | 1 active body + 1 static partner |
| `rigid_body_pairff_unified_env_kernel` | 2643+ | 1 active body + many env molecules (tiled) |
| `rigid_body_pairff_unified_faf_kernel` | 2700+ | + FAF substrate (fused PairFF+FAF) |
| `rigid_body_pairff_unified_env_faf_kernel` | 2734+ | + env + FAF |
| `rigid_body_pairff_unified_allmol[_faf]_kernel` | 2888+ | Multi-body shared buffers (preferred for multi-mol) |

### 4.3b Legacy PairFF model (4-loop Morse+Coulomb / Lorentzian — superseded)

**This is the older model in `demo_pairff.py`, kept for comparison.** Uses 4 separate loops with `if (atom_idx < n_dyn_atoms)` branching → warp divergence. From `rigid.cl:2198` (`rigid_body_pairff_kernel`).

**Atom-atom** (type=0 ↔ type=0): Morse + damped Coulomb (same as §4.1).

**Atom-epair / atom-sigma** (type=0 ↔ type=1 or type=2): Lorentzian Hbond:
$$
V_{\text{Hbond}}(r) = \min(0, Q_{\text{atom}} \cdot Q_{\text{dummy}}) \cdot f_{\text{cut}}(r/r_c) \cdot \frac{1}{w^2 + r^2}
$$

where `f_cut = smoothstep(1 − r/r_c) = 3x² − 2x³` (C¹ cutoff), `Q_dummy = He` (epair, negative) or `Hs` (sigma-hole, positive). The `min(0, ...)` clips to attractive-only.

**Force:**
$$
\mathbf{F} = -\frac{dV}{dr} \hat{r} = -\left[ \text{coeff} \cdot (f'_{\text{cut}} \cdot L + f_{\text{cut}} \cdot L') \right] \hat{r}
$$

where `L = 1/(w²+r²)`, `L' = −2r·L²`, `f'_cut = −6x(1−x)/r_c`.

**Design:** epairs/sigma-holes are pseudo-atoms with R=0, E=0. They participate ONLY in Hbond/sigma interactions, not Morse/Coulomb. Pseudo-charge stored in REQ.z. Epair-epair and sigma-sigma interactions are skipped.

**Why superseded:** the 4-loop design with type branching causes warp divergence on GPU. The unified compact-exp kernel (§4.3) uses the **same instructions** for all site types — only parameters differ (via the `g_ij` core flag). The Lorentzian also requires a separate cutoff function, while compact-exp has built-in compact support.

### 4.4 Pure-tail polynomial Morse (analytical, no fitting)

From `fit_radial.py` + `FastPairwisePotentials.chat.md` (line 944+). Sets `c=0` and solves for pair-specific cutoff.

$$
V_n(r) = a_n (r - r_{\text{node}}) \cdot f(r)^n
$$

where `f(r) = (1 - r²/r_c²)²` (fc22 cutoff), `r_node = R₀ − ln(2)/β` (exact Morse zero crossing).

**Pure-tail solution** (c=0, no repulsive bump):
$$
\boxed{r_c^2 = R_0^2 + 4n \, \Delta \, R_0, \qquad \Delta = \frac{\ln 2}{\beta}}
$$

$$
a_n = -\frac{E_0}{\Delta \cdot f(R_0)^n}
$$

**Interpolation between f² and f⁴** (tunable tail length):
$$
V_\lambda(r) = (r - r_{\text{node}}) \left[ c_2 f^2 + a_4 f^4 \right]
$$

$$
c_2 = -\frac{E_0}{\Delta} \frac{1-\lambda}{f_0^2}, \qquad a_4 = -\frac{E_0}{\Delta} \frac{\lambda}{f_0^4}
$$

$$
r_c^2 = R_0^2 + 8(1+\lambda) \Delta R_0
$$

`λ=0` → pure f² (longer tail, stiffer minimum); `λ=1` → pure f⁴ (shorter tail, softer minimum). Both coefficients non-positive → purely attractive tail, no bump.

---

## 5. Split-collision for position-based dynamics (Axis 3, PBD-specific)

> **Corrected 2026-08-28 (multiple issues):**
>
> **"Convex" → "locally projectable" (point 8):** The previous version called `½k[R−r]₊²` "always convex." It is convex as a scalar function of `r` inside the active interval, but **not globally convex as a function of the Cartesian relative vector `d`**: its tangential Hessian eigenvalue is `U'(r)/r < 0` during penetration. XPBD mainly needs a **well-defined local constraint/projection**, not true Cartesian convexity. "Convex" is replaced by "locally projectable" throughout this section.
>
> **Exact vs approximate splits (point 9):** "Split potential" means two different things:
> - **Exact algebraic decomposition** (3b-iii): `U_h + U_s = U_ref` to floating-point accuracy. Used for integrator study.
> - **Approximate replacement** (3b-i, 3b-ii): `U_h + U_s ≈ U_ref` but not exact. Used for production.
>
> These need separate validation criteria.
>
> **Curvature is the right criterion (point 10):** The reason for splitting is not "repulsion is hard, attraction is soft" — it is: **put the large Hessian eigenvalues into the implicit inner problem and leave an explicit residual with the smallest possible curvature.** The outer-step stability is controlled by `L_s ~ max_r |U_s''(r)|`. See §11 for the curvature-optimized split criterion.

### 5.0 Morse inflection point — the natural split boundary

> **Corrected 2026-08-28 (point 11):** The previous version called the concave quadratic tail "unphysical (attractive where Morse is repulsive)." This is misleading — **Morse IS attractive for `r > R_0`**, and that attraction is exactly what you want. The more useful observation is that Morse changes curvature at:

$$
R_{\text{inf}} = R_0 + \frac{\ln 2}{\beta}
$$

- `r < R_inf`: Morse is **convex** (positive curvature) — the stiff inner basin
- `r > R_inf`: Morse is **concave** (negative curvature) — the soft dissociative tail

This suggests a **physically motivated split boundary**: let the implicit solver handle the high-curvature inner basin approximately up to `R_inf`, and let the explicit outer force handle the soft concave tail. This is a better framing than "repulsion vs attraction."

### 5.1 Piecewise quadratic split (3b-i) — approximate replacement

From FireCore `Forces.h:getSR_x2_smooth` l.511-539.

$$
U_1(r) = \frac{1}{2} k_1 (r - R_{\min})^2 + E_{\min}, \qquad r < R_{\text{cut}}
$$

$$
U_2(r) = \frac{1}{2} k_2 (r - R_{\text{cut2}})^2, \qquad R_{\text{cut}} \leq r < R_{\text{cut2}}
$$

where:
$$
d_1 = R_{\text{cut}} - R_{\min}, \qquad d_2 = R_{\text{cut2}} - R_{\text{cut}}
$$

$$
k_1 = \frac{-2 E_{\min}}{d_1 (d_1 + d_2)} \quad (\text{locally projectable, positive}), \qquad k_2 = -k_1 \frac{d_1}{d_2} \quad (\text{concave, negative})
$$

- `U₁` (r < R_cut): **locally projectable** → XPBD constraint (implicit solve)
- `U₂` (R_cut ≤ r < R_cut2): **concave** → explicit external force
- C¹ continuous at R_cut; `U(R_cut2) = 0`, `U'(R_cut2) = 0`.

> **Reframed (point 11):** Rather than calling `U₂` "unphysical," describe it as: **a deliberately simplified convex-inner / concave-outer approximation of a Morse well.** This is almost tailor-made for an implicit/explicit integrator. The inner region captures the stiff basin; the outer concave tail approximates the dissociative tail. A good initial choice for `R_cut` is `R_inf = R_0 + ln(2)/β` (the Morse inflection point).

**Pros:** simple (~20 lines), C¹ smooth, physically motivated split at inflection. **Cons:** approximate (not exact Morse); `U₂` is a quadratic approximation of the exponential tail, not exact.

### 5.2 One-sided hard contact + erf/erfc Coulomb split (3b-ii)

From `SoftSplineHardAtomCore.chat.md`; `pyBall/OCL/Surface_utils.py:2700-2830`.

> **Reclassified 2026-08-28 (point 12):** The erf/erfc Coulomb decomposition is primarily a **short-range/long-range spatial decomposition** for grid methods (PME-like). It solves the "how to efficiently evaluate long-range Coulomb" problem, **not** the "how to split for PBD stability" problem. For the CPU physics prototype, leave Coulomb entirely in the outer soft force — hard-core repulsion already prevents the `r→0` singularity. Introduce erf/erfc only when studying grid acceleration (Phase 4).

**One-sided hard contact** (XPBD constraint — locally projectable):
$$
\boxed{U_{\text{hard}}(r) = \frac{1}{2} k_h [R_h - r]_+^2}
$$

where `[x]_+ = max(0, x)`. This is a **one-sided** contact — zero for `r > R_h`, quadratic for `r < R_h`. Locally projectable, perfect for PBD.

> **Note on convexity (point 8):** This is convex as a scalar function of `r` inside the active interval, but **not globally convex in Cartesian coordinates** — the tangential Hessian eigenvalue is `U'(r)/r < 0` during penetration. This is not fatal; XPBD needs a well-defined local projection, not global convexity.

**Coulomb split** (spatial short-range/long-range, for grid acceleration — **not needed for CPU prototype**):
$$
\frac{1}{r} = \frac{\text{erfc}(r/\sigma)}{r} + \frac{\text{erf}(r/\sigma)}{r}
$$

- `V_core = Σ_j [Morse(r_j) + k_e Q_j erfc(r_j/σ)/r_j]` — short-range, per-atom, explicit
- `V_smooth = Σ_j k_e Q_j erf(r_j/σ)/r_j` — long-range, on B-spline grid

**Pros:** physically correct contact (one-sided), clean spatial split for GPU grid. **Cons:** erf/erfc + grid is unnecessary complexity for CPU prototype; the PBD stiffness problem is solved by `U_hard`, not by the Coulomb split.

### 5.3 Split using compact-exp potential (3b-iii) — exact algebraic decomposition

The compact-exp family (§4.3) can be split exactly for PBD. The repulsive part `V_rep = E₀ α y²` becomes the XPBD constraint; the attractive part `V_attr = −E₀(1+α) y` becomes the explicit force.

$$
V = \underbrace{E_0 \alpha \, y^2}_{\text{XPBD constraint}} - \underbrace{E_0 (1+\alpha) \, y}_{\text{explicit force}}
$$

For `α = 1` (atom-atom): `V_rep = E₀ y²`, `V_attr = −2 E₀ y`. This is an **exact decomposition** — `V_rep + V_attr = V` to floating-point accuracy. Uses the **same** compact-exp kernel for both parts.

> **⚠ Curvature analysis (corrected 2026-08-28, point 10):** This split is **algebraically exact but numerically poor as a stiffness split.** Verified with SymPy for n=8, α=1:
>
> | Component | `V''(R₀)` | Fraction of total |
> |-----------|-----------|-------------------|
> | Full `V` | `2 E₀ β²` | 100% |
> | Implicit `V_rep` | `15/4 E₀ β² = 3.75 E₀ β²` | 187.5% |
> | Explicit `V_attr` | `−7/4 E₀ β² = −1.75 E₀ β²` | **87.5%** |
>
> The explicit attractive part retains **87.5% of the total curvature** at the equilibrium distance. This means the explicit part is almost as stiff as the original potential, so it barely helps the maximum timestep. The split is useful as an **exact decomposition for integrator testing** (verify `U_h + U_s = U_ref`), but **not recommended as a production stiffness split.**
>
> **Better approach:** Fit a simple quadratic/projective surrogate `U_h(r)` specifically to reduce `max_r |U_s''(r)|` where `U_s = U_compact − U_h`. See §11 for the curvature-optimized split criterion.

---

## 6. Quaternion update rules (summary)

| Method | Formula | When to use | Source |
|--------|---------|-------------|--------|
| **Symplectic Euler** (dynamic) | `ω ← (1−γ)ω + I⁻¹τ dt; q ← normalize(dq(ω,dt) ⊗ q)` | Force-based MD with physical inertia | `rigid_sp3.rs:183` |
| **Taylor quaternion** (dynamic, fast) | `dq ≈ [ω·dt/2, 1]` (1st order Taylor of exp) | GPU, small dt | `Rigid.cl:make_qrot_taylor` l.237 |
| **Exact axis-angle** (dynamic) | `dq = [ω̂ sin(θ/2), cos(θ/2)], θ=|ω|dt` | CPU reference, large dt | `Rigid.cl:make_qrot` l.245 |
| **Newton–Schulz polar** (analytical) | `R ← ½R(3I−RᵀR)`, 3-5 iters; `q ← mat3_to_quat(R)` | Memoryless, GPU-friendly | `RRsp3.cl:1089` |
| **Horn K-matrix** (analytical) | `q ← normalize(K·q)`, 4 power iters, warm-started | Memoryless, robust, warm-start | `RRsp3.cl:1260` |
| **Newton in ω** (analytical) | `ω ← H_rot⁻¹ τ`, substeps; `q ← dq(ω) ⊗ q` | Memoryless, fast convergence | `RRsp3.cl:916` |
| **XPBD rotation** (constraint) | `Δθ ← λ I⁻¹ (r×n); q ← normalize(dq(Δθ) ⊗ q)` | XPBD with physical inertia | `RRsp3.cl:659-911` |

**Quaternion convention (SurfMol):** `Quat4d = (x, y, z, w)`, identity = `(0, 0, 0, 1)`. Same as SPAMMM `RigidEnsemble.py`.

**Quaternion multiplication:**
$$
q_a \otimes q_b = (w_a w_b - \mathbf{v}_a \cdot \mathbf{v}_b, \; w_a \mathbf{v}_b + w_b \mathbf{v}_a + \mathbf{v}_a \times \mathbf{v}_b)
$$

**Rotation by quaternion:**
$$
R(q) \, \mathbf{v} = q \otimes (0, \mathbf{v}) \otimes q^*
$$

---

## 7. GPU kernel layout variants (Axis 4)

| Layout | Workgroup | Pattern | Best for | Source |
|--------|-----------|---------|----------|--------|
| **One WG per body** | WG=32, 4 atoms/thread | Force/torque reduced in local mem | Rigid molecules (≤128 atoms) | `Rigid.cl:191` |
| **Cluster-sorted** | WG=64, nodes first + ghosts | `bkSlots`/`revSlot` gather, no atomics | Port-based rigid atoms | `RRsp3.cl:97` |
| **One WG per molecule (LFF)** | WG=64 | Diagonal Jacobi, barriers | Linearized spring relaxation | `LFF.cl:20` |
| **Per-bond / per-atom** | 1 thread per bond/atom | Global reduction | Large systems, UFF | `UFF.cl` |
| **Small-system single WG** | Entire system in 1 WG | No global memory round-trip | Small molecules (≤ WG size) | Design goal |

---

## 8. Variant comparison matrix

| Variant | Rotation (Axis 1) | Dynamics (Axis 2) | Non-bonded (Axis 3) | Conserved | Stability | Speed | Complexity |
|---------|-------------------|-------------------|---------------------|-----------|-----------|-------|------------|
| **A: Dynamic+ForceMD** | Dynamic DOF | Force MD | Full Morse+Coulomb | P, L_total | dt-limited | Baseline | Low |
| **B: Dynamic+XPBD** | Dynamic DOF | XPBD | Split (hard contact) | P, L_total | Stable | Fast | Medium |
| **C: Analytical+ForceMD** | Polar/Eigen/Newton | Force MD | Full Morse+Coulomb | P, L_trans | dt-limited | Fast (no ω integ) | Medium |
| **D: Analytical+XPBD** | Polar/Eigen/Newton | XPBD | Split (hard contact) | P, L_trans | Stable | Fastest | Medium-High |
| **E: LFF (PD surrogate)** | N/A (no ports) | Projective Jacobi | N/A or FAF | P | Very stable | Fastest relax | Low |

**Key tradeoffs:**
- **Dynamic vs Analytical:** Dynamic gives true rotational inertia (L_total conserved) but needs small dt for stiff rotations. Analytical is memoryless (faster, no ω integration). At exact adiabatic convergence, ordinary off-center port forces conserve angular momentum without center-line projection (§2.3).
- **ForceMD vs XPBD:** ForceMD tracks energy conservation, physical dynamics. XPBD is stable for stiff constraints, handles collisions natively, but needs iterative convergence. Both solve the same proximal problem (§11) with different algorithms.
- **Full vs Split non-bonded:** Full Morse+Coulomb is the physics reference. Split enables PBD stability (locally projectable constraint) but the quality is measured by **residual curvature** `max|U_s''|`, not energy fitting (§11.3).

---

## 9. Source inventory (where each equation comes from)

| Topic | Primary source | Secondary source |
|-------|---------------|-----------------|
| **Review & corrections (12 fixes)** | **`notes/chats/REAFF.chat.md`** | **Applied 2026-08-28 to all sections** |
| Port energy / ARAP theory | `NumericalMathPlayground/topics/ReactiveFF/RigidAtomicRotatingFrameFF.chat.md` | SurfMol `rigid_sp3.rs` |
| Dynamic rotation (symplectic Euler) | SurfMol `rigid_sp3.rs:183-236` | FireCore `Rigid.cl:make_qrot` |
| Newton–Schulz polar | `Analytic_Procrustes_doc.md` §A | `RRsp3.cl:1089` |
| Horn K-matrix eigen | `Analytic_Procrustes_doc.md` §B | `RRsp3.cl:1260` |
| Newton in ω-space | `RRsp3.cl:916` | — |
| XPBD port constraint | `RRsp3.cl:659-911` | `RRsp3_momentum_design.md` §2.2 |
| XPBD heavy-ball | `RRsp3_momentum_design.md` §3.1 | `RRsp3.cl:1640` |
| Projective Dynamics | FireCore `ProjectiveDynamics_d.h` | `RRsp3_momentum_design.md` §2.2 |
| LFF (linearized PD) | SPAMMM `LFF_ProjectiveRelax.md` | SPAMMM `LFF.cl:61` |
| Compact polynomial Morse | `FastPairwisePotentials.chat.md` §1-2 | `fit_radial.py` |
| Compact exponential Morse | `FastPairwisePotentials.chat.md` (l.1392+) | SPAMMM `Forces.cl:260` |
| Pure-tail polynomial | `FastPairwisePotentials.chat.md` (l.944+) | `fit_radial.py` |
| Piecewise quadratic split | FireCore `Forces.h:511-539` | `ToDo_FastCollision_2.md` |
| Hard contact + erf/erfc | `SoftSplineHardAtomCore.chat.md` | `Surface_utils.py:2700` |
| Conservation (massfull vs massless) | `RRsp3_momentum_design.md` §6 | — |
| PairFF rigid-body demo | SPAMMM `demos/PairFF_manual.md` | `demo_pairff.py` |
| PairFF non-bonded model (legacy+unified) | SPAMMM `rigid.cl:2198` (legacy), `rigid.cl:2452` (unified) | `Forces.cl:260` (`compact_exp_pair_EF`), `Forces.cl:279` (`pairff_unified_site_EF`) |
| GPU kernel layouts | FireCore `Rigid.cl`, `RRsp3.cl` | SPAMMM `LFF.cl` |

---

## 10. Chronology — how the ideas developed over time

Reconstructed from git logs. See `Import_other_Repos.md` §2b for the full table. Key milestones:

1. **2026-07-15-16 (NMP):** Port-based FF theory — `RigidAtomicRotatingFrameFF.chat.md`. ARAP equivalence, rotation regimes, novelty assessment.
2. **2026-07-22 (NMP):** Compact non-bonded potential design — `FastPairwisePotentials.chat.md`. Polynomial family → pure-tail → **compact exponential** (the breakthrough: converges to Morse, branch-free for atoms+epairs). `fit_radial.py` fitting code. First `demo_pairff.py` with Vispy.
3. **2026-07-23 (NMP→SPAMM):** Unified kernel `rigid_body_pairff_unified_kernel` born. `Forces.cl:compact_exp_pair_EF` (n=8, soft radius) implemented. SPAMMM `demo_pairff.py` shows it working.
4. **2026-07-24 - Aug 10 (SPAMM):** Production hardening — FAF substrate fusion, multi-body allmol shared buffers, GUI integration, refactoring. 5 kernel variants all using the same compact-exp model.

**What superseded what:**
- Compact polynomial Morse → **superseded by** compact exponential Morse (better Morse tail).
- Legacy 4-loop kernel (Morse+Coulomb / Lorentzian, with branching) → **superseded by** unified compact-exp kernel (single branch-free loop). Legacy kept for comparison.
- `NMP/demo_pairff.py` (legacy default, single 2-molecule) → **superseded by** `SPAMMM/demos/demo_pairff.py` (unified default, multi-body, FAF).

---

## 11. The common proximal problem (unifying framework)

> **Added 2026-08-28 based on `REAFF.chat.md` review.** This is the conceptual heart of the project: XPBD, PD, VBD, analytical rotations, and the nonbonded split are all different answers to the **same question**.

### 11.1 The central equation

For the position-based branch, compute the expensive soft interaction once at the beginning of a macrostep `H`:

$$
F_s^n = -\nabla U_s(x^n)
$$

Define the inertial/external-force target:

$$
\boxed{y = x^n + H v^n + H^2 M^{-1} F_s^n}
$$

Then solve approximately:

$$
\boxed{x^{n+1} = \arg\min_x \left[ \frac{1}{2H^2}(x-y)^T M (x-y) + U_h(x, R) \right]}
$$

For dynamic orientations, there is an analogous rotational inertial term `δθ^T I δθ / (2H²)`. For adiabatic orientations, `R_i = R_i*(x)` is simply minimized as an internal variable.

**Key insight:** the inertial stiffness is `M/H²`. The long-range `O(N²)` force enters **only through `y`**. The `O(N)` ports and contacts are solved repeatedly inside the proximal problem. This is clean IMEX: the soft energy is linearized once (`U_s(x) ≈ U_s(x^n) − F_s^n·(x−x^n)`), while the hard energy is treated implicitly.

### 11.2 All solvers solve the same problem

| Solver | How it solves the proximal problem |
|--------|-------------------------------------|
| **XPBD** | Per-constraint Gauss-Seidel projections |
| **PD-Jacobi** | Nonlinear local projections + fixed global quadratic step |
| **VBD** | Block-coordinate Gauss-Seidel: each atom minimizes locally |
| **Analytical rotation** | `R_i*` minimized as internal variable (adiabatic elimination) |

This avoids the conceptual mistake of treating PD and XPBD as different force fields. They are alternative **algorithms** for the same port model.

### 11.3 Curvature-optimized split criterion

The reason for splitting the nonbonded potential is not "repulsion vs attraction" — it is:

> **Put the large Hessian eigenvalues into the implicit inner problem and leave an explicit residual with the smallest possible curvature.**

The outer-step stability is controlled by the Lipschitz constant of the explicit force:

$$
L_s \sim \max_r |U_s''(r)|
$$

Therefore choose the hard surrogate by:

$$
\boxed{\theta^* = \arg\min_\theta \max_{r \in \Omega} |U_{\text{ref}}''(r) - U_h''(r; \theta)|}
$$

Or use a weighted `L²` curvature error. This is much more directly connected to **maximum permissible outer timestep** than fitting energy values.

### 11.4 Two research tracks

| Track | Goal | Validation |
|-------|------|------------|
| **Exact numerical split** | `U_s = U_ref − U_h` exactly | `U_h + U_s = U_ref` to floating-point |
| **Fast approximate production** | `U_fast = U_inner^implicit + U_tail^explicit` | Reproduce `R_0, E_0, U''(R_0), tail range` |

These are currently mixed in the roadmap; they should be separate experiments.

### 11.5 Iterations vs substeps experiment

Compare at equal hard-work budget:

$$
1 \times H, \; 16 \text{ hard iterations} \quad \text{vs} \quad 16 \times h, \; 1 \text{ hard iteration}, \quad h = H/16
$$

The "Small Steps" result (Macklin) suggests many small substeps may win, but RAFF's cost asymmetry (`O(N²)` soft ≫ `O(N)` hard) might flip the answer — both do the **same number of expensive soft evaluations**.

**Metrics:** `H_max` (max stable macrostep), `N_soft` (soft evaluations to reach minimum), `t_wall` (wall time to tolerance). Not "constraint error after one frame."

### 11.6 The central research question

> **How much of the stiff Hessian can we remove from the explicit outer dynamics using only cheap atom-local implicit solves?**

Then XPBD, PD, VBD, analytical rotations, and the short-range split are all different answers to this one question. This formulation makes the eventual publication much stronger: the port representation, the auxiliary rotations, the atom-local proximal solver, and the curvature-designed nonbonded split become parts of one coherent story.

### 11.7 Essential diagnostics

| Test | What it catches |
|------|-----------------|
| Finite-difference `E(x+ε) − E(x)` vs `F` | Signs/factors |
| Finite-difference quaternion rotation vs `τ` | Torque convention |
| Global translation invariance | `Σ F = 0` |
| Global rotation invariance | `Σ x×F + τ = 0` |
| Reciprocal-port stiffness test | Hidden factor-of-two |
| Exact adiabatic rotation torque residual | Analytical solver correctness |
| Hessian eigenvalues around sp3/sp2 equilibrium | Actual stiffness spectrum |
| Two-atom collision approach | Signs/contact stability |
| `H`-vs-`N_inner` stability map | Core research metric |
| Relaxation vs number of **soft evaluations** | Actual performance objective |
| **Plot `U, U', U''` for every nonbonded split** | Immediately exposes bad stiffness splits |

> **The last diagnostic is especially important.** Looking only at `U(r)` can make two potentials look similar while their timestep behavior differs dramatically. For this project, **`U''(r)` may be the most important plot.**

### 11.8 Distortion types for relaxation benchmarks

The initial geometry perturbation determines **which mode of the Hessian** the
optimizer must dissipate. Different distortions probe different failure modes.
A solver that relaxes random noise quickly may stall on a low-frequency stretch
— the benchmark must cover both.

#### Three distortion classes

**D1 — Random displacement (high-frequency white noise).**
Each atom gets an independent random displacement ~0.1-0.3 Å. This excites the
**high-frequency end** of the vibrational spectrum (bond stretches, angle
bends). Most optimizers relax this well: the stiff modes have large gradients
and the Newton/XPBD local solve captures them directly. **Easy case** — if a
solver fails here, it is fundamentally broken.

**D2 — Uniaxial stretch along the long axis (low-frequency, pathological).**
Scale the molecule along its principal axis (PCA longest component) by 1.2-1.5×.
This excites the **lowest-frequency acoustic mode** — a collective, nearly
zero-curvature deformation. The optimizer must propagate information across the
entire molecule to relax it. This is the **long-narrow-valley pathology**
(continuous analogue of the Rosenbrock function): the gradient is small along
the valley floor and the Hessian is ill-conditioned (λ_max/λ_min ≫ 1).
Traditional local solvers (force-MD, Gauss-Seidel XPBD) converge slowly here
because each step only fixes local bonds, and the global compression must
diffuse atom-by-atom. **This is the pathological case that motivates
multi-grid / coarse-to-fine strategies.** Projective Dynamics' global solve
should handle this better than Gauss-Seidel XPBD — the benchmark must verify.

**D3 — Soft degrees of freedom (near-zero curvature).**
Two sub-cases:

  - **D3a — Dihedral rotation (e.g. H2O2).** The O-O bond has a torsional
    barrier but the curvature at the minimum is very low. The port model alone
    (bond constraints) does NOT constrain the dihedral — it is a genuine free
    DOF unless an explicit torsion term is added. H2O2 is the minimal test case:
    4 atoms, one dihedral, well-studied equilibrium (~111° HOOH torsion).
    **Without an explicit dihedral term, all solvers reach E≈0 at any dihedral**
    — the benchmark documents this null space, same as chain4.

  - **D3b — Non-covalent assembly (e.g. benzoic acid dimer).** Two monomers
    held by hydrogen bonds. Requires either explicit electron-pair sites
    (sigma-hole + lone pair) or at minimum atom-atom electrostatics + LJ.
    The RAFF port model only covers covalent bonds; non-bonded interactions
    (`eval_nonbonded`) must be enabled. **Not yet benchmarkable** — H-bond
    directionality needs the electron-pair site system (roadmap Phase 2c).
    The `benzoicacid_dimer.xyz` file (30 atoms, 2 H-bonds) is staged for this.

#### Two convergence targets

**T1 — Accurate convergence.** Residual geometry RMSD < 0.0001 Å, or
max|F| < 0.001 meV/Å (~1.6e-5 eV/Å). Required for final geometry output,
frequency analysis, and any downstream QM calculation. All solvers must reach
this eventually — the question is **how many soft evaluations** it costs.

**T2 — Rough geometry, fast.** Residual RMSD < 0.05 Å, or max|F| < 0.1 eV/Å.
This is the target for **interactive pre-optimization** (user drags an atom,
the structure snaps to a reasonable shape in <100 ms) and for **global
relaxation** inside a larger optimization loop (basin hopping, simulated
annealing) where many candidate geometries are evaluated and only the promising
ones are refined to T1. **Currently the higher-priority target.** The benchmark
must report steps-to-T2 separately from steps-to-T1, because a solver that
reaches T2 in 10 steps but T1 in 10000 may be the right choice for interactive
use even if another solver reaches T1 in 500.

#### What the benchmark must report

Per solver × distortion × molecule:
  - **Convergence curve**: residual RMSD (log-scale) vs macrostep / soft-eval count
  - **Force curve**: max|F| (log-scale) vs macrostep / soft-eval count
  - **Steps-to-T2** (rough) and **steps-to-T1** (accurate) — both, separately
  - **`N_soft`** (soft evaluations) — the cross-solver performance objective
  - **`t_wall`** — single-thread wall time (CPU reference for GPU port)
