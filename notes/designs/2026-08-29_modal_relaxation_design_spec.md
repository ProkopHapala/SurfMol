---
type: design-spec
title: Coarse-grained modal relaxation — design specification (v2)
description: Two complementary approaches for coarse-grained molecular relaxation. (A) Fitted modal: fit stiffness K once, Newton steps (exact for quadratic, 53× speedup on pentacene). (B) Force-projection Galerkin V-shape: project atomic forces onto modes without fitting K, large-timestep modal damped MD alternating with fine smoothing (robust for nonlinear systems). Both exploit timestep scaling — freezing hard modes allows 100-1000× larger dt. Pentacene is the ideal test case because its low-energy subspace is approximately quadratic bending along soft eigenmodes.
tags: [multigrid, modal, coarse-graining, timestep-scaling, relaxation, pentacene, design-spec, amortized-fitting, galerkin, v-shape, force-projection, newton]
timestamp: 2026-08-29
---

# Coarse-Grained Modal Relaxation — Design Specification (v2)

## 1. Big picture: why coarse-graining and where the speedup comes from

### 1.1 The intended application

The target application is **global optimization of molecules on surfaces**: thousands of molecules, each performing millions of MD steps / relaxation steps, exploring conformational space on a surface potential. The coarse-grained model is **built once per molecule type** (e.g., pentacene) and **reused across all instances and all timesteps**. Any setup cost (mode generation, stiffness fitting) is amortized over the entire simulation campaign.

**Setup cost is NOT counted in the per-simulation benchmark.** The benchmark measures only the simulation phase: coarse steps (with syncs) + fine finishing steps.

### 1.2 Where the speedup comes from: TIMESTEP SCALING

The primary speedup mechanism is **timestep scaling**, not fewer iterations:

- **Full-atom dynamics:** the highest frequency is set by the stiffest mode (bond stretch, H wiggle). For UFF: f_max ≈ sqrt(k_bond / m) ≈ sqrt(200/12) ≈ 4.1. Stable dt ≈ 0.02. Each step evaluates the full forcefield (O(N) for N atoms).

- **Modal coarse dynamics:** only soft modes (bend, twist, stretch) are evolved. Hard modes are FROZEN. The highest frequency is set by the softest retained mode. For pentacene modal K: f_max ≈ sqrt(K_twist) ≈ sqrt(0.20) ≈ 0.45. Stable dt ≈ **22** — **1000× larger** than full-atom dt.

- **Newton step** (for fitted modal): for a quadratic model, one Newton step reaches the exact equilibrium — equivalent to infinite timestep. Even with a trust region for large distortions, 3-5 Newton steps suffice.

- **The coarse phase converges soft DOFs in a few large-timestep steps.** Then the full-atom phase only needs to relax hard DOFs (bond stretches, H atoms), which converge fast because they're stiff and local.

**This is the key insight that was missing in the first benchmark (v1).** Running the modal model at dt=0.05 (the full-atom timestep) completely defeats the purpose. The modal timestep must be scaled to the modal frequencies, not the full-atom frequencies.

### 1.3 Why pentacene is the ideal test case

Pentacene is chosen deliberately because its low-energy subspace is **approximately quadratic bending along soft eigenmodes**:

- Aromatic rings are rigid → the soft DOFs are global bend/twist, not local torsions
- sp2 carbons have no free rotation → no highly nonlinear torsional DOFs
- The inversion + dihedral terms provide smooth quadratic bending stiffness
- The molecule is planar at equilibrium → out-of-plane bend/twist are the softest modes

Aliphatic chains (hexadecane) have free rotations around single bonds — these are highly nonlinear torsional DOFs that a quadratic modal model cannot capture. Pentacene avoids this complication, making it the **minimal test system** for validating the modal approach. For nonlinear systems, see Approach B below.

## 2. Two complementary approaches

There are two ways to do coarse-grained modal relaxation. They share the same mode basis Φ and the same timestep scaling principle, but differ in how the coarse dynamics is computed:

### Approach A: Fitted modal (coarse-first, then fine)

**Fit stiffness K once, then use Newton steps (exact for quadratic model).**

```
SETUP (one-time, amortized):
  1. Relax molecule fully → reference geometry x_ref
  2. Build modes Φ from x_ref (bend, twist, stretch, ...)
  3. Fit K from 2×n_modes force evals (central differences at x_ref)
  4. Factor K (Cholesky) — reusable

SIMULATION (per instance):
  COARSE PHASE:
    repeat:
      sync: g = ΦᵀF(x)           — 1 full-force eval
      Newton: dq = K⁻¹·g          — cheap (n_modes × n_modes solve)
      reconstruct: x = x_ref + Φ·(q + dq)
      trust region: scale dq if |dq| > trust_radius
    until |g| < threshold (soft DOFs converged)
  FINE PHASE:
    FIRE on full atom until |F| < threshold (hard DOFs only)
```

**Pros:**
- Newton is exact for quadratic → 1-5 steps for nearly-quadratic systems (pentacene: 4 Newton + 5 syncs = 5 full-force evals, 53× speedup)
- Between syncs, can extrapolate using cached K: g(q) ≈ g_sync - K·(q - q_sync) → fewer syncs needed
- Setup cost is truly negligible for large campaigns (4 force evals, done once)

**Cons:**
- K is fitted at x_ref → inaccurate far from reference (large distortions, conformational changes)
- Quadratic model breaks down for nonlinear DOFs (torsions, cis-trans isomerization)
- Modes are fixed → may not span the relevant subspace if the molecule changes shape significantly
- Requires periodic refitting for strongly nonlinear systems (adds setup cost)

**When to use:** approximately quadratic systems (aromatic molecules, small distortions, rigid frameworks). The intended application: thousands of pentacene molecules on a surface, each relaxing from slightly different distorted geometries.

### Approach B: Force-projection Galerkin V-shape (no fitting)

**Don't fit K. Project atomic forces onto modes at each sync. Use large-timestep damped MD in modal space. Alternate coarse and fine phases (V-shape).**

```
SETUP (one-time, amortized):
  1. Relax molecule fully → reference geometry x_ref
  2. Build modes Φ from x_ref (bend, twist, stretch, ...)
  — NO stiffness fitting needed —

SIMULATION (per instance):
  V-CYCLE (repeat until converged):
    COARSE PHASE (soft DOFs):
      sync: g = ΦᵀF(x)           — 1 full-force eval (TRUE force, no model)
      modal damped MD: q ← q + dt_modal · v, v ← damp·v + dt_modal·g
        dt_modal ≈ 10/f_max_modal (large, because only soft modes)
        — NO K needed: g is the true projected force, re-evaluated each sync —
      reconstruct: x = x_ref + Φ·q
      run n_coarse steps between syncs (using cached g, or sync every step)
    FINE PHASE (hard DOFs):
      few FIRE steps on full atom (relax bond stretches, H atoms)
      — only hard modes, converges in 1-5 steps —
    check: if |F| < threshold → done
```

**Key insight: the timestep is still large even without K.** The timestep stability limit is set by the highest frequency in the evolved subspace. Since we only evolve soft modes (bend, twist), dt_modal ≈ 22 regardless of whether we know K or not. The force projection g = ΦᵀF(x) gives the true modal force — no quadratic assumption needed.

**What "Galerkin" means here:** In a Galerkin method, you project the full equation onto a reduced basis. Here, the full equation is F(x) = 0 (force balance). Projecting onto modes: ΦᵀF(x) = 0 (modal force balance). The coarse solve finds q such that ΦᵀF(x_ref + Φ·q) ≈ 0. This is exactly what the force projection + modal damped MD does — it's a Galerkin restriction of the force balance equation.

**The V-shape:** Unlike Approach A (coarse-first, then fine), the V-shape alternates between coarse and fine phases. This handles coupling between soft and hard modes:
1. Coarse step: move soft DOFs (may excite hard DOFs slightly)
2. Fine step: relax hard DOFs (may slightly change soft DOF forces)
3. Repeat until both are converged

This is important for nonlinear systems where coarse and fine modes are coupled — you can't just do all coarse then all fine.

**Pros:**
- No quadratic assumption → works for nonlinear systems (torsions, large distortions, conformational changes)
- True force at each sync → always accurate, no model breakdown
- No fitting cost → simpler setup
- V-shape handles soft-hard coupling → more robust
- Adapts naturally: if the molecule changes shape, the projected force reflects the new landscape

**Cons:**
- No K → can't do Newton (must use damped MD or FIRE in modal space)
- Damped MD converges slower than Newton (needs ~2-5 periods vs 1 Newton step)
- Without K, can't extrapolate between syncs → may need more syncs
- Can estimate K online from sync history (secant/BFGS): K ≈ -Δg/Δq → hybrid approach

**When to use:** nonlinear systems (aliphatic chains with torsions, large distortions, conformational changes, molecules on surfaces with significant restructuring). Also useful as a fallback when Approach A's quadratic model breaks down.

### 2.3 Hybrid: online K estimation

Approach B can be enhanced by estimating K online from the sync history:

```
After 2+ syncs with different q values:
  g1 = ΦᵀF(x1), q1 known
  g2 = ΦᵀF(x2), q2 known
  K_secant ≈ -(g2 - g1) / (q2 - q1)   (secant approximation)

After K_secant is available:
  Switch from damped MD to Newton steps
  Or use K_secant for extrapolation between syncs
```

This is a BFGS-like update in modal space. After 2-3 syncs, the secant K is often accurate enough for Newton steps. This combines the robustness of Approach B (no initial fitting, true forces) with the speed of Approach A (Newton convergence).

### 2.4 Comparison table

| Aspect | Approach A (fitted) | Approach B (force-projection) |
|--------|--------------------|-----------------------------|
| Setup cost | 2×n_modes force evals | 0 (just modes) |
| Coarse step | Newton (exact for quadratic) | Damped MD with large dt |
| Syncs needed | 1-5 (Newton converges fast) | 3-10 (damped MD slower) |
| Extrapolation between syncs | Yes (cached K) | No (or online K estimate) |
| Quadratic assumption | Yes | No |
| Nonlinear systems | Breaks down | Robust |
| V-shape (alternating) | No (coarse-first) | Yes (alternating) |
| Pentacene speedup | 53× (verified) | ~30-50× (estimated) |
| Best for | Aromatic, rigid, small distortions | Flexible, torsional, large distortions |

## 3. Modal basis design

### 3.1 Modes for pentacene (6 DOF recommended)

| Mode | Formula | Physical meaning | Frequency |
|------|---------|-----------------|-----------|
| Out-of-plane bend | `n · sin(πs)` | global bowing | very soft |
| Axial twist | `u × r_perp · (2s-1)` | torsion around long axis | soft |
| Longitudinal stretch | `u · cos(πs)` | compression/extension | medium |
| In-plane transverse bend | `v · sin(πs)` | in-plane bowing | medium |
| Rigid translation x | `u` | center-of-mass shift | free (zero) |
| Rigid translation y | `v` | center-of-mass shift | free (zero) |

Rigid rotations are removed by the orthonormalization (Gram-Schmidt). Rigid translations have zero stiffness and should be excluded from the modal solve (or handled separately as free DOFs).

### 3.2 Why fewer modes can be faster

The coarse phase converges faster with fewer modes because:
- The modal system is smaller (n_modes × n_modes solve)
- The highest retained frequency is lower (only the softest modes)
- The timestep is larger (dt ∝ 1/f_max_retained)
- For Approach A: Newton step is cheaper (smaller K⁻¹)
- For Approach B: damped MD converges in fewer periods

Adding a stiff mode (e.g., longitudinal stretch) increases f_max → decreases dt → more steps. **The optimal mode set is the softest modes only.** Bend + twist (2 modes) may be optimal for pentacene if stretch is much stiffer.

### 3.3 Mode generation

Modes are generated analytically from the molecular axes (PCA long axis + plane normal), then:
1. Remove net translation (subtract mean)
2. Gram-Schmidt orthogonalize (bend first, then twist against bend)
3. Normalize to unit norm

This is already implemented in `build_bend_twist_modes`.

### 3.4 Mode validity far from reference

Modes are built from the planar reference geometry. For large distortions or conformational changes, the modes may no longer span the relevant subspace. Options:
- **Accept it:** for small-moderate distortions, the modes are still approximately valid (bend/twist shapes don't change much)
- **Rebuild modes periodically:** after each V-cycle, rebuild modes from the current geometry (costs ~nothing — just PCA + analytic formulas)
- **Use curvilinear modes:** for large conformational changes, use modes that follow the molecular shape (e.g., arc-based bend instead of sin(πs))

## 4. Modal dynamics: why large timestep is essential

### 4.1 Frequency analysis

For pentacene with real UFF:
- Bond stretch: k ≈ 200 eV/Å², m ≈ 12 amu → f ≈ 4.1 → dt_max ≈ 0.02
- Angle bend: k ≈ 50 eV/rad² → f ≈ 2.0 → dt_max ≈ 0.05
- Inversion (out-of-plane): k ≈ 5 eV/Å² → f ≈ 0.6 → dt_max ≈ 1.7
- Modal bend: K_bend ≈ 0.058 eV/Å² → f ≈ 0.24 → **dt_max ≈ 42**
- Modal twist: K_twist ≈ 0.20 eV/Å² → f ≈ 0.45 → **dt_max ≈ 22**

The modal timestep is **1000× larger** than the full-atom timestep. This is the speedup.

**This applies to BOTH approaches.** In Approach B, even without knowing K, the timestep stability limit is set by the highest frequency in the evolved subspace. Since we only evolve soft modes, dt_modal ≈ 22 regardless. We can estimate f_max_modal from the first sync (g = ΦᵀF, the force-displacement relationship gives an effective stiffness) or use a conservative default.

### 4.2 Approach A: Newton step

For a quadratic model, **one Newton step reaches the exact equilibrium**:
```
dq = K⁻¹ · g    where g = ΦᵀF
q_eq = q + dq
```

This is exact when the forcefield is quadratic in the modal subspace. For pentacene (approximately quadratic bending), this converges in 3-5 Newton steps even from large distortions (verified: 4 Newton steps for 0.5 Å bend).

For large distortions (beyond the fitted radius), use a **trust region**:
1. Compute Newton step dq = K⁻¹·g
2. If |dq| > trust_radius, scale: dq *= trust_radius / |dq|
3. Apply, sync, check if force decreased
4. Adapt trust radius (increase on success, decrease on failure)

### 4.3 Approach B: Modal damped MD with large timestep

Without K, we use damped MD in modal space:
```
v_q ← damp · v_q + dt_modal · g    (g = ΦᵀF, true projected force)
q ← q + dt_modal · v_q
```

The timestep dt_modal is large (≈ 10/f_max_modal ≈ 22) because only soft modes are evolved. Damping (damp ≈ 0.9-0.95) removes oscillations. Convergence in ~2-5 periods of the slowest mode.

For better convergence, use **modal FIRE** instead of damped MD:
- FIRE adaptively increases dt when force and velocity are aligned
- FIRE resets velocity when force opposes motion
- This combines the large-timestep benefit with FIRE's acceleration

### 4.4 Synchronization cadence

**Approach A:** Between syncs, the modal model uses the cached projected force with K extrapolation:
```
g(q) ≈ g_sync - K·(q - q_sync)
```
For a quadratic model, this is exact — so one Newton step between syncs suffices. For pentacene: 5 syncs total.

**Approach B:** Without K, the force between syncs is just g_sync (constant). Options:
1. **Sync every step:** each modal step costs 1 full-force eval. Simple but expensive.
2. **Extrapolate with online K:** after 2 syncs, estimate K_secant ≈ -Δg/Δq, then extrapolate. Reduces syncs to ~3-5.
3. **Multiple cached-force steps:** take several damped MD steps with cached g_sync, then sync. Works if the landscape is smooth (g doesn't change much between syncs).

The sync cost is one full-force evaluation. The goal is to minimize syncs while maintaining accuracy.

## 5. The V-shape: alternating coarse and fine

### 5.1 Why V-shape for nonlinear systems

In Approach A (coarse-first), we do all coarse steps, then all fine steps. This works when soft and hard modes are decoupled (pentacene: bend/twist doesn't affect bond stretches).

For nonlinear systems, soft and hard modes can be coupled:
- A large coarse step (soft DOF) may excite hard DOFs (bond stretches change)
- Fine smoothing (hard DOFs) may change the forces on soft DOFs
- Need to alternate: coarse → fine → coarse → fine → ...

This is the V-shape (or F-shape for multiple cycles):
```
V-cycle:
  1. Coarse step: move soft DOFs (may excite hard DOFs)
  2. Fine smoothing: few FIRE steps (relax hard DOFs)
  3. Check convergence
  4. Repeat if not converged
```

### 5.2 V-shape with force projection (Approach B)

```
for each V-cycle:
  sync: g = ΦᵀF(x)                    — 1 full-force eval
  modal damped MD: q ← q + dt·v        — n_coarse steps (cheap, no full-force)
  reconstruct: x = x_ref + Φ·q
  fine smoothing: 1-5 FIRE steps       — 1-5 full-force evals
  check: |F| < threshold?
```

Cost per V-cycle: 1 sync + 1-5 fine = 2-6 full-force evals.
Number of V-cycles: 2-5 for moderately nonlinear systems.
Total: 4-30 full-force evals (vs 100s-1000s for plain FIRE).

### 5.3 When V-shape is NOT needed

For pentacene (approximately quadratic, soft-hard decoupled), the V-shape is unnecessary. Coarse-first (Approach A) converges soft DOFs, then 1 FIRE step finishes hard DOFs. The V-shape adds overhead (extra syncs) without benefit.

Use V-shape when:
- Soft and hard modes are coupled (large coarse steps excite hard DOFs)
- The system is nonlinear (force landscape changes during relaxation)
- The molecule undergoes conformational changes (modes shift)

## 6. Benchmark protocol

### 6.1 What to count

- **Setup cost (NOT counted):** mode generation, stiffness fitting (Approach A only), Cholesky factorization
- **Coarse phase:** N_sync (full-force evals for synchronization) + N_modal_steps (cheap, ~free)
- **Fine phase:** N_fire (full-force evals for FIRE finishing/smoothing)
- **Total:** N_total = N_sync + N_fire
- **Baseline:** N_plain = plain FIRE steps to same threshold

### 6.2 What to compare

Compare against **FIRE only** (the best available optimizer). Do NOT compare against damped MD — that would be dishonest.

### 6.3 Distortion

- **Low-frequency:** parabolic bend + axial twist (the soft modes the coarse model should handle)
- **Small high-frequency:** white noise (the hard modes the fine phase should handle)
- **Amplitude sweep:** test multiple amplitudes to find where Approach A breaks down and Approach B is needed

### 6.4 Convergence criterion

All strategies must reach the same final force threshold (fmax < 1e-3 eV/Å) AND the same minimum (check z_rms, energy, and trajectory visually).

### 6.5 Different minima

If the modal approach finds a different minimum than plain FIRE, investigate:
1. Is plain FIRE trapped in a local minimum? (UFF dihedral multi-well — verified for pentacene)
2. Is the modal approach projecting out the distortion incorrectly?
3. Is the reference geometry properly relaxed? (must relax with ALL terms ON)

**Always save trajectories and inspect visually.** For pentacene, the modal approach correctly finds the planar ground state (E=4.67e-9) while FIRE gets trapped in a non-planar local minimum (E=1.06e-7, 23× higher energy). This is a real benefit, not a bug.

## 7. Implementation notes

### 7.1 Current status

- `build_bend_twist_modes`: implemented, orthonormal, tested
- `ModalQuadratic::fit_central`: implemented, tested (parity to 1e-16)
- `ModalQuadratic::solve_force`: implemented (Newton step: dq = K⁻¹·g, dx = Φ·dq)
- `ModalQuadratic::project_force`: implemented (g = ΦᵀF)
- `UffHessianOp`: implemented (finite-difference full UFF Hessian) — useful for diagnostics, NOT for production (too expensive: 2×n_dof force evals)

### 7.2 Approach A: implemented and verified

- Newton step with trust region: implemented in `relax_pentacene_speedup.rs`
- 53× speedup on pentacene (5 syncs + 4 Newton + 1 FIRE = 6 full-force evals vs 323 for plain FIRE)
- Amplitude sweep: 37-60× speedup across 0.01-1.0 Å distortions
- Finds true planar ground state (FIRE gets trapped in dihedral multi-well)

### 7.3 Approach B: to be implemented

- Force projection: g = ΦᵀF(x) — already available via `ModalQuadratic::project_force`
- Modal damped MD: q ← q + dt·v, v ← damp·v + dt·g — trivial to implement
- Modal FIRE: same as full-atom FIRE but in modal space — straightforward
- V-shape loop: alternate coarse + fine — straightforward
- Online K estimation (secant/BFGS): K ≈ -Δg/Δq — straightforward
- Needs testing on nonlinear systems (hexadecane with torsional DOFs)

### 7.4 What was abandoned (and why)

- **Finite-difference Hessian (UffHessianOp) for production:** costs 2×n_dof force evals — never competitive. Keep for diagnostics only (eigenvalue analysis, stiffness verification).
- **Geometric prolongation / pivot-based V-cycle as the primary strategy:** the pivot-based coarse space is not physically motivated (unlike modal bend/twist). The modal basis is better because it captures the actual soft modes of the molecule. The V-cycle infrastructure (`TrussOp`, `galerkin_coarse`, `solve_two_grid`, etc.) is retained as diagnostic infrastructure and may be reused for the linear inner solve if needed.

## 8. Test molecules and coarse-point assignments

### 8.1 Pentacene — rigid aromatic stick (36 atoms)

**Geometry:** 22 C + 14 H, flat in xy-plane, elongated along x (~13Å). Five fused benzene rings. Essentially a rigid rod.

**Soft modes:** long-axis bending (beam flex), axial twist. C-C and C-H stretches are much stiffer.

**Coarse modes (modal approach):** out-of-plane bend `n·sin(πs)` + axial twist `(2s-1)·[u×(r-r_axis)]`. 2 modes, 2 coarse DOF. Captures the softest internal modes.

**Coarse points (geometric pivot approach):** 2–3 pivots (both ends + optional midpoint). 6–9 coarse DOF. Captures rigid motion + bending but not twist.

**Why it's the ideal test case:** aromatic rings are rigid → soft DOFs are global bend/twist, not local torsions. sp2 carbons have no free rotation → no highly nonlinear torsional DOFs. The inversion + dihedral terms provide smooth quadratic bending stiffness. The molecule is planar at equilibrium → out-of-plane bend/twist are the softest modes.

### 8.2 n-Hexadecane — flexible rope (50 atoms)

**Geometry:** 16 C backbone (zigzag, ~19Å end-to-end) + 34 H.

**Soft modes:** many low-frequency bending/torsion modes. The chain can fold, twist, and bend in many ways. This is the **hardest case for Jacobi** and where multigrid/modal should shine.

**Coarse points (geometric):** 4–8 pivots along the chain (12–24 coarse DOF). Captures overall shape + several bending modes.

**Spectral approach:** expect a dense spectrum of soft bending/torsion modes with no clear gap. May need 15–20 modes. This is where Approach B (force-projection, no fitting) is likely needed — the quadratic model breaks down for torsional DOFs.

### 8.3 DiTriptyceno_helicene — branching I-beam (104 atoms)

**Geometry:** flat aromatic core + 4 triptycene protrusions sticking out of plane.

**Soft modes:** protrusion rotation/twisting relative to core (hinge modes). Possible spectral gap between hinge modes and internal stiff modes.

**Coarse points (geometric):** 6 pivots (2 core ends + 4 protrusion tips). 18 coarse DOF. Captures core rigid motion + 4 protrusion displacements.

**Spectral approach:** best candidate for spectral-gap approach — the rigid core creates a natural separation.

### 8.4 Automatic coarse-point assignment (future)

1. **Shape-based:** PCA of the molecule. Elongated (one dominant axis → stick/rope) → place pivots along that axis. Flat (two dominant axes → sheet) → place a grid. Branched → detect branches (graph articulation points) and place one pivot per branch tip + core.
2. **Spectral-gap-based:** relax, compute vibration spectrum, find the largest gap in low eigenvalues, take all modes below the gap.
3. **Hybrid:** geometric pivots to bootstrap (cheap), then switch to spectral once roughly relaxed.

## 9. RAFF extension (6-DOF atoms)

### 9.1 The challenge

RAFF atoms have position `x_i ∈ ℝ³` + orientation `q_i ∈ S³` (unit quaternion). The port energy

$$E_{i\alpha} = \frac{k_p}{2} |x_j - x_i - R(q_i)\,(l_{i\alpha}\,a_{i\alpha})|^2$$

couples translation and rotation. The Hessian w.r.t. `(δx_i, δθ_i)` (using `δq ≈ ½ δθ` for small rotations) is a **6×6 block** per atom:

$$H_i = \begin{pmatrix} H_{xx} & H_{x\theta} \\ H_{\theta x} & H_{\theta\theta} \end{pmatrix}$$

where:
- `H_{xx} = Σ_α k_p I` (3×3, translation stiffness)
- `H_{xθ} = −Σ_α k_p [r_α]_×` (3×3, translation-rotation cross)
- `H_{θθ} = Σ_α k_p [r_α]_×^T [r_α]_×` (3×3, rotational stiffness)

### 9.2 Route A — Adiabatic elimination (reduce to 3 DOF) — FIRST

In adiabatic mode (`OrientMode::Adiabatic`), rotations are solved out each step (Wahba solve). The effective 3-DOF stiffness is the Schur complement:

$$H_i^{\text{eff}} = H_{xx} − H_{x\theta}\,H_{\theta\theta}^{-1}\,H_{\theta x}$$

This is a 3×3 block per atom → **reuses the UFF modal approach verbatim** (just with a modified stiffness that encodes the Schur-complemented contribution). The Schur-complemented stiffness is not a simple axial spring — it's a dense 3×3 per atom with angle-like coupling.

### 9.3 Route B — Full 6-DOF multigrid — LATER

Keep all 6 DOF. Block size becomes 6×6. Prolongation P maps coarse 6-DOF atoms to fine 6-DOF atoms. A coarse atom's rotation represents a rigid-body rotation of a cluster — this is the **rigid-cluster coarse graining** idea, closely related to the beam prolongation and to the Blended Rigid Body Frames work in `MultiGridFF/`.

More powerful (captures rotational soft modes directly) but more complex. Defer to after Route A is validated.

### 9.4 RAFF plan

1. Generalize the modal stiffness to accept per-edge **3×3 stiffness blocks** (not just `k_eff·n⊗n`). The bond-only case sets the block to `k_eff·n⊗n`; the RAFF-adiabatic case sets it to the Schur-complemented contribution.
2. Build the RAFF-adiabatic stiffness from `RaffTopology` + `RaffState` (compute `H_i^eff` per atom).
3. Test: modal relaxation on RAFF-adiabatic system, parity vs dense, convergence.
4. Wrap as `relax_raff_modal` (adiabatic).

## 10. GPU (OpenCL) roadmap

### 10.1 Kernels copied (not yet wired)

- `opencl/multigrid.cl` ← `NumericalMathPlayground/.../kernels_multigrid.cl` (restriction, prolongation, coarse Cholesky)
- `opencl/block_jacobi.cl` ← `NumericalMathPlayground/.../kernels_block_jacobi.cl` (block Jacobi smoother, residual, Dinv)

These use the truss layout (CSR neighbors, patch clusters). They will need adaptation to SurfMol's buffer conventions (`AlignedVec`, `bytemuck` casts, `float4` packing per AGENTS.md GPU rules).

### 10.2 GPU plan

1. Adapt `block_jacobi.cl` to SurfMol bond layout (`Uff::bon_atoms` → CSR neighbor lists).
2. Adapt `multigrid.cl` prolongation/restriction to the pivot-based P (the reference uses a patch-cluster layout; we may keep that or switch to a CSR-ish P layout).
3. Host orchestration in a new `crates/libs/molff/src/multigrid_ocl.rs` (or an `opencl` crate if one materializes — currently SurfMol has no OpenCL Rust crate; `ocl` 0.19 is the chosen dep per `DESIGN_GOALS.md`).
4. Parity: GPU V-cycle vs CPU V-cycle, same residual history to tol 1e-5 (f32).
5. Performance: benchmark vs CPU on ≥ 1000-atom systems. Target: match or beat the 60–246× speedup seen in the reference (NVIDIA GTX 1650).

**Note:** the modal approach (Approach A/B) may not need GPU acceleration for the coarse phase — it's already tiny (2–20 DOF). GPU acceleration is most relevant for the fine-phase full-force evaluation, which is the existing UFF/RAFF OpenCL pipeline. The linear V-cycle GPU kernels are retained for potential future use as a linear preconditioner.

## 11. See also

- `notes/reports/2026-08-29_multigrid_consolidated_report.md` — **consolidated report** (all history, benchmarks, insights, negative results)
- `doc/topical_audit/multigrid.md` — cross-implementation map (SurfMol vs NumericalMathPlayground)
- `NumericalMathPlayground/topics/LinarElasticity/` — reference Python+OpenCL implementation
