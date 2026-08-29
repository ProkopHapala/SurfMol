---
type: design-spec
title: Coarse-grained modal relaxation — design specification (v2)
description: Two complementary coarse-grained relaxation contracts. (A) Staged coarse-manifold/decoder relaxation fits reduced stiffness, evolves only coarse coordinates, decodes canonical atomistic geometry, then refines; it gives 57× on a pure in-manifold pentacene distortion and 53.8× on the mixed decoder workflow. (B) Additive Galerkin relaxation preserves unresolved atomistic coordinates and gives 1.66× on the mixed input. Force-projection V-shape cycling is the nonlinear fallback when a fixed fitted model or decoder is inadequate.
tags: [multigrid, modal, coarse-graining, timestep-scaling, relaxation, pentacene, design-spec, amortized-fitting, galerkin, v-shape, force-projection, newton]
timestamp: 2026-08-29
---

# Coarse-Grained Modal Relaxation — Design Specification (v2)

> **Two contracts must remain separate.** For additive same-state relaxation, `x←x+ΦΔq` must preserve `(I−ΦΦᵀ)(x−x_ref)`; this gives 323→194 evaluations (1.66×) on the mixed pentacene input. For staged coarse-manifold simulation, restriction intentionally drops that complement and decoding uses `x=D(q)` before atomistic refinement; this restores the 323→6 (53.8×) workflow. A pure linear-modal input has zero complement, so additive and decoder updates coincide and both give 285→5 (**57×**), proving that the large acceleration is real for coarse-representable deformation rather than solely an artifact of deleting noise.

## 1. Big picture: why coarse-graining and where the speedup comes from

### 1.1 The intended application

The target application is **global optimization of molecules on surfaces**: thousands of molecules, each performing millions of MD steps / relaxation steps, exploring conformational space on a surface potential. The coarse-grained model is **built once per molecule type** (e.g., pentacene) and **reused across all instances and all timesteps**. Any setup cost (mode generation, stiffness fitting) is amortized over the entire simulation campaign.

**Setup cost is NOT counted in the per-simulation benchmark.** The benchmark measures only the simulation phase: coarse steps (with syncs) + fine finishing steps.

### 1.2 Where the speedup comes from: TIMESTEP SCALING

The primary acceleration mechanism is **stiffness separation**:

- **Full-atom dynamics:** the stable explicit timestep is constrained by the stiffest atomistic mode (bond stretch, local H motion).
- **Reduced dynamics:** directions outside `span(Φ)` are frozen during a pure coarse update, so the relevant spectrum is `M_c⁻¹H_c`, with `M_c=ΦᵀMΦ` and `H_c=ΦᵀHΦ`. If Φ contains only genuinely soft modes, a larger step is possible.
- **Reduced Newton:** when a fitted K accurately represents `H_c`, `Δq=K⁻¹g` removes quadratic coarse error directly; a true-energy trust region globalizes the step away from the fit point.
- **Cost separation:** acceleration is useful only if reducing coarse error decreases the subsequent fine work, or if several coarse updates can be taken using a reusable/independently evaluable reduced model.

A numerical timestep ratio cannot be inferred from K alone without consistent masses and integrator units, and it is not constant far from the reference. The benchmark therefore treats reduced curvature as a proposal mechanism and validates nonlinear progress with full energy evaluations.

### 1.3 Why pentacene is the ideal test case

Pentacene is chosen deliberately because its low-energy subspace is **approximately quadratic bending along soft eigenmodes**:

- Aromatic rings are rigid → the soft DOFs are global bend/twist, not local torsions
- sp2 carbons have no free rotation → no highly nonlinear torsional DOFs
- The inversion + dihedral terms provide smooth quadratic bending stiffness
- The molecule is planar at equilibrium → out-of-plane bend/twist are the softest modes

Aliphatic chains (hexadecane) have free rotations around single bonds — these are highly nonlinear torsional DOFs that a quadratic modal model cannot capture. Pentacene avoids this complication, making it the **minimal test system** for validating the modal approach. For nonlinear systems, see Approach B below.

## 2. Two complementary approaches

There are two ways to do coarse-grained modal relaxation. They share the same mode basis Φ and the same timestep scaling principle, but differ in how the coarse dynamics is computed:

### Approach A: Fitted staged coarse-to-fine relaxation

**Fit a reusable reduced internal model, evolve only coarse coordinates, decode an atomistic geometry, then refine.**

```
SETUP (one-time, amortized):
  1. Relax a canonical molecule → reference geometry x_ref
  2. Define encoder q=R(x) and decoder x=D(q); linear prototype D(q)=x_ref+Φq
  3. Fit reduced internal stiffness K (or nonlinear E_c(q))
  4. Factor K / prepare the tiny coarse optimizer

SIMULATION (per coarse molecule instance):
  1. Restrict/initialize q; unresolved atomistic coordinates are not part of coarse state
  2. Relax E_c(q)+E_external(D(q)) with Newton or large-step modal FIRE
  3. Decode x=D(q)
  4. Run full-atom FIRE only for reconstruction strain and missing interactions
```

This is the preferred fast path when the simulation genuinely uses a coarse molecular state. Decoding a canonical geometry is then intentional—not a claim that discarded atomistic noise was dynamically relaxed. On a pure in-manifold pentacene distortion, additive and decoder updates coincide and both give **57×** (285→5 evaluations). On the mixed input, decoder mode gives **53.8×** (323→6), while same-state additive relaxation gives 1.66×.

**Primary optimization:** stop evaluating full internal UFF at every coarse step. The fitted internal force `g_int(q)≈−Kq` is already available cheaply. Evaluate only external/surface/intermolecular contributions in coarse form (or project those forces), take many cheap coarse steps, then perform one full synchronization before refinement. For internal-only relaxation around the canonical reference, the quadratic model reaches `q=0` in one Newton step; repeated full-UFF syncs are conservative validation, not the intended production algorithm.

**Pros:**
- Removes stiff atomistic directions from the active state, enabling large reduced updates or direct Newton equilibrium steps.
- Canonical decoding initializes fine coordinates close to valid bond/angle geometry.
- Fitting cost is negligible when amortized across many instances of one molecule type.

**Cons / contract:**
- It does not preserve arbitrary atomistic microstate information; use additive Galerkin if that information matters.
- Decoder quality is crucial. A linear tangent decoder is inadequate for large finite rotations; use curvilinear internal coordinates/rigid fragments.
- K fitted at one reference can fail under strong nonlinear deformation or changing environment; validate/refit or fall back to Approach B.

**When to use:** coarse-grained global search, adsorption/packing, or long coarse dynamics where each molecule is represented by a small set of conformational coordinates and canonical fine geometry can be regenerated before final UFF refinement.

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
      estimate a safe reduced step scale from projected-force history or backtracking
      modal FIRE / gradient step using g = ΦᵀF(x)
      update: x ← x + Φ·dq         — preserve unresolved fine coordinates
      evaluate the true force after every accepted nonlinear coarse step unless a validated surrogate predicts it
    FINE PHASE (hard DOFs):
      few FIRE steps on full atom (relax bond stretches, H atoms)
      — only hard modes, converges in 1-5 steps —
    check: if |F| < threshold → done
```

**Key insight, with an important qualification:** projection removes atomistic directions outside `span(Φ)`, so the relevant stability scale is the largest eigenvalue of the **current reduced Hessian** `H_c(x)=ΦᵀH(x)Φ`, not the largest eigenvalue of the full Hessian. This often permits a much larger step than atomistic dynamics, but a value such as `dt=22` is not justified without estimating `H_c` (and the reduced mass matrix if using dynamics). Nonlinearity can increase `λ_max(H_c)` far from the reference. Use backtracking, projected secants/Barzilai–Borwein, or a small online BFGS model to set the step. The force projection gives the true modal gradient, not a free prediction of that gradient at future points.

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

| Aspect | Approach A: staged decoder | Approach B: additive force-projection |
|--------|----------------------------|---------------------------------------|
| State | Coarse coordinates q only; decoder regenerates atoms | Full atomistic state preserved |
| Setup cost | Fit reusable K/E_c and decoder | Modes only; optional online Hessian |
| Coarse step | Newton or large-step modal FIRE on cheap fitted model | Projected gradient/FIRE, optionally online BFGS |
| Full-force syncs | Sparse; ideally only validation + final refinement | At least one per true nonlinear trial unless using a validated surrogate |
| Fine complement | Intentionally replaced by canonical decode | Preserved by `x←x+ΦΔq` |
| Nonlinear systems | Needs nonlinear/curvilinear decoder and model | More robust gradient, but basis can still be inadequate |
| V-shape | Usually staged coarse→fine | Yes when soft-hard coupling is significant |
| Pentacene result | 57× pure in-manifold; 53.8× mixed decoder workflow | 57× pure in-manifold; 1.66× mixed same-state workflow |
| Best for | Coarse global search and canonical reconstruction | Existing atomistic state, nonlinear correction/fallback |

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

For pentacene with real UFF, the fitted two-mode stiffness is approximately `K_bend=0.058` and `K_twist=0.20` in the normalized coordinate convention, much smaller than local bond-stretch curvature. This confirms qualitative soft/hard separation, but these numbers alone are not physical frequencies: the mode normalization is Euclidean rather than mass-normalized, and the FIRE timestep uses code-specific units.

For explicit dynamics the limit depends on the eigenvalues of `M_c⁻¹H_c`, where `M_c=ΦᵀMΦ`; for optimization it depends on the globalization method. The previous `dt≈22` and unconditional 1000× timestep claims were unsupported. Use the fitted K for a safeguarded Newton proposal, or compute consistent reduced masses and verify stability numerically before making a dynamics claim.

**This applies to both approaches only locally.** Approach A obtains a curvature model from fitted K. Approach B must estimate reduced curvature online or use an energy-decreasing line search. One force sample `g=ΦᵀF` does not determine `λ_max(H_c)`; at least a displacement/force difference or a backtracking trial is required.

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

Without fitted K, use a projected gradient or modal FIRE step with nonlinear globalization:
```
g_k = ΦᵀF(x_k)
dq_k = α_k g_k                         # or modal FIRE velocity update
x_trial = x_k + Φ dq_k                 # additive: preserve fine complement
accept if E(x_trial) < E(x_k); otherwise reduce α_k
```

Initialize `α_k` from a conservative value or a projected secant, e.g. Barzilai–Borwein `α≈(sᵀs)/(sᵀy)` with `s=q_k−q_{k−1}` and `y=−(g_k−g_{k−1})`, then protect it with bounds and backtracking. Modal FIRE can adapt the scale and momentum, but it still requires true projected-force evaluations and must reset velocity after a rejected step or fine smoothing.

### 4.4 Synchronization cadence

**Approach A:** Between syncs, the modal model uses the cached projected force with K extrapolation:
```
g(q) ≈ g_sync - K·(q - q_sync)
```
For a quadratic model, this is exact — so one Newton step between syncs suffices. For pentacene: 5 syncs total.

**Approach B:** a cached projected force is the gradient at one point, not a force oracle. Safe options:
1. **Evaluate every trial/accepted step:** simplest truthful nonlinear Galerkin method; each trial costs one full-force evaluation.
2. **Online reduced model:** after accepted distinct q values, update a small SPD BFGS/secant approximation and use it to propose the next step; validate every proposal against the true energy/force.
3. **Independent coarse forcefield:** only this permits many genuinely cheap coarse steps without fine synchronization.

Taking many dynamics steps under a constant cached force is not a controlled approximation and can overshoot. The optimization target is fewer **accepted fine-force evaluations**, not a nominal count of cheap steps.

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

**Always save trajectories and inspect visually.** The earlier claim that the modal strategy found the planar ground state was produced by the invalid reconstruction that erased the fine complement. After correction, both strategies reach geometrically similar non-planar states at `fmax<1e-3`, while their reported energies still differ (`1.06e-7` vs `1.42e-5` in the main case). Therefore the present stopping threshold is insufficient for a same-minimum claim; compare after tighter convergence or evaluate both geometries with a common post-processing relaxation.

## 7. Implementation notes

### 7.1 Current status

- `build_bend_twist_modes`: implemented, orthonormal, tested
- `ModalQuadratic::fit_central`: implemented, tested (parity to 1e-16)
- `ModalQuadratic::solve_force`: implemented (Newton step: dq = K⁻¹·g, dx = Φ·dq)
- `ModalQuadratic::project_force`: implemented (g = ΦᵀF)
- `UffHessianOp`: implemented (finite-difference full UFF Hessian) — useful for diagnostics, NOT for production (too expensive: 2×n_dof force evals)

### 7.2 Implemented contract-separated benchmark

- `CoarseContract::Additive`: applies `x←x+ΦΔq` and asserts complement invariance. Mixed input: **1.66×** (323→194).
- `CoarseContract::Decode`: intentionally reconstructs `x=D(q)=x_ref+Φq`. Mixed coarse-to-fine workflow: **53.8×** (323→6).
- Pure in-manifold input (`fine_rms≈7e-17`): additive and decoder are identical and both give **57×** (285→5). This is the fair demonstration that coarse soft-mode relaxation itself can produce an order-of-magnitude speedup.
- Both use true-energy step acceptance; rejected trials reuse the cached previous force.
- The mixed input has `fine_rms≈0.305 Å`, so its additive/decoder difference measures the chosen state contract, not solver quality.

### 7.3 Approach B: to be implemented

- Force projection `g=ΦᵀF(x)` is already available via `ModalQuadratic::project_force`.
- Implement projected gradient/modal FIRE with backtracking or a safeguarded Barzilai–Borwein scale—not an assumed fixed `dt`.
- Alternate coarse updates with a small, measured number of fine smoothing steps only when fine smoothing regenerates significant `|ΦᵀF|`.
- Update a tiny SPD BFGS model from accepted `(Δq,Δg)` pairs; use it for proposals but validate against true energy.
- Test first on controlled nonlinear reduced potentials, then on hexadecane torsions.

### 7.4 Priorities: expensive reasoning vs delegable work

**P0 — explicit state contract (done here):** additive runs must preserve the complement; decoder runs may replace it but must say so. Pure in-manifold inputs are the fair common benchmark.

**P1 — exploit the fitted model fully:** move the internal coarse loop off full UFF. Use `g_int=−Kq` (or fitted nonlinear `E_c(q)`) for cheap steps, add only coarse external/surface/intermolecular forces, and synchronize full UFF sparsely. This is where timestep scaling matters for modal MD/FIRE; for minimization, reduced Newton is faster than integrating many timesteps.

**P2 — improve the decoder/manifold:** finite twist is a curvilinear rotation, while the current Φ is only its tangent. Implement rigid-ring/fragment transforms or nonlinear bend/twist coordinates so large conformations stay chemically valid and external forces can be evaluated on `D(q)`.

**P3 — robust nonlinear Galerkin fallback:** implement Approach B as projected gradient/FIRE + backtracking, then online SPD BFGS. Trigger fine smoothing based on regenerated `|ΦᵀF|` rather than a fixed cycle count.

**P4 — cost-optimal handoff:** sweep coarse tolerance and fine steps; minimize total expensive evaluations. Do not over-solve coarse coordinates once further reduction no longer lowers fine refinement work.

**Delegate to cheaper agents:** parameter sweeps (coarse tolerance, trust radius, fit radius, fine steps/cycle), TSV aggregation, larger molecule runs, documentation tables, and mechanical extraction of modal helpers from the benchmark into `molff`. Reserve expert work for curvilinear coordinate design, force/Jacobian consistency, reduced-mass/stability derivation, and interpreting changes of basin.

### 7.5 What was abandoned (and why)

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
