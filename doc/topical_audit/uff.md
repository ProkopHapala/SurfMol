---
type: topical-audit
title: "UFF — Universal Force Field"
description: Cross-implementation map of UFF (bonds, angles, dihedrals, inversions) across SurfMol Rust, FireCore C++, and SPAMMM Python/OpenCL.
tags: [topic, forcefield, uff, cross-language, parity]
timestamp: 2026-09-29
---

# UFF — Universal Force Field

Cross-implementation map of the Universal Force Field (Rappé et al., 1992) across the three codebases that implement it. SurfMol Rust is the active implementation; FireCore C++ is the correctness/performance reference; SPAMMM Python/OpenCL is the production GPU reference.

## Summary

UFF models four bonded interaction types: bond stretching, angle bending (Fourier form), dihedral torsion, and inversion (out-of-plane bending at sp² centers). The parameter assignment pipeline derives all four from atom types + hybridization + UFF formulas. The critical implementation detail is the **inversion term** — without it, aromatic molecules like pentacene have no planarity enforcement and buckle under bond strain.

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `crates/libs/molff/src/uff.rs` | **active** | SurfMol Rust CPU. Full 4-term UFF with `setup_params` (real params) and `set_dummy_params` (testing). Force piece assembly with complex-number angle powers. |
| `crates/libs/moltopo/src/params.rs` | **active** | UFF formula functions: `uff_bond_length`, `uff_bond_k`, `uff_angle_sp2/sp3/sp1`. `.dat` file parser. |
| `crates/libs/moltopo/src/assign_uff.rs` | **active** | UFF type assignment from topology + hybridization. |
| `crates/libs/moltopo/src/topology.rs` | **active** | Topology builders: `build_dihedrals_from_bonds` (dedup `i4>i1`), `build_inversions_from_bonds` (3 per trigonal center). |
| `crates/libs/surfmol/src/mol_world.rs` | **active** | `MolWorld::setup_uff_params` wrapper, `BondedFFMode::Uff` dispatch. |
| `opencl/UFF.cl` | **active** | OpenCL kernel for UFF bond forces. |
| `crates/libs/oclff/src/uff.rs` | **active** | `UffOcl` — OpenCL harness for UFF bonds. |
| `FireCore/src/forcefields/uff.cpp` | **active** | C++ reference. Full UFF with `evalUFFf` force kernels. |
| `FireCore/src/forcefields/UFFbuilder.h` | **active** | C++ reference. `assignUFFparams` — the canonical parameter assignment pipeline. |
| `SPAMMM/UFF.py` | **active** | Python/OpenCL reference. UFF + SPFF with pi-orbital bending. |

## Parameter assignment pipeline

The canonical pipeline (ported from FireCore `UFFbuilder.h:assignUFFparams`):

| Step | FireCore | SurfMol Rust | Parity |
|------|----------|-------------|--------|
| Atom type assignment | `assignUFFtypes` | `moltopo::assign_uff::assign_uff_types` | ✅ verified |
| Bond params (k, l0) | `uff_bond_length` + `uff_bond_k` | `moltopo::params::uff_bond_length` + `uff_bond_k` | ✅ verified |
| Angle params (Fourier c0-c3) | `setAngleParams` | `Uff::setup_params` angle loop | ✅ verified |
| Dihedral params (V, d, n) | `setDihedralParams` by hybridization | `Uff::setup_params` dihedral loop | ✅ verified |
| Inversion params (K, C0-C2) | `setInversionParams` | `Uff::setup_params` inversion loop | ✅ verified |

### Key formulas

**Bond length:** `l0 = r_i + r_j + r_bo - 0.08·|χi-χj|` where `r_bo` depends on bond order (0.0 for single, 0.21 for double, 0.44 for triple, **0.0 for aromatic**), `χ` is electronegativity correction.

**Bond stiffness:** `k = 0.5·28.8·Qi·Qj/l0³` (eV/Å²), where `Qi` is the UFF tabulated bond force parameter.

**Angle Fourier:** `E = k·(Σ Cn·cos(n·θ))` where coefficients depend on central atom hybridization:
- sp³: `c0=2.12, c1=0, c2=-0.53, c3=0` (tetrahedral, θ₀=109.47°)
- sp²: `c0=1.5, c1=0, c2=-0.5, c3=0` (trigonal, θ₀=120°)
- sp¹: `c0=1.0, c1=0, c2=0, c3=0` (linear, θ₀=180°)

**Dihedral:** `E = ½·V·(1+d·cos(n·φ))` — V, d, n from central atom hybridizations:
- sp³-sp³: `V=2.0, d=-1, n=3` (staggered)
- sp³-sp²: `V=1.0, d=-1, n=3`
- sp²-sp²: `V=2.0, d=+1, n=2` (planar, minima at φ=0 and φ=π)

**Inversion:** `E = K·(C₀+C₁cosω+C₂cos2ω)` — K from atom type:
- sp² C, N: `K=6` kcal/mol
- Carbonyl C: `K=50` kcal/mol
- sp³: `K=0` (no inversion)

### Bond order for aromatic bonds

**Critical:** Aromatic `C_R–C_R` bonds use bond order **1.0**, not 1.5 or 2.0. This matches FireCore. Using bond order 2.0 produces l0=1.254 Å (vs. correct 1.458 Å), causing ~3.8 eV compressive strain in pentacene — enough to buckle the molecule out of plane.

## Topology enumeration

| Feature | FireCore | SurfMol Rust | Parity |
|---------|----------|-------------|--------|
| Dihedral dedup | `i4 > i1` in `UFFbuilder.h:1159` | `if l > i` in `build_dihedrals_from_bonds` | ✅ fixed (was generating both `(i,j,k,l)` and `(l,k,j,i)`, causing force cancellation) |
| Inversions per center | 3 per trigonal center (`UFFbuilder.h:1318-1332`) | 3 per trigonal center (`build_inversions_from_bonds`) | ✅ fixed (was generating only 1) |

## Force evaluation

| Feature | FireCore | SurfMol Rust | Parity |
|---------|----------|-------------|--------|
| Bond forces | Direct in per-atom loop | Direct in per-atom loop (not bucketed) | ✅ |
| Angle forces | Trig-based | Complex-number powers (`Vec2d` as cosθ+isinθ) | ✅ (no trig in inner loop) |
| Dihedral forces | Trig-based | Complex-number powers | ✅ |
| Inversion forces | Trig-based | Complex-number powers | ✅ |
| `hneigh` cache | Bond vectors cached | Bond vectors + inverse distance cached (`Quat4d.xyz=dir, .w=1/|r|`) | ✅ |
| Force piece assembly | Per-term buffers → assembled | Per-term buffers (`fang`,`fdih`,`finv`) → `Buckets` (`a2f`) | ✅ |

## Parity Status

| Test | Reference | Tolerance | Status |
|------|-----------|-----------|--------|
| Pentacene planar energy | FireCore | — | ✅ E_bond=3.83, E_angle=0.01, E_dih=0, E_inv=0 |
| Pentacene out-of-plane force | Physical (restoring force) | sign check | ✅ Fz=-11.7 eV/Å for 0.5 Å displacement |
| Pentacene pyramidalization relaxation | Physical (returns to planar) | z_rms < 0.05 | ✅ z_rms 0.05→0.012 in 177 FIRE steps |
| Pentacene bend relaxation | Physical (reduces bend) | z_rms reduced | ✅ z_rms 0.15→0.11 (partial — UFF dihedral multi-well) |
| Xylitol bond relaxation | `set_dummy_params` baseline | convergence | ✅ converges with bond-only params |

## Open Issues

- **SPFF not on CPU:** The `Ksp` pi-sigma orthogonality term (SPFF) is implemented in `opencl/SPFF.cl` and `oclff/src/spff.rs` but not in the Rust CPU `molff` path. SPFF is needed for global bending stiffness of aromatic sheets.
- **molengine param setup:** `load_topology_from_json` creates a `Uff` with zero parameters. `molengine` does not call `setup_uff_params` — physical UFF runs require wiring this in.
- **Editor param setup:** The editor hand-rolls UFF parameter setup in `src/main.rs:308-311` using `.dat` lookups, not via `MolWorld::setup_uff_params`. Both paths produce equivalent parameters but the duplication should be consolidated.
- **Dihedral multi-well:** UFF sp²-sp² dihedrals `V·(1+cos(2φ))` have minima at both φ=0 and φ=π, allowing non-planar configurations with zero dihedral energy. This is a known UFF limitation for aromatic systems — SPFF's `Ksp` term resolves it.

## See also

- [`/userguide/uff_spff.md`](/userguide/uff_spff.md) — end-user guide for UFF/SPFF setup and relaxation
- [`/crates/libs/molff/README.md`](/crates/libs/molff/README.md) — `molff` crate docs
- [`/crates/libs/moltopo/README.md`](/crates/libs/moltopo/README.md) — `moltopo` crate docs (topology, params, type assignment)
- `FireCore/src/forcefields/uff.cpp` — C++ reference force evaluation
- `FireCore/src/forcefields/UFFbuilder.h` — C++ reference parameter assignment (`assignUFFparams`)
- `SPAMMM/UFF.py` — Python/OpenCL reference
- Rappé, A. K., et al. "UFF, a full periodic table force field..." *J. Am. Chem. Soc.* 114.25 (1992): 10024-10035.
