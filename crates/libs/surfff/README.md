---
type: rust-crate
title: surfff
description: Surface–molecule interaction forcefield — folded periodic potential with separable tensor-product Fourier basis, complex recurrence for integer harmonics, NaCl surface template.
tags: [rust, crate, forcefield, surface, fourier, periodic, nacl]
timestamp: 2026-08-25
---

# surfff

Surface–molecule interaction forcefield using a **folded periodic potential** with separable tensor-product basis. Models the substrate as a 2D periodic lattice with z-decaying interactions. Three independent interaction channels: electrostatics (charge), Pauli repulsion (short-range), London dispersion (medium-range).

## What it does

`SurfaceFolded` evaluates energy and force for an atom at position `(x, y, z)` above a periodic surface. The potential is a sum of tensor-product basis functions:

```
E(x,y,z) = Σ_{i,j,k} c_{ijk} · bx_i(x) · by_j(y) · bz_k(z)
```

where `bx`, `by` are Fourier harmonics (cos/sin of integer multiples of lattice coordinates) and `bz` is exponential z-decay `exp(-kz·(z-z0))`. The separable structure reduces per-atom cost from O(N³) to O(N) per dimension.

## Key algorithms

- **Complex recurrence for harmonics** (`precompute_harmonics`) — when harmonic frequencies are integers (0, 1, 2, ...), computes all cos(nφ)/sin(nφ) with **1 trig call + nmax complex multiplies** instead of nmax trig calls. Uses `z = exp(iφ)` and recurrence `z^(n+1) = z^n · z`. The `Complex` struct is a minimal 2-f64 pack with a single `mul` operation. `all_integer_harmonics` checks at construction time whether the optimization applies.

- **Lattice coordinate transform** — Cartesian (x,y) → fractional (u,v) via inverse 2D lattice matrix. `u, v` wrapped to [0,1) via `u - u.floor()` for periodic boundary handling. Forces converted back to Cartesian via chain rule with the inverse lattice matrix.

- **REQ→PLQ conversion** (`req2plq`) — maps standard vdW parameters `[R_vdw, E_vdw, Q, H]` to basis coefficients `[Pauli, London, Q, H]` with exponential scaling: `k = -α`, `e = exp(k·R)`, `c_l = e·sqrt(E)`, `c_p = e·c_l`. The `α` parameter controls how steeply the Pauli/London amplitudes grow as atoms get larger.

- **Force clamping** (`clamp_force`) — when atoms penetrate the surface (z < z0), forces can diverge. `clamp_force` caps the force magnitude to `fmax` (default 100 eV/Å) to prevent MD blowup while preserving the energy value.

- **NaCl surface template** (`setup_nacl_surface`) — constructs a NaCl-like substrate with 3 z-basis layers: `iz=0` Pauli repulsion (kz = 2·β_morse, steeper), `iz=1` London attraction (kz = β_morse), `iz=2` electrostatics (kz = β_charge, slower decay). X/Y harmonics `k=[0,1]` give period a/2 matching Na-Cl spacing. Charge pattern: checkerboard with +1 at Na sites, -1 at Cl sites. Pauli prefactor amplified 40× to overcome the atomic P<<L ratio and create a proper minimum ~1.5 Å above the surface.

## Structs

- **`SurfaceFolded`** — main potential: 2D lattice vectors + inverse, harmonic frequencies (`kx`, `ky`, `kz`), z-reference positions (`z0`), coefficient arrays (`coeffs_q`, `coeffs_p`, `coeffs_l`), integer harmonic caches (`kx_int`, `ky_int`, `kx_max`, `ky_max`).
- **`SurfaceScratch`** — preallocated buffers (`bx`, `by`, `bz`, derivatives, `cux`/`sux`/`cvy`/`svy`) to avoid per-atom allocation in hot loops. Use `eval_all_scratch` instead of `eval_all` for batched evaluation.

## Design decisions

- **Separable basis** — `B(x,y,z) = bx(x)·by(y)·bz(z)` reduces the triple loop from O(N³) to O(N) per dimension with small constant factors.
- **Integer harmonic detection at construction** — `kx_int`/`ky_int` precomputed in `new()`, so the per-atom evaluation branch is predictable.
- **Scratch buffer reuse** — `SurfaceScratch` eliminates per-atom allocations. Sized to `kx_max+1` / `ky_max+1` for the complex recurrence arrays.
- **Three coefficient channels** — `coeffs_q` (electrostatics), `coeffs_p` (Pauli), `coeffs_l` (London) stored separately so the same basis evaluation serves all three with different amplitudes.

## What does NOT belong here

- Intra-molecular forcefields → `molff`
- MD orchestration → `surfmol`
- Topology → `moltopo`

## See also

- `surfmol` — `MolWorld` owns `Option<SurfaceFolded>` and calls `eval_all_clamped` during force evaluation
- `tests/test_surface.rs` — z-scan and x-scan plots, spot checks for neutral/positive/negative charge states
