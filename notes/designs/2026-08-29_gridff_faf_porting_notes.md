---
type: design-notes
title: GridFF + Folded Atomic Forcefield Porting Notes
description: Equations, memory layout, strides, boundary conditions, and performance-critical design decisions for the FireCore/SPAMMM GridFF and FAF kernels copied to opencl/.
tags: [opencl, gridff, faf, b-spline, ewald2d, porting]
timestamp: 2026-08-29
---

# GridFF + Folded Atomic Forcefield Porting Notes

Copied OpenCL sources:

- `opencl/gridff_spammm.cl` — SPAMMM `kernels/gridFF.cl` (2106 lines). Grid construction, tricubic B-spline interpolation, Poisson solver, charge projection.
- `opencl/surface_spammm.cl` — SPAMMM `kernels/surface.cl` (1867 lines). Brute surface Morse, folded basis (FAF), 2D Ewald, isosurfaces.
- `opencl/grids.cl`, `opencl/PME.cl`, `opencl/PME8.cl`, `opencl/contact_surface.cl` — supporting SPAMMM kernels.

Existing SurfMol sources:

- `opencl/GridFF.cl` — FireCore `common_resources/cl/GridFF.cl` (1666 lines). Canonical tricubic B-spline for substrate PLQ grids.
- `opencl/Surface.cl` — FireCore `common_resources/cl/Surface.cl` (partial, missing trailing `Hybrid Potential` section).
- `crates/libs/surfff/src/lib.rs` — CPU reference `SurfaceFolded` (separable Fourier × exponential basis, complex recurrence).

This document records the two system parts — **builder** and **evaluator** — the equations, memory layouts, strides, boundary conditions, and the basis-set / hierarchical-recursive evaluation that makes the GPU kernels fast.

---

## 1. Two-part architecture

Both `GridFF` and `FAF` are deliberately split into a slow, run-once **builder** and a fast, run-many **evaluator**.

### 1.1 GridFF

**Builder** (host/GPU, once per substrate):
1. Evaluate reference Pauli/London/Coulomb on a 3-D grid by brute PBC summation of substrate atoms.
2. Fit cubic B-spline coefficients to the reference grid.
3. Pack Pauli, London, Coulomb into one `float4` / `Vec3d` per voxel (`Bspline_PLQ`).

**Evaluator** (GPU, per probe position):
1. Tricubic B-spline interpolation.
2. Dot the interpolated `(Pauli, London, Coulomb)` with the probe atom's `PLQ` vector.
3. Output `(Fx, Fy, Fz, E)`.

### 1.2 Folded Atomic Forcefield (FAF)

**Builder** (host, once per substrate):
1. Sample the reference energy/force (brute Morse + 2D Ewald Coulomb) on a sparse grid of (x, y, z) points above the surface.
2. Fit the separable basis coefficients `c_{kx,ky,kz}` by least squares.
3. Store either typed coefficients `(ntypes, nbasis)` or factorized `float4` coefficients `(nbasis, 4)` = `(Pauli, London, Coulomb, H-bond)`.

**Evaluator** (GPU, per atom):
1. Fold (x, y) into fractional surface coordinates `(u, v)`.
2. Evaluate the folded basis recursively (two `sincos` + complex powers for x/y, one `exp` per z layer).
3. Contract with coefficients and back-transform to Cartesian force.

---

## 2. Equations

### 2.1 Cubic B-spline basis (GridFF)

Standard uniform cubic B-spline, parameter `u ∈ [0,1]` (FireCore `GridFF.cl:66-87`, `Bspline.h:254-276`):

```
B0(u) =  (1-u)^3 / 6
B1(u) =  (3 u^2 (u-2) + 4) / 6
B2(u) =  (3 u (1+u-u^2) + 1) / 6
B3(u) =  u^3 / 6
```

Derivatives:

```
B'0(u) = -0.5 (1-u)^2
B'1(u) =  0.5 (3u^2 - 4u)
B'2(u) =  0.5 (-3u^2 + 2u + 1)
B'3(u) =  0.5 u^2
```

### 2.2 Tricubic interpolation (GridFF)

For probe at world position `p`, normalized grid coordinate:

```
u = (p - g0) ⊘ dg
```

(⊘ = elementwise divide).  Let `ix,iy,iz = floor(u)` and `tx,ty,tz` the fractions.  The OpenCL `fe3d_pbc` (`GridFF.cl:108-160`, `gridff_spammm.cl:fe3d_pbc`) does:

1. Choose 4×4×4 voxel indices with PBC in x and y, zero or clamped in z.
2. Interpolate along z: `fe1D` gives `(E, dE/dz)` for each `(x, y)` pair.
3. Interpolate along y: `fe2d` gives `(Fy, Fz, E)` for each x.
4. Interpolate along x: `fe3d_pbc` gives `(Fx, Fy, Fz, E)`.

For the combined `PLQ` grid (`sample3D_comb`), the atom-specific mixing vector is dotted at the 1-D stage:

```
cs = (dot(PLQ, E[0]), dot(PLQ, E[1]), dot(PLQ, E[2]), dot(PLQ, E[3]))
E  = dot(b, cs)
Fx = dot(db, cs) * (-inv_dg.x)
```

### 2.3 Morse reference (builder)

For a probe-surface pair at distance `r` with combined parameters `R0`, `E0`, and `K = -alphaMorse`:

```
e   = exp(K (r - R0))
eM  = E0 * e
Pauli  =  eM * e  =  E0 e^{2K(r-R0)}
London = -2 eM     = -2 E0 e^{K(r-R0)}
```

Coulomb is damped:

```
E_C = COULOMB_CONST * Q / sqrt(r^2 + R2damp)
F_C = -E_C * (r^2 + R2damp)^{-1} * r
```

### 2.4 Folded basis (FAF)

A single folded basis function (`surface_spammm.cl:215-221`):

```
B(u,v,z; ku,kv,kz,z0) = cos(2π ku u) cos(2π kv v) exp(-kz (z - z0))   for z ≥ z0
```

Full potential for an atom of type `t`:

```
E_t(x,y,z) = Σ_b c_{t,b} B(u,v,z; ku_b, kv_b, kz_b, z0_b)
```

Fractional coordinates for a 2-D lattice `a=(ax,ay)`, `b=(bx,by)`:

```
det = ax*by - bx*ay
u = ( by*x - bx*y) / det   # = invLvec.x*x + invLvec.y*y
v = (-ay*x + ax*y) / det   # = invLvec.z*x + invLvec.w*y
u -= floor(u);  v -= floor(v)
```

Gradient in fractional coordinates (`surface_spammm.cl:folded_eval_grad`):

```
dB/du = -2π ku sin(2π ku u) cos(2π kv v) bz
dB/dv = -2π kv cos(2π ku u) sin(2π kv v) bz
dB/dz = -kz * B             (if z > z0, else 0)
```

Cartesian force:

```
Fx = -(dE/du * du/dx + dE/dv * dv/dx)
Fy = -(dE/du * du/dy + dE/dv * dv/dy)
Fz = -dE/dz
```

with `du/dx = by/det`, `du/dy = -bx/det`, `dv/dx = -ay/det`, `dv/dy = ax/det`.

### 2.5 2D Ewald electrostatics

For a 2-D periodic slab in the `xy` plane, reciprocal lattice vectors `G = h b1 + k b2` with magnitude `|G| = Gn` (`surface_spammm.cl:1036-1173`):

**Coefficients:**

```
C_G = (2π / (A Gn)) Σ_i q_i exp(Gn z_i) exp(-i G·ρ_i)
```

**Vacuum potential (z above all ions):**

```
φ(ρ,z) = Re Σ_G C_G exp(i G·ρ) exp(-Gn z)
```

**Full potential (arbitrary z):**

```
φ(ρ,z) = -(2π/A) Σ_i q_i |z - z_i|
       + Re Σ_G Σ_i w[g,i] exp(i G·ρ) exp(-Gn |z - z_i|)
```

The complex phases `exp(i G·ρ)` are built from two base phases `exp(i b1·ρ)` and `exp(i b2·ρ)` by recurrence (see §4.3).

### 2.6 Poisson / FFT solver (gridff_spammm.cl)

`poissonW` (`gridff_spammm.cl:1518-1560`) filters a charge-density FFT:

```
kx = (ix <= nx/2 ? ix : ix - nx) * 2π/Lx
k2 = kx^2 + ky^2 + kz^2
f  = coefs.w * exp(-params.x * k2)       # Gaussian damping
if (params.y > 0.5) f /= k2              # 1/k^2 for Poisson
V_k = rho_k * f
```

`laplace_real_pbc` is an SOR relaxation for real-space Laplace with PBC.

---

## 3. Memory layout and strides

### 3.1 GridFF reference grids

All 3-D arrays in `GridFF.cl` and `gridff_spammm.cl` are **x-fastest** (z-major in the flattened index, but x changes fastest along a row):

```
index = ix + nx * (iy + ny * iz)
```

The grid shape is passed as `int4(nx, ny, nz, nxyz)` in `ng` or `ns`, with `nxyz = nx*ny*nz`.

### 3.2 Packed B-spline PLQ

FireCore `GridFF.h:pack_Bspline_d` **transposes** the scalar Pauli/London/Coulomb B-spline coefficients into a **z-fastest** `Vec3d`/`float4` array:

```
source ibuff = ix + nx * (iy + iz * ny)     # scalar, xyz
 target i     = iz + nz * (iy + ix * ny)     # z-fastest, zyx
```

The OpenCL `fe3d_pbc` kernels assume this z-fastest layout; the 4×4×4 stencil is fetched starting at

```
i0 = (iz - 1) + nz * (iy + nx * ix)
```

### 3.3 PBC index patterns for cubic B-spline

Because the 4-point B-spline support can wrap the periodic `x`/`y` boundaries, the kernels precompute four possible index patterns (`GridFF.cl:33-63`, `gridff_spammm.cl:38-59`):

```
iqs[0] = {0, 1,   2,   3  }
iqs[1] = {0, 1,   2,   3-n}
iqs[2] = {0, 1,   2-n, 3-n}
iqs[3] = {0, 1-n, 2-n, 3-n}
```

`choose_inds_pbc(i, n, iqs)` picks the pattern when `i >= n-3`; otherwise `iqs[0]` is used.  This removes per-point modulo branches.  The patterns are held in `__local int4 xqs[4]` / `yqs[4]` (or passed as tables on CPU).

### 3.4 Quintic projection patterns

`project_atoms_on_grid_quintic_pbc` (`gridff_spammm.cl:1385-1493`) extends this to a 6-point stencil with `make_inds_pbc_5` / `choose_inds_pbc_5` (`gridff_spammm.cl:1358-1376`).  The starting knot is `gi - 2` and the support is 6 voxels wide.

### 3.5 z boundary condition

`z` is **not periodic** (surface → vacuum).  Different evaluators handle it slightly differently:

- `GridFF.cl fe3d_pbc` (`GridFF.cl:118`): return zero if `iz < 1` or `iz >= nz - 2`.
- `Bspline.h fe3d_pbc` (`Bspline.h:637`): return zero if `iz < 2` or `iz >= nz - 3`.
- `Bspline.h fe3d_pbc_comb3` (`Bspline.h:759`): clamp `iz` to `[3, nz - 5]`.

These guards keep the 4-point support inside the allocated grid.  The host must upload a grid that is thick enough in `z` for the interpolation stencil.

### 3.6 Folded-basis coefficient layout

Two modes are supported by `surface_spammm.cl`:

- **Typed mode** (`folded_meta[1] = ntypes > 0`): `folded_coeffs[ntypes * nbasis]` flat.  Index for atom type `it` and basis `ib` is `it * nbasis + ib`.
- **Factorized PLQH mode** (`folded_meta[1] = -1`): `folded_coeffs[nbasis, 4]` stores `float4` `(Pauli, London, Coulomb, H-bond)` per basis.  At runtime the atom's `PLQH` is dotted in.

Padding constants from `RigidBodyDynamics.py:init_folded`:

```
FOLDED_BASIS_MAX = 128
FOLDED_TYPES_MAX = 8
```

### 3.7 Local-memory tiling

- `make_MorseFF` / `make_Coulomb_points`: `LATOMS[32]`, `LCLJS[32]` — tile of 32 substrate atoms loaded once and reused across PBC image loops.
- `getSurfFolded_workgroup`: `LCOEFFS[8*64]`, `LPARAMS_XY[4]`, `LPARAMS_Z[4]`, plus per-atom `LBX/LdBX/LBY/LdBY/LBZ/LdBZ` arrays. 64-thread workgroup, one thread per atom.  Trig/exp are computed once, stored to `__local`, then the O(N_basis) triple loop streams from local memory.

---

## 4. Basis-set and hierarchical / recursive evaluation

### 4.1 GridFF B-spline hierarchy

The 3-D field is represented by tensor-product cubic B-spline coefficients.  Evaluation is a **separable 1-D convolution hierarchy**:

1. Compute `b(t)` and `db(t)` for `t = tx, ty, tz`.
2. Interpolate along z for each `(x, y)` in the 4×4×4 support.
3. Interpolate along y for each `x`.
4. Interpolate along x to get the final value and gradient.

The PBC index tables (§3.3) are the "recursive" reuse part: the same near-boundary patterns are used for every point.

### 4.2 FAF separable basis

The folded basis is a product of three 1-D functions:

```
B(u,v,z) = (x-plane-wave) × (y-plane-wave) × (z-decay)
```

- `x`: `cos(2π ku u)`
- `y`: `cos(2π kv v)`
- `z`: `exp(-kz (z - z0))` or polynomial power `t^m` (`getSurfFolded_tensor_poly`)

The `getSurfFolded_workgroup` kernel precomputes all 1-D factors into `__local` memory, then assembles the 64 (or more) tensor-product terms by pure multiplication.  This reduces the per-basis trig count from `O(N_basis)` to `O(Nx + Ny + Nz)`.

### 4.3 Complex recurrence for 2-D plane waves (FAF and Ewald2D)

Instead of `Nx*Ny` `cos/sin` calls per atom, the fast FAF and Ewald2D kernels compute only two base complex phases:

```
zu = exp(i 2π u)   = (cos(2π u), sin(2π u))
zv = exp(i 2π v)   = (cos(2π v), sin(2π v))
```

Then for integer harmonics `(ku, kv) = (h, k)`:

```
cos(2π h u) = Re(zu^h)
cos(2π k v) = Re(zv^k)
sin(2π h u) = Im(zu^h)
```

The powers are built by repeated `float2` complex multiplication (`cmul`):

```
float2 cmul(float2 a, float2 b) {
    return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
```

For Ewald2D the same idea is applied to the reciprocal-lattice base phases `exp(i b1·ρ)` and `exp(i b2·ρ)`; all `G = h b1 + k b2` phases come from `z1_b1^h * z1_b2^k`.

### 4.4 Polynomial z-basis (`getSurfFolded_tensor_poly`)

For a compact z-basis, the polynomial variant uses:

```
t = 1.0 - clamp((z - zmin)/zcut, 0.0, 1.0)
B_z(iz) = t^{m_start + iz}
dB_z(iz) = -(m_start + iz)/zcut * t^{m_start + iz - 1}
```

The powers `t^p` are updated by `tpow *= t` each iteration, avoiding per-`iz` `pow` calls.

### 4.5 Factorized PLQH contraction

In factorized mode the folded energy is a polynomial in the shared basis product `B = bx * by * bz`:

```
E(B) = cCoulomb * B + cLondon * B^2 + cPauli * B^3
```

This is evaluated by Horner-like accumulation (surface_spammm.cl:856-862):

```
E    += c.x*B^3 + c.y*B^2 + c.z*B + c.w   (H-bond)
dE_dB = c.z + B*(2*c.y + B*3*c.x)
```

The single derivative `dE/dB` is then distributed to `dE/du`, `dE/dv`, `dE/dz` by the chain rule:

```
dEdu = dE_dB * dB/du
dEdv = dE_dB * dB/dv
dEdz = dE_dB * dB/dz
```

This is the key to making the O(N_basis) contraction cheap: only one `float4` dot / polynomial per basis, and the expensive trig lives outside the triple loop.

---

## 5. Critical performance design decisions

1. **One packed PLQ(H) grid instead of three separate grids.**
   Pauli, London, Coulomb (and H-bond) are packed into one `float4` per voxel.  The atom's mixing vector is dotted at the 1-D interpolation level; the rest of the tricubic blending is scalar.

2. **PBC index tables eliminate runtime modulo.**
   `make_inds_pbc` / `choose_inds_pbc` tables are precomputed per workgroup, removing branches from the hot interpolation loop.

3. **Folded-basis 1-D factor precomputation.**
   `getSurfFolded_workgroup` moves `native_cos`/`native_exp` out of the triple loop.  The hot loop is only FMAs and coefficient fetches.

4. **Complex recurrence for 2-D plane waves.**
   Two `sincos` per atom build all harmonics; Ewald2D uses the same trick for all reciprocal `G` vectors.

5. **Local-memory tiling of heavy pairwise kernels.**
   `make_MorseFF`, `make_Coulomb_points`, `getSurfMorse`, and `getSurfFolded_workgroup` preload 32- or 64-atom/coefficient tiles into `__local` and reuse them across the inner PBC image or basis loops.

6. **Kahan summation in `make_Coulomb_points`.**
   `float4` Coulomb sums use a Kahan accumulator to preserve accuracy over many small PBC contributions.

7. **Texture build path (`make_GridFF`).**
   `gridff_spammm.cl:1951` writes the PLQ fields to `image3d_t` textures, which is the reference path for texture sampling in `sampleGridFF` (`gridff_spammm.cl:1865`).

8. **Lazy file caching of fitted grids.**
   `GridFF.h:tryLoad_new` checks for precomputed `Bspline_PLQd.npy` and only refits if missing.  Fits are expensive; caching is essential.

9. **Coulomb via 2D Ewald, not brute PBC, when possible.**
   For 2-D periodic slabs, Ewald2D is exact in the infinite lateral limit and much faster than brute image summation.  The same functional form (`cos(ku u) cos(kv v) exp(-|G| z)`) can be absorbed into the FAF fit.

---

## 6. Rust harness design (target)

The harness lives in the Rust workspace.  Following the `molff` / `molff-ocl` pattern, the target is `crates/libs/surfff-ocl`:

- `surfff` (`crates/libs/surfff`) keeps the CPU reference `SurfaceFolded` and the separable-basis math.
- `surfff-ocl` loads the copied kernels, manages OpenCL buffers, and dispatches builder/evaluator kernels.

### 6.1 Kernel concatenation order

For GridFF builder + evaluator:

```
common.cl + Forces.cl + gridff_spammm.cl
```

For surface/FAF + Ewald2D:

```
common.cl + Forces.cl + surface_spammm.cl
```

`common.cl` and `Forces.cl` are already in `opencl/`.  The SPAMMM kernels depend on the symbols defined there (`cl_Mat3`, `COULOMB_CONST`, `modulo`, `getMorsePLQH`, ...).

### 6.2 Host-side data flow

**GridFF:**
1. Upload substrate `REQ` parameters and positions (`float4 apos`, `float4 REQs`).
2. Allocate `FE_Paul`, `FE_Lond`, `FE_Coul` grids (or one `Bspline_PLQ` grid).
3. Run `make_MorseFF` / `make_Coulomb_points` or `make_GridFF`.
4. Run `BsplineConv3D` or host FFT to fit B-spline coefficients.
5. Upload `Bspline_PLQ` and run `sample3D_comb` for each probe.

**FAF:**
1. Upload `lvec2d`, `folded_params`, `folded_coeffs`, and `PLQH` per atom.
2. Run `getSurfFolded_workgroup` for all atoms (one workgroup per batch).
3. Optionally run `compute_ewald_coefficients` first if fitting the Coulomb channel from Ewald2D.

### 6.3 Memory layout the Rust harness must respect

- `ng` shape as `int4(nx, ny, nz, nxyz)`.
- Flat index `ix + nx * (iy + ny * iz)` for reference grids.
- Z-fastest `iz + nz * (iy + ny * ix)` for the packed `Bspline_PLQ`.
- PBC index tables `iqs[4]` for cubic B-spline; `iqs[6]` for quintic.
- Folded coefficients typed `(ntypes, nbasis)` or factorized `(nbasis, 4)`.
- `PLQH` per atom as `float4`.
- Ewald2D reciprocal vector list `(h, k, Gn)` and per-ion `w[g,i]`.

---

## 7. Open issues / caveats

1. `surface_spammm.cl:folded_eval_grad` has a reported `dudy ↔ dvdx` swap for non-orthogonal 2-D lattices.  Use the tensor kernels (`getSurfFolded_tensor_exp` / `_poly`) or the fixed `rigid.cl` version for production.
2. `surface_spammm.cl:addDipoleField` may have an off-by-one (`>` vs `>=`) at the last grid point.
3. Float32 Ewald2D accumulators (`eval_potential_vacuum` / `eval_potential_full`) may show ~1e-4 RMSE; use `eval_potential_cluster` (double-single) for validation.
4. The FireCore `Surface.cl` in `opencl/Surface.cl` is missing the trailing `Hybrid Potential` section (lines 983-1106 in the original).  If the hard-core hybrid grid is needed, either restore from FireCore or use the SPAMMM `contact_surface.cl`.
5. GridFF `GridFF.cl` (FireCore) and `gridff_spammm.cl` (SPAMMM) are both present.  They have overlapping but not identical kernels.  The FireCore version is the canonical tricubic B-spline; the SPAMMM version adds texture build/sampling and the `sampleGridFF_Bspline_points` path.  Decide per use-case which to load.
