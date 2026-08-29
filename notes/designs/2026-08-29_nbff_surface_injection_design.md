# Non-Bonded Kernel + Surface Injection: Macro-Assembler Design

## Question

How do we integrate GridFF and FAF surface interactions into the non-bonded (NB)
kernels for UFF and SPFF, without writing N×M separate kernel variants?

## Findings from FireCore

### The N×M explosion in FireCore

FireCore's `cpp/common_resources/cl/` has these `getNonBond*` kernels hand-written:

| Kernel | Pairwise | Exclusion | Surface | File:line |
|--------|----------|-----------|---------|-----------|
| `getNonBond` | LJQH | neighs4 | none | UFF.cl:1023 |
| `getNonBond_ex2` | LJQH | excl-list | none | UFF.cl:1208 |
| `getNonBond_GridFF` | LJQH | neighs4 | GridFF-tex | relax_multi.cl:2413 |
| `getNonBond_GridFF_Bspline` | LJQH | neighs4 | GridFF-Bspline | UFF.cl:1523 |
| `getNonBond_GridFF_Bspline_ex2` | LJQH | excl-list | GridFF-Bspline | UFF.cl:1721 |
| `getNonBond_GridFF_Bspline_tex` | LJQH | neighs4 | GridFF-tex | relax_multi.cl:3573 |

Plus standalone surface-only kernels (no NB loop):

| Kernel | Surface type | File:line |
|--------|-------------|-----------|
| `getSurfMorse` | Morse | Surface.cl:334 |
| `getSurfFolded` | FAF | Surface.cl:432 |
| `getSurfFolded_workgroup` | FAF (wg-opt) | Surface.cl:512 |
| `getSurfFolded_harmonics` | FAF (harmonic) | Surface.cl:656 |

Adding FAF injection to the NB loop in the FireCore style would require 2 more
hand-written kernels: `getNonBond_FAF` and `getNonBond_FAF_ex2`. Adding SPFF
pairwise variant would double it again. This is the combinatoric explosion.

### Key structural insight

Examining `getNonBond_GridFF_Bspline` (UFF.cl:1523-1717), the kernel has **three
orthogonal, decoupled axes**:

```
__kernel void getNonBond_GridFF_Bspline(...) {
    // === AXIS 1+2: NB loop (pairwise potential + exclusion strategy) ===
    if(ns.w>=0) {  // insulate nbff
        for (j0 chunks) {
            for (jl) {
                // pairwise: getLJQH(dp, REQK, R2damp)   <-- AXIS 1
                // exclusion: neighs4 check              <-- AXIS 2
                // PBC loop
            }
        }
    }

    if(iG>=natoms) return;

    // === AXIS 3: Surface injection (completely decoupled from NB loop) ===
    { // insulate gridff
        // sample Bspline grid at posi using fe3d_pbc_comb()
        // fe += fg
    }

    forces[iaa] += fe;
}
```

**The surface block only needs `posi`, `REQKi`, and `fe` (the accumulator).**
It does not touch the NB loop variables. This is true for all three surface
types (GridFF-Bspline, GridFF-tex, FAF).

### UFF vs SPFF: same NB kernel

Both UFF and SPFF share the **same** `getNonBond` kernel for non-bonded
interactions — both call `getLJQH(dp, REQK, R2damp)`. The difference between
UFF and SPFF is only in the **bonded** kernels (`getUFFf4` vs `getSPFFf4`):
bonds, angles, pi-pi, pi-sigma. The NB loop is identical.

SPFF's `getSPFFf4` (SPFF.cl:323) does have a `getLJQH` call at line 491, but
that is the "subtractVdW" 1-3 correction inside the angle loop, not the main
NB loop.

### Existing eval fragments in SurfMol

The split fragments already expose the low-level inline functions:

- `gridff_eval.cl`: `fe3d_pbc_comb(u, n, BsplinePLQ, PLQH, xqs, yqs)` (line 234)
- `faf_eval.cl`: `folded_eval_basis(u, v, z, prm)` + `folded_eval_grad(...)` (lines 112, 134)

But they do **not** yet have high-level injection macros that wrap the setup
(local memory, grid coord transform, PLQH prefactor) into a single
`SURF_INJECT_*(posi, REQKi, fe)` call.

## Proposed Design: 3-axis macro assembler

### Three macro slots

**Axis 1 — `NB_PAIR_FORCE(dp, REQK, R2damp)`** (pairwise potential):
- `NB_PAIR_LJQH` → `getLJQH(dp, REQK, R2damp)`  (UFF + SPFF shared)
- Future: `NB_PAIR_MORSE`, `NB_PAIR_BUCK` etc.

**Axis 2 — `NB_EXCLUDE(ja, ng, ngC, ipbc, bBonded)`** (exclusion strategy):
- `NB_EXCL_NEIGHS4` → 4-neighbor int4 check (UFF.cl style)
- `NB_EXCL_LIST` → packed sorted exclusion list (ex2 style)

**Axis 3 — `SURF_INJECT(posi, REQKi, fe)`** (surface/grid injection):
- `SURF_NONE` → empty
- `SURF_GRIDFF_BSPLINE` → setup + `fe3d_pbc_comb()` + `fe += fg`
- `SURF_GRIDFF_TEX` → texture sampling variant
- `SURF_FAF` → `folded_eval_basis/grad` loop + `fe += fg`
- `SURF_MORSE` → Morse surface

### Template kernel

```c
// getNonBond_generic.cl — assembled by ClAssembler
__kernel void getNonBond_generic(
    // common args
    const int4 ns,
    __global float4* atoms, __global float4* forces,
    __global float4* REQKs,
    NB_EXCL_ARGS,           // macro-expanded exclusion args
    __global cl_Mat3* lvecs, const int4 nPBC, const float4 GFFParams,
    SURF_ARGS                // macro-expanded surface args (can be empty)
){
    // ... setup ...
    float4 fe = float4Zero;

    // === NB loop ===
    for (j0 chunks) {
        // load local memory
        for (jl) {
            if (NB_EXCLUDE(...)) {
                fe += NB_PAIR_FORCE(dp, REQK, R2damp);
            }
        }
    }

    if(iG>=natoms) return;

    // === Surface injection ===
    SURF_INJECT(posi, REQKi, fe);

    forces[iaa] += fe;
}
```

### Variants generated (N+M instead of N×M)

Fragments needed: 1 template + 1 pairwise + 2 exclusion + 4 surface = **8 fragments**
Variants generated: 1 × 1 × 2 × 4 = **8 kernels** (vs 6+ hand-written in FireCore, and growing)

New combinations (e.g. SPFF+FAF, UFF+GridFF+FAF) are just a different assembly
recipe — no new kernel code.

### Surface injection macro shape

Each `SURF_INJECT_*` macro expands to a self-contained block:

```c
// gridff_eval.cl — add at bottom
#define SURF_ARGS_GRIDFF_BSPLINE  \
    __global float4* BsplinePLQ, const int4 grid_ns, \
    const float4 grid_invStep, const float4 grid_p0

#define SURF_INJECT_GRIDFF_BSPLINE(posi, REQKi, fe)  {  \
    __local int4 xqs[4]; __local int4 yqs[4];           \
    /* cooperative load of pbc indices */               \
    if(iL<4) xqs[iL]=make_inds_pbc(grid_ns.x,iL);      \
    else if(iL<8){ int i=iL-4; yqs[i]=make_inds_pbc(grid_ns.y,i);} \
    barrier(CLK_LOCAL_MEM_FENCE);                       \
    const float ej = exp(GFFParams.y*REQKi.x);          \
    const float4 PLQH = {ej*ej*REQKi.y, ej*REQKi.y, REQKi.z, 0}; \
    const float3 u = (posi - grid_p0.xyz)*grid_invStep.xyz; \
    float4 fg = fe3d_pbc_comb(u, grid_ns.xyz, BsplinePLQ, PLQH, xqs, yqs); \
    fg.xyz *= -grid_invStep.xyz;                        \
    fe += fg;                                           \
}
```

```c
// faf_eval.cl — add at bottom
#define SURF_ARGS_FAF  \
    __global float* folded_coeffs, __global float4* folded_kxyz, \
    __global int* folded_atom_type, const int4 folded_meta, \
    const float4 folded_lvec2d

#define SURF_INJECT_FAF(posi, REQKi, fe)  {             \
    /* local memory load of basis + coeffs */           \
    /* ... see getSurfFolded (Surface.cl:432) ... */    \
    /* u,v from posi via invLvec2d */                   \
    /* loop over nbasis: E += c*b; F -= c*g */          \
    fe += (float4)(F.x, F.y, F.z, -E);                  \
}
```

### Argument-list challenge

OpenCL kernels have flat argument lists. Different surface injectables need
different extra args. Two approaches:

**A. Full union arg list** — always declare all possible surface args; unused
ones are dummy buffers. Simple, but wastes a few arg slots.

**B. Macro-expanded arg list** — `SURF_ARGS_*` macro injects only the needed
args. Cleaner, but the Rust harness must set args in the exact order the
assembled kernel expects (the assembler must report the arg list).

Recommend **B** — the `ClAssembler` already tracks fragments; it can also emit
the arg list. The Rust harness uses the assembler's output to set args.

## Next steps

1. Add `SURF_INJECT_*` / `SURF_ARGS_*` macros to `gridff_eval.cl` and `faf_eval.cl`
2. Write `getNonBond_generic.cl` template with `NB_PAIR_FORCE`, `NB_EXCLUDE`, `SURF_INJECT` slots
3. Add `NB_EXCL_NEIGHS4` / `NB_EXCL_LIST` macros to a new `nb_common.cl`
4. Extend `ClAssembler` to assemble NB+surface variants and report arg lists
5. Wire `oclff` harness to call assembled NB kernels
6. Parity test: assembled `getNonBond_uff_gridff` vs FireCore `getNonBond_GridFF_Bspline`
