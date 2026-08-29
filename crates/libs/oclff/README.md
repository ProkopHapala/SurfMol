---
type: folder
title: oclff
description: OpenCL GPU harness for forcefields — UFF, SPFFsp3, RAFF/RRsp3, GridFF, FAF. Consolidated from molff-ocl + surfff-ocl. CPU references in molff/surfff remain authoritative for correctness.
tags: [opencl, gpu, forcefield, uff, spff, raff, rrsp3, gridff, faf, harness]
timestamp: 2026-08-29
---

# oclff — OpenCL GPU Forcefield Harness

OpenCL GPU acceleration for all SurfMol forcefields. CPU references in `molff` and `surfff` remain authoritative for correctness — GPU must match CPU within f32 tolerance.

## Modules

| Module | File | GPU kernel source | CPU reference | Status |
|--------|------|-------------------|---------------|--------|
| **rrsp3** | `src/rrsp3.rs` | `opencl/RRsp3.cl` (14 kernels) | `molff::raff` | **active** — 5 port kernel variants, step_cluster + step_dynamics, CPU↔GPU parity verified |
| **pack** | `src/pack.rs` | (host-side) | — | **active** — cluster-sorted GPU layout, topology helpers |
| **uff** | `src/uff.rs` | `opencl/UFF.cl` | `molff::uff::Uff` | **partial** — bond eval only, parity test passes |
| **spff** | `src/spff.rs` | `opencl/SPFF.cl` (+common+Forces) | SPAMMM (Rust ref TODO) | **stub** — compile smoke test only |
| **surfff** | `src/surfff.rs` | `opencl/gridff_*.cl`, `opencl/faf_*.cl` | `surfff::SurfaceFolded` | **infrastructure** — ClAssembler wired, kernels not dispatched |
| **assemble** | `src/assemble.rs` | (host-side) | SPAMMM `OpenCLBase` | **active** — macro/fragment composition for GridFF/FAF |

## RAFF/RRsp3 GPU Implementation

The RRsp3 harness (`RRsp3` struct) wraps the OpenCL rigid-atom port forcefield. It is a direct port of FireCore `pyBall/RigidAtomFF/RRsp3/RRsp3.py`.

### What it can do

- **5 port kernel variants** selectable at runtime via `PortKernel` enum:
  - `Current` — massfull XPBD with physical inertia (quaternion DOF + torque → δθ → q)
  - `Orig` — original massfull solver (same physics, no `rot_mass_scale` tuning)
  - `Substep` — massless Newton-Raphson in ω-space, iterative substep alignment
  - `Shapematch` — massless Kabsch/polar-decomposition orientation solve
  - `Eigen` — massless Davenport q-method eigenproblem (Horn quaternion eigen)
- **Two step modes**:
  - `step_cluster` — relaxation step: bboxes → topology → collision → ports → corrections
  - `step_dynamics` — MD step: predict → collision → ports → corrections → update_velocities
- **14 OpenCL kernels** in `RRsp3.cl`: predict_dynamics, update_velocities_dynamics, update_bboxes_rigid, build_local_topology_rigid, compute_collision_cluster_rigid, 5 port kernel variants, compute_tips, compute_optimal_rotation_eigen, apply_corrections_rigid_ports
- **Cluster-sorted GPU layout**: molecules packed into workgroups (group_size=64), ghost atoms for cross-group neighbors, local topology rebuilt each step from AABBs
- **Host-side data prep** (`pack.rs`): `pack_molecules`, `build_neighs_from_bonds`, `make_exclusions_1st_2nd`, `make_bk_slots_clustered`, `make_ports_from_neighs`, `masses_from_elems`

### CPU↔GPU parity verified

Two apples-to-apples comparisons (in `raff_ocl_smoke --parity`):

| Variant | CPU | GPU | Kabsch RMSD | Status |
|---------|-----|-----|-------------|--------|
| Memoryless | Adiabatic (Wahba from scratch) | Shapematch (Kabsch from scratch) | 0.0096 Å | PASS |
| Massfull | Dynamic (quaternion + inertia) | Current (quaternion + inertia) | 0.000000 Å | PASS |

### Binaries

| Binary | Description | Usage |
|--------|-------------|-------|
| `raff_ocl_smoke` | 2× water smoke test: single step, perturbed relaxation, optional CPU↔GPU parity | `cargo run -p oclff --bin raff_ocl_smoke [--port current\|shapematch] [--parity] [--traj PATH] [--tsv PATH]` |
| `raff_ocl_xylitol` | 4× xylitol relaxation: force-based convergence, trajectory + convergence TSV | `cargo run -p oclff --bin raff_ocl_xylitol [--port current\|shapematch] [--n_copies N]` |

Debug artifacts saved to `debug/raff_ocl_smoke/` and `debug/raff_ocl_xylitol/` (see `notes/conventions/relaxation_convergence.md` for methodology).

### Tests

| Test file | What it tests |
|-----------|---------------|
| `tests/test_uff_cl.rs` | UFF OpenCL bond eval parity vs `molff::uff::Uff` CPU (H₂) |
| `tests/test_spff_cl.rs` | SPFFsp3 OpenCL compile smoke test |
| `tests/test_assemble_fragments.rs` | ClAssembler parse + inject: 21 functions from gridff_build.cl, 7 macros from gridff_eval.cl, etc. |

## Open issues

- [ ] **1-4 exclusions missing**: `make_exclusions_1st_2nd` only excludes 1st and 2nd neighbors. Xylitol has 1-4 atom pairs within collision radius (2.0 Å) that are not excluded, causing collision forces to fight port springs. Workaround: set `radius=0` to disable collisions. Fix: add 1-4 exclusion list.
- [ ] **`k_coll` not used by collision kernel**: `compute_collision_cluster_rigid` uses geometric overlap (rsum = ri + rj), not `k_coll`. Setting `k_coll=0` has no effect. To disable collisions, set `radius=0`.
- [ ] **`faf_eval.cl` is a placeholder**: empty file — GridFF/FAF eval macros not yet extracted from `surface_spammm.cl`. `FafEvalOcl::new()` will fail.
- [ ] **SPFF harness is a stub**: `SpffOcl` compiles the kernel program but does not dispatch any kernels. Needs force evaluation + parity vs reference.
- [ ] **UFF harness is partial**: `UffOcl::eval_bonds` works (parity tested) but angles/dihedrals/inversions not dispatched.
- [ ] **No inter-molecular non-bonded on GPU**: the collision kernel handles sphere-sphere overlap but there is no LJ/Coulomb non-bonded kernel. CPU `eval_nonbonded` is the only option.
- [ ] **No f64 support**: all GPU kernels are f32. CPU reference is f64. Parity tolerance must account for f32 precision limits (especially for large molecules with many coupled nodes — xylitol plateaus at ~2e-5 force).
- [ ] **`step_dynamics` not tested**: only `step_cluster` (relaxation) is tested. The MD path (predict → velocities) has no parity test.
- [ ] **No multi-molecule ghost atom test**: smoke tests use molecules far apart (no ghost atoms needed). Ghost atom path (cross-workgroup neighbors) is untested with actual overlapping bboxes.
- [ ] **Port kernel `Eigen` not parity-tested**: only `Current` and `Shapematch` have CPU↔GPU parity. `Orig`, `Substep`, `Eigen` compile and run but no parity comparison.
- [ ] **No performance benchmark**: no GPU vs CPU wall-time comparison. Smoke tests verify correctness only.
- [x] **CPU↔GPU parity (memoryless)**: Adiabatic vs Shapematch, RMSD=0.0096 Å — PASS
- [x] **CPU↔GPU parity (massfull)**: Dynamic vs Current, RMSD=0.000000 Å — PASS
- [x] **Force-based convergence**: relaxation runs until max|correction| < threshold, final geometry as reference (see `notes/conventions/relaxation_convergence.md`)
- [x] **NVIDIA GPU selection**: `nvidia_proque()` explicitly selects NVIDIA platform, never PoCL/CPU
- [x] **Kernel embedded at compile time**: `include_str!("../../../../opencl/RRsp3.cl")` — no runtime file loading

## API reference

### `RRsp3` struct

```rust
pub struct RRsp3 {
    pub natoms: usize,
    pub group_size: usize,
    pub num_groups: usize,
    pub max_ghosts: usize,
    pub nnode_per_group: i32,
    pub nnode_tot: usize,
}
```

### Key methods

```rust
// Constructor — creates OpenCL context on NVIDIA GPU
RRsp3::new(natoms, group_size, max_ghosts, prefer_gpu) -> ocl::Result<Self>

// Upload
upload_state(&mut self, pos3: &[[f32;3]], inv_mass: &[f32], quat: Option<&[[f32;4]]>)
upload_radius(&mut self, radius: &[f32])
upload_neighs_and_exclusions(&mut self, neighs, excl1, excl2)
upload_cluster_ports(&mut self, port_local, kflat, nnode_per_group)
upload_bk_slots(&mut self, bk_slots)

// Step
step_cluster(&mut self, port_kernel: PortKernel, cfg: &StepConfig)  // relaxation
step_dynamics(&mut self, port_kernel: PortKernel, cfg: &StepConfig) // MD

// Download
download_pos_quat() -> (Vec<[f32;4]>, Vec<[f32;4]>)
download_dpos_coll() -> Vec<[f32;4]>   // collision corrections (force proxy)
download_dpos_node() -> Vec<[f32;4]>   // port corrections (force proxy)
```

### `PortKernel` enum

```rust
pub enum PortKernel { Current, Orig, Substep, Shapematch, Eigen }
// from_str: "current"|"rigid", "orig"|"original", "substep", "shapematch"|"shape_match", "eigen"|"q_eigen"
// is_massless(): Shapematch, Eigen, Substep = true; Current, Orig = false
```

### `StepConfig`

```rust
pub struct StepConfig {
    pub dt: f32,              // timestep
    pub k_coll: f32,          // collision stiffness (currently unused by kernel — see open issues)
    pub relaxation: f32,      // Jacobi relaxation factor (0-1)
    pub bbox_margin: f32,     // AABB expansion for neighbor search
    pub momentum_beta: f32,   // heavy-ball momentum (0 = disabled)
    pub rot_mass_scale: f32,  // rotational mass scaling (Current kernel)
    pub n_rot_substeps: i32,  // rotation substeps (Substep kernel)
    pub rot_eps: f32,         // convergence eps (Substep kernel)
    pub theta_max: f32,       // max rotation angle (Substep kernel)
    pub damp: f32,            // velocity damping (dynamics mode)
}
```

## See also

- [`/doc/topical_audit/raff.md`](/doc/topical_audit/raff.md) — RAFF cross-implementation map (CPU vs GPU vs FireCore)
- [`/opencl/README.md`](/opencl/README.md) — OpenCL kernel inventory and conventions
- [`/notes/conventions/relaxation_convergence.md`](/notes/conventions/relaxation_convergence.md) — relaxation convergence methodology
- [`/userguide/raff.md`](/userguide/raff.md) — RAFF solver modes end-user guide (CPU editor)
- `opencl/RRsp3.cl` — the OpenCL kernel source (1748 lines, 14 kernels)
- FireCore `pyBall/RigidAtomFF/RRsp3/RRsp3.py` — original Python harness (parity reference)
- FireCore `pyBall/RigidAtomFF/RRsp3/test_RRsp3_smoke.py` — original smoke test (ported as `raff_ocl_smoke`)
