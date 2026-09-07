---
type: task
title: "GPU RAFF Performance Benchmark — Multi-Replica Stress Test"
description: Plan for measuring actual GPU throughput of the RRsp3 RAFF harness with thousands of independent replicas. Covers the 2D (iatom, isystem) indexing strategy, workgroup padding, broad-phase replacement, and benchmark methodology. Reference: FireCore xylitol/NaCl/GridFF 5000-replica throughput test.
tags: [gpu, opencl, raff, rrsp3, benchmark, throughput, multi-replica, stress-test, performance, workgroup, padding, broad-phase]
timestamp: 2026-08-29
---

# GPU RAFF Performance Benchmark — Multi-Replica Stress Test

## Goal

Measure the **actual GPU throughput** of the RRsp3 RAFF harness on a realistically-sized system: thousands of independent molecular replicas relaxed in parallel on the GPU. This is the use case for free-energy sampling, global optimization, and batch surface-scanning — not a single large molecule, but many small ones processed simultaneously.

### Why thousands of atoms?

A modern NVIDIA GPU has ~4000–10000 threads resident simultaneously (e.g. RTX 3060 has 3584 CUDA cores, 48 SMs × 64 warps). To saturate the GPU:

- Each workgroup has `GROUP_SIZE` threads (32 or 64).
- We need enough workgroups to fill all SMs many times over (occupancy ~50–100 workgroups per SM for good latency hiding).
- For `GROUP_SIZE=64` and 48 SMs: need `48 × 8 = 384` workgroups minimum for full occupancy, but `48 × 32 = 1536` workgroups for good latency hiding.
- With xylitol (21 atoms, padded to 64 per workgroup): **1000–5000 replicas** = 1000–5000 workgroups = 64000–320000 atoms total. This is the target scale.

### FireCore reference benchmark

FireCore's throughput test (`tests/tMMFFmulti/run_throughput_MD.py`) runs:
- **6000 replicas** of xylitol (default `--nSys 6000`)
- On NaCl surface (`--surf_name data/xyz/surfaces_for_throughput/NaCl_NxN_L3`)
- With GridFF surface potential (`--GridFF 1`)
- 10000 MD loop iterations
- Measures wall-clock time per iteration

We will not reproduce the GridFF surface part yet (no GPU GridFF in SurfMol), but the **molecular relaxation throughput** (port forces + collisions + corrections) is the same and is what we want to measure.

## Background: Current vs Required Architecture

### Current RRsp3 harness (1D, one molecule per workgroup)

The current `oclff::rrsp3::RRsp3` harness uses a **1D global index**:

```
global_size = natoms,  local_size = GROUP_SIZE
num_groups = natoms / GROUP_SIZE
```

Each workgroup = one molecule. The workgroup ID `grp = get_group_id(0)` identifies the molecule. Atom index within the molecule = `lid = get_local_id(0)`.

**This works for a few molecules** (smoke tests: 2×water, 4×xylitol) but has two problems at scale:

1. **Quadratic broad phase**: `build_local_topology_rigid` loops over ALL other workgroups to find ghost atoms:
   ```c
   for (int other_g = 0; other_g < num_groups; other_g++) {
       // test AABB overlap with every other group
   }
   ```
   With 5000 workgroups, this is 25M AABB tests per step — **completely dominates runtime**.

2. **No system dimension**: all workgroups are in one 1D pool. There's no way to distinguish "replica 42 of xylitol" from "a different molecule". The kernel treats all workgroups as potential collision partners.

### Required: 2D indexing (iatom, isystem) — the FireCore pattern

FireCore's `relax_multi.cl` uses a **2D NDRange**:

```c
const int iG = get_global_id(0);   // atom index within system  (0..natoms-1)
const int iS = get_global_id(1);   // system/replica index      (0..nSys-1)
const int i0a = iS * natoms;       // offset to this replica's atoms
const int iaa = iG + i0a;          // global flat index
```

Launch: `global_size = (natoms, nSys)`, `local_size = (GROUP_SIZE, 1)`.

**Key advantage**: replicas are **completely independent** — no cross-replica neighbor lookups, no ghost atoms between replicas. Each workgroup processes one replica (or one cluster within a replica). The GPU schedules workgroups across SMs freely, maximizing occupancy.

### Independent replicas vs interacting molecules

There are two use cases for multi-molecule GPU simulation:

| Use case | Replicas interact? | Ghost atoms needed? | Example |
|----------|-------------------|---------------------|---------|
| **Independent replicas** (free-energy sampling, GOpt, batch scan) | NO | NO — each replica is isolated | 5000× xylitol on surface, each at different (x,y,θ) |
| **Interacting molecules** (condensed phase, multi-molecule MD) | YES | YES — cross-workgroup neighbors | 100 molecules in a box, colliding |

**For this benchmark, we focus on independent replicas.** This is the simpler case (no cross-replica ghosts) and the most common use case for the applications mentioned. Interacting molecules require a broad-phase acceleration structure (bucket/cell list) and are a separate task.

## Design: Multi-Replica RRsp3 Harness

### Step 1: 2D kernel indexing

Modify the RRsp3 OpenCL kernels (or create a multi-replica variant `RRsp3_multi.cl`) to accept a 2D NDRange:

```c
__kernel void step_cluster_multi(
    const int natoms_per_sys,   // atoms per replica (including padding)
    const int nnode_per_sys,    // nodes per replica
    // ... all buffers are now [nSys * natoms_per_sys] or [nSys * nnode_per_sys] ...
)
{
    const int iG = get_global_id(0);   // atom within replica (0..GROUP_SIZE-1 for one-WG-per-replica)
    const int iS = get_global_id(1);   // replica index
    const int i0 = iS * natoms_per_sys;  // offset to this replica's data
    // ... all array accesses use i0 + iG ...
}
```

**For independent replicas with one workgroup per replica:**
- `global_size = (GROUP_SIZE, nSys)`
- `local_size = (GROUP_SIZE, 1)`
- Each workgroup = one replica. `grp = get_group_id(1)` = replica index.
- No cross-replica ghost atoms needed → `build_local_topology_rigid` can be **skipped entirely** (or reduced to a no-op) since all neighbors are within the same workgroup.

**This is a massive simplification**: the entire ghost atom machinery (AABB overlap, ghost list, global→local mapping) is unnecessary when replicas are independent. The neighbor list is built once on the host and reused for all replicas (same molecule = same topology).

### Step 2: Shared topology, per-replica state

For N replicas of the same molecule:

| Buffer | Size | Per-replica? | Notes |
|--------|------|--------------|-------|
| `pos` | `nSys × natoms` | YES | positions differ per replica |
| `quat` | `nSys × nnode` | YES | orientations differ per replica |
| `vel`, `omega` | `nSys × natoms` | YES | velocities differ per replica |
| `neighs` | `natoms × 4` | NO (shared) | same topology for all replicas |
| `excl1`, `excl2` | `natoms × 4` | NO (shared) | same exclusions for all replicas |
| `port_local` | `nnode × 4 × 4` | NO (shared) | same port geometry |
| `kflat` (stiffness) | `nnode × 4` | NO (shared) | same bond stiffness |
| `bk_slots` | `natoms × 4` | NO (shared) | same back-slot mapping |
| `radius` | `natoms` | NO (shared) | same vdW radii |
| `fixmask` | `natoms` | NO (shared) | same constraints |
| `dpos_coll`, `dpos_node`, etc. | `nSys × ...` | YES | per-replica corrections |

**Memory savings**: for 5000 replicas of xylitol (21 atoms, 10 nodes):
- Shared topology: ~21×4×4 + 10×4×4 + 10×4 + 21×4 ≈ 1.3 KB (negligible)
- Per-replica state: 5000 × (21×16 + 10×16 + 21×16) ≈ 5000 × 832 = 4.2 MB (easily fits in GPU memory)

### Step 3: Workgroup size and padding strategy

**The problem**: GPU workgroups have fixed size (32 or 64 on NVIDIA). Molecules rarely have exactly that many atoms.

| Molecule | Atoms | Nodes | GROUP_SIZE=32 | GROUP_SIZE=64 |
|----------|-------|-------|---------------|---------------|
| Water (H₂O) | 3 | 1 | 29 padding (91%) | 61 padding (95%) |
| Methanol (CH₃OH) | 6 | 2 | 26 padding (81%) | 58 padding (91%) |
| Xylitol (C₅H₁₂O₅) | 21 | 10 | 11 padding (34%) | 43 padding (67%) |
| Pentacene (C₂₂H₁₄) | 36 | 22 | — (too big for 32) | 28 padding (44%) |

**Padding strategy** (current approach in `pack.rs`):
- Pad with dummy atoms (`invM=0`, `pos=NaN`) to fill the workgroup.
- Kernels check `p.w > 1e-12f` (invM > 0) to skip padding.
- **Waste**: padded slots consume registers and local memory but do no useful work.

**Alternative: multiple molecules per workgroup** (for small molecules):
- Pack 2–10 water molecules into one 64-thread workgroup.
- Each molecule occupies a contiguous slice of the local memory.
- Needs intra-workgroup molecule boundaries (a `mol_start[igroup]` array).
- More complex but eliminates padding waste for small molecules.

**For this benchmark**: use xylitol (21 atoms) with `GROUP_SIZE=64` — 33% padding is acceptable and matches the FireCore reference. We can explore packing optimization later.

**GROUP_SIZE=32 vs 64**: NVIDIA GPUs have 32-thread warps. `GROUP_SIZE=32` = one warp per workgroup (minimum latency, but less shared memory per workgroup). `GROUP_SIZE=64` = two warps (more shared memory, better latency hiding). The benchmark should test both.

### Step 4: Skip broad phase for independent replicas

Since replicas don't interact, `build_local_topology_rigid` is unnecessary — all neighbors are within the same workgroup. The kernel dispatch simplifies to:

```
step_cluster_multi:
  1. zero_corrections        (per-replica)
  2. compute_collision       (intra-replica only, no ghosts)
  3. compute_ports           (intra-replica only, no ghosts)
  4. apply_corrections       (per-replica)
```

No `update_bboxes`, no `build_local_topology`. The neighbor list is precomputed on the host and shared across all replicas. This eliminates the O(N²) bottleneck entirely.

### Step 5: Collision handling for independent replicas

For independent replicas, collisions are **intra-molecular only** (atoms within the same molecule overlapping due to perturbation). The collision kernel simplifies to:

```c
// No ghost loading needed — all atoms are in the workgroup
__local float4 l_pos[GROUP_SIZE];
l_pos[iG] = pos[i0 + iG];
barrier(CLK_LOCAL_MEM_FENCE);

for (int j = 0; j < natoms_per_sys; j++) {
    if (j == iG) continue;
    if (excluded8(j, excl1[iG], excl2[iG])) continue;
    // sphere-sphere collision between l_pos[iG] and l_pos[j]
}
```

**Note**: for independent replicas on a surface, inter-replica collisions would matter (molecules bumping into each other), but that requires a broad-phase structure and is out of scope for this benchmark.

## Benchmark Methodology

### System sizes

| Configuration | nSys | Atoms/replica | Total atoms | GROUP_SIZE | Workgroups |
|---------------|------|---------------|-------------|------------|------------|
| Small | 64 | 64 (padded xylitol) | 4096 | 64 | 64 |
| Medium | 512 | 64 | 32768 | 64 | 512 |
| Large | 2000 | 64 | 128000 | 64 | 2000 |
| Stress | 5000 | 64 | 320000 | 64 | 5000 |
| Stress-32 | 5000 | 32 (padded) | 160000 | 32 | 5000 |

### What to measure

| Metric | How | Why |
|--------|-----|-----|
| **Wall time per step** | `std::time::Instant` around `step_cluster_multi` | Primary throughput metric |
| **Steps/second** | `n_steps / total_time` | GPU throughput |
| **Atoms/second** | `nSys × natoms × steps/sec` | Normalized throughput |
| **GPU kernel time** | `cl::Event` profiling (if supported) | Breakdown by kernel |
| **Convergence** | max\|dpos\| per step → plot | Verify physics is correct, not just fast |
| **CPU reference time** | Same system on CPU `molff::raff` | Speedup ratio |

### Test protocol

1. **Setup**: Load xylitol, build topology, pack N replicas with random perturbations (0.2–0.5 Å)
2. **Warmup**: 10 GPU steps (not timed) — JIT compilation, buffer allocation
3. **Benchmark**: 100 GPU steps, measure wall time
4. **Convergence check**: run 5000 steps, verify max\|dpos\| decreases monotonically
5. **CPU parity**: run 1 replica on CPU, compare final geometry via Kabsch RMSD
6. **Sweep**: repeat for each system size and GROUP_SIZE

### Output

- `debug/raff_ocl_benchmark/timing.tsv` — columns: `nSys, group_size, n_steps, wall_time_s, steps_per_s, atoms_per_s, cpu_time_s, speedup`
- `debug/raff_ocl_benchmark/convergence_N5000.tsv` — columns: `step, max_dpos, max_force, rms_dpos, rms_force`
- `debug/raff_ocl_benchmark/plot_timing.py` — plot throughput vs nSys
- `debug/raff_ocl_benchmark/plot_convergence.py` — plot convergence for stress test

## Implementation Plan

### Phase 1: Multi-replica kernel (minimal changes)

**Goal**: Get a working multi-replica benchmark with minimal kernel changes.

1. **Extend `RRsp3` struct** in `oclff::rrsp3` — add multi-replica support to the existing harness
   - Add `nsys: usize` field (number of independent replicas)
   - Buffers: `pos[nSys×natoms]`, `quat[nSys×nnode]`, shared topology buffers (neighs, excl, port_local, bk_slots, radius, fixmask — uploaded once)
   - `new(natoms_per_sys, nnode_per_sys, nsys, group_size, max_ghosts)` constructor
   - `upload_state_multi(pos: &[f32], quat: &[f32], inv_mass: &[f32])` — uploads all replicas at once
   - `step_cluster_multi(port_kernel, cfg)` — one step for ALL replicas (2D dispatch)

2. **Modify `RRsp3.cl`** — add `iS = get_global_id(1)` indexing to existing kernels
   - Add `nSys` parameter and `iS` dimension to: `compute_collision`, `compute_ports_*`, `apply_corrections`
   - For independent replicas: skip `update_bboxes` and `build_local_topology` (no ghosts needed — all neighbors intra-workgroup)
   - Use `neighs` directly (identity mapping, no local remapping)
   - The ghost machinery stays in the kernel for the interacting-molecules case but is not dispatched when `nSys > 1` and replicas are independent

3. **Create benchmark binary** `raff_ocl_benchmark.rs` in `oclff/src/bin/`
   - Loads xylitol, builds topology, packs N replicas
   - Random perturbation per replica
   - Runs benchmark protocol (warmup → timed → convergence)
   - Uses `cl.finish()` + wall-time for Phase 1 timing
   - Saves TSV + plots to `debug/raff_ocl_benchmark/`

4. **Run benchmark**, measure throughput, identify bottlenecks

### Phase 2: Optimization (after Phase 1 baseline)

Based on Phase 1 results, optimize. **Each optimization must be benchmarked in isolation — only keep changes that improve throughput without breaking parity.**

#### Kernel fusion — CAUTIOUS, likely NOT worth it

**Concern 1: combinatorial explosion of kernel variants.** Fusing collision + ports + corrections would require a separate fused kernel per port variant (Current, Orig, Substep, Shapematch, Eigen) × collision on/off × dynamics on/off. That's 20+ kernels.

**Solution: `ClAssembler` macro metaprogramming** (`crates/libs/oclff/src/assemble.rs`). The assembler parses `.cl` "libraries" for `//>>>function` / `//>>>macro` blocks and preprocesses templates with `//<<<` sentinels (`//<<<file`, `//<<<macro`, `//<<<function`). This lets us generate only the fused variant we need at build time, without maintaining 20 hand-written kernels. Used by `surfff.rs` and `spff.rs` already. If we do kernel fusion, use this system — do NOT hand-write fused variants.

**Concern 2: register spill and local memory limits.** Larger kernels use more registers and `__local` memory. When registers spill to private memory (off-chip), performance drops sharply. `__local` memory is shared across all resident workgroups — too much reduces occupancy (fewer workgroups resident per SM), killing latency hiding.

- OpenCL requires `__local` arrays declared at the top of the kernel (not in scopes). So `__local` memory from fused collision + ports + corrections is **additive** — `l_pos[GROUP_SIZE]` from collision + `l_pos[GROUP_SIZE]` from ports = 2× the local memory of either alone.
- **Scoping local (private) variables** (not `__local`) into `{}` blocks CAN help the compiler shorten register lifetimes, reducing spill. This is worth doing regardless of fusion.
- **Decision**: do NOT fuse kernels for Phase 2. The 4-kernel-launch overhead is ~0.12ms (launch-bound regime), which is small relative to the 0.39ms compute time at 5000 replicas. Fusion would save ~30% in the launch-bound regime but risk register spill in the compute-bound regime where it matters more. **Profile first with `cl::Event` to confirm launch overhead is actually the bottleneck before attempting fusion.**

#### Ranked optimization priorities

Apply one change at a time and retain it only after direct parity plus timing at both 64 and 5000 replicas.

| Priority | Change | Expected gain | Risk / reason |
|----------|--------|---------------|---------------|
| **P0** | Add direct `RRsp3` ↔ multi-system parity for one identical replica | correctness prerequisite | Current finite/convergent checks do not prove equivalent corrections or geometry |
| **P1 — implemented** | Cache persistent `ocl::Kernel` objects instead of rebuilding them every step | optimized 64-replica path is ~10× faster than original combined with zero-launch removal | scalar step parameters are updated with `set_arg`; saturated 5000-replica timing is unchanged |
| **P1 — implemented** | Remove redundant `zero_corrections_multi` for Current mode | +19% at 5000 replicas with WG32 (3.69×10⁸ → 4.39×10⁸ atom-steps/s) | safe while collision and Current ports overwrite every consumed correction slot; accumulating variants may still need zeroing |
| **P1 — implemented/configured** | Benchmark and prefer GROUP_SIZE=32 for 22-atom xylitol | +30% at 5000 replicas (2.83×10⁸ → 3.69×10⁸ atom-steps/s before zero removal) | benchmark default is 32; device/molecule dependent, not a universal harness hard-code |
| **P2** | Skip collision dispatch when collisions are disabled, while keeping `dpos_coll` deterministically zero | one fewer launch | must fix/define `k_coll` semantics first: current collision kernels accept `k_coll` but do not use it; radius=0 is presently what disables collisions |
| **P2** | Add `cl::Event` timings and device/kernel resource queries | diagnostic, not direct speedup | needed to distinguish enqueue/build overhead, GPU execution, register spill, and local-memory occupancy |
| **P3** | Scope private variables into `{}` blocks | possible spill reduction | compiler-dependent; measure private-memory usage and runtime |
| **P3** | Consider staging shared topology/port constants in `__local` | uncertain | all replicas read the same small arrays, so hardware constant/L1/L2 cache may already serve them efficiently; cooperative copies add instructions, barriers, and local-memory occupancy |
| **P4** | Kernel fusion via `ClAssembler` generated variants | uncertain; launch-bound cases only | variant explosion, additive `__local` use, register pressure/spill; profile first |

#### Current static kernel findings

- Layout `pos[iS*natoms + lid]` is coalesced within each workgroup; do not transpose it without evidence.
- `compute_collision_multi` writes `dpos_coll` for every lane, including padding/inactive lanes.
- `compute_ports_current_multi` zeros all four `dpos_neigh` slots for every node and then overwrites `dpos_node`/`drot_node` for every active node. Therefore the preceding global zero kernel is redundant for **Current** mode if these invariants are asserted and covered by parity tests.
- `apply_corrections_multi` is one-owner gather code and should remain separate unless profiling proves fusion worthwhile.
- `radius` is passed to `compute_ports_current_multi` but not used; remove this argument when touching the persistent kernel interface.
- `k_coll` is passed to collision kernels but currently does not scale or gate the correction. This is a semantic issue to resolve before using `k_coll == 0` as a dispatch condition.
- `port_local`, `kflat`, neighbor and exclusion data are tiny and identical across replicas. They are strong cache candidates; explicit `__local` staging is not automatically faster.

#### Verification gate for every optimization

1. Identical initial state in legacy `RRsp3` and multi-system path (`nSys=1`), Current port mode, collision disabled consistently.
2. Compare one-step positions, quaternions, node corrections, and final relaxed geometry; initial target tolerance `max_abs ≤ 1e-6` for one step and Kabsch RMSD `< 1e-3 Å` after relaxation.
3. Assert every real output is finite and padding remains inactive.
4. Benchmark release build after warmup using `queue.finish()` outside the timed loop; sizes 64 and 5000 distinguish host/launch-bound and saturated throughput.
5. Keep full stdout and record before/after numbers in the existing benchmark report.

#### Multiple molecules per workgroup — reducing padding waste

**Problem**: xylitol (22 atoms) padded to GROUP_SIZE=64 = 65% padding waste. Water (3 atoms) padded to 64 = 95% waste.

**Approach A: smaller workgroups.** Use GROUP_SIZE=32 for xylitol (22 atoms, 31% waste) or GROUP_SIZE=16 (22 atoms doesn't fit). For water, GROUP_SIZE=8 (3 atoms, 62% waste) or GROUP_SIZE=4 (3 atoms, 25% waste). NVIDIA warps are 32 threads — sub-warp workgroups (16, 8, 4) are valid but may underutilize the warp. Benchmark to find the sweet spot.

**Approach B: multiple molecules per workgroup.** Pack 2 xylitols (44 atoms) into one GROUP_SIZE=64 workgroup (31% waste). Each molecule occupies a contiguous slice of local memory. Needs:
- `mol_start[igroup]` array — offset to each molecule's atoms within the workgroup
- `nnode_per_mol` — nodes per molecule (constant if all same molecule)
- Neighbor indices become intra-molecule (local within the molecule's slice)
- Collision loop iterates only over the same molecule's atoms (not the whole workgroup)

**Concern with Approach B**: if molecules fly apart (e.g. during dynamics with large kicks), the AABB of the workgroup becomes unnecessarily large, and the collision loop may test pairs that are far apart. Covalent bonds keep molecules compact, so this is usually not a problem for bonded relaxation. But for surface scanning where molecules are at different (x,y) positions, the workgroup AABB would span all of them — bad for broad-phase (if we ever add inter-replica collisions).

**Approach C: multiple collision-groups (AABB boxes) per workgroup.** Load several independent AABB-bounded groups into one workgroup, each processed by a subset of threads. This would allow fine-grained collision detection without inflating the AABB. BUT this unnecessarily complicates the kernels — each group needs its own neighbor list, exclusion list, and barrier synchronization. **Not recommended for Phase 2.**

**Decision for Phase 2**: try Approach A (smaller GROUP_SIZE) first — it's free (just a parameter change). Approach B is worth it for small molecules (water, methanol) but adds kernel complexity. Approach C is YAGNI.

### Phase 3: CPU comparison and parity

- Run same N-replica system on CPU (parallelized with `rayon` if available)
- Compare wall time → speedup ratio
- Verify Kabsch RMSD < 1e-3 for at least one replica

## Multi-replica porting status

### Kernels in `RRsp3.cl`

| Kernel | Single-system (1D) | Multi-replica (2D) | Notes |
|--------|:------------------:|:------------------:|-------|
| `zero_corrections` | (host-side write) | ✅ `zero_corrections_multi` | Zero dpos buffers for all replicas |
| `compute_collision` | ✅ `compute_collision_cluster_rigid` | ✅ `compute_collision_multi` | No ghosts — all intra-workgroup |
| `compute_ports` (Current) | ✅ `compute_ports_cluster_rigid` | ✅ `compute_ports_current_multi` | Massfull XPBD, rot_mass_scale |
| `compute_ports` (Orig) | ✅ `compute_ports_cluster_rigid_orig` | ❌ TODO | Same as Current but no rot_mass_scale |
| `compute_ports` (Substep) | ✅ `compute_ports_cluster_rigid_substep_optimized` | ❌ TODO | Massless Newton-Raphson in omega-space |
| `compute_ports` (Shapematch) | ✅ `compute_ports_cluster_rigid_shapematch` | ❌ TODO | Massless polar/Kabsch decomposition |
| `compute_ports` (Eigen) | ✅ `compute_optimal_rotation_eigen` + `compute_ports_cluster_rigid_eigen_tips` | ❌ TODO | Massless Horn quaternion eigen (two-pass) |
| `compute_tips` | ✅ `compute_tips` | ❌ TODO | Helper for Eigen variant (rotates ports to world-space tips) |
| `apply_corrections` | ✅ `apply_corrections_rigid_ports` | ✅ `apply_corrections_multi` | Position + quaternion corrections |
| `update_bboxes` | ✅ `update_bboxes_rigid` | ❌ N/A | Not needed for independent replicas (no ghosts) |
| `build_local_topology` | ✅ `build_local_topology_rigid` | ❌ N/A | Not needed for independent replicas (no ghosts) |
| `predict_dynamics` | ✅ `predict_dynamics` | ❌ TODO | Leapfrog predict step (for dynamics mode) |
| `update_velocities` | ✅ `update_velocities_dynamics` | ❌ TODO | Velocity update (for dynamics mode) |

### Rust harness methods in `rrsp3.rs`

| Method | `RRsp3` (single) | `RRsp3Multi` | Notes |
|--------|:-----------------:|:------------:|-------|
| `new()` | ✅ | ✅ | Context + buffer allocation |
| `upload_state` | ✅ | ✅ `upload_state_multi` | Per-replica pos/quat/inv_mass |
| `upload_radius` | ✅ | ✅ | Shared |
| `upload_neighs_and_exclusions` | ✅ | ✅ | Shared |
| `upload_fixmask` | ✅ | ✅ | Shared |
| `upload_cluster_ports` | ✅ | ✅ `upload_cluster_ports_multi` | Shared port_local + kflat |
| `upload_bk_slots` | ✅ | ✅ `upload_bk_slots_multi` | Shared |
| `upload_rev_slot` | ✅ | ❌ TODO | Only needed for node-node reciprocal slots (Eigen variant?) |
| `step_cluster` | ✅ | ✅ `step_cluster_multi` | Current port kernel only |
| `step_dynamics` | ✅ | ❌ TODO | Dynamics mode (predict + relax + update_velocities) |
| `reset_momentum` | ✅ | ❌ TODO | Zero dpos_mom + dquat_mom |
| `reset_dynamics` | ✅ | ❌ TODO | Zero vel + omega |
| `download_pos` | ✅ | ✅ `download_pos_multi` / `download_pos_replica` | |
| `download_quat` | ✅ | ✅ `download_quat_replica` | |
| `download_dpos_coll` | ✅ | ✅ `download_dpos_coll_replica` | |
| `download_dpos_node` | ✅ | ✅ `download_dpos_node_replica` | |
| `download_ghost_counts` | ✅ | ❌ N/A | No ghosts for independent replicas |
| `download_neighs_local` | ✅ | ❌ N/A | No local remapping (neighs used directly) |

### Priority for porting remaining variants

1. **`step_dynamics_multi`** (predict + relax + update_velocities) — needed for MD simulations, not just relaxation. High priority.
2. **Shapematch multi** — massless, good for rigid molecules (aromatics). Medium priority.
3. **Substep multi** — massless Newton-Raphson, good for large perturbations. Medium priority.
4. **Eigen multi** — massless Horn quaternion, most accurate rotation. Low priority (two-pass, more complex).
5. **Orig multi** — nearly identical to Current, trivial port. Low priority.

## Key Algorithms to Understand (from FireCore)

### Neighbor reindexing (global → local + ghosts)

In the current cluster-sorted layout, each workgroup has its own local index space `[0, GROUP_SIZE)`. Atoms from other workgroups that are close enough to interact are loaded as "ghosts" with local indices `[GROUP_SIZE, GROUP_SIZE + n_ghosts)`.

The mapping function (`RRsp3.cl:441`):
```c
inline int map_global_to_local(int t, int grp, int total_ghosts, __local const int* l_ghost_list){
    if (t < 0) return -1;
    int tgrp = t / GROUP_SIZE;
    if (tgrp == grp) return t % GROUP_SIZE;      // same group → local index
    for (int g = 0; g < total_ghosts; g++)        // other group → ghost slot
        if (l_ghost_list[g] == t) return GROUP_SIZE + g;
    return -1;                                    // not found → skip
}
```

**For independent replicas**: this entire mechanism is unnecessary. All neighbors are in the same workgroup. `neighs_local[i] = neighs_global[i]` (identity mapping). The ghost machinery can be completely removed.

### 1-2 / 1-3 exclusion algorithm

Host-side (`pack.rs:make_exclusions_1st_2nd`, ported from `RRsp3.py:58`):
- `excl1[i]` = up to 4 first neighbors (bonded atoms) of atom `i`
- `excl2[i]` = up to 4 second neighbors (neighbors of neighbors, excluding self and 1st neighbors)
- Both are `int4` arrays (4 slots, -1 for unused)

GPU-side (`RRsp3.cl:206`):
```c
inline int excluded8(int j, int4 a, int4 b){
    if( (j==a.x) || (j==a.y) || (j==a.z) || (j==a.w) ) return 1;  // 1-2
    if( (j==b.x) || (j==b.y) || (j==b.z) || (j==b.w) ) return 1;  // 1-3
    return 0;
}
```

The collision kernel skips pairs where `excluded8(j, excl1_local[i], excl2_local[i])` returns 1.

**Known issue**: `make_exclusions_1st_2nd` does NOT build 1-4 exclusions. Molecules with 1-4 pairs within collision radius (like xylitol) will have spurious collision forces. Workaround: set `radius=0` to disable collisions. Fix: add 1-4 exclusion list (would need `excl3` or a wider exclusion mask).

### Back-slot mapping (bk_slots)

For each node `ia` and port `k` pointing to neighbor `ja`, `bk_slots[ja*4 + s] = inode*4 + k` records where the recoil force for `ja` from port `(ia,k)` is stored. The correction kernel gathers these:
```c
// apply_corrections_rigid_ports
for (int s = 0; s < 4; s++) {
    int idx = bkSlots[my_global_id * 4 + s];
    if (idx >= 0) dpos_total += dpos_neigh[idx];
}
```

This is the gather pattern that avoids atomics — each atom collects its recoil contributions from the pre-recorded slots.

## Read-only performance audit after the first optimization pass

Three delegated audits reviewed legacy RRsp3 host dispatch, other `oclff` harnesses, and multi-system kernels. These are analysis findings only; no additional code was changed.

### P0 — correctness or active hot-loop violations

1. **Legacy `RRsp3` repeatedly builds kernels.** `step_cluster` creates 5 OpenCL kernel objects per step; `step_dynamics` creates 7 or more. Builders occur in `run_bboxes_and_topology`, `run_collision`, all five `run_ports` variants, `run_corrections`, `run_predict`, and `run_update_velocities`. This violates the persistent-object rule and should be converted to the same initialization-time kernel cache now used by the multi-system path.
2. **Legacy `margin_sq()` performs GPU→CPU readback and allocation every step.** It allocates `Vec<f32>[natoms]`, downloads invariant radii, reduces `rmax`, then computes the margin. Cache host `radius_max` in `upload_radius`; only the scalar bbox margin may vary per step.
3. **Legacy zero/reset paths allocate and transfer host arrays every step.** `zero_corrections()` allocates three zero vectors and performs up to four H→D writes; dynamics also allocates another zero vector and writes two momentum buffers. Producers should own complete outputs where possible; otherwise use persistent fill storage/commands.
4. **Legacy Eigen cluster mode can reuse stale tips.** `step_dynamics` invalidates `tips_valid`, but `step_cluster` does not even though positions/quaternions change. This is a correctness bug, not only performance work.
5. **`UffOcl::eval_bonds()` rebuilds its entire GPU invocation per evaluation.** It allocates every input/output buffer, builds clear + force kernels, creates flattening/zero vectors, uploads all inputs, and reads output back. This API is test-oriented and cannot be used as a simulation hot path until buffers/kernels are persistent and readback is optional.
6. **Collision semantics are internally inconsistent.** Both collision kernels accept `k_coll` but never use it. Thus `k_coll=0` does not disable or scale collisions; current xylitol runs are collision-free only because all radii are zero. Resolve against the FireCore reference before optimizing dispatch.

### P1 — high-value multi-system improvements

| Finding | Measured/status | Constraints / verification |
|---------|-----------------|----------------------------|
| **Implemented: omit collision when all uploaded radii are zero** | +16.5% at 5000 replicas; +14.6% at 64 replicas | `dpos_coll` is zeroed once during construction/radius reconfiguration; does not reinterpret unresolved `k_coll` semantics |
| **Implemented: one write per `dpos_neigh` slot** | ~1.7% at 5000 replicas | invalid/inactive ports write zero once; valid ports write recoil once; inactive nodes still clear all outputs |
| **Implemented: avoid old momentum reads when `beta=0`** | ~7% at 5000 replicas | output momentum is still written exactly as before, preserving behavior if beta becomes nonzero later; uniform branch |
| Full no-momentum generated correction variant | not implemented; could also remove momentum writes/allocations | macro-generate/select variant; requires explicit policy for changing beta and direct parity |
| Partial replica downloads | current `download_*_replica(i)` allocates/downloads the entire `nSys` buffer | use offset/sub-buffer reads and caller-provided reusable output storage; diagnostics only, not timed hot path |
| Current-only correction signature | `tips` allocation and `port_local`/`tips` correction arguments are dead when `massless_rot=0` | macro-generated Current variant can omit them; keep future massless variants separate |
| Node-only quaternion storage | `quat` and `dquat_mom` are per atom although only node lanes use them | meaningful memory reduction, but indexing/API change is moderate risk |
| Shared topology address space/cache experiment | topology is tiny and invariant across replicas | `__constant` may help but lane indices are not always broadcast-uniform; compare event timings before retaining |

### P2 — lower priority or higher-risk kernel ideas

- Collision computes each pair twice. An unordered-pair algorithm could halve distance/sqrt work but requires contested writes, atomics, or additional local reduction/barriers. Preserve the current owner-gather implementation unless profiling proves collision dominates.
- Node-node ports are also evaluated twice with a `0.5` correction. `rev_slot` exists and could deduplicate them, but this changes dynamics/physics and requires CPU parity—not a pure optimization.
- Replace `int*` aliasing casts of `int4` (`neighbors`, `bk`) with explicit components for OpenCL portability; expected speed gain is small.
- Remove the unused `radius` argument from `compute_ports_current_multi`; clarity gain, negligible runtime gain after kernels are persistent.
- **Rejected experiment:** storing precomputed compliance `1/K` and replacing per-port `1/(K·dt²)` with multiplication changed rounding (`1.147e-5 → 1.146e-5` residual) and improved wall time only ~0.7%, within run variance. Reverted to preserve stiffness-buffer semantics.
- Scope port-loop private temporaries to shorten register lifetimes, then inspect kernel private-memory/resource reports. Do not assume source scopes change generated registers.
- Explicit `__local` staging and kernel fusion remain profiling-dependent. Current local arrays are only ~1.0–1.3 KiB/workgroup at WG64, so local capacity is not presently the primary concern; register pressure is more plausible.

### Latent API hazards elsewhere in `oclff`

`SpffOcl::kernel()` and GridFF/FAF `kernel()` methods return a newly built kernel on every call. They are currently factories used outside a demonstrated hot loop, so they are not yet active violations, but their API shape invites misuse. Before adding eval/scan loops, replace them with initialization-time named kernel caches or make the one-shot nature explicit. Program assembly/compilation in constructors and topology fragment parsing are legitimate setup work.

### Recommended implementation order from this audit

1. Direct legacy-vs-multi parity test (Current, identical state).
2. Fix legacy Eigen stale-tip correctness.
3. Convert legacy RRsp3 kernels/radius/zero storage to persistent state.
4. Define and test collision semantics; then omit collision work when disabled.
5. Single-write `dpos_neigh` and no-momentum generated correction variant, measured separately with events.
6. Refactor `UffOcl::eval_bonds` before using it in any simulation loop.
7. Only then consider constant address space, node-only quaternion packing, pair deduplication, or fusion.

## References

- FireCore throughput benchmark: `/home/prokophapala/git/FireCore-master/tests/tMMFFmulti/run_throughput_MD.py` (6000 replicas of xylitol on NaCl with GridFF)
- FireCore multi-replica kernels: `/home/prokophapala/git/FireCore/cpp/common_resources/cl/relax_multi.cl` (2D `(iG, iS)` indexing)
- FireCore RRsp3 Python: `/home/prokophapala/git/FireCore/pyBall/RigidAtomFF/RRsp3/RRsp3.py` (host-side packing, exclusion builder)
- SurfMol RRsp3 kernel: `opencl/RRsp3.cl` (14 kernels, cluster-sorted layout, ghost atoms)
- SurfMol RRsp3 harness: `crates/libs/oclff/src/rrsp3.rs` (Rust wrapper, 1D dispatch)
- SurfMol pack helpers: `crates/libs/oclff/src/pack.rs` (pack_molecules, exclusions, bk_slots)
- SurfMol CPU reference: `crates/libs/molff/src/raff.rs` (CPU RAFF, parity target)
- Convergence methodology: `notes/conventions/relaxation_convergence.md`

## Decisions

1. **No separate kernel file — modify `RRsp3.cl` directly.** Multi-system simulation is the **primary goal** of `RRsp3.cl`, not a special mode. The current single-system smoke tests are the special case. The kernel must support `nSys` replicas natively. The ghost-atom machinery stays for the interacting-molecules case but is bypassed when replicas are independent (no cross-replica neighbors).

2. **2D NDRange for readability.** `get_global_id(0)` = atom, `get_global_id(1)` = system. On hardware this is identical to 1D index unfolding — same speed, same register allocation. But 2D is more readable and less error-prone (no manual `iS = gid / natoms; iG = gid % natoms` math). Use the `ocl` crate's 2D `global_work_size((natoms, nsys))` / `local_work_size((group_size, 1))`.

3. **Timing: `cl.finish()` wall-time for Phase 1, `cl::Event` profiling for Phase 2.** See §`cl::Event` below.

4. **Separate benchmark binary in `oclff/src/bin/`.** Not molengine+Rhai — a standalone binary gives fine control over warmup, timing, and output without complicating the engine with benchmark-specific Rhai functions. The molengine Rhai path is for user-driven simulation; the benchmark is a developer tool. If the benchmark reveals useful patterns, we can later expose a `benchmark` Rhai function in molengine.

## `cl::Event` profiling — what it is and why it's useful

OpenCL `cl::Event` objects are returned by kernel enqueue calls. They serve two purposes:

### 1. Synchronization (wait for a specific kernel to finish)

```rust
let mut event = ocl::Event::empty()?;
kernel.cmd().arg(...).enew(&mut event).enq()?;
event.wait_for()?;  // block until this specific kernel completes
```

This is finer-grained than `queue.finish()` — you can wait for one kernel without waiting for all queued work.

### 2. Profiling (measure GPU-side kernel execution time)

When the command queue is created with profiling enabled, each event records timestamps:

```rust
// Queue with profiling:
let queue = ocl::Queue::new(&device, ocl::CommandQueueProperties::PROFILING)?;

// After enqueue:
let start = event.profiling_start()?;  // nanoseconds since some reference point
let end = event.profiling_end()?;
let kernel_time_ns = end - start;
let kernel_time_ms = kernel_time_ns as f64 / 1e6;
```

**Why this matters for benchmarking:**

| Method | What it measures | Accuracy |
|--------|-----------------|----------|
| `Instant::now()` / `cl.finish()` / `Instant::now()` | Host wall time including queue dispatch overhead, buffer transfers, kernel execution | ±10–100 µs (host timer resolution + queue overhead) |
| `cl::Event` profiling | **GPU-side kernel execution time only** — excludes host overhead, queue dispatch, buffer transfers | ±1 µs (GPU hardware timer) |

For a single kernel that takes 0.1 ms, the host wall-time method has 10–100% error. For 100 kernels in a loop, the overhead accumulates. `cl::Event` profiling gives the true GPU cost.

**For Phase 1**: use `cl.finish()` + wall-time — it's simpler and sufficient for comparing GPU vs CPU at the system level. The queue dispatch overhead is small relative to 100 steps of relaxation.

**For Phase 2**: enable profiling and measure each kernel separately (`update_bboxes`, `build_topology`, `collision`, `ports`, `corrections`) to identify which kernel dominates. This is essential for optimization — you can't optimize what you can't measure.

**`ocl` crate API**: `ocl::Event` has `profiling_start()` and `profiling_end()` methods returning `Result<cl_ulong>` (nanoseconds). The queue must be created with `CommandQueueProperties::PROFILING`. See `ocl` docs: https://docs.rs/ocl/0.19/ocl/struct.Event.html
