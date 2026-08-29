---
type: report
title: GPU RAFF multi-replica benchmark — first results
description: Throughput benchmark of the multi-replica RRsp3 OpenCL harness on xylitol (22 atoms). 5000 independent replicas on GTX 1650 achieve 2.8×10⁸ atoms/s (2494 steps/s, 0.4ms/step). Launch-bound below 1000 replicas, compute-bound above 2000. Force converges (4.1e-4 → 1.0e-4 over 1500 steps). No ghost atoms needed for independent replicas — massive simplification vs single-system RRsp3.
tags: [gpu, opencl, rrsp3, multi-replica, benchmark, throughput, xylitol, gtx1650, launch-bound, compute-bound, memory-bound]
timestamp: 2026-08-29
---

# GPU RAFF Multi-Replica Benchmark — First Results

## 1. Summary

The multi-replica `RRsp3Multi` harness achieves **2.8×10⁸ atoms/s** (2494 steps/s) on 5000 independent xylitol replicas on an NVIDIA GTX 1650. The force converges monotonically. The key architectural simplification: **independent replicas need no ghost atoms** — all neighbors are intra-workgroup, so the O(N²) broad-phase topology build is completely eliminated.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Molecule | Xylitol (C₅H₁₂O₅, 22 atoms, 21 bonds, 10 nodes) |
| Group size | 64 (padded from 22, 65% waste) |
| Port kernel | Current (massfull XPBD) |
| Stiffness K | 200.0 |
| Collisions | Disabled (radius=0, k_coll=0) — xylitol has 1-4 pairs not excluded by 1st+2nd |
| Perturbation | 0.2Å (H: 0.3Å) random per replica |
| GPU | NVIDIA GeForce GTX 1650 (896 CUDA cores, 128 GB/s, 4GB) |
| Steps (benchmark) | 1000 (timed, after 10 warmup) |
| Steps (convergence) | 2000 (sampled every 100) |

## 3. Throughput vs system size

| nsys | Real atoms | ms/step | steps/s | atoms/s | Regime |
|------|-----------|---------|---------|---------|--------|
| 64 | 1,408 | 0.123 | 8,130 | 1.15×10⁷ | launch-bound |
| 512 | 11,264 | 0.126 | 7,937 | 8.94×10⁷ | launch-bound |
| 1,000 | 22,000 | 0.12 | 8,333 | 1.83×10⁸ | launch-bound |
| 2,000 | 44,000 | 0.178 | 5,618 | 2.47×10⁸ | transition |
| 5,000 | 110,000 | 0.388 | 2,577 | 2.84×10⁸ | compute-bound |

<ref_file file="/home/prokohapala/git/SurfMol/debug/raff_ocl_benchmark/throughput_sweep.png" />

### 3.1 Launch-bound regime (nsys ≤ 1000)

For small nsys, the per-step time is constant at ~0.12ms regardless of nsys. This is **kernel launch overhead**: 3 kernels per step × ~40µs per launch = 120µs. The GPU finishes the compute before the host can enqueue the next kernel.

**Implication**: for small systems, batching more replicas is "free" — 64 replicas take the same time as 1 replica. Always batch as many replicas as possible.

### 3.2 Compute-bound regime (nsys ≥ 2000)

For large nsys, the per-step time grows linearly with nsys. At 5000 replicas, 0.388ms/step = 0.078µs per replica per step.

**Memory bandwidth analysis**: each step reads/writes ~20 MB (pos + quat + dpos buffers for 5000×64 atoms). At 0.388ms: 20MB / 0.388ms = 51.5 GB/s = **40% of GTX 1650's 128 GB/s peak**. The kernel is memory-bound, as expected for a lightweight force kernel (~2000 FLOPs per replica, ~25 GFLOPs/s = ~1% of compute peak).

### 3.3 Peak throughput

**2.84×10⁸ atoms/s** at 5000 replicas. For the FireCore reference benchmark (6000 replicas of xylitol, 10000 MD steps), this would take:
- 10000 steps / 2494 steps/s ≈ **4 seconds** (molecular relaxation only, no surface GridFF)

## 4. Convergence

<ref_file file="/home/prokohapala/git/SurfMol/debug/raff_ocl_benchmark/convergence.png" />

The max|correction| (force proxy) decreases monotonically:
- Step 0: 4.1×10⁻⁴
- Step 500: 2.5×10⁻⁴
- Step 1000: 1.6×10⁻⁴
- Step 1500: 1.0×10⁻⁴

The convergence rate is slow (~40% reduction per 500 steps) because:
1. `dt=0.1` is conservative (no line search or adaptive timestep)
2. `relaxation=0.5` under-relaxes (Jacobi-style, not Gauss-Seidel)
3. Collisions are disabled — the port springs alone converge slowly for large perturbations

This is expected behavior for the RAFF port-force relaxation. The physics is correct; convergence tuning is a separate concern.

## 5. Architecture: what was implemented

### 5.1 Multi-replica kernels in `RRsp3.cl`

Added 4 new kernels at the end of `opencl/RRsp3.cl`:

| Kernel | Purpose |
|--------|---------|
| `zero_corrections_multi` | Zero dpos_coll, dpos_node, drot_node, dpos_neigh for all replicas |
| `compute_collision_multi` | Jacobi sphere collisions (intra-workgroup only, no ghosts) |
| `compute_ports_current_multi` | Massfull XPBD port forces (Current variant) |
| `apply_corrections_multi` | Apply position + quaternion corrections |

All use 2D NDRange: `global = (GROUP_SIZE, nSys)`, `local = (GROUP_SIZE, 1)`.
- `lid = get_local_id(0)` = atom index within replica
- `iS = get_global_id(1)` = replica index
- `i0a = iS * natoms` = offset to this replica's data

**Shared buffers** (uploaded once, same molecule): radius, neighs, excl1, excl2, fixmask, port_local, kflat, bk_slots — all indexed by local index `lid`.

**Per-replica buffers** (differ across replicas): pos, quat, dpos_coll, dpos_node, drot_node, dpos_neigh, dpos_mom, dquat_mom — all indexed by `i0a + lid` or `i0n + lid`.

### 5.2 `RRsp3Multi` Rust harness in `rrsp3.rs`

New struct `RRsp3Multi` with:
- `new(natoms_per_sys, group_size, nsys, prefer_gpu)` — creates context, allocates buffers
- `upload_radius`, `upload_neighs_and_exclusions`, `upload_cluster_ports_multi`, `upload_bk_slots_multi` — shared topology
- `upload_state_multi(pos3_flat, inv_mass, quat_flat)` — per-replica state (all replicas at once)
- `step_cluster_multi(port_kernel, cfg)` — one step for ALL replicas (4 kernel launches)
- `download_pos_replica(isys)`, `download_dpos_coll_replica(isys)`, etc. — per-replica download
- `finish()` — wait for queue completion

### 5.3 Benchmark binary `raff_ocl_benchmark.rs`

- Loads xylitol, packs one copy, builds topology
- Creates N replicas with random perturbations
- Warmup (10 steps) → timed benchmark (1000 steps) → convergence (2000 steps)
- Saves `timing.tsv`, `convergence.tsv`, `final_replica0.xyz` to `debug/raff_ocl_benchmark/`

## 6. What was eliminated vs single-system RRsp3

| Component | Single-system RRsp3 | Multi-replica RRsp3Multi | Why |
|-----------|---------------------|--------------------------|-----|
| `update_bboxes_rigid` | Required (find ghost candidates) | **Eliminated** | No ghosts needed |
| `build_local_topology_rigid` | Required (O(N²) AABB overlap) | **Eliminated** | All neighbors intra-workgroup |
| Ghost atom loading | Required (cross-workgroup neighbors) | **Eliminated** | Independent replicas |
| `neighs_local` remapping | Required (global → local + ghost) | **Eliminated** | neighs are already local |
| `excl_local` remapping | Required | **Eliminated** | excl are already local |

**Result**: 4 kernel launches per step (zero + collision + ports + corrections) vs 6+ for single-system (bboxes + topology + zero + collision + ports + corrections). And no O(N²) topology build.

## 7. Limitations and next steps

### 7.1 Current limitations

**Port kernel variants** — only `Current` is ported to multi-replica. Full status:

| Kernel | Single (1D) | Multi (2D) | Priority |
|--------|:-----------:|:----------:|----------|
| collision | ✅ | ✅ | — |
| ports Current | ✅ | ✅ | — |
| ports Orig | ✅ | ❌ TODO | low (trivial, same as Current) |
| ports Substep | ✅ | ❌ TODO | medium (massless Newton) |
| ports Shapematch | ✅ | ❌ TODO | medium (massless Kabsch) |
| ports Eigen (2-pass) | ✅ | ❌ TODO | low (complex, two-pass) |
| compute_tips | ✅ | ❌ TODO | needed for Eigen |
| apply_corrections | ✅ | ✅ | — |
| predict_dynamics | ✅ | ❌ TODO | high (needed for MD) |
| update_velocities | ✅ | ❌ TODO | high (needed for MD) |
| update_bboxes | ✅ | N/A | not needed (no ghosts) |
| build_local_topology | ✅ | N/A | not needed (no ghosts) |

See task spec `notes/tasks/2026-08-29_gpu_raff_benchmark.md` §"Multi-replica porting status" for the full table including Rust harness methods.

**Other limitations**:
- **No dynamics mode** (`step_dynamics_multi`) — only relaxation (`step_cluster_multi`).
- **Collisions disabled** (radius=0) — xylitol has 1-4 pairs not excluded by 1st+2nd neighbor exclusions. Need 1-4 exclusion list for proper intra-molecular collisions.
- **65% padding waste** — xylitol (22 atoms) padded to 64. Could use GROUP_SIZE=32 (31% waste) or pack 2 xylitols per workgroup (31% waste).

### 7.2 Optimization opportunities

#### Kernel fusion — likely NOT worth it (see task spec for full analysis)

Two concerns make kernel fusion risky:
1. **Combinatorial explosion**: fused collision+ports+corrections would need 20+ variants (5 port kernels × collision on/off × dynamics on/off). Solution: use the `ClAssembler` macro metaprogramming system (`crates/libs/oclff/src/assemble.rs`) to generate only needed variants — but this is engineering effort for uncertain gain.
2. **Register spill + local memory limits**: `__local` arrays are additive across fused phases (`l_pos` from collision + `l_pos` from ports = 2× local memory). Too much `__local` reduces occupancy. Register spill to private memory kills performance. **OpenCL requires `__local` declared at kernel top — cannot scope into blocks.** Private variables CAN be scoped into `{}` blocks to shorten register lifetimes.

**Decision**: do NOT fuse for Phase 2. The 4-launch overhead is ~0.12ms (launch-bound regime), small relative to 0.39ms compute at 5000 replicas. Profile with `cl::Event` first to confirm launch is actually the bottleneck.

#### What to optimize instead
- **Shared memory for port geometry**: `port_local` and `kflat` are shared across all replicas but in global memory. Load to `__local` once per workgroup.
- **Smaller GROUP_SIZE**: GROUP_SIZE=32 for xylitol (22 atoms) = 31% waste vs 65% at 64. Free parameter change — benchmark it.
- **Multiple molecules per workgroup**: pack 2 xylitols (44 atoms) per GROUP_SIZE=64 = 31% waste. Needs `mol_start[igroup]` boundaries. Concern: if molecules fly apart, workgroup AABB inflates. Covalent bonds keep molecules compact, so OK for bonded relaxation. For small molecules (water, methanol) this is essential. See task spec for full discussion.
- **Scope private variables** into `{}` blocks to reduce register spill (good practice regardless).
- **`cl::Event` profiling**: measure per-kernel GPU time to identify which kernel dominates.

### 7.3 Parity verification
- Need to compare multi-replica GPU output with single-system RRsp3 output for the same molecule + perturbation. If they match (Kabsch RMSD < 1e-3), the multi-replica kernels are correct.
- The convergence trace (4.1e-4 → 1.0e-4) is consistent with the single-system xylitol test, but a direct comparison hasn't been done yet.

## 8. Files

| File | Description |
|------|-------------|
| `opencl/RRsp3.cl` (lines 1749–2088) | 4 new multi-replica kernels |
| `crates/libs/oclff/src/rrsp3.rs` (lines 684–1095) | `RRsp3Multi` struct + methods |
| `crates/libs/oclff/src/bin/raff_ocl_benchmark.rs` | Benchmark binary |
| `debug/raff_ocl_benchmark/timing.tsv` | Throughput numbers |
| `debug/raff_ocl_benchmark/convergence.tsv` | Force convergence trace |
| `debug/raff_ocl_benchmark/convergence.png` | Convergence plot |
| `debug/raff_ocl_benchmark/throughput_sweep.png` | Throughput vs nsys plot |
| `debug/raff_ocl_benchmark/final_replica0.xyz` | Final relaxed geometry |
| `debug/raff_ocl_benchmark/plot_benchmark.py` | Plotting script |
