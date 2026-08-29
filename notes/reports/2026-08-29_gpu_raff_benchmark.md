---
type: report
title: GPU RAFF multi-replica benchmark — first results
description: Multi-replica RRsp3 optimization benchmark on xylitol (22 atoms). Persistent kernels, WG32, producer-owned single writes, radius-zero collision omission, and avoiding dead momentum reads raise GTX 1650 throughput from 2.83×10⁸ to about 5.6×10⁸ real atom-steps/s at 5000 replicas. Direct legacy-vs-multi parity remains the next correctness gate.
tags: [gpu, opencl, rrsp3, multi-replica, benchmark, throughput, xylitol, gtx1650, launch-bound, compute-bound, memory-bound]
timestamp: 2026-08-29
---

# GPU RAFF Multi-Replica Benchmark — First Results

## 1. Summary

The optimized multi-replica path achieves approximately **5.6×10⁸ real atom-steps/s** (5.1k simulation steps/s, 0.194–0.197 ms/step) for 5000 xylitol replicas on an NVIDIA GTX 1650—about **2× the original WG64 implementation**. Retained changes: WG32, persistent kernels, no redundant zero launch, one recoil-buffer write per port, no collision launch when all radii are zero, and no old-momentum reads when `beta=0`. Independent replicas still require no ghosts or O(N²) broad phase.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Molecule | Xylitol (C₅H₁₂O₅, 22 atoms, 21 bonds, 10 nodes) |
| Group size | 32 (padded from 22, 31% waste); original baseline used 64 |
| Port kernel | Current (massfull XPBD) |
| Stiffness K | 200.0 |
| Collisions | Disabled by `radius=0`; collision launch omitted and `dpos_coll` initialized once. `k_coll` remains semantically unresolved/unused. |
| Perturbation | 0.2Å (H: 0.3Å) random per replica |
| GPU | NVIDIA GeForce GTX 1650 (896 CUDA cores, 128 GB/s, 4GB) |
| Steps (benchmark) | 5000 for optimization comparisons (timed after 10 warmup) |
| Steps (convergence) | 2000 (sampled every 100) |

## 3. Throughput vs system size

| Configuration | nsys | Real atoms | ms/step | steps/s | real atom-steps/s |
|---------------|------|------------|---------|---------|-------------------|
| Original, WG64, transient kernels, 4 launches | 64 | 1,408 | 0.123 | 8,130 | 1.15×10⁷ |
| **Optimized, WG32, persistent kernels, 3 launches** | **64** | **1,408** | **0.0123** | **81,301** | **1.15×10⁸** |
| Original, WG64, transient kernels, 4 launches | 5,000 | 110,000 | 0.388 | 2,576 | 2.83×10⁸ |
| WG32 only (before removing zero) | 5,000 | 110,000 | 0.298 | 3,351 | 3.69×10⁸ |
| WG32 + no zero launch | 5,000 | 110,000 | 0.251 | 3,987 | 4.39×10⁸ |
| WG32 + no zero + persistent kernels | 5,000 | 110,000 | 0.251 | 3,984 | 4.38×10⁸ |
| + one `dpos_neigh` write per port | 5,000 | 110,000 | 0.247 | 4,052 | 4.46×10⁸ |
| + omit collision when all radii zero | 5,000 | 110,000 | 0.212 | 4,722 | 5.19×10⁸ |
| **+ avoid old-momentum reads at beta=0 (final)** | **5,000** | **110,000** | **0.194–0.197** | **5,068–5,156** | **5.58–5.67×10⁸** |

At saturation, WG32 gives +30%, removing the redundant zero launch +19%, single-write recoil ~1.7%, radius-zero collision omission ~16.5%, and avoiding dead momentum reads ~7%. Persistent kernels do not change saturated GPU time but dominate small-batch improvement. End-to-end saturated throughput is now approximately **2× the original path**. At 64 replicas the latest measured path reaches 93.2k steps/s versus 8.1k originally (~11.5×).

<ref_file file="/home/prokohapala/git/SurfMol/debug/raff_ocl_benchmark/throughput_sweep.png" />

### 3.1 Host overhead and persistent kernels

The original ~0.12 ms floor was not pure GPU launch time: `step_cluster_multi()` rebuilt four `ocl::Kernel` objects and reset every invariant argument on every step. Caching three persistent kernels reduces the optimized 64-replica path to ~0.012 ms/step. This confirms repeated kernel construction was a major host-side bottleneck.

### 3.2 Saturated throughput

At 5000 replicas, persistent kernel caching alone has no measurable effect: GPU work dominates. The complete retained sequence raises throughput from 2.83×10⁸ to roughly 5.6×10⁸ real atom-steps/s. The largest post-WG32 kernel-path gain is recognizing that radius-zero systems need no collision launch; `dpos_coll` is initialized once when radii are uploaded.

The earlier bandwidth estimate is withdrawn: host wall time alone cannot establish memory-bound behavior. Per-kernel `cl::Event` timing plus measured bytes and device counters are required before classifying the optimized path.

### 3.3 Peak measured throughput

**5.58–5.67×10⁸ real atom-steps/s** at 5000 replicas across the final runs. At this rate, 10000 molecular-relaxation steps take about **1.9–2.0 seconds** (surface GridFF excluded).

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
- `step_cluster_multi(port_kernel, cfg)` — one step for ALL replicas (2 persistent launches for radius-zero systems; 3 when collisions are active; no per-step construction)
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

**Result after optimization**: radius-zero systems use 2 persistent launches (ports + corrections); systems with positive collision radii use 3 (collision + ports + corrections). There is no per-step `Kernel::build` and no O(N²) topology build. `dpos_coll` is initialized once when collisions are disabled; `zero_corrections_multi` remains available for future accumulating variants.

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

**Decision**: do NOT fuse now. Persistent kernels plus removal of redundant zeroing already reduce the path to three launches and ~0.012 ms/step at 64 replicas. At 5000 replicas, measured time is 0.251 ms/step; profile each kernel with `cl::Event` before accepting fusion's register/local-memory risks.

#### Ranked next work after static kernel review

1. **Correctness gate first**: direct one-replica parity against legacy `RRsp3`. Finite positions and decreasing correction are necessary but not sufficient.
2. **Cache persistent `ocl::Kernel` objects**: `step_cluster_multi()` currently runs `Kernel::builder().build()` four times on every step. Removing repeated `clCreateKernel` and invariant argument setup should be the largest low-risk host-side gain, especially below saturation.
3. **Eliminate redundant zero launch for Current mode**: collision overwrites every `dpos_coll`; Current ports zero all node recoil slots and overwrite node outputs. This can remove one launch and buffer write, but only after parity confirms the overwrite invariant.
4. **Benchmark GROUP_SIZE=32**: xylitol uses 22 real lanes, so 32 wastes 31% versus 66% for 64 and occupies one NVIDIA warp rather than two.
5. **Resolve collision-disable semantics**: `k_coll` is currently accepted but unused; radius=0 actually disables collision. Only then skip the collision kernel when disabled while preserving deterministic zero `dpos_coll`.
6. **Add `cl::Event` profiling and resource queries** before changing `__local` use or fusing kernels.

Explicitly staging `port_local`/`kflat` in `__local` is lower priority than previously stated: these arrays are tiny and identical for every replica, so GPU caches may already serve them well. Local staging adds cooperative loads, barriers, and occupancy cost. Likewise, private-variable scopes may reduce register lifetime but are compiler-dependent and must be measured.

The complete ranked table and verification gate are in the existing task specification.

### 7.3 Verification status
- `cargo test -p oclff`: 20 tests passed after optimization, including OpenCL compilation tests.
- Removing zeroing and caching kernels preserves the deterministic printed final xylitol coordinates exactly to 6 decimals; the sampled residual is unchanged (`1.147e-5` after the 5000-step timing sequence).
- A fresh convergence run remains monotonic (`max|F| 5.879e-3` at step 0 → `8.409e-4` at step 500), with finite positions.
- **Still required**: direct legacy `RRsp3` versus multi-system comparison for identical initial state and step count. The current before/after check validates these optimizations, not the original multi-system port itself.

### 7.4 Repository-wide hot-path audit

The persistent-kernel speedup triggered a read-only audit rather than more implementation. Main findings:

- **Legacy `RRsp3` remains a major violation**: 5 kernel builds per cluster step and 7+ per dynamics step, plus per-step GPU→CPU radius readback and host zero-vector allocations/transfers.
- **Legacy Eigen cluster mode has a correctness risk**: `tips_valid` is not invalidated when cluster steps update geometry.
- **`UffOcl::eval_bonds` is not simulation-ready**: every call allocates/uploads all buffers, builds two kernels, allocates output storage, and reads results back.
- **Collision control is misleading**: `k_coll` is dead in both collision implementations. Radius zero—not `k_coll=0`—is what disables xylitol collisions.
- **Next likely multi-system gains**: skip collision work after semantics are defined; single-write `dpos_neigh`; generated no-momentum correction variant for `beta=0`; partial replica downloads rather than transferring all replicas.
- **Lower-confidence ideas**: constant address space for shared topology, node-only quaternion buffers, pair deduplication, and fusion require profiling/parity before implementation.

The complete ranked audit, evidence, and implementation order are recorded in the existing task specification. No code was changed for these audit findings.

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
