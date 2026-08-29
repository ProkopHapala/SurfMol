//! raff_ocl_benchmark — GPU throughput benchmark for multi-replica RRsp3.
//!
//! Loads xylitol (21 atoms), packs N independent replicas with random perturbations,
//! and measures GPU throughput (steps/second, atoms/second) vs system size.
//!
//! Methodology (per notes/tasks/2026-08-29_gpu_raff_benchmark.md):
//!   1. Setup: load xylitol, build topology, pack N replicas with random perturbations
//!   2. Warmup: 10 GPU steps (not timed) — JIT compilation, buffer allocation
//!   3. Benchmark: 100 GPU steps, measure wall time (cl.finish() before stop)
//!   4. Convergence: 5000 steps, track max|correction| per step
//!   5. Output: timing.tsv + convergence.tsv to debug/raff_ocl_benchmark/
//!
//! Usage: cargo run -p oclff --bin raff_ocl_benchmark [--nsys N] [--group-size G] [--steps N]

use oclff::pack::{pack_molecules, MolInput, build_neighs_from_bonds, make_exclusions_1st_2nd, make_bk_slots_clustered, make_ports_from_neighs, masses_from_elems};
use oclff::rrsp3::{RRsp3Multi, PortKernel, StepConfig};
use moltopo::xyz;
use moltopo::builder::Builder;
use std::path::Path;
use std::io::Write;
use std::time::Instant;

fn covalent_radii(elems: &[String]) -> Vec<f64> {
    elems.iter().map(|el| match el.as_str() {
        "H" => 0.31, "C" => 0.76, "N" => 0.71, "O" => 0.66,
        "F" => 0.57, "P" => 1.07, "S" => 1.05, "Cl" => 1.02,
        _ => 0.7,
    }).collect()
}

/// Load xylitol from XYZ, detect bonds, return (positions, bonds, elems, nnode).
fn load_xylitol() -> (Vec<[f32;3]>, Vec<(usize,usize)>, Vec<String>, usize) {
    let path = Path::new("data/xyz/xylitol.xyz");
    let sys = xyz::read_xyz(path).unwrap_or_else(|e| panic!("read_xyz failed for {}: {e}", path.display()));
    let radii = covalent_radii(&sys.elems);
    let top = Builder::from_positions_and_radii(&sys.apos, &sys.elems, &radii, 0.4).bake();
    let natoms = sys.apos.len();
    let pos_f32: Vec<[f32;3]> = sys.apos.iter().map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
    let bonds: Vec<(usize,usize)> = top.bonds.iter().map(|b| (b[0] as usize, b[1] as usize)).collect();
    let mut deg = vec![0i32; natoms];
    for &(i,j) in &bonds { deg[i] += 1; deg[j] += 1; }
    let nnode = deg.iter().filter(|&&d| d > 1).count();
    eprintln!("xylitol: {natoms} atoms, {} bonds, {nnode} nodes", bonds.len());
    (pos_f32, bonds, sys.elems, nnode)
}

fn parse_args() -> (usize, usize, usize, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut nsys = 500usize;
    let mut group_size = 32usize;
    let mut bench_steps = 100usize;
    let mut conv_steps = 5000usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--nsys" => { i += 1; if i < args.len() { nsys = args[i].parse().unwrap_or(500); } }
            "--group-size" => { i += 1; if i < args.len() { group_size = args[i].parse().unwrap_or(32); } }
            "--steps" => { i += 1; if i < args.len() { bench_steps = args[i].parse().unwrap_or(100); } }
            "--conv-steps" => { i += 1; if i < args.len() { conv_steps = args[i].parse().unwrap_or(5000); } }
            _ => {}
        }
        i += 1;
    }
    (nsys, group_size, bench_steps, conv_steps)
}

fn main() -> ocl::Result<()> {
    let (nsys, group_size, bench_steps, conv_steps) = parse_args();
    eprintln!("=== raff_ocl_benchmark: nsys={nsys}, group_size={group_size}, bench_steps={bench_steps}, conv_steps={conv_steps} ===");

    // 1. Load xylitol and pack ONE copy (nodes first, then caps, then padding)
    let (pos_xyl, bonds, elems, nnode) = load_xylitol();
    let natoms_real = pos_xyl.len();
    assert!(natoms_real <= group_size, "xylitol has {natoms_real} atoms > group_size={group_size}");

    let mol = MolInput { elems: elems.clone(), pos: pos_xyl.clone(), bonds: bonds.clone(), nnode };
    let packed = pack_molecules(&[mol], group_size);
    let natoms_per_sys = packed.natoms;  // padded to group_size
    let nnode_per_group = packed.nnode_per_group[0] as i32;
    eprintln!("packed: natoms_per_sys={natoms_per_sys} (padded from {natoms_real}), nnode_per_group={nnode_per_group}");

    // 2. Build shared topology (local indices 0..natoms_per_sys-1)
    let neighs = build_neighs_from_bonds(natoms_per_sys, &packed.bonds);
    let (excl1, excl2) = make_exclusions_1st_2nd(&neighs, natoms_per_sys);
    let (port_local, kflat) = make_ports_from_neighs(&packed.pos, &neighs, natoms_per_sys, 200.0);
    let bk_slots = make_bk_slots_clustered(&neighs, group_size, &packed.nnode_per_group, natoms_per_sys);

    // Masses + inv mass (shared, same molecule)
    let masses = masses_from_elems(&packed);
    let inv_mass: Vec<f32> = (0..natoms_per_sys).map(|i| if packed.is_padding[i] { 0.0 } else { 1.0 / masses[i] }).collect();

    // Radius=0: disable collisions (xylitol has 1-4 pairs not excluded by 1st+2nd only)
    let radius: Vec<f32> = vec![0.0; natoms_per_sys];

    // 3. Create N replicas with random perturbations
    let perturb = 0.2f32;
    let mut pos3_flat = vec![0.0f32; nsys * natoms_per_sys * 3];
    let mut seed = 12345u64;
    for is in 0..nsys {
        for i in 0..natoms_per_sys {
            if packed.is_padding[i] {
                pos3_flat[(is * natoms_per_sys + i) * 3 + 0] = 0.0;
                pos3_flat[(is * natoms_per_sys + i) * 3 + 1] = 0.0;
                pos3_flat[(is * natoms_per_sys + i) * 3 + 2] = 0.0;
                continue;
            }
            let is_h = packed.elems[i] == "H";
            let amp = if is_h { perturb * 1.5 } else { perturb };
            for d in 0..3 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let r = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
                pos3_flat[(is * natoms_per_sys + i) * 3 + d] = packed.pos[i][d] + amp * r * 2.0;
            }
        }
    }

    // 4. Create GPU harness and upload
    eprintln!("creating RRsp3Multi harness (nsys={nsys}, group_size={group_size})...");
    let mut sim = RRsp3Multi::new(natoms_per_sys, group_size, nsys, true)?;
    eprintln!("OpenCL device: {}", sim.device_name());

    sim.upload_radius(&radius)?;
    sim.upload_neighs_and_exclusions(&neighs, &excl1, &excl2)?;
    sim.upload_cluster_ports_multi(&port_local, &kflat, nnode_per_group)?;
    sim.upload_bk_slots_multi(&bk_slots)?;
    sim.upload_state_multi(&pos3_flat, &inv_mass, None)?;

    let cfg = StepConfig { dt: 0.1, k_coll: 0.0, relaxation: 0.5, ..Default::default() };
    let port_kernel = PortKernel::Current;

    // 5. Warmup (not timed)
    eprintln!("warmup: 10 steps (not timed)...");
    for _ in 0..10 {
        sim.step_cluster_multi(port_kernel, &cfg)?;
    }
    sim.finish()?;
    eprintln!("warmup done");

    // 6. Benchmark: measure wall time for bench_steps steps
    eprintln!("benchmark: {bench_steps} steps (timed)...");
    let t0 = Instant::now();
    for _ in 0..bench_steps {
        sim.step_cluster_multi(port_kernel, &cfg)?;
    }
    sim.finish()?;  // wait for all kernels to complete
    let elapsed = t0.elapsed();
    let wall_s = elapsed.as_secs_f64();
    let wall_ms = wall_s * 1000.0;
    let steps_per_s = bench_steps as f64 / wall_s;
    let atoms_per_s = steps_per_s * (nsys * natoms_real) as f64;
    eprintln!("benchmark result: {bench_steps} steps in {wall_ms:.3}ms = {steps_per_s:.1} steps/s, {atoms_per_s:.3e} atoms/s (real atoms = {nsys}×{natoms_real}={})", nsys * natoms_real);

    // 7. Convergence: run conv_steps steps, track max|correction| via download
    // For throughput we don't download every step (that would dominate time).
    // Instead, download every 100 steps to track convergence.
    eprintln!("convergence: {conv_steps} steps (sampling every 100 steps)...");
    let mut conv_data: Vec<(usize, f32, f32)> = Vec::new();  // (step, max_dpos_coll, max_dpos_node)
    for step in 0..conv_steps {
        sim.step_cluster_multi(port_kernel, &cfg)?;
        if step % 100 == 0 || step < 5 {
            sim.finish()?;
            // Download correction buffers for replica 0 to track convergence
            let dpos_coll = sim.download_dpos_coll_replica(0)?;
            let dpos_node = sim.download_dpos_node_replica(0)?;
            let max_coll = dpos_coll.iter().take(natoms_real * 3).map(|v| v.abs()).fold(0.0f32, f32::max);
            let max_node = dpos_node.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let max_force = max_coll.max(max_node);
            conv_data.push((step, max_coll, max_node));
            if step % 500 == 0 || step < 5 {
                eprintln!("  step {step}: max|F|={max_force:.3e} (coll={max_coll:.3e} node={max_node:.3e})");
            }
        }
    }
    sim.finish()?;
    eprintln!("convergence done ({} samples)", conv_data.len());

    // 8. Verify final geometry is finite (sanity check)
    let final_pos = sim.download_pos_replica(0)?;
    for i in 0..natoms_real {
        for d in 0..3 {
            assert!(final_pos[i * 3 + d].is_finite(), "final pos replica 0 atom {i} dim {d} not finite: {}", final_pos[i * 3 + d]);
        }
    }
    eprintln!("sanity: all positions finite for replica 0 — PASS");

    // 9. Save outputs
    let outdir = Path::new("debug/raff_ocl_benchmark");
    std::fs::create_dir_all(outdir).expect("create debug dir");

    // timing.tsv
    let timing_path = outdir.join("timing.tsv");
    let mut tf = std::fs::File::create(&timing_path).expect("create timing.tsv");
    writeln!(tf, "nsys\tgroup_size\tbench_steps\twall_ms\tsteps_per_s\tatoms_per_s\ttotal_real_atoms").expect("tsv write");
    writeln!(tf, "{nsys}\t{group_size}\t{bench_steps}\t{wall_ms}\t{steps_per_s:.6}\t{atoms_per_s:.6}\t{}", nsys * natoms_real).expect("tsv write");
    tf.flush().expect("tsv flush");
    eprintln!("saved: {}", timing_path.display());

    // convergence.tsv
    let conv_path = outdir.join("convergence.tsv");
    let mut cf = std::fs::File::create(&conv_path).expect("create convergence.tsv");
    writeln!(cf, "step\tmax_dpos_coll_replica0\tmax_dpos_node_replica0").expect("tsv write");
    for (step, max_coll, max_node) in &conv_data {
        writeln!(cf, "{step}\t{max_coll:.10}\t{max_node:.10}").expect("tsv write");
    }
    cf.flush().expect("tsv flush");
    eprintln!("saved: {} ({} samples)", conv_path.display(), conv_data.len());

    // Save final geometry of replica 0 as XYZ
    let final_xyz_path = outdir.join("final_replica0.xyz");
    let mut ff = std::fs::File::create(&final_xyz_path).expect("create final.xyz");
    writeln!(ff, "{natoms_real}").expect("xyz write");
    writeln!(ff, "final replica 0 after {conv_steps} steps").expect("xyz write");
    for i in 0..natoms_real {
        writeln!(ff, "{} {:.6} {:.6} {:.6}", packed.elems[i], final_pos[i*3], final_pos[i*3+1], final_pos[i*3+2]).expect("xyz write");
    }
    eprintln!("saved: {}", final_xyz_path.display());

    // Summary
    eprintln!("\n=== BENCHMARK SUMMARY ===");
    eprintln!("  device:       {}", sim.device_name());
    eprintln!("  nsys:         {nsys}");
    eprintln!("  group_size:   {group_size}");
    eprintln!("  atoms/replica:{natoms_real} (padded to {natoms_per_sys})");
    eprintln!("  total atoms:  {} (real), {} (padded)", nsys * natoms_real, nsys * natoms_per_sys);
    eprintln!("  bench_steps:  {bench_steps}");
    eprintln!("  wall_time:    {wall_ms:.3}ms");
    eprintln!("  throughput:   {steps_per_s:.1} steps/s");
    eprintln!("  throughput:   {atoms_per_s:.3e} atoms/s");
    eprintln!("  per-step:     {:.3} ms/step", wall_ms as f64 / bench_steps as f64);
    eprintln!("  per-replica:  {:.3} ms/replica/step", wall_ms as f64 / bench_steps as f64 / nsys as f64);
    eprintln!("=== raff_ocl_benchmark: DONE ===");
    Ok(())
}
