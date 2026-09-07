//! raff_ocl_smoke — CLI smoke test for the RRsp3 OpenCL harness.
//!
//! Port of FireCore `pyBall/RigidAtomFF/RRsp3/test_RRsp3_smoke.py`.
//! Sets up 2 water molecules, packs them into cluster-sorted layout, uploads to GPU,
//! runs one step_cluster, downloads diagnostics, and checks local index ranges.
//!
//! Then runs a perturbed relaxation (GPU only) with trajectory save, and optionally
//! a CPU↔GPU parity check (final geometry comparison via Kabsch RMSD).
//!
//! Usage: cargo run -p oclff --bin raff_ocl_smoke [--port current|orig|substep|shapematch|eigen] [--parity] [--traj PATH] [--tsv PATH]

use oclff::pack::{pack_molecules, MolInput, build_neighs_from_bonds, make_exclusions_1st_2nd, make_bk_slots_clustered, make_ports_from_neighs, make_h2o_geometry, masses_from_elems};
use oclff::rrsp3::{RRsp3, PortKernel, StepConfig};
use numtypes::Vec3d;

fn check_local_ranges(neighs_local: &[i32], excl1_local: &[i32], excl2_local: &[i32], ghost_counts: &[i32], group_size: usize) {
    let natoms = neighs_local.len() / 4;
    let ng = natoms / group_size;
    for ig in 0..ng {
        let g = ghost_counts[ig];
        let hi = (group_size as i32 + g) as usize;
        let abase = ig * group_size;
        for (arr, name) in [
            (&neighs_local[abase*4..(abase+group_size)*4], "neighs_local"),
            (&excl1_local[abase*4..(abase+group_size)*4], "excl1_local"),
            (&excl2_local[abase*4..(abase+group_size)*4], "excl2_local"),
        ] {
            for atom in 0..group_size {
                for k in 0..4 {
                    let val = arr[atom * 4 + k];
                    if val != -1 && (val as usize) >= hi {
                        panic!("check_local_ranges: {name} out of range in group {ig}: hi={hi} atom={atom} k={k} val={val}");
                    }
                }
            }
        }
    }
}

/// Parse command-line args. Returns (port_kernel, do_parity, traj_path, tsv_path).
fn parse_args() -> (PortKernel, bool, Option<String>, Option<String>) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut port = PortKernel::Current;
    let mut parity = false;
    let mut traj: Option<String> = None;
    let mut tsv: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--parity" => { parity = true; }
            "--traj" => { i += 1; if i < args.len() { traj = Some(args[i].clone()); } }
            "--tsv" => { i += 1; if i < args.len() { tsv = Some(args[i].clone()); } }
            s if !s.starts_with("--") => { port = PortKernel::from_str(s); }
            _ => {}
        }
        i += 1;
    }
    (port, parity, traj, tsv)
}

/// Write an XYZ frame to a string (only real atoms, skip padding).
fn xyz_frame(elems: &[String], is_pad: &[bool], pos4: &[[f32; 4]]) -> String {
    let n_real = is_pad.iter().filter(|&&p| !p).count();
    let mut s = format!("{n_real}\nstep\n");
    for i in 0..pos4.len() {
        if is_pad[i] { continue; }
        s += &format!("{} {:.6} {:.6} {:.6}\n", elems[i], pos4[i][0], pos4[i][1], pos4[i][2]);
    }
    s
}

/// Build a CPU RAFF system for 2 water molecules (same geometry as GPU).
/// Returns (state, topology) ready for step_xpbd.
fn build_cpu_water2(perturb: f32) -> (molff::raff::RaffState, molff::raff::RaffTopology) {
    use molff::raff::*;
    let (pos_h2o, bonds, _nnode, _elems) = make_h2o_geometry_f64();
    let pos2: Vec<Vec3d> = pos_h2o.iter().map(|p| Vec3d::new((p[0] + 4.0) as f64, p[1] as f64, p[2] as f64)).collect();
    let mut apos = Vec::new();
    for p in &pos_h2o { apos.push(Vec3d::new(p[0] as f64, p[1] as f64, p[2] as f64)); }
    for p in &pos2 { apos.push(Vec3d::new(p.x, p.y, p.z)); }
    for mol_start in [0usize, 3] {
        let o = apos[mol_start];
        for h in 1..=2 {
            let i = mol_start + h;
            let dx = apos[i].x - o.x; let dy = apos[i].y - o.y; let dz = apos[i].z - o.z;
            let len = (dx*dx + dy*dy + dz*dz).sqrt();
            let scale = 1.0 + perturb as f64 / len;
            apos[i].x = o.x + dx * scale; apos[i].y = o.y + dy * scale; apos[i].z = o.z + dz * scale;
        }
    }
    let bonds_arr: Vec<[i32; 2]> = vec![[0,1],[0,2],[3,4],[3,5]];
    let natoms = apos.len();
    let mut topo = RaffTopology::new(natoms);
    topo.bond_params = bonds_arr.iter().map(|_| PortParam { k_p: 200.0, l0: 0.96 }).collect();
    topo.build_neighs_from_bonds(&bonds_arr);
    topo.build_exclusions();
    topo.compute_inertia();
    topo.set_port_geometry_from_reference(&apos);
    let mut state = RaffState::new(natoms);
    state.set_positions(&apos);
    (state, topo)
}

fn make_h2o_geometry_f64() -> (Vec<[f32; 3]>, Vec<(usize, usize)>, usize, Vec<String>) { make_h2o_geometry() }

fn main() -> ocl::Result<()> {
    let (port_kernel, do_parity, traj_path, tsv_path) = parse_args();
    eprintln!("=== raff_ocl_smoke: port_kernel={port_kernel:?}, parity={do_parity}, traj={traj_path:?}, tsv={tsv_path:?} ===");

    let group_size = 64usize;
    let (pos_h2o, _bonds, nnode, elems) = make_h2o_geometry();
    let m1 = MolInput { elems: elems.clone(), pos: pos_h2o.clone(), bonds: _bonds.clone(), nnode };
    let m2 = MolInput { elems, pos: pos_h2o.iter().map(|p| [p[0] + 4.0, p[1], p[2]]).collect(), bonds: _bonds, nnode };
    let packed = pack_molecules(&[m1, m2], group_size);
    let natoms = packed.natoms;
    let ng = packed.num_groups;
    eprintln!("packed: natoms={natoms}, ng={ng}, nnode_per_group={:?}", packed.nnode_per_group);
    assert!(ng == 2, "expected 2 groups, got ng={ng}");

    let masses = masses_from_elems(&packed);
    let inv_mass: Vec<f32> = (0..natoms).map(|i| if packed.is_padding[i] { 0.0 } else { 1.0 / masses[i] }).collect();
    let neighs = build_neighs_from_bonds(natoms, &packed.bonds);
    let nnode_per_group = packed.nnode_per_group[0] as i32;
    assert!(packed.nnode_per_group.iter().all(|&n| n == nnode_per_group), "smoke test assumes constant nnode_per_group, got {:?}", packed.nnode_per_group);
    let (excl1, excl2) = make_exclusions_1st_2nd(&neighs, natoms);
    let (port_local, kflat) = make_ports_from_neighs(&packed.pos, &neighs, natoms, 200.0);
    let radius: Vec<f32> = (0..natoms).map(|i| if packed.is_padding[i] { 0.0 } else { 1.0 }).collect();
    let bk_slots = make_bk_slots_clustered(&neighs, group_size, &packed.nnode_per_group, natoms);

    let max_ghosts = 64usize;
    let mut sim = RRsp3::new(natoms, group_size, max_ghosts, true)?;
    eprintln!("OpenCL device: {}", sim.device_name());

    let quat: Vec<[f32; 4]> = (0..natoms).map(|_| [0.0, 0.0, 0.0, 1.0]).collect();
    sim.upload_state(&packed.pos, &inv_mass, Some(&quat))?;
    sim.upload_radius(&radius)?;
    sim.upload_neighs_and_exclusions(&neighs, &excl1, &excl2)?;
    sim.upload_cluster_ports(&port_local, &kflat, nnode_per_group)?;
    sim.upload_bk_slots(&bk_slots)?;

    // === Part 1: Single step at equilibrium (smoke test) ===
    eprintln!("\n--- Part 1: single step at equilibrium ---");
    let cfg = StepConfig { dt: 0.1, k_coll: 50.0, relaxation: 0.5, bbox_margin: 0.5, ..Default::default() };
    sim.step_cluster(port_kernel, &cfg)?;
    let ghost_counts = sim.download_ghost_counts()?;
    let neighs_local = sim.download_neighs_local()?;
    let (excl1_local, excl2_local) = sim.download_excl_local()?;
    let dpos_coll = sim.download_dpos_coll()?;
    let (pos4, _quat4) = sim.download_pos_quat()?;
    check_local_ranges(&neighs_local, &excl1_local, &excl2_local, &ghost_counts, group_size);
    println!("ghost_counts: {:?}", ghost_counts);
    println!("pos after step group0 atoms0..2:");
    for i in 0..3 { println!("  [{:.6}, {:.6}, {:.6}, {:.6}]", pos4[i][0], pos4[i][1], pos4[i][2], pos4[i][3]); }
    for i in 0..natoms {
        if !packed.is_padding[i] {
            for d in 0..3 { assert!(pos4[i][d].is_finite(), "pos after step: atom {i} dim {d} is not finite: {}", pos4[i][d]); }
        }
    }
    eprintln!("Part 1: PASS (all outputs finite, local ranges valid)");

    // === Part 2: Perturbed relaxation with force-based convergence ===
    // Methodology (see notes/conventions/relaxation_convergence.md):
    //   1. Run until forces (correction magnitudes) converge below threshold
    //   2. Record trajectory positions in memory
    //   3. Take FINAL converged geometry as reference (NOT the initial geometry)
    //   4. Compute backward displacement |x_step - x_final| for each frame
    eprintln!("\n--- Part 2: perturbed relaxation (GPU, force-based convergence) ---");
    let perturb = 0.3f32;
    let mut perturbed_pos = packed.pos.clone();
    for ig in 0..ng {
        let abase = ig * group_size;
        let o = perturbed_pos[abase];
        for h in 1..=2 {
            let i = abase + h;
            let dx = perturbed_pos[i][0] - o[0]; let dy = perturbed_pos[i][1] - o[1]; let dz = perturbed_pos[i][2] - o[2];
            let len = (dx*dx + dy*dy + dz*dz).sqrt();
            let scale = 1.0 + perturb / len;
            perturbed_pos[i][0] = o[0] + dx * scale; perturbed_pos[i][1] = o[1] + dy * scale; perturbed_pos[i][2] = o[2] + dz * scale;
        }
    }
    sim.upload_state(&perturbed_pos, &inv_mass, Some(&quat))?;
    sim.reset_momentum()?;

    use std::io::Write;
    let f_thresh = 1e-6f32; // force convergence threshold (max correction magnitude)
    let max_steps = 2000;   // safety cap
    let mut traj_frames: Vec<Vec<[f32;4]>> = Vec::new(); // record positions every step
    eprintln!("running step_cluster until max|correction| < {f_thresh} or {max_steps} steps...");
    let mut n_done = 0usize;
    for step in 0..max_steps {
        sim.step_cluster(port_kernel, &cfg)?;
        let (pos4_step, _) = sim.download_pos_quat()?;
        for i in 0..natoms {
            if !packed.is_padding[i] {
                for d in 0..3 { assert!(pos4_step[i][d].is_finite(), "step {step} atom {i} dim {d} not finite: {}", pos4_step[i][d]); }
            }
        }
        traj_frames.push(pos4_step);
        n_done = step + 1;
        // Check force convergence: max correction magnitude across all kernels
        let dpos_coll = sim.download_dpos_coll()?;
        let dpos_node = sim.download_dpos_node()?;
        let max_dpos_coll = dpos_coll.iter().take(natoms).map(|v| v[0].abs().max(v[1].abs()).max(v[2].abs())).fold(0.0f32, f32::max);
        let max_dpos_node = dpos_node.iter().map(|v| v[0].abs().max(v[1].abs()).max(v[2].abs())).fold(0.0f32, f32::max);
        let max_force = max_dpos_coll.max(max_dpos_node);
        if step % 50 == 0 || step < 5 || max_force < f_thresh {
            eprintln!("  step {step}: max|F|={max_force:.3e} (coll={max_dpos_coll:.3e} node={max_dpos_node:.3e})");
        }
        if max_force < f_thresh {
            eprintln!("  CONVERGED at step {step} (max|F|={max_force:.3e} < {f_thresh})");
            break;
        }
    }
    if n_done == max_steps {
        eprintln!("  WARNING: did not converge in {max_steps} steps (force threshold {f_thresh} not reached)");
    }

    // Take final converged geometry as reference
    let final_pos = traj_frames.last().expect("no trajectory frames").clone();
    eprintln!("  final geometry taken as reference ({} frames recorded)", traj_frames.len());

    // Compute backward displacement from final geometry and write TSV + trajectory
    let mut traj_file = traj_path.as_ref().map(|p| std::fs::File::create(p).expect("failed to create traj file"));
    let mut tsv_file = tsv_path.as_ref().map(|p| {
        let mut f = std::fs::File::create(p).expect("failed to create tsv file");
        writeln!(f, "step\tmax_disp_from_final\trms_disp_from_final").expect("tsv write");
        f
    });
    for (step, frame) in traj_frames.iter().enumerate() {
        let mut max_disp = 0.0f32; let mut sum_sq = 0.0f32; let mut n_real = 0u32;
        for i in 0..natoms {
            if packed.is_padding[i] { continue; }
            n_real += 1;
            for d in 0..3 { let dd = frame[i][d] - final_pos[i][d]; max_disp = max_disp.max(dd.abs()); sum_sq += dd * dd; }
        }
        let rms_disp = (sum_sq / n_real as f32).sqrt();
        if let Some(ref mut f) = tsv_file {
            writeln!(f, "{step}\t{max_disp:.10}\t{rms_disp:.10}").expect("tsv write");
        }
        // Save XYZ every 10 steps + first 5 + last frame
        let save_traj = step % 10 == 0 || step < 5 || step == traj_frames.len() - 1;
        if let Some(ref mut f) = traj_file {
            if save_traj {
                let frame_str = xyz_frame(&packed.elems, &packed.is_padding, frame);
                write!(f, "{frame_str}").expect("failed to write traj frame");
            }
        }
    }
    if let Some(ref mut f) = tsv_file { f.flush().expect("tsv flush"); }
    if let Some(ref mut f) = traj_file { f.flush().expect("traj flush"); }
    eprintln!("Part 2: GPU relaxation complete ({n_done} steps, converged to max|F| < {f_thresh})");

    // === Part 3: CPU↔GPU parity (optional) ===
    if do_parity {
        eprintln!("\n--- Part 3: CPU↔GPU parity ---");
        use molff::raff::*;
        let nbcfg = NbConfig { enabled: false, ..Default::default() };
        let n_parity_steps = 500usize;
        let gpu_real_idx: Vec<usize> = (0..ng).flat_map(|ig| { let abase = ig * group_size; (0..3).map(move |h| abase + h) }).collect();
        let extract_gpu = |pos4: &[[f32;4]]| -> Vec<Vec3d> {
            gpu_real_idx.iter().map(|&i| Vec3d::new(pos4[i][0] as f64, pos4[i][1] as f64, pos4[i][2] as f64)).collect()
        };

        // 3A: Memoryless — CPU Adiabatic (Wahba) vs GPU shapematch (Kabsch)
        eprintln!("\n  3A: Memoryless — CPU Adiabatic (Wahba) vs GPU shapematch (Kabsch)");
        let (mut cpu_state_a, cpu_topo_a) = build_cpu_water2(perturb);
        let cpu_cfg_a = RaffConfig {
            orient_mode: OrientMode::Adiabatic, pos_solver: PosSolver::Projective,
            dt: 0.1, cdamp: 0.0, rot_damp: 0.0, flim: 0.0,
            xpbd_iters: 1, xpbd_over_relax: 1.0, pd_inertia: false, vel_reset: false,
            bmix_start: 0.0, bmix_end: 0.0, bmix_istart: 0, bmix_iend: 0,
            box_cfg: BoxCfg::default(), ..Default::default()
        };
        eprintln!("  CPU: running {n_parity_steps} steps (Adiabatic, Projective, 1 inner iter)...");
        let mut cpu_e_a = 0.0;
        for step in 0..n_parity_steps {
            cpu_e_a = step_position_based(&mut cpu_state_a, &cpu_topo_a, &cpu_cfg_a, &nbcfg);
            if step % 100 == 0 { eprintln!("    CPU step {step}: E={cpu_e_a:.6}"); }
        }
        let cpu_pos_a: Vec<Vec3d> = cpu_state_a.pos[0..6].to_vec();

        let mut sim_a = RRsp3::new(natoms, group_size, max_ghosts, true)?;
        sim_a.upload_state(&perturbed_pos, &inv_mass, Some(&quat))?;
        sim_a.upload_radius(&radius)?;
        sim_a.upload_neighs_and_exclusions(&neighs, &excl1, &excl2)?;
        sim_a.upload_cluster_ports(&port_local, &kflat, nnode_per_group)?;
        sim_a.upload_bk_slots(&bk_slots)?;
        sim_a.reset_momentum()?;
        let cfg_a = StepConfig { dt: 0.1, k_coll: 50.0, relaxation: 0.5, bbox_margin: 0.5, ..Default::default() };
        eprintln!("  GPU: running {n_parity_steps} steps (shapematch, massless)...");
        for step in 0..n_parity_steps {
            sim_a.step_cluster(PortKernel::Shapematch, &cfg_a)?;
            if step % 100 == 0 {
                let (p, _) = sim_a.download_pos_quat()?;
                let mut maxd = 0.0f32;
                for i in 0..natoms { if !packed.is_padding[i] { for d in 0..3 { maxd = maxd.max((p[i][d]-packed.pos[i][d]).abs()); } } }
                eprintln!("    GPU step {step}: max|dx|={maxd:.6}");
            }
        }
        let (gpu_pos4_a, _) = sim_a.download_pos_quat()?;
        let gpu_pos_a = extract_gpu(&gpu_pos4_a);
        let rmsd_a = kabsch_rmsd(&cpu_pos_a, &gpu_pos_a);
        eprintln!("  3A RESULT: Kabsch RMSD (CPU Adiabatic vs GPU shapematch) = {rmsd_a:.6} Å");
        eprintln!("  CPU final E = {cpu_e_a:.6}");
        for i in 0..6 {
            let d = ((cpu_pos_a[i].x-gpu_pos_a[i].x).powi(2) + (cpu_pos_a[i].y-gpu_pos_a[i].y).powi(2) + (cpu_pos_a[i].z-gpu_pos_a[i].z).powi(2)).sqrt();
            eprintln!("    atom {i}: CPU=({:.4},{:.4},{:.4}) GPU=({:.4},{:.4},{:.4}) |d|={d:.6}",
                cpu_pos_a[i].x, cpu_pos_a[i].y, cpu_pos_a[i].z, gpu_pos_a[i].x, gpu_pos_a[i].y, gpu_pos_a[i].z);
        }
        if rmsd_a.is_nan() || rmsd_a > 0.1 { eprintln!("  3A: FAIL — RMSD={rmsd_a:.6} > 0.1 Å. Memoryless variants disagree."); }
        else { eprintln!("  3A: PASS — memoryless variants agree (RMSD={rmsd_a:.6} Å)"); }

        // 3B: Massfull — CPU Dynamic (quaternion+inertia) vs GPU current (quaternion+inertia)
        eprintln!("\n  3B: Massfull — CPU Dynamic (quaternion+inertia) vs GPU current (quaternion+inertia)");
        let (mut cpu_state_b, cpu_topo_b) = build_cpu_water2(perturb);
        let cpu_cfg_b = RaffConfig {
            orient_mode: OrientMode::Dynamic, pos_solver: PosSolver::Projective,
            dt: 0.1, cdamp: 0.0, rot_damp: 0.0, flim: 0.0,
            xpbd_iters: 1, xpbd_over_relax: 1.0, pd_inertia: false, vel_reset: false,
            bmix_start: 0.0, bmix_end: 0.0, bmix_istart: 0, bmix_iend: 0,
            box_cfg: BoxCfg::default(), ..Default::default()
        };
        eprintln!("  CPU: running {n_parity_steps} steps (Dynamic, Projective, 1 inner iter)...");
        let mut cpu_e_b = 0.0;
        for step in 0..n_parity_steps {
            cpu_e_b = step_position_based(&mut cpu_state_b, &cpu_topo_b, &cpu_cfg_b, &nbcfg);
            if step % 100 == 0 { eprintln!("    CPU step {step}: E={cpu_e_b:.6}"); }
        }
        let cpu_pos_b: Vec<Vec3d> = cpu_state_b.pos[0..6].to_vec();

        let mut sim_b = RRsp3::new(natoms, group_size, max_ghosts, true)?;
        sim_b.upload_state(&perturbed_pos, &inv_mass, Some(&quat))?;
        sim_b.upload_radius(&radius)?;
        sim_b.upload_neighs_and_exclusions(&neighs, &excl1, &excl2)?;
        sim_b.upload_cluster_ports(&port_local, &kflat, nnode_per_group)?;
        sim_b.upload_bk_slots(&bk_slots)?;
        sim_b.reset_momentum()?;
        let cfg_b = StepConfig { dt: 0.1, k_coll: 50.0, relaxation: 0.5, bbox_margin: 0.5, momentum_beta: 0.0, ..Default::default() };
        eprintln!("  GPU: running {n_parity_steps} steps (current, massfull)...");
        for step in 0..n_parity_steps {
            sim_b.step_cluster(PortKernel::Current, &cfg_b)?;
            if step % 100 == 0 {
                let (p, _) = sim_b.download_pos_quat()?;
                let mut maxd = 0.0f32;
                for i in 0..natoms { if !packed.is_padding[i] { for d in 0..3 { maxd = maxd.max((p[i][d]-packed.pos[i][d]).abs()); } } }
                eprintln!("    GPU step {step}: max|dx|={maxd:.6}");
            }
        }
        let (gpu_pos4_b, _) = sim_b.download_pos_quat()?;
        let gpu_pos_b = extract_gpu(&gpu_pos4_b);
        let rmsd_b = kabsch_rmsd(&cpu_pos_b, &gpu_pos_b);
        eprintln!("  3B RESULT: Kabsch RMSD (CPU Dynamic vs GPU current) = {rmsd_b:.6} Å");
        eprintln!("  CPU final E = {cpu_e_b:.6}");
        for i in 0..6 {
            let d = ((cpu_pos_b[i].x-gpu_pos_b[i].x).powi(2) + (cpu_pos_b[i].y-gpu_pos_b[i].y).powi(2) + (cpu_pos_b[i].z-gpu_pos_b[i].z).powi(2)).sqrt();
            eprintln!("    atom {i}: CPU=({:.4},{:.4},{:.4}) GPU=({:.4},{:.4},{:.4}) |d|={d:.6}",
                cpu_pos_b[i].x, cpu_pos_b[i].y, cpu_pos_b[i].z, gpu_pos_b[i].x, gpu_pos_b[i].y, gpu_pos_b[i].z);
        }
        if rmsd_b.is_nan() || rmsd_b > 0.1 { eprintln!("  3B: FAIL — RMSD={rmsd_b:.6} > 0.1 Å. Massfull variants disagree."); }
        else { eprintln!("  3B: PASS — massfull variants agree (RMSD={rmsd_b:.6} Å)"); }
        println!("parity: memoryless_rmsd={rmsd_a:.6} massfull_rmsd={rmsd_b:.6} cpu_e_memless={cpu_e_a:.6} cpu_e_massfull={cpu_e_b:.6}");
    }

    eprintln!("\n=== raff_ocl_smoke: ALL CHECKS PASSED ===");
    Ok(())
}
