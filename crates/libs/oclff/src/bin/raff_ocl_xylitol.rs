//! raff_ocl_xylitol — RRsp3 GPU relaxation of xylitol (21 atoms, 5C+5O+11H).
//!
//! Loads xylitol from data/xyz/xylitol.xyz, detects bonds via covalent radii,
//! packs 4 copies into separate workgroups, perturbs, and relaxes on GPU.
//! Saves trajectory (.xyz) and convergence (.tsv) to debug/raff_ocl_xylitol/.
//!
//! Methodology: see notes/conventions/relaxation_convergence.md
//!   - Run until forces converge below threshold
//!   - Take final converged geometry as reference
//!   - Plot backward displacement from final
//!
//! Usage: cargo run -p oclff --bin raff_ocl_xylitol [--port current|shapematch] [--n_copies N]

use oclff::pack::{pack_molecules, MolInput, build_neighs_from_bonds, make_exclusions_1st_2nd, make_bk_slots_clustered, make_ports_from_neighs, masses_from_elems};
use oclff::rrsp3::{RRsp3, PortKernel, StepConfig};
use moltopo::xyz;
use moltopo::builder::Builder;
use numtypes::Vec3d;
use std::path::Path;
use std::io::Write;

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
    // Count nodes (degree > 1) — matches pack.rs inference
    let mut deg = vec![0i32; natoms];
    for &(i,j) in &bonds { deg[i] += 1; deg[j] += 1; }
    let nnode = deg.iter().filter(|&&d| d > 1).count();
    eprintln!("xylitol: {natoms} atoms, {} bonds, {nnode} nodes (degree>1), {} caps (degree<=1)",
        bonds.len(), natoms - nnode);
    // Print node elements
    let nodes: Vec<&String> = (0..natoms).filter(|&i| deg[i] > 1).map(|i| &sys.elems[i]).collect();
    eprintln!("  nodes: {:?}", nodes);
    (pos_f32, bonds, sys.elems, nnode)
}

fn parse_args() -> (PortKernel, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut port = PortKernel::Current;
    let mut n_copies = 4usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n_copies" => { i += 1; if i < args.len() { n_copies = args[i].parse().unwrap_or(4); } }
            s if !s.starts_with("--") => { port = PortKernel::from_str(s); }
            _ => {}
        }
        i += 1;
    }
    (port, n_copies)
}

fn main() -> ocl::Result<()> {
    let (port_kernel, n_copies) = parse_args();
    eprintln!("=== raff_ocl_xylitol: port={port_kernel:?}, n_copies={n_copies} ===");

    let group_size = 64usize;
    let (pos_xyl, bonds, elems, nnode) = load_xylitol();
    let natoms_per = pos_xyl.len();
    assert!(natoms_per <= group_size, "xylitol has {natoms_per} atoms > group_size={group_size}");

    // Pack n_copies of xylitol, each shifted along x
    let spacing = 15.0f32; // Å between copies
    let mols: Vec<MolInput> = (0..n_copies).map(|i| {
        let shift = [spacing * i as f32, 0.0, 0.0];
        MolInput {
            elems: elems.clone(),
            pos: pos_xyl.iter().map(|p| [p[0]+shift[0], p[1]+shift[1], p[2]+shift[2]]).collect(),
            bonds: bonds.clone(),
            nnode,
        }
    }).collect();
    let packed = pack_molecules(&mols, group_size);
    let natoms = packed.natoms;
    let ng = packed.num_groups;
    eprintln!("packed: natoms={natoms}, ng={ng}, nnode_per_group={:?}", packed.nnode_per_group);
    assert!(ng == n_copies, "expected {n_copies} groups, got {ng}");

    // Masses + inv masses
    let masses = masses_from_elems(&packed);
    let inv_mass: Vec<f32> = (0..natoms).map(|i| if packed.is_padding[i] { 0.0 } else { 1.0 / masses[i] }).collect();

    // Neighs + exclusions
    let neighs = build_neighs_from_bonds(natoms, &packed.bonds);
    let nnode_per_group = packed.nnode_per_group[0] as i32;
    assert!(packed.nnode_per_group.iter().all(|&n| n == nnode_per_group), "constant nnode_per_group required, got {:?}", packed.nnode_per_group);
    let (excl1, excl2) = make_exclusions_1st_2nd(&neighs, natoms);

    // Ports + stiffness (K=200)
    let (port_local, kflat) = make_ports_from_neighs(&packed.pos, &neighs, natoms, 200.0);

    // Radius=0: disable collisions. The collision kernel uses geometric overlap
    // (rsum = ri + rj), not k_coll. Xylitol has many 1-4 pairs within 2.0 Å
    // that are NOT excluded by make_exclusions_1st_2nd (only 1st+2nd).
    // TODO: add 1-4 exclusions for proper intra-molecular collisions.
    let radius: Vec<f32> = vec![0.0; natoms];

    // bk_slots
    let bk_slots = make_bk_slots_clustered(&neighs, group_size, &packed.nnode_per_group, natoms);

    // Create GPU harness
    let max_ghosts = 64usize;
    let mut sim = RRsp3::new(natoms, group_size, max_ghosts, true)?;
    eprintln!("OpenCL device: {}", sim.device_name());

    // Perturb: displace all non-H atoms by random small amount, H atoms more
    let perturb = 0.2f32;
    let mut perturbed_pos = packed.pos.clone();
    let mut seed = 12345u64;
    for i in 0..natoms {
        if packed.is_padding[i] { continue; }
        let is_h = packed.elems[i] == "H";
        let amp = if is_h { perturb * 1.5 } else { perturb };
        for d in 0..3 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            perturbed_pos[i][d] += amp * r * 2.0;
        }
    }

    let quat: Vec<[f32; 4]> = (0..natoms).map(|_| [0.0, 0.0, 0.0, 1.0]).collect();
    sim.upload_state(&perturbed_pos, &inv_mass, Some(&quat))?;
    sim.upload_radius(&radius)?;
    sim.upload_neighs_and_exclusions(&neighs, &excl1, &excl2)?;
    sim.upload_cluster_ports(&port_local, &kflat, nnode_per_group)?;
    sim.upload_bk_slots(&bk_slots)?;

    // === Relaxation with force-based convergence ===
    // k_coll=0: disable collisions — xylitol has 1-4 pairs within collision radius
    // that are not excluded by make_exclusions_1st_2nd (only 1st+2nd). Collisions
    // would fight the port springs. TODO: add 1-4 exclusions for proper intra-molecular collisions.
    let cfg = StepConfig { dt: 0.1, k_coll: 0.0, relaxation: 0.5, bbox_margin: 0.5, ..Default::default() };
    let f_thresh = 1e-6f32;
    let max_steps = 5000;
    let mut traj_frames: Vec<Vec<[f32;4]>> = Vec::new();
    eprintln!("relaxing until max|correction| < {f_thresh} or {max_steps} steps...");
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
        let dpos_coll = sim.download_dpos_coll()?;
        let dpos_node = sim.download_dpos_node()?;
        let max_dpos_coll = dpos_coll.iter().take(natoms).map(|v| v[0].abs().max(v[1].abs()).max(v[2].abs())).fold(0.0f32, f32::max);
        let max_dpos_node = dpos_node.iter().map(|v| v[0].abs().max(v[1].abs()).max(v[2].abs())).fold(0.0f32, f32::max);
        let max_force = max_dpos_coll.max(max_dpos_node);
        if step % 100 == 0 || step < 5 || max_force < f_thresh {
            eprintln!("  step {step}: max|F|={max_force:.3e} (coll={max_dpos_coll:.3e} node={max_dpos_node:.3e})");
        }
        if max_force < f_thresh {
            eprintln!("  CONVERGED at step {step} (max|F|={max_force:.3e} < {f_thresh})");
            break;
        }
    }
    if n_done == max_steps {
        eprintln!("  WARNING: did not converge in {max_steps} steps");
    }

    // Final geometry as reference
    let final_pos = traj_frames.last().expect("no frames").clone();
    eprintln!("  final geometry taken as reference ({} frames)", traj_frames.len());

    // Write outputs to debug/raff_ocl_xylitol/
    let outdir = Path::new("debug/raff_ocl_xylitol");
    std::fs::create_dir_all(outdir).expect("create debug dir");
    let traj_path = outdir.join("traj.xyz");
    let tsv_path = outdir.join("convergence.tsv");
    let mut traj_file = std::fs::File::create(&traj_path).expect("create traj");
    let mut tsv_file = std::fs::File::create(&tsv_path).expect("create tsv");
    writeln!(tsv_file, "step\tmax_disp_from_final\trms_disp_from_final").expect("tsv write");

    for (step, frame) in traj_frames.iter().enumerate() {
        let mut max_disp = 0.0f32; let mut sum_sq = 0.0f32; let mut n_real = 0u32;
        for i in 0..natoms {
            if packed.is_padding[i] { continue; }
            n_real += 1;
            for d in 0..3 { let dd = frame[i][d] - final_pos[i][d]; max_disp = max_disp.max(dd.abs()); sum_sq += dd * dd; }
        }
        let rms_disp = (sum_sq / n_real as f32).sqrt();
        writeln!(tsv_file, "{step}\t{max_disp:.10}\t{rms_disp:.10}").expect("tsv write");
        // Save XYZ every 20 steps + first 5 + last
        if step % 20 == 0 || step < 5 || step == traj_frames.len() - 1 {
            let n_real = packed.is_padding.iter().filter(|&&p| !p).count();
            write!(traj_file, "{n_real}\nstep {step}\n").expect("traj write");
            for i in 0..natoms {
                if packed.is_padding[i] { continue; }
                write!(traj_file, "{} {:.6} {:.6} {:.6}\n", packed.elems[i], frame[i][0], frame[i][1], frame[i][2]).expect("traj write");
            }
        }
    }
    tsv_file.flush().expect("tsv flush");
    traj_file.flush().expect("traj flush");
    eprintln!("saved: {} ({} frames), {} ({} steps)", traj_path.display(), traj_frames.len(), tsv_path.display(), n_done);

    // Also save the initial perturbed and final equilibrium as separate XYZ for comparison
    let init_path = outdir.join("initial_perturbed.xyz");
    let final_path = outdir.join("final_equilibrium.xyz");
    for (path, pos, label) in [(&init_path, &traj_frames[0], "initial"), (&final_path, &final_pos, "final")] {
        let mut f = std::fs::File::create(path).expect("create xyz");
        let n_real = packed.is_padding.iter().filter(|&&p| !p).count();
        write!(f, "{n_real}\n{label}\n").expect("xyz write");
        for i in 0..natoms {
            if packed.is_padding[i] { continue; }
            write!(f, "{} {:.6} {:.6} {:.6}\n", packed.elems[i], pos[i][0], pos[i][1], pos[i][2]).expect("xyz write");
        }
    }
    eprintln!("saved: {} , {}", init_path.display(), final_path.display());
    eprintln!("\n=== raff_ocl_xylitol: DONE ({n_done} steps) ===");
    Ok(())
}
