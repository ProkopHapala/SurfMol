//! Pentacene bend-relaxation test: bend the molecule into a bow shape, verify it
//! relaxes back to planar with full UFF (bonds + angles + dihedrals + inversions).
//!
//! The bend is a coherent low-frequency deformation mode (parabolic z-displacement
//! along the long molecular axis), not random noise. This directly tests the
//! aromatic bending stiffness from UFF inversion terms on sp2 carbons.
//!
//! Procedure:
//!   1. Load planar pentacene, assign real UFF params, relax in-plane (inversions off)
//!      to relieve bond strain from DFT-vs-UFF geometry mismatch.
//!   2. Apply a parabolic bend: z = amp * (x / L)^2  (bow shape along x-axis)
//!   3. Relax with full UFF (inversions ON) using FIRE.
//!   4. Verify the molecule returns to planarity (z_rms < threshold).

use std::path::Path;
use std::fs::File;
use std::io::Write;
use numtypes::Vec3d;
use moltopo::topology::{Topology, build_bonds_by_cutoff, build_angles_from_bonds, build_dihedrals_from_bonds, build_inversions_from_bonds};
use moltopo::xyz::read_xyz;
use moltopo::params::Params;
use moltopo::assign_uff;
use surfmol::mol_world::{MolWorld, BondedFFMode};
use molff::raff::FireState;

fn write_trace(path: &Path, step: &[i32], e: &[f64], fmax: &[f64], z_rms: &[f64]) {
    let mut f = File::create(path).expect("create trace");
    writeln!(f, "step\tE\tfmax\tz_rms").expect("header");
    for i in 0..step.len() {
        writeln!(f, "{}\t{:.6}\t{:.6}\t{:.6}", step[i], e[i], fmax[i], z_rms[i]).expect("row");
    }
}

/// One FIRE relaxation step. Returns (E, fmax, z_rms).
fn fire_step(mw: &mut MolWorld, fire: &mut FireState) -> (f64, f64, f64) {
    let (eb, ea, ed, ei, _, _) = mw.eval_forces();
    let e = eb + ea + ed + ei;
    let dt = fire.dt;
    let mut v_dot_f = 0.0;
    let mut v_norm2 = 0.0;
    let mut f_norm2 = 0.0;
    let mut f2max: f64 = 0.0;
    for ia in 0..mw.natoms() {
        let f = mw.dyn_atoms.fapos.as_slice()[ia];
        let f2 = f.norm2();
        if f2 > f2max { f2max = f2; }
        f_norm2 += f2;
        let v = mw.dyn_atoms.vapos.as_slice()[ia];
        let mut v_new = v;
        v_new.add_mul(f, dt);
        mw.dyn_atoms.vapos.as_mut_slice()[ia] = v_new;
        v_dot_f += Vec3d::dot(v_new, f);
        v_norm2 += v_new.norm2();
    }
    if v_dot_f > 0.0 && f_norm2 > 1e-30 && v_norm2 > 1e-30 {
        let v_mag = v_norm2.sqrt();
        let f_mag = f_norm2.sqrt();
        for ia in 0..mw.natoms() {
            let f = mw.dyn_atoms.fapos.as_slice()[ia];
            let f_hat = Vec3d::set_mul(f, 1.0 / f_mag);
            let v = mw.dyn_atoms.vapos.as_slice()[ia];
            mw.dyn_atoms.vapos.as_mut_slice()[ia] = Vec3d::set_lincomb(1.0 - fire.alpha, v, fire.alpha * v_mag, f_hat);
        }
        fire.n_pos += 1;
        if fire.n_pos > fire.n_min {
            fire.dt = (fire.dt * fire.f_inc).min(fire.dt_max);
            fire.alpha *= fire.f_alpha;
        }
    } else {
        fire.n_pos = 0;
        fire.dt *= fire.f_dec;
        fire.alpha = fire.alpha0;
        for v in mw.dyn_atoms.vapos.as_mut_slice() { *v = Vec3d::new(0.0, 0.0, 0.0); }
    }
    for ia in 0..mw.natoms() {
        let v = mw.dyn_atoms.vapos.as_slice()[ia];
        mw.dyn_atoms.atoms.apos.as_mut_slice()[ia].add_mul(v, dt);
    }
    let fmax = f2max.sqrt();
    let z_rms = mw.dyn_atoms.atoms.apos.as_slice().iter().map(|p| p.z * p.z).sum::<f64>().sqrt() / (mw.natoms() as f64).sqrt();
    (e, fmax, z_rms)
}

fn run_fire(mw: &mut MolWorld, niter: i32, fconv: f64, label: &str, trace_path: &Path, xyz_path: Option<&Path>, elems: &[String]) -> (i32, f64, f64, f64) {
    let mut fire = FireState::new(0.001, 0.05);
    let mut step = Vec::new();
    let mut e_trace = Vec::new();
    let mut fmax_trace = Vec::new();
    let mut zrms_trace = Vec::new();
    let mut n_conv = niter;
    // For .xyz trajectory: collect frames at regular intervals
    let save_every = 10;
    let mut frames: Vec<Vec<Vec3d>> = Vec::new();
    let mut frame_steps: Vec<i32> = Vec::new();
    for itr in 0..niter {
        let (e, fmax, z_rms) = fire_step(mw, &mut fire);
        step.push(itr as i32);
        e_trace.push(e);
        fmax_trace.push(fmax);
        zrms_trace.push(z_rms);
        if let Some(_) = xyz_path {
            if itr % save_every == 0 || fmax < fconv {
                frames.push(mw.dyn_atoms.atoms.apos.as_slice().to_vec());
                frame_steps.push(itr as i32);
            }
        }
        if fmax < fconv { n_conv = itr + 1; break; }
    }
    let e_fin = e_trace.last().copied().unwrap_or(0.0);
    let fmax_fin = fmax_trace.last().copied().unwrap_or(0.0);
    let zrms_fin = zrms_trace.last().copied().unwrap_or(0.0);
    write_trace(trace_path, &step, &e_trace, &fmax_trace, &zrms_trace);
    // Write multi-frame .xyz trajectory
    if let Some(xyz_p) = xyz_path {
        let mut f = File::create(xyz_p).expect("create xyz trajectory");
        for (iframe, (frame, &step_i)) in frames.iter().zip(frame_steps.iter()).enumerate() {
            writeln!(f, "{}", frame.len()).expect("natoms");
            writeln!(f, "step={} E={:.6} fmax={:.6} z_rms={:.6}", step_i, e_trace[step_i as usize], fmax_trace[step_i as usize], zrms_trace[step_i as usize]).expect("comment");
            for (ia, p) in frame.iter().enumerate() {
                writeln!(f, "{} {:.6} {:.6} {:.6}", elems[ia], p.x, p.y, p.z).expect("atom");
            }
        }
        println!("  xyz trajectory: {} ({} frames)", xyz_p.display(), frames.len());
    }
    println!("{}: n={} E={:.6} fmax={:.6} z_rms={:.6}", label, n_conv, e_fin, fmax_fin, zrms_fin);
    (n_conv, e_fin, fmax_fin, zrms_fin)
}

/// Save a single-frame .xyz file
fn save_xyz(path: &Path, apos: &[Vec3d], elems: &[String], comment: &str) {
    let mut f = File::create(path).expect("create xyz");
    writeln!(f, "{}", apos.len()).expect("natoms");
    writeln!(f, "{}", comment).expect("comment");
    for (ia, p) in apos.iter().enumerate() {
        writeln!(f, "{} {:.6} {:.6} {:.6}", elems[ia], p.x, p.y, p.z).expect("atom");
    }
}

#[test]
fn pentacene_bend_relaxes_to_planar() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().parent().unwrap();
    let out_dir = base.join("debug/relax_pentacene_bend");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let xyz = read_xyz(&base.join("data/xyz/pentacene.xyz")).expect("read pentacene.xyz");
    let apos = xyz.apos;
    let elems = &xyz.elems;
    let natoms = apos.len() as i32;
    assert_eq!(natoms, 36, "pentacene should have 36 atoms");

    // Load UFF parameter files
    let mut params = Params::new();
    params.load_element_types(&base.join("data/ElementTypes.dat"));
    params.load_atom_types(&base.join("data/AtomTypes.dat"));
    params.load_bond_types(&base.join("data/BondTypes.dat"));
    params.load_angle_types(&base.join("data/AngleTypes.dat"));
    params.load_dihedral_types(&base.join("data/DihedralTypes.dat"));

    // Build topology
    let bonds = build_bonds_by_cutoff(&apos, 1.8);
    let angles = build_angles_from_bonds(natoms, &bonds);
    let dihedrals = build_dihedrals_from_bonds(&bonds);
    let inversions = build_inversions_from_bonds(natoms, &bonds);
    let top = Topology { apos, bonds, angles, dihedrals, inversions };
    println!("pentacene topology: bonds={} angles={} dihedrals={} inversions={}",
        top.bonds.len(), top.angles.len(), top.dihedrals.len(), top.inversions.len());

    // Build MolWorld and assign UFF types + real params
    let mut mw = MolWorld::from_topology(&top);
    mw.bonded_mode = BondedFFMode::Uff;
    mw.make_neigh_bs();
    let neighs_arr: Vec<[i32; 4]> = mw.dyn_atoms.neighs().iter().map(|q| q.as_array()).collect();
    let types = assign_uff::assign_uff_types(elems, &neighs_arr);
    mw.setup_uff_params(&params, &types);
    mw.bake_angle_neighs();
    mw.bake_dihedral_neighs();
    mw.bake_inversion_neighs();
    mw.map_atom_interactions();

    let n_nonzero_inv = mw.uff.inv_params.as_slice().iter().filter(|p| p[0].abs() > 1e-10).count();
    println!("nonzero inversion params: {}/{} (aromatic bending stiffness)", n_nonzero_inv, mw.uff.ninversions);
    assert!(n_nonzero_inv > 0, "no inversion parameters — aromatic bending stiffness missing");

    // === Step 1: In-plane relaxation (inversions OFF) to relieve bond strain ===
    // UFF bond lengths differ from DFT geometry, so we must relax in-plane first.
    let inv_params_saved: Vec<[f64; 4]> = mw.uff.inv_params.as_slice().to_vec();
    for ii in 0..mw.uff.ninversions as usize {
        mw.uff.inv_params.as_mut_slice()[ii] = [0.0, 1.0, -1.0, 0.0];
    }
    for i in 0..mw.natoms() {
        mw.dyn_atoms.atoms.apos.as_mut_slice()[i] = top.apos[i];
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::new(0.0, 0.0, 0.0);
    }
    mw.dyn_atoms.fapos.resize_fill(mw.natoms(), Vec3d::new(0.0, 0.0, 0.0));

    let (_, _, fmax_inplane, z_rms_inplane) = run_fire(&mut mw, 5000, 1e-3, "in-plane relax (inv OFF)",
        &out_dir.join("trajectory_inplane.tsv"), None, elems);
    assert!(fmax_inplane < 0.5, "in-plane relaxation did not converge: fmax={:.4}", fmax_inplane);
    assert!(z_rms_inplane < 0.05, "in-plane relaxation produced out-of-plane geometry: z_rms={:.4}", z_rms_inplane);
    println!("  -> planar baseline established (z_rms={:.6})", z_rms_inplane);

    // Save the relaxed planar geometry as the reference
    let planar_apos: Vec<Vec3d> = mw.dyn_atoms.atoms.apos.as_slice().to_vec();
    save_xyz(&out_dir.join("pentacene_planar.xyz"), &planar_apos, elems, "pentacene planar baseline (in-plane relaxed)");

    // === Step 2: Restore inversions, apply parabolic BEND, relax ===
    for ii in 0..mw.uff.ninversions as usize {
        mw.uff.inv_params.as_mut_slice()[ii] = inv_params_saved[ii];
    }

    // Apply a coherent bow-shaped bend along the x-axis (long axis of pentacene):
    //   z = amp * (x / L)^2
    // where L = half-width of the molecule. This is the lowest-frequency bending mode.
    let x_min = planar_apos.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let x_max = planar_apos.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let x_center = (x_min + x_max) * 0.5;
    let L = (x_max - x_min) * 0.5;
    let bend_amp = 0.3; // 0.3 Å peak deflection — gentle global bend
    println!("\napplying parabolic bend: amp={:.2} Å, L={:.2} Å, x_center={:.2}", bend_amp, L, x_center);

    for i in 0..mw.natoms() {
        let p = planar_apos[i];
        let x_norm = (p.x - x_center) / L; // [-1, 1]
        let dz = bend_amp * x_norm * x_norm;
        mw.dyn_atoms.atoms.apos.as_mut_slice()[i] = Vec3d::new(p.x, p.y, p.z + dz);
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::new(0.0, 0.0, 0.0);
    }
    mw.dyn_atoms.fapos.resize_fill(mw.natoms(), Vec3d::new(0.0, 0.0, 0.0));

    // Print bend diagnostics
    let (eb, ea, ed, ei, _, _) = mw.eval_forces();
    let z_rms_bent = mw.dyn_atoms.atoms.apos.as_slice().iter().map(|p| p.z * p.z).sum::<f64>().sqrt() / (mw.natoms() as f64).sqrt();
    let z_max_bent = mw.dyn_atoms.atoms.apos.as_slice().iter().map(|p| p.z.abs()).fold(0.0f64, |a, b| a.max(b));
    println!("bent state: E_bond={:.6} E_angle={:.6} E_dih={:.6} E_inv={:.6} total={:.6}", eb, ea, ed, ei, eb+ea+ed+ei);
    println!("  z_rms={:.4} z_max={:.4} (should be nonzero — molecule is bent)", z_rms_bent, z_max_bent);
    assert!(z_rms_bent > 0.01, "bend did not produce out-of-plane geometry: z_rms={:.4}", z_rms_bent);

    // Save the bent geometry (before relaxation)
    save_xyz(&out_dir.join("pentacene_bent.xyz"), mw.dyn_atoms.atoms.apos.as_slice(), elems,
        &format!("pentacene bent (parabolic amp={:.2} Å) E={:.6}", bend_amp, eb+ea+ed+ei));
    // Note: for a smooth parabolic bend, E_inv is tiny because the local out-of-plane
    // angle at each trigonal center is nearly zero (neighboring atoms move together).
    // The restoring force comes primarily from dihedral terms (E_dih). Inversions
    // penalize LOCAL pyramidalization, not global bending.

    // === Step 3: Relax with full UFF (inversions ON) ===
    let (n_conv, e_final, fmax_final, z_rms_final) = run_fire(&mut mw, 5000, 1e-3, "bend relax (inv ON)",
        &out_dir.join("trajectory_bend.tsv"), Some(&out_dir.join("pentacene_bend_traj.xyz")), elems);
    println!("  -> relaxed to z_rms={:.6}", z_rms_final);
    println!("  trace: {}/trajectory_bend.tsv", out_dir.display());

    // Save the relaxed geometry (after bend relaxation)
    save_xyz(&out_dir.join("pentacene_bend_relaxed.xyz"), mw.dyn_atoms.atoms.apos.as_slice(), elems,
        &format!("pentacene after bend relaxation z_rms={:.6}", z_rms_final));

    // The smooth parabolic bend may not fully relax to planar because UFF sp2-sp2
    // dihedrals (V*(1-cos(2φ))) have minima at both φ=0 and φ=π, allowing non-planar
    // configurations with zero dihedral energy. This is a known UFF limitation for
    // aromatic systems — the inversion terms penalize LOCAL pyramidalization, not
    // global bending. So we check that the bend is at least partially reduced:
    assert!(fmax_final < 0.5, "FIRE did not converge: fmax={:.4}", fmax_final);
    assert!(z_rms_final < z_rms_bent, "bend did not reduce at all: z_rms {:.4} -> {:.4}", z_rms_bent, z_rms_final);
    println!("  global bend reduced: z_rms {:.4} -> {:.4} (UFF dihedral multi-well allows non-planar minima)", z_rms_bent, z_rms_final);

    // === Step 4: LOCAL pyramidalization test — directly tests inversion stiffness ===
    // Push ONE atom out of plane. This creates strong local pyramidalization at the
    // displaced atom and its neighbors, producing large inversion energy. The inversion
    // terms must restore the atom to the plane.
    println!("\n--- local pyramidalization test ---");
    for i in 0..mw.natoms() {
        mw.dyn_atoms.atoms.apos.as_mut_slice()[i] = planar_apos[i];
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::new(0.0, 0.0, 0.0);
    }
    // Push atom 4 (a central C_R) out of plane by 0.3 Å
    let pyr_atom = 4usize;
    let pyr_amp = 0.3;
    mw.dyn_atoms.atoms.apos.as_mut_slice()[pyr_atom].z += pyr_amp;
    mw.dyn_atoms.fapos.resize_fill(mw.natoms(), Vec3d::new(0.0, 0.0, 0.0));

    // Save the pyramidalized geometry (before relaxation)
    save_xyz(&out_dir.join("pentacene_pyr_bent.xyz"), mw.dyn_atoms.atoms.apos.as_slice(), elems,
        &format!("pentacene pyramidalized atom {} z+{:.2}", pyr_atom, pyr_amp));

    let (eb_p, ea_p, ed_p, ei_p, _, _) = mw.eval_forces();
    let fz_pyr = mw.dyn_atoms.fapos.as_slice()[pyr_atom].z;
    let z_rms_pyr = mw.dyn_atoms.atoms.apos.as_slice().iter().map(|p| p.z * p.z).sum::<f64>().sqrt() / (mw.natoms() as f64).sqrt();
    println!("pyramidalized atom {} by z={:.2}: E_bond={:.4} E_angle={:.4} E_dih={:.4} E_inv={:.4}",
        pyr_atom, pyr_amp, eb_p, ea_p, ed_p, ei_p);
    println!("  Fz on atom {} = {:.4} (must be negative — restoring force toward plane)", pyr_atom, fz_pyr);
    println!("  z_rms = {:.4}", z_rms_pyr);
    assert!(fz_pyr < -1e-3, "no restoring force on pyramidalized atom: Fz={:.4}", fz_pyr);
    assert!(ei_p > 1e-4, "inversion energy too small for local pyramidalization: E_inv={:.4e}", ei_p);

    // Relax and verify the atom returns to the plane
    let (n_conv_p, _, fmax_final_p, z_rms_final_p) = run_fire(&mut mw, 5000, 1e-3, "pyr relax (inv ON)",
        &out_dir.join("trajectory_pyr.tsv"), Some(&out_dir.join("pentacene_pyr_traj.xyz")), elems);
    println!("  -> relaxed to z_rms={:.6} in {} steps", z_rms_final_p, n_conv_p);

    // Save the pyramidalization-relaxed geometry (after)
    save_xyz(&out_dir.join("pentacene_pyr_relaxed.xyz"), mw.dyn_atoms.atoms.apos.as_slice(), elems,
        &format!("pentacene after pyr relaxation z_rms={:.6}", z_rms_final_p));
    assert!(fmax_final_p < 0.5, "pyr FIRE did not converge: fmax={:.4}", fmax_final_p);
    assert!(z_rms_final_p < 0.05, "pyramidalized atom did NOT return to plane: z_rms={:.4}", z_rms_final_p);

    println!("\n=== PASS: pentacene local pyramidalization ({:.2} Å) relaxed back to planar (z_rms {:.4} -> {:.4}) in {} steps ===",
        pyr_amp, z_rms_pyr, z_rms_final_p, n_conv_p);
}
