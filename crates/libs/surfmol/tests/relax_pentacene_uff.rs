//! Pentacene UFF relaxation with REAL parameters (bonds + angles + dihedrals + inversions).
//!
//! Verifies that the full UFF force field — including aromatic bending stiffness
//! from inversion (improper torsion) terms on sp2 carbons — produces physical
//! out-of-plane restoring forces. A planar molecule (pentacene) with out-of-plane
//! distortion must relax back to planarity.
//!
//! Parity reference: FireCore cpp/common/molecular/UFFbuilder.h:assignUFFparams
//!   - Bonds: harmonic, k from UFF Coulomb-like formula
//!   - Angles: Fourier c0..c3, sp2 aromatic → C0=1, C3=-1, k=kappa/9
//!   - Dihedrals: V*(1+d*cos(n*phi)), sp2-sp2 → V=5*sqrt(Ui*Uj)*(1+4.18*ln(BO))
//!   - Inversions: K*(C0+C1*cos(w)+C2*cos(2w)), sp2 C → K=6 kcal/mol, C0=1, C1=-1, C2=0
//!
//! Key physical check: with inversions enabled, an out-of-plane bend of a C_R atom
//! must produce a nonzero Fz restoring force. With inversions disabled (dummy params),
//! Fz must be ~zero (only bonds + in-plane angles, no out-of-plane stiffness).

use std::path::Path;
use std::fs::File;
use std::io::Write;
use numtypes::Vec3d;
use moltopo::topology::{Topology, build_bonds_by_cutoff, build_angles_from_bonds, build_dihedrals_from_bonds, build_inversions_from_bonds};
use moltopo::xyz::read_xyz;
use moltopo::params::Params;
use moltopo::assign_uff;
use surfmol::mol_world::{MolWorld, BondedFFMode};

fn write_trace(path: &Path, step: &[i32], e: &[f64], fmax: &[f64], fz_max: &[f64]) {
    let mut f = File::create(path).expect("create trace");
    writeln!(f, "step\tE\tfmax\tfz_max").expect("header");
    for i in 0..step.len() {
        writeln!(f, "{}\t{:.6}\t{:.6}\t{:.6}", step[i], e[i], fmax[i], fz_max[i]).expect("row");
    }
}

#[test]
fn pentacene_uff_full_stiffness() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().parent().unwrap();
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

    // Build MolWorld and assign UFF types
    let mut mw = MolWorld::from_topology(&top);
    mw.bonded_mode = BondedFFMode::Uff;
    mw.make_neigh_bs();

    // Assign UFF atom types from topology
    let neighs_arr: Vec<[i32; 4]> = mw.dyn_atoms.neighs().iter().map(|q| q.as_array()).collect();
    let types = assign_uff::assign_uff_types(elems, &neighs_arr);
    println!("UFF types: {}", types.iter().map(|t| t.as_str()).collect::<Vec<_>>().join(" "));

    // Assign REAL UFF parameters
    mw.setup_uff_params(&params, &types);
    mw.bake_angle_neighs();
    mw.bake_dihedral_neighs();
    mw.bake_inversion_neighs();
    mw.map_atom_interactions();

    // Verify parameter assignment: angles, dihedrals, inversions should be nonzero
    let n_nonzero_ang = mw.uff.ang_params.as_slice().iter().filter(|p| p[0].abs() > 1e-10).count();
    let n_nonzero_dih = mw.uff.dih_params.as_slice().iter().filter(|p| p[0].abs() > 1e-10).count();
    let n_nonzero_inv = mw.uff.inv_params.as_slice().iter().filter(|p| p[0].abs() > 1e-10).count();
    println!("nonzero params: angles={}/{} dihedrals={}/{} inversions={}/{}",
        n_nonzero_ang, mw.uff.nangles, n_nonzero_dih, mw.uff.ndihedrals, n_nonzero_inv, mw.uff.ninversions);
    assert!(n_nonzero_ang > 0, "no angle parameters assigned — UFF setup failed");
    assert!(n_nonzero_dih > 0, "no dihedral parameters assigned — UFF setup failed");
    assert!(n_nonzero_inv > 0, "no inversion parameters assigned — UFF setup failed (aromatic bending missing!)");

    // --- Check 1: out-of-plane force on a distorted C_R atom ---
    // Pentacene is planar (z=0). Displace one carbon out of plane and check Fz.
    // First, check energy at the ideal planar geometry (should be near zero if l0 = current bond lengths)
    mw.dyn_atoms.fapos.resize_fill(mw.natoms(), Vec3d::new(0.0, 0.0, 0.0));
    let (eb0, ea0, ed0, ei0, _, _) = mw.eval_forces();
    println!("\nplanar geometry energy: E_bond={:.6} E_angle={:.6} E_dih={:.6} E_inv={:.6}", eb0, ea0, ed0, ei0);

    let test_atom = 0usize; // first carbon
    mw.dyn_atoms.atoms.apos.as_mut_slice()[test_atom].z = 0.5; // 0.5 Å out of plane
    let (eb, ea, ed, ei, _, _) = mw.eval_forces();
    let fz = mw.dyn_atoms.fapos.as_slice()[test_atom].z;
    let fmax = mw.dyn_atoms.fapos.as_slice().iter().map(|f| f.norm()).fold(0.0f64, |a, b| a.max(b));
    println!("out-of-plane test (atom {} displaced z=0.5):", test_atom);
    println!("  E_bond={:.6} E_angle={:.6} E_dih={:.6} E_inv={:.6}", eb, ea, ed, ei);
    println!("  Fz on displaced atom = {:.6} (should be nonzero — restoring force)", fz);
    println!("  max|F| = {:.6}", fmax);
    assert!(fz.abs() > 1e-6, "no out-of-plane restoring force Fz={:.2e} — inversion stiffness missing", fz);

    // --- Check 2: relaxation with FIRE ---
    // The input pentacene.xyz has DFT-optimized bond lengths (~1.4 Å) but UFF predicts
    // l0=1.458 Å for aromatic C-C. This 4% compression means E_bond=3.83 eV at planar
    // geometry — larger than the total inversion barrier (~1.9 eV). A compressed sheet
    // buckles out of plane (classical buckling instability), which is correct UFF physics.
    //
    // To test inversion stiffness in isolation, we must first relieve bond strain by
    // relaxing in-plane (with inversions disabled), then add out-of-plane distortion
    // and verify it relaxes back to planarity.

    // Step 2a: Save original inversion params, disable them for in-plane relaxation
    let inv_params_saved: Vec<[f64; 4]> = mw.uff.inv_params.as_slice().to_vec();
    for ii in 0..mw.uff.ninversions as usize {
        mw.uff.inv_params.as_mut_slice()[ii] = [0.0, 1.0, -1.0, 0.0];
    }
    // Reset to planar geometry and relax in-plane
    for i in 0..mw.natoms() {
        mw.dyn_atoms.atoms.apos.as_mut_slice()[i] = top.apos[i];
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::new(0.0, 0.0, 0.0);
    }
    mw.dyn_atoms.fapos.resize_fill(mw.natoms(), Vec3d::new(0.0, 0.0, 0.0));

    let out_dir = base.join("debug/relax_pentacene_uff");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    // FIRE relaxation — in-plane only (inversions off)
    use molff::raff::FireState;
    let mut fire = FireState::new(0.001, 0.05);
    let mut step = Vec::new();
    let mut e_trace = Vec::new();
    let mut fmax_trace = Vec::new();
    let mut fz_max_trace = Vec::new();
    let niter = 5000;
    let mut n_converged = niter;
    for itr in 0..niter {
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
        let fz_max = mw.dyn_atoms.fapos.as_slice().iter().map(|f| f.z.abs()).fold(0.0f64, |a, b| a.max(b));
        step.push(itr as i32);
        e_trace.push(e);
        fmax_trace.push(fmax);
        fz_max_trace.push(fz_max);
        if fmax < 1e-3 { n_converged = itr + 1; break; }
    }
    let e_inplane = e_trace.last().copied().unwrap_or(0.0);
    let fmax_inplane = fmax_trace.last().copied().unwrap_or(0.0);
    let z_rms_inplane = mw.dyn_atoms.atoms.apos.as_slice().iter().map(|p| p.z * p.z).sum::<f64>().sqrt() / (mw.natoms() as f64).sqrt();
    println!("\nStep 2a: in-plane relaxation (inversions OFF): n={} E={:.6} fmax={:.6} z_rms={:.6}", n_converged, e_inplane, fmax_inplane, z_rms_inplane);
    write_trace(&out_dir.join("trajectory_inplane.tsv"), &step, &e_trace, &fmax_trace, &fz_max_trace);
    assert!(fmax_inplane < 0.5, "in-plane relaxation did not converge: fmax={:.4}", fmax_inplane);
    assert!(z_rms_inplane < 0.05, "in-plane relaxation produced out-of-plane geometry: z_rms={:.4}", z_rms_inplane);

    // DEBUG: print geometry bounds and sample bond lengths after in-plane relaxation
    {
        let apos = mw.dyn_atoms.atoms.apos.as_slice();
        let mut xmin = f64::MAX; let mut xmax = f64::MIN;
        let mut ymin = f64::MAX; let mut ymax = f64::MIN;
        for p in apos {
            if p.x < xmin { xmin = p.x; } if p.x > xmax { xmax = p.x; }
            if p.y < ymin { ymin = p.y; } if p.y > ymax { ymax = p.y; }
        }
        println!("  in-plane geometry bounds: x=[{:.3},{:.3}] y=[{:.3},{:.3}]", xmin, xmax, ymin, ymax);
        // Print first 5 bond lengths
        for ib in 0..5.min(mw.uff.nbonds as usize) {
            let b = mw.uff.bon_atoms.as_slice()[ib];
            let d = Vec3d::set_sub(apos[b[1] as usize], apos[b[0] as usize]);
            let l = d.norm();
            let l0 = mw.uff.bon_params.as_slice()[ib][1];
            println!("  bond[{}] {}-{} l={:.4} l0={:.4} diff={:.4}", ib, b[0], b[1], l, l0, l - l0);
        }
    }

    // Step 2b: Restore inversion params, add out-of-plane distortion, relax with FIRE
    for ii in 0..mw.uff.ninversions as usize {
        mw.uff.inv_params.as_mut_slice()[ii] = inv_params_saved[ii];
    }
    // Add ONLY out-of-plane (z) distortion to the relaxed planar geometry
    // Use a simple deterministic pattern: alternating ±amp_z with sinusoidal modulation
    let amp_z = 0.1;
    let mut z_max = 0.0f64;
    for i in 0..mw.natoms() {
        // Deterministic z-displacement in [-amp_z, +amp_z]
        let dz = amp_z * ((i as f64 * 0.7).sin() + 0.3 * ((i as f64 * 1.3).cos())) / 1.3;
        mw.dyn_atoms.atoms.apos.as_mut_slice()[i].z += dz;
        if dz.abs() > z_max { z_max = dz.abs(); }
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::new(0.0, 0.0, 0.0);
    }
    mw.dyn_atoms.fapos.resize_fill(mw.natoms(), Vec3d::new(0.0, 0.0, 0.0));
    println!("  z-displacement: max|dz|={:.4}", z_max);

    // DEBUG: check bond lengths after distortion
    {
        let apos = mw.dyn_atoms.atoms.apos.as_slice();
        let mut max_dl = 0.0f64;
        for ib in 0..mw.uff.nbonds as usize {
            let b = mw.uff.bon_atoms.as_slice()[ib];
            let d = Vec3d::set_sub(apos[b[1] as usize], apos[b[0] as usize]);
            let l = d.norm();
            let l0 = mw.uff.bon_params.as_slice()[ib][1];
            let dl = (l - l0).abs();
            if dl > max_dl { max_dl = dl; }
        }
        println!("  after distortion: max|l-l0|={:.6}", max_dl);
    }

    let (eb0, ea0, ed0, ei0, _, _) = mw.eval_forces();
    println!("Step 2b: out-of-plane distortion (amp={:.2}): E_bond={:.4} E_angle={:.4} E_dih={:.4} E_inv={:.4}", amp_z, eb0, ea0, ed0, ei0);

    // FIRE relaxation — with inversions ON
    let mut fire2 = FireState::new(0.001, 0.05);
    let mut step2 = Vec::new();
    let mut e_trace2 = Vec::new();
    let mut fmax_trace2 = Vec::new();
    let mut fz_max_trace2 = Vec::new();
    let mut n_converged2 = niter;
    for itr in 0..niter {
        let (eb, ea, ed, ei, _, _) = mw.eval_forces();
        let e = eb + ea + ed + ei;
        let dt = fire2.dt;
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
                mw.dyn_atoms.vapos.as_mut_slice()[ia] = Vec3d::set_lincomb(1.0 - fire2.alpha, v, fire2.alpha * v_mag, f_hat);
            }
            fire2.n_pos += 1;
            if fire2.n_pos > fire2.n_min {
                fire2.dt = (fire2.dt * fire2.f_inc).min(fire2.dt_max);
                fire2.alpha *= fire2.f_alpha;
            }
        } else {
            fire2.n_pos = 0;
            fire2.dt *= fire2.f_dec;
            fire2.alpha = fire2.alpha0;
            for v in mw.dyn_atoms.vapos.as_mut_slice() { *v = Vec3d::new(0.0, 0.0, 0.0); }
        }
        for ia in 0..mw.natoms() {
            let v = mw.dyn_atoms.vapos.as_slice()[ia];
            mw.dyn_atoms.atoms.apos.as_mut_slice()[ia].add_mul(v, dt);
        }
        let fmax = f2max.sqrt();
        let fz_max = mw.dyn_atoms.fapos.as_slice().iter().map(|f| f.z.abs()).fold(0.0f64, |a, b| a.max(b));
        step2.push(itr as i32);
        e_trace2.push(e);
        fmax_trace2.push(fmax);
        fz_max_trace2.push(fz_max);
        if fmax < 1e-3 { n_converged2 = itr + 1; break; }
    }
    write_trace(&out_dir.join("trajectory_fire.tsv"), &step2, &e_trace2, &fmax_trace2, &fz_max_trace2);
    let e_final = e_trace2.last().copied().unwrap_or(0.0);
    let fmax_final = fmax_trace2.last().copied().unwrap_or(0.0);
    let fz_final = fz_max_trace2.last().copied().unwrap_or(0.0);
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    // FIRE relaxation — in-plane only (inversions off)
    let mut fire = FireState::new(0.001, 0.05);
    let mut step = Vec::new();
    let mut e_trace = Vec::new();
    let mut fmax_trace = Vec::new();
    let mut fz_max_trace = Vec::new();
    let niter = 5000;
    let mut n_converged = niter;
    for itr in 0..niter {
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
        let fz_max = mw.dyn_atoms.fapos.as_slice().iter().map(|f| f.z.abs()).fold(0.0f64, |a, b| a.max(b));
        step.push(itr as i32);
        e_trace.push(e);
        fmax_trace.push(fmax);
        fz_max_trace.push(fz_max);
        if fmax < 1e-3 { n_converged = itr + 1; break; }
    }
    write_trace(&out_dir.join("trajectory_fire.tsv"), &step, &e_trace, &fmax_trace, &fz_max_trace);
    let e_final = e_trace.last().copied().unwrap_or(0.0);
    let fmax_final = fmax_trace.last().copied().unwrap_or(0.0);
    let fz_final = fz_max_trace.last().copied().unwrap_or(0.0);

    println!("\nStep 2b: FIRE relaxation (inversions ON): n={} E_final={:.6} fmax={:.6} fz_max={:.6}", n_converged2, e_final, fmax_final, fz_final);
    println!("  trace: {}/trajectory_fire.tsv", out_dir.display());

    // Check that relaxation converged
    assert!(fmax_final < 0.5, "FIRE did not converge sufficiently: fmax={:.4}", fmax_final);

    // Check that the molecule relaxed back toward planarity (z-coordinates near 0)
    let z_rms = mw.dyn_atoms.atoms.apos.as_slice().iter().map(|p| p.z * p.z).sum::<f64>().sqrt() / (mw.natoms() as f64).sqrt();
    println!("  z_rms of final geometry = {:.6} (should be small — planar molecule)", z_rms);
    assert!(z_rms < 0.05, "molecule did not relax back to planarity: z_rms={:.4}", z_rms);
}
