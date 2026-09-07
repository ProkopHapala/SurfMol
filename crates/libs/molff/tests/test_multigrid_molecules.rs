//! Multigrid solver on real molecules — pentacene, n-hexadecane, DiTriptyceno-helicene.
//!
//! Corrected benchmark per notes/reports/2026-08-29_multigrid_consolidated_report.md:
//!   - Low-frequency distortion initialization (parabolic bend + small noise), NOT x0=0 + pure force.
//!     This isolates coarse-solver performance: the bend is the low-freq error the coarse solver
//!     should capture; the noise is the high-freq error the smoother handles.
//!   - Penalty mass for pinned atoms (×1000), matching the reference demo's boundary treatment.
//!   - Heavy-ball momentum (beta=0.5) in the Jacobi baseline for fair comparison.
//!   - Coarse-projected residual |Pᵀ·r| measured separately from total residual |r|.
//!     The coarse residual isolates the coarse solver's job (low-freq modes) from the smoother's
//!     (high-freq modes like H-atom wiggles). See v2 analysis §0 Correction 2.
//!   - Static transverse-load RHS with a smooth bent/stretched initial guess plus local noise.
//!     This remains a linear diagnostic; production benefit must be measured in the nonlinear outer relaxation.

use molff::multigrid::*;
use molff::uff::Uff;
use moltopo::builder::Builder;
use moltopo::xyz;
use numtypes::Vec3d;
use std::path::PathBuf;

const REPO: &str = "../../..";

fn covalent_radii(elems: &[String]) -> Vec<f64> {
    elems.iter().map(|el| match el.as_str() {
        "H" => 0.31, "C" => 0.76, "N" => 0.71, "O" => 0.66,
        "F" => 0.57, "Si" => 1.11, "P" => 1.07, "S" => 1.05,
        "Cl" => 1.02, _ => 1.0,
    }).collect()
}

/// Load XYZ → topology → UFF (dummy params k=100) → TrussOp.
/// mass_dt2 = 1/dt² (reference uses dt=0.02 → mass_dt2=2500). Pinned atoms get ×1000 penalty.
/// Returns (TrussOp, positions, n_atoms).
fn load_molecule(xyz_path: &str, pinned: &[usize]) -> (TrussOp, Vec<Vec3d>) {
    let path = PathBuf::from(xyz_path);
    let sys = xyz::read_xyz(&path).unwrap_or_else(|e| panic!("read_xyz failed for {xyz_path}: {e}"));
    let radii = covalent_radii(&sys.elems);
    let top = Builder::from_positions_and_radii(&sys.apos, &sys.elems, &radii, 0.4).bake();
    let mut uff = Uff::from_topology(&top);
    uff.set_dummy_params(&top.apos);  // k=100, l0=actual bond length
    let dt = 0.02;
    let mut mass_dt2 = vec![1.0 / dt / dt; uff.natoms as usize];  // 2500 for free atoms
    for &i in pinned { mass_dt2[i] *= 1000.0; }  // penalty for pinned atoms
    let op = TrussOp::from_uff_bonds(&uff, &top.apos, &mass_dt2);
    (op, top.apos)
}

/// Build a low-frequency distortion: parabolic bend in y + uniform stretch along x + small noise.
/// The bend is the low-freq error the coarse solver should capture.
/// The noise is the high-freq error the smoother handles.
/// `bend_amp` = max y-displacement at the midpoint (Å). `stretch` = fractional x-stretch.
/// `noise_amp` = per-atom random displacement (Å).
fn make_low_freq_distortion(apos: &[Vec3d], bend_amp: f64, stretch: f64, noise_amp: f64, seed: u64) -> Vec<f64> {
    let n = apos.len();
    let mut x0 = vec![0.0f64; n * 3];
    // Find x-extent for parabolic bend parameterization
    let xmin = apos.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let xmax = apos.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let span = (xmax - xmin).max(1e-12);
    let mut rng = seed;
    for i in 0..n {
        let p = apos[i];
        // Parabolic bend: y += bend_amp * (1 - (2*(x-xmin)/span - 1)^2) — max at center, 0 at ends
        let t = (p.x - xmin) / span;  // 0 at left, 1 at right
        let bend = bend_amp * (1.0 - (2.0*t - 1.0).powi(2));
        // Uniform stretch: x += stretch * (x - x_center)
        let xcenter = (xmin + xmax) * 0.5;
        let stretch_dx = stretch * (p.x - xcenter);
        // Small random noise
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r1 = ((rng >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r2 = ((rng >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r3 = ((rng >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        x0[i*3]     = stretch_dx + noise_amp * r1;
        x0[i*3 + 1] = bend      + noise_amp * r2;
        x0[i*3 + 2] =              noise_amp * r3;
    }
    x0
}

/// Compute total residual |b - A·x| / |b| over free DOFs, and coarse-projected |Pᵀ·r| / |Pᵀ·b|.
fn compute_residuals(op: &TrussOp, p: &[f64], n_coarse: usize, b: &[f64], x: &[f64],
                     free_mask: &[bool], b_norm: f64, b_c_norm: f64) -> (f64, f64) {
    let n = op.natoms * 3;
    let ax = op.matvec(x);
    let mut r = vec![0.0f64; n];
    let mut rn = 0.0;
    for i in 0..op.natoms {
        if free_mask[i] {
            for d in 0..3 { let dd = b[i*3+d] - ax[i*3+d]; r[i*3+d] = dd; rn += dd*dd; }
        }
    }
    let total = rn.sqrt() / b_norm;
    let mut rcn = 0.0;
    for j in 0..n_coarse {
        let mut s = 0.0;
        for i in 0..n { s += p[i * n_coarse + j] * r[i]; }
        rcn += s*s;
    }
    (total, rcn.sqrt() / b_c_norm)
}

/// Corrected benchmark: low-freq distortion init, penalty mass, heavy-ball baseline,
/// coarse-projected residual measurement.
fn benchmark(op: &TrussOp, apos: &[Vec3d], pinned: &[usize], loaded: &[usize],
             label: &str, manual_pivots: &[usize], auto_n_pivots: usize) {
    let n = op.natoms * 3;
    let bonds: Vec<[i32;2]> = op.ei.iter().zip(op.ej.iter()).map(|(&i,&j)| [i,j]).collect();
    println!("\n================================================================================");
    println!("[{}] {} atoms, {} bonds, {} DOF, {} pinned", label, op.natoms, op.ei.len(), n, pinned.len());

    let mut free_mask = vec![true; op.natoms];
    for &i in pinned { free_mask[i] = false; }

    // Low-frequency initial guess: parabolic bend (1Å) + 5% stretch + 0.01Å noise.
    // The solution x* = A⁻¹·b is the TRUE bending shape (from the transverse force).
    // The error x0 - x* = (parabolic - true_bend) + noise — a low-freq smooth error
    // (the difference between approximate and exact bend) + high-freq noise.
    // This properly exercises both the coarse solver (low-freq) and smoother (high-freq).
    let mut x0 = make_low_freq_distortion(apos, 1.0, 0.05, 0.01, 42);
    // Zero out pinned atoms — their displacement must be 0 (Dirichlet BC).
    // Otherwise the penalty mass (2.5M) creates a huge residual that destabilizes Jacobi.
    for &i in pinned { for d in 0..3 { x0[i*3 + d] = 0.0; } }

    // RHS: transverse (y) force on loaded atoms — the external load that causes bending.
    // The solution x* is the true static bending displacement.
    let mut b = vec![0.0; n];
    for &i in loaded { b[i*3 + 1] = -10.0; }  // -10 in y on each loaded atom
    let b_norm: f64 = b.iter().map(|x| x*x).sum::<f64>().sqrt().max(1e-30);

    // --- Direct solve (reference) ---
    let a_dense = op.assemble_dense();
    let x_direct = dense_solve(&a_dense, &b, n);
    let (direct_total, _) = compute_residuals(op, &[], 0, &b, &x_direct, &free_mask, b_norm, 1.0);
    println!("[{}] Direct solve: residual = {:.3e}", label, direct_total);

    // --- Jacobi baselines: plain (β=0) and heavy-ball (β=0.5) ---
    let d = op.diagonal_blocks();
    let dinv = invert_3x3_blocks(&d);
    for (beta, blabel) in [(0.0, "plain"), (0.5, "HB(β=0.5)")] {
        let mut x_jac = x0.clone();
        let mut vel_jac = vec![0.0f64; n];
        let mut jac_res = vec![];
        for _ in 0..5000 {
            let ax = op.matvec(&x_jac);
            let mut rn = 0.0;
            for i in 0..n { let r = b[i] - ax[i]; rn += r*r; }
            jac_res.push(rn.sqrt() / b_norm);
            jacobi_smooth_momentum(op, &dinv, &b, &mut x_jac, &free_mask, 0.8, beta, &mut vel_jac, 1);
        }
        let jac_final = jac_res[jac_res.len()-1];
        let jac_to_1e3 = jac_res.iter().position(|&r| r < 1e-3).unwrap_or(99999);
        let jac_to_1e6 = jac_res.iter().position(|&r| r < 1e-6).unwrap_or(99999);
        let status = if jac_final > jac_res[0] { "DIVERGED" } else if jac_to_1e3 < 99999 { "converged" } else { "stalled" };
        println!("[{}] Jacobi {} 5000 iters: final={:.3e}, to-1e-3={}, to-1e-6={} [{}]",
            label, blabel, jac_final, jac_to_1e3, jac_to_1e6, status);
    }

    // --- MG with manual pivots ---
    if !manual_pivots.is_empty() {
        run_mg(op, apos, &free_mask, manual_pivots, &b, &x0, label, "manual");
    }
    // --- MG with automatic maximin pivots ---
    if auto_n_pivots > 0 {
        let pivots = select_pivots_maximin(&bonds, op.natoms, auto_n_pivots, &free_mask);
        println!("[{}] Auto pivots ({}): {:?}", label, auto_n_pivots, pivots);
        run_mg(op, apos, &free_mask, &pivots, &b, &x0, label, "auto");
    }
}

/// Run MG with given pivots, print total + coarse-projected residuals.
fn run_mg(op: &TrussOp, apos: &[Vec3d], free_mask: &[bool], pivots: &[usize], b: &[f64], x0: &[f64], label: &str, mode: &str) {
    let n = op.natoms * 3;
    let p = build_pivot_prolongation(apos, pivots, 2.0, free_mask);
    let n_coarse = pivots.len() * 3;
    let (x_stage, stage_res, coarse_energy, _) = solve_coarse_first(op, &p, n_coarse, b, x0, free_mask, 0.8, 0.0, 500, 1e-6);
    let stage_fine = stage_res.len() - 1;
    let mut stage_err = 0.0f64;
    let x_direct = dense_solve(&op.assemble_dense(), b, n);
    for i in 0..n { stage_err = stage_err.max((x_stage[i] - x_direct[i]).abs()); }
    let stage_fine_a = 2 + 2*stage_fine;
    println!("[{}] MG {} coarse-first: 1 coarse correction + {} fine steps, fine-A solve/setup={}/{}, residual={:.3e}, coarse-energy={:.3e}, max|err|={:.3e}", label, mode, stage_fine, stage_fine_a, n_coarse, stage_res[stage_res.len()-1], coarse_energy, stage_err);

    // MG with beta=0.0 in smoother (reference default) — test beta=0.5 too
    for (beta, beta_label) in [(0.0, "β=0"), (0.5, "β=0.5")] {
        let (x_mg, mg_total, mg_coarse, _) = solve_multigrid(op, &p, n_coarse, &b, &x0, free_mask,
                                                             0.8, beta, 3, 3, 500, 1e-10);
        let mg_final = mg_total[mg_total.len()-1];
        let mg_to_1e3 = mg_total.iter().position(|&r| r < 1e-3).unwrap_or(99999);
        let mg_to_1e6 = mg_total.iter().position(|&r| r < 1e-6).unwrap_or(99999);
        let mg_smooth_to_1e3 = mg_to_1e3 * 6;
        // Coarse residual: how well is the low-freq error being captured?
        let c_final = mg_coarse[mg_coarse.len()-1];
        let c_to_1e3 = mg_coarse.iter().position(|&r| r < 1e-3).unwrap_or(99999);
        // Compare to direct
        let a_dense = op.assemble_dense();
        let x_direct = dense_solve(&a_dense, &b, n);
        let mut max_err = 0.0;
        for i in 0..n { let e = (x_mg[i] - x_direct[i]).abs(); if e > max_err { max_err = e; } }
        let compress = 100.0 * (1.0 - n_coarse as f64 / n as f64);
        let mg_fine_a_to_1e6 = 1 + mg_to_1e6*8;
        println!("[{}] MG {} {}pivots ({} DOF, {:.0}% comp): total={:.3e}, cycles to 1e-3/1e-6={}/{}, smooth to 1e-3={}, fine-A to 1e-6/setup={}/{}",
            label, mode, beta_label, n_coarse, compress, mg_final, mg_to_1e3, mg_to_1e6, mg_smooth_to_1e3, mg_fine_a_to_1e6, n_coarse);
        println!("[{}]   max|mg-direct|={:.3e}", label, max_err);
        println!("[{}]   coarse-proj residual: final={:.3e}, to-1e-3={} — low-freq capture",
            label, c_final, c_to_1e3);
        if beta == 0.0 {
            let preview: Vec<String> = mg_total.iter().take(10).map(|r| format!("{:.2e}", r)).collect();
            println!("[{}]   total residual curve (first 10): {:?}", label, preview);
            let cpreview: Vec<String> = mg_coarse.iter().take(10).map(|r| format!("{:.2e}", r)).collect();
            println!("[{}]   coarse residual curve (first 10): {:?}", label, cpreview);
        }
    }
}

fn find_x_extremes(apos: &[Vec3d]) -> (usize, usize, f64, f64) {
    let mut xmin_i = 0; let mut xmax_i = 0;
    let mut xmin = f64::INFINITY; let mut xmax = f64::NEG_INFINITY;
    for (i, &p) in apos.iter().enumerate() {
        if p.x < xmin { xmin = p.x; xmin_i = i; }
        if p.x > xmax { xmax = p.x; xmax_i = i; }
    }
    (xmin_i, xmax_i, xmin, xmax)
}

#[test]
fn test_pentacene() {
    let (xmin_i, xmax_i, xmin, xmax) = find_x_extremes(&read_positions("pentacene.xyz"));
    let pinned: Vec<usize> = read_positions("pentacene.xyz").iter().enumerate()
        .filter(|(_, &p)| p.x < xmin + 1.5).map(|(i, _)| i).collect();
    let loaded: Vec<usize> = read_positions("pentacene.xyz").iter().enumerate()
        .filter(|(_, &p)| p.x > xmax - 1.5).map(|(i, _)| i).collect();
    let (op, apos) = load_molecule(&format!("{REPO}/data/xyz/pentacene.xyz"), &pinned);
    let xmid_i = { let xavg: f64 = apos.iter().map(|p| p.x).sum::<f64>() / apos.len() as f64;
        apos.iter().enumerate().min_by(|(_, a), (_, b)| (a.x - xavg).abs().partial_cmp(&(b.x - xavg).abs()).unwrap()).unwrap().0 };
    let span = xmax - xmin;
    let xavg: f64 = apos.iter().map(|p| p.x).sum::<f64>() / apos.len() as f64;
    let mut q1_i = 0; let mut q1_d = f64::INFINITY;
    let mut q3_i = 0; let mut q3_d = f64::INFINITY;
    for (i, &p) in apos.iter().enumerate() {
        let d1 = (p.x - (xavg - span*0.25)).abs(); if d1 < q1_d { q1_d = d1; q1_i = i; }
        let d3 = (p.x - (xavg + span*0.25)).abs(); if d3 < q3_d { q3_d = d3; q3_i = i; }
    }
    let manual = vec![xmin_i, q1_i, xmid_i, q3_i, xmax_i];
    println!("[pentacene] pinned={:?}, loaded={:?}, manual pivots={:?}", pinned, loaded, manual);
    benchmark(&op, &apos, &pinned, &loaded, "pentacene", &manual, 6);
}

#[test]
fn test_nhexadecane() {
    let pinned = vec![0usize, 1];
    let loaded = vec![14usize, 15];
    let (op, apos) = load_molecule(&format!("{REPO}/data/xyz/nHexadecan.xyz"), &pinned);
    let manual = vec![0usize, 3, 6, 9, 12, 15];
    println!("[hexadecane] pinned={:?}, loaded={:?}, manual pivots={:?}", pinned, loaded, manual);
    benchmark(&op, &apos, &pinned, &loaded, "hexadecane", &manual, 10);
}

#[test]
fn test_ditriptyceno() {
    let apos_pre = read_positions("DiTriptyceno_helicene_3a.xyz");
    let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
    for &p in &apos_pre { cx += p.x; cy += p.y; cz += p.z; }
    let n = apos_pre.len() as f64;
    let centroid = Vec3d::new(cx/n, cy/n, cz/n);
    let mut idx_dist: Vec<(usize, f64)> = apos_pre.iter().enumerate()
        .map(|(i, &p)| (i, (p - centroid).norm())).collect();
    idx_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let pinned: Vec<usize> = idx_dist.iter().take(3).map(|(i, _)| *i).collect();
    idx_dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let loaded: Vec<usize> = idx_dist.iter().take(6).map(|(i, _)| *i).collect();
    let (op, apos) = load_molecule(&format!("{REPO}/data/xyz/DiTriptyceno_helicene_3a.xyz"), &pinned);
    let manual: Vec<usize> = loaded.clone();
    println!("[ditriptyceno] pinned (core)={:?}, loaded (tips)={:?}, manual pivots={:?}", pinned, loaded, manual);
    benchmark(&op, &apos, &pinned, &loaded, "ditriptyceno", &manual, 12);
}

/// Helper: read positions only (for pin selection before load_molecule).
fn read_positions(name: &str) -> Vec<Vec3d> {
    let path = PathBuf::from(format!("{REPO}/data/xyz/{name}"));
    let sys = xyz::read_xyz(&path).unwrap_or_else(|e| panic!("read_xyz failed for {name}: {e}"));
    sys.apos
}

// ============================================================================
// Real UFF test: full Hessian (bonds + angles + dihedrals + inversions)
// ============================================================================
//
// The bond-only TrussOp has ZERO out-of-plane bending stiffness at a planar
// equilibrium — axial springs only resist along the bond direction. Real UFF
// adds inversion (improper torsion) and dihedral terms that provide the
// out-of-plane bending stiffness critical for aromatic molecules like pentacene.
//
// This test builds the full UFF Hessian via finite differences (UffHessianOp)
// and compares multigrid convergence against:
//   1. Bond-only TrussOp (dummy params) — no bending stiffness
//   2. Full UFF UffHessianOp — physical bending stiffness
//
// The RHS is a transverse (z) force causing out-of-plane bending. With real UFF,
// the solution is a smooth bend; with bond-only, the mass term is the only
// resistance (unphysical).

use moltopo::topology::{Topology, build_bonds_by_cutoff, build_angles_from_bonds, build_dihedrals_from_bonds, build_inversions_from_bonds};
use moltopo::params::Params;
use moltopo::assign_uff;
use surfmol::mol_world::{MolWorld, BondedFFMode};

/// Load pentacene with REAL UFF parameters via MolWorld.
/// Returns (MolWorld, positions, bonds, pinned, loaded).
/// In-plane relaxation is performed first to relieve DFT-vs-UFF bond strain,
/// then inversions are re-enabled for the bending test.
fn load_pentacene_real_uff() -> (MolWorld, Vec<Vec3d>, Vec<[i32;2]>, Vec<usize>, Vec<usize>) {
    let base = PathBuf::from(REPO);
    let xyz = xyz::read_xyz(&base.join("data/xyz/pentacene.xyz")).expect("read pentacene.xyz");
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
    println!("[pentacene_uff] topology: bonds={} angles={} dihedrals={} inversions={}",
        top.bonds.len(), top.angles.len(), top.dihedrals.len(), top.inversions.len());

    // Build MolWorld and assign real UFF types + params
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

    // Verify real params: angles, dihedrals, inversions should be nonzero
    let n_ang = mw.uff.ang_params.as_slice().iter().filter(|p| p[0].abs() > 1e-10).count();
    let n_dih = mw.uff.dih_params.as_slice().iter().filter(|p| p[0].abs() > 1e-10).count();
    let n_inv = mw.uff.inv_params.as_slice().iter().filter(|p| p[0].abs() > 1e-10).count();
    println!("[pentacene_uff] nonzero params: angles={}/{} dihedrals={}/{} inversions={}/{}",
        n_ang, mw.uff.nangles, n_dih, mw.uff.ndihedrals, n_inv, mw.uff.ninversions);
    assert!(n_inv > 0, "no inversion params — aromatic bending stiffness missing");

    // Step 1: in-plane relaxation (inversions OFF) to relieve bond strain.
    // UFF bond lengths (1.458 Å for aromatic C-C) differ from DFT geometry (~1.4 Å).
    // This 4% compression causes ~3.8 eV bond strain that would buckle the molecule.
    let inv_saved: Vec<[f64; 4]> = mw.uff.inv_params.as_slice().to_vec();
    for ii in 0..mw.uff.ninversions as usize {
        mw.uff.inv_params.as_mut_slice()[ii] = [0.0, 1.0, -1.0, 0.0];
    }
    for i in 0..mw.natoms() {
        mw.dyn_atoms.atoms.apos.as_mut_slice()[i] = top.apos[i];
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::new(0.0, 0.0, 0.0);
    }
    mw.dyn_atoms.fapos.resize_fill(mw.natoms(), Vec3d::new(0.0, 0.0, 0.0));

    // Simple damped MD for in-plane relaxation (no FIRE needed — just relieve strain)
    let dt = 0.02;
    let cdamp = 0.9;
    for itr in 0..2000 {
        let (_, _, _, _, _, _) = mw.eval_forces();
        let mut f2max = 0.0f64;
        for ia in 0..mw.natoms() {
            let f = mw.dyn_atoms.fapos.as_slice()[ia];
            let f2 = f.norm2();
            if f2 > f2max { f2max = f2; }
            let v = mw.dyn_atoms.vapos.as_slice()[ia];
            let mut v_new = v;
            v_new.add_mul(f, dt);
            v_new = v_new * cdamp;
            mw.dyn_atoms.vapos.as_mut_slice()[ia] = v_new;
            mw.dyn_atoms.atoms.apos.as_mut_slice()[ia].add_mul(v_new, dt);
        }
        if f2max.sqrt() < 1e-3 { println!("[pentacene_uff] in-plane relaxed at step {itr}"); break; }
    }

    // Restore inversions for the bending test
    for ii in 0..mw.uff.ninversions as usize {
        mw.uff.inv_params.as_mut_slice()[ii] = inv_saved[ii];
    }

    let relaxed_apos: Vec<Vec3d> = mw.dyn_atoms.atoms.apos.as_slice().to_vec();
    let bonds_i32: Vec<[i32; 2]> = top.bonds.iter().map(|b| [b[0], b[1]]).collect();

    // Pin left end, load right end (transverse z-force → out-of-plane bending)
    let xmin = relaxed_apos.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let xmax = relaxed_apos.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let pinned: Vec<usize> = relaxed_apos.iter().enumerate()
        .filter(|(_, &p)| p.x < xmin + 1.5).map(|(i, _)| i).collect();
    let loaded: Vec<usize> = relaxed_apos.iter().enumerate()
        .filter(|(_, &p)| p.x > xmax - 1.5).map(|(i, _)| i).collect();

    println!("[pentacene_uff] pinned={:?} (left end), loaded={:?} (right end)", pinned, loaded);
    (mw, relaxed_apos, bonds_i32, pinned, loaded)
}

/// Run a benchmark on any LinearOp: Jacobi baseline + MG with manual/auto pivots.
/// `bonds` are needed for pivot selection (not from the operator).
fn benchmark_op(op: &impl LinearOp, apos: &[Vec3d], bonds: &[[i32; 2]], pinned: &[usize],
                loaded: &[usize], b: &[f64], x0: &[f64], label: &str,
                manual_pivots: &[usize], auto_n_pivots: usize) {
    let natoms = op.natoms();
    let n = natoms * 3;
    let mut free_mask = vec![true; natoms];
    for &i in pinned { free_mask[i] = false; }
    println!("\n================================================================================");
    println!("[{}] {} atoms, {} bonds, {} DOF, {} pinned", label, natoms, bonds.len(), n, pinned.len());

    let b_norm: f64 = b.iter().map(|x| x*x).sum::<f64>().sqrt().max(1e-30);

    // Direct solve (reference)
    let a_dense = op.assemble_dense();
    let x_direct = dense_solve(&a_dense, b, n);
    let ax_direct = op.matvec(&x_direct);
    let mut direct_res = 0.0;
    for i in 0..natoms {
        if free_mask[i] { for d in 0..3 { let r = b[i*3+d] - ax_direct[i*3+d]; direct_res += r*r; } }
    }
    println!("[{}] Direct solve: residual = {:.3e}", label, direct_res.sqrt() / b_norm);

    // Jacobi baseline (plain + heavy-ball)
    let d = op.diagonal_blocks();
    let dinv = invert_3x3_blocks(&d);
    for (beta, blabel) in [(0.0, "plain"), (0.5, "HB(β=0.5)")] {
        let mut x_jac = x0.to_vec();
        let mut vel_jac = vec![0.0f64; n];
        let mut jac_res = vec![];
        for _ in 0..5000 {
            let ax = op.matvec(&x_jac);
            let mut rn = 0.0;
            for i in 0..n { let r = b[i] - ax[i]; rn += r*r; }
            jac_res.push(rn.sqrt() / b_norm);
            jacobi_smooth_momentum(op, &dinv, b, &mut x_jac, &free_mask, 0.8, beta, &mut vel_jac, 1);
        }
        let jac_final = *jac_res.last().unwrap();
        let jac_to_1e3 = jac_res.iter().position(|&r| r < 1e-3).unwrap_or(99999);
        let jac_to_1e6 = jac_res.iter().position(|&r| r < 1e-6).unwrap_or(99999);
        let status = if jac_final > jac_res[0] { "DIVERGED" } else if jac_to_1e3 < 99999 { "converged" } else { "stalled" };
        println!("[{}] Jacobi {} 5000 iters: final={:.3e}, to-1e-3={}, to-1e-6={} [{}]",
            label, blabel, jac_final, jac_to_1e3, jac_to_1e6, status);
    }

    // MG with manual pivots
    if !manual_pivots.is_empty() {
        run_mg_op(op, apos, &free_mask, manual_pivots, b, x0, label, "manual");
    }
    // MG with automatic maximin pivots
    if auto_n_pivots > 0 {
        let pivots = select_pivots_maximin(bonds, natoms, auto_n_pivots, &free_mask);
        println!("[{}] Auto pivots ({}): {:?}", label, auto_n_pivots, pivots);
        run_mg_op(op, apos, &free_mask, &pivots, b, x0, label, "auto");
    }
}

fn run_mg_op(op: &impl LinearOp, apos: &[Vec3d], free_mask: &[bool], pivots: &[usize],
             b: &[f64], x0: &[f64], label: &str, mode: &str) {
    let n = op.natoms() * 3;
    let p = build_pivot_prolongation(apos, pivots, 2.0, free_mask);
    let n_coarse = pivots.len() * 3;

    // Coarse-first: 1 coarse correction + fine smoothing
    let (x_stage, stage_res, coarse_energy, _) = solve_coarse_first(op, &p, n_coarse, b, x0, free_mask, 0.8, 0.0, 500, 1e-6);
    let stage_fine = stage_res.len() - 1;
    let x_direct = dense_solve(&op.assemble_dense(), b, n);
    let mut stage_err = 0.0f64;
    for i in 0..n { stage_err = stage_err.max((x_stage[i] - x_direct[i]).abs()); }
    println!("[{}] MG {} coarse-first: 1 coarse + {} fine steps, residual={:.3e}, coarse-E={:.3e}, max|err|={:.3e}",
        label, mode, stage_fine, *stage_res.last().unwrap(), coarse_energy, stage_err);

    // Full V-cycle with β=0 and β=0.5
    for (beta, beta_label) in [(0.0, "β=0"), (0.5, "β=0.5")] {
        let (x_mg, mg_total, mg_coarse, _) = solve_multigrid(op, &p, n_coarse, b, x0, free_mask,
                                                             0.8, beta, 3, 3, 500, 1e-10);
        let mg_final = *mg_total.last().unwrap();
        let mg_to_1e3 = mg_total.iter().position(|&r| r < 1e-3).unwrap_or(99999);
        let mg_to_1e6 = mg_total.iter().position(|&r| r < 1e-6).unwrap_or(99999);
        let c_final = *mg_coarse.last().unwrap();
        let c_to_1e3 = mg_coarse.iter().position(|&r| r < 1e-3).unwrap_or(99999);
        let mut max_err = 0.0;
        for i in 0..n { let e = (x_mg[i] - x_direct[i]).abs(); if e > max_err { max_err = e; } }
        let compress = 100.0 * (1.0 - n_coarse as f64 / n as f64);
        println!("[{}] MG {} {}pivots ({} DOF, {:.0}% comp): total={:.3e}, cycles to 1e-3/1e-6={}/{}, coarse={:.3e}/to-1e-3={}, max|err|={:.3e}",
            label, mode, beta_label, n_coarse, compress, mg_final, mg_to_1e3, mg_to_1e6, c_final, c_to_1e3, max_err);
    }
}

/// Out-of-plane bending distortion: parabolic z-bend along x + small noise.
fn make_bend_distortion_z(apos: &[Vec3d], bend_amp: f64, noise_amp: f64, seed: u64) -> Vec<f64> {
    let n = apos.len();
    let mut x0 = vec![0.0f64; n * 3];
    let xmin = apos.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let xmax = apos.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let span = (xmax - xmin).max(1e-12);
    let mut rng = seed;
    for i in 0..n {
        let p = apos[i];
        let t = (p.x - xmin) / span;
        let bend = bend_amp * (1.0 - (2.0*t - 1.0).powi(2)); // parabolic: max at center, 0 at ends
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r1 = ((rng >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r2 = ((rng >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r3 = ((rng >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        x0[i*3]     = noise_amp * r1;
        x0[i*3 + 1] = noise_amp * r2;
        x0[i*3 + 2] = bend      + noise_amp * r3;
    }
    x0
}

#[test]
fn test_pentacene_real_uff() {
    let (mut mw, apos, bonds, pinned, loaded) = load_pentacene_real_uff();
    let natoms = mw.natoms();
    let n = natoms * 3;

    // --- Build full UFF Hessian operator (bonds + angles + dihedrals + inversions) ---
    let dt = 0.02;
    let mut mass_dt2 = vec![1.0 / dt / dt; natoms]; // 2500 for free atoms
    for &i in &pinned { mass_dt2[i] *= 1000.0; }    // penalty for pinned atoms
    let eps_fd = 1e-4; // finite-difference step (Å)
    println!("[pentacene_uff] building full UFF Hessian (eps_fd={eps_fd}, {} force evals)...", 2*n);
    let op_uff = UffHessianOp::from_uff(
        &mut mw.uff, &apos, mw.dyn_atoms.neighs(), mw.dyn_atoms.neigh_bs(),
        &mass_dt2, eps_fd);
    println!("[pentacene_uff] Hessian built: {}×{} dense matrix", n, n);

    // --- Build bond-only Hessian for comparison (zero angles/dihedrals/inversions) ---
    // Save and zero out angle/dihedral/inversion params to isolate bond stiffness.
    let ang_saved: Vec<[f64; 5]> = mw.uff.ang_params.as_slice().to_vec();
    let dih_saved: Vec<[f64; 3]> = mw.uff.dih_params.as_slice().to_vec();
    let inv_saved2: Vec<[f64; 4]> = mw.uff.inv_params.as_slice().to_vec();
    for ia in 0..mw.uff.nangles as usize { mw.uff.ang_params.as_mut_slice()[ia] = [0.0, 1.0, -1.0, 0.0, 0.0]; }
    for id in 0..mw.uff.ndihedrals as usize { mw.uff.dih_params.as_mut_slice()[id] = [0.0, 1.0, 3.0]; }
    for ii in 0..mw.uff.ninversions as usize { mw.uff.inv_params.as_mut_slice()[ii] = [0.0, 1.0, -1.0, 0.0]; }
    println!("[pentacene_uff] building bond-only Hessian (angles/dih/inversions zeroed)...");
    let op_bond = UffHessianOp::from_uff(
        &mut mw.uff, &apos, mw.dyn_atoms.neighs(), mw.dyn_atoms.neigh_bs(),
        &mass_dt2, eps_fd);
    // Restore full params
    for ia in 0..mw.uff.nangles as usize { mw.uff.ang_params.as_mut_slice()[ia] = ang_saved[ia]; }
    for id in 0..mw.uff.ndihedrals as usize { mw.uff.dih_params.as_mut_slice()[id] = dih_saved[id]; }
    for ii in 0..mw.uff.ninversions as usize { mw.uff.inv_params.as_mut_slice()[ii] = inv_saved2[ii]; }

    // --- RHS: transverse (z) force on loaded atoms → out-of-plane bending ---
    let mut b = vec![0.0f64; n];
    for &i in &loaded { b[i*3 + 2] = -10.0; } // -10 in z on each loaded atom

    // --- Initial guess: parabolic z-bend + small noise (low-freq + high-freq error) ---
    let mut x0 = make_bend_distortion_z(&apos, 0.5, 0.01, 42);
    for &i in &pinned { for d in 0..3 { x0[i*3 + d] = 0.0; } }

    // --- Compare stiffness: isolate K·x by subtracting mass·x from A·x ---
    // A = K + diag(mass_dt2), so K·x = A·x - mass_dt2·x
    let ax_bond = op_bond.matvec(&x0);
    let ax_uff = op_uff.matvec(&x0);
    let mut kzz_bond = 0.0;
    let mut kzz_uff = 0.0;
    for i in 0..natoms {
        let mass_contrib = mass_dt2[i] * x0[i*3+2];
        kzz_bond += (ax_bond[i*3+2] - mass_contrib).abs();
        kzz_uff += (ax_uff[i*3+2] - mass_contrib).abs();
    }
    println!("\n--- stiffness comparison (K·x out-of-plane, mass subtracted) ---");
    println!("[pentacene_uff] bond-only: Σ|K·x|_z = {:.4} (bonds only, no bending)", kzz_bond);
    println!("[pentacene_uff] full UFF:  Σ|K·x|_z = {:.4} (bonds + angles + dih + inv)", kzz_uff);
    let kzz_bending = kzz_uff - kzz_bond;
    println!("[pentacene_uff] bending contribution (inv+dih): {:.4}", kzz_bending);
    assert!(kzz_bending > 1.0,
        "full UFF should have significant bending stiffness beyond bonds: bending={kzz_bending:.4} UFF={kzz_uff:.4} bond={kzz_bond:.4}");

    // --- Benchmark 1: bond-only Hessian (no bending stiffness) ---
    println!("\n--- benchmark 1: bond-only Hessian (angles/dih/inversions zeroed) ---");
    benchmark_op(&op_bond, &apos, &bonds, &pinned, &loaded, &b, &x0, "pentacene_bond", &[], 6);

    // --- Benchmark 2: full UFF Hessian (physical bending stiffness) ---
    println!("\n--- benchmark 2: full UFF Hessian (real params, bending from inv+dih) ---");
    // Manual pivots: 5 evenly spaced along x-axis
    let xmin = apos.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let xmax = apos.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let span = xmax - xmin;
    let xavg: f64 = apos.iter().map(|p| p.x).sum::<f64>() / natoms as f64;
    let mut q1_i = 0; let mut q1_d = f64::INFINITY;
    let mut q3_i = 0; let mut q3_d = f64::INFINITY;
    let mut xmid_i = 0; let mut xmid_d = f64::INFINITY;
    let mut xmin_i = 0; let mut xmax_i = 0;
    for (i, &p) in apos.iter().enumerate() {
        if p.x < apos[xmin_i].x { xmin_i = i; }
        if p.x > apos[xmax_i].x { xmax_i = i; }
        let dm = (p.x - xavg).abs(); if dm < xmid_d { xmid_d = dm; xmid_i = i; }
        let d1 = (p.x - (xavg - span*0.25)).abs(); if d1 < q1_d { q1_d = d1; q1_i = i; }
        let d3 = (p.x - (xavg + span*0.25)).abs(); if d3 < q3_d { q3_d = d3; q3_i = i; }
    }
    let manual = vec![xmin_i, q1_i, xmid_i, q3_i, xmax_i];
    benchmark_op(&op_uff, &apos, &bonds, &pinned, &loaded, &b, &x0, "pentacene_uff", &manual, 6);

    // --- Key result: verify the direct solve gives a physical bending shape ---
    let x_direct_uff = dense_solve(&op_uff.assemble_dense(), &b, n);
    let z_max_uff = (0..natoms).map(|i| x_direct_uff[i*3+2].abs()).fold(0.0f64, f64::max);
    let z_rms_uff = (0..natoms).map(|i| x_direct_uff[i*3+2].powi(2)).sum::<f64>().sqrt() / (natoms as f64).sqrt();
    println!("\n--- direct solve: out-of-plane bending shape ---");
    println!("[pentacene_uff] full UFF:  z_max={:.6} z_rms={:.6} (physical bend from inv+dih)", z_max_uff, z_rms_uff);

    let x_direct_bond = dense_solve(&op_bond.assemble_dense(), &b, n);
    let z_max_bond = (0..natoms).map(|i| x_direct_bond[i*3+2].abs()).fold(0.0f64, f64::max);
    println!("[pentacene_uff] bond-only: z_max={:.6} (no bending stiffness — only mass resists)", z_max_bond);
    assert!(z_max_uff > 1e-4, "full UFF direct solve gave no bending: z_max={z_max_uff:.2e}");

    println!("\n=== PASS: pentacene real UFF multigrid test ===");
    println!("  bending stiffness (inv+dih) = {:.4} (beyond bond-only {:.4})", kzz_bending, kzz_bond);
    println!("  full UFF bending z_max={:.6} vs bond-only z_max={:.6}", z_max_uff, z_max_bond);
}
