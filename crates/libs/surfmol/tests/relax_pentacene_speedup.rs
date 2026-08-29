//! Pentacene modal relaxation speedup benchmark — v2 (proper timestep scaling + Newton).
//!
//! Design spec: `notes/designs/2026-08-29_modal_relaxation_design_spec.md`
//!
//! Key fix vs v1: the speedup comes from TIMESTEP SCALING. Freezing hard modes allows
//! the modal Newton step to converge soft DOFs in 1-3 steps (exact for quadratic model).
//! Pentacene's bending is approximately quadratic (small angles), so Newton is nearly exact.
//! The fine phase (FIRE) then only relaxes hard DOFs (bond stretches, H atoms) — fast.
//!
//! Fitting cost is NOT counted — it's a one-time setup amortized over thousands of
//! molecules × millions of steps.
//!
//! Distortion: parabolic out-of-plane bend + axial twist + small white noise.
//! Pentacene is the ideal test case: aromatic rings are rigid, sp2 carbons have no
//! free rotation, so the low-energy subspace is approximately linear bending along
//! soft eigenmodes (no nonlinear torsional DOFs like aliphatic chains).

use std::path::Path;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use numtypes::Vec3d;
use moltopo::topology::{Topology, build_bonds_by_cutoff, build_angles_from_bonds, build_dihedrals_from_bonds, build_inversions_from_bonds};
use moltopo::xyz::read_xyz;
use moltopo::params::Params;
use moltopo::assign_uff;
use surfmol::mol_world::{MolWorld, BondedFFMode};
use molff::raff::FireState;
use molff::multigrid::{build_bend_twist_modes, ModalQuadratic};

const DIM: usize = 3;

// ============================================================================
// Helpers
// ============================================================================

fn save_xyz(path: &Path, apos: &[Vec3d], elems: &[String], comment: &str) {
    let mut f = File::create(path).expect("create xyz");
    writeln!(f, "{}", apos.len()).expect("natoms");
    writeln!(f, "{}", comment).expect("comment");
    for (ia, p) in apos.iter().enumerate() {
        writeln!(f, "{} {:.6} {:.6} {:.6}", elems[ia], p.x, p.y, p.z).expect("atom");
    }
}

/// Save a multi-frame xyz trajectory.
fn save_traj(path: &Path, frames: &[(Vec<Vec3d>, String)], elems: &[String]) {
    let mut f = File::create(path).expect("create traj");
    for (apos, comment) in frames {
        writeln!(f, "{}", apos.len()).expect("natoms");
        writeln!(f, "{}", comment).expect("comment");
        for (ia, p) in apos.iter().enumerate() {
            writeln!(f, "{} {:.6} {:.6} {:.6}", elems[ia], p.x, p.y, p.z).expect("atom");
        }
    }
}

fn z_rms(apos: &[Vec3d]) -> f64 {
    apos.iter().map(|p| p.z * p.z).sum::<f64>().sqrt() / (apos.len() as f64).sqrt()
}

fn fmax_flat(force: &[f64], natoms: usize) -> f64 {
    let mut f2max = 0.0f64;
    for i in 0..natoms {
        let f2 = force[i*3]*force[i*3] + force[i*3+1]*force[i*3+1] + force[i*3+2]*force[i*3+2];
        if f2 > f2max { f2max = f2; }
    }
    f2max.sqrt()
}

/// Projected force norm |ΦᵀF| — measures how much force is in the modal subspace.
fn projected_fmax(g: &[f64]) -> f64 {
    g.iter().map(|v| v*v).sum::<f64>().sqrt()
}

// ============================================================================
// Result structs
// ============================================================================

struct FireResult {
    n_force_evals: usize,   // full-force evals during THIS phase only (no setup)
    n_steps: i32,
    e_final: f64,
    fmax_final: f64,
    z_rms_final: f64,
    wall_ms: u128,
    converged: bool,
    trace: Vec<(i32, f64, f64)>, // (step, fmax, z_rms) convergence trace
}

struct ModalResult {
    n_sync: usize,          // full-force evals during coarse phase
    n_newton: usize,        // cheap modal Newton steps (no full-force)
    fmax_after_coarse: f64,
    gmax_after_coarse: f64, // projected force after coarse phase
    z_rms_after_coarse: f64,
    wall_ms: u128,
    trace: Vec<(usize, f64, f64, f64)>, // (sync, fmax, gmax, z_rms)
    final_apos: Vec<Vec3d>,
}

// ============================================================================
// FIRE relaxation (full-atom UFF) with convergence trace
// ============================================================================

/// Run full-atom FIRE. `n_force_offset` = full-force evals already done (e.g. coarse phase).
fn run_fire(mw: &mut MolWorld, niter: i32, fconv: f64, label: &str,
            n_force_offset: usize) -> FireResult {
    let mut fire = FireState::new(0.001, 0.05);
    let t0 = Instant::now();
    let mut n_evals = n_force_offset;
    let mut e_last = 0.0f64;
    let mut fmax_last = 0.0f64;
    let mut n_conv = niter;
    let mut converged = false;
    let mut trace = Vec::new();
    for itr in 0..niter {
        let (eb, ea, ed, ei, _, _) = mw.eval_forces();
        n_evals += 1;
        e_last = eb + ea + ed + ei;
        let dt = fire.dt;
        let mut v_dot_f = 0.0;
        let mut v_norm2 = 0.0;
        let mut f_norm2 = 0.0;
        let mut f2max = 0.0f64;
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
        fmax_last = f2max.sqrt();
        // Trace every 10 steps or at convergence
        if itr % 10 == 0 || fmax_last < fconv {
            let z = z_rms(mw.dyn_atoms.atoms.apos.as_slice());
            trace.push((itr + 1, fmax_last, z));
        }
        if fmax_last < fconv { n_conv = itr + 1; converged = true; break; }
    }
    let wall_ms = t0.elapsed().as_millis();
    let z = z_rms(mw.dyn_atoms.atoms.apos.as_slice());
    println!("  {}: n_steps={} n_force={} fmax={:.4e} z_rms={:.4e} wall={}ms{}",
        label, n_conv, n_evals, fmax_last, z, wall_ms,
        if converged { " [CONVERGED]" } else { " [NOT CONVERGED]" });
    FireResult { n_force_evals: n_evals, n_steps: n_conv, e_final: e_last,
                fmax_final: fmax_last, z_rms_final: z, wall_ms, converged, trace }
}

// ============================================================================
// Distortion generator
// ============================================================================

fn apply_bend_twist_noise(apos: &[Vec3d], bend_amp: f64, twist_amp: f64, noise_amp: f64, seed: u64,
                          axis: Vec3d, normal: Vec3d) -> Vec<Vec3d> {
    let n = apos.len();
    let u = axis * (1.0 / axis.norm());
    let mut nrm = normal - u * u.dot(normal);
    nrm = nrm * (1.0 / nrm.norm());
    let center: Vec3d = apos.iter().fold(Vec3d::new(0.0,0.0,0.0), |a, &p| a + p) * (1.0 / n as f64);
    let s: Vec<f64> = apos.iter().map(|p| (*p - center).dot(u)).collect();
    let smin = s.iter().copied().fold(f64::INFINITY, f64::min);
    let smax = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (smax - smin).max(1e-12);
    let mut rng = seed;
    let mut distorted = apos.to_vec();
    for i in 0..n {
        let t = (s[i] - smin) / span;
        let bend = nrm * (bend_amp * (std::f64::consts::PI * t).sin());
        let theta = twist_amp * (2.0 * t - 1.0);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let radial = apos[i] - center - u * s[i];
        let v_inplane = radial - nrm * radial.dot(nrm);
        let v_normal = nrm * radial.dot(nrm);
        let twisted = v_inplane * cos_t + v_inplane.cross(u) * sin_t + v_normal;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r1 = ((rng >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r2 = ((rng >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r3 = ((rng >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        distorted[i] = center + u * s[i] + twisted + bend
                     + Vec3d::new(noise_amp * r1, noise_amp * r2, noise_amp * r3);
    }
    distorted
}

// ============================================================================
// Setup: load pentacene with real UFF, in-plane relax, save state
// ============================================================================

struct Setup {
    planar_apos: Vec<Vec3d>,
    elems: Vec<String>,
    bonds: Vec<[i32; 2]>,
    axis: Vec3d,
    normal: Vec3d,
    top: Topology,
    bon_params: Vec<[f64; 2]>,
    ang_params: Vec<[f64; 5]>,
    dih_params: Vec<[f64; 3]>,
    inv_params: Vec<[f64; 4]>,
}

fn load_pentacene_setup(base: &Path) -> Setup {
    let xyz = read_xyz(&base.join("data/xyz/pentacene.xyz")).expect("read pentacene.xyz");
    let apos = xyz.apos;
    let elems = xyz.elems.clone();
    let natoms = apos.len() as i32;
    assert_eq!(natoms, 36);

    let mut params = Params::new();
    params.load_element_types(&base.join("data/ElementTypes.dat"));
    params.load_atom_types(&base.join("data/AtomTypes.dat"));
    params.load_bond_types(&base.join("data/BondTypes.dat"));
    params.load_angle_types(&base.join("data/AngleTypes.dat"));
    params.load_dihedral_types(&base.join("data/DihedralTypes.dat"));

    let bonds = build_bonds_by_cutoff(&apos, 1.8);
    let angles = build_angles_from_bonds(natoms, &bonds);
    let dihedrals = build_dihedrals_from_bonds(&bonds);
    let inversions = build_inversions_from_bonds(natoms, &bonds);
    let top = Topology { apos, bonds, angles, dihedrals, inversions };
    println!("[setup] topology: bonds={} angles={} dihedrals={} inversions={}",
        top.bonds.len(), top.angles.len(), top.dihedrals.len(), top.inversions.len());

    let mut mw = MolWorld::from_topology(&top);
    mw.bonded_mode = BondedFFMode::Uff;
    mw.make_neigh_bs();
    let neighs_arr: Vec<[i32; 4]> = mw.dyn_atoms.neighs().iter().map(|q| q.as_array()).collect();
    let types = assign_uff::assign_uff_types(&elems, &neighs_arr);
    mw.setup_uff_params(&params, &types);
    mw.bake_angle_neighs();
    mw.bake_dihedral_neighs();
    mw.bake_inversion_neighs();
    mw.map_atom_interactions();

    let n_inv = mw.uff.inv_params.as_slice().iter().filter(|p| p[0].abs() > 1e-10).count();
    println!("[setup] nonzero inversion params: {}/{}", n_inv, mw.uff.ninversions);
    assert!(n_inv > 0, "no inversion params — aromatic bending stiffness missing");

    let bon_params = mw.uff.bon_params.as_slice().to_vec();
    let ang_params = mw.uff.ang_params.as_slice().to_vec();
    let dih_params = mw.uff.dih_params.as_slice().to_vec();
    let inv_params = mw.uff.inv_params.as_slice().to_vec();

    // In-plane relaxation (inversions OFF — only relax in-plane strain)
    for ii in 0..mw.uff.ninversions as usize {
        mw.uff.inv_params.as_mut_slice()[ii] = [0.0, 1.0, -1.0, 0.0];
    }
    for i in 0..mw.natoms() {
        mw.dyn_atoms.atoms.apos.as_mut_slice()[i] = top.apos[i];
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::new(0.0, 0.0, 0.0);
    }
    mw.dyn_atoms.fapos.resize_fill(mw.natoms(), Vec3d::new(0.0, 0.0, 0.0));

    let dt = 0.02;
    let cdamp = 0.9;
    for itr in 0..3000 {
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
        if f2max.sqrt() < 1e-3 { println!("[setup] in-plane relaxed at step {itr}"); break; }
    }

    // Restore inversions and do FULL relaxation (inversions ON) to find the true minimum.
    // The in-plane-relaxed geometry has residual inversion forces (fmax ≈ 1e-3).
    // We need the true minimum of the full forcefield as the modal reference.
    for ii in 0..mw.uff.ninversions as usize {
        mw.uff.inv_params.as_mut_slice()[ii] = inv_params[ii];
    }
    // Reset velocities for the second phase
    for i in 0..mw.natoms() {
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::new(0.0, 0.0, 0.0);
    }
    let (eb, ea, ed, ei, _, _) = mw.eval_forces();
    let e_before = eb + ea + ed + ei;
    let mut f2max_before = 0.0f64;
    for ia in 0..mw.natoms() {
        let f = mw.dyn_atoms.fapos.as_slice()[ia];
        let f2 = f.norm2();
        if f2 > f2max_before { f2max_before = f2; }
    }
    println!("[setup] after restoring inversions: E={:.6e} fmax={:.6e}", e_before, f2max_before.sqrt());

    // Full relaxation with inversions ON (damped MD, tight threshold)
    for itr in 0..10000 {
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
        if f2max.sqrt() < 1e-5 { println!("[setup] full relaxed at step {itr} (fmax={:.6e})", f2max.sqrt()); break; }
    }
    let (eb, ea, ed, ei, _, _) = mw.eval_forces();
    let e_after = eb + ea + ed + ei;
    let mut f2max_after = 0.0f64;
    for ia in 0..mw.natoms() {
        let f = mw.dyn_atoms.fapos.as_slice()[ia];
        let f2 = f.norm2();
        if f2 > f2max_after { f2max_after = f2; }
    }
    let planar_apos: Vec<Vec3d> = mw.dyn_atoms.atoms.apos.as_slice().to_vec();
    let z_ref = z_rms(&planar_apos);
    println!("[setup] full relaxed: E={:.6e} fmax={:.6e} z_rms={:.6e}", e_after, f2max_after.sqrt(), z_ref);
    if z_ref > 0.01 {
        println!("[setup] WARNING: reference geometry is non-planar (z_rms={:.4e})!", z_ref);
        println!("[setup]   The UFF forcefield has a non-planar ground state for pentacene.");
        println!("[setup]   This means the dihedral/inversion terms favor a twisted geometry.");
    }

    // PCA for molecular axes
    let center = planar_apos.iter().fold(Vec3d::new(0.0,0.0,0.0), |a, &p| a + p) * (1.0 / planar_apos.len() as f64);
    let mut cov = [[0.0f64; 3]; 3];
    for &p in &planar_apos {
        let d = p - center;
        for a in 0..3 { for b in 0..3 { cov[a][b] += d.array()[a] * d.array()[b]; } }
    }
    let mut u = Vec3d::new(1.0, 0.0, 0.0);
    for _ in 0..50 {
        let mut nx = cov[0][0]*u.x + cov[0][1]*u.y + cov[0][2]*u.z;
        let mut ny = cov[1][0]*u.x + cov[1][1]*u.y + cov[1][2]*u.z;
        let mut nz = cov[2][0]*u.x + cov[2][1]*u.y + cov[2][2]*u.z;
        let nm = (nx*nx + ny*ny + nz*nz).sqrt().max(1e-30);
        nx /= nm; ny /= nm; nz /= nm;
        u = Vec3d::new(nx, ny, nz);
    }
    let mut n_axis = u.cross(Vec3d::new(0.0, 1.0, 0.0));
    let nm = n_axis.norm();
    if nm < 1e-6 { n_axis = u.cross(Vec3d::new(0.0, 0.0, 1.0)); }
    n_axis = n_axis * (1.0 / n_axis.norm());
    if n_axis.z < 0.0 { n_axis = n_axis * (-1.0); }
    println!("[setup] long axis u=({:.3},{:.3},{:.3}) normal n=({:.3},{:.3},{:.3})", u.x, u.y, u.z, n_axis.x, n_axis.y, n_axis.z);

    Setup { planar_apos, elems, bonds: top.bonds.clone(), axis: u, normal: n_axis,
            top, bon_params, ang_params, dih_params, inv_params }
}

fn build_molworld(setup: &Setup, initial_apos: &[Vec3d]) -> MolWorld {
    let mut mw = MolWorld::from_topology(&setup.top);
    mw.bonded_mode = BondedFFMode::Uff;
    mw.make_neigh_bs();
    mw.uff.bon_params.as_mut_slice().copy_from_slice(&setup.bon_params);
    mw.uff.ang_params.as_mut_slice().copy_from_slice(&setup.ang_params);
    mw.uff.dih_params.as_mut_slice().copy_from_slice(&setup.dih_params);
    mw.uff.inv_params.as_mut_slice().copy_from_slice(&setup.inv_params);
    mw.bake_angle_neighs();
    mw.bake_dihedral_neighs();
    mw.bake_inversion_neighs();
    mw.map_atom_interactions();
    for i in 0..mw.natoms() {
        mw.dyn_atoms.atoms.apos.as_mut_slice()[i] = initial_apos[i];
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::new(0.0, 0.0, 0.0);
    }
    mw.dyn_atoms.fapos.resize_fill(mw.natoms(), Vec3d::new(0.0, 0.0, 0.0));
    mw
}

// ============================================================================
// Strategy 1: Plain FIRE (baseline)
// ============================================================================

fn strategy_plain_fire(setup: &Setup, distorted: &[Vec3d], fconv: f64) -> (FireResult, Vec<Vec3d>) {
    let mut mw = build_molworld(setup, distorted);
    println!("\n--- Strategy 1: Plain FIRE (baseline) ---");
    let r = run_fire(&mut mw, 20000, fconv, "plain FIRE", 0);
    let final_apos = mw.dyn_atoms.atoms.apos.as_slice().to_vec();
    (r, final_apos)
}

// ============================================================================
// Strategy 2: Modal Newton + FIRE (proper timestep scaling)
// ============================================================================

/// Modal coarse phase: Newton steps with trust region.
///
/// For a quadratic model, one Newton step reaches the exact equilibrium:
///   dq = K⁻¹·g,  x_new = x_planar + Φ·(q + dq)
///
/// The trust region limits the step size for large distortions where the
/// quadratic model may be inaccurate. If the full-atom force increases after
/// a step, the trust radius is reduced and the step is retried.
///
/// Fitting cost is NOT counted — it's a one-time setup.
fn strategy_modal_fire(setup: &Setup, distorted: &[Vec3d], fconv: f64,
                       fit_radius: f64, max_syncs: usize, trust_radius: f64,
                       out_dir: &Path) -> (ModalResult, FireResult, Vec<Vec3d>) {
    let mut mw = build_molworld(setup, distorted);
    let natoms = mw.natoms();
    let n = natoms * DIM;

    println!("\n--- Strategy 2: Modal Newton + FIRE ---");
    println!("  [modal] fit_radius={} max_syncs={} trust_radius={}", fit_radius, max_syncs, trust_radius);

    // --- SETUP PHASE (not counted in benchmark) ---
    // Build modes and fit stiffness. This is done ONCE per molecule type.
    let phi = build_bend_twist_modes(&setup.planar_apos, setup.axis, setup.normal);
    let n_modes = 2;
    println!("  [modal] built {} modes (bend + twist) from planar reference", n_modes);

    let neighs = mw.dyn_atoms.neighs().to_vec();
    let neigh_bs = mw.dyn_atoms.neigh_bs().to_vec();

    // Fit K from 2*n_modes force evals (central differences at planar geometry)
    let mut force_minus = vec![0.0f64; n * n_modes];
    let mut force_plus = vec![0.0f64; n * n_modes];
    let t_fit = Instant::now();
    for mode in 0..n_modes {
        let mut dx_plus = setup.planar_apos.clone();
        let mut dx_minus = setup.planar_apos.clone();
        for i in 0..natoms {
            for d in 0..DIM {
                let q = i * DIM + d;
                let phi_val = phi[q * n_modes + mode];
                dx_plus[i].array_mut()[d] += fit_radius * phi_val;
                dx_minus[i].array_mut()[d] -= fit_radius * phi_val;
            }
        }
        let mut fpos = vec![Vec3d::new(0.0,0.0,0.0); natoms];
        mw.uff.eval_forces(&dx_plus, &mut fpos, &neighs, &neigh_bs);
        for i in 0..natoms {
            for d in 0..DIM { force_plus[mode * n + i * DIM + d] = fpos[i].array()[d]; }
        }
        mw.uff.eval_forces(&dx_minus, &mut fpos, &neighs, &neigh_bs);
        for i in 0..natoms {
            for d in 0..DIM { force_minus[mode * n + i * DIM + d] = fpos[i].array()[d]; }
        }
    }
    let fit_ms = t_fit.elapsed().as_millis();
    let modal = ModalQuadratic::fit_central(&phi, n, n_modes, fit_radius, &force_minus, &force_plus);
    // Modal frequencies: f = sqrt(K/m), dt_max = 10/f
    let k_max = modal.k[0].max(modal.k[3]).max(modal.k[1].abs());
    let f_max_modal = k_max.sqrt();
    let dt_max_modal = 10.0 / f_max_modal;
    println!("  [modal] K = [{:.4}, {:.4}; {:.4}, {:.4}]  f_max={:.4}  dt_max={:.1}",
        modal.k[0], modal.k[1], modal.k[2], modal.k[3], f_max_modal, dt_max_modal);
    println!("  [modal] fit cost: {} force evals, {}ms (NOT counted — setup)", 2*n_modes, fit_ms);

    // --- SIMULATION PHASE (counted in benchmark) ---
    let t0 = Instant::now();
    let mut n_sync = 0;       // full-force evals during coarse phase
    let mut n_newton = 0;     // cheap modal Newton steps
    let mut current_apos = distorted.to_vec();
    let mut traj_frames: Vec<(Vec<Vec3d>, String)> = Vec::new();

    // Helper: project positions to modal coords q = Φᵀ(x - x_planar)
    let project_to_modal = |apos: &[Vec3d]| -> Vec<f64> {
        let mut q = vec![0.0f64; n_modes];
        for i in 0..apos.len() {
            for d in 0..DIM {
                let qi = i * DIM + d;
                let dx = apos[i].array()[d] - setup.planar_apos[i].array()[d];
                for m in 0..n_modes { q[m] += phi[qi * n_modes + m] * dx; }
            }
        }
        q
    };

    // Helper: reconstruct x = x_planar + Φ·q
    let reconstruct = |q: &[f64]| -> Vec<Vec3d> {
        let mut apos = setup.planar_apos.clone();
        for i in 0..natoms {
            for d in 0..DIM {
                let qi = i * DIM + d;
                for m in 0..n_modes { apos[i].array_mut()[d] += phi[qi * n_modes + m] * q[m]; }
            }
        }
        apos
    };

    // Initial full force evaluation (sync 0)
    let mut fpos = vec![Vec3d::new(0.0,0.0,0.0); natoms];
    mw.uff.eval_forces(&current_apos, &mut fpos, &neighs, &neigh_bs);
    n_sync += 1;
    let mut force_flat = vec![0.0f64; n];
    for i in 0..natoms {
        force_flat[i*3] = fpos[i].x; force_flat[i*3+1] = fpos[i].y; force_flat[i*3+2] = fpos[i].z;
    }
    let mut fmax_curr = fmax_flat(&force_flat, natoms);
    let mut g_curr = modal.project_force(&force_flat);
    let mut gmax_curr = projected_fmax(&g_curr);
    let q_init = project_to_modal(&current_apos);
    println!("  [modal] sync 0: fmax={:.4e} gmax={:.4e} q=[{:.4}, {:.4}]", fmax_curr, gmax_curr, q_init[0], q_init[1]);
    traj_frames.push((current_apos.clone(), format!("sync 0: fmax={:.4e} gmax={:.4e}", fmax_curr, gmax_curr)));

    let mut trace: Vec<(usize, f64, f64, f64)> = vec![(0, fmax_curr, gmax_curr, z_rms(&current_apos))];
    let mut trust = trust_radius;

    for sync_iter in 0..max_syncs {
        // Check convergence: if projected force is small, soft DOFs are converged
        if gmax_curr < fconv * 0.1 {
            println!("  [modal] soft DOFs converged (gmax={:.4e} < {:.4e}) after {} syncs, {} Newton steps",
                gmax_curr, fconv * 0.1, n_sync, n_newton);
            break;
        }

        // Newton step: dq = K⁻¹·g, then x_new = x_planar + Φ·(q + dq)
        // For quadratic model, this reaches equilibrium EXACTLY.
        // Trust region: scale step if |dq| > trust
        let q = project_to_modal(&current_apos);
        let mut dx = vec![0.0f64; n];
        let dq = modal.solve_force(&force_flat, &mut dx);
        n_newton += 1;

        let dq_norm = dq.iter().map(|v| v*v).sum::<f64>().sqrt();
        let scale = if dq_norm > trust { trust / dq_norm } else { 1.0 };
        let q_new: Vec<f64> = q.iter().zip(dq.iter()).map(|(&qi, &dqi)| qi + scale * dqi).collect();
        current_apos = reconstruct(&q_new);

        // Sync: evaluate full force at new position
        n_sync += 1;
        mw.uff.eval_forces(&current_apos, &mut fpos, &neighs, &neigh_bs);
        for i in 0..natoms {
            force_flat[i*3] = fpos[i].x; force_flat[i*3+1] = fpos[i].y; force_flat[i*3+2] = fpos[i].z;
        }
        let fmax_new = fmax_flat(&force_flat, natoms);
        let g_new = modal.project_force(&force_flat);
        let gmax_new = projected_fmax(&g_new);

        // Trust region adaptation: if force increased, reduce trust and retry
        if fmax_new > fmax_curr * 1.01 {
            trust *= 0.5;
            println!("  [modal] sync {}: fmax INCREASED ({:.4e} → {:.4e}), reducing trust to {:.4}",
                n_sync, fmax_curr, fmax_new, trust);
            // Revert: go back to previous position
            current_apos = reconstruct(&q);
            // Re-evaluate force at reverted position (but don't count — it's the same as before)
            // Actually we need to restore force_flat too. Simplest: just continue from here.
            mw.uff.eval_forces(&current_apos, &mut fpos, &neighs, &neigh_bs);
            for i in 0..natoms {
                force_flat[i*3] = fpos[i].x; force_flat[i*3+1] = fpos[i].y; force_flat[i*3+2] = fpos[i].z;
            }
            n_sync += 1; // count the revert eval
            continue;
        } else {
            trust = (trust * 1.5).min(trust_radius);
        }

        fmax_curr = fmax_new;
        g_curr = g_new;
        gmax_curr = gmax_new;
        let z = z_rms(&current_apos);
        trace.push((n_sync, fmax_curr, gmax_curr, z));
        traj_frames.push((current_apos.clone(),
            format!("sync {}: fmax={:.4e} gmax={:.4e} q=[{:.4},{:.4}]", n_sync, fmax_curr, gmax_curr, q_new[0], q_new[1])));
        println!("  [modal] sync {}: fmax={:.4e} gmax={:.4e} q=[{:.4}, {:.4}] trust={:.3}",
            n_sync, fmax_curr, gmax_curr, q_new[0], q_new[1], trust);
    }

    let coarse_wall = t0.elapsed().as_millis();
    let modal_result = ModalResult {
        n_sync, n_newton,
        fmax_after_coarse: fmax_curr,
        gmax_after_coarse: gmax_curr,
        z_rms_after_coarse: z_rms(&current_apos),
        wall_ms: coarse_wall,
        trace,
        final_apos: current_apos.clone(),
    };

    // Save coarse trajectory
    save_traj(&out_dir.join("traj_modal_coarse.xyz"), &traj_frames, &setup.elems);

    // Apply modal-relaxed positions to MolWorld for FIRE finishing
    for i in 0..natoms {
        mw.dyn_atoms.atoms.apos.as_mut_slice()[i] = current_apos[i];
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::new(0.0, 0.0, 0.0);
    }
    mw.dyn_atoms.fapos.resize_fill(mw.natoms(), Vec3d::new(0.0, 0.0, 0.0));

    println!("  [modal] coarse phase: {} syncs (full-force), {} Newton steps, wall={}ms",
        n_sync, n_newton, coarse_wall);
    println!("  [modal] after coarse: fmax={:.4e} gmax={:.4e} z_rms={:.4e}",
        fmax_curr, gmax_curr, z_rms(&current_apos));

    // Fine phase: FIRE to relax hard DOFs
    let fire_result = run_fire(&mut mw, 20000, fconv, "modal + FIRE", n_sync);
    let final_apos = mw.dyn_atoms.atoms.apos.as_slice().to_vec();
    println!("  [modal] TOTAL simulation: {} full-force evals ({} sync + {} FIRE)",
        fire_result.n_force_evals, n_sync, fire_result.n_force_evals - n_sync);

    (modal_result, fire_result, final_apos)
}

// ============================================================================
// Main test
// ============================================================================

#[test]
fn pentacene_speedup_benchmark() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().parent().unwrap();
    let out_dir = base.join("debug/relax_pentacene_speedup");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    println!("============================================================");
    println!("Pentacene modal relaxation speedup benchmark — v2");
    println!("  Newton step + trust region + proper work accounting");
    println!("============================================================");

    let setup = load_pentacene_setup(base);
    save_xyz(&out_dir.join("pentacene_planar.xyz"), &setup.planar_apos, &setup.elems,
        "pentacene planar baseline (in-plane relaxed, inv ON)");

    // Distortion parameters
    let bend_amp = 0.5;
    let twist_amp = 0.3;
    let noise_amp = 0.02;
    let fconv = 1e-3;

    let distorted = apply_bend_twist_noise(&setup.planar_apos, bend_amp, twist_amp, noise_amp, 42,
        setup.axis, setup.normal);
    save_xyz(&out_dir.join("pentacene_distorted.xyz"), &distorted, &setup.elems,
        &format!("pentacene distorted (bend={:.2}Å twist={:.2}rad noise={:.3}Å)", bend_amp, twist_amp, noise_amp));
    println!("\ndistorted geometry: z_rms={:.4} (bend+twist+noise)", z_rms(&distorted));

    // --- Strategy 1: Plain FIRE ---
    let (r1, final_plain) = strategy_plain_fire(&setup, &distorted, fconv);
    save_xyz(&out_dir.join("result_plain_fire.xyz"), &final_plain, &setup.elems,
        &format!("plain FIRE: {} evals, fmax={:.4e}, z_rms={:.4e}", r1.n_force_evals, r1.fmax_final, r1.z_rms_final));

    // Save plain FIRE trajectory trace
    {
        let mut f = File::create(&out_dir.join("trace_plain_fire.tsv")).expect("create trace");
        writeln!(f, "step\tfmax\tz_rms").expect("header");
        for (s, fm, z) in &r1.trace { writeln!(f, "{}\t{:.6e}\t{:.6e}", s, fm, z).expect("row"); }
    }

    // --- Strategy 2: Modal Newton + FIRE ---
    let (modal_res, r2, final_modal) = strategy_modal_fire(&setup, &distorted, fconv, 0.1, 30, 2.0, &out_dir);
    save_xyz(&out_dir.join("result_modal_fire.xyz"), &final_modal, &setup.elems,
        &format!("modal + FIRE: {} evals, fmax={:.4e}, z_rms={:.4e}", r2.n_force_evals, r2.fmax_final, r2.z_rms_final));

    // Save modal trace
    {
        let mut f = File::create(&out_dir.join("trace_modal.tsv")).expect("create trace");
        writeln!(f, "sync\tfmax\tgmax\tz_rms").expect("header");
        for (s, fm, gm, z) in &modal_res.trace { writeln!(f, "{}\t{:.6e}\t{:.6e}\t{:.6e}", s, fm, gm, z).expect("row"); }
    }

    // --- Summary ---
    println!("\n============================================================");
    println!("=== SPEEDUP BENCHMARK RESULTS (v2) ===");
    println!("============================================================");
    println!("distortion: bend={:.2}Å twist={:.2}rad noise={:.3}Å  fconv={:.0e}", bend_amp, twist_amp, noise_amp, fconv);
    println!();
    println!("{:<25} {:>10} {:>10} {:>10} {:>10} {:>8}", "strategy", "N_force", "N_steps", "fmax", "z_rms", "wall[ms]");
    println!("{}", "-".repeat(83));
    println!("{:<25} {:>10} {:>10} {:>10.4e} {:>10.4e} {:>8}",
        "plain FIRE", r1.n_force_evals, r1.n_steps, r1.fmax_final, r1.z_rms_final, r1.wall_ms);
    println!("{:<25} {:>10} {:>10} {:>10.4e} {:>10.4e} {:>8}",
        "modal + FIRE", r2.n_force_evals, r2.n_steps, r2.fmax_final, r2.z_rms_final, r2.wall_ms);
    println!();
    let speedup = r1.n_force_evals as f64 / r2.n_force_evals as f64;
    println!("speedup: {:.2}× (N_force {} → {})", speedup, r1.n_force_evals, r2.n_force_evals);
    println!("  coarse phase: {} syncs + {} Newton steps ({} full-force evals)",
        modal_res.n_sync, modal_res.n_newton, modal_res.n_sync);
    println!("  fine phase:   {} FIRE steps ({} full-force evals)",
        r2.n_steps, r2.n_force_evals - modal_res.n_sync);

    // --- Check for different minima ---
    println!("\n--- Minimum comparison ---");
    println!("  plain FIRE:  z_rms={:.6e}  E={:.6e}", r1.z_rms_final, r1.e_final);
    println!("  modal + FIRE: z_rms={:.6e}  E={:.6e}", r2.z_rms_final, r2.e_final);
    let z_diff = (r1.z_rms_final - r2.z_rms_final).abs();
    if z_diff > 0.01 {
        println!("  WARNING: different minima! z_rms diff = {:.4e}", z_diff);
        println!("    → Check trajectories visually: debug/relax_pentacene_speedup/traj_modal_coarse.xyz");
        println!("    → This is SUSPICIOUS — plain FIRE should also find the planar minimum.");
        println!("    → Possible causes: UFF dihedral multi-well, FIRE momentum trap, or forcefield bug.");
    }

    // --- Planar stability check: FIRE from planar + small noise ---
    println!("\n--- Planar stability check ---");
    let small_noise = apply_bend_twist_noise(&setup.planar_apos, 0.0, 0.0, 0.02, 99,
        setup.axis, setup.normal);
    let (r_stable, final_stable) = strategy_plain_fire(&setup, &small_noise, fconv);
    save_xyz(&out_dir.join("result_planar_stability.xyz"), &final_stable, &setup.elems,
        &format!("planar stability: {} evals, z_rms={:.4e}", r_stable.n_force_evals, r_stable.z_rms_final));
    println!("  planar + noise → FIRE: z_rms={:.6e} E={:.6e} (should be ~0 if planar is the ground state)", r_stable.z_rms_final, r_stable.e_final);

    // --- Energy at exact planar geometry ---
    {
        let mut mw = build_molworld(&setup, &setup.planar_apos);
        let (eb, ea, ed, ei, _, _) = mw.eval_forces();
        let e_planar = eb + ea + ed + ei;
        let fmax_planar = {
            let mut f2max = 0.0f64;
            for ia in 0..mw.natoms() {
                let f = mw.dyn_atoms.fapos.as_slice()[ia];
                let f2 = f.norm2();
                if f2 > f2max { f2max = f2; }
            }
            f2max.sqrt()
        };
        println!("  exact planar: E={:.6e} fmax={:.6e}", e_planar, fmax_planar);
    }

    // --- Distortion amplitude sweep: when do minima diverge? ---
    println!("\n============================================================");
    println!("=== DISTORTION AMPLITUDE SWEEP ===");
    println!("============================================================");
    println!("Sweeping bend amplitude to find where plain FIRE diverges from planar minimum");
    println!();

    let sweep_amps: Vec<f64> = vec![0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0];
    let mut sweep_rows: Vec<Vec<f64>> = Vec::new();

    for &amp in &sweep_amps {
        let dist = apply_bend_twist_noise(&setup.planar_apos, amp, amp * 0.6, 0.01, 42,
            setup.axis, setup.normal);

        // Plain FIRE
        let (r_fire, _) = strategy_plain_fire(&setup, &dist, fconv);

        // Modal + FIRE
        let (_, r_modal, _) = strategy_modal_fire(&setup, &dist, fconv, 0.1, 30, 2.0, &out_dir);

        let same = (r_fire.z_rms_final - r_modal.z_rms_final).abs() < 0.01;
        let speedup = r_fire.n_force_evals as f64 / r_modal.n_force_evals as f64;
        println!("  amp={:.2}: FIRE(z={:.4e},E={:.4e},N={}) modal(z={:.4e},E={:.4e},N={}) speedup={:.1}x {}",
            amp, r_fire.z_rms_final, r_fire.e_final, r_fire.n_force_evals,
            r_modal.z_rms_final, r_modal.e_final, r_modal.n_force_evals,
            speedup, if same { "[SAME]" } else { "[DIFFERENT]" });

        sweep_rows.push(vec![amp, r_fire.n_force_evals as f64, r_fire.z_rms_final, r_fire.e_final,
            r_modal.n_force_evals as f64, r_modal.z_rms_final, r_modal.e_final,
            speedup, if same { 1.0 } else { 0.0 }]);
    }

    // Write sweep TSV
    let sweep_path = out_dir.join("sweep_amplitude.tsv");
    let mut f = File::create(&sweep_path).expect("create sweep");
    writeln!(f, "amp\tN_fire\tz_fire\tE_fire\tN_modal\tz_modal\tE_modal\tspeedup\tsame").expect("header");
    for row in &sweep_rows {
        let s: Vec<String> = row.iter().map(|v| format!("{:.6e}", v)).collect();
        writeln!(f, "{}", s.join("\t")).expect("row");
    }
    println!("\nsweep: {}", sweep_path.display());

    // --- Assertions ---
    assert!(r1.converged, "plain FIRE did not converge");
    assert!(r2.converged, "modal + FIRE did not converge");

    // Write summary TSV
    let trace_path = out_dir.join("speedup_summary.tsv");
    let mut f = File::create(&trace_path).expect("create summary");
    writeln!(f, "strategy\tN_force\tN_steps\tfmax\tz_rms\tE\twall_ms").expect("header");
    writeln!(f, "plain_fire\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}\t{}", r1.n_force_evals, r1.n_steps, r1.fmax_final, r1.z_rms_final, r1.e_final, r1.wall_ms).expect("row");
    writeln!(f, "modal_fire\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}\t{}", r2.n_force_evals, r2.n_steps, r2.fmax_final, r2.z_rms_final, r2.e_final, r2.wall_ms).expect("row");
    println!("\nsummary: {}", trace_path.display());

    println!("\n=== PASS: all strategies converged to force threshold {:.0e} ===", fconv);
}
