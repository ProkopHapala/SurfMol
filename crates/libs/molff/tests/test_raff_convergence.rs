//! RAFF convergence-to-same-geometry benchmark (Q2a).
//!
//! Verifies that force-based MD and the three position-based solvers
//! (PBD-with-compliance, true XPBD, Projective Dynamics) relax a perturbed
//! molecule to the **same** equilibrium geometry, up to rigid-body drift.
//!
//! Both families are translation+rotation invariant → absolute frames drift,
//! so we compare via `kabsch_rmsd` (optimal rigid alignment). This is the
//! correctness gate before the parameter-sweep/perf benchmark (Q2b): if the
//! solvers do not agree on the answer, comparing their speed is meaningless.
//!
//! Reference geometry: the **input** geometry is the exact E=0 equilibrium by
//! construction — `l0` is set to each bond's input length and ports use per-atom
//! ARAP (`set_port_geometry_from_reference`), so identity rotation gives tip=x_j
//! for every port. This makes the reference exact and free (no slow force-MD
//! pre-relaxation), and lets us test force-MD *itself* against the same answer.
//!
//! Per AGENTS.md §Tests Are Diagnostics: on failure, prints per-atom residuals
//! and the worst contributor so the bug is locatable without re-running.

use molff::raff::*;
use numtypes::{Quat4d, Vec3d};

fn nb_off() -> NbConfig { NbConfig { enabled: false, ..Default::default() } }

/// Build a molecule from positions + bonds. Sets `l0` = actual input bond length
/// per bond and uses per-atom ARAP port geometry, so the input is the exact E=0
/// equilibrium (identity rotation → tip = x_j for every port).
fn make_molecule(apos: &[Vec3d], bonds: &[[i32; 2]], k_bond: f64) -> (RaffState, RaffTopology) {
    let n = apos.len();
    let mut topo = RaffTopology::new(n);
    topo.bond_params = bonds.iter().map(|b| {
        let d = apos[b[1] as usize] - apos[b[0] as usize];
        PortParam { k_p: k_bond / 2.0, l0: d.norm() }   // k_p = K_bond/2 (reciprocal ports)
    }).collect();
    topo.build_neighs_from_bonds(bonds);
    topo.set_port_geometry_from_reference(apos);        // ARAP: ports = input neighbor directions
    let mut state = RaffState::new(n);
    state.set_positions(apos);
    (state, topo)
}

fn make_ch4() -> (RaffState, RaffTopology) {
    let s = 1.0 / 3.0_f64.sqrt();
    make_molecule(
        &[Vec3d::new(0.0,0.0,0.0), Vec3d::new(s,s,s), Vec3d::new(s,-s,-s),
          Vec3d::new(-s,s,-s), Vec3d::new(-s,-s,s)],
        &[[0,1],[0,2],[0,3],[0,4]], 100.0)
}

fn make_water() -> (RaffState, RaffTopology) {
    make_molecule(
        &[Vec3d::new(0.0,0.0,0.0), Vec3d::new(0.96,0.0,0.0), Vec3d::new(-0.24,0.93,0.0)],
        &[[0,1],[0,2]], 100.0)
}

/// 4-atom bent chain A-B-C-D. NOTE: the port model has a **dihedral null space**
/// for chains (each atom's rotation is set by its own neighbors via Wahba; the
/// A-B-C-D dihedral is not constrained). So "same geometry" only holds up to a
/// dihedral flip — see `test_same_geometry_chain4_dihedral`.
fn make_chain4() -> (RaffState, RaffTopology) {
    make_molecule(
        &[Vec3d::new(0.0,0.0,0.0), Vec3d::new(1.0,0.0,0.0),
          Vec3d::new(1.7,0.8,0.0), Vec3d::new(2.4,0.0,0.0)],
        &[[0,1],[1,2],[2,3]], 100.0)
}

/// Perturb a copy: displace a few atoms moderately (stays in the basin of the
/// original equilibrium so all solvers converge back to it).
fn perturb(state: &RaffState) -> RaffState {
    let mut s = state.clone();
    s.pos[1] = s.pos[1] + Vec3d::new(0.15, 0.0, 0.0);
    if s.natoms > 2 { s.pos[2] = s.pos[2] + Vec3d::new(0.0, 0.10, 0.0); }
    if s.natoms > 3 { s.pos[3] = s.pos[3] + Vec3d::new(-0.08, 0.0, 0.05); }
    // zero velocities / angular velocities for a fresh relaxation
    for i in 0..s.natoms { s.vel[i] = Vec3d::new(0.0,0.0,0.0); s.omega[i] = Vec3d::new(0.0,0.0,0.0); }
    s
}

/// Relax with force-MD (Adiabatic). Returns (final_E, n_steps). Direct loop (controlled verbosity).
fn relax_md(init: &RaffState, topo: &RaffTopology, max_steps: usize, f_tol: f64) -> (RaffState, f64, usize) {
    let mut s = init.clone();
    solve_all_rotations(&mut s, topo);
    let cfg = RaffConfig { orient_mode: OrientMode::Adiabatic, dyn_mode: DynMode::ForceMD,
        pos_solver: PosSolver::PbdCompliance, dt: 0.002, cdamp: 0.85, rot_damp: 0.85, flim: 0.0,
        xpbd_iters: 16, xpbd_over_relax: 1.0, ..Default::default() };
    let n = s.natoms;
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut tau   = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut e_last = f64::INFINITY;
    let mut step = 0;
    for st in 0..max_steps {
        solve_all_rotations(&mut s, topo);
        let (e, max_f, _max_t) = step_force_md(&mut s, topo, &cfg, &mut fapos, &mut tau, &nb_off());
        e_last = e; step = st + 1;
        if max_f < f_tol { break; }
    }
    let _ = (fapos, tau);
    (s, e_last, step)
}

/// Relax with a position-based solver. Returns (final_E, n_macrosteps, n_port_evals).
fn relax_pos(init: &RaffState, topo: &RaffTopology, solver: PosSolver, dt: f64, iters: usize, max_macro: usize, e_tol: f64)
    -> (RaffState, f64, usize, usize) {
    let mut s = init.clone();
    solve_all_rotations(&mut s, topo);
    let cfg = RaffConfig { orient_mode: OrientMode::Adiabatic, dyn_mode: DynMode::Xpbd,
        pos_solver: solver, dt, cdamp: 0.0, rot_damp: 0.0, flim: 0.0,
        xpbd_iters: iters, xpbd_over_relax: 1.0, ..Default::default() };
    let mut last_e = f64::INFINITY;
    let mut n_evals = 0usize;
    let mut step = 0;
    for st in 0..max_macro {
        let e = step_position_based(&mut s, topo, &cfg, &nb_off());
        n_evals += 1; step = st + 1;
        if (last_e - e).abs() < e_tol && st > 5 { last_e = e; break; }
        last_e = e;
    }
    (s, last_e, step, n_evals)
}

fn report_residuals(label: &str, got: &[Vec3d], ref_pos: &[Vec3d]) {
    let mut worst = (0usize, 0.0f64);
    for i in 0..got.len() {
        let d = (got[i] - ref_pos[i]).norm();
        if d > worst.1 { worst = (i, d); }
        eprintln!("  {} atom {:>2}: |Δ|={:.4e}", label, i, d);
    }
    eprintln!("  {} worst: atom {} |Δ|={:.4e}", label, worst.0, worst.1);
}

/// Run the full convergence comparison for one molecule with an exact (input) reference.
/// Each solver starts from the SAME perturbed copy of `init`.
fn check_same_geometry(name: &str, init: &RaffState, topo: &RaffTopology, rmsd_tol: f64) {
    let ref_pos: Vec<Vec3d> = init.pos.clone();   // exact E=0 equilibrium (by construction)
    eprintln!("\n=== {} (ref = input geometry, exact E=0) ===", name);

    // Force-MD
    let p0 = perturb(init);
    let (s_md, e_md, n_md) = relax_md(&p0, topo, 200_000, 1e-10);
    let r_md = kabsch_rmsd(&s_md.pos, &ref_pos);
    eprintln!("ForceMD      : n_steps={:>6} E={:.4e}  RMSD={:.4e}", n_md, e_md, r_md);
    if r_md > rmsd_tol { report_residuals("ForceMD", &s_md.pos, &ref_pos); }
    assert!(r_md < rmsd_tol, "{}: ForceMD did not return to input geometry: RMSD={:.4e} > tol={:.4e}", name, r_md, rmsd_tol);

    // Position-based solvers — each from a fresh perturbed copy
    let cases: [(PosSolver, &str, f64, usize); 3] = [
        (PosSolver::PbdCompliance, "PBD-compl",   0.05, 32),
        (PosSolver::Xpbd,          "XPBD-lag",    0.05, 32),
        (PosSolver::Projective,    "Projective",  0.05, 32),
    ];
    for (solver, label, dt, iters) in cases {
        let p = perturb(init);
        let (s, e, n, n_ev) = relax_pos(&p, topo, solver, dt, iters, 50_000, 1e-14);
        let r = kabsch_rmsd(&s.pos, &ref_pos);
        eprintln!("{:12}: n_macro={:>6} n_evals={:>6} E={:.4e}  RMSD={:.4e}", label, n, n_ev, e, r);
        if r > rmsd_tol { report_residuals(label, &s.pos, &ref_pos); }
        assert!(r < rmsd_tol,
            "{}: {} did not converge to the input geometry: RMSD={:.4e} > tol={:.4e} (E={:.3e}). See per-atom residuals above.",
            name, label, r, rmsd_tol, e);
    }
}

#[test]
fn test_same_geometry_ch4() {
    let (state, topo) = make_ch4();
    check_same_geometry("CH4", &state, &topo, 1e-3);
}

#[test]
fn test_same_geometry_water() {
    let (state, topo) = make_water();
    check_same_geometry("water", &state, &topo, 1e-3);
}

/// Chain4 has a dihedral null space in the port model — all solvers reach E≈0
/// (constraints satisfied) but the A-B-C-D dihedral can drift differently per
/// solver. This test documents that: it asserts E→0 for every solver (the
/// *constraint-satisfied* geometry is reached) and reports the RMSD vs the
/// planar input as a diagnostic, but does NOT assert RMSD < tol, because the
/// dihedral is a genuine free DOF, not a bug.
#[test]
fn test_same_geometry_chain4_dihedral() {
    let (state, topo) = make_chain4();
    let ref_pos: Vec<Vec3d> = state.pos.clone();
    eprintln!("\n=== chain4 (dihedral null space — informational) ===");
    let p0 = perturb(&state);
    let (s_md, e_md, n_md) = relax_md(&p0, &topo, 200_000, 1e-10);
    eprintln!("ForceMD      : n_steps={:>6} E={:.4e}  RMSD={:.4e}", n_md, e_md, kabsch_rmsd(&s_md.pos, &ref_pos));
    let cases: [(PosSolver, &str, f64, usize); 3] = [
        (PosSolver::PbdCompliance, "PBD-compl",   0.05, 32),
        (PosSolver::Xpbd,          "XPBD-lag",    0.05, 32),
        (PosSolver::Projective,    "Projective",  0.05, 32),
    ];
    for (solver, label, dt, iters) in cases {
        let p = perturb(&state);
        let (s, e, n, n_ev) = relax_pos(&p, &topo, solver, dt, iters, 50_000, 1e-14);
        let r = kabsch_rmsd(&s.pos, &ref_pos);
        eprintln!("{:12}: n_macro={:>6} n_evals={:>6} E={:.4e}  RMSD={:.4e}  (dihedral drift — not asserted)", label, n, n_ev, e, r);
        // Every solver must satisfy the port constraints (E→0). The dihedral may differ.
        assert!(e < 1e-6, "chain4: {} did not satisfy port constraints: E={:.4e}", label, e);
    }
}

/// Kabsch invariants: identical/translated/rotated → 0; genuinely different → > 0.
#[test]
fn test_kabsch_invariants() {
    let a = vec![
        Vec3d::new(0.0,0.0,0.0), Vec3d::new(1.0,0.0,0.0),
        Vec3d::new(0.0,1.0,0.0), Vec3d::new(0.0,0.0,1.0),
    ];
    assert!(kabsch_rmsd(&a, &a) < 1e-12, "identical RMSD should be 0, got {:.2e}", kabsch_rmsd(&a, &a));
    let b: Vec<Vec3d> = a.iter().map(|p| *p + Vec3d::new(5.0,-3.0,2.0)).collect();
    assert!(kabsch_rmsd(&a, &b) < 1e-12, "translated RMSD should be 0, got {:.2e}", kabsch_rmsd(&a, &b));
    let c: Vec<Vec3d> = a.iter().map(|p| Vec3d::new(-p.y, p.x, p.z)).collect(); // 90° around z
    assert!(kabsch_rmsd(&a, &c) < 1e-12, "rotated RMSD should be 0, got {:.2e}", kabsch_rmsd(&a, &c));
    let mut d = a.clone(); d[1] = d[1] + Vec3d::new(0.5,0.0,0.0);
    let r = kabsch_rmsd(&a, &d);
    assert!(r > 1e-6, "different config RMSD should be > 0, got {:.2e}", r);
    eprintln!("kabsch invariants OK (different-config RMSD = {:.4e})", r);
}

#[test]
fn test_wahba_single_call_from_bad_orientation() {
    let (mut s, topo) = make_ch4();
    s.quat[0] = Quat4d::new(1.0, 0.0, 0.0, 0.0); // 180° about x: deliberately poor warm start
    solve_all_rotations(&mut s, &topo);
    let mut f = vec![Vec3d::new(0.0, 0.0, 0.0); s.natoms];
    let mut tau = f.clone();
    let e = eval_port_forces(&s, &topo, &mut f, &mut tau);
    let tmax = tau.iter().map(|t| t.norm()).fold(0.0f64, f64::max);
    eprintln!("Wahba bad-start diagnostic: E={:.6e} max|tau|={:.6e}", e, tmax);
    assert!(e < 1e-10 && tmax < 1e-8, "one Wahba solve did not recover equilibrium from a poor quaternion: E={:.3e}, max|tau|={:.3e}", e, tmax);
}

#[test]
fn test_projective_dynamic_orientation_carries_omega() {
    let (state, topo) = make_ch4();
    let mut s = perturb(&state);
    let q0 = s.quat[0];
    let cfg = RaffConfig { orient_mode: OrientMode::Dynamic, dyn_mode: DynMode::Xpbd, pos_solver: PosSolver::Projective,
        dt: 0.001, cdamp: 1.0, rot_damp: 1.0, flim: 0.0, xpbd_iters: 1, xpbd_over_relax: 1.0,
        pd_inertia: true, vel_reset: false, bmix_start: 0.0, bmix_end: 0.0, bmix_istart: 0, bmix_iend: 1, ..Default::default() };
    let e = step_position_based(&mut s, &topo, &cfg, &nb_off());
    let w = s.omega[0].norm();
    let dq = ((s.quat[0].x-q0.x).powi(2) + (s.quat[0].y-q0.y).powi(2) + (s.quat[0].z-q0.z).powi(2) + (s.quat[0].w-q0.w).powi(2)).sqrt();
    eprintln!("Projective dynamic rotation diagnostic: E={:.6e} |omega_0|={:.6e} |dq_0|={:.6e}", e, w, dq);
    assert!(e.is_finite() && w > 1e-12 && dq > 1e-12, "Projective dynamic orientation did not advance torque/omega state: E={e}, |omega|={w}, |dq|={dq}");
}

#[test]
fn test_pd_dynamic_inner_rotation_converges() {
    // Dynamic PD with inner-loop rotational Jacobi: translation and rotation updated together
    // each inner iteration. Converges at dt=0.1 (where outer-only dynamic diverged).
    let (state, topo) = make_ch4();
    let mut s = perturb(&state);
    let cfg = RaffConfig { orient_mode: OrientMode::Dynamic, dyn_mode: DynMode::Xpbd, pos_solver: PosSolver::Projective,
        dt: 0.1, cdamp: 1.0, rot_damp: 1.0, flim: 0.0, xpbd_iters: 4, xpbd_over_relax: 1.0,
        pd_inertia: true, vel_reset: true, bmix_start: 0.0, bmix_end: 0.0, bmix_istart: 1, bmix_iend: 2, ..Default::default() };
    let mut e_last = f64::INFINITY;
    let mut max_f = f64::INFINITY;
    let n = s.natoms;
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut tau = vec![Vec3d::new(0.0,0.0,0.0); n];
    for _ in 0..2000 {
        e_last = step_position_based(&mut s, &topo, &cfg, &nb_off());
        eval_port_forces(&s, &topo, &mut fapos, &mut tau);
        max_f = 0.0;
        for f in &fapos { max_f = max_f.max(f.norm()); }
        if max_f < 1e-3 { break; }
    }
    eprintln!("Dynamic PD with inner rotation (dt=0.1, i4): E={:.6e} max|F|={:.6e}", e_last, max_f);
    assert!(e_last.is_finite() && max_f < 0.1, "Dynamic PD with inner rotation did not converge at dt=0.1: E={:.3e}, max|F|={:.3e}", e_last, max_f);
}

#[test]
fn test_pd_outer_inertia_retains_velocity() {
    let topo = RaffTopology::new(1);
    let mut s = RaffState::new(1);
    s.vel[0] = Vec3d::new(2.0, -1.0, 0.5);
    let cfg = RaffConfig { orient_mode: OrientMode::Adiabatic, dyn_mode: DynMode::Xpbd, pos_solver: PosSolver::Projective,
        dt: 0.1, cdamp: 1.0, rot_damp: 1.0, flim: 0.0, xpbd_iters: 1, xpbd_over_relax: 1.0,
        pd_inertia: true, vel_reset: false, bmix_start: 0.0, bmix_end: 0.0, bmix_istart: 0, bmix_iend: 1, ..Default::default() };
    step_position_based(&mut s, &topo, &cfg, &nb_off());
    step_position_based(&mut s, &topo, &cfg, &nb_off());
    let p_ref = Vec3d::new(0.4, -0.2, 0.1);
    let dp = (s.pos[0] - p_ref).norm();
    let dv = (s.vel[0] - Vec3d::new(2.0, -1.0, 0.5)).norm();
    eprintln!("PD inertia diagnostic: pos=({:.6},{:.6},{:.6}) |pos-ref|={:.3e} |vel-ref|={:.3e}", s.pos[0].x, s.pos[0].y, s.pos[0].z, dp, dv);
    assert!(dp < 1e-12 && dv < 1e-12, "PD outer loop did not preserve unconstrained inertial motion: dp={:.3e}, dv={:.3e}", dp, dv);
}

#[test]
fn test_pd_i4_heavy_ball_is_active() {
    let (state, topo) = make_ch4();
    let mut s0 = perturb(&state);
    let mut sh = s0.clone();
    solve_all_rotations(&mut s0, &topo);
    solve_all_rotations(&mut sh, &topo);
    let base = RaffConfig { orient_mode: OrientMode::Adiabatic, dyn_mode: DynMode::Xpbd, pos_solver: PosSolver::Projective,
        dt: 0.2, cdamp: 0.0, rot_damp: 0.0, flim: 0.0, xpbd_iters: 4, xpbd_over_relax: 1.0,
        pd_inertia: false, vel_reset: false, bmix_start: 0.0, bmix_end: 0.0, bmix_istart: 1, bmix_iend: 2, ..Default::default() };
    let accelerated = RaffConfig { bmix_start: 0.75, bmix_end: 0.75, ..base };
    step_position_based(&mut s0, &topo, &base, &nb_off());
    step_position_based(&mut sh, &topo, &accelerated, &nb_off());
    let mut dx2 = 0.0;
    for i in 0..s0.natoms { dx2 += (sh.pos[i] - s0.pos[i]).norm2(); }
    let dx = dx2.sqrt();
    eprintln!("PD heavy-ball schedule diagnostic: |x_hb-x_plain|={:.6e}", dx);
    assert!(dx > 1e-10, "i4 heavy-ball schedule was inactive: |x_hb-x_plain|={:.3e}", dx);
}
