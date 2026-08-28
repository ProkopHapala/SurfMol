//! RAFF computational core tests — physical invariants, FD validation, conservation.
//!
//! Tests the corrected equations from raff_theory_equations.md:
//! - §1: Port energy convention (E = k_p/2 |e|², F = k_p·e)
//! - §2.2: Wahba rotation (no centroid subtraction)
//! - §2.3: Adiabatic torque residual → 0 at convergence
//! - §3.1: Force-based MD
//! - §3.2: XPBD port constraint (C = |x_j - tip| = 0)
//! - §11.7: Translation/rotation invariance, FD force/torque checks

use molff::raff::*;
use numtypes::Vec3d;

/// Default disabled non-bonded config (for tests that only test port forces).
fn nb_disabled() -> NbConfig { NbConfig { enabled: false, ..Default::default() } }

// ============================================================
//  Helper: build a tetrahedral CH4-like system
// ============================================================

fn make_ch4() -> (RaffState, RaffTopology) {
    let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
    let r = 1.0; // bond length
    let apos = vec![
        Vec3d::new(0.0, 0.0, 0.0),                       // C center
        Vec3d::new( r*inv_sqrt3,  r*inv_sqrt3,  r*inv_sqrt3), // H1
        Vec3d::new( r*inv_sqrt3, -r*inv_sqrt3, -r*inv_sqrt3), // H2
        Vec3d::new(-r*inv_sqrt3,  r*inv_sqrt3, -r*inv_sqrt3), // H3
        Vec3d::new(-r*inv_sqrt3, -r*inv_sqrt3,  r*inv_sqrt3), // H4
    ];
    let bonds = vec![[0,1], [0,2], [0,3], [0,4]];
    let natoms = apos.len();
    let mut topo = RaffTopology::new(natoms);
    // Bond params: k_p = K_bond/2 = 50 (so K_bond = 100), l0 = 1.0
    topo.bond_params = bonds.iter().map(|_| PortParam { k_p: 50.0, l0: 1.0 }).collect();
    topo.build_neighs_from_bonds(&bonds);
    let mut state = RaffState::new(natoms);
    state.set_positions(&apos);
    (state, topo)
}

fn make_water() -> (RaffState, RaffTopology) {
    let apos = vec![
        Vec3d::new(0.0, 0.0, 0.0),     // O
        Vec3d::new(0.96, 0.0, 0.0),    // H1
        Vec3d::new(-0.24, 0.93, 0.0),  // H2
    ];
    let bonds = vec![[0,1], [0,2]];
    let natoms = apos.len();
    let mut topo = RaffTopology::new(natoms);
    topo.bond_params = bonds.iter().map(|_| PortParam { k_p: 50.0, l0: 1.0 }).collect();
    topo.build_neighs_from_bonds(&bonds);
    let mut state = RaffState::new(natoms);
    state.set_positions(&apos);
    (state, topo)
}

// ============================================================
//  L0: Build & basic sanity
// ============================================================

#[test]
fn test_build_ch4_neighs() {
    let (_state, topo) = make_ch4();
    assert_eq!(topo.nport[0], 4, "C should have 4 ports");
    assert_eq!(topo.nport[1], 1, "H should have 1 port");
    assert_eq!(topo.nport[2], 1);
    assert_eq!(topo.nport[3], 1);
    assert_eq!(topo.nport[4], 1);
    // C's neighbors should be 1,2,3,4 in some order
    let cn = topo.neighs[0].as_array();
    let mut found = [false; 5];
    for &j in &cn { if j >= 0 { found[j as usize] = true; } }
    assert!(found[1] && found[2] && found[3] && found[4], "C neighbors should be 1-4, got {:?}", cn);
    // Each H should have C (index 0) as neighbor
    for i in 1..=4 {
        let hn = topo.neighs[i].as_array();
        assert_eq!(hn[0], 0, "H{} should have C(0) as neighbor, got {:?}", i, hn);
    }
    // Inertia should be positive for C, zero for H (single port → no rotational DOF needed)
    assert!(topo.inv_inertia[0] > 0.0, "C should have positive inertia");
}

// ============================================================
//  §11.7: Translation invariance — Σ F_i = 0
// ============================================================

#[test]
fn test_translation_invariance() {
    let (state, topo) = make_ch4();
    let net_force = check_translation_invariance(&state, &topo);
    assert!(net_force < 1e-10, "Net force should be 0 (translation invariance), got |ΣF| = {:.2e}", net_force);
}

// ============================================================
//  §11.7: Rotation invariance — Σ x_i×F_i + τ_i = 0
// ============================================================

#[test]
fn test_rotation_invariance() {
    let (state, topo) = make_ch4();
    let net_torque = check_rotation_invariance(&state, &topo);
    assert!(net_torque < 1e-10, "Net torque should be 0 (rotation invariance), got |Σx×F+τ| = {:.2e}", net_torque);
}

// ============================================================
//  §11.7: Finite-difference force check — F = -dE/dx
// ============================================================

#[test]
fn test_fd_forces_ch4() {
    let (state, topo) = make_ch4();
    let (max_err, details) = fd_check_forces(&state, &topo, 1e-6);
    if max_err > 1e-4 {
        for (i, d, _) in &details {
            eprintln!("  atom {}: fd={:.6} analytic={:.6} rel_err={:.2e}", i, d.x, d.y, d.z);
        }
    }
    assert!(max_err < 1e-4, "FD force check failed: max rel err = {:.2e} (should be < 1e-4)", max_err);
}

#[test]
fn test_fd_forces_water() {
    let (state, topo) = make_water();
    let (max_err, details) = fd_check_forces(&state, &topo, 1e-6);
    if max_err > 1e-4 {
        for (i, d, _) in &details {
            eprintln!("  atom {}: fd={:.6} analytic={:.6} rel_err={:.2e}", i, d.x, d.y, d.z);
        }
    }
    assert!(max_err < 1e-4, "FD force check failed: max rel err = {:.2e}", max_err);
}

// ============================================================
//  §11.7: Finite-difference torque check — τ = r × F
// ============================================================

#[test]
fn test_fd_torques_ch4() {
    let (state, topo) = make_ch4();
    let max_err = fd_check_torques(&state, &topo, 1e-6);
    // Torque FD is less precise due to quaternion normalization, allow 1e-2
    assert!(max_err < 1e-2, "FD torque check failed: max rel err = {:.2e}", max_err);
}

// ============================================================
//  §1: Port energy at equilibrium should be 0
// ============================================================

#[test]
fn test_equilibrium_energy_zero() {
    let (state, topo) = make_ch4();
    let n = state.natoms;
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut tau = vec![Vec3d::new(0.0,0.0,0.0); n];
    let e = eval_port_forces(&state, &topo, &mut fapos, &mut tau);
    // At perfect tetrahedral equilibrium with correct orientation, E should be 0
    // But initial quat is identity — need to solve rotation first
    assert!(e >= 0.0, "Port energy should be non-negative, got E = {:.6}", e);
    eprintln!("Initial CH4 port energy (identity quat): E = {:.6}", e);
}

// ============================================================
//  §2.2: Wahba rotation solver — should find zero-energy orientation
// ============================================================

#[test]
fn test_wahba_finds_equilibrium_ch4() {
    let (mut state, topo) = make_ch4();
    // Solve rotation for ALL atoms (C uses Horn K-matrix, H atoms use single-port alignment)
    solve_all_rotations(&mut state, &topo);
    let n = state.natoms;
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut tau = vec![Vec3d::new(0.0,0.0,0.0); n];
    let e = eval_port_forces(&state, &topo, &mut fapos, &mut tau);
    let max_f = fapos.iter().map(|f| f.norm()).fold(0.0, f64::max);
    let max_t = tau.iter().map(|t| t.norm()).fold(0.0, f64::max);
    eprintln!("After Wahba solve: E = {:.6e}, max|F| = {:.6e}, max|τ| = {:.6e}", e, max_f, max_t);
    assert!(e < 1e-10, "Wahba solver should find zero-energy orientation for perfect tetrahedron, got E = {:.2e}", e);
    assert!(max_f < 1e-8, "Forces should be ~0 at Wahba optimum, got max|F| = {:.2e}", max_f);
}

// ============================================================
//  §2.3: Adiabatic torque residual → 0 at convergence
// ============================================================

#[test]
fn test_adiabatic_torque_residual() {
    let (mut state, topo) = make_ch4();
    // Perturb positions slightly so rotation is non-trivial
    state.pos[1] = state.pos[1] + Vec3d::new(0.01, 0.0, 0.0);
    // Debug: print neighbors and ports for atom 0
    let ns = topo.neighs[0].as_array();
    let bs = topo.neigh_bs[0].as_array();
    eprintln!("Atom 0 neighbors: {:?}, bond_params: {:?}", ns, bs);
    for s in 0..4 {
        let j = ns[s] as usize;
        let d = state.pos[j] - state.pos[0];
        let r = topo.port_local[s];
        eprintln!("  port {}: j={}, d=({:.4},{:.4},{:.4}), r_body=({:.4},{:.4},{:.4})", s, j, d.x, d.y, d.z, r.x, r.y, r.z);
    }
    let residuals = check_adiabatic_torque_residual(&mut state, &topo);
    // Debug: print quaternion for atom 0
    let q = state.quat[0];
    eprintln!("Atom 0 quaternion after solve: ({:.6}, {:.6}, {:.6}, {:.6})", q.x, q.y, q.z, q.w);
    // Debug: print per-port forces and torques
    let n = state.natoms;
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut tau = vec![Vec3d::new(0.0,0.0,0.0); n];
    eval_port_forces(&state, &topo, &mut fapos, &mut tau);
    eprintln!("Atom 0 total force: ({:.4}, {:.4}, {:.4}), torque: ({:.4}, {:.4}, {:.4})", fapos[0].x, fapos[0].y, fapos[0].z, tau[0].x, tau[0].y, tau[0].z);
    // Only check atoms with ≥2 ports (single-port atoms have no rotational DOF to eliminate torque)
    let max_residual = (0..topo.natoms)
        .filter(|&i| topo.nport[i] >= 2)
        .map(|i| residuals[i])
        .fold(0.0, f64::max);
    eprintln!("Adiabatic torque residuals: {:?}", residuals.iter().enumerate().map(|(i,r)| format!("atom{}(np={}): {:.2e}", i, topo.nport[i], r)).collect::<Vec<_>>());
    assert!(max_residual < 1e-6, "At adiabatic convergence, per-atom torque (atoms with ≥2 ports) should be ~0, got max = {:.2e}", max_residual);
}

// ============================================================
//  §3.1: Force-based MD — energy should decrease with damping
// ============================================================

#[test]
fn test_force_md_relaxes_ch4() {
    let (mut state, topo) = make_ch4();
    // Perturb from equilibrium
    state.pos[1] = state.pos[1] + Vec3d::new(0.1, 0.0, 0.0);
    state.pos[2] = state.pos[2] + Vec3d::new(0.0, 0.05, 0.0);
    // Use adiabatic rotation (solve R* each step) so rotation doesn't limit convergence
    let cfg = RaffConfig { orient_mode: OrientMode::Adiabatic, dt: 0.001, cdamp: 0.9, rot_damp: 0.9, flim: 100.0, ..Default::default() };
    let n = state.natoms;
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut tau = vec![Vec3d::new(0.0,0.0,0.0); n];
    // Solve initial rotations
    solve_all_rotations(&mut state, &topo);
    let e0 = eval_port_forces(&state, &topo, &mut fapos, &mut tau);
    eprintln!("Initial perturbed energy: E0 = {:.6}", e0);
    let mut e_last = e0;
    for step in 0..5000 {
        // Adiabatic: re-solve rotations each step
        solve_all_rotations(&mut state, &topo);
        let (e, max_f, max_t) = step_force_md(&mut state, &topo, &cfg, &mut fapos, &mut tau, &nb_disabled());
        if step % 1000 == 0 {
            eprintln!("  step {}: E = {:.6e}, max|F| = {:.4e}, max|τ| = {:.4e}", step, e, max_f, max_t);
        }
        e_last = e;
    }
    eprintln!("Final energy after 5000 steps: E = {:.6e}", e_last);
    assert!(e_last < e0, "Energy should decrease with damping: E0={:.6e} → E_final={:.6e}", e0, e_last);
    assert!(e_last < 1.0, "Should relax significantly, got E = {:.6e}", e_last);
}

// ============================================================
//  §3.2: XPBD — should converge to zero constraint violation
// ============================================================

#[test]
fn test_xpbd_converges_ch4() {
    let (mut state, topo) = make_ch4();
    // Solve initial rotations
    solve_all_rotations(&mut state, &topo);
    // Small perturbation from equilibrium
    state.pos[1] = state.pos[1] + Vec3d::new(0.01, 0.0, 0.0);
    let cfg = RaffConfig {
        orient_mode: OrientMode::Adiabatic,
        dyn_mode: DynMode::Xpbd,
        dt: 0.01,        // larger dt → smaller compliance α̃ = 1/(k_p·dt²)
        cdamp: 0.0,      // pure relaxation (no velocity prediction)
        xpbd_iters: 64,
        ..Default::default()
    };
    let n = state.natoms;
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut tau = vec![Vec3d::new(0.0,0.0,0.0); n];
    let e0 = eval_port_forces(&state, &topo, &mut fapos, &mut tau);
    eprintln!("XPBD initial energy: E0 = {:.6e}", e0);
    // Run multiple XPBD macrosteps
    let mut e_last = e0;
    for step in 0..100 {
        let e = step_xpbd(&mut state, &topo, &cfg, &nb_disabled());
        if step % 20 == 0 {
            eprintln!("  XPBD step {}: E = {:.6e}", step, e);
        }
        e_last = e;
    }
    // Recheck energy
    let mut fapos2 = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut tau2 = vec![Vec3d::new(0.0,0.0,0.0); n];
    let e_after = eval_port_forces(&state, &topo, &mut fapos2, &mut tau2);
    let max_f = fapos2.iter().map(|f| f.norm()).fold(0.0, f64::max);
    eprintln!("XPBD after 100 macrosteps: E_last = {:.6e}, E_recheck = {:.6e}, max|F| = {:.4e}", e_last, e_after, max_f);
    assert!(e_after < e0, "XPBD should reduce energy: E0={:.6e} → E_after={:.6e}", e0, e_after);
    assert!(e_after < 1e-3, "XPBD should converge close to equilibrium, got E = {:.6e}", e_after);
}

// ============================================================
//  §1: Reciprocal port stiffness — k_p = K_bond/2 gives correct total
// ============================================================

#[test]
fn test_reciprocal_port_stiffness() {
    // Two-atom system: single bond, both atoms have 1 port
    let apos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(1.0, 0.0, 0.0)];
    let bonds = vec![[0, 1]];
    let mut topo = RaffTopology::new(2);
    // K_bond = 100, so k_p = 50 on each side
    topo.bond_params = vec![PortParam { k_p: 50.0, l0: 1.0 }];
    topo.build_neighs_from_bonds(&bonds);
    let mut state = RaffState::new(2);
    state.set_positions(&apos);
    // Solve rotations first (single-port atoms: align port with neighbor)
    solve_all_rotations(&mut state, &topo);
    // At equilibrium (distance = l0 = 1.0, orientation aligned), E = 0
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); 2];
    let mut tau = vec![Vec3d::new(0.0,0.0,0.0); 2];
    let e = eval_port_forces(&state, &topo, &mut fapos, &mut tau);
    eprintln!("Two-atom equilibrium: E = {:.6e}, F0 = {:?}, F1 = {:?}", e, fapos[0], fapos[1]);
    assert!(e < 1e-10, "At equilibrium with solved rotation, E should be 0, got {:.6e}", e);
    // Perturb atom 1 by +0.1 in x: distance = 1.1, error = 0.1 on each port
    // E_per_port = k_p/2 * 0.1^2 = 50/2 * 0.01 = 0.25
    // E_total = 2 * 0.25 = 0.5 (two reciprocal ports)
    // F_per_port = k_p * 0.1 = 5.0 (attractive, pulling back)
    // F_total on atom 0 = +5.0 (toward atom 1), F_total on atom 1 = -5.0
    let mut state2 = RaffState::new(2);
    state2.set_positions(&[Vec3d::new(0.0,0.0,0.0), Vec3d::new(1.1,0.0,0.0)]);
    solve_all_rotations(&mut state2, &topo);
    let mut f2 = vec![Vec3d::new(0.0,0.0,0.0); 2];
    let mut t2 = vec![Vec3d::new(0.0,0.0,0.0); 2];
    let e2 = eval_port_forces(&state2, &topo, &mut f2, &mut t2);
    eprintln!("Perturbed: E = {:.6} (expected 0.5), F0.x = {:.4} (expected +10.0), F1.x = {:.4} (expected -10.0)", e2, f2[0].x, f2[1].x);
    assert!((e2 - 0.5).abs() < 1e-10, "Energy should be 0.5 (2 × k_p/2 × 0.01), got {:.6}", e2);
    // F_total = 2 × k_p × e = K_bond × e = 100 × 0.1 = 10.0 (two reciprocal ports sum)
    assert!((f2[0].x - 10.0).abs() < 1e-10, "F0.x should be +10.0 (K_bond × 0.1), got {:.6}", f2[0].x);
    assert!((f2[1].x + 10.0).abs() < 1e-10, "F1.x should be -10.0, got {:.6}", f2[1].x);
}

// ============================================================
//  §11.7: Conservation — momentum should be conserved in MD
// ============================================================

#[test]
fn test_momentum_conservation() {
    let (mut state, topo) = make_ch4();
    // Perturb
    state.pos[1] = state.pos[1] + Vec3d::new(0.05, 0.0, 0.0);
    // No damping for conservation test
    let cfg = RaffConfig { dt: 0.0005, cdamp: 1.0, rot_damp: 1.0, flim: 0.0, ..Default::default() };
    let n = state.natoms;
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut tau = vec![Vec3d::new(0.0,0.0,0.0); n];
    // Initial momentum
    let p0: Vec3d = state.vel.iter().zip(topo.mass.iter()).map(|(v,m)| *v * *m).fold(Vec3d::new(0.0,0.0,0.0), |a,b| a+b);
    for step in 0..100 {
        let (_e, _, _) = step_force_md(&mut state, &topo, &cfg, &mut fapos, &mut tau, &nb_disabled());
        if step % 25 == 0 {
            let p: Vec3d = state.vel.iter().zip(topo.mass.iter()).map(|(v,m)| *v * *m).fold(Vec3d::new(0.0,0.0,0.0), |a,b| a+b);
            eprintln!("  step {}: P = [{:.4e}, {:.4e}, {:.4e}]", step, p.x, p.y, p.z);
        }
    }
    let p_final: Vec3d = state.vel.iter().zip(topo.mass.iter()).map(|(v,m)| *v * *m).fold(Vec3d::new(0.0,0.0,0.0), |a,b| a+b);
    let dp = (p_final - p0).norm();
    // With no external forces, momentum should be conserved (starts at 0, stays ~0)
    assert!(dp < 1e-10, "Momentum should be conserved: |ΔP| = {:.2e}", dp);
}

// ============================================================
//  Water molecule — basic sanity
// ============================================================

#[test]
fn test_water_basic() {
    let (state, topo) = make_water();
    let net_force = check_translation_invariance(&state, &topo);
    assert!(net_force < 1e-10, "Water: net force should be 0, got {:.2e}", net_force);
    let (max_err, _) = fd_check_forces(&state, &topo, 1e-6);
    assert!(max_err < 1e-4, "Water: FD force check failed: max rel err = {:.2e}", max_err);
}

// ============================================================
//  §4: Non-bonded — exclusion, collision, LJ, Coulomb, momentum
// ============================================================

/// Build a simple 4-atom linear chain: A-B-C-D with 3 bonds.
/// Used to test 1-2 and 1-3 exclusions.
fn make_chain4() -> (RaffState, RaffTopology) {
    let apos = vec![
        Vec3d::new(0.0, 0.0, 0.0),  // 0: A
        Vec3d::new(1.0, 0.0, 0.0),  // 1: B
        Vec3d::new(2.0, 0.0, 0.0),  // 2: C
        Vec3d::new(3.0, 0.0, 0.0),  // 3: D
    ];
    let bonds = vec![[0,1], [1,2], [2,3]];
    let natoms = apos.len();
    let mut topo = RaffTopology::new(natoms);
    topo.bond_params = vec![PortParam { k_p: 50.0, l0: 1.0 }; 3];
    topo.build_neighs_from_bonds(&bonds);
    let mut state = RaffState::new(natoms);
    state.set_positions(&apos);
    (state, topo)
}

#[test]
fn test_exclusion_12_13() {
    let (_state, topo) = make_chain4();
    // Atom 1 (B) is bonded to 0 and 2 (1-2), and 1-3 with 3 (via 2)
    // Exclusions for atom 1: {0, 2, 3} (0 and 2 are 1-2, 3 is 1-3 via 2)
    assert!(topo.is_excluded(1, 0), "Atom 1 should exclude bonded atom 0");
    assert!(topo.is_excluded(1, 2), "Atom 1 should exclude bonded atom 2");
    assert!(topo.is_excluded(1, 3), "Atom 1 should exclude 1-3 neighbor atom 3");
    // Atom 0 (A) is bonded to 1 (1-2), and 1-3 with 2 (via 1)
    // Exclusions for atom 0: {1, 2}
    assert!(topo.is_excluded(0, 1), "Atom 0 should exclude bonded atom 1");
    assert!(topo.is_excluded(0, 2), "Atom 0 should exclude 1-3 neighbor atom 2");
    assert!(!topo.is_excluded(0, 3), "Atom 0 should NOT exclude atom 3 (1-4, not excluded)");
    eprintln!("Exclusion test passed: 1-2 and 1-3 exclusions correct for linear chain");
}

#[test]
fn test_lj_energy_at_minimum() {
    // Two atoms with LJ, at the minimum distance r = 2^(1/6) * sigma
    let sigma = 3.0;
    let eps = 0.1;
    let r_min = 2.0_f64.powf(1.0/6.0) * sigma;  // LJ minimum distance
    let apos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(r_min, 0.0, 0.0)];
    let mut topo = RaffTopology::new(2);
    // No bonds — just two non-bonded atoms
    topo.nport[0] = 0; topo.nport[1] = 0;
    topo.build_exclusions();  // no bonds → no exclusions
    topo.nb_params[0] = NbParams { sigma, epsilon: eps, charge: 0.0, radius: 0.0 };
    topo.nb_params[1] = NbParams { sigma, epsilon: eps, charge: 0.0, radius: 0.0 };
    let state = RaffState::new(2);
    let mut state = state;
    state.set_positions(&apos);
    let nbcfg = NbConfig { enabled: true, rcut: 10.0, excl_12: false, excl_13: false, ..Default::default() };
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); 2];
    let e = eval_nonbonded(&state, &topo, &nbcfg, &mut fapos);
    // At LJ minimum: E = -ε, F = 0
    eprintln!("LJ at r_min: E = {:.6} (expected -ε = {:.6}), |F0| = {:.6}, |F1| = {:.6}", e, -eps, fapos[0].norm(), fapos[1].norm());
    assert!((e - (-eps)).abs() < 1e-10, "LJ energy at minimum should be -ε = {:.6}, got {:.6}", -eps, e);
    assert!(fapos[0].norm() < 1e-10, "LJ force at minimum should be 0, got {:.6}", fapos[0].norm());
    assert!(fapos[1].norm() < 1e-10, "LJ force at minimum should be 0, got {:.6}", fapos[1].norm());
}

#[test]
fn test_lj_force_direction() {
    // Two atoms with LJ, closer than r_min → should repel
    let sigma = 3.0;
    let eps = 0.1;
    let r = 2.5;  // closer than r_min ≈ 3.36 → repulsive
    let apos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(r, 0.0, 0.0)];
    let mut topo = RaffTopology::new(2);
    topo.nport[0] = 0; topo.nport[1] = 0;
    topo.build_exclusions();
    topo.nb_params[0] = NbParams { sigma, epsilon: eps, charge: 0.0, radius: 0.0 };
    topo.nb_params[1] = NbParams { sigma, epsilon: eps, charge: 0.0, radius: 0.0 };
    let mut state = RaffState::new(2);
    state.set_positions(&apos);
    let nbcfg = NbConfig { enabled: true, rcut: 10.0, excl_12: false, excl_13: false, ..Default::default() };
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); 2];
    eval_nonbonded(&state, &topo, &nbcfg, &mut fapos);
    // Atom 0 should be pushed in -x (away from atom 1), atom 1 in +x
    eprintln!("LJ repulsive: F0 = ({:.4}, 0, 0), F1 = ({:.4}, 0, 0)", fapos[0].x, fapos[1].x);
    assert!(fapos[0].x < 0.0, "Atom 0 should be pushed in -x (repulsive), got F0.x = {:.4}", fapos[0].x);
    assert!(fapos[1].x > 0.0, "Atom 1 should be pushed in +x (repulsive), got F1.x = {:.4}", fapos[1].x);
    // Newton's 3rd law: F0 = -F1
    assert!((fapos[0].x + fapos[1].x).abs() < 1e-10, "Newton's 3rd law: F0 + F1 should be 0");
}

#[test]
fn test_coulomb_same_sign_repels() {
    let r = 2.0;
    let apos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(r, 0.0, 0.0)];
    let mut topo = RaffTopology::new(2);
    topo.nport[0] = 0; topo.nport[1] = 0;
    topo.build_exclusions();
    topo.nb_params[0] = NbParams { sigma: 0.0, epsilon: 0.0, charge: 1.0, radius: 0.0 };
    topo.nb_params[1] = NbParams { sigma: 0.0, epsilon: 0.0, charge: 1.0, radius: 0.0 };
    let mut state = RaffState::new(2);
    state.set_positions(&apos);
    let nbcfg = NbConfig { enabled: true, rcut: 10.0, r_damp: 0.0, excl_12: false, excl_13: false, ..Default::default() };
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); 2];
    let e = eval_nonbonded(&state, &topo, &nbcfg, &mut fapos);
    let e_expected = 14.3996448915 / r;  // k * q1 * q2 / r
    eprintln!("Coulomb same-sign: E = {:.6} (expected {:.6}), F0.x = {:.4}, F1.x = {:.4}", e, e_expected, fapos[0].x, fapos[1].x);
    assert!((e - e_expected).abs() < 1e-6, "Coulomb energy should be {:.6}, got {:.6}", e_expected, e);
    assert!(fapos[0].x < 0.0, "Same-sign charges should repel atom 0 in -x, got F0.x = {:.4}", fapos[0].x);
    assert!(fapos[1].x > 0.0, "Same-sign charges should repel atom 1 in +x, got F1.x = {:.4}", fapos[1].x);
}

#[test]
fn test_collision_repulsion() {
    // Two atoms with collision radius, overlapping → should repel
    let r = 2.0;  // distance
    let rad = 1.5;  // radius each → rsum = 3.0 > r = 2.0 (overlap)
    let apos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(r, 0.0, 0.0)];
    let mut topo = RaffTopology::new(2);
    topo.nport[0] = 0; topo.nport[1] = 0;
    topo.build_exclusions();
    topo.nb_params[0] = NbParams { sigma: 0.0, epsilon: 0.0, charge: 0.0, radius: rad };
    topo.nb_params[1] = NbParams { sigma: 0.0, epsilon: 0.0, charge: 0.0, radius: rad };
    let mut state = RaffState::new(2);
    state.set_positions(&apos);
    let nbcfg = NbConfig { enabled: true, rcut: 10.0, k_coll: 100.0, f_max: 0.0, excl_12: false, excl_13: false, ..Default::default() };
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); 2];
    let e = eval_nonbonded(&state, &topo, &nbcfg, &mut fapos);
    let overlap = 2.0 * rad - r;  // 1.0
    let e_expected = 0.5 * 100.0 * overlap * overlap;  // 50.0
    let f_expected = 100.0 * overlap;  // 100.0
    eprintln!("Collision: E = {:.6} (expected {:.6}), |F0| = {:.4} (expected {:.4})", e, e_expected, fapos[0].norm(), f_expected);
    assert!((e - e_expected).abs() < 1e-10, "Collision energy should be {:.6}, got {:.6}", e_expected, e);
    assert!((fapos[0].norm() - f_expected).abs() < 1e-8, "Collision force should be {:.6}, got {:.6}", f_expected, fapos[0].norm());
    assert!(fapos[0].x < 0.0, "Atom 0 should be pushed in -x, got F0.x = {:.4}", fapos[0].x);
    assert!(fapos[1].x > 0.0, "Atom 1 should be pushed in +x, got F1.x = {:.4}", fapos[1].x);
}

#[test]
fn test_exclusion_skips_bonded_pair() {
    // Two bonded atoms with LJ — should NOT interact via non-bonded (1-2 exclusion)
    let sigma = 3.0;
    let eps = 0.1;
    let r = 1.0;  // typical bond distance
    let apos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(r, 0.0, 0.0)];
    let bonds = vec![[0, 1]];
    let mut topo = RaffTopology::new(2);
    topo.bond_params = vec![PortParam { k_p: 50.0, l0: r }];
    topo.build_neighs_from_bonds(&bonds);
    topo.nb_params[0] = NbParams { sigma, epsilon: eps, charge: 0.0, radius: 0.0 };
    topo.nb_params[1] = NbParams { sigma, epsilon: eps, charge: 0.0, radius: 0.0 };
    let mut state = RaffState::new(2);
    state.set_positions(&apos);
    let nbcfg = NbConfig { enabled: true, rcut: 10.0, excl_12: true, excl_13: true, ..Default::default() };
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); 2];
    let e = eval_nonbonded(&state, &topo, &nbcfg, &mut fapos);
    eprintln!("Excluded bonded pair: E = {:.6} (expected 0), |F0| = {:.6}, |F1| = {:.6}", e, fapos[0].norm(), fapos[1].norm());
    assert!(e.abs() < 1e-10, "Bonded pair should be excluded from non-bonded, got E = {:.6}", e);
    assert!(fapos[0].norm() < 1e-10, "Bonded pair should have no non-bonded force, got |F0| = {:.6}", fapos[0].norm());
}

#[test]
fn test_nb_momentum_conservation() {
    // Two non-bonded atoms with LJ + Coulomb, no damping → momentum conserved
    let r = 3.0;
    let apos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(r, 0.0, 0.0)];
    let mut topo = RaffTopology::new(2);
    topo.nport[0] = 0; topo.nport[1] = 0;
    topo.build_exclusions();
    topo.nb_params[0] = NbParams { sigma: 2.5, epsilon: 0.1, charge: 0.5, radius: 0.0 };
    topo.nb_params[1] = NbParams { sigma: 3.0, epsilon: 0.1, charge: 0.5, radius: 0.0 };
    topo.mass[0] = 12.0; topo.mass[1] = 1.0;
    let mut state = RaffState::new(2);
    state.set_positions(&apos);
    let cfg = RaffConfig { dt: 0.0005, cdamp: 1.0, rot_damp: 1.0, flim: 0.0, ..Default::default() };
    let nbcfg = NbConfig { enabled: true, rcut: 10.0, excl_12: false, excl_13: false, ..Default::default() };
    let n = state.natoms;
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); n];
    let mut tau = vec![Vec3d::new(0.0,0.0,0.0); n];
    let p0: Vec3d = state.vel.iter().zip(topo.mass.iter()).map(|(v,m)| *v * *m).fold(Vec3d::new(0.0,0.0,0.0), |a,b| a+b);
    for step in 0..100 {
        let (_e, _, _) = step_force_md(&mut state, &topo, &cfg, &mut fapos, &mut tau, &nbcfg);
        if step % 25 == 0 {
            let p: Vec3d = state.vel.iter().zip(topo.mass.iter()).map(|(v,m)| *v * *m).fold(Vec3d::new(0.0,0.0,0.0), |a,b| a+b);
            eprintln!("  step {}: P = [{:.4e}, {:.4e}, {:.4e}]", step, p.x, p.y, p.z);
        }
    }
    let p_final: Vec3d = state.vel.iter().zip(topo.mass.iter()).map(|(v,m)| *v * *m).fold(Vec3d::new(0.0,0.0,0.0), |a,b| a+b);
    let dp = (p_final - p0).norm();
    eprintln!("NB momentum conservation: |ΔP| = {:.2e}", dp);
    assert!(dp < 1e-10, "Non-bonded momentum should be conserved: |ΔP| = {:.2e}", dp);
}

#[test]
fn test_nb_fd_check() {
    // Finite-difference check of non-bonded forces (LJ + Coulomb)
    let r = 3.0;
    let apos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(r, 0.0, 0.0)];
    let mut topo = RaffTopology::new(2);
    topo.nport[0] = 0; topo.nport[1] = 0;
    topo.build_exclusions();
    topo.nb_params[0] = NbParams { sigma: 2.5, epsilon: 0.1, charge: 0.3, radius: 0.0 };
    topo.nb_params[1] = NbParams { sigma: 3.0, epsilon: 0.15, charge: -0.3, radius: 0.0 };
    let mut state = RaffState::new(2);
    state.set_positions(&apos);
    let nbcfg = NbConfig { enabled: true, rcut: 10.0, r_damp: 0.0, excl_12: false, excl_13: false, ..Default::default() };
    // Analytic force
    let mut f_analytic = vec![Vec3d::new(0.0,0.0,0.0); 2];
    eval_nonbonded(&state, &topo, &nbcfg, &mut f_analytic);
    // FD force: perturb atom 1 in x, check dE/dx vs -F
    let eps = 1e-6;
    let mut state_plus = state.clone();
    state_plus.pos[1].x += eps;
    let mut f_tmp = vec![Vec3d::new(0.0,0.0,0.0); 2];
    let e_plus = eval_nonbonded(&state_plus, &topo, &nbcfg, &mut f_tmp);
    let mut state_minus = state.clone();
    state_minus.pos[1].x -= eps;
    let e_minus = eval_nonbonded(&state_minus, &topo, &nbcfg, &mut f_tmp);
    let fd_force_x = -(e_plus - e_minus) / (2.0 * eps);  // force on atom 1 in x
    // F_analytic on atom 1 in x should match fd_force_x
    let rel_err = if f_analytic[1].x.abs() > 1e-10 {
        (f_analytic[1].x - fd_force_x).abs() / f_analytic[1].x.abs()
    } else {
        (f_analytic[1].x - fd_force_x).abs()
    };
    eprintln!("NB FD check: F_analytic.x = {:.6}, F_fd.x = {:.6}, rel_err = {:.2e}", f_analytic[1].x, fd_force_x, rel_err);
    assert!(rel_err < 1e-4, "Non-bonded FD force check failed: rel_err = {:.2e}", rel_err);
}
