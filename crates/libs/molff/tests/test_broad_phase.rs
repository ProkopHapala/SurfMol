//! Broad-phase AABB collision parity tests.
//!
//! Verifies that `NonBondedFF::eval_broad` (AABB-culled) produces **identical**
//! forces and energy as `NonBondedFF::eval` (O(N²) all-pairs) on multi-molecule systems.
//! Also tests `molff::raff::eval_nonbonded_broad` vs `eval_nonbonded`.
//!
//! Per AGENTS.md "Tests Are Diagnostics": this is a parity check, not a smoke test.
//! Any difference indicates a bug in the broad-phase culling logic.

use molff::nonbonded::{NonBondedFF, BroadPhase};
use molff::raff::*;
use numtypes::Vec3d;

/// Build a 2-molecule system: two benzoic-acid-like clusters placed near each other.
/// Returns (positions, cluster_ranges, reqs).
fn make_two_molecules() -> (Vec<Vec3d>, Vec<[u32; 2]>, Vec<[f64; 4]>) {
    // Simplified benzoic acid: 15 atoms (6C ring + COOH + 3H ring + 2H COOH + 1H OH)
    // Just use a compact set of atoms with realistic VdW radii
    let mol_atoms = vec![
        ("C", Vec3d::new(0.0, 0.0, 0.0)),
        ("C", Vec3d::new(1.4, 0.0, 0.0)),
        ("C", Vec3d::new(2.1, 1.2, 0.0)),
        ("C", Vec3d::new(1.4, 2.4, 0.0)),
        ("C", Vec3d::new(0.0, 2.4, 0.0)),
        ("C", Vec3d::new(-0.7, 1.2, 0.0)),
        ("C", Vec3d::new(2.8, 3.6, 0.0)),  // COOH carbon
        ("O", Vec3d::new(2.2, 4.7, 0.0)),  // =O
        ("O", Vec3d::new(4.1, 3.6, 0.0)),  // -O-H
        ("H", Vec3d::new(-1.7, 1.2, 0.0)),
        ("H", Vec3d::new(2.1, -1.0, 0.0)),
        ("H", Vec3d::new(3.5, 0.0, 0.0)),  // not realistic but ok for test
        ("H", Vec3d::new(4.8, 4.4, 0.0)),
        ("H", Vec3d::new(1.7, 5.5, 0.0)),
        ("H", Vec3d::new(-0.7, 3.4, 0.0)),
    ];
    let natoms_per_mol = mol_atoms.len();
    // Place two molecules: molecule 0 at origin, molecule 1 shifted by 8 Å in x (within cutoff)
    let shift = Vec3d::new(8.0, 0.0, 0.0);
    let mut apos = Vec::new();
    let mut reqs = Vec::new();
    for (el, p) in &mol_atoms {
        apos.push(*p);
        let req = match *el {
            "C" => [1.7, 0.1, 0.0, 0.0],
            "O" => [1.52, 0.07, 0.0, 0.0],
            "H" => [1.2, 0.05, 0.0, 0.0],
            _ => [1.5, 0.1, 0.0, 0.0],
        };
        reqs.push(req);
    }
    for (el, p) in &mol_atoms {
        apos.push(Vec3d::set_add(*p, shift));
        let req = match *el {
            "C" => [1.7, 0.1, 0.0, 0.0],
            "O" => [1.52, 0.07, 0.0, 0.0],
            "H" => [1.2, 0.05, 0.0, 0.0],
            _ => [1.5, 0.1, 0.0, 0.0],
        };
        reqs.push(req);
    }
    let cluster_ranges = vec![
        [0, natoms_per_mol as u32],
        [natoms_per_mol as u32, (2 * natoms_per_mol) as u32],
    ];
    (apos, cluster_ranges, reqs)
}

/// Build a NonBondedFF from reqs, with no exclusions (inter-molecular test).
fn make_nbff(natoms: usize, reqs: &[[f64; 4]]) -> NonBondedFF {
    let mut nb = NonBondedFF::new(natoms);
    for i in 0..natoms {
        nb.reqs.as_mut_slice()[i] = reqs[i];
    }
    nb.set_cutoff(8.0);
    nb
}

#[test]
fn test_broad_phase_parity_nonbonded() {
    let (apos, cluster_ranges, reqs) = make_two_molecules();
    let natoms = apos.len();
    let mut nb = make_nbff(natoms, &reqs);

    // O(N²) all-pairs
    let mut fapos_on2 = vec![numtypes::VEC3D_ZERO; natoms];
    let e_on2 = nb.eval(&mut fapos_on2, &apos);

    // Broad-phase AABB-culled
    let mut bp = BroadPhase::new(cluster_ranges, 8.0);
    bp.rebuild(&apos);
    let mut fapos_bp = vec![numtypes::VEC3D_ZERO; natoms];
    let e_bp = nb.eval_broad(&mut fapos_bp, &apos, &bp);

    // Parity check: forces and energy must be identical
    let e_diff = (e_on2 - e_bp).abs();
    assert!(e_diff < 1e-10, "Energy mismatch: O(N²)={}, broad={}, diff={}", e_on2, e_bp, e_diff);
    let mut max_f_diff = 0.0;
    for i in 0..natoms {
        let df = Vec3d::set_sub(fapos_on2[i], fapos_bp[i]);
        let d = df.norm2().sqrt();
        if d > max_f_diff { max_f_diff = d; }
    }
    assert!(max_f_diff < 1e-10, "Force mismatch: max|ΔF|={}", max_f_diff);
    println!("[parity] O(N²) E={:.6}, broad E={:.6}, max|ΔF|={:.2e}, BP pairs={}", e_on2, e_bp, max_f_diff, bp.pairs().len());
}

#[test]
fn test_broad_phase_parity_far_molecules() {
    // Two molecules far apart (no overlap) — broad phase should skip all inter-mol pairs
    let (mut apos, cluster_ranges, reqs) = make_two_molecules();
    let natoms = apos.len();
    // Shift molecule 1 far away (100 Å)
    for i in 15..natoms {
        apos[i].x += 100.0;
    }
    let mut nb = make_nbff(natoms, &reqs);

    let mut fapos_on2 = vec![numtypes::VEC3D_ZERO; natoms];
    let e_on2 = nb.eval(&mut fapos_on2, &apos);

    let mut bp = BroadPhase::new(cluster_ranges, 8.0);
    bp.rebuild(&apos);
    let mut fapos_bp = vec![numtypes::VEC3D_ZERO; natoms];
    let e_bp = nb.eval_broad(&mut fapos_bp, &apos, &bp);

    let e_diff = (e_on2 - e_bp).abs();
    assert!(e_diff < 1e-10, "Energy mismatch (far): O(N²)={}, broad={}, diff={}", e_on2, e_bp, e_diff);
    let mut max_f_diff = 0.0;
    for i in 0..natoms {
        let df = Vec3d::set_sub(fapos_on2[i], fapos_bp[i]);
        let d = df.norm2().sqrt();
        if d > max_f_diff { max_f_diff = d; }
    }
    assert!(max_f_diff < 1e-10, "Force mismatch (far): max|ΔF|={}", max_f_diff);
    // When far apart, broad phase should find 0 overlapping pairs
    assert!(bp.pairs().is_empty(), "Far molecules should have 0 BP pairs, got {}", bp.pairs().len());
    println!("[parity-far] O(N²) E={:.6}, broad E={:.6}, max|ΔF|={:.2e}, BP pairs={}", e_on2, e_bp, max_f_diff, bp.pairs().len());
}

#[test]
fn test_broad_phase_parity_raff() {
    // Test RAFF nonbonded broad-phase parity: eval_nonbonded_broad vs eval_nonbonded
    let (apos, cluster_ranges, _) = make_two_molecules();
    let natoms = apos.len();

    // Build a minimal RaffTopology + RaffState
    let mut topo = RaffTopology::new(natoms);
    for i in 0..natoms {
        let el_rvdw = if i % 15 < 6 { 1.7 } else if i % 15 < 8 { 1.52 } else { 1.2 }; // C/O/H
        topo.nb_params[i] = NbParams { sigma: 2.0 * el_rvdw, epsilon: 0.01, charge: 0.0, radius: el_rvdw * 0.8 };
    }
    let mut state = RaffState::new(natoms);
    state.set_positions(&apos);

    let nbcfg = NbConfig { enabled: true, rcut: 8.0, r_damp: 0.1, f_max: 50.0, k_coll: 100.0, excl_12: true, excl_13: true };

    // O(N²)
    let mut fapos_on2 = vec![numtypes::VEC3D_ZERO; natoms];
    let e_on2 = eval_nonbonded(&state, &topo, &nbcfg, &mut fapos_on2);

    // Broad-phase
    let mut bp = BroadPhase::new(cluster_ranges, 8.0);
    bp.rebuild(&state.pos[0..natoms]);
    let mut fapos_bp = vec![numtypes::VEC3D_ZERO; natoms];
    let e_bp = eval_nonbonded_broad(&state, &topo, &nbcfg, &bp, &mut fapos_bp);

    let e_diff = (e_on2 - e_bp).abs();
    assert!(e_diff < 1e-10, "RAFF Energy mismatch: O(N²)={}, broad={}, diff={}", e_on2, e_bp, e_diff);
    let mut max_f_diff = 0.0;
    for i in 0..natoms {
        let df = Vec3d::set_sub(fapos_on2[i], fapos_bp[i]);
        let d = df.norm2().sqrt();
        if d > max_f_diff { max_f_diff = d; }
    }
    assert!(max_f_diff < 1e-10, "RAFF Force mismatch: max|ΔF|={}", max_f_diff);
    println!("[parity-raff] O(N²) E={:.6}, broad E={:.6}, max|ΔF|={:.2e}, BP pairs={}", e_on2, e_bp, max_f_diff, bp.pairs().len());
}
