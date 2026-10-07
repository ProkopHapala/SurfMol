//! RARFF-2D tests: FD parity, pair equilibrium, anti-node repulsion, relaxation.
//! Diagnostic tests — print actual numbers, assert on physical invariants.

use molff::rarff2d::{Bond2d, Rarff2d, Ring2d, TYPE_SP2, TYPE_SP3};
use numtypes::Vec2d;
use std::f64::consts::PI;

const R0: f64 = 1.3; // TYPE_SP2.r_cov * 2 = aromatic C-C

/// 2 sp2 atoms at distance r, i at origin φ=0, j at (r,0) φ=φ_j.
fn pair(r: f64, phi_j: f64) -> Rarff2d {
    let mut ff = Rarff2d::new(vec![TYPE_SP2, TYPE_SP2], vec![Vec2d::new(0.0, 0.0), Vec2d::new(r, 0.0)]);
    ff.set_orient(1, phi_j);
    ff
}

#[test]
fn test_fd_parity() {
    // random-ish config: 5 atoms, mixed types, arbitrary orientations
    let pos = vec![
        Vec2d::new(0.0, 0.0), Vec2d::new(1.4, 0.3), Vec2d::new(0.4, 1.5),
        Vec2d::new(-1.2, 0.8), Vec2d::new(2.6, 1.1)];
    let types = vec![TYPE_SP2, TYPE_SP2, TYPE_SP3, TYPE_SP2, TYPE_SP2];
    let mut ff = Rarff2d::new(types, pos);
    for i in 0..ff.natom() { ff.set_orient(i, 0.37 * i as f64 + 0.11); }
    ff.set_rings(vec![
        Ring2d { pos: Vec2d::new(0.7, 0.9), zc: Vec2d::new((0.3f64).cos(), (0.3f64).sin()), nfold: 6, radius: 1.3, a: 2.0, h: 0.5, a_core: 1.0, r0_core: 0.9, b_core: 3.0 },
        Ring2d { pos: Vec2d::new(2.3, 0.6), zc: Vec2d::new(1.0, 0.0), nfold: 5, radius: 1.1, a: 2.0, h: 0.5, a_core: 1.0, r0_core: 0.8, b_core: 3.0 }]);
    ff.set_bonds(vec![
        Bond2d::from_ends(Vec2d::new(0.4, 0.2), Vec2d::new(1.6, 0.9), 1.0, 0.9, 0.3),  // near atoms 0,1 — nonzero on all DOFs
        Bond2d { pos: Vec2d::new(-1.8, 1.6), zc: Vec2d::new((0.9f64).cos(), (0.9f64).sin()), half: 0.7, half0: 0.7, a: 1.0, b: 0.9, w: 0.3 }]);
    let (f_err, t_err) = ff.fd_check(1e-5);
    println!("FD parity: max|F_err|={f_err:.3e} max|T_err|={t_err:.3e}");
    assert!(f_err < 1e-5, "force FD err {f_err}");
    assert!(t_err < 1e-5, "torque FD err {t_err}");
}

#[test]
fn test_pair_equilibrium() {
    // aligned pair at r0: j faces back at i (φ_j=π) → Y=1 → E = a(e²−2e) = −a = −1
    let mut ff = pair(R0, PI);
    let e = ff.eval();
    println!("aligned pair at r0: E={e:.6} maxF={:.3e} maxT={:.3e}", ff.max_f2().sqrt(), ff.max_t2().sqrt());
    assert!((e + 1.0).abs() < 1e-9, "E={e} expected -1.0");
    assert!(ff.max_f2() < 1e-12, "force at minimum should vanish");
    assert!(ff.max_t2() < 1e-12, "torque at minimum should vanish");
}

#[test]
fn test_anti_node_repels() {
    // j misoriented by 60° (anti-node for nfold=3): Y≈0.02 → pair repels at r0
    let mut ff = pair(R0, PI + PI / 3.0);
    ff.eval();
    let f_radial = ff.force[1].x; // hij = +x → positive = pushed away from i
    println!("anti-node pair at r0: F_j.x={f_radial:.4} (repulsion wants >0) eatom={:?}", ff.eatom);
    assert!(f_radial > 0.0, "misaligned pair should repel, got F={f_radial}");
}

#[test]
fn test_offaxis_repels() {
    // j at 60° from i's port dir (at (r0·cos60, r0·sin60)); both face each other:
    // Δ_i = 60°, Δ_j = 60° → Y = g(60)² ≈ tiny → repulsive
    let mut ff = Rarff2d::new(vec![TYPE_SP2, TYPE_SP2],
        vec![Vec2d::new(0.0, 0.0), Vec2d::new(R0 * 0.5, R0 * (3.0f64).sqrt() * 0.5)]);
    ff.set_orient(1, PI / 3.0 + PI); // j points back toward i (th=60°, so φ_j=60+180=240°)
    ff.eval();
    let f = ff.force[1];
    let hij = Vec2d::new(0.5, (3.0f64).sqrt() * 0.5);
    let f_radial = f.x * hij.x + f.y * hij.y;
    println!("off-axis pair: F_radial={f_radial:.4} (repulsion wants >0)");
    assert!(f_radial > 0.0, "off-axis pair should repel, got F_r={f_radial}");
}

#[test]
fn test_relax_bend3() {
    // 3 trigonal atoms in a shallow V (~135°), random orientations.
    // sp2 atoms cannot make a straight chain (180° = anti-node) → must bend to ~120°.
    // NOTE: starting perfectly straight, the end atoms get repelled out of the
    // attraction tail before bonds form (local basin) — start slightly bent.
    let pos = vec![Vec2d::new(0.0, 0.0), Vec2d::new(R0, 0.0), Vec2d::new(R0 + R0 * 0.7, R0 * 0.5)];
    let mut ff = Rarff2d::new(vec![TYPE_SP2; 3], pos);
    for i in 0..3 { ff.set_orient(i, 0.4 * i as f64); }
    let (e, n, conv) = ff.relax(20000, 0.02, 0.9, 1e-4, true);
    println!("bend3: E={e:.4} steps={n} converged={conv}");
    for i in 0..3 { println!("  atom{i} pos=({:.4},{:.4}) phi={:.1}deg E={:.4}",
        ff.pos[i].x, ff.pos[i].y, ff.phi(i).to_degrees(), ff.eatom[i]); }
    let v1 = ff.pos[0] - ff.pos[1];
    let v2 = ff.pos[2] - ff.pos[1];
    let cos_ang = v1.dot(v2) / (v1.norm() * v2.norm());
    println!("  angle at middle atom: {:.1}deg  |v1|={:.4} |v2|={:.4}", cos_ang.acos().to_degrees(), v1.norm(), v2.norm());
    assert!(conv, "bend3 did not converge");
    assert!(cos_ang.abs() < 0.9, "chain should bend (cos={cos_ang}), not stay linear");
    assert!((v1.norm() - R0).abs() < 0.15, "bond length off: {}", v1.norm());
}

#[test]
fn test_relax_hexagon() {
    // 6 trigonal atoms perturbed around a hexagon → relax back to regular r0-sided hexagon.
    // This is the PAH repair primitive: positions+orientations self-organize into a ring.
    let n = 6;
    let mut pos = Vec::new();
    for i in 0..n {
        let th = 2.0 * PI * i as f64 / n as f64;
        let r = R0 * (1.0 + 0.15 * (0.7 * i as f64).sin()); // perturbed radius
        pos.push(Vec2d::new(r * th.cos(), r * th.sin()));
    }
    let mut ff = Rarff2d::new(vec![TYPE_SP2; n], pos);
    for i in 0..n { ff.set_orient(i, 0.9 * i as f64); } // random-ish orientations
    let (e, nsteps, conv) = ff.relax(8000, 0.02, 0.92, 1e-4, true);
    println!("hexagon: E={e:.4} steps={nsteps} converged={conv}");
    let mut max_len_err = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let d = (ff.pos[j] - ff.pos[i]).norm();
        max_len_err = max_len_err.max((d - R0).abs());
        println!("  edge {i}-{j} len={d:.4}");
    }
    println!("  E/atom={:.4} max|len-r0|={max_len_err:.4}", e / n as f64);
    assert!(conv, "hexagon did not converge");
    assert!(max_len_err < 0.1, "hexagon edges not at r0: max err {max_len_err}");
    assert!(e < -4.0, "hexagon energy too high: {e} (expect ~ -6 + strain)");
}

#[test]
fn test_ring_pulls_atoms_to_sites() {
    // ring entity n=6 R=1.3 alone (no bonds): 6 atoms placed inside the annulus
    // at wrong angles get dragged onto the 6 sites. With a>0 attractive sites.
    let n = 6;
    let mut pos = Vec::new();
    for i in 0..n {
        let th = 2.0 * PI * (i as f64 + 0.09 * (i as f64).sin()) / n as f64; // ~site angles but perturbed
        let r = 1.3 + 0.1 * (1.7 * i as f64).cos();
        pos.push(Vec2d::new(r * th.cos(), r * th.sin()));
    }
    let mut ff = Rarff2d::new(vec![TYPE_SP2; n], pos);
    ff.set_rings(vec![Ring2d { pos: Vec2d::new(0.0, 0.0), zc: Vec2d::new(1.0, 0.0), nfold: 6, radius: 1.3, a: 2.0, h: 0.5, a_core: 1.5, r0_core: 0.7, b_core: 2.5 }]);
    let (e, nsteps, conv) = ff.relax(4000, 0.02, 0.9, 1e-3, true);
    let mut max_r_err = 0.0f64;
    for i in 0..n { max_r_err = max_r_err.max((ff.pos[i].norm() - 1.3).abs()); }
    println!("ring: E={e:.4} steps={nsteps} conv={conv} max|r-1.3|={max_r_err:.4} eatom={:?}", ff.eatom);
    assert!(conv, "ring relax did not converge");
    assert!(max_r_err < 0.08, "atoms not on ring radius: {max_r_err}");
    assert!(e < -11.0, "6 atoms should each sit in a -2 eV site, E={e}");
}

#[test]
fn test_ring_plus_pairs_hexagon() {
    // ring + pair bonding together: ring pins sites, pairs bond the neighbors.
    // Atoms start slightly displaced; final E should be deep (~ring 6*-2 + bonds ~6*-1).
    let n = 6;
    let mut pos = Vec::new();
    for i in 0..n {
        let th = 2.0 * PI * i as f64 / n as f64;
        let r = 1.3 * (1.0 + 0.08 * (2.3 * i as f64).sin());
        pos.push(Vec2d::new(r * th.cos(), r * th.sin()));
    }
    let mut ff = Rarff2d::new(vec![TYPE_SP2; n], pos);
    for i in 0..n { ff.set_orient(i, 1.3 * i as f64); }
    ff.set_rings(vec![Ring2d { pos: Vec2d::new(0.05, -0.03), zc: Vec2d::new(1.0, 0.0), nfold: 6, radius: 1.3, a: 2.0, h: 0.5, a_core: 1.5, r0_core: 0.7, b_core: 2.5 }]);
    let (e, nsteps, conv) = ff.relax(20000, 0.02, 0.9, 1e-3, true);
    println!("ring+pairs: E={e:.4} steps={nsteps} conv={conv} eatom={:?}", ff.eatom);
    assert!(conv, "ring+pairs relax did not converge");
    assert!(e < -12.0, "combined energy too high: {e}");
}

// NOTE: bond-entity behavior tests live in Python (invPPAFM
// export_invAFM/scripts/test_rarff2d_bond.py) — they drive the FFI, which is
// the interface invAFM actually calls. Kept here: fd_parity (engine-internal
// FD-vs-analytic check, covers bond/ring DOFs via the scene below) plus the
// pre-existing atom/ring/grid tests.

#[test]
fn test_grid_parity() {
    // CSR grid eval must produce identical energy/forces as O(N2) eval.
    // 40 pseudo-random atoms (xorshift) in a 12x12 A domain, dx=1.0 A cells.
    let n = 40;
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng >> 11) as f64 / (1u64 << 53) as f64 };
    let pos: Vec<Vec2d> = (0..n).map(|_| Vec2d::new(12.0 * next() - 6.0, 12.0 * next() - 6.0)).collect();
    let phis: Vec<f64> = (0..n).map(|_| 6.28 * next()).collect();
    let types = vec![TYPE_SP2; n];

    let mut ff_ref = Rarff2d::new(types.clone(), pos.clone());
    for i in 0..n { ff_ref.set_orient(i, phis[i]); }
    let e_ref = ff_ref.eval();
    let f_ref = ff_ref.force.clone();
    let ea_ref = ff_ref.eatom.clone();
    let t_ref = ff_ref.torq.clone();

    let mut ff_g = Rarff2d::new(types, pos.clone());
    for i in 0..n { ff_g.set_orient(i, phis[i]); }
    ff_g.set_grid(1.0, [-7.0, -7.0], 14, 14);   // 1 A cells, covers atoms (clamped at edges)
    let e_g = ff_g.eval();

    let mut max_ef = 0.0f64; let mut max_ee = 0.0f64; let mut max_et = 0.0f64;
    for i in 0..n {
        max_ef = max_ef.max((ff_g.force[i] - f_ref[i]).norm());
        max_ee = max_ee.max((ff_g.eatom[i] - ea_ref[i]).abs());
        max_et = max_et.max((ff_g.torq[i] - t_ref[i]).abs());
    }
    println!("grid parity: dE={:.3e} max|dF|={max_ef:.3e} max|deatom|={max_ee:.3e} max|dT|={max_et:.3e}", (e_g - e_ref).abs());
    assert!((e_g - e_ref).abs() < 1e-10, "grid eval E mismatch: O(N2)={e_ref} grid={e_g}");
    assert!(max_ef < 1e-10 && max_et < 1e-10 && max_ee < 1e-10, "grid eval force/torque/eatom mismatch");

    // same scene through COG collision groups — must also be identical
    let mut ff_c = Rarff2d::new(vec![TYPE_SP2; n], pos.clone());
    for i in 0..n { ff_c.set_orient(i, phis[i]); }
    ff_c.set_groups(6.0, 16);
    let e_c = ff_c.eval();
    let mut max_ef2 = 0.0f64;
    for i in 0..n { max_ef2 = max_ef2.max((ff_c.force[i] - f_ref[i]).norm()); }
    println!("groups parity: dE={:.3e} max|dF|={max_ef2:.3e} ngroups={}", (e_c - e_ref).abs(), ff_c.groups.as_ref().unwrap().ngroup());
    assert!((e_c - e_ref).abs() < 1e-10, "groups eval E mismatch: O(N2)={e_ref} groups={e_c}");
    assert!(max_ef2 < 1e-10, "groups eval force mismatch");
}
