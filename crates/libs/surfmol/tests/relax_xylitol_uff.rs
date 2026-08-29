//! UFF xylitol relaxation: invariants, inertial-reset, and FIRE.
//!
//! Reproduces the RAFF solver-style comparison for UFF CPU:
//! - invariants: damped=0, no velocity purge, momentum must be conserved.
//! - inertial:   damped=0, dot(v,F)<0 -> v=0 (simple FIRE, constant dt).
//! - FIRE:       full FIRE with adaptive dt and velocity steering.

use std::path::Path;
use std::fs::File;
use std::io::Write;
use numtypes::Vec3d;
use moltopo::topology::{Topology, build_bonds_by_cutoff, build_angles_from_bonds, build_dihedrals_from_bonds, build_inversions_from_bonds};
use moltopo::xyz::read_xyz;
use surfmol::mol_world::{MolWorld, BondedFFMode};
use molff::raff::FireState;
use molff::multigrid::{TrussOp, GalerkinLevel, select_pivots_maximin, build_pivot_prolongation};

fn xorshift(seed: &mut u64) -> f64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed & 0x7fffffffffffffu64) as f64) * (2.0 / ((1u64 << 52) as f64)) - 1.0
}

fn momentum(apos: &[Vec3d], vapos: &[Vec3d]) -> (Vec3d, Vec3d) {
    let mut p = Vec3d::new(0.0, 0.0, 0.0);
    let mut l = Vec3d::new(0.0, 0.0, 0.0);
    for i in 0..apos.len() {
        p.add(vapos[i]);
        l.add(apos[i].cross(vapos[i]));
    }
    (p, l)
}

fn write_trace(path: &Path, step: &[i32], e: &[f64], fmax: &[f64]) {
    let mut f = File::create(path).expect("create trace");
    writeln!(f, "step\tE\tfmax").expect("header");
    for i in 0..step.len() {
        writeln!(f, "{}\t{:.6}\t{:.6}", step[i], e[i], fmax[i]).expect("row");
    }
}

fn run_fire(mw: &mut MolWorld, initial: &[Vec3d], niter: i32, flim: f64) -> (i32, f64, f64, Vec<i32>, Vec<f64>, Vec<f64>) {
    mw.dyn_atoms.atoms.apos.as_mut_slice().copy_from_slice(initial);
    mw.dyn_atoms.clean_velocity();
    let mut fire = FireState::new(0.001, 0.02);
    let mut step = Vec::new();
    let mut energy = Vec::new();
    let mut fmax_trace = Vec::new();
    let mut n_done = niter;
    for itr in 0..niter {
        let (eb, ea, ed, ei, _, _) = mw.eval_forces();
        let e = eb + ea + ed + ei;
        energy.push(e);
        step.push(itr);
        let dt = fire.dt;
        let mut v_dot_f = 0.0;
        let mut v_norm2 = 0.0;
        let mut f_norm2 = 0.0;
        let mut f2max = 0.0f64;
        for ia in 0..mw.natoms() {
            let f = mw.dyn_atoms.fapos.as_slice()[ia];
            let f2 = f.norm2();
            if f2 > flim * flim { panic!("run_fire: force exceeds limit at step={itr} atom={ia}: |F|={} flim={flim}", f2.sqrt()); }
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
                let f_hat = Vec3d::set_mul(mw.dyn_atoms.fapos.as_slice()[ia], 1.0 / f_mag);
                let v = mw.dyn_atoms.vapos.as_slice()[ia];
                mw.dyn_atoms.vapos.as_mut_slice()[ia] = Vec3d::set_lincomb(1.0 - fire.alpha, v, fire.alpha * v_mag, f_hat);
            }
            fire.n_pos += 1;
            if fire.n_pos > fire.n_min { fire.dt = (fire.dt * fire.f_inc).min(fire.dt_max); fire.alpha *= fire.f_alpha; }
        } else {
            fire.n_pos = 0;
            fire.dt *= fire.f_dec;
            fire.alpha = fire.alpha0;
            mw.dyn_atoms.clean_velocity();
        }
        for ia in 0..mw.natoms() {
            let v = mw.dyn_atoms.vapos.as_slice()[ia];
            mw.dyn_atoms.atoms.apos.as_mut_slice()[ia].add_mul(v, dt);
        }
        let fmax = f2max.sqrt();
        fmax_trace.push(fmax);
        if fmax < 1e-3 { n_done = itr + 1; break; }
    }
    let e_final = *energy.last().expect("run_fire: no iterations executed");
    let f_final = *fmax_trace.last().expect("run_fire: no iterations executed");
    (n_done, e_final, f_final, step, energy, fmax_trace)
}

#[test]
fn uff_xylitol_solvers() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().parent().unwrap().join("data/xyz/xylitol.xyz");
    let xyz = read_xyz(&path).expect("read xylitol.xyz");
    let apos = xyz.apos;
    let natoms = apos.len() as i32;

    let bonds = build_bonds_by_cutoff(&apos, 1.8);
    let angles = build_angles_from_bonds(natoms, &bonds);
    let dihedrals = build_dihedrals_from_bonds(&bonds);
    let inversions = build_inversions_from_bonds(natoms, &bonds);
    let top = Topology { apos, bonds, angles, dihedrals, inversions };

    let mut mw = MolWorld::from_topology(&top);
    mw.bonded_mode = BondedFFMode::Uff;
    mw.set_dummy_params();
    mw.make_neigh_bs();
    mw.bake_angle_neighs();
    mw.bake_dihedral_neighs();
    mw.bake_inversion_neighs();
    mw.map_atom_interactions();

    // Random distortion
    let mut seed = 0x12345678u64;
    let amp = 0.3;
    for i in 0..mw.natoms() {
        let dx = Vec3d::set_mul(Vec3d::new(xorshift(&mut seed), xorshift(&mut seed), xorshift(&mut seed)), amp);
        mw.dyn_atoms.atoms.apos.as_mut_slice()[i] = Vec3d::set_add(mw.dyn_atoms.atoms.apos.as_slice()[i], dx);
    }
    let apos_distorted: Vec<Vec3d> = mw.dyn_atoms.atoms.apos.as_slice().to_vec();

    // ---- one cached coarse-preconditioned nonlinear force step ----
    let free_mask = vec![true; mw.natoms()];
    let bonds: Vec<[i32;2]> = mw.uff.bon_atoms.as_slice().to_vec();
    let pivots = select_pivots_maximin(&bonds, mw.natoms(), 4, &free_mask);
    let p = build_pivot_prolongation(mw.dyn_atoms.apos(), &pivots, 2.0, &free_mask);
    let mass_dt2 = vec![100.0; mw.natoms()];
    let op = TrussOp::from_uff_bonds(&mw.uff, mw.dyn_atoms.apos(), &mass_dt2);
    let level = GalerkinLevel::new(&op, &p, pivots.len()*3);
    let (eb0c, ea0c, ed0c, ei0c, _, _) = mw.eval_forces();
    let e0c = eb0c + ea0c + ed0c + ei0c;
    let f0c = mw.dyn_atoms.fapos.as_slice().iter().map(|f| f.norm()).fold(0.0f64, f64::max);
    let (coarse_energy, coarse_step) = mw.apply_coarse_force_step(&level, &free_mask, 0.5);
    let (eb1c, ea1c, ed1c, ei1c, _, _) = mw.eval_forces();
    let e1c = eb1c + ea1c + ed1c + ei1c;
    let f1c = mw.dyn_atoms.fapos.as_slice().iter().map(|f| f.norm()).fold(0.0f64, f64::max);
    println!("xylitol UFF cached coarse step: pivots={pivots:?} E={e0c:.6}->{e1c:.6} fmax={f0c:.6}->{f1c:.6} coarse_energy={coarse_energy:.6} max_step={coarse_step:.6}");
    assert!(e1c < e0c, "cached coarse step did not lower nonlinear UFF energy: E0={e0c:.15e} E1={e1c:.15e} fmax0={f0c:.15e} fmax1={f1c:.15e} coarse_energy={coarse_energy:.15e} max_step={coarse_step:.15e}");
    let apos_coarse = mw.dyn_atoms.apos().to_vec();
    mw.dyn_atoms.atoms.apos.as_mut_slice().copy_from_slice(&apos_distorted);
    mw.dyn_atoms.clean_velocity();

    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().parent().unwrap().join("debug/relax_xylitol_uff");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    // ---- invariants (damping=0, no purge) ----
    for i in 0..mw.natoms() {
        mw.dyn_atoms.vapos.as_mut_slice()[i] = Vec3d::set_mul(Vec3d::new(xorshift(&mut seed), xorshift(&mut seed), xorshift(&mut seed)), 0.1);
    }
    let mut vsum = Vec3d::new(0.0, 0.0, 0.0);
    for &v in mw.dyn_atoms.vapos.as_slice() { vsum.add(v); }
    let vcorr = Vec3d::set_mul(vsum, 1.0 / mw.natoms() as f64);
    for v in mw.dyn_atoms.vapos.as_mut_slice() { v.sub(vcorr); }
    let (p0, l0) = momentum(mw.dyn_atoms.apos(), mw.dyn_atoms.vapos.as_slice());

    let dt = 0.01;
    let flim = 1e18;
    let niter = 500;
    let mut step = Vec::new();
    let mut e_trace = Vec::new();
    let mut fmax_trace = Vec::new();
    for itr in 0..niter {
        let (eb, ea, ed, ei, _, _) = mw.eval_forces();
        e_trace.push(eb + ea + ed + ei);
        step.push(itr);
        let mut f2max = 0.0;
        for ia in 0..mw.natoms() {
            let (_, _, f2) = mw.dyn_atoms.move_atom_md(ia, dt, flim, 1.0);
            if f2 > f2max { f2max = f2; }
        }
        fmax_trace.push(f2max.sqrt());
    }
    write_trace(&out_dir.join("trajectory_invariants.tsv"), &step, &e_trace, &fmax_trace);
    let (p1, l1) = momentum(mw.dyn_atoms.apos(), mw.dyn_atoms.vapos.as_slice());
    let dp = Vec3d::set_sub(p1, p0).norm();
    let dl = Vec3d::set_sub(l1, l0).norm();

    // ---- inertial reset (damping=0, dot(v,F)<0 -> v=0) ----
    mw.dyn_atoms.atoms.apos.as_mut_slice().copy_from_slice(&apos_distorted);
    for v in mw.dyn_atoms.vapos.as_mut_slice() { *v = Vec3d::new(0.0, 0.0, 0.0); }
    step.clear(); e_trace.clear(); fmax_trace.clear();
    let mut n_inertial = niter;
    for itr in 0..niter {
        let (eb, ea, ed, ei, _, _) = mw.eval_forces();
        e_trace.push(eb + ea + ed + ei);
        step.push(itr);
        let mut v_dot_f = 0.0;
        let mut f2max = 0.0;
        for ia in 0..mw.natoms() {
            let (ff, _, f2) = mw.dyn_atoms.move_atom_md(ia, dt, flim, 1.0);
            v_dot_f += ff;
            if f2 > f2max { f2max = f2; }
        }
        fmax_trace.push(f2max.sqrt());
        if v_dot_f < 0.0 { for v in mw.dyn_atoms.vapos.as_mut_slice() { *v = Vec3d::new(0.0, 0.0, 0.0); } }
        if f2max.sqrt() < 1e-3 { n_inertial = itr + 1; break; }
    }
    write_trace(&out_dir.join("trajectory_inertial.tsv"), &step, &e_trace, &fmax_trace);
    let e_inertial = e_trace.last().copied().unwrap_or(0.0);
    let f_inertial = fmax_trace.last().copied().unwrap_or(0.0);

    // ---- FIRE and one-coarse-step + FIRE at the same force threshold ----
    let (n_fire, e_fire, f_fire, step_fire, e_fire_trace, f_fire_trace) = run_fire(&mut mw, &apos_distorted, niter, flim);
    write_trace(&out_dir.join("trajectory_fire.tsv"), &step_fire, &e_fire_trace, &f_fire_trace);
    let (n_mg_fire, e_mg_fire, f_mg_fire, step_mg, e_mg_trace, f_mg_trace) = run_fire(&mut mw, &apos_coarse, niter, flim);
    write_trace(&out_dir.join("trajectory_mg_fire.tsv"), &step_mg, &e_mg_trace, &f_mg_trace);
    let full_eval_mg_fire = 1 + n_mg_fire;

    println!("xylitol UFF solvers:");
    println!("  invariants: |dP|={dp:.6} |dL|={dl:.6}");
    println!("  inertial:   n={n_inertial} E={e_inertial:.6} fmax={f_inertial:.6}");
    println!("  FIRE:       full evaluations={n_fire} E={e_fire:.6} fmax={f_fire:.6}");
    println!("  MG→FIRE:    full evaluations={full_eval_mg_fire} (1 coarse + {n_mg_fire} FIRE) E={e_mg_fire:.6} fmax={f_mg_fire:.6}");
    println!("  traces: {}/trajectory_*.tsv", out_dir.display());

    assert!(dp < 1e-8, "linear momentum not conserved: |dP|={dp}");
    assert!(dl < 1e-6, "angular momentum not conserved: |dL|={dl}");
    assert!(f_inertial < 1e-3, "inertial did not converge: fmax={f_inertial}");
    assert!(f_fire < 1e-3, "FIRE did not converge: fmax={f_fire}");
    assert!(f_mg_fire < 1e-3, "coarse-step + FIRE did not converge: fmax={f_mg_fire}");
    assert!((e_mg_fire-e_fire).abs() < 1e-6, "coarse-step + FIRE converged to a different energy: E_mg={e_mg_fire:.15e} E_fire={e_fire:.15e} dE={:.15e}", e_mg_fire-e_fire);
}
