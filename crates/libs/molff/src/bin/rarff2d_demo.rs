//! rarff2d_demo — dynamics demo of the rod-free reactive forcefield.
//!
//! Scene: naphthalene = two fused hexagons (10 sp2 atoms) + 2 ring entities.
//! A) "clean": one atom displaced (bad NN guess) -> should fully repair.
//! B) "bad":   displacement + spurious atom in a ring hole -> extra atom steals
//!    a site, displaced atom orphaned -> eatom inconsistency map flags it.
//! Relax = overdamped GD (step_gd). Dumps CSV/XYZ to debug/rarff2d_dyn/
//! for scripts/plot_rarff2d_dyn.py.
//!
//! Run: cargo run --release --bin rarff2d_demo

use molff::rarff2d::{Rarff2d, Ring2d, TYPE_SP2};
use numtypes::Vec2d;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

const R0: f64 = 1.3;                        // aromatic C-C
const OUT: &str = "debug/rarff2d_dyn";

fn hexagon(cx: f64, cy: f64, r: f64, phi0: f64) -> Vec<Vec2d> {
    (0..6).map(|k| {
        let t = phi0 + k as f64 * PI / 3.0;
        Vec2d::new(cx + r * t.cos(), cy + r * t.sin())
    }).collect()
}

fn run(tag: &str, spurious: bool) {
    let c = R0 * (PI / 6.0).cos();          // fused centers at x=±1.1258, shared edge x=0
    let mut pos = hexagon(-c, 0.0, R0, PI / 2.0);
    for p in hexagon(c, 0.0, R0, PI / 2.0) {
        if p.x.abs() > 0.1 { pos.push(p); }  // right ring's 4 exclusive verts
    }
    pos[0].x -= 0.30; pos[0].y -= 0.25;      // corrupt: displace top atom OUTWARD (inward push would land it inside a repulsive wall)
    if spurious { pos.push(Vec2d::new(-c + 0.15, 0.1)); }  // extra atom in left hole
    let n = pos.len();
    println!("\n=== scene {tag}: {n} atoms (spurious={spurious}) ===");

    let mut ff = Rarff2d::new(vec![TYPE_SP2; n], pos.clone());
    for i in 0..n { ff.set_orient(i, 1.7 * i as f64 + 0.4); ff.inv_i[i] = 2.0; }  // random orientations, faster rotation
    let ring_z = Vec2d::new((PI / 2.0).cos(), (PI / 2.0).sin());
    ff.set_rings(vec![
        Ring2d { pos: Vec2d::new(-c, 0.0), zc: ring_z, nfold: 6, radius: R0, a: 2.0, h: 0.5, a_core: 1.5, r0_core: 0.7, b_core: 2.5 },
        Ring2d { pos: Vec2d::new(c, 0.0), zc: ring_z, nfold: 6, radius: R0, a: 2.0, h: 0.5, a_core: 1.5, r0_core: 0.7, b_core: 2.5 }]);

    let (f_err, t_err) = ff.fd_check(1e-5);
    println!("FD check (initial): max|F_err|={f_err:.3e} max|T_err|={t_err:.3e}");
    let e0 = ff.eval();
    let eatom0 = ff.eatom.clone();
    println!("initial E={e0:.4}  eatom={eatom0:.3?}");

    let mut fcsv = File::create(format!("{OUT}/{tag}_relax.csv")).expect("relax.csv");
    writeln!(fcsv, "step,E,maxF,maxT").unwrap();
    let mut fxyz = File::create(format!("{OUT}/{tag}_traj.xyz")).expect("traj.xyz");
    // overdamped GD (repair mode): dt, trust-radius dmax [A], dphi_max [rad]
    // dphi_max << gate w (0.3) else orientations overshoot the gate edge -> limit cycle
    let (dt, dmax, dphi_max, nsteps) = (0.01, 0.02, 0.05, 40000);
    let mut step = 0usize;
    let mut e;
    let t0 = std::time::Instant::now();
    while step < nsteps {
        e = ff.eval();                                   // refresh forces EVERY step
        let (mf, mt) = (ff.max_f2().sqrt(), ff.max_t2().sqrt());
        if step % 10 == 0 || step == nsteps - 1 {
            writeln!(fcsv, "{step},{e:.6},{mf:.6},{mt:.6}").unwrap();
            if step % 1000 == 0 {
                writeln!(fxyz, "{}\nstep {step} E={e:.6}", ff.natom()).unwrap();
                for i in 0..ff.natom() { writeln!(fxyz, "C {:.6} {:.6} 0.0  {:.6}", ff.pos[i].x, ff.pos[i].y, ff.eatom[i]).unwrap(); }
                eprintln!("step {step:5}  E={e:10.4}  max|F|={mf:.4}  max|T|={mt:.4}");
            }
        }
        if mf < 1e-3 && mt < 1e-3 { eprintln!("converged at step {step}"); break; }
        ff.step_gd(dt, dmax, dphi_max);
        step += 1;
    }
    eprintln!("done {step} steps in {:.2?}", t0.elapsed());
    e = ff.eval();
    println!("final   E={e:.4}  eatom={:.3?}", ff.eatom);
    let imax = ff.force.iter().map(|f| f.norm2()).enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
    let imaxt = ff.torq.iter().map(|t| t.abs()).enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
    println!("  maxF on atom {imax} ({:.3})  maxT on atom {imaxt} ({:.3})  zc={:?}", ff.force[imax].norm(), ff.torq[imaxt], ff.zc[imaxt]);

    let mut fa = File::create(format!("{OUT}/{tag}_atoms.csv")).expect("atoms.csv");
    writeln!(fa, "i,x0,y0,xf,yf,eatom0,eatomf").unwrap();
    for i in 0..n {
        writeln!(fa, "{i},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}", pos[i].x, pos[i].y, ff.pos[i].x, ff.pos[i].y, eatom0[i], ff.eatom[i]).unwrap();
    }
    let (f_err2, t_err2) = ff.fd_check(1e-5);
    println!("FD check (final):   max|F_err|={f_err2:.3e} max|T_err|={t_err2:.3e}");
}

fn main() {
    std::fs::create_dir_all(OUT).expect("create debug dir");
    run("clean", false);
    run("bad", true);
    let mut fr = File::create(format!("{OUT}/rings.csv")).expect("rings.csv");
    let c = R0 * (PI / 6.0).cos();
    writeln!(fr, "x,y,nfold,R,a,h").unwrap();
    for sx in [-1.0f64, 1.0] { writeln!(fr, "{:.6},0.0,6,{:.6},2.0,0.5", c * sx, R0).unwrap(); }
    println!("\nwrote {OUT}/{{clean,bad}}_{{relax,atoms,traj}}.csv/xyz + rings.csv");
}
