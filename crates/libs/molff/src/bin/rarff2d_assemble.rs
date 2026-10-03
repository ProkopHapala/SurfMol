//! rarff2d_assemble — self-assembly demo: random atoms -> bonded geometry.
//!
//! N sp2 atoms seeded randomly inside a soft confinement "arena" (keeps unbound
//! atoms from evaporating), damped-MD dynamics; bonds emerge from the gated
//! Morse pair potential — no topology given. Records debug/rarff2d_assemble/
//! traj.xyz (E + eatom in comment/4th column) + relax.csv + summary of detected
//! bonds (r<1.6 A, gate Y>0.3) and rings.
//!
//! Run: cargo run --release --bin rarff2d_assemble [-- N] [--seed S]

use molff::rarff2d::{Rarff2d, TYPE_SP2};
use numtypes::Vec2d;
use std::fs::File;
use std::io::Write;

const OUT_DEF: &str = "debug/rarff2d_assemble";

/// xorshift64* — tiny deterministic RNG (no deps, reproducible seed).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {  // uniform [0,1)
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.windows(2).find(|w| w[0] == "--out").map(|w| w[1].clone()).unwrap_or_else(|| OUT_DEF.to_string());
    std::fs::create_dir_all(&out).expect("create debug dir");
    let argf = |k: &str, d: f64| args.windows(2).find(|w| w[0] == k).map(|w| w[1].parse().unwrap_or(d)).unwrap_or(d);
    let natom = argf("--n", 14.0) as usize;
    let seed = (argf("--seed", 7.0) as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let arena_r = argf("--arena", 4.5);          // arena radius [A]
    let arena_k = argf("--kwall", 2.0);          // wall stiffness [eV/A^2]
    let nsteps = argf("--steps", 60000.0) as usize;
    let (dt, cdamp) = (0.02, argf("--cdamp", 0.98));

    // --- seed atoms: random pos in disc r<0.8*arena, random orientations ---
    let mut rng = Rng(seed | 1);
    let pos: Vec<Vec2d> = (0..natom).map(|_| {
        let r = arena_r * 0.8 * rng.next().sqrt();
        let t = rng.next() * std::f64::consts::TAU;
        Vec2d::new(r * t.cos(), r * t.sin())
    }).collect();
    let mut fi = File::create(format!("{out}/init.csv")).expect("init.csv");
    writeln!(fi, "i,x,y").unwrap();
    for (i, p) in pos.iter().enumerate() { writeln!(fi, "{i},{:.6},{:.6}", p.x, p.y).unwrap(); }
    let mut ff = Rarff2d::new(vec![TYPE_SP2; natom], pos);
    for i in 0..natom { ff.set_orient(i, rng.next() * std::f64::consts::TAU); }
    ff.arena = Some((arena_r, arena_k));
    println!("assemble: {natom} sp2 atoms in arena R={arena_r} k={arena_k}, dt={dt} cdamp={cdamp}, {nsteps} steps");

    let mut fcsv = File::create(format!("{out}/relax.csv")).expect("relax.csv");
    writeln!(fcsv, "step,E,maxF,maxT").unwrap();
    let mut fxyz = File::create(format!("{out}/traj.xyz")).expect("traj.xyz");
    let t0 = std::time::Instant::now();
    let mut e = 0.0;
    for step in 0..nsteps {
        e = ff.eval();
        if step % 500 == 0 {
            let (mf, mt) = (ff.max_f2().sqrt(), ff.max_t2().sqrt());
            writeln!(fcsv, "{step},{e:.6},{mf:.6},{mt:.6}").unwrap();
            writeln!(fxyz, "{}\nstep {step} E={e:.6}", natom).unwrap();
            for i in 0..natom { writeln!(fxyz, "C {:.6} {:.6} 0.0  {:.6}", ff.pos[i].x, ff.pos[i].y, ff.eatom[i]).unwrap(); }
            eprintln!("step {step:6}  E={e:9.4}  max|F|={mf:.3}  max|T|={mt:.3}");
            if mf < 1e-3 && mt < 1e-3 { eprintln!("converged at step {step}"); break; }
        }
        ff.step_md(dt, cdamp);
    }
    eprintln!("done in {:.2?}", t0.elapsed());
    e = ff.eval();
    println!("final E={e:.4}");

    // --- bond detection + summary ---
    let mut bonds = Vec::new();
    let mut coord = vec![0usize; natom];
    for i in 0..natom { for j in (i + 1)..natom {
        let d = (ff.pos[j] - ff.pos[i]).norm();
        if d < 1.6 {
            let th = (ff.pos[j] - ff.pos[i]).y.atan2((ff.pos[j] - ff.pos[i]).x);
            let gi = molff::rarff2d::gate(3, 0.3, th - ff.phi(i)).0;
            let gj = molff::rarff2d::gate(3, 0.3, th + std::f64::consts::PI - ff.phi(j)).0;
            if gi * gj > 0.3 { bonds.push((i, j, d)); coord[i] += 1; coord[j] += 1; }
        }
    }}
    println!("detected {} bonds:", bonds.len());
    for (i, j, d) in &bonds { println!("  {i}-{j}  r={d:.4}"); }
    for i in 0..natom { println!("  atom{i}: coord={} E={:.3} pos=({:.3},{:.3}) phi={:.0}deg", coord[i], ff.eatom[i], ff.pos[i].x, ff.pos[i].y, ff.phi(i).to_degrees()); }

    let mut fa = File::create(format!("{out}/atoms.csv")).expect("atoms.csv");
    writeln!(fa, "i,x,y,phi,eatom,coord").unwrap();
    for i in 0..natom { writeln!(fa, "{i},{:.6},{:.6},{:.6},{:.6},{}", ff.pos[i].x, ff.pos[i].y, ff.phi(i), ff.eatom[i], coord[i]).unwrap(); }
    let mut fb = File::create(format!("{out}/bonds.csv")).expect("bonds.csv");
    writeln!(fb, "i,j,r").unwrap();
    for (i, j, d) in &bonds { writeln!(fb, "{i},{j},{d:.6}").unwrap(); }
    println!("wrote {out}/{{traj.xyz,relax.csv,atoms.csv,bonds.csv}}");
}
