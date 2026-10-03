//! rarff2d_bench — O(N^2) vs CSR-grid vs COG-group pair eval benchmark.
//! Realistic scenario: K bonded hexamer fragments scattered in a domain
//! (molecular packing ~1 atom/1.8 A^2 inside fragments, gaps between).
//! Writes debug/rarff2d_bench/scenario.csv (x,y,cell) for the plot script.
//! Run: cargo run --release -p molff --bin rarff2d_bench [--n 100] [--reps 50]

use molff::rarff2d::{Rarff2d, TYPE_SP2};
use numtypes::Vec2d;
use std::f64::consts::TAU;
use std::time::Instant;

struct Rng(u64);
impl Rng { fn next(&mut self) -> f64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 11) as f64 / (1u64 << 53) as f64 } }

/// Hexagonal lattice at bond-length spacing a=1.3 A (graphene-like dense
/// sheet), ~70% random fill, ±0.3 A jitter, random orientations.
/// Placement lattice is deliberately NON-COMMENSURATE with the acceleration
/// grid (dx=6.0 = rcut) — a real stress test, not an aligned best case.
/// Returns (ff, half_domain). Atom count ≈ n (fill noise ±few %).
fn make_ff(n: usize, seed: u64) -> (Rarff2d, f64) {
    let mut rng = Rng(seed | 1);
    let a = 1.3;                                    // lattice const = bond length
    let fill = 0.70;
    // sites per A^2 = 1/(a^2 sin60); side so that filled sites ≈ n
    let side = (n as f64 / (fill * 0.68)).sqrt();
    let mut pos = Vec::new();
    let mut phis = Vec::new();
    let ny = (side / (a * 0.866)) as usize;
    for iy in 0..ny {
        let y = iy as f64 * a * 0.866 - side * 0.5;
        let xoff = if iy % 2 == 0 { 0.0 } else { 0.5 * a };
        let nx = (side / a) as usize;
        for ix in 0..nx {
            if rng.next() > fill { continue; }
            let x = ix as f64 * a + xoff - side * 0.5;
            pos.push(Vec2d::new(x + 0.3 * (rng.next() - 0.5), y + 0.3 * (rng.next() - 0.5)));
            phis.push(rng.next() * TAU);
        }
    }
    let mut ff = Rarff2d::new(vec![TYPE_SP2; pos.len()], pos);
    for i in 0..ff.natom() { ff.set_orient(i, phis[i]); }
    (ff, side * 0.5)
}

/// Packed baseline: all pairs inside rcut (dense blob, diam < rcut) →
/// t_per_pair = the price of one VALID evaluation incl. force/torque/eatom writes.
/// This is the physics floor every method must pay for ninside pairs.
fn packed_pair_cost() -> f64 {
    let mut rng = Rng(3);
    let mut pos = Vec::new();
    // lattice a=1.0 cropped to disc r<2.9 → all pairs < 6 A = inside rcut
    let a = 1.0;
    for iy in 0..6 { for ix in 0..6 {
        let p = Vec2d::new(ix as f64 * a - 2.5 + 0.2 * (rng.next() - 0.5), iy as f64 * a * 0.866 - 2.2 + 0.2 * (rng.next() - 0.5));
        if p.norm() < 2.9 { pos.push(p); }
    }}
    let n = pos.len();
    let mut ff = Rarff2d::new(vec![TYPE_SP2; n], pos);
    for i in 0..n { ff.set_orient(i, rng.next() * TAU); }
    let reps = 2000;
    let t0 = Instant::now();
    let mut e = 0.0;
    for _ in 0..reps { e += std::hint::black_box(ff.eval()); }
    let t = t0.elapsed().as_secs_f64() / reps as f64;
    let npair = (n * (n - 1) / 2) as f64;
    println!("packed baseline: {n} atoms all-inside, {:.1} ns/valid-pair (eval {:.1} us, E={e:.1})", t / npair * 1e9, t * 1e6);
    t / npair
}

/// Max and mean neighbors within rcut, counted directly (O(N^2), one-off).
fn neighbor_stats(pos: &[Vec2d], rcut: f64) -> (usize, f64) {
    let n = pos.len();
    let mut cnt = vec![0usize; n];
    let mut max_n = 0usize;
    for i in 0..n { for j in (i + 1)..n {
        if (pos[j] - pos[i]).norm2() < rcut * rcut { cnt[i] += 1; cnt[j] += 1; }
    }}
    for &c in &cnt { max_n = max_n.max(c); }
    (max_n, cnt.iter().sum::<usize>() as f64 / n as f64)
}
/// (sec/eval, energy, pair_ef calls/eval, in-cutoff evals/eval)
/// Timing excludes the counter overhead: counters are counted on a separate
/// single instrumented eval (dbg_timing on), the loop runs untimed-hot.
fn bench_eval(ff: &mut Rarff2d, reps: usize) -> (f64, f64, f64, f64) {
    let mut e = 0.0;
    let t0 = Instant::now();
    for _ in 0..reps { e += std::hint::black_box(ff.eval()); }
    let t = t0.elapsed().as_secs_f64() / reps as f64;
    std::hint::black_box(e);
    ff.dbg_timing = true;
    let e2 = ff.eval();
    ff.dbg_timing = false;
    (t, e2, ff.dbg_npair as f64, ff.dbg_ninside as f64)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut n: usize = 100; let mut reps: usize = 50;
    for w in args.windows(2) { match w[0].as_str() { "--n" => n = w[1].parse().unwrap(), "--reps" => reps = w[1].parse().unwrap(), _ => {} } }

    // --- pair_math micro-cost: in-cutoff vs out-cutoff, streaming a prebuilt
    // halo-like buffer (isolates math from gather/scatter/rebuild overheads)
    {
        let (mut f0, _) = make_ff(100, 5);
        let t = f0.types[0];
        let zi = f0.zc[0]; let pi = Vec2d::new(0.0, 0.0);
        let zj = f0.zc[1]; let pj_in = Vec2d::new(1.35, 0.2); let pj_out = Vec2d::new(10.0, 10.0);
        let mut acc = 0.0;
        let m = 1_000_000usize;
        let t0 = Instant::now();
        for _ in 0..m { let r = std::hint::black_box(f0.pair_math(t, zi, pi, t, zj, pj_in)); acc += r.0; }
        let t_in = t0.elapsed().as_secs_f64() / m as f64;
        let t0 = Instant::now();
        for _ in 0..m { let r = std::hint::black_box(f0.pair_math(t, zi, pi, t, zj, pj_out)); acc += r.0; }
        let t_out = t0.elapsed().as_secs_f64() / m as f64;
        println!("pair_math: inside={:.1}ns/call outside={:.1}ns/call (acc={:.2})", t_in * 1e9, t_out * 1e9, acc);
    }

    println!("rarff2d_bench: N~{n} reps={reps}  (hexagonal lattice a=1.3, 70% fill, us/eval)");
    let t_packed = packed_pair_cost();
    let (mut ff, half) = make_ff(n, 42);
    let rdom = half + 4.0;
    let (max_nbr, mean_nbr) = neighbor_stats(&ff.pos, ff.rcut);
    println!("neighbors inside rcut={:.1}: mean={:.1} max={}", ff.rcut, mean_nbr, max_nbr);
    let (t_n2, e_n2, np_n2, ni_n2) = bench_eval(&mut ff, reps);
    println!("O(N^2) : {:>10.2} us  pair_ef={np_n2:>9.0} inside={ni_n2:>8.0}  eff={:.0}% (baseline ninside*t_packed={:.1}us)",
        t_n2 * 1e6, 100.0 * ni_n2 * t_packed / t_n2, ni_n2 * t_packed * 1e6);
    for dx in [6.0, 3.0, 1.0] {
        let (mut ff_g, _) = make_ff(n, 42);
        let ncell = (2.0 * rdom / dx).ceil() as usize + 1;
        ff_g.set_grid(dx, [-rdom, -rdom], ncell, ncell);
        let (t_g, e_g, np_g, ni_g) = bench_eval(&mut ff_g, reps);
        assert!((e_g - e_n2).abs() < 1e-8, "grid parity broken: e_n2={e_n2} e_g={e_g} (dx={dx})");
        assert!((ni_g - ni_n2).abs() < 1.0, "grid inside-count broken: {ni_n2} vs {ni_g} (dx={dx})");
        println!("gr dx={dx} : {:>10.2} us  pair_ef={np_g:>9.0} inside={ni_g:>8.0}  eff={:.0}% ({:.1}x fewer calls)",
            t_g * 1e6, 100.0 * ni_g * t_packed / t_g, np_n2 / np_g);
    }
    // per-pass timing breakdown for dx=6: [intra, gather, inner, scatter]
    {
        let (mut ff_g, _) = make_ff(n, 42);
        let ncell = (2.0 * rdom / 6.0).ceil() as usize + 1;
        ff_g.set_grid(6.0, [-rdom, -rdom], ncell, ncell);
        ff_g.dbg_timing = true;
        ff_g.eval();
        let d = &ff_g.dbg_t;
        println!("passes [intra={:.2} gather={:.2} inner={:.2} scatter={:.2}] us", d[0]*1e6, d[1]*1e6, d[2]*1e6, d[3]*1e6);
    }
    {
        let (mut ff_g, _) = make_ff(n, 42);
        ff_g.set_groups(6.0, (n / 4).max(16));
        let (t_g, e_g, np_g, ni_g) = bench_eval(&mut ff_g, reps);
        assert!((e_g - e_n2).abs() < 1e-8, "groups parity broken: e_n2={e_n2} e_g={e_g}");
        let ng = ff_g.groups.as_ref().unwrap().ngroup();
        println!("grp r_j=6: {:>10.2} us  pair_ef={np_g:>9.0} inside={ni_g:>8.0}  eff={:.0}% ({:.1}x fewer calls, ngroups={ng})",
            t_g * 1e6, 100.0 * ni_g * t_packed / t_g, np_n2 / np_g);
    }

    // scaling table: N sweep
    println!("\nscaling (hexagonal lattice a=1.3 70% fill, dx=6=rcut):");
    println!("{:>6} | {:>11} {:>11} {:>11} | {:>8} {:>8} | {:>9} {:>6}", "N", "O(N^2)", "gr dx=6", "grp r_j=6", "dx6", "grp", "pair_ef", "eff%");
    for nn in [24usize, 48, 96, 192, 384, 768, 1536, 3072] {
        let reps2 = (reps.max(10) * 1000 / nn.max(10)).max(5);
        let mut ts = [0.0f64; 3]; let (nreal2, np2, npg, nig);
        { let (mut f, _) = make_ff(nn, 7); let (t,_,p,_) = bench_eval(&mut f, reps2); ts[0] = t; nreal2 = f.natom(); np2 = p; }
        { let (mut f, h) = make_ff(nn, 7); let rdom2 = h + 4.0; let nc = (2.0 * rdom2 / 6.0).ceil() as usize + 1; f.set_grid(6.0, [-rdom2, -rdom2], nc, nc); let (t,_,p,q) = bench_eval(&mut f, reps2); ts[1] = t; npg = p; nig = q; }
        { let (mut f, _) = make_ff(nn, 7); f.set_groups(6.0, (nn / 4).max(16)); let (t,_,_,_) = bench_eval(&mut f, reps2); ts[2] = t; }
        println!("{:>6} | {:>11.2} {:>11.2} {:>11.2} | {:>7.2}x {:>7.2}x | {:>9.2} {:>5.0}%", nreal2, ts[0] * 1e6, ts[1] * 1e6, ts[2] * 1e6, ts[0] / ts[1], ts[0] / ts[2], np2 / npg, 100.0 * nig * t_packed / ts[1]);
    }

    // scenario dump for the reality-check plot (atoms + cells + occupancy hist)
    let out = "debug/rarff2d_bench";
    std::fs::create_dir_all(out).unwrap();
    let (mut ff_s, _) = make_ff(n, 42);
    let ncell = (2.0 * rdom / 6.0).ceil() as usize + 1;
    ff_s.set_grid(6.0, [-rdom, -rdom], ncell, ncell);
    ff_s.eval();
    let mut csv = String::from("x,y,cell\n");
    for i in 0..ff_s.natom() {
        csv += &format!("{:.4},{:.4},{}\n", ff_s.pos[i].x, ff_s.pos[i].y, ff_s.grid.as_ref().unwrap().cell_of[i]);
    }
    std::fs::write(format!("{out}/scenario.csv"), &csv).unwrap();
    // group AABBs for the same scene (fit box + rcut halo in the plot)
    let (mut ff_s2, _) = make_ff(n, 42);
    ff_s2.set_groups(6.0, (n / 4).max(16));
    ff_s2.eval();
    let mut aab = String::from("g,xmin,ymin,xmax,ymax,natom\n");
    if let Some(gp) = &ff_s2.groups {
        for g in 0..gp.ngroup() {
            let a = gp.aabb[g];
            let n_a = gp.buckets.offsets[g + 1] - gp.buckets.offsets[g];
            aab += &format!("{g},{:.3},{:.3},{:.3},{:.3},{n_a}\n", a[0], a[1], a[2], a[3]);
        }
    }
    std::fs::write(format!("{out}/groups.csv"), &aab).unwrap();
    println!("wrote {out}/scenario.csv + {out}/groups.csv  (dom=±{rdom:.1}, grid dx=6, ncell={ncell})");
    println!("done.");
}
