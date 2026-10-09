//! Reactive RAFF tests: FD parity, facing-pair E(r), grid parity, invariances,
//! methane self-assembly. Diagnostic tests — print actual numbers, assert on
//! physical invariants (notes/designs/2026-10-09_raff_reactive_3d_inventory.md §2.5).

use molff::raff::{self, RaffConfig, RaffState, RaffTopology};
use molff::raff_reactive::{pair_math, set_reactive_inertia, ReactParams, Reactive};
use numtypes::{Quat4d, Vec3d, VEC3D_ZERO};
use std::cell::RefCell;
use std::f64::consts::PI;

const REPO: &str = "../../..";
const C: ReactParams = ReactParams { r_cov: 0.77, a: 1.0, b: 0.9 };   // sp3 carbon
const H: ReactParams = ReactParams { r_cov: 0.37, a: 1.0, b: 0.9 };   // 1-port hydrogen (port_local +x)
const PEX_M: u32 = 3;

/// Tiny deterministic LCG (no rand dep): x *= 6364136223846793005 + c → [0,1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 11) as f64) * (1.0 / 9007199254740992.0) }
    fn vec3(&mut self) -> Vec3d { Vec3d::new(self.next(), self.next(), self.next()) }
}

/// Quaternion rotating unit vector a onto b (shortest arc).
fn quat_align(a: Vec3d, b: Vec3d) -> Quat4d {
    let ax = Vec3d::cross(a, b);
    let s = ax.norm();
    let c = a.dot(b);
    if s < 1e-12 {
        if c > 0.0 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); }
        // a ≈ -b: 180° about any axis ⊥ a — 180° about a itself leaves it invariant!
        let mut ax = Vec3d::cross(a, if a.x.abs() < 0.9 { Vec3d::new(1.0, 0.0, 0.0) } else { Vec3d::new(0.0, 1.0, 0.0) });
        ax.normalize();
        return Quat4d::new(ax.x, ax.y, ax.z, 0.0);
    }
    raff::quat_from_omega_dt(ax * (s.atan2(c) / s), 1.0)
}

/// Random unit quaternion: random axis + random angle.
fn quat_random(rng: &mut Lcg) -> Quat4d {
    let mut v = rng.vec3() * 2.0 - Vec3d::new(1.0, 1.0, 1.0);
    v.normalize();
    raff::quat_from_omega_dt(v * (rng.next() * 2.0 * PI), 1.0)
}

fn make_topo(n: usize, sp3: &[usize], h: &[usize]) -> RaffTopology {
    let mut topo = RaffTopology::new(n);
    for &i in sp3 { topo.set_sp3(i); }
    for &i in h { topo.set_1port(i, Vec3d::new(1.0, 0.0, 0.0)); }
    topo
}

// ========== 1. FD parity: forces + torques vs energy ==========

#[test]
fn test_fd_parity() {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let mut topo = make_topo(6, &[0, 1, 2], &[3, 4, 5]);
    let params = vec![C, C, C, H, H, H];
    set_reactive_inertia(&mut topo, &params);
    let mut state = RaffState::new(6);
    for i in 0..6 { state.pos[i] = rng.vec3() * 4.0; state.quat[i] = quat_random(&mut rng); }
    let rx = RefCell::new(Reactive::new(&topo, params, PEX_M));
    let eval = |s: &RaffState, f: &mut [Vec3d], t: &mut [Vec3d]| rx.borrow_mut().eval_reactive(s, &topo, f, t);
    let (fe, det) = raff::fd_check_forces_with(&state, &topo, 1e-5, &eval);
    let te = raff::fd_check_torques_with(&state, &topo, 1e-5, &eval);
    println!("FD parity: max rel force err = {fe:.3e}, max rel torque err = {te:.3e} (npair={} ninside={})", rx.borrow().dbg_npair, rx.borrow().dbg_ninside);
    for (i, d, _) in &det { println!("  force fd mismatch atom {i}: fd={:.6e} analytic={:.6e} rel={:.3e}", d.x, d.y, d.z); }
    assert!(fe < 1e-5, "force FD rel err {fe:.3e} >= 1e-5");
    assert!(te < 1e-5, "torque FD rel err {te:.3e} >= 1e-5");
}

// ========== 2. Two facing sp3 atoms: E(r) well at r0=2·r_cov ==========

#[test]
fn test_facing_pair_energy() {
    let topo = make_topo(2, &[0, 1], &[]);
    let params = vec![C, C];
    let mut rx = Reactive::new(&topo, params, PEX_M);
    let mut state = RaffState::new(2);
    let a0 = topo.port_local[0];                              // sp3 port 0 = (1,1,1)/√3
    state.quat[0] = quat_align(a0, Vec3d::new(1.0, 0.0, 0.0));  // i port0 → +x
    state.quat[1] = quat_align(a0, Vec3d::new(-1.0, 0.0, 0.0)); // j port0 → −x (facing i)
    let mut fapos = vec![VEC3D_ZERO; 2];
    let mut tau = vec![VEC3D_ZERO; 2];
    let mut e_min = f64::INFINITY;
    let mut r_min = 0.0;
    println!("facing sp3 pair E(r):");
    let mut r = 1.2;
    while r <= 3.0 + 1e-9 {
        state.pos[1] = Vec3d::new(r, 0.0, 0.0);
        let e = rx.eval_reactive(&state, &topo, &mut fapos, &mut tau);
        println!("  r={r:.3}  E={e:.6}");
        if e < e_min { e_min = e; r_min = r; }
        r += 0.05;
    }
    println!("  min: E_min={e_min:.6} at r={r_min:.3} (expect −1.0 at 1.54)");
    assert!((r_min - 1.54).abs() < 0.02, "E_min at r={r_min}, expected 1.54");
    // Exact minimum: at r=r0, e=pex(0)=1 and facing ports give Y=1 → E=a(1−2)=−1 exactly
    state.pos[1] = Vec3d::new(1.54, 0.0, 0.0);
    let e0 = rx.eval_reactive(&state, &topo, &mut fapos, &mut tau);
    println!("  E(r0=1.54)={e0:.9} (expect −1.0, Y=1 gated well bottom)");
    assert!((e0 + 1.0).abs() < 1e-6, "E(r0)={e0}, expected −1.0 (a·(e²−2eY) at e=Y=1)");
    // Anti-facing: j port0 flipped to +x. Residual gate leak ~3·(1/9)⁴ → E_min ≈ −2e-7.
    state.quat[1] = quat_align(a0, Vec3d::new(1.0, 0.0, 0.0));
    let mut e_worst = 0.0f64;
    let mut r = 1.2;
    while r <= 3.0 + 1e-9 {
        state.pos[1] = Vec3d::new(r, 0.0, 0.0);
        let e = rx.eval_reactive(&state, &topo, &mut fapos, &mut tau);
        e_worst = e_worst.min(e);
        r += 0.05;
    }
    println!("anti-facing pair: min E over scan = {e_worst:.3e} (expect ≥ ~−2e-7 gate leak)");
    assert!(e_worst > -1e-5, "anti-facing pair attracted: E={e_worst}");
    // Torque sign: rotate j by +10° about z → τ_j.z must restore alignment (<0).
    state.quat[1] = quat_align(a0, Vec3d::new(-1.0, 0.0, 0.0));
    let dq = raff::quat_from_omega_dt(Vec3d::new(0.0, 0.0, 10.0 * PI / 180.0), 1.0);
    state.quat[1] = raff::quat_normalize(raff::quat_mul(dq, state.quat[1]));
    state.pos[1] = Vec3d::new(1.54, 0.0, 0.0);
    rx.eval_reactive(&state, &topo, &mut fapos, &mut tau);
    println!("j tilted +10° about z: tau_j = ({:.4}, {:.4}, {:.4}) — tau_z<0 restores", tau[1].x, tau[1].y, tau[1].z);
    assert!(tau[1].z < 0.0, "torque should rotate j back toward alignment, got tau_z={}", tau[1].z);
}

// ========== 3. Grid parity: O(N²) vs Grid3 ==========

fn random_cloud(seed: u64, n_c: usize, n_h: usize, lo: f64, hi: f64) -> (RaffState, RaffTopology, Vec<ReactParams>) {
    let mut rng = Lcg(seed);
    let n = n_c + n_h;
    let mut topo = make_topo(n, &(0..n_c).collect::<Vec<_>>(), &(n_c..n).collect::<Vec<_>>());
    let params = vec![C; n_c].into_iter().chain(vec![H; n_h]).collect::<Vec<_>>();
    set_reactive_inertia(&mut topo, &params);
    let mut state = RaffState::new(n);
    for i in 0..n { state.pos[i] = rng.vec3() * (hi - lo) + Vec3d::new(lo, lo, lo); state.quat[i] = quat_random(&mut rng); }
    (state, topo, params)
}

#[test]
fn test_grid_parity() {
    let (state, topo, params) = random_cloud(0xDEADBEEFCAFEF00D, 100, 50, 0.5, 12.5);
    let mut rx = Reactive::new(&topo, params, PEX_M);
    let (mut fa0, mut ta0) = (vec![VEC3D_ZERO; 150], vec![VEC3D_ZERO; 150]);
    let e0 = rx.eval_reactive(&state, &topo, &mut fa0, &mut ta0);
    let ea0 = rx.eatom.clone();
    let (np0, ni0) = (rx.dbg_npair, rx.dbg_ninside);
    // Grid domain slightly SMALLER than the cloud → clamping exercised
    rx.set_grid([1.0, 1.0, 1.0], [3, 3, 3], 4.0);
    let (mut fa1, mut ta1) = (vec![VEC3D_ZERO; 150], vec![VEC3D_ZERO; 150]);
    let e1 = rx.eval_reactive(&state, &topo, &mut fa1, &mut ta1);
    let (np1, ni1) = (rx.dbg_npair, rx.dbg_ninside);
    let mut df = 0.0f64;
    let mut dt = 0.0f64;
    let mut de = 0.0f64;
    for i in 0..150 {
        df = df.max((fa0[i] - fa1[i]).norm());
        dt = dt.max((ta0[i] - ta1[i]).norm());
        de = de.max((ea0[i] - rx.eatom[i]).abs());
    }
    println!("grid parity: |ΔE|={:.3e}  max|ΔF|={df:.3e}  max|Δτ|={dt:.3e}  max|Δeatom|={de:.3e}", (e0 - e1).abs());
    println!("  O(N²): npair={np0} ninside={ni0} | grid: npair={np1} ninside={ni1}  rcut={:.3}", rx.cfg.rcut);
    assert_eq!(np1, np0, "grid skipped pairs: {np1} != {np0}");
    assert!((e0 - e1).abs() < 1e-10, "|ΔE|={:.3e}", (e0 - e1).abs());
    assert!(df < 1e-10 && dt < 1e-10 && de < 1e-10, "grid parity failed: dF={df:.3e} dτ={dt:.3e} dE_at={de:.3e}");
}

// ========== 4. Translation / rotation invariance ==========

#[test]
fn test_invariances() {
    let (state, topo, params) = random_cloud(0xDEADBEEFCAFEF00D, 100, 50, 0.5, 12.5);
    let rx = RefCell::new(Reactive::new(&topo, params, PEX_M));
    rx.borrow_mut().set_grid([1.0, 1.0, 1.0], [3, 3, 3], 4.0);
    let eval = |s: &RaffState, f: &mut [Vec3d], t: &mut [Vec3d]| rx.borrow_mut().eval_reactive(s, &topo, f, t);
    let sf = raff::check_translation_invariance_with(&state, &topo, &eval);
    let st = raff::check_rotation_invariance_with(&state, &topo, &eval);
    println!("invariance: |ΣF|={sf:.3e}  |Σ(x×F + τ)|={st:.3e}");
    assert!(sf < 1e-9, "ΣF = {sf:.3e} — pair forces not Newton-symmetric");
    assert!(st < 1e-9, "Σ(x×F + τ) = {st:.3e} — port-torque bookkeeping off");
}

// ========== 5. Methane self-assembly ==========

/// Shared methane-relaxation driver: relax `state` for 3000 damped steps,
/// write traj/energy to debug/raff_reactive/{tag}_*, return (rx, final E).
fn methane_run(state: &mut RaffState, topo: &RaffTopology, params: Vec<ReactParams>, tag: &str) -> (Reactive, f64) {
    let mut rx = Reactive::new(topo, params, PEX_M);
    let cfg = RaffConfig { dt: 0.02, cdamp: 0.9, rot_damp: 0.9, flim: 20.0, ..Default::default() };
    let (mut fapos, mut tau) = (vec![VEC3D_ZERO; 5], vec![VEC3D_ZERO; 5]);
    let out_dir = format!("{REPO}/debug/raff_reactive");
    std::fs::create_dir_all(&out_dir).expect("cannot create debug/raff_reactive");
    let mut traj = String::new();
    let mut etsv = String::from("step\tE\tmaxF\tmaxT\n");
    let mut e_prev = f64::INFINITY;
    for step in 0..3000 {
        let (e, mf, mt) = rx.step_reactive_md(state, topo, &cfg, &mut fapos, &mut tau);
        assert!(e.is_finite(), "energy non-finite at step {step}: E={e}");
        if step % 200 == 0 || step == 2999 {
            println!("step {step:4}: E={e:9.5}  max|F|={mf:.4e}  max|τ|={mt:.4e}");
            e_prev = e;
        }
        etsv.push_str(&format!("{step}\t{e:.6e}\t{mf:.6e}\t{mt:.6e}\n"));
        if step % 10 == 0 {
            traj.push_str("5\nstep ");
            traj.push_str(&step.to_string());
            traj.push('\n');
            for i in 0..5 {
                let el = if i == 0 { "C" } else { "H" };
                traj.push_str(&format!("{el}  {:.6} {:.6} {:.6}\n", state.pos[i].x, state.pos[i].y, state.pos[i].z));
            }
        }
    }
    std::fs::write(format!("{out_dir}/{tag}_traj.xyz"), &traj).expect("write traj");
    std::fs::write(format!("{out_dir}/{tag}_energy.tsv"), &etsv).expect("write energy tsv");
    println!("wrote {out_dir}/{tag}_traj.xyz + {tag}_energy.tsv, final E={e_prev:.6}");
    (rx, e_prev)
}

/// Docking test: 4 H placed in-cone (tetrahedral sites + 0.15 Å jitter) with
/// port-first quats must all bond — verifies the full pipeline (gate, torques,
/// geometry → CH4) for well-posed approaches.
#[test]
fn test_methane_assembly() {
    let mut rng = Lcg(0xB5297A4DB5297A4D);
    let mut topo = make_topo(5, &[0], &[1, 2, 3, 4]);
    let params = vec![C, H, H, H, H];
    set_reactive_inertia(&mut topo, &params);
    let mut state = RaffState::new(5);
    let s = 1.0 / 3.0f64.sqrt();
    let dirs = [Vec3d::new(s, s, s), Vec3d::new(s, -s, -s), Vec3d::new(-s, s, -s), Vec3d::new(-s, -s, s)];
    for i in 1..5 {
        let j = rng.vec3() * 0.3 - Vec3d::new(0.15, 0.15, 0.15);
        let p = dirs[i - 1] * 2.0 + j;              // in-cone on a 2.0 Å sphere
        state.pos[i] = p;
        // 1-port H arrives port-first (radical lobe toward the substrate):
        // a port aiming away is outside the gate ⇒ no torque ⇒ it can never
        // reorient (rigid-port dead zone — documented in the notes).
        state.quat[i] = quat_align(Vec3d::new(1.0, 0.0, 0.0), p * (-1.0 / p.norm()));
    }
    let (mut rx, _e) = methane_run(&mut state, &topo, params, "methane");
    // Emerged bonds: expect 4 C–H pairs at r ≈ r0_CH = 1.14
    let bonds = rx.emerged_bonds(&state, &topo, -0.3);
    println!("emerged bonds (E<−0.3): {bonds:?}");
    assert_eq!(bonds.len(), 4, "expected 4 C–H bonds, got {bonds:?}");
    for &(i, j, ep) in &bonds {
        let r = (state.pos[j as usize] - state.pos[i as usize]).norm();
        println!("  bond {i}-{j}: r={r:.4} (expect ~1.14) E={ep:.4}");
        assert!((r - 1.14).abs() < 0.05, "bond {i}-{j} length {r} != 1.14±0.05");
        assert!(i == 0 || j == 0, "bond {i}-{j} is not a C–H bond");
    }
    // All 6 H–C–H angles ≈ 109.47°
    for a in 1..5 {
        for b in (a + 1)..5 {
            let da = (state.pos[a] - state.pos[0]) * (1.0 / (state.pos[a] - state.pos[0]).norm());
            let db = (state.pos[b] - state.pos[0]) * (1.0 / (state.pos[b] - state.pos[0]).norm());
            let ang = da.dot(db).clamp(-1.0, 1.0).acos() * 180.0 / PI;
            println!("  H{a}-C-H{b} angle = {ang:.3}° (expect 109.47±5)");
            assert!((ang - 109.47).abs() < 5.0, "H{a}-C-H{b} = {ang:.3}°");
        }
    }
    // Direct pair_math check on the final geometry (single-source sanity)
    let p0 = rx.params[0];
    let np0 = topo.nport[0] as usize;
    for j in 1..5 {
        let npj = topo.nport[j] as usize;
        let res = pair_math(&rx.cfg, &p0, &rx.h_world[0..np0], state.pos[0], &rx.params[j], &rx.h_world[j * 4..j * 4 + npj], state.pos[j]);
        println!("  pair 0-{j}: {res:?}");
    }
}

/// DIAGNOSTIC (run `cargo test -- --ignored`): random approach directions.
/// With the two-sided gate, an H landing outside all 4 tetrahedral port cones
/// sees only the ungated e² wall and is ejected into the zero-force tail, where
/// damped MD freezes it — that IS directional bonding (the feature), not a bug.
/// Prints how many of the 4 random starts dock; no hard assertion.
#[test]
#[ignore]
fn test_methane_random_start() {
    let mut rng = Lcg(0xB5297A4DB5297A4D);
    let mut topo = make_topo(5, &[0], &[1, 2, 3, 4]);
    let params = vec![C, H, H, H, H];
    set_reactive_inertia(&mut topo, &params);
    let mut state = RaffState::new(5);
    for i in 1..5 {
        let mut d = rng.vec3() * 2.0 - Vec3d::new(1.0, 1.0, 1.0);
        d.normalize();
        state.pos[i] = d * 2.0;
        state.quat[i] = quat_align(Vec3d::new(1.0, 0.0, 0.0), d * (-1.0));   // port-first
    }
    let (mut rx, _e) = methane_run(&mut state, &topo, params, "methane_random");
    let bonds = rx.emerged_bonds(&state, &topo, -0.3);
    println!("random-start: {} of 4 H docked (out-of-cone H's are repelled by design)", bonds.len());
    for &(i, j, ep) in &bonds {
        let r = (state.pos[j as usize] - state.pos[i as usize]).norm();
        println!("  bond {i}-{j}: r={r:.4} E={ep:.4}");
    }
}
