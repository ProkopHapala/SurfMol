//! raff_reactive_demo — headless reactive-RAFF movie generator.
//!
//! Scenes (bonds EMERGE from geometry — no bond list):
//!   pair    — 2 sp3 C 3.0 A apart, random orientations → rotate into alignment + bond
//!   methane — C at origin + 4 H on a 1.6 A tetrahedral sphere + 0.15 A LCG jitter
//!   cloud   — 12 sp3 C + 12 H uniform-random in a 6 A box (harmonic walls)
//!
//! Writes <out>/traj.xyz (multi-frame XYZ, comment `step <i> E=<e>`) and
//! <out>/energy.tsv (step, E, maxF, maxT). Default out = debug/raff_reactive_demo/<scene>.
//!
//! Run: cargo run --release -p molff --bin raff_reactive_demo -- --scene methane --steps 1500

use std::io::Write;
use std::path::PathBuf;

use molff::raff::{self, BoxCfg, RaffConfig, RaffState, RaffTopology};
use molff::raff_reactive::{set_reactive_inertia, write_xyz_frame, ReactParams, Reactive, REACT_PEX_M};
use numtypes::{Quat4d, Vec3d, VEC3D_ZERO};

const C: ReactParams = ReactParams { r_cov: 0.77, a: 1.0, b: 0.9 };   // sp3 carbon
const H: ReactParams = ReactParams { r_cov: 0.37, a: 1.0, b: 0.9 };   // 1-port hydrogen
const E_BOND: f64 = -0.3;                                           // emerged-bond threshold (well depth ~ -1)

/// Tiny deterministic LCG (no rand dep) — same as tests/test_raff_reactive.rs.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 11) as f64) * (1.0 / 9007199254740992.0) }
    fn vec3(&mut self) -> Vec3d { Vec3d::new(self.next(), self.next(), self.next()) }
}

/// Random unit quaternion: random axis + random angle.
fn quat_random(rng: &mut Lcg) -> Quat4d {
    let mut v = rng.vec3() * 2.0 - Vec3d::new(1.0, 1.0, 1.0);
    v.normalize();
    raff::quat_from_omega_dt(v * (rng.next() * 2.0 * std::f64::consts::PI), 1.0)
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

// XYZ frame writer is shared: molff::raff_reactive::write_xyz_frame (used by the FFI traj too).

/// Scene builder: (state, topo, params, box_cfg).
fn build_scene(scene: &str) -> (RaffState, RaffTopology, Vec<ReactParams>, BoxCfg) {
    let mut rng = Lcg(0xB5297A4DB5297A4D);
    match scene {
        "pair" => {
            let mut topo = RaffTopology::new(2);
            topo.set_sp3(0); topo.set_sp3(1);
            let params = vec![C, C];
            set_reactive_inertia(&mut topo, &params);
            let mut state = RaffState::new(2);
            state.pos[0] = Vec3d::new(-1.5, 0.0, 0.0);
            state.pos[1] = Vec3d::new( 1.5, 0.0, 0.0);
            state.quat[0] = quat_random(&mut rng);
            state.quat[1] = quat_random(&mut rng);
            (state, topo, params, BoxCfg::default())
        }
        "methane" => {
            let mut topo = RaffTopology::new(5);
            topo.set_sp3(0);
            for i in 1..5 { topo.set_1port(i, Vec3d::new(1.0, 0.0, 0.0)); }
            let params = vec![C, H, H, H, H];
            set_reactive_inertia(&mut topo, &params);
            let mut state = RaffState::new(5);
            let s = 1.0 / 3.0f64.sqrt();
            let dirs = [Vec3d::new(s, s, s), Vec3d::new(s, -s, -s), Vec3d::new(-s, s, -s), Vec3d::new(-s, -s, s)];
            for i in 1..5 {   // 1.6 A sphere + deterministic 0.15 A jitter (beats random-start metastability)
                let j = rng.vec3() * 0.3 - Vec3d::new(0.15, 0.15, 0.15);
                let p = dirs[i - 1] * 1.6 + j;
                state.pos[i] = p;
                // 1-port H arrives port-first (lobe toward C); a port aiming
                // away is outside the gate ⇒ no torque ⇒ rigid-port dead zone.
                state.quat[i] = quat_align(Vec3d::new(1.0, 0.0, 0.0), p * (-1.0 / p.norm()));
            }
            (state, topo, params, BoxCfg::default())
        }
        "cloud" => {
            let n = 24;
            let mut topo = RaffTopology::new(n);
            for i in 0..12 { topo.set_sp3(i); }
            for i in 12..n { topo.set_1port(i, Vec3d::new(1.0, 0.0, 0.0)); }
            let params = vec![C; 12].into_iter().chain(vec![H; 12]).collect::<Vec<_>>();
            set_reactive_inertia(&mut topo, &params);
            let mut state = RaffState::new(n);
            for i in 0..n {
                state.pos[i] = rng.vec3() * 6.0 - Vec3d::new(3.0, 3.0, 3.0);   // uniform in [-3,3]^3
                state.quat[i] = quat_random(&mut rng);
            }
            let box_cfg = BoxCfg { enabled: true, min: Vec3d::new(-3.0, -3.0, -3.0), max: Vec3d::new(3.0, 3.0, 3.0), k: 20.0 };
            (state, topo, params, box_cfg)
        }
        _ => panic!("unknown scene '{scene}' — expected methane|cloud|pair"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone());
    let scene = get("--scene").unwrap_or_else(|| "methane".to_string());
    let steps: usize = get("--steps").map(|s| s.parse().expect("--steps must be integer")).unwrap_or(2000);
    let stride: usize = get("--stride").map(|s| s.parse().expect("--stride must be integer")).unwrap_or(10);
    let dt: f64 = get("--dt").map(|s| s.parse().expect("--dt must be float")).unwrap_or(0.02);
    let cdamp: f64 = get("--cdamp").map(|s| s.parse().expect("--cdamp must be float")).unwrap_or(0.9);
    let use_grid = args.iter().any(|a| a == "--grid");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");   // crates/libs/molff → repo root
    let out_dir = get("--out").map(PathBuf::from).unwrap_or_else(|| root.join("debug/raff_reactive_demo").join(&scene));
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("cannot create out dir {}: {e}", out_dir.display()));

    let (mut state, topo, params, box_cfg) = build_scene(&scene);
    let mut rx = Reactive::new(&topo, params, REACT_PEX_M);
    if use_grid { rx.set_grid([-4.0, -4.0, -4.0], [4, 4, 4], 2.0); }
    let cfg = RaffConfig { dt, cdamp, rot_damp: cdamp, flim: 20.0, box_cfg, ..Default::default() };
    let n = state.natoms;
    let (mut fapos, mut tau) = (vec![VEC3D_ZERO; n], vec![VEC3D_ZERO; n]);
    let (mut traj, mut etsv) = (String::new(), String::from("step\tE\tmaxF\tmaxT\n"));
    eprintln!("raff_reactive_demo: scene={scene} natoms={n} steps={steps} dt={dt} cdamp={cdamp} grid={use_grid} rcut={:.3} -> {}", rx.cfg.rcut, out_dir.display());
    std::io::stderr().flush().expect("flush stderr");

    write_xyz_frame(&mut traj, &state, &topo, 0, 0.0).expect("write_xyz_frame");
    let mut e0 = f64::NAN;
    let mut e_last = f64::NAN;
    for step in 0..steps {
        let (e, mf, mt) = rx.step_reactive_md(&mut state, &topo, &cfg, &mut fapos, &mut tau);
        assert!(e.is_finite() && mf.is_finite() && mt.is_finite(), "non-finite at step {step}: E={e} maxF={mf} maxT={mt} pos0={:?}", state.pos[0]);
        if step == 0 { e0 = e; }
        e_last = e;
        if step % stride == 0 { write_xyz_frame(&mut traj, &state, &topo, step + 1, e).expect("write_xyz_frame"); }
        etsv.push_str(&format!("{}\t{e:.6e}\t{mf:.6e}\t{mt:.6e}\n", step + 1));
        if step % 100 == 0 { eprintln!("step {step:6}: E={e:12.6}  max|F|={mf:.4e}  max|tau|={mt:.4e}"); std::io::stderr().flush().expect("flush stderr"); }
    }
    eprintln!("step {steps:6}: E={e_last:12.6}  (E_start={e0:.6}, dE={:.6})", e_last - e0);

    let bonds = rx.emerged_bonds(&state, &topo, E_BOND);
    println!("emerged bonds (E_pair < {E_BOND}): {}", bonds.len());
    for &(i, j, ep) in &bonds {
        let r = (state.pos[j as usize] - state.pos[i as usize]).norm();
        println!("  bond {}-{}: E_pair={:.4} r={:.4}", i, j, ep, r);
    }
    if scene == "methane" && bonds.len() < 4 {
        println!("DIAGNOSTIC: methane formed only {} of 4 C-H bonds — likely metastable port alignment (see notes/reports/raff_reactive_debug.md)", bonds.len());
    }

    let f_xyz = out_dir.join("traj.xyz");
    let f_tsv = out_dir.join("energy.tsv");
    std::fs::write(&f_xyz, &traj).unwrap_or_else(|e| panic!("write {}: {e}", f_xyz.display()));
    std::fs::write(&f_tsv, &etsv).unwrap_or_else(|e| panic!("write {}: {e}", f_tsv.display()));
    println!("wrote {} ({} frames) + {}", f_xyz.display(), traj.matches("\nstep ").count() + 1, f_tsv.display());
}
