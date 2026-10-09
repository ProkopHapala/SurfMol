//! ffi_raff_reactive.rs — extern "C" opaque-handle API for the 3D reactive
//! forcefield (`raff_reactive`), exported by libmolff.so (cdylib).
//!
//! Contract: `SurfMol/scripts/raff3d_ffi.py` — all arrays numpy float64
//! C-contiguous; `types` is uint8 (0=sp3, 1=sp2, 2=sp1, 3=H 1-port).
//! `quat` (n,4) is (x,y,z,w), renormalized on ingest. Positions/forces/torques
//! are (n,3). Energy eval ~ O(N²) CPU (or Grid3 after `raff3d_set_grid`).
//!
//! NOTE: `unsafe` at this FFI boundary is unavoidable (raw C pointers); the
//! crate-wide convention "only OpenCL uses unsafe" is waived here by design —
//! every entry point asserts non-null + consistent lengths (fail loud).

use std::io::Write;
use std::os::raw::c_char;
use numtypes::{Quat4d, Vec3d, VEC3D_ZERO};
use spacc::uniform_grid::neighbors_within;
use crate::raff::{quat_normalize, BoxCfg, RaffConfig, RaffState, RaffTopology};
use crate::raff_reactive::{pair_math, project_ports, set_reactive_inertia, write_xyz_frame, ReactParams, Reactive, REACT_PEX_M};

/// Multi-frame XYZ trajectory sink (see `raff3d_set_traj`): BufWriter + write
/// stride + steps completed since attach. Written inside `do_step`.
pub struct TrajOut {
    pub f: std::io::BufWriter<std::fs::File>,
    pub stride: usize,
    pub step: usize,
}

/// Opaque FFI state: rigid-atom state + port topology + reactive evaluator +
/// MD config + persistent force/torque scratch (allocated once in `raff3d_new`).
pub struct Raff3dFF {
    pub state: RaffState,
    pub topo: RaffTopology,
    pub rx: Reactive,
    pub cfg: RaffConfig,
    pub fapos: Vec<Vec3d>,
    pub tau: Vec<Vec3d>,
    pub traj: Option<TrajOut>,
}

#[inline] unsafe fn hf(h: *mut Raff3dFF) -> &'static mut Raff3dFF {
    assert!(!h.is_null(), "raff3d FFI: null handle");
    unsafe { &mut *h }
}
#[inline] unsafe fn rd_f64<'a>(p: *const f64, n: usize) -> &'a [f64] {
    assert!(!p.is_null(), "raff3d FFI: null f64 array");
    unsafe { std::slice::from_raw_parts(p, n) }
}
#[inline] unsafe fn wr_f64<'a>(p: *mut f64, n: usize) -> &'a mut [f64] {
    assert!(!p.is_null(), "raff3d FFI: null f64 out-array");
    unsafe { std::slice::from_raw_parts_mut(p, n) }
}

/// Default covalent radii per type code (0=sp3, 1=sp2, 2=sp1, 3=H 1-port).
const RCOV: [f64; 4] = [0.77, 0.73, 0.66, 0.37];

/// n atoms, types u8 (0=sp3, 1=sp2, 2=sp1, 3=H 1-port). Positions init to
/// origin, quats to identity — call raff3d_set_state before eval.
#[no_mangle]
pub extern "C" fn raff3d_new(n: i32, types: *const u8) -> *mut Raff3dFF {
    assert!(n > 0, "raff3d_new: n={n} <= 0");
    assert!(!types.is_null(), "raff3d_new: null types");
    let n = n as usize;
    let ty = unsafe { std::slice::from_raw_parts(types, n) };
    let mut topo = RaffTopology::new(n);
    let mut params = Vec::with_capacity(n);
    for (i, &t) in ty.iter().enumerate() {
        assert!(t <= 3, "raff3d_new: bad type code {t} at atom {i} (0=sp3, 1=sp2, 2=sp1, 3=H)");
        match t { 1 => topo.set_sp2(i), 2 => topo.set_sp1(i), 3 => topo.set_1port(i, Vec3d::new(1.0, 0.0, 0.0)), _ => topo.set_sp3(i) }
        params.push(ReactParams { r_cov: RCOV[t as usize], a: 1.0, b: 0.9 });
    }
    set_reactive_inertia(&mut topo, &params);          // mass=1 (RaffTopology::new default)
    let rx = Reactive::new(&topo, params, REACT_PEX_M);
    let cfg = RaffConfig { dt: 0.02, cdamp: 0.9, rot_damp: 0.9, flim: 20.0, ..Default::default() };
    Box::into_raw(Box::new(Raff3dFF { state: RaffState::new(n), topo, rx, cfg, fapos: vec![VEC3D_ZERO; n], tau: vec![VEC3D_ZERO; n], traj: None }))
}

#[no_mangle]
pub extern "C" fn raff3d_free(h: *mut Raff3dFF) {
    if !h.is_null() { drop(unsafe { Box::from_raw(h) }); }
}

/// pos (n,3) f64, quat (n,4) as (x,y,z,w) — renormalized on ingest (fail-loud
/// on degenerate/non-finite). Velocities reset to zero (fresh state).
#[no_mangle]
pub extern "C" fn raff3d_set_state(h: *mut Raff3dFF, pos: *const f64, quat: *const f64) {
    let ff = unsafe { hf(h) };
    let n = ff.state.natoms;
    let p = unsafe { rd_f64(pos, 3 * n) };
    let q = unsafe { rd_f64(quat, 4 * n) };
    for i in 0..n {
        let (x, y, z) = (p[3 * i], p[3 * i + 1], p[3 * i + 2]);
        assert!(x.is_finite() && y.is_finite() && z.is_finite(), "raff3d_set_state: non-finite pos at atom {i}: ({x}, {y}, {z})");
        ff.state.pos[i].set(x, y, z);
        let qq = Quat4d::new(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
        let n2 = qq.x * qq.x + qq.y * qq.y + qq.z * qq.z + qq.w * qq.w;
        assert!(n2.is_finite() && n2 > 1e-24, "raff3d_set_state: degenerate quat at atom {i}: ({}, {}, {}, {})", qq.x, qq.y, qq.z, qq.w);
        ff.state.quat[i] = quat_normalize(qq);
        ff.state.vel[i] = VEC3D_ZERO;
        ff.state.omega[i] = VEC3D_ZERO;
    }
}

/// pos (n,3), quat (n,4) as (x,y,z,w).
#[no_mangle]
pub extern "C" fn raff3d_get_state(h: *mut Raff3dFF, pos: *mut f64, quat: *mut f64) {
    let ff = unsafe { hf(h) };
    let n = ff.state.natoms;
    let po = unsafe { wr_f64(pos, 3 * n) };
    let qo = unsafe { wr_f64(quat, 4 * n) };
    for i in 0..n {
        po[3 * i] = ff.state.pos[i].x; po[3 * i + 1] = ff.state.pos[i].y; po[3 * i + 2] = ff.state.pos[i].z;
        qo[4 * i] = ff.state.quat[i].x; qo[4 * i + 1] = ff.state.quat[i].y; qo[4 * i + 2] = ff.state.quat[i].z; qo[4 * i + 3] = ff.state.quat[i].w;
    }
}

/// Eval → (E, f (n,3) = -∇E, tau (n,3), eatom (n,) consistency map).
#[no_mangle]
pub extern "C" fn raff3d_eval(h: *mut Raff3dFF, f: *mut f64, tau: *mut f64, eatom: *mut f64) -> f64 {
    let ff = unsafe { hf(h) };
    let n = ff.state.natoms;
    let e = ff.rx.eval_reactive(&ff.state, &ff.topo, &mut ff.fapos, &mut ff.tau);
    let fo = unsafe { wr_f64(f, 3 * n) };
    let to = unsafe { wr_f64(tau, 3 * n) };
    let ea = unsafe { wr_f64(eatom, n) };
    for i in 0..n {
        fo[3 * i] = ff.fapos[i].x; fo[3 * i + 1] = ff.fapos[i].y; fo[3 * i + 2] = ff.fapos[i].z;
        to[3 * i] = ff.tau[i].x; to[3 * i + 1] = ff.tau[i].y; to[3 * i + 2] = ff.tau[i].z;
        ea[i] = ff.rx.eatom[i];
    }
    e
}

/// Shared step body for raff3d_step/raff3d_relax: one damped-MD step, then an
/// optional traj frame when `traj.step % stride == 0` (step 0 writes a frame).
fn do_step(ff: &mut Raff3dFF) -> (f64, f64, f64) {
    let (e, mf, mt) = ff.rx.step_reactive_md(&mut ff.state, &ff.topo, &ff.cfg, &mut ff.fapos, &mut ff.tau);
    if let Some(t) = ff.traj.as_mut() {
        if t.step % t.stride == 0 {
            let mut s = String::with_capacity(ff.state.natoms * 40);
            write_xyz_frame(&mut s, &ff.state, &ff.topo, t.step, e).expect("do_step: write_xyz_frame fmt failed");
            t.f.write_all(s.as_bytes()).unwrap_or_else(|er| panic!("do_step: traj write failed at step {}: {er}", t.step));
        }
        t.step += 1;
    }
    (e, mf, mt)
}

/// One damped-MD step (eval + box + integrate_md via step_reactive_md).
/// Sets cfg.dt/cdamp/rot_damp first (cdamp drives both). Returns total E.
#[no_mangle]
pub extern "C" fn raff3d_step(h: *mut Raff3dFF, dt: f64, cdamp: f64) -> f64 {
    let ff = unsafe { hf(h) };
    assert!(dt.is_finite() && dt > 0.0, "raff3d_step: dt={dt} must be finite > 0");
    assert!(cdamp.is_finite(), "raff3d_step: cdamp={cdamp} non-finite");
    ff.cfg.dt = dt; ff.cfg.cdamp = cdamp; ff.cfg.rot_damp = cdamp;
    let (e, _, _) = do_step(ff);
    e
}

/// Damped-MD relax to fixpoint: converged when max|F| < tol_f && max|τ| < tol_t
/// (separate force vs torque thresholds). → (E_final, out_steps, out_conv).
/// Progress eprintln every 200 steps.
#[no_mangle]
pub extern "C" fn raff3d_relax(h: *mut Raff3dFF, nsteps: i32, dt: f64, cdamp: f64, tol_f: f64, tol_t: f64, out_steps: *mut i32, out_conv: *mut i32) -> f64 {
    assert!(nsteps > 0, "raff3d_relax: nsteps={nsteps} <= 0");
    assert!(tol_f.is_finite() && tol_f > 0.0 && tol_t.is_finite() && tol_t > 0.0, "raff3d_relax: tol_f={tol_f} tol_t={tol_t} must be finite > 0");
    let ff = unsafe { hf(h) };
    ff.cfg.dt = dt; ff.cfg.cdamp = cdamp; ff.cfg.rot_damp = cdamp;
    let (mut e, mut steps, mut conv) = (0.0f64, 0usize, false);
    for s in 0..nsteps as usize {
        let (es, mf, mt) = do_step(ff);
        assert!(es.is_finite(), "raff3d_relax: non-finite E at step {s}: E={es} max|F|={mf} max|τ|={mt}");
        e = es; steps = s + 1;
        if s % 200 == 0 { eprintln!("raff3d_relax step {s:5}: E={e:10.6}  max|F|={mf:.4e}  max|τ|={mt:.4e}"); }
        if mf < tol_f && mt < tol_t { conv = true; break; }
    }
    unsafe {
        if !out_steps.is_null() { *out_steps = steps as i32; }
        if !out_conv.is_null() { *out_conv = conv as i32; }
    }
    e
}

/// Attach a multi-frame XYZ trajectory writer (`{n}\nstep {s} E={e}\n` + `El
/// x y z` per atom; El from nport 4→C 3→N 2→O 1→H else X) — a frame is written
/// every `stride` completed steps (raff3d_step or inside raff3d_relax).
/// Fails loud if the file cannot be created. Replaces any previous traj.
#[no_mangle]
pub extern "C" fn raff3d_set_traj(h: *mut Raff3dFF, path: *const c_char, stride: i32) -> i32 {
    assert!(!path.is_null(), "raff3d_set_traj: null path");
    assert!(stride >= 1, "raff3d_set_traj: stride={stride} < 1");
    let ff = unsafe { hf(h) };
    let p = unsafe { std::ffi::CStr::from_ptr(path) }.to_str().expect("raff3d_set_traj: path not valid UTF-8");
    let f = std::fs::File::create(p).unwrap_or_else(|e| panic!("raff3d_set_traj: cannot create '{p}': {e}"));
    ff.traj = Some(TrajOut { f: std::io::BufWriter::new(f), stride: stride as usize, step: 0 });
    0
}

/// Detach the trajectory writer: flush + close. No-op if none attached.
#[no_mangle]
pub extern "C" fn raff3d_clear_traj(h: *mut Raff3dFF) {
    let ff = unsafe { hf(h) };
    if let Some(mut t) = ff.traj.take() {
        t.f.flush().unwrap_or_else(|e| panic!("raff3d_clear_traj: flush failed: {e}"));
    }
}

/// Geometric neighbor query: all pairs (i,j) with |x_j - x_i| < rc plus their
/// distances. Uses rx.grid when configured (rebuilt for current positions),
/// else a one-shot auto-fit Grid3 (AABB + cells of size rc). pairs (cap,2) i32
/// + dists (cap,) f64 out; returns TOTAL count found (cap/retry contract like
/// raff3d_emerged_bonds — pass null/0 to probe the count first).
#[no_mangle]
pub extern "C" fn raff3d_pairs_within(h: *mut Raff3dFF, rc: f64, pairs: *mut i32, dists: *mut f64, cap: i32) -> i32 {
    let ff = unsafe { hf(h) };
    let mut v = Vec::new();
    match ff.rx.grid.as_mut() {
        Some(g) => g.pairs_within(&ff.state.pos, rc, &mut v),
        None => neighbors_within(&ff.state.pos, rc, &mut v),
    }
    if !pairs.is_null() && !dists.is_null() {
        let cap = cap.max(0) as usize;
        let po = unsafe { std::slice::from_raw_parts_mut(pairs, 2 * cap) };
        let dd = unsafe { std::slice::from_raw_parts_mut(dists, cap) };
        for (k, &(i, j)) in v.iter().enumerate().take(cap) {
            po[2 * k] = i as i32; po[2 * k + 1] = j as i32;
            dd[k] = (ff.state.pos[j as usize] - ff.state.pos[i as usize]).norm();
        }
    }
    v.len() as i32
}

/// World-frame port dirs → dirs (n,4,3) f64, unused slots = 0. Re-projected
/// from state.quat every call (projectBonds) — safe at any time, no eval needed.
#[no_mangle]
pub extern "C" fn raff3d_get_ports(h: *mut Raff3dFF, dirs: *mut f64) {
    let ff = unsafe { hf(h) };
    let n = ff.state.natoms;
    let o = unsafe { wr_f64(dirs, 12 * n) };
    for v in o.iter_mut() { *v = 0.0; }
    project_ports(&ff.state, &ff.topo, &mut ff.rx.h_world);
    for i in 0..n {
        let np = ff.topo.nport[i] as usize;
        for s in 0..np {
            let hw = ff.rx.h_world[i * 4 + s];
            o[(i * 4 + s) * 3] = hw.x; o[(i * 4 + s) * 3 + 1] = hw.y; o[(i * 4 + s) * 3 + 2] = hw.z;
        }
    }
}

/// Ports per atom → out (n,) i32.
#[no_mangle]
pub extern "C" fn raff3d_nports(h: *mut Raff3dFF, out: *mut i32) {
    let ff = unsafe { hf(h) };
    assert!(!out.is_null(), "raff3d_nports: null out");
    let o = unsafe { std::slice::from_raw_parts_mut(out, ff.state.natoms) };
    for (i, v) in o.iter_mut().enumerate() { *v = ff.topo.nport[i] as i32; }
}

/// Uniform neighbor grid for eval. Domain origin + (nx,ny,nz)*dx; dx >= rcut
/// gives a strict 3x3x3 stencil. Call again if natom changes (reconfig).
#[no_mangle]
pub extern "C" fn raff3d_set_grid(h: *mut Raff3dFF, dx: f64, ox: f64, oy: f64, oz: f64, nx: i32, ny: i32, nz: i32) {
    assert!(dx.is_finite() && dx > 0.0 && nx > 0 && ny > 0 && nz > 0, "raff3d_set_grid: dx={dx} n=({nx},{ny},{nz})");
    unsafe { hf(h) }.rx.set_grid([ox, oy, oz], [nx as usize, ny as usize, nz as usize], dx);
}

/// Drop the neighbor grid → O(N²) reference loop.
#[no_mangle]
pub extern "C" fn raff3d_clear_grid(h: *mut Raff3dFF) { unsafe { hf(h) }.rx.grid = None; }

/// Point-probe energy at p=(x,y,z): Σ_i pair_energy(i, probe) where the probe
/// is a portless atom with params (r_cov, a, b) — repulsive wall + one-sided
/// port gate. Ports are re-projected from state.quat each call, so no prior
/// eval is required. skip >= 0 excludes atom `skip` (probe on an atom site).
#[no_mangle]
pub extern "C" fn raff3d_field_at(h: *mut Raff3dFF, x: f64, y: f64, z: f64, skip: i32, r_cov: f64, a: f64, b: f64) -> f64 {
    let ff = unsafe { hf(h) };
    assert!(r_cov > 0.0 && b > 0.0, "raff3d_field_at: probe r_cov={r_cov} b={b} must be > 0");
    let n = ff.state.natoms;
    assert!(skip < n as i32, "raff3d_field_at: skip={skip} >= natom={n}");
    let p = Vec3d::new(x, y, z);
    let pj = ReactParams { r_cov, a, b };
    project_ports(&ff.state, &ff.topo, &mut ff.rx.h_world);
    let mut e = 0.0;
    for i in 0..n {
        if i as i32 == skip { continue; }
        let npi = ff.topo.nport[i] as usize;
        if let Some((ep, _, _, _)) = pair_math(&ff.rx.cfg, &ff.rx.params[i], &ff.rx.h_world[i * 4..i * 4 + npi], ff.state.pos[i], &pj, &[], p) {
            e += ep;
        }
    }
    e
}

/// Emerged bonds: pairs (i,j) with pair energy < e_thr. pairs (cap,2) i32 out;
/// returns TOTAL count found (if > cap, caller reallocs and retries).
#[no_mangle]
pub extern "C" fn raff3d_emerged_bonds(h: *mut Raff3dFF, e_thr: f64, pairs: *mut i32, cap: i32) -> i32 {
    let ff = unsafe { hf(h) };
    let v = ff.rx.emerged_bonds(&ff.state, &ff.topo, e_thr);
    if !pairs.is_null() {
        let cap = cap.max(0) as usize;
        let o = unsafe { std::slice::from_raw_parts_mut(pairs, 2 * cap) };
        for (k, &(i, j, _)) in v.iter().enumerate().take(cap) { o[2 * k] = i as i32; o[2 * k + 1] = j as i32; }
    }
    v.len() as i32
}

/// Override per-atom reactive params (calibration): r_cov + Morse a, b.
/// NOTE: reconfigure drops the grid — re-run raff3d_set_grid if still wanted.
#[no_mangle]
pub extern "C" fn raff3d_set_param(h: *mut Raff3dFF, i: i32, r_cov: f64, a: f64, b: f64) {
    let ff = unsafe { hf(h) };
    let n = ff.state.natoms;
    assert!((i as usize) < n, "raff3d_set_param: i={i} >= natom={n}");
    assert!(r_cov.is_finite() && r_cov > 0.0 && b.is_finite() && b > 0.0, "raff3d_set_param: atom {i} r_cov={r_cov} b={b} must be finite > 0");
    let mut ps = ff.rx.params.clone();
    ps[i as usize] = ReactParams { r_cov, a, b };
    let m = ff.rx.cfg.pex_m;
    ff.rx.reconfigure(ps, m);
}

/// pex squaring count m — controls BOTH approx quality and true support
/// r0 + 2^m/(b_i+b_j). NOTE: drops the grid (see raff3d_set_param).
#[no_mangle]
pub extern "C" fn raff3d_set_pex_m(h: *mut Raff3dFF, m: i32) {
    assert!((0..32).contains(&m), "raff3d_set_pex_m: m={m} out of [0,32)");
    let ff = unsafe { hf(h) };
    let ps = ff.rx.params.clone();
    ff.rx.reconfigure(ps, m as u32);
}

/// Port-pair gate flag: 1 = full (ci·cj·cij)^4 (default, physical);
/// 0 = drop h_i·h_j factor (debug/ablation only).
#[no_mangle]
pub extern "C" fn raff3d_set_cij(h: *mut Raff3dFF, flag: i32) {
    unsafe { hf(h) }.rx.cfg.use_cij = flag != 0;
}

/// Harmonic AABB confinement cfg.box_cfg: F = k·(limit − x) outside [x0,x1].
/// Disabled when k <= 0 or min == max on all axes.
#[no_mangle]
pub extern "C" fn raff3d_set_box(h: *mut Raff3dFF, x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64, k: f64) {
    let ff = unsafe { hf(h) };
    let en = k > 0.0 && !(x0 == x1 && y0 == y1 && z0 == z1);
    ff.cfg.box_cfg = BoxCfg { enabled: en, min: Vec3d::new(x0, y0, z0), max: Vec3d::new(x1, y1, z1), k };
}
