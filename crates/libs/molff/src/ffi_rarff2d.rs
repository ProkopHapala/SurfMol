//! ffi_rarff2d.rs — extern "C" opaque-handle API exported by libmolff.so (cdylib).
//!
//! Contract: invPPAFM `export_invAFM/scripts/rarff2d_ffi.py` — all arrays are
//! numpy float64 C-contiguous; `types` is uint8 (0=sp2, 1=sp3, 2=H).
//! `zc` = unit-complex orientation (cosφ, sinφ), NOT angles.
//! Design doc: notes/designs/2026-10-07_bond2d_entity_pic_loss.md.
//!
//! NOTE: `unsafe` at this FFI boundary is unavoidable (raw C pointers); the
//! crate-wide convention "only OpenCL uses unsafe" is waived here by design —
//! every entry point asserts non-null + consistent lengths (fail loud).

use crate::rarff2d::{Bond2d, Rarff2d, Rarff2dType, Ring2d, TYPE_H, TYPE_SP2, TYPE_SP3};
use numtypes::Vec2d;

#[inline] unsafe fn hf(h: *mut Rarff2d) -> &'static mut Rarff2d {
    assert!(!h.is_null(), "rarff2d FFI: null handle");
    unsafe { &mut *h }
}
#[inline] unsafe fn rd_f64<'a>(p: *const f64, n: usize) -> &'a [f64] {
    assert!(!p.is_null(), "rarff2d FFI: null f64 array");
    unsafe { std::slice::from_raw_parts(p, n) }
}
#[inline] unsafe fn wr_f64<'a>(p: *mut f64, n: usize) -> &'a mut [f64] {
    assert!(!p.is_null(), "rarff2d FFI: null f64 out-array");
    unsafe { std::slice::from_raw_parts_mut(p, n) }
}

/// n atoms, types u8 (0=sp2,1=sp3,2=H). Positions init to origin — call
/// rarff2d_set_state before eval.
#[no_mangle]
pub extern "C" fn rarff2d_new(n: i32, types: *const u8) -> *mut Rarff2d {
    assert!(n > 0, "rarff2d_new: n={n} <= 0");
    assert!(!types.is_null(), "rarff2d_new: null types");
    let n = n as usize;
    let ty = unsafe { std::slice::from_raw_parts(types, n) };
    let types: Vec<Rarff2dType> = ty.iter().map(|&t| match t { 1 => TYPE_SP3, 2 => TYPE_H, _ => TYPE_SP2 }).collect();
    Box::into_raw(Box::new(Rarff2d::new(types, vec![Vec2d::new(0.0, 0.0); n])))
}

#[no_mangle]
pub extern "C" fn rarff2d_free(h: *mut Rarff2d) {
    if !h.is_null() { drop(unsafe { Box::from_raw(h) }); }
}

/// pos (n,2) f64, zc (n,2) unit-complex — renormalized on ingest (fail-loud on degenerate).
#[no_mangle]
pub extern "C" fn rarff2d_set_state(h: *mut Rarff2d, pos: *const f64, zc: *const f64) {
    let ff = unsafe { hf(h) };
    let n = ff.natom();
    let p = unsafe { rd_f64(pos, 2 * n) };
    let z = unsafe { rd_f64(zc, 2 * n) };
    for i in 0..n {
        ff.pos[i].set(p[2 * i], p[2 * i + 1]);
        let nz = Vec2d::new(z[2 * i], z[2 * i + 1]).norm();
        assert!(nz.is_finite() && nz > 1e-12, "rarff2d_set_state: degenerate zc at atom {i}: ({},{})", z[2 * i], z[2 * i + 1]);
        ff.zc[i].set(z[2 * i] / nz, z[2 * i + 1] / nz);
    }
}

/// Eval → (E, f (n,2) = -∇E, torq (n,), eatom (n,) consistency map).
#[no_mangle]
pub extern "C" fn rarff2d_eval(h: *mut Rarff2d, f: *mut f64, torq: *mut f64, eatom: *mut f64) -> f64 {
    let ff = unsafe { hf(h) };
    let e = ff.eval();
    let n = ff.natom();
    let fo = unsafe { wr_f64(f, 2 * n) };
    let tq = unsafe { wr_f64(torq, n) };
    let ea = unsafe { wr_f64(eatom, n) };
    for i in 0..n { fo[2 * i] = ff.force[i].x; fo[2 * i + 1] = ff.force[i].y; tq[i] = ff.torq[i]; ea[i] = ff.eatom[i]; }
    e
}

/// Orientations relaxed at frozen positions, then eval → E_min(φ), F = -dE_min/dpos.
#[no_mangle]
pub extern "C" fn rarff2d_eval_oriented(h: *mut Rarff2d, iters: i32, dt: f64, dphi: f64, f: *mut f64, eatom: *mut f64) -> f64 {
    assert!(iters >= 0, "rarff2d_eval_oriented: iters={iters} < 0");
    let ff = unsafe { hf(h) };
    let e = ff.eval_oriented(iters as usize, dt, dphi);
    let n = ff.natom();
    let fo = unsafe { wr_f64(f, 2 * n) };
    let ea = unsafe { wr_f64(eatom, n) };
    for i in 0..n { fo[2 * i] = ff.force[i].x; fo[2 * i + 1] = ff.force[i].y; ea[i] = ff.eatom[i]; }
    e
}

/// One overdamped-GD repair step (trust region dmax / dphi).
#[no_mangle]
pub extern "C" fn rarff2d_step_gd(h: *mut Rarff2d, dt: f64, dmax: f64, dphi: f64) {
    unsafe { hf(h) }.step_gd(dt, dmax, dphi);
}

/// Entity DOF step (overdamped GD, trust region): ring pos/orientation +
/// bond midpoint/axis/half-length. Prefer rarff2d_relax_entities (damped MD)
/// for convergence — pure GD creeps asymptotically near the minimum.
#[no_mangle]
pub extern "C" fn rarff2d_step_entities(h: *mut Rarff2d, dt: f64, dmax: f64, dphi: f64, dl: f64) {
    unsafe { hf(h) }.step_entities(dt, dmax, dphi, dl);
}

/// One damped-MD step on entity DOFs (same scheme as atom step_md).
#[no_mangle]
pub extern "C" fn rarff2d_step_entities_md(h: *mut Rarff2d, dt: f64, cdamp: f64) {
    unsafe { hf(h) }.step_entities_md(dt, cdamp);
}

/// Relax entity DOFs by damped MD until entity residual < tol.
/// move_atoms!=0: atoms step_md too (joint relax); 0: atoms pinned
/// (entities snap onto the atom layout). -> (E_final, out_steps, out_conv).
#[no_mangle]
pub extern "C" fn rarff2d_relax_entities(h: *mut Rarff2d, nsteps: i32, dt: f64, cdamp: f64, tol: f64, move_atoms: i32, out_steps: *mut i32, out_conv: *mut i32) -> f64 {
    assert!(nsteps > 0, "rarff2d_relax_entities: nsteps={nsteps} <= 0");
    let ff = unsafe { hf(h) };
    let (e, n, conv) = ff.relax_entities(nsteps as usize, dt, cdamp, tol, move_atoms != 0, false);
    unsafe {
        if !out_steps.is_null() { *out_steps = n as i32; }
        if !out_conv.is_null() { *out_conv = conv as i32; }
    }
    e
}

/// Damped-MD relax to fixpoint → (E_final, out_steps, out_conv).
#[no_mangle]
pub extern "C" fn rarff2d_relax(h: *mut Rarff2d, nsteps: i32, dt: f64, cdamp: f64, tol: f64, out_steps: *mut i32, out_conv: *mut i32) -> f64 {
    assert!(nsteps > 0, "rarff2d_relax: nsteps={nsteps} <= 0");
    let ff = unsafe { hf(h) };
    let (e, n, conv) = ff.relax(nsteps as usize, dt, cdamp, tol, false);
    unsafe {
        if !out_steps.is_null() { *out_steps = n as i32; }
        if !out_conv.is_null() { *out_conv = conv as i32; }
    }
    e
}

#[no_mangle]
pub extern "C" fn rarff2d_get_state(h: *mut Rarff2d, pos: *mut f64, zc: *mut f64) {
    let ff = unsafe { hf(h) };
    let n = ff.natom();
    let po = unsafe { wr_f64(pos, 2 * n) };
    let zo = unsafe { wr_f64(zc, 2 * n) };
    for i in 0..n { po[2 * i] = ff.pos[i].x; po[2 * i + 1] = ff.pos[i].y; zo[2 * i] = ff.zc[i].x; zo[2 * i + 1] = ff.zc[i].y; }
}

/// pex squaring count m — controls BOTH approx quality and true support r0+2^m/b.
#[no_mangle]
pub extern "C" fn rarff2d_set_pex_m(h: *mut Rarff2d, m: i32) {
    unsafe { hf(h) }.set_pex_m(m as u32);
}

/// Override per-atom type params (calibration): (nfold, r_cov, a, b, w).
#[no_mangle]
pub extern "C" fn rarff2d_set_type(h: *mut Rarff2d, i: i32, nfold: i32, r_cov: f64, a: f64, b: f64, w: f64) {
    let ff = unsafe { hf(h) };
    assert!((i as usize) < ff.natom(), "rarff2d_set_type: i={i} >= natom={}", ff.natom());
    ff.types[i as usize] = Rarff2dType { nfold: nfold as u32, r_cov, a, b, w };
}

/// Append a ring entity (hexagon/pentagon constraint). Default site params as
/// the rarff2d_view scene DSL. Returns ring index.
#[no_mangle]
pub extern "C" fn rarff2d_add_ring(h: *mut Rarff2d, x: f64, y: f64, nfold: i32, r: f64) -> i32 {
    let ff = unsafe { hf(h) };
    let mut rings = ff.rings.clone();
    rings.push(Ring2d { pos: Vec2d::new(x, y), zc: Vec2d::new(1.0, 0.0), nfold: nfold as u32, radius: r, a: 2.0, h: 0.5, a_core: 1.5, r0_core: r - 0.6, b_core: 2.5 });
    let idx = rings.len() as i32 - 1;
    ff.set_rings(rings);
    idx
}

/// Append a bond entity from an endpoint pair (img2mol codec convention:
    /// endpoints e1,e2; internally midpoint/axis/half). a,b,w = Morse depth,
/// steepness, axis-gate width of the atom–bond potential. Returns bond index.
#[no_mangle]
pub extern "C" fn rarff2d_add_bond(h: *mut Rarff2d, x1: f64, y1: f64, x2: f64, y2: f64, a: f64, b: f64, w: f64) -> i32 {
    let ff = unsafe { hf(h) };
    let mut bonds = ff.bonds.clone();
    bonds.push(Bond2d::from_ends(Vec2d::new(x1, y1), Vec2d::new(x2, y2), a, b, w));
    let idx = bonds.len() as i32 - 1;
    ff.set_bonds(bonds);
    idx
}

#[no_mangle]
pub extern "C" fn rarff2d_clear_bonds(h: *mut Rarff2d) { unsafe { hf(h) }.set_bonds(Vec::new()); }

#[no_mangle]
pub extern "C" fn rarff2d_clear_rings(h: *mut Rarff2d) { unsafe { hf(h) }.set_rings(Vec::new()); }

/// Per-endpoint consistency readout — call AFTER eval/eval_oriented.
/// esite (nb,2): site energy per bond end (≈0 ⇒ bare end = no atom).
/// fdof  (nb,4): generalized forces (fx, fy, torque, stretch) on bond DOFs.
#[no_mangle]
pub extern "C" fn rarff2d_bond_report(h: *mut Rarff2d, esite: *mut f64, fdof: *mut f64) {
    let ff = unsafe { hf(h) };
    let nb = ff.bonds.len();
    let es = unsafe { wr_f64(esite, 2 * nb) };
    let fd = unsafe { wr_f64(fdof, 4 * nb) };
    for ib in 0..nb {
        es[2 * ib] = ff.bond_esite[ib][0]; es[2 * ib + 1] = ff.bond_esite[ib][1];
        fd[4 * ib] = ff.bond_f[ib].x; fd[4 * ib + 1] = ff.bond_f[ib].y;
        fd[4 * ib + 2] = ff.bond_t[ib]; fd[4 * ib + 3] = ff.bond_fh[ib];
    }
}

/// Decode bond entities back to endpoint pairs: out (nb,4) = (x1,y1,x2,y2).
#[no_mangle]
pub extern "C" fn rarff2d_get_bonds(h: *mut Rarff2d, out: *mut f64) -> i32 {
    let ff = unsafe { hf(h) };
    let nb = ff.bonds.len();
    let o = unsafe { wr_f64(out, 4 * nb) };
    for (ib, bd) in ff.bonds.iter().enumerate() {
        let (e0, e1) = bd.ends();
        o[4 * ib] = e0.x; o[4 * ib + 1] = e0.y; o[4 * ib + 2] = e1.x; o[4 * ib + 3] = e1.y;
    }
    nb as i32
}

/// Ring entities: out (nr,5) = (x, y, nfold, radius, phi).
#[no_mangle]
pub extern "C" fn rarff2d_get_rings(h: *mut Rarff2d, out: *mut f64) -> i32 {
    let ff = unsafe { hf(h) };
    let nr = ff.rings.len();
    let o = unsafe { wr_f64(out, 5 * nr) };
    for (ir, rg) in ff.rings.iter().enumerate() {
        o[5 * ir] = rg.pos.x; o[5 * ir + 1] = rg.pos.y; o[5 * ir + 2] = rg.nfold as f64;
        o[5 * ir + 3] = rg.radius; o[5 * ir + 4] = rg.zc.y.atan2(rg.zc.x);
    }
    nr as i32
}

/// Uniform neighbor grid for eval (~15x at large N). Domain origin+(nx,ny)*dx.
/// Call again whenever natom changes. Disables groups (and vice versa).
#[no_mangle]
pub extern "C" fn rarff2d_set_grid(h: *mut Rarff2d, dx: f64, ox: f64, oy: f64, nx: i32, ny: i32) {
    assert!(dx > 0.0 && nx > 0 && ny > 0, "rarff2d_set_grid: dx={dx} nx={nx} ny={ny}");
    unsafe { hf(h) }.set_grid(dx, [ox, oy], nx as usize, ny as usize);
}

/// Probe-atom field at (x,y): energy an optimally-oriented probe of the given
/// type would feel (atoms except `skip`, + rings + bond wells). "Is there /
/// should there be an atom here" oracle.
#[no_mangle]
pub extern "C" fn rarff2d_field_at(h: *mut Rarff2d, x: f64, y: f64, skip: i32, nfold: i32, r_cov: f64, a: f64, b: f64, w: f64) -> f64 {
    let ff = unsafe { hf(h) };
    let tp = Rarff2dType { nfold: nfold as u32, r_cov, a, b, w };
    let sk = if skip < 0 { usize::MAX } else { skip as usize };   // -1 = no atom skipped
    ff.field_at(Vec2d::new(x, y), sk, tp)
}

/// Interaction-channel mask: bit0 atom-atom pairs, bit1 ring-atom,
/// bit2 atom-bond, bit3 ring-bond, bit4 bond-bond. Default 0b11111 (all on).
#[no_mangle]
pub extern "C" fn rarff2d_set_inter(h: *mut Rarff2d, mask: i32) {
    let ff = unsafe { hf(h) };
    ff.inter.atom_pair = mask & 1 != 0;
    ff.inter.ring_atom = mask & 2 != 0;
    ff.inter.bond_atom = mask & 4 != 0;
    ff.inter.ring_bond = mask & 8 != 0;
    ff.inter.bond_bond = mask & 16 != 0;
}

/// Atom-bond angular-gate mask: bit0 atom-side gate g_i, bit1 bond-side gate
/// g_b (endpoint lobes). Default 3; 0 = pure radial Morse to midpoint.
#[no_mangle]
pub extern "C" fn rarff2d_set_bond_gate(h: *mut Rarff2d, mode: i32) {
    unsafe { hf(h) }.bond_gate = mode as u8;
}

/// Soft confinement arena E = k(r-R)^2 for |x|>R; r_arena<=0 disables.
#[no_mangle]
pub extern "C" fn rarff2d_set_arena(h: *mut Rarff2d, r_arena: f64, k: f64) {
    let ff = unsafe { hf(h) };
    ff.arena = if r_arena > 0.0 { Some((r_arena, k)) } else { None };
}

/// Emerged bonds: pairs (i,j) with pair energy < e_thr. pairs (cap,2) i32
/// out; returns TOTAL count found (if > cap, caller reallocs and retries).
#[no_mangle]
pub extern "C" fn rarff2d_emerged_bonds(h: *mut Rarff2d, e_thr: f64, pairs: *mut i32, cap: i32) -> i32 {
    let ff = unsafe { hf(h) };
    let v = ff.emerged_bonds(e_thr);
    if !pairs.is_null() {
        let o = unsafe { std::slice::from_raw_parts_mut(pairs, 2 * (cap.max(0) as usize)) };
        for (k, &(i, j, _)) in v.iter().enumerate().take(cap.max(0) as usize) {
            o[2 * k] = i as i32; o[2 * k + 1] = j as i32;
        }
    }
    v.len() as i32
}
