//! ffi_uff.rs — extern "C" opaque-handle API for explicit-bond UFF (3D, f64).
//!
//! Contract: invPPAFM `export_invAFM/scripts/uff_ffi.py` — numpy float64
//! C-contiguous; elements as CSV string; bonds as (nb,2) int32.
//! Build order mirrors editor/molengine: make_neigh_bs → topology builders →
//! Uff::new → assign_uff_types → Params (.dat) → setup_params → bake_*_neighs
//! → map_atom_interactions. Bonded terms only (no nonbonded — see
//! surfmol_forcefields.md §8.2 caveats).

use crate::uff::Uff;
use moltopo::assign_uff::assign_uff_types;
use moltopo::molecular::DynamicAtoms;
use moltopo::params::Params;
use moltopo::topology::{build_angles_from_bonds, build_dihedrals_from_bonds, build_inversions_from_bonds};
use numtypes::Vec3d;
use std::ffi::CStr;
use std::path::Path;

/// One UFF system: forcefield + dynamic state (apos/fapos/vapos) + type names.
pub struct UffSys { pub uff: Uff, pub da: DynamicAtoms, pub types: Vec<String> }

#[inline] unsafe fn hs(h: *mut UffSys) -> &'static mut UffSys {
    assert!(!h.is_null(), "uff FFI: null handle");
    unsafe { &mut *h }
}
#[inline] unsafe fn rd_f64<'a>(p: *const f64, n: usize) -> &'a [f64] {
    assert!(!p.is_null(), "uff FFI: null f64 array");
    unsafe { std::slice::from_raw_parts(p, n) }
}
#[inline] unsafe fn wr_f64<'a>(p: *mut f64, n: usize) -> &'a mut [f64] {
    assert!(!p.is_null(), "uff FFI: null f64 out-array");
    unsafe { std::slice::from_raw_parts_mut(p, n) }
}
#[inline] unsafe fn cstr(p: *const std::os::raw::c_char) -> &'static str {
    assert!(!p.is_null(), "uff FFI: null cstr");
    unsafe { CStr::from_ptr(p) }.to_str().expect("uff FFI: elems/data_dir not UTF-8")
}

/// n atoms, elems_csv "C,C,H,...", bonds (nb,2) i32, pos (n,3) f64,
/// data_dir with the five UFF *.dat tables. Returns opaque handle.
#[no_mangle]
pub extern "C" fn uff_new(n: i32, elems_csv: *const std::os::raw::c_char, bonds: *const i32, nbonds: i32, pos: *const f64, data_dir: *const std::os::raw::c_char) -> *mut UffSys {
    assert!(n > 0, "uff_new: n={n} <= 0");
    assert!(nbonds >= 0, "uff_new: nbonds={nbonds} < 0");
    let n = n as usize;
    let elems: Vec<String> = unsafe { cstr(elems_csv) }.split(',').map(|s| s.trim().to_string()).collect();
    assert_eq!(elems.len(), n, "uff_new: elems.len()={} != n={}", elems.len(), n);
    let bonds: Vec<[i32; 2]> = if nbonds > 0 {
        assert!(!bonds.is_null(), "uff_new: nbonds={nbonds} but null bonds");
        unsafe { std::slice::from_raw_parts(bonds, 2 * nbonds as usize) }
            .chunks_exact(2).map(|c| [c[0], c[1]]).collect()
    } else { Vec::new() };
    let pv = unsafe { rd_f64(pos, 3 * n) };
    let apos: &[Vec3d] = bytemuck::cast_slice(pv);

    let mut da = DynamicAtoms::new(n as i32);
    da.atoms.apos.as_mut_slice().copy_from_slice(apos);
    da.atoms.make_neigh_bs(&bonds);
    let angles = build_angles_from_bonds(n as i32, &bonds);
    let dihs = build_dihedrals_from_bonds(&bonds);
    let invs = build_inversions_from_bonds(n as i32, &bonds);
    let mut uff = Uff::new(n as i32, &bonds, &angles, &dihs, &invs);

    let ng4: Vec<[i32; 4]> = da.neighs().iter().map(|q| q.as_array()).collect();
    let types = assign_uff_types(&elems, &ng4);
    let dir = Path::new(unsafe { cstr(data_dir) });
    let mut params = Params::new();
    params.load_element_types(dir.join("ElementTypes.dat"));
    params.load_atom_types(dir.join("AtomTypes.dat"));
    params.load_bond_types(dir.join("BondTypes.dat"));
    params.load_angle_types(dir.join("AngleTypes.dat"));
    params.load_dihedral_types(dir.join("DihedralTypes.dat"));
    uff.setup_params(&params, &types, da.neighs());
    uff.bake_angle_neighs(da.neighs());
    uff.bake_dihedral_neighs(da.neighs());
    uff.bake_inversion_neighs(da.neighs());
    uff.map_atom_interactions();
    Box::into_raw(Box::new(UffSys { uff, da, types }))
}

#[no_mangle]
pub extern "C" fn uff_free(h: *mut UffSys) {
    if !h.is_null() { drop(unsafe { Box::from_raw(h) }); }
}

#[no_mangle]
pub extern "C" fn uff_set_pos(h: *mut UffSys, pos: *const f64) {
    let s = unsafe { hs(h) };
    let pv = unsafe { rd_f64(pos, 3 * s.da.natoms()) };
    s.da.atoms.apos.as_mut_slice().copy_from_slice(bytemuck::cast_slice(pv));
    s.da.clean_velocity();   // stale velocities after teleport would inject phantom momentum
}

/// Eval → (E, f (n,3), terms (4: eb,ea,ed,ei)).
#[no_mangle]
pub extern "C" fn uff_eval(h: *mut UffSys, f: *mut f64, terms: *mut f64) -> f64 {
    let s = unsafe { hs(h) };
    let n = s.da.natoms();
    let (eb, ea, ed, ei) = s.uff.eval_forces(s.da.atoms.apos.as_slice(), s.da.fapos.as_mut_slice(), s.da.atoms.neighs.as_slice(), s.da.atoms.neigh_bs.as_slice());
    let fo = unsafe { wr_f64(f, 3 * n) };
    let to = unsafe { wr_f64(terms, 4) };
    let fs: &[f64] = bytemuck::cast_slice(s.da.fapos.as_slice());
    fo.copy_from_slice(fs);
    to[0] = eb; to[1] = ea; to[2] = ed; to[3] = ei;
    eb + ea + ed + ei
}

#[no_mangle]
pub extern "C" fn uff_get_pos(h: *mut UffSys, pos: *mut f64) {
    let s = unsafe { hs(h) };
    let n = s.da.natoms();
    let po = unsafe { wr_f64(pos, 3 * n) };
    po.copy_from_slice(bytemuck::cast_slice(s.da.atoms.apos.as_slice()));
}

/// Assigned UFF type names as CSV into buf → returns bytes written (0 if truncated).
#[no_mangle]
pub extern "C" fn uff_get_types(h: *mut UffSys, buf: *mut std::os::raw::c_char, nbuf: i32) -> i32 {
    let s = unsafe { hs(h) };
    let csv = s.types.join(",");
    let bytes = csv.as_bytes();
    if bytes.len() + 1 > nbuf as usize { return 0; }   // fail: caller sees 0 → buffer too small
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, bytes.len());
        *(buf as *mut u8).add(bytes.len()) = 0;
    }
    bytes.len() as i32
}

/// Damped-MD relax (same loop as DynamicAtoms::run_md / MolWorld::run_md):
/// eval → move (flim clamp, cdamp) → v·f<0 resets velocity → conv at Σf²<tol².
#[no_mangle]
pub extern "C" fn uff_relax(h: *mut UffSys, nsteps: i32, dt: f64, cdamp: f64, flim: f64, tol: f64, out_steps: *mut i32, out_conv: *mut i32) -> f64 {
    assert!(nsteps > 0, "uff_relax: nsteps={nsteps} <= 0");
    let s = unsafe { hs(h) };
    let mut e = 0.0;
    let mut conv = 0i32;
    let mut steps = nsteps;
    let f2c = tol * tol;
    for it in 0..nsteps {
        let (eb, ea, ed, ei) = s.uff.eval_forces(s.da.atoms.apos.as_slice(), s.da.fapos.as_mut_slice(), s.da.atoms.neighs.as_slice(), s.da.atoms.neigh_bs.as_slice());
        e = eb + ea + ed + ei;
        let (mut ff, mut vf) = (0.0, 0.0);
        for i in 0..s.da.natoms() {
            let (ff_, _vv, f2_) = s.da.move_atom_md(i, dt, flim, cdamp);
            ff += ff_; vf += f2_;
        }
        if ff < 0.0 { s.da.clean_velocity(); }
        if vf < f2c { conv = 1; steps = it + 1; break; }
    }
    unsafe {
        if !out_steps.is_null() { *out_steps = steps; }
        if !out_conv.is_null() { *out_conv = conv; }
    }
    e
}
