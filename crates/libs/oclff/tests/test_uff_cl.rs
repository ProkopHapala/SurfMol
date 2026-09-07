//! UFF OpenCL parity test vs `molff::uff::Uff` CPU reference.

use molff::uff::Uff;
use oclff::UffOcl;
use numtypes::{Quat4i, Vec3d};

#[test]
fn uff_bond_parity_h2() {
    let natoms = 2i32;
    let k = 100.0f64;
    let l0 = 0.9f64;
    let apos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(1.0, 0.0, 0.0)];

    let mut uff = Uff::new(natoms, &[[0, 1]], &[], &[], &[]);
    uff.bon_params.as_mut_slice()[0] = [k, l0];

    let neighs = vec![Quat4i::new(1, -1, -1, -1), Quat4i::new(0, -1, -1, -1)];
    let neigh_bs = vec![Quat4i::new(0, -1, -1, -1), Quat4i::new(0, -1, -1, -1)];
    let mut fcpu = vec![Vec3d::default(); natoms as usize];
    // `eval_forces` needs a2f assembly map even for bond-only; call `eval_atom_bonds` directly.
    for ia in 0..natoms as usize {
        uff.eval_atom_bonds(ia, &apos, &mut fcpu, &neighs, &neigh_bs);
    }

    let dl = 1.0 - l0;
    let fx = 2.0 * k * dl;
    let e_atom = k * dl * dl;

    // GPU: one bond, no PBC, no NB subtraction
    let apos_f32: Vec<[f32; 4]> = apos.iter().map(|v| [v.x as f32, v.y as f32, v.z as f32, 0.0]).collect();
    let ucl = UffOcl::new().expect("OpenCL device/context init");
    let fgpu = ucl
        .eval_bonds(
            &apos_f32,
            &[[0, 1]],
            &[[k as f32, l0 as f32]],
            &[[1, -1, -1, -1], [0, -1, -1, -1]],
            &[[0, -1, -1, -1], [0, -1, -1, -1]], // neighCell: all PBC index 0 (zero shift)
            &[[0, -1, -1, -1], [0, -1, -1, -1]], // neighBs
            &[[0.0, 0.0, 0.0, 0.0]],
            &[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], // REQs not used
            0.0,                                                  // Rdamp
            1e6,                                                  // FmaxNonBonded
            0,                                                    // bSubtractBondNonBond
        )
        .expect("eval_bonds kernel dispatch");

    let tol = 1e-4f64;
    assert!((fgpu[0][0] as f64 - fx).abs() < tol, "atom 0 fx: gpu={} cpu={}", fgpu[0][0], fx);
    assert!((fgpu[1][0] as f64 + fx).abs() < tol, "atom 1 fx: gpu={} cpu={}", fgpu[1][0], -fx);
    assert!((fgpu[0][3] as f64 - e_atom).abs() < tol, "atom 0 E: gpu={} cpu={}", fgpu[0][3], e_atom);
    assert!((fgpu[1][3] as f64 - e_atom).abs() < tol, "atom 1 E: gpu={} cpu={}", fgpu[1][3], e_atom);

    assert!(fgpu[0][1].abs() < tol as f32 && fgpu[0][2].abs() < tol as f32, "atom 0 non-bond force components: {:?}", fgpu[0]);
    assert!(fgpu[1][1].abs() < tol as f32 && fgpu[1][2].abs() < tol as f32, "atom 1 non-bond force components: {:?}", fgpu[1]);
}
