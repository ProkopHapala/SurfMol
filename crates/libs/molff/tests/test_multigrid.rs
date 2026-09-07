//! Multigrid solver parity + convergence tests.
//!
//! T1: matvec parity (TrussOp::matvec vs dense A·x)
//! T2: diagonal-block parity (TrussOp::diagonal_blocks vs dense diagonal)
//! T3: direct-solve parity (multigrid V-cycle vs dense Gaussian elimination)
//! T4: convergence (multigrid vs plain Jacobi, residual curves)
//! T5: cached coarse-force step parity with Galerkin correction
//! T6: fitted modal quadratic parity with analytical coupled modes
//! T7: bend/twist mode orthonormality
//!
//! Per AGENTS.md §Tests Are Diagnostics: on failure, prints per-atom residuals
//! and the worst contributor so the bug is locatable without re-running.

use molff::multigrid::*;
use numtypes::Vec3d;

/// Small 5-atom chain: 0-1-2-3-4 along x, bond stiffness k=100, mass=1, dt=0.02.
/// Fixed endpoints (atoms 0 and 4) → 3 free atoms, 9 free DOFs.
fn chain5() -> (TrussOp, Vec<Vec3d>, Vec<bool>) {
    let apos = vec![
        Vec3d::new(0.0, 0.0, 0.0),
        Vec3d::new(1.0, 0.0, 0.0),
        Vec3d::new(2.0, 0.0, 0.0),
        Vec3d::new(3.0, 0.0, 0.0),
        Vec3d::new(4.0, 0.0, 0.0),
    ];
    let bonds = [[0,1],[1,2],[2,3],[3,4]];
    let k = vec![100.0; 4];
    let dt = 0.02;
    let mass_dt2 = vec![1.0/dt/dt; 5];
    // Fix endpoints with huge mass
    let mut mass_dt2 = mass_dt2;
    mass_dt2[0] = 1e8;
    mass_dt2[4] = 1e8;
    let free_mask = vec![true, true, true, true, true];  // all free; penalty handles fixed
    // Actually mark fixed nodes as not-free for the smoother
    let free_mask = vec![false, true, true, true, false];
    let op = TrussOp::from_bonds(&bonds, &k, &apos, &mass_dt2);
    (op, apos, free_mask)
}

#[test]
fn t1_matvec_parity() {
    let (op, _apos, _free) = chain5();
    let n = op.natoms * 3;
    let a_dense = op.assemble_dense();
    // Random x
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let ax_op = op.matvec(&x);
    // Dense A·x
    let mut ax_dense = vec![0.0f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n { s += a_dense[i*n + j] * x[j]; }
        ax_dense[i] = s;
    }
    let mut max_err = 0.0;
    let mut worst = 0;
    for i in 0..n {
        let e = (ax_op[i] - ax_dense[i]).abs();
        if e > max_err { max_err = e; worst = i; }
    }
    assert!(max_err < 1e-6, "t1_matvec_parity: max|matvec - dense|={max_err:.3e} at dof {worst} (n={n}).\n  matvec={ax_op:?}\n  dense ={ax_dense:?}");
    println!("[t1_matvec_parity] OK: max|matvec - dense|={max_err:.3e} (n={n} DOF)");
}

#[test]
fn t2_diagonal_parity() {
    let (op, _apos, _free) = chain5();
    let n = op.natoms * 3;
    let a_dense = op.assemble_dense();
    let d = op.diagonal_blocks();
    let mut max_err = 0.0;
    let mut worst = (0, 0);
    for i in 0..op.natoms {
        for a in 0..3 {
            for b in 0..3 {
                let exp = a_dense[(i*3+a)*n + (i*3+b)];
                let got = d[i][a*3 + b];
                let e = (got - exp).abs();
                if e > max_err { max_err = e; worst = (i, a*3+b); }
            }
        }
    }
    assert!(max_err < 1e-10, "t2_diagonal_parity: max|diag - dense_diag|={max_err:.3e} at atom {} block {}", worst.0, worst.1);
    println!("[t2_diagonal_parity] OK: max|diag - dense_diag|={max_err:.3e}");
}

#[test]
fn t3_direct_solve_parity() {
    let (op, _apos, free_mask) = chain5();
    let n = op.natoms * 3;
    let a_dense = op.assemble_dense();
    // RHS: gravity-like load on y
    let b: Vec<f64> = (0..op.natoms).flat_map(|i| {
        if free_mask[i] { vec![0.0, -9.81, 0.0] } else { vec![0.0, 0.0, 0.0] }
    }).collect();
    // Direct solve
    let x_direct = dense_solve(&a_dense, &b, n);
    // Multigrid: 4 pivots (atoms 1,2,3 are free; pick 2 pivots → 6 coarse DOF)
    let bonds = [[0,1],[1,2],[2,3],[3,4]];
    let pivots = select_pivots_maximin(&bonds, op.natoms, 3, &free_mask);
    println!("[t3] pivots = {pivots:?}");
    let p = build_pivot_prolongation(&_apos_placeholder(&op), &pivots, 2.0, &free_mask);
    let n_coarse = pivots.len() * 3;
    let x0 = vec![0.0; n];
    let (x_mg, res, _cres, _a_c) = solve_multigrid(&op, &p, n_coarse, &b, &x0, &free_mask,
                                       0.8, 0.0, 3, 3, 200, 1e-10);
    println!("[t3] MG residuals: first={:.3e}, last={:.3e}, n_cycles={}", res[0], res[res.len()-1], res.len()-1);
    // Compare solutions on free DOFs
    let mut max_err = 0.0;
    let mut worst = 0;
    for i in 0..op.natoms {
        if !free_mask[i] { continue; }
        for d in 0..3 {
            let idx = i*3 + d;
            let e = (x_mg[idx] - x_direct[idx]).abs();
            if e > max_err { max_err = e; worst = idx; }
        }
    }
    assert!(max_err < 1e-5, "t3_direct_solve_parity: max|mg - direct|={max_err:.3e} at dof {worst}.\n  mg     ={x_mg:?}\n  direct ={x_direct:?}");
    println!("[t3_direct_solve_parity] OK: max|mg - direct|={max_err:.3e} (residual converged to {:.3e} in {} cycles)", res[res.len()-1], res.len()-1);
}

#[test]
fn t4_convergence_vs_jacobi() {
    // 2D triangular grid — the canonical case where Jacobi stalls on low-frequency modes.
    // Ported from demo_MultiGrid.main_grid geometry. nx×ny grid with bottom row fixed.
    let (nx, ny) = (8, 8);
    let natoms = nx * ny;
    let a_grid = 1.0;
    // Positions on a 2D grid (z=0), with triangular diagonals
    let apos: Vec<Vec3d> = (0..natoms).map(|i| {
        let ix = i % nx; let iy = i / nx;
        Vec3d::new(ix as f64 * a_grid, iy as f64 * a_grid, 0.0)
    }).collect();
    // Edges: horizontal, vertical, and one diagonal per cell (i, i+nx+1)
    let mut bonds: Vec<[i32;2]> = Vec::new();
    for iy in 0..ny {
        for ix in 0..nx {
            let i = (iy * nx + ix) as i32;
            if ix + 1 < nx { bonds.push([i, i + 1]); }              // horizontal
            if iy + 1 < ny { bonds.push([i, i + nx as i32]); }      // vertical
            if ix + 1 < nx && iy + 1 < ny { bonds.push([i, i + nx as i32 + 1]); }  // diagonal
        }
    }
    let k = vec![20000.0; bonds.len()];
    let dt = 0.02;
    let mut mass_dt2 = vec![1.0/dt/dt; natoms];
    // Fix bottom row (iy=0) — moderate penalty (1000×, matching reference demo_MultiGrid)
    let free_mask: Vec<bool> = (0..natoms).map(|i| (i / nx) != 0).collect();
    for i in 0..natoms { if !free_mask[i] { mass_dt2[i] *= 1000.0; } }
    let op = TrussOp::from_bonds(&bonds, &k, &apos, &mass_dt2);
    let n = op.natoms * 3;
    // RHS: gravity-like load on y for free nodes
    let b: Vec<f64> = (0..natoms).flat_map(|i| {
        if free_mask[i] { vec![0.0, -9.81, 0.0] } else { vec![0.0, 0.0, 0.0] }
    }).collect();

    // Plain Jacobi: 1000 iterations
    let d = op.diagonal_blocks();
    let dinv = invert_3x3_blocks(&d);
    let mut x_jac = vec![0.0; n];
    let b_norm: f64 = b.iter().map(|x| x*x).sum::<f64>().sqrt().max(1e-30);
    let mut jac_res = vec![];
    for _itr in 0..1000 {
        let ax = op.matvec(&x_jac);
        let mut rn = 0.0;
        for i in 0..natoms {
            if free_mask[i] { for dd in 0..3 { let r = b[i*3+dd] - ax[i*3+dd]; rn += r*r; } }
        }
        jac_res.push(rn.sqrt() / b_norm);
        jacobi_smooth(&op, &dinv, &b, &mut x_jac, &free_mask, 0.8, 1);
    }
    println!("[t4] Jacobi 1000 iters: final res = {:.3e}", jac_res[jac_res.len()-1]);

    // Multigrid: 12 pivots
    let pivots = select_pivots_maximin(&bonds, natoms, 12, &free_mask);
    println!("[t4] pivots = {pivots:?}");
    let p = build_pivot_prolongation(&apos, &pivots, 2.0, &free_mask);
    let n_coarse = pivots.len() * 3;
    let x0 = vec![0.0; n];
    let (_x_mg, mg_res, _mg_cres, _a_c) = solve_multigrid(&op, &p, n_coarse, &b, &x0, &free_mask,
                                          0.8, 0.0, 3, 3, 100, 1e-10);
    let mg_final = mg_res[mg_res.len()-1];
    let mg_cycles = mg_res.len() - 1;
    println!("[t4] MG {mg_cycles} cycles: final res = {:.3e}", mg_final);
    println!("[t4] MG residual curve: {:?}", mg_res.iter().enumerate().map(|(i,r)| (i, format!("{r:.2e}"))).collect::<Vec<_>>());

    // MG should reach a given tolerance in far fewer smooth-steps than Jacobi.
    // This is the standard MG benchmark: compare work-to-tolerance, not final residual.
    let tol = 1e-6;
    let jac_steps_to_tol = jac_res.iter().position(|&r| r < tol).unwrap_or(jac_res.len());
    let mg_steps_to_tol = mg_res.iter().position(|&r| r < tol).unwrap_or(mg_res.len());
    let mg_smooth_to_tol = mg_steps_to_tol * 6;  // 6 smooth steps per V-cycle (3 pre + 3 post)
    let jac_final = jac_res[jac_res.len()-1];
    println!("[t4] Jacobi reaches {tol:.0e} in {jac_steps_to_tol} steps; MG reaches it in {mg_steps_to_tol} cycles ({mg_smooth_to_tol} smooth-steps)");
    assert!(mg_smooth_to_tol < jac_steps_to_tol,
        "t4_convergence: MG needs {mg_smooth_to_tol} smooth-steps to reach {tol:.0e}, Jacobi needs {jac_steps_to_tol}. MG should be faster.\n  MG final={mg_final:.3e}, Jacobi final={jac_final:.3e}");
    println!("[t4_convergence] OK: MG reaches {tol:.0e} in {mg_smooth_to_tol} smooth-steps vs Jacobi {jac_steps_to_tol} steps ({:.1}× speedup)", jac_steps_to_tol as f64 / mg_smooth_to_tol as f64);
}

#[test]
fn t7_bend_twist_mode_orthonormality() {
    let apos: Vec<Vec3d> = (0..5).flat_map(|i| [-0.5,0.5].map(|y| Vec3d::new(i as f64,y,0.0))).collect();
    let phi = build_bend_twist_modes(&apos, Vec3d::new(1.0,0.0,0.0), Vec3d::new(0.0,0.0,1.0));
    let mut gram = vec![0.0;4];
    for a in 0..2 { for b in 0..2 { gram[a*2+b]=(0..apos.len()*3).map(|i| phi[i*2+a]*phi[i*2+b]).sum(); } }
    let err = gram.iter().zip([1.0,0.0,0.0,1.0]).map(|(a,b)| (a-b).abs()).fold(0.0f64,f64::max);
    assert!(err < 1e-14, "t7_bend_twist_mode_orthonormality: Gram={gram:?} max_err={err:.3e}");
    println!("[t7_bend_twist_mode_orthonormality] OK: Gram={gram:?}");
}

#[test]
fn t6_fitted_modal_quadratic_parity() {
    let phi = vec![1.0,0.0, 0.0,1.0, 0.0,0.0];
    let eps = 0.1;
    let k = [[4.0,1.0],[1.0,3.0]];
    let mut fminus = vec![0.0; 6];
    let mut fplus = vec![0.0; 6];
    for mode in 0..2 {
        for i in 0..2 {
            fplus[mode*3+i] = -k[i][mode]*eps;
            fminus[mode*3+i] = k[i][mode]*eps;
        }
    }
    let model = ModalQuadratic::fit_central(&phi, 3, 2, eps, &fminus, &fplus);
    let mut dx = vec![0.0; 3];
    let dq = model.solve_force(&[5.0,4.0,0.0], &mut dx);
    let max_k_err = model.k.iter().zip([4.0,1.0,1.0,3.0]).map(|(a,b)| (a-b).abs()).fold(0.0f64,f64::max);
    let max_q_err = dq.iter().zip([1.0,1.0]).map(|(a,b)| (a-b).abs()).fold(0.0f64,f64::max);
    assert!(max_k_err < 1e-14 && max_q_err < 1e-14 && dx[2] == 0.0,
        "t6_fitted_modal_quadratic_parity: max_k_err={max_k_err:.3e} max_q_err={max_q_err:.3e} K={:?} dq={dq:?} dx={dx:?}", model.k);
    println!("[t6_fitted_modal_quadratic_parity] OK: max_k_err={max_k_err:.3e}, max_q_err={max_q_err:.3e}");
}

#[test]
fn t5_cached_coarse_force_parity() {
    let (op, apos, free_mask) = chain5();
    let bonds = [[0,1],[1,2],[2,3],[3,4]];
    let pivots = select_pivots_maximin(&bonds, op.natoms, 3, &free_mask);
    let p = build_pivot_prolongation(&apos, &pivots, 2.0, &free_mask);
    let n_coarse = pivots.len()*3;
    let level = GalerkinLevel::new(&op, &p, n_coarse);
    let force: Vec<f64> = (0..op.natoms).flat_map(|i| if free_mask[i] { vec![0.3*i as f64, -9.81, 0.2] } else { vec![0.0; 3] }).collect();
    let mut dx_cached = vec![0.0; op.natoms*3];
    let e_cached = level.solve_force(&force, &free_mask, &mut dx_cached);
    let mut dx_reference = vec![0.0; op.natoms*3];
    let e_reference = coarse_correct(&op, &p, &level.a_c_chol, n_coarse, &force, &mut dx_reference, &free_mask);
    let max_err = dx_cached.iter().zip(dx_reference.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f64, f64::max);
    let force_v: Vec<Vec3d> = force.chunks_exact(3).map(|f| Vec3d::new(f[0], f[1], f[2])).collect();
    let mut moved = apos.clone();
    let (e_applied, max_step) = level.apply_force_step(&mut moved, &force_v, &free_mask, 0.5);
    let max_apply_err = moved.iter().zip(apos.iter()).zip(dx_cached.chunks_exact(3)).map(|((p1,p0),d)| {
        (Vec3d::set_sub(*p1,*p0) - Vec3d::new(0.5*d[0],0.5*d[1],0.5*d[2])).norm()
    }).fold(0.0f64, f64::max);
    assert!(max_err < 1e-14 && (e_cached-e_reference).abs() < 1e-14 && (e_applied-e_cached).abs() < 1e-14 && max_apply_err < 1e-14,
        "t5_cached_coarse_force_parity: max_dx_err={max_err:.3e} energy cached/reference/applied={e_cached:.15e}/{e_reference:.15e}/{e_applied:.15e} max_apply_err={max_apply_err:.3e} max_step={max_step:.3e}");
    println!("[t5_cached_coarse_force_parity] OK: max_dx_err={max_err:.3e}, coarse_energy={e_cached:.3e}, max_apply_err={max_apply_err:.3e}, max_step={max_step:.3e}");
}

/// Helper to reconstruct positions for pivot prolongation in t3 (chain5 has fixed geometry).
fn _apos_placeholder(op: &TrussOp) -> Vec<Vec3d> {
    // chain5 positions — must match chain5()
    match op.natoms {
        5 => vec![
            Vec3d::new(0.0, 0.0, 0.0),
            Vec3d::new(1.0, 0.0, 0.0),
            Vec3d::new(2.0, 0.0, 0.0),
            Vec3d::new(3.0, 0.0, 0.0),
            Vec3d::new(4.0, 0.0, 0.0),
        ],
        n => (0..n).map(|i| Vec3d::new(i as f64, 0.0, 0.0)).collect(),
    }
}
