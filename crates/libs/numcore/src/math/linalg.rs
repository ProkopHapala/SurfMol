/// Symmetric 3×3 eigendecomposition — analytical (closed-form), no iteration.
/// Ported from FireCore/cpp/common/math/Mat3.h:Mat3T::eigenvals() + eigenvec()
/// Original algorithm: Smith, Oliver K. (April 1961), "Eigenvalues of a symmetric 3×3 matrix.",
/// Communications of the ACM 4 (4): 168.
/// Reference: http://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf
/// Replaces nalgebra::Matrix3::symmetric_eigen for the only use case in SurfMol (PCA in thumbnailer).
///
/// Input: row-major [xx,xy,xz, yx,yy,yz, zx,zy,zz] (symmetric, only 6 unique values used).
/// Output: [(eigenvalue, [eigenvector_x,y,z]); 3], sorted ascending by eigenvalue.

pub fn symmetric_eigen_3x3(a: [f32; 9]) -> [(f32, [f32; 3]); 3] {
    let [xx, xy, xz, _yx, yy, yz, _zx, _zy, zz] = a; // symmetric: yx=xy, zx=xz, zy=yz

    // --- Eigenvalues: Smith 1961 trigonometric solution of the characteristic cubic ---
    // char. poly: λ³ - c2·λ² + c1·λ - c0 = 0, where:
    //   c0 = det(A), c1 = sum of 2×2 principal minors, c2 = trace(A)
    let inv3 = 1.0 / 3.0;
    let root3 = 3.0_f32.sqrt();
    let amax = a.iter().copied().fold(a[0], f32::max);
    let amax2 = amax * amax;
    let c0 = xx * yy * zz + 2.0 * xy * xz * yz - xx * yz * yz - yy * xz * xz - zz * xy * xy;
    let c1 = xx * yy - xy * xy + xx * zz - xz * xz + yy * zz - yz * yz;
    let c2 = xx + yy + zz;
    // Normalize by amax to avoid overflow/underflow
    let c2 = c2 / amax;
    let c1 = c1 / amax2;
    let c0 = c0 / (amax2 * amax);

    let c2_div3 = c2 * inv3;
    let a_div3 = (c1 - c2 * c2_div3) * inv3;
    let a_div3 = if a_div3 > 0.0 { 0.0 } else { a_div3 }; // clamp: a_div3 ≤ 0 for real eigenvalues
    let mb_div2 = 0.5 * (c0 + c2_div3 * (2.0 * c2_div3 * c2_div3 - c1));
    let q = mb_div2 * mb_div2 + a_div3 * a_div3 * a_div3;
    let q = if q > 0.0 { 0.0 } else { q }; // clamp: q ≤ 0 for real eigenvalues
    let magnitude = (-a_div3).sqrt();
    let angle = atan2_safe((-q).sqrt(), mb_div2) * inv3;
    let cs = angle.cos();
    let sn = angle.sin();

    let ev0 = amax * (c2_div3 + 2.0 * magnitude * cs);
    let ev1 = amax * (c2_div3 - magnitude * (cs + root3 * sn));
    let ev2 = amax * (c2_div3 - magnitude * (cs - root3 * sn));

    // --- Eigenvectors: cross-product method (FireCore Mat3.h:eigenvec) ---
    // For each eigenvalue λ, form (A - λI) rows, cross-product pairs → null space vectors.
    // The cross product with largest |·|² is the most stable eigenvector.
    let v0 = eigenvec_symmetric(xx, xy, xz, yy, yz, zz, ev0);
    let v1 = eigenvec_symmetric(xx, xy, xz, yy, yz, zz, ev1);
    let v2 = eigenvec_symmetric(xx, xy, xz, yy, yz, zz, ev2);

    let mut pairs = [(ev0, v0), (ev1, v1), (ev2, v2)];
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

/// Compute eigenvector for a given eigenvalue of a symmetric 3×3 matrix.
/// Ported from FireCore Mat3.h:eigenvec. Uses cross products of (A-λI) rows.
fn eigenvec_symmetric(xx: f32, xy: f32, xz: f32, yy: f32, yz: f32, zz: f32, eval: f32) -> [f32; 3] {
    // Rows of (A - λI), using symmetry: row0=(xx-λ, xy, xz), row1=(xy, yy-λ, yz), row2=(xz, yz, zz-λ)
    let r0 = [xx - eval, xy, xz];
    let r1 = [xy, yy - eval, yz];
    let r2 = [xz, yz, zz - eval];
    let c01 = cross3(r0, r1); // r0 × r1 — in null space
    let c02 = cross3(r0, r2);
    let c12 = cross3(r1, r2);
    let d0 = dot3(c01, c01);
    let d1 = dot3(c02, c02);
    let d2 = dot3(c12, c12);
    // Pick the cross product with largest |·|² (most stable)
    if d0 >= d1 && d0 >= d2 {
        mul3(c01, 1.0 / d0.sqrt())
    } else if d1 >= d2 {
        mul3(c02, 1.0 / d1.sqrt())
    } else {
        mul3(c12, 1.0 / d2.sqrt())
    }
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn mul3(a: [f32; 3], s: f32) -> [f32; 3] { [a[0] * s, a[1] * s, a[2] * s] }

/// atan2 that handles the FireCore convention: atan2(sqrt(-q), mb_div2).
/// When q=0 (degenerate), sqrt(-q)=0, so angle depends on sign of mb_div2.
fn atan2_safe(y: f32, x: f32) -> f32 {
    if y < 1e-15 && x.abs() < 1e-15 { return std::f32::consts::FRAC_PI_2; } // both ~0 → π/2 (FireCore line 718)
    y.atan2(x)
}

// ============================================================================
// Dense f64 linear algebra for small matrices (multigrid coarse-space solves).
// Coarse space is ≤ a few hundred DOF, so dense is fine. Row-major layout.
// ============================================================================

/// Cholesky factorization of a dense SPD matrix A (n×n, row-major) into L (lower triangular,
/// row-major) with A = L·Lᵀ. Returns L packed into n² (full n×n, upper triangle zeroed).
/// Fails loud (per AGENTS.md Fail Fast) if A is not SPD: diagonal ≤ 0 or negative sqrt.
/// Ported conceptually from NumericalMathPlayground/LinarElasticity/MultiGrid.galerkin_coarse_operator
/// which uses scipy.linalg.lu_factor; Cholesky is preferred for SPD coarse operators.
pub fn cholesky_factor_f64(a: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n, "cholesky_factor_f64: input len {} != n²={}", a.len(), n * n);
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j { s -= l[i * n + k] * l[j * n + k]; }
            if i == j {
                assert!(s > 0.0, "cholesky_factor_f64: matrix not SPD at diag[{i}]={s:.3e} (≤0). A may be singular — check Galerkin operator regularization or prolongation rank.");
                l[i * n + j] = s.sqrt();
            } else {
                let ljj = l[j * n + j];
                assert!(ljj > 0.0, "cholesky_factor_f64: zero pivot L[{j},{j}] during factor at i={i}");
                l[i * n + j] = s / ljj;
            }
        }
    }
    l
}

/// Solve A·x = b given Cholesky factor L (from cholesky_factor_f64). A = L·Lᵀ.
/// Forward solve L·y = b, then back solve Lᵀ·x = y. Returns x (length n).
pub fn cholesky_solve_f64(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(l.len(), n * n, "cholesky_solve_f64: L len {} != n²={}", l.len(), n * n);
    assert_eq!(b.len(), n, "cholesky_solve_f64: b len {} != n={}", b.len(), n);
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i { s -= l[i * n + k] * y[k]; }
        let lii = l[i * n + i];
        assert!(lii > 0.0, "cholesky_solve_f64: zero diagonal L[{i},{i}]={lii}");
        y[i] = s / lii;
    }
    let mut x = vec![0.0f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut s = y[i];
        for k in (i + 1)..n { s -= l[k * n + i] * x[k]; }   // Lᵀ[i,k] = L[k,i]
        let lii = l[i * n + i];
        x[i] = s / lii;
    }
    x
}

/// Dense matvec y = A·x for row-major n×n A. (Used by Galerkin coarse operator application.)
#[inline(always)]
pub fn dense_matvec_f64(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    assert_eq!(x.len(), n);
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n { s += a[i * n + j] * x[j]; }
        y[i] = s;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagonal_matrix() {
        let a = [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let eig = symmetric_eigen_3x3(a);
        assert!((eig[0].0 - 1.0).abs() < 1e-4, "eigenvalue 0: {}", eig[0].0);
        assert!((eig[1].0 - 2.0).abs() < 1e-4, "eigenvalue 1: {}", eig[1].0);
        assert!((eig[2].0 - 3.0).abs() < 1e-4, "eigenvalue 2: {}", eig[2].0);
    }

    #[test]
    fn test_identity() {
        let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let eig = symmetric_eigen_3x3(a);
        for (val, _) in &eig { assert!((val - 1.0).abs() < 1e-4, "val={}", val); }
    }

    #[test]
    fn test_offdiagonal() {
        // [[2, 1, 0], [1, 2, 0], [0, 0, 3]] → eigenvalues 1, 3, 3
        let a = [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let eig = symmetric_eigen_3x3(a);
        assert!((eig[0].0 - 1.0).abs() < 1e-4, "smallest: {}", eig[0].0);
        // Repeated eigenvalue (3,3): analytical Smith method loses ~1e-3 precision in degenerate cases
        assert!((eig[2].0 - 3.0).abs() < 2e-3, "largest: {}", eig[2].0);
        // Eigenvector for eigenvalue 1 is [1/√2, -1/√2, 0] (or its negative)
        let (val, vec) = &eig[0];
        assert!((val - 1.0).abs() < 1e-4);
        let s2 = 2.0_f32.sqrt();
        let ok_a = (vec[0] - 1.0 / s2).abs() < 1e-4 && (vec[1] + 1.0 / s2).abs() < 1e-4;
        let ok_b = (vec[0] + 1.0 / s2).abs() < 1e-4 && (vec[1] - 1.0 / s2).abs() < 1e-4;
        assert!(ok_a || ok_b, "vec=[{}, {}, {}]", vec[0], vec[1], vec[2]);
    }

    #[test]
    fn test_random_symmetric() {
        // Construct A = R D R^T where D=diag(1,2,3) and R is a 30° rotation around z
        let r = [0.866, -0.5, 0.0, 0.5, 0.866, 0.0, 0.0, 0.0, 1.0];
        let d = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let mut rd = [0.0f32; 9];
        for i in 0..3 { for j in 0..3 { for k in 0..3 { rd[i*3+j] += r[i*3+k] * d[k*3+j]; } } }
        let mut a = [0.0f32; 9];
        for i in 0..3 { for j in 0..3 { for k in 0..3 { a[i*3+j] += rd[i*3+k] * r[j*3+k]; } } }
        // Ensure symmetric
        assert!((a[1] - a[3]).abs() < 1e-5);
        assert!((a[2] - a[6]).abs() < 1e-5);
        assert!((a[5] - a[7]).abs() < 1e-5);
        let eig = symmetric_eigen_3x3(a);
        assert!((eig[0].0 - 1.0).abs() < 1e-4, "eig[0]={}", eig[0].0);
        assert!((eig[1].0 - 2.0).abs() < 1e-4, "eig[1]={}", eig[1].0);
        assert!((eig[2].0 - 3.0).abs() < 1e-4, "eig[2]={}", eig[2].0);
        // Verify eigenvectors are unit length
        for (_, v) in &eig {
            let n = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
            assert!((n - 1.0).abs() < 1e-4, "eigenvector not unit: |v|={}", n);
        }
    }

    #[test]
    fn test_pca_inertia_tensor() {
        // Simulate the thumbnailer use case: inertia tensor of a linear molecule along x
        // Points along x-axis → I_xx small, I_yy = I_zz large
        // Inertia tensor (symmetric): I = diag(I_xx, I_yy, I_zz) with off-diagonals ~0
        let ixx = 0.1f32; let iyy = 10.0; let izz = 10.0;
        let a = [ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz];
        let eig = symmetric_eigen_3x3(a);
        // smallest eigenvalue = ixx → x axis (longest extent)
        assert!((eig[0].0 - ixx).abs() < 1e-3, "eig[0]={} expected {}", eig[0].0, ixx);
        // eigenvector for smallest should be ~[1,0,0]
        let (_, v) = &eig[0];
        assert!(v[0].abs() > 0.99, "expected x-axis eigenvector, got [{},{},{}]", v[0], v[1], v[2]);
    }

    #[test]
    fn test_cholesky_identity() {
        let n = 3;
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let l = cholesky_factor_f64(&a, n);
        // L should be identity
        for i in 0..n { for j in 0..n { let exp = if i == j { 1.0 } else { 0.0 }; assert!((l[i*n+j] - exp).abs() < 1e-12, "L[{},{}]={}", i, j, l[i*n+j]); } }
        let b = vec![1.0, 2.0, 3.0];
        let x = cholesky_solve_f64(&l, &b, n);
        for i in 0..n { assert!((x[i] - b[i]).abs() < 1e-12, "x[{}]={} expected {}", i, x[i], b[i]); }
    }

    #[test]
    fn test_cholesky_spd_solve() {
        // A = [[4, 2], [2, 3]] — SPD (eigenvalues 1, 5). Solve A x = [1, 1].
        // x = A⁻¹ b = (1/8) [[3,-2],[-2,4]] [1,1] = [1/8, 2/8] = [0.125, 0.25]
        let n = 2;
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky_factor_f64(&a, n);
        let b = vec![1.0, 1.0];
        let x = cholesky_solve_f64(&l, &b, n);
        assert!((x[0] - 0.125).abs() < 1e-12, "x[0]={} expected 0.125", x[0]);
        assert!((x[1] - 0.25).abs() < 1e-12, "x[1]={} expected 0.25", x[1]);
        // Verify residual A x - b ≈ 0
        let ax = dense_matvec_f64(&a, &x, n);
        for i in 0..n { assert!((ax[i] - b[i]).abs() < 1e-12, "residual[{}]={}", i, ax[i] - b[i]); }
    }

    #[test]
    fn test_cholesky_4x4() {
        // A = L0 L0ᵀ with known L0, then check factor recovers it (up to signs)
        let n = 4;
        let l0 = vec![2.0, 0.0, 0.0, 0.0,  1.0, 3.0, 0.0, 0.0,  0.5, 1.0, 2.0, 0.0,  1.0, 0.0, 1.0, 4.0];
        let mut a = vec![0.0f64; n * n];
        for i in 0..n { for j in 0..n { for k in 0..n { a[i*n+j] += l0[i*n+k] * l0[j*n+k]; } } } // A = L0 L0ᵀ
        let l = cholesky_factor_f64(&a, n);
        for i in 0..n { for j in 0..=i { assert!((l[i*n+j] - l0[i*n+j]).abs() < 1e-9, "L[{},{}]={} expected {}", i, j, l[i*n+j], l0[i*n+j]); } }
        let b = vec![3.0, 5.0, 7.0, 11.0];
        let x = cholesky_solve_f64(&l, &b, n);
        let ax = dense_matvec_f64(&a, &x, n);
        for i in 0..n { assert!((ax[i] - b[i]).abs() < 1e-9, "residual[{}]={}", i, ax[i] - b[i]); }
    }
}
