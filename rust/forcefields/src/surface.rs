use surfmol_common::math::vec3::{Vec3d, VEC3_ZERO};

const TAU: f64 = 6.283185307179586476925286766559;

/// Complex number for cos/sin optimization via complex multiplication
#[derive(Copy, Clone, Debug)]
struct Complex { re: f64, im: f64 }

impl Complex {
    #[inline(always)] fn new(re: f64, im: f64) -> Self { Self { re, im } }
    #[inline(always)] fn mul(self, other: Self) -> Self {
        Self { re: self.re * other.re - self.im * other.im, im: self.re * other.im + self.im * other.re }
    }
}

pub struct SurfaceScratch {
    pub bx: Vec<f64>, pub by: Vec<f64>, pub bz: Vec<f64>,
    pub dbx: Vec<f64>, pub dby: Vec<f64>, pub dbz: Vec<f64>,
    pub cux: Vec<f64>, pub sux: Vec<f64>,
    pub cvy: Vec<f64>, pub svy: Vec<f64>,
}

impl SurfaceScratch {
    pub fn new(surf: &SurfaceFolded) -> Self {
        Self {
            bx: vec![0.0; surf.nx], by: vec![0.0; surf.ny], bz: vec![0.0; surf.nz],
            dbx: vec![0.0; surf.nx], dby: vec![0.0; surf.ny], dbz: vec![0.0; surf.nz],
            cux: vec![0.0; surf.kx_max + 1], sux: vec![0.0; surf.kx_max + 1],
            cvy: vec![0.0; surf.ky_max + 1], svy: vec![0.0; surf.ky_max + 1],
        }
    }
}

/// Precompute cos(n·φ), sin(n·φ) for n=0..nmax using complex recurrence.
/// n=0: cos=1, sin=0; n>=1: z^n where z=exp(i·φ), computed by recurrence.
/// Only 1 cos/sin call + nmax complex multiplies for all harmonics.
/// Output arrays must have length >= nmax+1.
#[inline] fn precompute_harmonics(phi: f64, nmax: usize, cos_vals: &mut [f64], sin_vals: &mut [f64]) {
    cos_vals[0] = 1.0;
    sin_vals[0] = 0.0;
    if nmax == 0 { return; }
    // z = exp(i·φ) = cos(φ) + i·sin(φ)  -- THE ONLY sin/cos CALL
    let z = Complex::new(phi.cos(), phi.sin());
    // Recurrence: z^(n+1) = z^n · z, read cos(nφ)=Re(z^n), sin(nφ)=Im(z^n)
    let mut zn = z;
    for i in 1..=nmax {
        cos_vals[i] = zn.re;
        sin_vals[i] = zn.im;
        zn = zn.mul(z);
    }
}

/// Check if all k values are integers (0,1,2,3...) -- complex recurrence applies.
#[inline] fn all_integer_harmonics(k: &[f64]) -> bool {
    for &ki in k.iter() {
        if ki < 0.0 { return false; }
        if (ki - ki.round()).abs() > 1e-12 { return false; }
    }
    true
}

/// Precompute separable 1D basis functions and derivatives for one atom position.
/// Core optimization: for integer harmonics, uses complex recurrence (1 cos/sin + max_k complex muls).
/// For non-integer k, falls back to direct cos/sin per basis.
#[inline] fn precompute_1d_bases(
    u: f64, v: f64, z: f64,
    kx: &[f64], ky: &[f64], kz: &[f64], z0: &[f64],
    bx: &mut [f64], by: &mut [f64], bz: &mut [f64],
    dbx: &mut [f64], dby: &mut [f64], dbz: &mut [f64],
) {
    let nx = kx.len();
    let ny = ky.len();
    let nz = kz.len();

    // --- X harmonics ---
    if all_integer_harmonics(kx) {
        let kx_int: Vec<usize> = kx.iter().map(|&ki| ki.round() as usize).collect();
        let kx_max = kx_int.iter().copied().max().unwrap_or(0);
        let mut cux = vec![0.0f64; kx_max + 1];
        let mut sux = vec![0.0f64; kx_max + 1];
        precompute_harmonics(TAU * u, kx_max, &mut cux, &mut sux);
        for i in 0..nx {
            let n = kx_int[i];
            bx[i] = cux[n];
            dbx[i] = -TAU * kx[i] * sux[n];
        }
    } else {
        for i in 0..nx {
            let phi = TAU * kx[i] * u;
            bx[i] = phi.cos();
            dbx[i] = -TAU * kx[i] * phi.sin();
        }
    }

    // --- Y harmonics ---
    if all_integer_harmonics(ky) {
        let ky_int: Vec<usize> = ky.iter().map(|&ki| ki.round() as usize).collect();
        let ky_max = ky_int.iter().copied().max().unwrap_or(0);
        let mut cvy = vec![0.0f64; ky_max + 1];
        let mut svy = vec![0.0f64; ky_max + 1];
        precompute_harmonics(TAU * v, ky_max, &mut cvy, &mut svy);
        for i in 0..ny {
            let n = ky_int[i];
            by[i] = cvy[n];
            dby[i] = -TAU * ky[i] * svy[n];
        }
    } else {
        for i in 0..ny {
            let phi = TAU * ky[i] * v;
            by[i] = phi.cos();
            dby[i] = -TAU * ky[i] * phi.sin();
        }
    }

    // --- Z decay (exponential, each independent) ---
    for i in 0..nz {
        let dz = z - z0[i];
        let bz_i = (-kz[i] * dz).exp();
        bz[i] = bz_i;
        dbz[i] = -kz[i] * bz_i;
    }
}

/// Folded surface potential with tensor-product separable basis.
/// Uses complex recurrence for harmonics: O(1) sin/cos per dimension instead of O(N_harmonics).
pub struct SurfaceFolded {
    // 2D lattice vectors (ax,bx,ay,by) where a=(ax,ay), b=(bx,by)
    pub ax: f64, pub bx: f64, pub ay: f64, pub by: f64,
    // Inverse lattice
    pub inv_ax: f64, pub inv_bx: f64,
    pub inv_ay: f64, pub inv_by: f64,
    // 1D harmonic frequencies [N_x], [N_y] and z-decay params [N_z]
    pub kx: Vec<f64>,
    pub ky: Vec<f64>,
    pub kz: Vec<f64>,
    pub z0: Vec<f64>,
    pub nx: usize, pub ny: usize, pub nz: usize,
    // Coefficients [natom_types * (nx*ny*nz)] for each interaction channel
    pub coeffs_q: Vec<f64>,   // electrostatics (charge)
    pub coeffs_p: Vec<f64>,  // Pauli repulsion
    pub coeffs_l: Vec<f64>,  // London dispersion
    pub ntypes: usize,

    pub kx_int: Option<Vec<usize>>, pub ky_int: Option<Vec<usize>>,
    pub kx_max: usize, pub ky_max: usize,
}

impl SurfaceFolded {
    pub fn new(ax: f64, bx: f64, ay: f64, by: f64,
               kx: Vec<f64>, ky: Vec<f64>, kz: Vec<f64>, z0: Vec<f64>, ntypes: usize) -> Self {
        let det = ax * by - bx * ay;
        assert!(det.abs() > 1e-12, "SurfaceFolded: degenerate 2D lattice");
        let idet = 1.0 / det;
        let nx = kx.len();
        let ny = ky.len();
        let nz = kz.len();
        let nbasis = nx * ny * nz;
        let ncoef = ntypes * nbasis;

        let (kx_int, kx_max) = if all_integer_harmonics(&kx) {
            let vi: Vec<usize> = kx.iter().map(|&ki| ki.round() as usize).collect();
            let km = vi.iter().copied().max().unwrap_or(0);
            (Some(vi), km)
        } else {
            (None, 0)
        };
        let (ky_int, ky_max) = if all_integer_harmonics(&ky) {
            let vi: Vec<usize> = ky.iter().map(|&ki| ki.round() as usize).collect();
            let km = vi.iter().copied().max().unwrap_or(0);
            (Some(vi), km)
        } else {
            (None, 0)
        };
        Self {
            ax, bx, ay, by,
            inv_ax: by * idet,  inv_bx: -bx * idet,
            inv_ay: -ay * idet, inv_by: ax * idet,
            kx, ky, kz, z0, nx, ny, nz,
            coeffs_q: vec![0.0; ncoef],
            coeffs_p: vec![0.0; ncoef],
            coeffs_l: vec![0.0; ncoef],
            ntypes,

            kx_int, ky_int, kx_max, ky_max,
        }
    }

    #[inline(always)] pub fn nbasis(&self) -> usize { self.nx * self.ny * self.nz }

    #[inline(always)] fn precompute_1d_bases_scratch(&self, u: f64, v: f64, z: f64, s: &mut SurfaceScratch) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;

        if let Some(ref kx_int) = self.kx_int {
            if self.kx_max > 0 { precompute_harmonics(TAU * u, self.kx_max, &mut s.cux, &mut s.sux); }
            else { s.cux[0] = 1.0; s.sux[0] = 0.0; }
            for i in 0..nx {
                let n = kx_int[i];
                s.bx[i] = s.cux[n];
                s.dbx[i] = -TAU * self.kx[i] * s.sux[n];
            }
        } else {
            for i in 0..nx {
                let phi = TAU * self.kx[i] * u;
                s.bx[i] = phi.cos();
                s.dbx[i] = -TAU * self.kx[i] * phi.sin();
            }
        }

        if let Some(ref ky_int) = self.ky_int {
            if self.ky_max > 0 { precompute_harmonics(TAU * v, self.ky_max, &mut s.cvy, &mut s.svy); }
            else { s.cvy[0] = 1.0; s.svy[0] = 0.0; }
            for i in 0..ny {
                let n = ky_int[i];
                s.by[i] = s.cvy[n];
                s.dby[i] = -TAU * self.ky[i] * s.svy[n];
            }
        } else {
            for i in 0..ny {
                let phi = TAU * self.ky[i] * v;
                s.by[i] = phi.cos();
                s.dby[i] = -TAU * self.ky[i] * phi.sin();
            }
        }

        for i in 0..nz {
            let dz = z - self.z0[i];
            let bz_i = (-self.kz[i] * dz).exp();
            s.bz[i] = bz_i;
            s.dbz[i] = -self.kz[i] * bz_i;
        }
    }

    /// Set coefficients for atom type `ityp` and basis index `ib` (flattened ix,iy,iz)
    #[inline(always)] pub fn set_coeffs(&mut self, ityp: usize, ib: usize, q: f64, p: f64, l: f64) {
        let ioff = ityp * self.nbasis() + ib;
        self.coeffs_q[ioff] = q;
        self.coeffs_p[ioff] = p;
        self.coeffs_l[ioff] = l;
    }

    /// Clamp force magnitude to prevent numerical blowup when atom penetrates surface.
    /// Returns (energy, clamped_force).
    #[inline(always)] fn clamp_force(&self, e: f64, f: Vec3d, fmax: f64) -> (f64, Vec3d) {
        let f2 = f.x*f.x + f.y*f.y + f.z*f.z;
        if f2 > fmax*fmax {
            let s = fmax / f2.sqrt();
            (e, Vec3d::new(f.x*s, f.y*s, f.z*s))
        } else {
            (e, f)
        }
    }

    /// Convert REQ [RvdW, EvdW, Q, H] → PLQ [Pauli, London, Q, H]
    #[inline(always)] pub fn req2plq(req: [f64; 4], alpha: f64) -> [f64; 4] {
        let k = -alpha;
        let e = (k * req[0]).exp();
        let c_l = e * req[1].sqrt();
        let c_p = e * c_l;
        [c_p, c_l, req[2], req[3]]
    }

    #[inline(always)] pub fn eval_atom(&self, pos: Vec3d, plq: [f64; 4]) -> (f64, Vec3d) {
        let mut scratch = SurfaceScratch::new(self);
        self.eval_atom_scratch(pos, plq, &mut scratch)
    }

    /// Evaluate with force clamping (default fmax=100 eV/Å).
    #[inline(always)] pub fn eval_atom_clamped(&self, pos: Vec3d, plq: [f64; 4], fmax: f64) -> (f64, Vec3d) {
        let (e, f) = self.eval_atom(pos, plq);
        self.clamp_force(e, f, fmax)
    }

    /// Evaluate energy and force for one atom using reusable scratch buffers (no per-atom allocations).
    /// plq = [Pauli, London, Q, Hb] precomputed per-atom coefficients.
    #[inline(always)] pub fn eval_atom_scratch(&self, pos: Vec3d, plq: [f64; 4], s: &mut SurfaceScratch) -> (f64, Vec3d) {
        let u = self.inv_ax * pos.x + self.inv_ay * pos.y;
        let v = self.inv_bx * pos.x + self.inv_by * pos.y;
        let u = u - u.floor();
        let v = v - v.floor();

        self.precompute_1d_bases_scratch(u, v, pos.z, s);

        let ioff = 0; // ntypes=1, all atoms share same basis coefficients
        let mut ic = ioff;

        let mut e_tot = 0.0;
        let mut dEdu = 0.0;
        let mut dEdv = 0.0;
        let mut dEdz = 0.0;

        for iz in 0..self.nz {
            let bz_iz = s.bz[iz];
            let dbz_iz = s.dbz[iz];
            for iy in 0..self.ny {
                let by_iy = s.by[iy];
                let dby_iy = s.dby[iy];
                let bz_by = bz_iz * by_iy;
                let dbz_by = dbz_iz * by_iy;
                let bz_dby = bz_iz * dby_iy;
                for ix in 0..self.nx {
                    let c = self.coeffs_q[ic] * plq[2] + self.coeffs_p[ic] * plq[0] + self.coeffs_l[ic] * plq[1];
                    ic += 1;
                    let bx_ix = s.bx[ix];
                    let dbx_ix = s.dbx[ix];
                    e_tot  += c * (bx_ix * bz_by);
                    dEdu   += c * (dbx_ix * bz_by);
                    dEdv   += c * (bx_ix * bz_dby);
                    dEdz   += c * (bx_ix * dbz_by);
                }
            }
        }

        let fx = -(dEdu * self.inv_ax + dEdv * self.inv_bx);
        let fy = -(dEdu * self.inv_ay + dEdv * self.inv_by);
        let fz = -dEdz;

        (e_tot, Vec3d::new(fx, fy, fz))
    }

    /// Evaluate for all atoms, accumulate forces into `fapos`
    pub fn eval_all(&self, apos: &[Vec3d], plqs: &[[f64; 4]], fapos: &mut [Vec3d]) -> f64 {
        let mut scratch = SurfaceScratch::new(self);
        self.eval_all_scratch(apos, plqs, fapos, &mut scratch)
    }

    /// Evaluate for all atoms, reusing scratch buffers (no per-atom allocations).
    pub fn eval_all_scratch(&self, apos: &[Vec3d], plqs: &[[f64; 4]], fapos: &mut [Vec3d], scratch: &mut SurfaceScratch) -> f64 {
        let mut etot = 0.0;
        for ia in 0..apos.len() {
            let (e, f) = self.eval_atom_scratch(apos[ia], plqs[ia], scratch);
            etot += e;
            fapos[ia].add(f);
        }
        etot
    }

    /// Evaluate with force clamping for all atoms.
    pub fn eval_all_clamped(&self, apos: &[Vec3d], plqs: &[[f64; 4]], fapos: &mut [Vec3d], fmax: f64) -> f64 {
        let mut scratch = SurfaceScratch::new(self);
        let mut etot = 0.0;
        for ia in 0..apos.len() {
            let (e, f) = self.eval_atom_scratch(apos[ia], plqs[ia], &mut scratch);
            let (_, fc) = self.clamp_force(e, f, fmax);
            etot += e;
            fapos[ia].add(fc);
        }
        etot
    }
}

/// Create a NaCl-like substrate surface with separable Morse + electrostatics basis:
///
/// Z-basis structure (Morse exponent > charge exponent):
///   iz=0: kz = 2*β_morse  → Pauli repulsion  (square of attractive = exp(-2β_morse z))
///   iz=1: kz = β_morse    → London attraction (attractive Morse = exp(-β_morse z))
///   iz=2: kz = β_charge   → Electrostatics   (slower decay than Morse)
///
/// X/Y harmonics: k=1.0 gives period a/2, matching Na-Cl spacing.
///
/// Charge/element assignment:
///   (1,1,2) checkerboard: +1 at Na sites, -1 at Cl sites.
pub fn setup_nacl_surface(a: f64, z0: f64, beta_charge: f64, beta_morse_ratio: f64, q_amp: f64, plq_amp: f64) -> SurfaceFolded {
    let beta_morse = beta_charge * beta_morse_ratio;   // steeper than charge

    let kx = vec![0.0, 1.0];  // 0: constant, 1.0: cos(2π*x/a)
    let ky = vec![0.0, 1.0];  // 0: constant, 1.0: cos(2π*y/a)
    let kz = vec![2.0 * beta_morse, beta_morse, beta_charge];
    let z0s = vec![z0, z0, z0];
    let nx = kx.len();
    let ny = ky.len();
    let nz = kz.len();

    let mut surf = SurfaceFolded::new(a, 0.0, 0.0, a, kx, ky, kz, z0s, 1);

    // Basis index: ib = ix + nx*(iy + ny*iz)
    for iz in 0..nz {
        let iz_off = iz * nx * ny;
        let i_const = iz_off + 0 + nx * (0 + ny * 0);  // (0,0,iz)
        let i_cos_x = iz_off + 1 + nx * (0 + ny * 0);  // (1,0,iz)
        let i_cos_y = iz_off + 0 + nx * (1 + ny * 0);  // (0,1,iz)
        let i_check = iz_off + 1 + nx * (1 + ny * 0);  // (1,1,iz)

        if iz == 0 {
            // Pauli repulsion on short-range z-decay (square of attractive)
            // Prefactor must overcome atomic P<<L ratio from req2plq to create
            // a proper minimum.  ~40x gives equilibrium ~1.5Å above surface.
            surf.set_coeffs(0, i_const, 0.0,  plq_amp * 40.0,  0.0);
            surf.set_coeffs(0, i_cos_x, 0.0, 0.0,      0.0);
            surf.set_coeffs(0, i_cos_y, 0.0, 0.0,      0.0);
            surf.set_coeffs(0, i_check, 0.0, 0.0,      0.0);
        } else if iz == 1 {
            // London attraction on medium-range z-decay
            surf.set_coeffs(0, i_const, 0.0,  0.0,     -plq_amp);
            surf.set_coeffs(0, i_cos_x, 0.0,  0.0,      0.0);
            surf.set_coeffs(0, i_cos_y, 0.0,  0.0,      0.0);
            surf.set_coeffs(0, i_check, 0.0,  0.0,      0.0);
        } else {
            // Electrostatics on slow z-decay (checkerboard only)
            surf.set_coeffs(0, i_const, 0.0,  0.0,      0.0);
            surf.set_coeffs(0, i_cos_x, 0.0,  0.0,      0.0);
            surf.set_coeffs(0, i_cos_y, 0.0,  0.0,      0.0);
            surf.set_coeffs(0, i_check, q_amp, 0.0,     0.0);
        }
    }

    surf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonics_recurrence() {
        let phi = 0.7;
        let nmax = 5;
        let mut cos_vals = vec![0.0; nmax + 1];  // n=0..nmax
        let mut sin_vals = vec![0.0; nmax + 1];
        precompute_harmonics(phi, nmax, &mut cos_vals, &mut sin_vals);

        // n=0: cos=1, sin=0
        assert!((cos_vals[0] - 1.0).abs() < 1e-12, "cos(0) should be 1");
        assert!(sin_vals[0].abs() < 1e-12, "sin(0) should be 0");

        // n=1..nmax
        for n in 1..=nmax {
            let expected_cos = (n as f64 * phi).cos();
            let expected_sin = (n as f64 * phi).sin();
            assert!((cos_vals[n] - expected_cos).abs() < 1e-12, "cos({}*{}) mismatch: {} vs {}", n, phi, cos_vals[n], expected_cos);
            assert!((sin_vals[n] - expected_sin).abs() < 1e-12, "sin({}*{}) mismatch: {} vs {}", n, phi, sin_vals[n], expected_sin);
        }
    }

    #[test]
    fn test_surface_eval_constant() {
        // 10x10 square lattice, one constant basis (nx=0 harmonic=1, ny=0 harmonic=1, z-constant)
        let mut surf = SurfaceFolded::new(
            10.0, 0.0, 0.0, 10.0,
            vec![0.0], vec![0.0], vec![0.0], vec![0.0], // kx=[0], ky=[0], kz=[0], z0=[0]
            1
        );
        // Constant basis: bx=cos(0)=1, by=cos(0)=1, bz=exp(0)=1 → basis=1 everywhere
        surf.set_coeffs(0, 0, 1.0, 0.0, 0.0); // Q-coeff=1

        let req = [3.0, 0.1, 0.5, 0.0]; // Q=0.5
        let plq = SurfaceFolded::req2plq(req, 2.0);
        let pos = Vec3d::new(3.0, 4.0, 1.0);
        let (e, f) = surf.eval_atom(pos, plq);

        // E = c_q * Q * basis = 1.0 * 0.5 * 1.0 = 0.5
        assert!((e - 0.5).abs() < 1e-10, "constant basis energy: {} vs 0.5", e);
        // Force should be zero for constant potential
        assert!(f.norm2() < 1e-10, "constant basis force should be zero: {:?}", f);
    }

    #[test]
    fn test_surface_eval_cos_x() {
        // cos(2π*x/10) basis in x, constant in y and z
        let mut surf = SurfaceFolded::new(
            10.0, 0.0, 0.0, 10.0,
            vec![1.0], vec![0.0], vec![0.0], vec![0.0],
            1
        );
        surf.set_coeffs(0, 0, 1.0, 0.0, 0.0);

        let req = [3.0, 0.1, 1.0, 0.0]; // Q=1.0
        let plq = SurfaceFolded::req2plq(req, 2.0);

        // At x=0: cos(0) = 1
        let (e0, _) = surf.eval_atom(Vec3d::new(0.0, 0.0, 0.0), plq);
        assert!((e0 - 1.0).abs() < 1e-10, "cos(0) should be 1, got {}", e0);

        // At x=5: cos(π) = -1
        let (e5, _) = surf.eval_atom(Vec3d::new(5.0, 0.0, 0.0), plq);
        assert!((e5 - (-1.0)).abs() < 1e-10, "cos(π) should be -1, got {}", e5);

        // At x=2.5: cos(π/2) = 0
        let (e25, _) = surf.eval_atom(Vec3d::new(2.5, 0.0, 0.0), plq);
        assert!(e25.abs() < 1e-10, "cos(π/2) should be 0, got {}", e25);
    }

    #[test]
    fn test_surface_eval_z_decay() {
        // Constant in x,y, exponential decay in z
        let mut surf = SurfaceFolded::new(
            10.0, 0.0, 0.0, 10.0,
            vec![0.0], vec![0.0], vec![1.0], vec![0.0], // kz=1.0, z0=0.0
            1
        );
        surf.set_coeffs(0, 0, 1.0, 0.0, 0.0);

        let req = [3.0, 0.1, 1.0, 0.0];
        let plq = SurfaceFolded::req2plq(req, 2.0);

        // At z=1: exp(-1.0 * 1.0) = exp(-1)
        let (e, _) = surf.eval_atom(Vec3d::new(0.0, 0.0, 1.0), plq);
        let expected = (-1.0_f64).exp();
        assert!((e - expected).abs() < 1e-10, "z decay: {} vs {}", e, expected);
    }

    #[test]
    fn test_req2plq() {
        let req = [3.0, 0.1, 0.5, 0.0]; // R=3, E=0.1, Q=0.5
        let plq = SurfaceFolded::req2plq(req, 2.0);
        let e = (-6.0_f64).exp();
        assert!((plq[0] - e * e * 0.1_f64.sqrt()).abs() < 1e-10, "Pauli mismatch");
        assert!((plq[1] - e * 0.1_f64.sqrt()).abs() < 1e-10, "London mismatch");
        assert!((plq[2] - 0.5).abs() < 1e-10, "Q should pass through");
    }
}
