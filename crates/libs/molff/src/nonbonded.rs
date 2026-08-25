use numcore::math::quat4::{Quat4i, QUAT4I_MINUS_ONES};
use numcore::math::vec3::{Vec3d, VEC3_ZERO};
use numcore::util::AlignedVec;

const EXCL_MAX: usize = 16;
const COULOMB_CONST: f64 = 14.3996448915; // eV*A / e^2

pub struct NonBondedFF {
    pub natoms: usize,
    pub reqs: AlignedVec<[f64; 4], 64>,      // [RvdW, sqrt(EvdW), Q, Hb] per atom
    pub plqs: AlignedVec<[f64; 4], 64>,      // [Pauli, London, Q, Hb] per atom (precomputed from reqs)
    pub excl: AlignedVec<i32, 64>,           // [natoms * EXCL_MAX] sorted exclusion list per atom
    pub pbc_shifts: Vec<Vec3d>,
    pub npbc: i32,
    pub lvec_a: Vec3d,
    pub lvec_b: Vec3d,
    pub lvec_c: Vec3d,
    pub n_pbc: [i32; 3],
    pub b_pbc: bool,
    pub rdamp: f64,
    pub fmax_nonbonded: f64,
    pub b_clamp_nonbonded: bool,

    pub rcut2: f64,
}

impl NonBondedFF {
    pub fn new(natoms: usize) -> Self {
        let mut reqs = AlignedVec::<[f64; 4], 64>::new();
        reqs.resize_fill(natoms, [0.0, 0.0, 0.0, 0.0]);
        let mut plqs = AlignedVec::<[f64; 4], 64>::new();
        plqs.resize_fill(natoms, [0.0, 0.0, 0.0, 0.0]);
        let mut excl = AlignedVec::<i32, 64>::new();
        excl.resize_fill(natoms * EXCL_MAX, -1);
        Self {
            natoms,
            reqs,
            plqs,
            excl,
            pbc_shifts: Vec::new(),
            npbc: 0,
            lvec_a: VEC3_ZERO,
            lvec_b: VEC3_ZERO,
            lvec_c: VEC3_ZERO,
            n_pbc: [0, 0, 0],
            b_pbc: false,
            rdamp: 0.1,
            fmax_nonbonded: 10.0,
            b_clamp_nonbonded: true,

            rcut2: 1e300,
        }
    }

    #[inline(always)] pub fn set_cutoff(&mut self, rcut: f64) { self.rcut2 = rcut * rcut; }

    /// Precompute PLQH coefficients from REQH for all atoms.
    /// req[0]=RvdW, req[1]=sqrt(EvdW), req[2]=Q, req[3]=Hb
    /// plq[0]=Pauli, plq[1]=London, plq[2]=Q, plq[3]=Hb
    pub fn make_plqs(&mut self, alpha: f64) {
        let reqs = self.reqs.as_slice();
        let plqs = self.plqs.as_mut_slice();
        for i in 0..self.natoms {
            let req = reqs[i];
            let k = -alpha;
            let e = (k * req[0]).exp();
            let c_l = e * req[1].sqrt();
            let c_p = e * c_l;
            plqs[i] = [c_p, c_l, req[2], req[3]];
        }
    }

    /// Build sorted exclusion list of 1-2 and 1-3 neighbors for each atom.
    pub fn make_second_neighs(&mut self, neighs: &[Quat4i], natoms: usize) {
        assert!(natoms <= self.natoms, "make_second_neighs natoms mismatch");
        self.excl.resize_fill(natoms * EXCL_MAX, -1);
        for ia in 0..natoms {
            let excli = &mut self.excl.as_mut_slice()[ia * EXCL_MAX..ia * EXCL_MAX + EXCL_MAX];
            for k in 0..EXCL_MAX { excli[k] = -1; }
            let mut n: usize = 0;
            let mut add_excl = |jb: i32| {
                if jb < 0 { return; }
                if jb == ia as i32 { return; }
                if jb >= natoms as i32 { return; }
                for m in 0..n { if excli[m] == jb { return; } }
                if n < EXCL_MAX { excli[n] = jb; n += 1; }
                else { panic!("NonBondedFF::make_second_neighs() ia={} n={} >= EXCL_MAX={}", ia, n, EXCL_MAX); }
            };
            let ng = neighs[ia].as_array();
            for s in 0..4 { add_excl(ng[s]); }
            for s in 0..4 {
                let ja = ng[s];
                if ja < 0 || ja >= natoms as i32 { continue; }
                let nj = neighs[ja as usize].as_array();
                for t in 0..4 { add_excl(nj[t]); }
            }
            excli[..n].sort_unstable();
        }
    }

    pub fn make_pbc_shifts(&mut self, n_pbc: [i32; 3], lvec_a: Vec3d, lvec_b: Vec3d, lvec_c: Vec3d) {
        self.b_pbc = true;
        self.n_pbc = n_pbc;
        self.lvec_a = lvec_a;
        self.lvec_b = lvec_b;
        self.lvec_c = lvec_c;
        let nx = (n_pbc[0] * 2 + 1) as usize;
        let ny = (n_pbc[1] * 2 + 1) as usize;
        let nz = (n_pbc[2] * 2 + 1) as usize;
        let npbc = nx * ny * nz;
        self.npbc = npbc as i32;
        self.pbc_shifts.resize(npbc, VEC3_ZERO);
        let mut ipbc = 0usize;
        for iz in -n_pbc[2]..=n_pbc[2] {
            for iy in -n_pbc[1]..=n_pbc[1] {
                for ix in -n_pbc[0]..=n_pbc[0] {
                    let sx = Vec3d::set_mul(lvec_a, ix as f64);
                    let sy = Vec3d::set_mul(lvec_b, iy as f64);
                    let sz = Vec3d::set_mul(lvec_c, iz as f64);
                    self.pbc_shifts[ipbc] = Vec3d::set_add(Vec3d::set_add(sx, sy), sz);
                    ipbc += 1;
                }
            }
        }
    }

    #[inline(always)]
    fn combine_req(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
        [a[0] + b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
    }

    #[inline(always)]
    fn get_ljqh(dp: Vec3d, reqh: [f64; 4], r2damp: f64) -> (f64, Vec3d) {
        let r2 = dp.norm2();
        let ir2_ = 1.0 / (r2 + r2damp);
        let e_coul = COULOMB_CONST * reqh[2] * ir2_.sqrt();
        let f_coul = e_coul * ir2_;
        let ir2 = 1.0 / r2;
        let u2 = reqh[0] * reqh[0] * ir2;
        let u6 = u2 * u2 * u2;
        let e_vdw = reqh[1] * reqh[1]; // reqh[1] is sqrt(EvdW)
        let vdw = u6 * e_vdw;
        let h = u6 * u6 * if reqh[3] < 0.0 { reqh[3] * e_vdw } else { 0.0 };
        let e = e_coul + (u6 - 2.0) * vdw + h;
        let f = f_coul + ((u6 - 1.0) * vdw + h) * ir2 * 12.0;
        let force = Vec3d::set_mul(dp, -f);
        (e, force)
    }

    #[inline(always)]
    fn clamp_force_vec(f: &mut Vec3d, f2max: f64) {
        let f2 = f.norm2();
        if f2 > f2max {
            f.mul(f2max.sqrt() / f2.sqrt());
        }
    }

    /// Evaluate non-bonded LJ+Coulomb for atom ia, skipping exclusions.
    /// Modifies fi (accumulates); returns energy.
    #[inline(always)]
    fn eval_nb_atom(&self, ia: usize, apos: &[Vec3d], fi: &mut Vec3d) -> f64 {
        let pi = apos[ia];
        let reqi = self.reqs.as_slice()[ia];
        let i0_ex = ia * EXCL_MAX;
        let excl = self.excl.as_slice();
        let mut iex = i0_ex;
        let iex_end = i0_ex + EXCL_MAX - 1;
        let mut jex = excl[iex];
        let mut e = 0.0;
        let r2damp = self.rdamp * self.rdamp;
        let f2max = self.fmax_nonbonded * self.fmax_nonbonded;
        let rcut2 = self.rcut2;
        for ja in 0..self.natoms {
            if ja == ia { continue; }
            if jex != -1 {
                if iex < iex_end && jex < ja as i32 { iex += 1; }
                jex = excl[iex];
            }
            if jex == ja as i32 { continue; }
            let dp = Vec3d::set_sub(apos[ja], pi);
            if dp.norm2() > rcut2 { continue; }
            let reqj = self.reqs.as_slice()[ja];
            let reqij = Self::combine_req(reqj, reqi);
            let (eij, mut fij) = Self::get_ljqh(dp, reqij, r2damp);
            if self.b_clamp_nonbonded { Self::clamp_force_vec(&mut fij, f2max); }
            e += eij;
            fi.add(fij);
        }
        e
    }

    /// Evaluate all non-bonded interactions (no PBC), accumulating into fapos.
    /// Returns total non-bonded energy.
    pub fn eval(&mut self, fapos: &mut [Vec3d], apos: &[Vec3d]) -> f64 {
        assert_eq!(fapos.len(), self.natoms, "fapos length mismatch");
        assert_eq!(apos.len(), self.natoms, "apos length mismatch");
        if self.b_pbc && self.npbc > 0 { return self.eval_pbc(fapos, apos); }
        let mut etot = 0.0;
        for ia in 0..self.natoms {
            let (e, mut fi) = (0.0, VEC3_ZERO);
            etot += self.eval_nb_atom(ia, apos, &mut fi);
            fapos[ia].add(fi);
        }
        etot
    }

    // ================== NBFF diagnostic prints (parity with C++ NBFF.h) ==================

    pub fn print_second_neighs(&self, mode: i32) {
        println!("NBFF::printSecondNeighs()");
        if self.excl.len() == 0 { println!("NBFF::printSecondNeighs() excl not built"); return; }
        for ia in 0..self.natoms {
            print!("excl[{:3}] ", ia);
            let lst = &self.excl.as_slice()[ia * EXCL_MAX..ia * EXCL_MAX + EXCL_MAX];
            for k in 0..EXCL_MAX {
                let v = lst[k];
                if v < 0 { print!(" -1"); continue; }
                if mode == 3 { print!("  {:02X}:{:06X}", (v >> 24) & 0xFF, v & 0x00FFFFFF); }
                else if mode == 2 { print!("  {:3}", (v >> 24) & 0xFF); }
                else              { print!("  {:3}", v); }
            }
            println!();
        }
    }

    pub fn print_nonbonded(&self, apos: &[Vec3d]) {
        println!("NBFF::print_nonbonded(n={})", self.natoms);
        for i in 0..self.natoms {
            let r = self.reqs.as_slice()[i];
            let p = apos[i];
            println!("nb_atom[{:3}] REQ({:7.3},{:12.8},{:12.8},{:12.8}) pos({:7.3},{:7.3},{:7.3})", i, r[0], r[1], r[2], r[3], p.x, p.y, p.z);
        }
    }

    pub fn check_req_limits(&self) {
        let vmin = [0.2, 0.0, -1.0, -1.0];
        let vmax = [3.0, 0.2, 1.0, 1.0];
        let reqs = self.reqs.as_slice();
        let mut ok = true;
        for i in 0..self.natoms {
            let ri = reqs[i];
            for j in 0..4 {
                let v = ri[j];
                if v < vmin[j] || v > vmax[j] || v.is_nan() {
                    println!("REQs[{:3},{:1}] {:12.6} out of limits [{:12.6},{:12.6}]", i, j, v, vmin[j], vmax[j]);
                    ok = false;
                }
            }
        }
        if !ok {
            self.print_nonbonded(&vec![Vec3d::new(0.0,0.0,0.0); self.natoms]);
            panic!("ERROR NBFF::check_req_limits(): REQs are out of range ({:7.3},{:7.3},{:7.3},{:7.3}) .. ({:7.3},{:7.3},{:7.3},{:7.3}) => Exit()", vmin[0],vmin[1],vmin[2],vmin[3], vmax[0],vmax[1],vmax[2],vmax[3]);
        }
    }

    /// Evaluate non-bonded with PBC.
    pub fn eval_pbc(&mut self, fapos: &mut [Vec3d], apos: &[Vec3d]) -> f64 {
        let natoms = self.natoms;
        let npbc = self.npbc as usize;
        let shifts = &self.pbc_shifts;
        let excl = self.excl.as_slice();
        let r2damp = self.rdamp * self.rdamp;
        let f2max = self.fmax_nonbonded * self.fmax_nonbonded;
        let mut etot = 0.0;
        for ia in 0..natoms {
            let pi = apos[ia];
            let reqi = self.reqs.as_slice()[ia];
            let i0_ex = ia * EXCL_MAX;
            let mut iex = i0_ex;
            let iex_end = i0_ex + EXCL_MAX - 1;
            let mut jex = excl[iex];
            let mut fx = 0.0;
            let mut fy = 0.0;
            let mut fz = 0.0;
            for ja in 0..natoms {
                if ja == ia { continue; }
                if jex != -1 {
                    if iex < iex_end && jex < ja as i32 { iex += 1; }
                    jex = excl[iex];
                }
                if jex == ja as i32 { continue; }
                let reqj = self.reqs.as_slice()[ja];
                let reqij = Self::combine_req(reqj, reqi);
                let dp0 = Vec3d::set_sub(apos[ja], pi);
                for ipbc in 0..npbc {
                    let dp = Vec3d::set_add(dp0, shifts[ipbc]);
                    let (eij, mut fij) = Self::get_ljqh(dp, reqij, r2damp);
                    if self.b_clamp_nonbonded { Self::clamp_force_vec(&mut fij, f2max); }
                    etot += eij;
                    fx += fij.x;
                    fy += fij.y;
                    fz += fij.z;
                }
            }
            let fi = &mut fapos[ia];
            fi.x += fx; fi.y += fy; fi.z += fz;
        }
        etot
    }
}
