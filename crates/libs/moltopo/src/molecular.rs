use numcore::math::quat4::{Quat4i, Quat4d, QUAT4I_MINUS_ONES};
use numcore::math::vec3::{Vec3d, VEC3_ZERO};
use numcore::util::AlignedVec;

/// Static atomic data: geometry + types + neighbor lists.
/// Shared by all forcefield modules.
pub struct Atoms {
    pub natoms: i32,
    pub atypes: Vec<usize>,
    pub apos: AlignedVec<Vec3d, 64>,
    pub neighs: AlignedVec<Quat4i, 64>,
    pub neigh_bs: AlignedVec<Quat4i, 64>,
}

impl Atoms {
    pub fn new(natoms: i32) -> Self {
        let mut apos = AlignedVec::<Vec3d, 64>::new();
        apos.resize_fill(natoms as usize, Vec3d::default());
        let mut neighs = AlignedVec::<Quat4i, 64>::new();
        let mut neigh_bs = AlignedVec::<Quat4i, 64>::new();
        neighs.resize_fill(natoms as usize, QUAT4I_MINUS_ONES);
        neigh_bs.resize_fill(natoms as usize, QUAT4I_MINUS_ONES);
        Self {
            natoms,
            atypes: vec![0; natoms as usize],
            apos,
            neighs,
            neigh_bs,
        }
    }

    /// Build neighbor lists from bond topology.
    /// Call after loading topology / before creating forcefields.
    pub fn make_neigh_bs(&mut self, bon_atoms: &[[i32; 2]]) {
        let natoms = self.natoms as usize;
        for ia in 0..natoms {
            self.neighs.as_mut_slice()[ia] = QUAT4I_MINUS_ONES;
            self.neigh_bs.as_mut_slice()[ia] = QUAT4I_MINUS_ONES;
        }
        for (ib, b) in bon_atoms.iter().enumerate() {
            let ia = b[0] as usize;
            let ja = b[1] as usize;
            let (ngi, ngj) = {
                let neighs = self.neighs.as_mut_slice();
                if ia < ja {
                    let (lo, hi) = neighs.split_at_mut(ja);
                    (&mut lo[ia], &mut hi[0])
                } else {
                    let (lo, hi) = neighs.split_at_mut(ia);
                    (&mut hi[0], &mut lo[ja])
                }
            };
            let (ngbi, ngbj) = {
                let neigh_bs = self.neigh_bs.as_mut_slice();
                if ia < ja {
                    let (lo, hi) = neigh_bs.split_at_mut(ja);
                    (&mut lo[ia], &mut hi[0])
                } else {
                    let (lo, hi) = neigh_bs.split_at_mut(ia);
                    (&mut hi[0], &mut lo[ja])
                }
            };
            let mut ai = [ngi.x, ngi.y, ngi.z, ngi.w];
            let mut aj = [ngj.x, ngj.y, ngj.z, ngj.w];
            let mut bi = [ngbi.x, ngbi.y, ngbi.z, ngbi.w];
            let mut bj = [ngbj.x, ngbj.y, ngbj.z, ngbj.w];
            for s in 0..4 {
                if ai[s] < 0 { ai[s] = ja as i32; bi[s] = ib as i32; break; }
            }
            for s in 0..4 {
                if aj[s] < 0 { aj[s] = ia as i32; bj[s] = ib as i32; break; }
            }
            *ngi = Quat4i::new(ai[0], ai[1], ai[2], ai[3]);
            *ngj = Quat4i::new(aj[0], aj[1], aj[2], aj[3]);
            *ngbi = Quat4i::new(bi[0], bi[1], bi[2], bi[3]);
            *ngbj = Quat4i::new(bj[0], bj[1], bj[2], bj[3]);
        }
    }
}

/// Dynamic atomic data: Atoms + forces + velocities + MD integrators.
pub struct DynamicAtoms {
    pub atoms: Atoms,
    pub fapos: AlignedVec<Vec3d, 64>,
    pub vapos: AlignedVec<Vec3d, 64>,
}

impl DynamicAtoms {
    pub fn new(natoms: i32) -> Self {
        let atoms = Atoms::new(natoms);
        let mut fapos = AlignedVec::<Vec3d, 64>::new();
        let mut vapos = AlignedVec::<Vec3d, 64>::new();
        fapos.resize_fill(natoms as usize, VEC3_ZERO);
        vapos.resize_fill(natoms as usize, VEC3_ZERO);
        Self { atoms, fapos, vapos }
    }

    #[inline(always)] pub fn natoms(&self) -> usize { self.atoms.natoms as usize }
    #[inline(always)] pub fn apos(&self) -> &[Vec3d] { self.atoms.apos.as_slice() }
    #[inline(always)] pub fn apos_mut(&mut self) -> &mut [Vec3d] { self.atoms.apos.as_mut_slice() }
    #[inline(always)] pub fn fapos(&self) -> &[Vec3d] { self.fapos.as_slice() }
    #[inline(always)] pub fn fapos_mut(&mut self) -> &mut [Vec3d] { self.fapos.as_mut_slice() }
    #[inline(always)] pub fn vapos(&self) -> &[Vec3d] { self.vapos.as_slice() }
    #[inline(always)] pub fn vapos_mut(&mut self) -> &mut [Vec3d] { self.vapos.as_mut_slice() }
    #[inline(always)] pub fn neighs(&self) -> &[Quat4i] { self.atoms.neighs.as_slice() }
    #[inline(always)] pub fn neigh_bs(&self) -> &[Quat4i] { self.atoms.neigh_bs.as_slice() }

    #[inline(always)] pub fn clean_force(&mut self) {
        for f in self.fapos.as_mut_slice() { *f = VEC3_ZERO; }
    }

    #[inline(always)] pub fn clean_velocity(&mut self) {
        for v in self.vapos.as_mut_slice() { *v = VEC3_ZERO; }
    }

    /// Single atom MD step. Returns (v·f, v·v, f·f) for convergence/instability checks.
    #[inline(always)]
    pub fn move_atom_md(&mut self, i: usize, dt: f64, flim: f64, cdamp: f64) -> (f64, f64, f64) {
        let f = self.fapos.as_slice()[i];
        let v = self.vapos.as_slice()[i];
        let p = self.atoms.apos.as_slice()[i];
        let ff = v.dot(f);
        let vv = v.norm2();
        let f2 = f.norm2();
        let mut f_clamped = f;
        if f2 > flim * flim {
            f_clamped.mul(flim / f2.sqrt());
        }
        let mut v_new = v;
        v_new.mul(cdamp);
        v_new.add_mul(f_clamped, dt);
        let mut p_new = p;
        p_new.add_mul(v_new, dt);
        self.atoms.apos.as_mut_slice()[i] = p_new;
        self.vapos.as_mut_slice()[i] = v_new;
        (ff, vv, f2)
    }

    /// Run MD for niter steps or until force convergence.
    /// Takes a force-evaluation closure so it works with any forcefield composition.
    pub fn run_md<F>(&mut self, eval_forces: &mut F, niter: i32, dt: f64, fconv: f64, flim: f64, damping: f64) -> i32
    where F: FnMut(&mut DynamicAtoms) {
        let f2conv = fconv * fconv;
        let cdamp = { let c = 1.0 - damping; if c < 0.0 { 0.0 } else { c } };
        for _itr in 0..niter {
            eval_forces(self);
            let mut ff = 0.0;
            let mut vv = 0.0;
            let mut vf = 0.0;
            for ia in 0..self.natoms() {
                let (ff_, vv_, vf_) = self.move_atom_md(ia, dt, flim, cdamp);
                ff += ff_;
                vv += vv_;
                vf += vf_;
            }
            if ff < 0.0 {
                self.clean_velocity();
            }
            if vf < f2conv {
                return _itr + 1;
            }
        }
        niter
    }
}
