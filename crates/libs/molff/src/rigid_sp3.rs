use numtypes::{Quat4d, Quat4i};
use numtypes::{Vec3d, VEC3D_ZERO as VEC3_ZERO};
use crate::uff::Uff;

pub struct RigidSp3FF {
    pub quat: Vec<Quat4d>,
    pub omega: Vec<Vec3d>,
    pub tau: Vec<Vec3d>,
    pub k_scale: f64,
    pub rot_damp: f64,

    pub nport: Vec<u8>,
    pub port_local: Vec<Vec3d>,
}

impl RigidSp3FF {
    #[inline(always)] pub fn new(natoms: usize) -> Self {
        Self {
            quat: vec![Quat4d::new(0.0, 0.0, 0.0, 1.0); natoms],
            omega: vec![VEC3_ZERO; natoms],
            tau: vec![VEC3_ZERO; natoms],
            k_scale: 1.0,
            rot_damp: 0.05,

            nport: vec![4u8; natoms],
            port_local: vec![VEC3_ZERO; natoms * 4],
        }
    }

    #[inline(always)] pub fn from_uff(natoms: usize) -> Self {
        let mut out = Self::new(natoms);
        for i in 0..natoms {
            out.set_sp3(i);
        }
        out
    }

    #[inline(always)] fn quat_normalize(mut q: Quat4d) -> Quat4d {
        let n2 = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
        if n2 > 0.0 {
            let inv = 1.0 / n2.sqrt();
            q.x *= inv; q.y *= inv; q.z *= inv; q.w *= inv;
        }
        q
    }

    #[inline(always)] fn quat_mul(a: Quat4d, b: Quat4d) -> Quat4d {
        Quat4d::new(
            a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
            a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
            a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
            a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        )
    }

    #[inline(always)] fn quat_conj(q: Quat4d) -> Quat4d { Quat4d::new(-q.x, -q.y, -q.z, q.w) }

    #[inline(always)] fn quat_rotate(q: Quat4d, v: Vec3d) -> Vec3d {
        let qv = Quat4d::new(v.x, v.y, v.z, 0.0);
        let r = Self::quat_mul(Self::quat_mul(q, qv), Self::quat_conj(q));
        Vec3d::new(r.x, r.y, r.z)
    }

    #[inline(always)] fn quat_from_omega_dt(omega: Vec3d, dt: f64) -> Quat4d {
        let w = omega.norm();
        if w < 1e-12 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); }
        let angle = w * dt;
        let s = (0.5 * angle).sin();
        let c = (0.5 * angle).cos();
        let invw = 1.0 / w;
        Quat4d::new(omega.x*invw*s, omega.y*invw*s, omega.z*invw*s, c)
    }

    #[inline(always)] fn set_sp3(&mut self, i: usize) {
        const INV_SQRT3: f64 = 0.57735026918962576451;
        self.nport[i] = 4;
        let o = i * 4;
        self.port_local[o + 0] = Vec3d::new( 1.0*INV_SQRT3,  1.0*INV_SQRT3,  1.0*INV_SQRT3);
        self.port_local[o + 1] = Vec3d::new( 1.0*INV_SQRT3, -1.0*INV_SQRT3, -1.0*INV_SQRT3);
        self.port_local[o + 2] = Vec3d::new(-1.0*INV_SQRT3,  1.0*INV_SQRT3, -1.0*INV_SQRT3);
        self.port_local[o + 3] = Vec3d::new(-1.0*INV_SQRT3, -1.0*INV_SQRT3,  1.0*INV_SQRT3);
    }

    #[inline(always)] fn set_sp2(&mut self, i: usize) {
        self.nport[i] = 3;
        let o = i * 4;
        self.port_local[o + 0] = Vec3d::new(1.0, 0.0, 0.0);
        self.port_local[o + 1] = Vec3d::new(-0.5, 0.8660254037844386, 0.0);
        self.port_local[o + 2] = Vec3d::new(-0.5, -0.8660254037844386, 0.0);
        self.port_local[o + 3] = VEC3_ZERO;
    }

    #[inline(always)] fn set_sp1(&mut self, i: usize) {
        self.nport[i] = 2;
        let o = i * 4;
        self.port_local[o + 0] = Vec3d::new(1.0, 0.0, 0.0);
        self.port_local[o + 1] = Vec3d::new(-1.0, 0.0, 0.0);
        self.port_local[o + 2] = VEC3_ZERO;
        self.port_local[o + 3] = VEC3_ZERO;
    }

    #[inline(always)] fn set_point(&mut self, i: usize) {
        self.nport[i] = 0;
        let o = i * 4;
        self.port_local[o + 0] = VEC3_ZERO;
        self.port_local[o + 1] = VEC3_ZERO;
        self.port_local[o + 2] = VEC3_ZERO;
        self.port_local[o + 3] = VEC3_ZERO;
    }

    pub fn set_port_geometry_from_types(&mut self, uff_types: &[String]) {
        assert_eq!(uff_types.len(), self.quat.len());
        for i in 0..uff_types.len() {
            let t = uff_types[i].as_str();
            if matches!(t, "C_R"|"C_2"|"N_R"|"O_2"|"O_R") {
                self.set_sp2(i);
            } else if matches!(t, "C_1"|"N_1") {
                self.set_sp1(i);
            } else if t == "H_" {
                self.set_point(i);
            } else {
                self.set_sp3(i);
            }
        }
    }

    #[inline(always)] pub fn get_port_tip(&self, apos: &[Vec3d], i: usize, slot: usize, l0: f64) -> Vec3d {
        let r0 = self.port_local[i * 4 + slot] * l0;
        apos[i] + Self::quat_rotate(self.quat[i], r0)
    }

    pub fn eval_forces(&mut self, apos: &[Vec3d], fapos: &mut [Vec3d], uff: &Uff, neighs: &[Quat4i], neigh_bs: &[Quat4i]) -> f64 {
        let natoms = uff.natoms as usize;
        assert_eq!(self.quat.len(), natoms);

        for i in 0..fapos.len() { fapos[i] = VEC3_ZERO; }
        for t in &mut self.tau { *t = VEC3_ZERO; }

        let bon_params = uff.bon_params.as_slice();

        let mut e = 0.0;

        for i in 0..natoms {
            let xi = apos[i];
            let q = self.quat[i];
            let ns = neighs[i].as_array();
            let bs = neigh_bs[i].as_array();

            let np = self.nport[i] as usize;

            for s in 0..4 {
                let j = ns[s];
                if j < 0 { continue; }
                let ib = bs[s];
                if ib < 0 { continue; }
                if s >= np { continue; }

                let par = bon_params[ib as usize];
                let k = par[0] * self.k_scale;
                if k <= 0.0 { continue; }
                let l0 = par[1];

                let r0 = self.port_local[i * 4 + s] * l0;
                let r = Self::quat_rotate(q, r0);
                let tip = xi + r;

                let xj = apos[j as usize];
                let diff = xj - tip;
                let f = diff * (0.5 * k);

                fapos[i].add(f);
                fapos[j as usize].sub(f);

                self.tau[i].add(Vec3d::cross(r, f));

                e += 0.25 * k * diff.norm2();
            }
        }

        e
    }

    #[inline(always)] pub fn move_atom_md(&mut self, i: usize, apos: &mut [Vec3d], fapos: &[Vec3d], vapos: &mut [Vec3d], uff: &Uff, neigh_bs: &[Quat4i], dt: f64, flim: f64, cdamp: f64) -> (f64, f64, f64) {
        let f = fapos[i];
        let v = vapos[i];
        let p = apos[i];
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
        apos[i] = p_new;
        vapos[i] = v_new;

        let mut inv_i = 0.0;
        {
            let bs = neigh_bs[i].as_array();
            let mut sum_l2 = 0.0;
            let mut n = 0.0;
            for s in 0..4 {
                let ib = bs[s];
                if ib < 0 { continue; }
                let l0 = uff.bon_params.as_slice()[ib as usize][1];
                sum_l2 += l0 * l0;
                n += 1.0;
            }
            if n > 0.0 {
                let l2 = sum_l2 / n;
                let i_mom = 0.4 * l2 + 1e-18;
                inv_i = 1.0 / i_mom;
            }
        }

        if inv_i > 0.0 {
            let tau = self.tau[i];
            let mut w = self.omega[i];
            let c = 1.0 - self.rot_damp;
            w.mul(if c < 0.0 { 0.0 } else { c });
            w.add_mul(tau, inv_i * dt);
            self.omega[i] = w;

            let dq = Self::quat_from_omega_dt(w, dt);
            let q_old = self.quat[i];
            let q_new = Self::quat_normalize(Self::quat_mul(dq, q_old));
            self.quat[i] = q_new;
        }

        (ff, vv, f2)
    }
}
