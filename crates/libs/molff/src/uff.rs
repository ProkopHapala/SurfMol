use numtypes::{Quat4d, Quat4i, QUAT4I_MINUS_ONES};
use numtypes::{Vec3d, VEC3D_ZERO as VEC3_ZERO};
use numtypes::AlignedVec;
use moltopo::topology::Topology;
use moltopo::params::{Params, uff_bond_length, uff_bond_k, uff_angle_sp3, bond_order_from_types, KCAL_TO_EV};

#[derive(Default)]
pub struct Buckets {
    pub cell_ns: Vec<i32>,
    pub cell_i0s: Vec<i32>,
    pub cell2obj: Vec<i32>,
    pub nobjs: i32,
}

impl Buckets {
    pub fn resize_cells(&mut self, ncells: i32) {
        let n = ncells as usize;
        self.cell_ns.resize(n, 0);
        self.cell_i0s.resize(n, 0);
    }
    pub fn resize_objs(&mut self, nobjs: i32) {
        self.nobjs = nobjs;
        self.cell2obj.resize(nobjs as usize, 0);
    }
    #[inline(always)] pub fn clean(&mut self) { for v in &mut self.cell_ns { *v = 0; } }
    pub fn update_offsets(&mut self) {
        let mut off = 0i32;
        for i in 0..self.cell_ns.len() {
            self.cell_i0s[i] = off;
            off += self.cell_ns[i];
        }
        assert!(off == self.nobjs, "Buckets.update_offsets off={} nobjs={}", off, self.nobjs);
        for v in &mut self.cell_ns { *v = 0; }
    }
    #[inline(always)] pub fn add_to_cell(&mut self, cell: i32, obj: i32) {
        let ic = cell as usize;
        let i0 = self.cell_i0s[ic];
        let i = i0 + self.cell_ns[ic];
        self.cell2obj[i as usize] = obj;
        self.cell_ns[ic] += 1;
    }
}

pub struct Uff {
    pub natoms: i32,
    pub nbonds: i32,
    pub nangles: i32,
    pub ndihedrals: i32,
    pub ninversions: i32,

    pub nf: i32,
    pub i0dih: i32,
    pub i0inv: i32,
    pub i0ang: i32,
    pub i0bon: i32,

    pub bon_atoms: AlignedVec<[i32; 2], 64>,
    pub ang_atoms: AlignedVec<[i32; 3], 64>,
    pub dih_atoms: AlignedVec<Quat4i, 64>,
    pub inv_atoms: AlignedVec<Quat4i, 64>,

    pub ang_ngs: AlignedVec<[i32; 2], 64>,
    pub dih_ngs: AlignedVec<[i32; 3], 64>,
    pub inv_ngs: AlignedVec<[i32; 3], 64>,

    pub hneigh: AlignedVec<Quat4d, 64>,

    pub fint: AlignedVec<Vec3d, 64>,
    pub fbon: AlignedVec<Vec3d, 64>,
    pub fang: AlignedVec<Vec3d, 64>,
    pub fdih: AlignedVec<Vec3d, 64>,
    pub finv: AlignedVec<Vec3d, 64>,

    pub a2f: Buckets,

    // --- Forcefield parameters
    pub bon_params: AlignedVec<[f64; 2], 64>, // [k, l0] per bond
    pub ang_params: AlignedVec<[f64; 5], 64>, // [k, c0, c1, c2, c3] per angle
    pub dih_params: AlignedVec<[f64; 3], 64>, // [V, d_sign, n] per dihedral
    pub inv_params: AlignedVec<[f64; 4], 64>, // [K, C0, C1, C2] per inversion

    // --- Control constants
    pub rdamp: f64,
    pub sub_nb_torsion_factor: f64,
}

impl Uff {
    pub fn new(natoms: i32, bonds: &[[i32; 2]], angles: &[[i32; 3]], dihedrals: &[Quat4i], inversions: &[Quat4i]) -> Self {
        let nbonds = bonds.len() as i32;
        let nangles = angles.len() as i32;
        let ndihedrals = dihedrals.len() as i32;
        let ninversions = inversions.len() as i32;

        let nf = ndihedrals * 4 + ninversions * 4 + nangles * 3 + nbonds;
        let i0dih = 0;
        let i0inv = i0dih + 4 * ndihedrals;
        let i0ang = i0inv + 4 * ninversions;
        let i0bon = i0ang + 3 * nangles;

        let mut bon_atoms = AlignedVec::<[i32; 2], 64>::new();
        bon_atoms.resize_fill(nbonds as usize, [-1, -1]);
        bon_atoms.as_mut_slice().copy_from_slice(bonds);

        let mut ang_atoms = AlignedVec::<[i32; 3], 64>::new();
        ang_atoms.resize_fill(nangles as usize, [-1, -1, -1]);
        ang_atoms.as_mut_slice().copy_from_slice(angles);

        let mut dih_atoms = AlignedVec::<Quat4i, 64>::new();
        dih_atoms.resize_fill(ndihedrals as usize, QUAT4I_MINUS_ONES);
        dih_atoms.as_mut_slice().copy_from_slice(dihedrals);

        let mut inv_atoms = AlignedVec::<Quat4i, 64>::new();
        inv_atoms.resize_fill(ninversions as usize, QUAT4I_MINUS_ONES);
        inv_atoms.as_mut_slice().copy_from_slice(inversions);

        let mut ang_ngs = AlignedVec::<[i32; 2], 64>::new();
        ang_ngs.resize_fill(nangles as usize, [-1, -1]);
        let mut dih_ngs = AlignedVec::<[i32; 3], 64>::new();
        dih_ngs.resize_fill(ndihedrals as usize, [-1, -1, -1]);
        let mut inv_ngs = AlignedVec::<[i32; 3], 64>::new();
        inv_ngs.resize_fill(ninversions as usize, [-1, -1, -1]);

        let mut hneigh = AlignedVec::<Quat4d, 64>::new();
        hneigh.resize_fill((natoms as usize) * 4, Quat4d::default());

        let mut fint = AlignedVec::<Vec3d, 64>::new();
        fint.resize_fill(nf as usize, VEC3_ZERO);
        let mut fbon = AlignedVec::<Vec3d, 64>::new();
        fbon.resize_fill(nbonds as usize, VEC3_ZERO);
        let mut fang = AlignedVec::<Vec3d, 64>::new();
        fang.resize_fill((nangles as usize) * 3, VEC3_ZERO);
        let mut fdih = AlignedVec::<Vec3d, 64>::new();
        fdih.resize_fill((ndihedrals as usize) * 4, VEC3_ZERO);
        let mut finv = AlignedVec::<Vec3d, 64>::new();
        finv.resize_fill((ninversions as usize) * 4, VEC3_ZERO);

        let mut bon_params = AlignedVec::<[f64; 2], 64>::new();
        bon_params.resize_fill(nbonds as usize, [0.0, 0.0]);
        let mut ang_params = AlignedVec::<[f64; 5], 64>::new();
        ang_params.resize_fill(nangles as usize, [0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut dih_params = AlignedVec::<[f64; 3], 64>::new();
        dih_params.resize_fill(ndihedrals as usize, [0.0, 0.0, 0.0]);
        let mut inv_params = AlignedVec::<[f64; 4], 64>::new();
        inv_params.resize_fill(ninversions as usize, [0.0, 0.0, 0.0, 0.0]);

        Self {
            natoms,
            nbonds,
            nangles,
            ndihedrals,
            ninversions,
            nf,
            i0dih,
            i0inv,
            i0ang,
            i0bon,
            bon_atoms,
            ang_atoms,
            dih_atoms,
            inv_atoms,
            ang_ngs,
            dih_ngs,
            inv_ngs,
            hneigh,
            fint,
            fbon,
            fang,
            fdih,
            finv,
            a2f: Buckets::default(),
            bon_params,
            ang_params,
            dih_params,
            inv_params,
            rdamp: 0.1,
            sub_nb_torsion_factor: 0.0,
        }
    }

    // ================== Diagnostic print functions (parity with C++ UFF.h) ==================

    pub fn print_sizes(&self) {
        println!("UFF::printSizes(): natoms({}) nbonds({}) nangles({}) ndihedrals({}) ninversions({})",
                 self.natoms, self.nbonds, self.nangles, self.ndihedrals, self.ninversions);
    }

    pub fn print_atom_params(&self, ia: usize, neighs: &[Quat4i]) {
        let n = neighs[ia];
        println!("atom[{:3}] neighs{{{:3},{:3},{:3},{:3}}}",
                 ia, n.x, n.y, n.z, n.w);
    }

    pub fn print_bond_params(&self, ib: usize) {
        let atoms = self.bon_atoms.as_slice()[ib];
        let p = self.bon_params.as_slice()[ib];
        println!("bond[{:3}] {{{}-{}}} K={:5.3} l0={:5.3}", ib, atoms[0], atoms[1], p[0], p[1]);
    }

    pub fn print_angle_params(&self, ia: usize) {
        let p = self.ang_params.as_slice()[ia];
        println!("angle[{:3}] K={:5.3} c0={:5.3} c1={:5.3} c2={:5.3} c3={:5.3}", ia, p[0], p[1], p[2], p[3], p[4]);
    }

    pub fn print_dihedral_params(&self, id: usize) {
        let p = self.dih_params.as_slice()[id];
        println!("dihedral[{:3}] V={:5.3} d={:5.3} n={:5.3}", id, p[0], p[1], p[2]);
    }

    pub fn print_inversion_params(&self, ii: usize) {
        let p = self.inv_params.as_slice()[ii];
        println!("inversion[{:3}] K={:5.3} c0={:5.3} c1={:5.3} c2={:5.3}", ii, p[0], p[1], p[2], p[3]);
    }

    pub fn print_all_params(&self, neighs: &[Quat4i], b_atoms: bool, b_bonds: bool, b_angles: bool, b_dihedrals: bool, b_inversions: bool) {
        if b_atoms {
            println!("\n=== Atoms ===");
            for i in 0..self.natoms as usize { self.print_atom_params(i, neighs); }
        }
        if b_bonds && self.nbonds > 0 {
            println!("\n=== Bonds ===");
            for i in 0..self.nbonds as usize { self.print_bond_params(i); }
        }
        if b_angles && self.nangles > 0 {
            println!("\n=== Angles ===");
            for i in 0..self.nangles as usize { self.print_angle_params(i); }
        }
        if b_dihedrals && self.ndihedrals > 0 {
            println!("\n=== Dihedrals ===");
            for i in 0..self.ndihedrals as usize { self.print_dihedral_params(i); }
        }
        if b_inversions && self.ninversions > 0 {
            println!("\n=== Inversions ===");
            for i in 0..self.ninversions as usize { self.print_inversion_params(i); }
        }
    }

    #[inline(always)]
    pub fn from_topology(top: &Topology) -> Self {
        Self::new(top.natoms(), &top.bonds, &top.angles, &top.dihedrals, &top.inversions)
    }

    #[inline(always)] pub fn clean_force(&mut self, fapos: &mut [Vec3d]) {
        for f in fapos { *f = VEC3_ZERO; }
        for f in self.fang.as_mut_slice() { *f = VEC3_ZERO; }
        for f in self.fdih.as_mut_slice() { *f = VEC3_ZERO; }
        for f in self.finv.as_mut_slice() { *f = VEC3_ZERO; }
        for f in self.fint.as_mut_slice() { *f = VEC3_ZERO; }
    }

    pub fn bake_angle_neighs(&mut self, neighs: &[Quat4i]) {
        for ia in 0..self.nangles as usize {
            let a = self.ang_atoms.as_slice()[ia];
            let i = a[0];
            let j = a[1];
            let k = a[2];
            let ings = neighs[j as usize].as_array();
            let mut ji = -1;
            let mut jk = -1;
            for s in 0..4 {
                let ing = ings[s];
                if ing < 0 { break; }
                if ing == i { ji = j * 4 + s as i32; }
                else if ing == k { jk = j * 4 + s as i32; }
            }
            self.ang_ngs.as_mut_slice()[ia] = [ji, jk];
        }
    }

    pub fn bake_dihedral_neighs(&mut self, neighs: &[Quat4i]) {
        for id in 0..self.ndihedrals as usize {
            let d = self.dih_atoms.as_slice()[id];
            let i = d.x;
            let j = d.y;
            let k = d.z;
            let l = d.w;
            let ingsj = neighs[j as usize].as_array();
            let ingsk = neighs[k as usize].as_array();
            let mut ji = -1;
            let mut jk = -1;
            let mut kl = -1;
            for s in 0..4 {
                let ing = ingsj[s];
                if ing < 0 { break; }
                if ing == i { ji = j * 4 + s as i32; }
                else if ing == k { jk = j * 4 + s as i32; }
            }
            for s in 0..4 {
                let ing = ingsk[s];
                if ing < 0 { break; }
                if ing == l { kl = k * 4 + s as i32; }
            }
            self.dih_ngs.as_mut_slice()[id] = [ji, jk, kl];
        }
    }

    pub fn bake_inversion_neighs(&mut self, neighs: &[Quat4i]) {
        for ii in 0..self.ninversions as usize {
            let v = self.inv_atoms.as_slice()[ii];
            let i = v.x;
            let j = v.y;
            let k = v.z;
            let l = v.w;
            let ings = neighs[i as usize].as_array();
            let mut ij = -1;
            let mut ik = -1;
            let mut il = -1;
            for s in 0..3 {
                let ing = ings[s];
                if ing == j { ij = i * 4 + s as i32; }
                else if ing == k { ik = i * 4 + s as i32; }
                else if ing == l { il = i * 4 + s as i32; }
            }
            self.inv_ngs.as_mut_slice()[ii] = [ij, ik, il];
        }
    }

    pub fn map_atom_interactions(&mut self) {
        self.a2f.resize_cells(self.natoms);
        // Match C++ UFF.h: we do NOT map bonds into a2f because bonds are evaluated in a per-atom loop.
        // Therefore a2f only needs to store dihedrals+inversions+angles pieces, i.e. indices < i0bon.
        self.a2f.resize_objs(self.i0bon);
        self.a2f.clean();

        for i in 0..self.ndihedrals as usize {
            let d = self.dih_atoms.as_slice()[i];
            self.a2f.cell_ns[d.x as usize] += 1;
            self.a2f.cell_ns[d.y as usize] += 1;
            self.a2f.cell_ns[d.z as usize] += 1;
            self.a2f.cell_ns[d.w as usize] += 1;
        }
        for i in 0..self.ninversions as usize {
            let v = self.inv_atoms.as_slice()[i];
            self.a2f.cell_ns[v.x as usize] += 1;
            self.a2f.cell_ns[v.y as usize] += 1;
            self.a2f.cell_ns[v.z as usize] += 1;
            self.a2f.cell_ns[v.w as usize] += 1;
        }
        for i in 0..self.nangles as usize {
            let a = self.ang_atoms.as_slice()[i];
            self.a2f.cell_ns[a[0] as usize] += 1;
            self.a2f.cell_ns[a[1] as usize] += 1;
            self.a2f.cell_ns[a[2] as usize] += 1;
        }

        self.a2f.update_offsets();

        for i in 0..self.ndihedrals as usize {
            let d = self.dih_atoms.as_slice()[i];
            let i0 = (i as i32) * 4 + self.i0dih;
            self.a2f.add_to_cell(d.x, i0);
            self.a2f.add_to_cell(d.y, i0 + 1);
            self.a2f.add_to_cell(d.z, i0 + 2);
            self.a2f.add_to_cell(d.w, i0 + 3);
        }
        for i in 0..self.ninversions as usize {
            let v = self.inv_atoms.as_slice()[i];
            let i0 = (i as i32) * 4 + self.i0inv;
            self.a2f.add_to_cell(v.x, i0);
            self.a2f.add_to_cell(v.y, i0 + 1);
            self.a2f.add_to_cell(v.z, i0 + 2);
            self.a2f.add_to_cell(v.w, i0 + 3);
        }
        for i in 0..self.nangles as usize {
            let a = self.ang_atoms.as_slice()[i];
            let i0 = (i as i32) * 3 + self.i0ang;
            self.a2f.add_to_cell(a[0], i0);
            self.a2f.add_to_cell(a[1], i0 + 1);
            self.a2f.add_to_cell(a[2], i0 + 2);
        }
    }

    #[inline(always)]
    pub fn assemble_atom_force(&mut self, ia: i32, fapos: &mut [Vec3d]) {
        let i0 = self.a2f.cell_i0s[ia as usize];
        let i1 = i0 + self.a2f.cell_ns[ia as usize];
        let mut f = fapos[ia as usize];
        for i in i0..i1 {
            let j = self.a2f.cell2obj[i as usize] as usize;
            f.add(self.fint.as_slice()[j]);
        }
        fapos[ia as usize] = f;
    }

    pub fn assemble_forces(&mut self, fapos: &mut [Vec3d]) {
        for ia in 0..self.natoms { self.assemble_atom_force(ia, fapos); }
    }

    pub fn update_hneigh(&mut self, apos: &[Vec3d], neighs: &[Quat4i]) {
        // Equivalent to having hneigh[j*4+slot].f = normalized bond vector, .w = invr.
        let hneigh = self.hneigh.as_mut_slice();
        for ia in 0..self.natoms as usize {
            let a = apos[ia];
            let ns = neighs[ia].as_array();
            for s in 0..4 {
                let ja = ns[s];
                let idx = ia * 4 + s;
                if ja < 0 { hneigh[idx] = Quat4d::default(); continue; }
                let b = apos[ja as usize];
                let mut d = Vec3d::set_sub(b, a);
                let r2 = d.norm2();
                let invr = 1.0 / r2.sqrt();
                d.mul(invr);
                hneigh[idx] = Quat4d::new(d.x, d.y, d.z, invr);
            }
        }
    }

    pub fn smoke_fill_term_forces(&mut self) {
        // Writes deterministic nonzero forces to term buffers and copies into fint in the same layout.
        for (i, f) in self.fdih.as_mut_slice().iter_mut().enumerate() { *f = Vec3d::new(0.001 * (i as f64 + 1.0), 0.0, 0.0); }
        for (i, f) in self.finv.as_mut_slice().iter_mut().enumerate() { *f = Vec3d::new(0.0, 0.001 * (i as f64 + 1.0), 0.0); }
        for (i, f) in self.fang.as_mut_slice().iter_mut().enumerate() { *f = Vec3d::new(0.0, 0.0, 0.001 * (i as f64 + 1.0)); }
        for (i, f) in self.fbon.as_mut_slice().iter_mut().enumerate() { *f = Vec3d::new(0.0001 * (i as f64 + 1.0), 0.0001, 0.0); }

        let fint = self.fint.as_mut_slice();
        for i in 0..self.ndihedrals as usize {
            let o = (i as i32) * 4 + self.i0dih;
            let src = &self.fdih.as_slice()[i * 4..i * 4 + 4];
            fint[o as usize + 0] = src[0]; fint[o as usize + 1] = src[1]; fint[o as usize + 2] = src[2]; fint[o as usize + 3] = src[3];
        }
        for i in 0..self.ninversions as usize {
            let o = (i as i32) * 4 + self.i0inv;
            let src = &self.finv.as_slice()[i * 4..i * 4 + 4];
            fint[o as usize + 0] = src[0]; fint[o as usize + 1] = src[1]; fint[o as usize + 2] = src[2]; fint[o as usize + 3] = src[3];
        }
        for i in 0..self.nangles as usize {
            let o = (i as i32) * 3 + self.i0ang;
            let src = &self.fang.as_slice()[i * 3..i * 3 + 3];
            fint[o as usize + 0] = src[0]; fint[o as usize + 1] = src[1]; fint[o as usize + 2] = src[2];
        }
        for i in 0..self.nbonds as usize {
            let o = (i as i32) + self.i0bon;
            fint[o as usize] = self.fbon.as_slice()[i];
        }
    }

    // ======================== REAL UFF FORCE EVALUATION ========================

    #[inline(always)]
    pub fn eval_atom_bonds(&mut self, ia: usize, apos: &[Vec3d], fapos: &mut [Vec3d], neighs: &[Quat4i], neigh_bs: &[Quat4i]) -> f64 {
        let hneigh = self.hneigh.as_mut_slice();
        let bon_params = self.bon_params.as_slice();

        let pa = apos[ia];
        let ings = neighs[ia].as_array();
        let inbs = neigh_bs[ia].as_array();
        let mut e = 0.0;
        for in_ in 0..4 {
            let ing = ings[in_];
            if ing < 0 { break; }
            let inn = ia * 4 + in_;
            let pb = apos[ing as usize];
            let mut dp = Vec3d::set_sub(pb, pa);
            let l = dp.norm();
            let invl = 1.0 / l;
            dp.mul(invl);
            hneigh[inn] = Quat4d::new(dp.x, dp.y, dp.z, invl);

            let ib = inbs[in_] as usize;
            let par = bon_params[ib];
            let dl = l - par[1];
            e += par[0] * dl * dl;
            let fr = 2.0 * par[0] * dl;
            let f = Vec3d::set_mul(dp, fr);
            fapos[ia].add(f);
        }
        e
    }

    #[inline(always)]
    pub fn eval_angle_prokop(&mut self, ia: usize) -> f64 {
        use numtypes::Vec2d;
        let ngs = self.ang_ngs.as_slice()[ia];
        if ngs[0] < 0 || ngs[1] < 0 { return 0.0; }
        let qij = self.hneigh.as_slice()[ngs[0] as usize]; // ji
        let qkj = self.hneigh.as_slice()[ngs[1] as usize]; // jk
        let h = Vec3d::set_add(qij.f(), qkj.f());
        let c = 0.5 * (h.norm2() - 2.0);
        let s = (1.0 - c * c + 1e-14).sqrt();
        let inv_sin = 1.0 / s;
        let par = self.ang_params.as_slice()[ia];
        let mut ee = par[1];
        let mut ff = par[2];
        let cs = Vec2d::new(c, s);
        let mut csn = cs;
        csn.mul_cmplx(cs); // (cos(2t), sin(2t))
        ee += par[3] * csn.x;
        ff += par[3] * csn.y * inv_sin * 2.0;
        csn.mul_cmplx(cs); // (cos(3t), sin(3t))
        ee += par[4] * csn.x;
        ff += par[4] * csn.y * inv_sin * 3.0;
        ee *= par[0];
        ff *= par[0];
        let fi = ff * qij.w;
        let fk = ff * qkj.w;
        let fic = fi * c;
        let fkc = fk * c;
        let fpi = Vec3d::set_lincomb(fic, qij.f(), -fi, qkj.f());
        let fpk = Vec3d::set_lincomb(-fk, qij.f(), fkc, qkj.f());
        let fpj = Vec3d::set_lincomb(fk - fic, qij.f(), fi - fkc, qkj.f());
        let i0 = (ia as i32) * 3 + self.i0ang;
        let fint = self.fint.as_mut_slice();
        fint[i0 as usize] = fpi;
        fint[i0 as usize + 1] = fpj;
        fint[i0 as usize + 2] = fpk;
        ee
    }

    #[inline(always)]
    pub fn eval_dihedral_prokop(&mut self, id: usize) -> f64 {
        use numtypes::Vec2d;
        let ngs = self.dih_ngs.as_slice()[id];
        if ngs[0] < 0 || ngs[1] < 0 || ngs[2] < 0 { return 0.0; }
        let q12 = self.hneigh.as_slice()[ngs[0] as usize]; // ji
        let q32 = self.hneigh.as_slice()[ngs[1] as usize]; // jk
        let q43 = self.hneigh.as_slice()[ngs[2] as usize]; // kl
        let n123 = Vec3d::cross(q12.f(), q32.f());
        let n234 = Vec3d::cross(q43.f(), q32.f());
        let il2_123 = 1.0 / n123.norm2();
        let il2_234 = 1.0 / n234.norm2();
        let inv_n12 = (il2_123 * il2_234).sqrt();
        let cs = Vec2d::new(
            n123.dot(n234) * inv_n12,
            -n123.dot(q43.f()) * inv_n12,
        );
        let mut csn = cs;
        let par = self.dih_params.as_slice()[id];
        let n = par[2] as i32;
        for _ in 1..n { csn.mul_cmplx(cs); }
        let ee = par[0] * (1.0 + par[1] * csn.x);
        let f = -par[0] * par[1] * par[2] * csn.y;
        let fp1 = Vec3d::set_mul(n123, -f * il2_123 * q12.w);
        let fp4 = Vec3d::set_mul(n234, f * il2_234 * q43.w);
        let c123 = q32.f().dot(q12.f()) * (q32.w / q12.w);
        let c432 = q32.f().dot(q43.f()) * (q32.w / q43.w);
        let fp3 = Vec3d::set_lincomb(-c123, fp1, -c432 - 1.0, fp4);
        let fp2 = Vec3d::set_lincomb(c123 - 1.0, fp1, c432, fp4);
        let i0 = (id as i32) * 4 + self.i0dih;
        let fint = self.fint.as_mut_slice();
        fint[i0 as usize] = fp1;
        fint[i0 as usize + 1] = fp2;
        fint[i0 as usize + 2] = fp3;
        fint[i0 as usize + 3] = fp4;
        ee
    }

    #[inline(always)]
    pub fn eval_inversion_prokop(&mut self, ii: usize) -> f64 {
        use numtypes::Vec2d;
        let ngs = self.inv_ngs.as_slice()[ii];
        if ngs[0] < 0 || ngs[1] < 0 || ngs[2] < 0 { return 0.0; }
        let q21 = self.hneigh.as_slice()[ngs[0] as usize]; // ji
        let q31 = self.hneigh.as_slice()[ngs[1] as usize]; // ki
        let q41 = self.hneigh.as_slice()[ngs[2] as usize]; // li
        let mut n123 = Vec3d::cross(q21.f(), q31.f());
        let il123 = 1.0 / n123.normalize();
        let s = -n123.dot(q41.f());
        let c = (1.0 - s * s + 1e-14).sqrt();
        let par = self.inv_params.as_slice()[ii];
        let cs = Vec2d::new(c, s);
        let mut cs2 = cs;
        cs2.mul_cmplx(cs);
        let ee = par[0] * (par[1] + par[2] * c + par[3] * cs2.x);
        let f = -par[0] * (par[2] * s + 2.0 * par[3] * cs2.y) / c;
        let fq41 = f * q41.w;
        let fi123 = f * il123;
        let tq = Vec3d::set_lincomb(s * fi123, n123, fi123, q41.f());
        let fp4 = Vec3d::set_lincomb(fq41, n123, s * fq41, q41.f());
        let fp2 = Vec3d::set_mul(Vec3d::cross(q31.f(), tq), q21.w);
        let fp3 = Vec3d::set_mul(Vec3d::cross(tq, q21.f()), q31.w);
        let fp1 = Vec3d::set_mul(Vec3d::set_add(Vec3d::set_add(fp2, fp3), fp4), -1.0);
        let i0 = (ii as i32) * 4 + self.i0inv;
        let fint = self.fint.as_mut_slice();
        fint[i0 as usize] = fp1;
        fint[i0 as usize + 1] = fp2;
        fint[i0 as usize + 2] = fp3;
        fint[i0 as usize + 3] = fp4;
        ee
    }

    // ======================== MD INTEGRATOR ========================

    #[inline(always)]
    pub fn move_atom_md(&mut self, i: usize, apos: &mut [Vec3d], fapos: &[Vec3d], vapos: &mut [Vec3d], dt: f64, flim: f64, cdamp: f64) -> (f64, f64, f64) {
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
        (ff, vv, f2)
    }

    pub fn clean_velocity(&mut self, vapos: &mut [Vec3d]) {
        for v in vapos { *v = VEC3_ZERO; }
    }

    // ======================== HIGH-LEVEL MD LOOP ========================

    pub fn eval_forces(&mut self, apos: &[Vec3d], fapos: &mut [Vec3d], neighs: &[Quat4i], neigh_bs: &[Quat4i]) -> (f64, f64, f64, f64) {
        let mut eb = 0.0;
        let mut ea = 0.0;
        let mut ed = 0.0;
        let mut ei = 0.0;
        // Zero fapos
        for i in 0..fapos.len() { fapos[i] = VEC3_ZERO; }
        // Bonds (also updates hneigh)
        for ia in 0..self.natoms as usize {
            eb += self.eval_atom_bonds(ia, apos, fapos, neighs, neigh_bs);
        }
        // Angles, dihedrals, inversions -> fint
        for ia in 0..self.nangles as usize {
            ea += self.eval_angle_prokop(ia);
        }
        for id in 0..self.ndihedrals as usize {
            ed += self.eval_dihedral_prokop(id);
        }
        for ii in 0..self.ninversions as usize {
            ei += self.eval_inversion_prokop(ii);
        }
        // Assemble fint into fapos
        for ia in 0..self.natoms {
            self.assemble_atom_force(ia, fapos);
        }
        (eb, ea, ed, ei)
    }

    /// Set minimal demo parameters based on current geometry.
    /// Not real UFF parameters — just enough to exercise the force kernels.
    pub fn set_dummy_params(&mut self, apos: &[Vec3d]) {
        // Bonds: k=100, l0 = 0.95 * current length (gentle initial strain)
        for ib in 0..self.nbonds as usize {
            let b = self.bon_atoms.as_slice()[ib];
            let ia = b[0] as usize;
            let ja = b[1] as usize;
            let d = Vec3d::set_sub(apos[ja], apos[ia]);
            let l0 = d.norm() * 0.95;
            self.bon_params.as_mut_slice()[ib] = [100.0, l0];
        }
        // Angles: disabled for stable demo (k=0)
        for ia in 0..self.nangles as usize {
            self.ang_params.as_mut_slice()[ia] = [0.0, 1.0, -1.0, 0.0, 0.0];
        }
        // Dihedrals: disabled
        for id in 0..self.ndihedrals as usize {
            self.dih_params.as_mut_slice()[id] = [0.0, 1.0, 3.0];
        }
        // Inversions: disabled
        for ii in 0..self.ninversions as usize {
            self.inv_params.as_mut_slice()[ii] = [0.0, 1.0, -1.0, 0.0];
        }
    }

    // ======================== REAL UFF PARAMETER ASSIGNMENT ========================
    // Ported from FireCore cpp/common/molecular/UFFbuilder.h:assignUFFparams
    // Fills bon_params, ang_params, dih_params, inv_params from Params + assigned UFF types.

    /// Hybridization suffix character from UFF atom type name: '1', '2', 'R', or '3'.
    /// Ported from FireCore UFFbuilder.h:1106 (params->atypes[...].name[2]).
    #[inline] fn hyb_suffix(tname: &str) -> char {
        if let Some(i) = tname.rfind('_') {
            if i + 1 < tname.len() { return tname.chars().nth(i + 1).unwrap_or('3'); }
        }
        '3'
    }

    /// Element symbol from UFF atom type name (everything before '_').
    #[inline] fn elem_sym(tname: &str) -> &str {
        if let Some(i) = tname.find('_') { &tname[..i] } else { tname }
    }

    /// Compute UFF bond length rIJ for a bond between atoms with types ta, tb.
    /// Ported from FireCore UFFbuilder.h:1043 assignUFFparams_calcrij.
    fn calc_rij(params: &Params, ta: &str, tb: &str, bo: f64) -> f64 {
        let (ti, ei) = match (params.get_atom_type(ta), params.element_of_atom_type(ta)) {
            (Some(t), Some(e)) => (t, e), _ => return 1.5,
        };
        let (tj, ej) = match (params.get_atom_type(tb), params.element_of_atom_type(tb)) {
            (Some(t), Some(e)) => (t, e), _ => return 1.5,
        };
        uff_bond_length(ti, tj, ei, ej, bo)
    }

    /// Assign real UFF parameters from atom types + topology.
    /// `types` = assigned UFF atom type names (e.g. "C_R", "H_", "O_2").
    /// `neighs` = per-atom neighbor lists (4 per atom, -1 padded).
    /// Ported from FireCore UFFbuilder.h:assignUFFparams (lines 1336-1362).
    pub fn setup_params(&mut self, params: &Params, types: &[String], neighs: &[Quat4i]) {
        let natoms = self.natoms as usize;
        assert_eq!(types.len(), natoms, "Uff::setup_params: types.len()={} != natoms={}", types.len(), natoms);
        assert_eq!(neighs.len(), natoms, "Uff::setup_params: neighs.len()={} != natoms={}", neighs.len(), natoms);

        // --- Bonds (UFFbuilder.h:1068) ---
        for ib in 0..self.nbonds as usize {
            let b = self.bon_atoms.as_slice()[ib];
            let ta = &types[b[0] as usize];
            let tb = &types[b[1] as usize];
            let bo = bond_order_from_types(ta, tb);
            let l0 = Self::calc_rij(params, ta, tb, bo);
            let (ei, ej) = match (params.element_of_atom_type(ta), params.element_of_atom_type(tb)) {
                (Some(ei), Some(ej)) => (ei, ej),
                _ => { self.bon_params.as_mut_slice()[ib] = [100.0, l0]; continue; }
            };
            let k = uff_bond_k(ei, ej, l0);
            self.bon_params.as_mut_slice()[ib] = [k, l0];
        }

        // --- Angles (UFFbuilder.h:1080) ---
        // ang_atoms = [i, j, k] where j is the central atom.
        // kappa = 28.7989 * Qi * Qk / rik^5 * (3*rij*rjk*st2 - rik^2*ct)
        // Then c0..c3 depend on hybridization of central atom j.
        let deg2rad = std::f64::consts::PI / 180.0;
        for ia in 0..self.nangles as usize {
            let a = self.ang_atoms.as_slice()[ia];
            let (i, j, k) = (a[0] as usize, a[1] as usize, a[2] as usize);
            let tj = &types[j];
            let ti_name = &types[i];
            let tk_name = &types[k];
            // Skip if central atom is H (no angle term)
            if Self::elem_sym(tj) == "H" {
                self.ang_params.as_mut_slice()[ia] = [0.0, 0.0, 0.0, 0.0, 0.0];
                continue;
            }
            let at_j = match params.get_atom_type(tj) {
                Some(at) => at, None => { self.ang_params.as_mut_slice()[ia] = [0.0, 0.0, 0.0, 0.0, 0.0]; continue; }
            };
            let ei = match params.element_of_atom_type(ti_name) {
                Some(e) => e, None => { self.ang_params.as_mut_slice()[ia] = [0.0, 0.0, 0.0, 0.0, 0.0]; continue; }
            };
            let ek = match params.element_of_atom_type(tk_name) {
                Some(e) => e, None => { self.ang_params.as_mut_slice()[ia] = [0.0, 0.0, 0.0, 0.0, 0.0]; continue; }
            };
            let ct = (at_j.a_ss * deg2rad).cos();
            let st2 = (at_j.a_ss * deg2rad).sin().powi(2);
            let bo_ij = bond_order_from_types(ti_name, tj);
            let bo_jk = bond_order_from_types(tj, tk_name);
            let rij = Self::calc_rij(params, ti_name, tj, bo_ij);
            let rjk = Self::calc_rij(params, tj, tk_name, bo_jk);
            let rik = (rij * rij + rjk * rjk - 2.0 * rij * rjk * ct).sqrt();
            let kappa = 28.7989689090648 * ei.q_uff * ek.q_uff / (rik.powi(5)) * (3.0 * rij * rjk * st2 - rik * rik * ct);
            let hyb = Self::hyb_suffix(tj);
            let (k_ang, c0, c1, c2, c3) = match hyb {
                '1' => (kappa, 1.0, 1.0, 0.0, 0.0),         // sp1 (UFFbuilder.h:1108-1114)
                '2' | 'R' => (kappa / 9.0, 1.0, 0.0, 0.0, -1.0), // sp2/aromatic (UFFbuilder.h:1115-1120)
                '3' => {                                      // sp3 (UFFbuilder.h:1122-1127)
                    let (c0, c1, c2, c3) = uff_angle_sp3(ct, st2);
                    (kappa, c0, c1, c2, c3)
                }
                _ => (kappa, 1.0, 0.0, 0.0, -1.0),           // fallback: treat as sp2
            };
            self.ang_params.as_mut_slice()[ia] = [k_ang, c0, c1, c2, c3];
        }

        // --- Dihedrals (UFFbuilder.h:1136) ---
        // dih_atoms = [i1, i2, i3, i4] where i2-i3 is the central bond.
        // V, d, n depend on hybridization of i2 and i3.
        // V is divided by 0.5*(nb2-1)*(nb3-1) where nb2, nb3 are neighbor counts of i2, i3.
        for id in 0..self.ndihedrals as usize {
            let d = self.dih_atoms.as_slice()[id];
            let (i1, i2, i3, i4) = (d.x as usize, d.y as usize, d.z as usize, d.w as usize);
            let t2 = &types[i2];
            let t3 = &types[i3];
            // Skip if either central atom is H or sp1 (UFFbuilder.h:1147-1148,1155)
            if Self::elem_sym(t2) == "H" || Self::elem_sym(t3) == "H" {
                self.dih_params.as_mut_slice()[id] = [0.0, 1.0, 3.0]; continue;
            }
            let h2 = Self::hyb_suffix(t2);
            let h3 = Self::hyb_suffix(t3);
            if h2 == '1' || h3 == '1' {
                self.dih_params.as_mut_slice()[id] = [0.0, 1.0, 3.0]; continue;
            }
            let bsp3_2 = h2 == '3';
            let bsp3_3 = h3 == '3';
            let e2 = params.element_of_atom_type(t2);
            let e3 = params.element_of_atom_type(t3);
            let bo_23 = bond_order_from_types(t2, t3);
            let (v, d_sign, n): (f64, f64, f64) = if bsp3_2 && bsp3_3 {
                // * - sp3 - sp3 - * (UFFbuilder.h:1174-1188)
                let v = (e2.map(|e| e.v_uff).unwrap_or(0.0) * e3.map(|e| e.v_uff).unwrap_or(0.0)).sqrt();
                // Special case: group 16 sp3 - sp3 (O, S)
                let el2 = Self::elem_sym(t2);
                let el3 = Self::elem_sym(t3);
                if (el2 == "O" || el2 == "S") && (el3 == "O" || el3 == "S") {
                    let mut k = KCAL_TO_EV; // 1 kcal/mol
                    if el2 == "O" { k *= 2.0; } else { k *= 6.8; }
                    if el3 == "O" { k *= 2.0; } else { k *= 6.8; }
                    (k.sqrt(), 1.0, 2.0)
                } else {
                    (v, 1.0, 3.0)
                }
            } else if (bsp3_2 && (h3 == '2' || h3 == 'R')) || ((h2 == '2' || h2 == 'R') && bsp3_3) {
                // * - sp3 - sp2 - * (UFFbuilder.h:1190-1234)
                let mut v = KCAL_TO_EV; // 1 kcal/mol
                let mut d_sign = -1.0;
                let mut n = 6.0;
                // Special case: group 16 sp3 - sp2 (UFFbuilder.h:1197-1204)
                let el_sp3 = if bsp3_2 { Self::elem_sym(t2) } else { Self::elem_sym(t3) };
                if el_sp3 == "O" || el_sp3 == "S" {
                    v = 5.0 * (e2.map(|e| e.u_uff).unwrap_or(0.0) * e3.map(|e| e.u_uff).unwrap_or(0.0)).sqrt() * (1.0 + 4.18 * bo_23.ln());
                    d_sign = 1.0;
                    n = 2.0;
                }
                // Special case: sp3 bonded to another sp2 (UFFbuilder.h:1206-1233)
                let (sp3_idx, sp2_idx) = if bsp3_2 { (i2, i3) } else { (i3, i2) };
                let sp2_neighs = neighs[sp2_idx].as_array();
                let has_sp2_neighbor = sp2_neighs.iter().take_while(|&&n| n >= 0).any(|&n| {
                    if n as usize == sp3_idx { return false; }
                    let hn = Self::hyb_suffix(&types[n as usize]);
                    hn == '2' || hn == 'R'
                });
                if has_sp2_neighbor {
                    v = 2.0 * KCAL_TO_EV; // 2 kcal/mol
                    d_sign = 1.0;
                    n = 3.0;
                }
                (v, d_sign, n)
            } else if (h2 == '2' || h2 == 'R') && (h3 == '2' || h3 == 'R') {
                // * - sp2 - sp2 - * (UFFbuilder.h:1236-1243)
                let v = 5.0 * (e2.map(|e| e.u_uff).unwrap_or(0.0) * e3.map(|e| e.u_uff).unwrap_or(0.0)).sqrt() * (1.0 + 4.18 * bo_23.ln());
                (v, -1.0, 2.0)
            } else {
                // Unknown case — disable (should not happen for well-typed molecules)
                eprintln!("WARNING: Uff::setup_params: dihedral case not found for {}-{}-{}-{} (types {} {} {} {})",
                    i1, i2, i3, i4, types[i1], types[i2], types[i3], types[i4]);
                (0.0, 1.0, 3.0)
            };
            // Divide by 0.5*(nb2-1)*(nb3-1) (UFFbuilder.h:1249)
            let nb2 = neighs[i2].as_array().iter().take_while(|&&n| n >= 0).count() as f64;
            let nb3 = neighs[i3].as_array().iter().take_while(|&&n| n >= 0).count() as f64;
            let denom = 0.5 * (nb2 - 1.0) * (nb3 - 1.0);
            let v_final = if denom > 0.0 { v / denom } else { v };
            self.dih_params.as_mut_slice()[id] = [v_final, d_sign, n];
        }

        // --- Inversions (UFFbuilder.h:1260-1334) ---
        // inv_atoms = [i1, i2, i3, i4] where i1 is the central trigonal atom.
        // 3 inversions per center (generated by build_inversions_from_bonds).
        // K is divided by 3 to avoid triple-counting.
        for ii in 0..self.ninversions as usize {
            let inv = self.inv_atoms.as_slice()[ii];
            let i1 = inv.x as usize;
            let t1 = &types[i1];
            let el1 = Self::elem_sym(t1);
            let h1 = Self::hyb_suffix(t1);
            let (k_inv, c0, c1, c2) = if el1 == "C" && (h1 == '2' || h1 == 'R') {
                // sp2 carbon (UFFbuilder.h:1276-1285)
                // Check for carbonyl (C=O neighbor)
                let neighbors = neighs[i1].as_array();
                let is_carbonyl = neighbors.iter().take_while(|&&n| n >= 0).any(|&n| {
                    let tn = &types[n as usize];
                    Self::elem_sym(tn) == "O" && Self::hyb_suffix(tn) == '2'
                });
                let k = if is_carbonyl { 50.0 * KCAL_TO_EV } else { 6.0 * KCAL_TO_EV };
                (k, 1.0, -1.0, 0.0)
            } else if el1 == "N" && (h1 == '2' || h1 == 'R') {
                // sp2 nitrogen (UFFbuilder.h:1288-1292)
                (6.0 * KCAL_TO_EV, 1.0, -1.0, 0.0)
            } else if el1 == "N" && h1 == '3' {
                // sp3 nitrogen — no inversion (UFFbuilder.h:1295-1299)
                (0.0, 0.0, 0.0, 0.0)
            } else if el1 == "P" {
                // Group 15 (UFFbuilder.h:1302-1307)
                let omega0 = 84.4339 * deg2rad;
                let c0 = 4.0 * omega0.cos().powi(2) - (2.0 * omega0).cos();
                let c1 = -4.0 * omega0.cos();
                let c2 = 1.0;
                let k = 22.0 * KCAL_TO_EV / (c0 + c1 + c2);
                (k, c0, c1, c2)
            } else {
                // No inversion for this atom type — disable
                (0.0, 0.0, 0.0, 0.0)
            };
            // Divide K by 3 (UFFbuilder.h:1313) — 3 inversions per center
            self.inv_params.as_mut_slice()[ii] = [k_inv / 3.0, c0, c1, c2];
        }
    }

}
