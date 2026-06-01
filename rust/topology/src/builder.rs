use surfmol_common::math::vec3::Vec3d;
use crate::topology::{Topology, build_angles_from_bonds, build_dihedrals_from_bonds, build_inversions_from_bonds};

#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, Debug)]
pub struct AtomH { pub idx: u32, pub gen: u32 }

#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, Debug)]
pub struct BondH { pub idx: u32, pub gen: u32 }

struct Slot<T> { gen: u32, val: Option<T> }

#[derive(Clone, Copy)]
pub struct AtomData {
    pub pos: Vec3d,
    pub neigh_bonds: [BondH; 4],
    pub nbond: u8,
}

#[derive(Clone, Copy)]
pub struct BondData {
    pub a: AtomH,
    pub b: AtomH,
}

pub struct Builder {
    atoms: Vec<Slot<AtomData>>,
    bonds: Vec<Slot<BondData>>,
    free_atoms: Vec<u32>,
    free_bonds: Vec<u32>,
    pub dirty_derived: bool,
}

impl Builder {
    pub fn new() -> Self {
        Self { atoms: Vec::new(), bonds: Vec::new(), free_atoms: Vec::new(), free_bonds: Vec::new(), dirty_derived: true }
    }

    #[inline(always)] fn slot_get<T>(slots: &Vec<Slot<T>>, h: u32, g: u32) -> &T {
        let s = &slots[h as usize];
        assert!(s.gen == g && s.val.is_some(), "stale handle idx={} gen={} slot.gen={} alive={}", h, g, s.gen, s.val.is_some());
        s.val.as_ref().unwrap()
    }
    #[inline(always)] fn slot_get_mut<T>(slots: &mut Vec<Slot<T>>, h: u32, g: u32) -> &mut T {
        let s = &mut slots[h as usize];
        assert!(s.gen == g && s.val.is_some(), "stale handle idx={} gen={} slot.gen={} alive={}", h, g, s.gen, s.val.is_some());
        s.val.as_mut().unwrap()
    }

    pub fn add_atom(&mut self, pos: Vec3d) -> AtomH {
        let data = AtomData { pos, neigh_bonds: [BondH::default(); 4], nbond: 0 };
        if let Some(i) = self.free_atoms.pop() {
            let s = &mut self.atoms[i as usize];
            s.val = Some(data);
            self.dirty_derived = true;
            AtomH { idx: i, gen: s.gen }
        } else {
            let idx = self.atoms.len() as u32;
            self.atoms.push(Slot { gen: 1, val: Some(data) });
            self.dirty_derived = true;
            AtomH { idx, gen: 1 }
        }
    }

    pub fn remove_atom(&mut self, a: AtomH) {
        let alive = {
            let s = &self.atoms[a.idx as usize];
            s.gen == a.gen && s.val.is_some()
        };
        assert!(alive, "remove_atom: stale handle {:?}", a);

        // remove incident bonds (iterate local copy because we will mutate nbond)
        let neighs: [BondH; 4] = self.atom(a).neigh_bonds;
        for bh in neighs {
            if bh.gen == 0 { continue; }
            self.remove_bond(bh);
        }

        let s = &mut self.atoms[a.idx as usize];
        s.val = None;
        s.gen += 1;
        self.free_atoms.push(a.idx);
        self.dirty_derived = true;
    }

    #[inline(always)] pub fn atom(&self, a: AtomH) -> &AtomData { Self::slot_get(&self.atoms, a.idx, a.gen) }
    #[inline(always)] pub fn atom_mut(&mut self, a: AtomH) -> &mut AtomData { Self::slot_get_mut(&mut self.atoms, a.idx, a.gen) }
    #[inline(always)] pub fn bond(&self, b: BondH) -> &BondData { Self::slot_get(&self.bonds, b.idx, b.gen) }

    fn add_bond_to_atom(&mut self, a: AtomH, b: BondH) {
        let ad = self.atom_mut(a);
        let n = ad.nbond as usize;
        assert!(n < 4, "atom {:?} exceeds max neighbors=4", a);
        ad.neigh_bonds[n] = b;
        ad.nbond += 1;
    }

    fn remove_bond_from_atom(&mut self, a: AtomH, b: BondH) {
        let ad = self.atom_mut(a);
        let n = ad.nbond as usize;
        for i in 0..n {
            if ad.neigh_bonds[i] == b {
                ad.neigh_bonds[i] = ad.neigh_bonds[n - 1];
                ad.neigh_bonds[n - 1] = BondH::default();
                ad.nbond -= 1;
                return;
            }
        }
        panic!("bond {:?} not found in atom {:?} neigh list", b, a);
    }

    pub fn add_bond(&mut self, a: AtomH, b: AtomH) -> BondH {
        // validate atoms
        let _ = self.atom(a);
        let _ = self.atom(b);

        let data = BondData { a, b };
        let bh = if let Some(i) = self.free_bonds.pop() {
            let s = &mut self.bonds[i as usize];
            s.val = Some(data);
            BondH { idx: i, gen: s.gen }
        } else {
            let idx = self.bonds.len() as u32;
            self.bonds.push(Slot { gen: 1, val: Some(data) });
            BondH { idx, gen: 1 }
        };

        self.add_bond_to_atom(a, bh);
        self.add_bond_to_atom(b, bh);
        self.dirty_derived = true;
        bh
    }

    pub fn remove_bond(&mut self, bh: BondH) {
        let alive = {
            let s = &self.bonds[bh.idx as usize];
            s.gen == bh.gen && s.val.is_some()
        };
        if !alive { return; }
        let (a, b) = {
            let bd = self.bond(bh);
            (bd.a, bd.b)
        };
        // detach from atoms (may be already removed)
        if self.is_atom_alive(a) { self.remove_bond_from_atom(a, bh); }
        if self.is_atom_alive(b) { self.remove_bond_from_atom(b, bh); }

        let s = &mut self.bonds[bh.idx as usize];
        s.val = None;
        s.gen += 1;
        self.free_bonds.push(bh.idx);
        self.dirty_derived = true;
    }

    #[inline(always)] fn is_atom_alive(&self, a: AtomH) -> bool {
        let s = &self.atoms[a.idx as usize];
        s.gen == a.gen && s.val.is_some()
    }

    // ================== MM::Builder diagnostic prints (parity with C++ MMFFBuilderBase.h) ==================

    pub fn print_bonds_of_atom(&self, ia: usize) {
        let ad = match &self.atoms[ia].val {
            Some(a) => a,
            None => { println!("printBondsOfAtom({}): atom is dead", ia); return; }
        };
        print!("printBondsOfAtom({}): ", ia);
        for i in 0..ad.nbond as usize {
            let bh = ad.neigh_bonds[i];
            if bh.gen == 0 { continue; }
            let bd = self.bond(bh);
            print!("({}|{:3},{:3}) ", bh.idx, bd.a.idx, bd.b.idx);
        }
        println!();
    }

    pub fn print_atom_neighs(&self, ia: usize) {
        let ad = match &self.atoms[ia].val {
            Some(a) => a,
            None => { println!("printAtomNeighs({}): atom is dead", ia); return; }
        };
        print!("atom[{:3}] nbond({:1}) neighs{{", ia, ad.nbond);
        for i in 0..4 {
            if i < ad.nbond as usize {
                let bh = ad.neigh_bonds[i];
                let bd = self.bond(bh);
                let ja = if bd.a.idx == ia as u32 { bd.b.idx } else { bd.a.idx };
                print!("{:3},", ja);
            } else {
                print!(" -1,");
            }
        }
        println!("}}");
    }

    pub fn bake(&mut self) -> Topology {
        // map live atoms to dense indices
        let mut map: Vec<i32> = vec![-1; self.atoms.len()];
        let mut apos: Vec<Vec3d> = Vec::new();
        for (i, s) in self.atoms.iter().enumerate() {
            if let Some(a) = &s.val {
                map[i] = apos.len() as i32;
                apos.push(a.pos);
            }
        }

        // export live bonds (dense atom indices)
        let mut bonds: Vec<[i32; 2]> = Vec::new();
        for s in &self.bonds {
            if let Some(bd) = &s.val {
                let ia = map[bd.a.idx as usize];
                let ja = map[bd.b.idx as usize];
                assert!(ia >= 0 && ja >= 0, "bond references dead atom");
                bonds.push([ia, ja]);
            }
        }

        let natoms = apos.len() as i32;
        let angles = build_angles_from_bonds(natoms, &bonds);
        let dihedrals = build_dihedrals_from_bonds(&bonds);
        let inversions = build_inversions_from_bonds(natoms, &bonds);

        self.dirty_derived = false;
        Topology { apos, bonds, angles, dihedrals, inversions }
    }

    pub fn from_positions_cutoff(apos: &[Vec3d], rcut: f64) -> Self {
        // convenience for current demo main: creates atoms then bonds by naive cutoff
        let mut b = Self::new();
        let mut hs: Vec<AtomH> = Vec::with_capacity(apos.len());
        for &p in apos { hs.push(b.add_atom(p)); }
        let r2cut = rcut * rcut;
        for i in 0..apos.len() {
            for j in (i + 1)..apos.len() {
                let d = Vec3d::set_sub(apos[j], apos[i]);
                if d.norm2() < r2cut { b.add_bond(hs[i], hs[j]); }
            }
        }
        b
    }

    /// Build bonds using element-specific covalent radii with tolerance.
    /// radii[i] is the covalent radius of atom i (in same units as apos).
    pub fn from_positions_and_radii(apos: &[Vec3d], radii: &[f64], tol: f64) -> Self {
        assert_eq!(apos.len(), radii.len(), "apos and radii must have same length");
        let mut b = Self::new();
        let mut hs: Vec<AtomH> = Vec::with_capacity(apos.len());
        for &p in apos { hs.push(b.add_atom(p)); }
        for i in 0..apos.len() {
            for j in (i + 1)..apos.len() {
                let d = Vec3d::set_sub(apos[j], apos[i]);
                let rcut = radii[i] + radii[j] + tol;
                if d.norm2() < rcut * rcut { b.add_bond(hs[i], hs[j]); }
            }
        }
        b
    }
}
