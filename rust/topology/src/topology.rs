use surfmol_common::math::quat4::Quat4i;
use surfmol_common::math::vec3::Vec3d;

pub struct Topology {
    pub apos: Vec<Vec3d>,
    pub bonds: Vec<[i32; 2]>,
    pub angles: Vec<[i32; 3]>,
    pub dihedrals: Vec<Quat4i>,
    pub inversions: Vec<Quat4i>,
}

impl Topology {
    #[inline(always)] pub fn natoms(&self) -> i32 { self.apos.len() as i32 }

    // ================== MM::Builder diagnostic prints (parity with C++ MMFFBuilderBase.h) ==================

    pub fn print_sizes(&self) {
        println!("sizes: atoms({}|0) bonds({}) angles({}) dihedrals({})",
                 self.apos.len(), self.bonds.len(), self.angles.len(), self.dihedrals.len());
    }

    pub fn print_atoms(&self) {
        println!(" # MM::Builder.printAtoms(na={}) ", self.apos.len());
        for (i, p) in self.apos.iter().enumerate() {
            println!("atom[{:3}] pos({:12.6},{:12.6},{:12.6})", i, p.x, p.y, p.z);
        }
    }

    pub fn print_bonds(&self) {
        println!(" # MM::Builder.printBonds(nb={}) ", self.bonds.len());
        for (i, b) in self.bonds.iter().enumerate() {
            println!("bond[{:3}] a({:3},{:3})", i, b[0], b[1]);
        }
    }

    pub fn print_angles(&self) {
        println!(" # MM::Builder.printAngles(ng={}) ", self.angles.len());
        for (i, a) in self.angles.iter().enumerate() {
            println!("angle[{:3}] a({:3},{:3},{:3})", i, a[0], a[1], a[2]);
        }
    }

    pub fn print_dihedrals(&self) {
        println!(" # MM::Builder.printDihedrals(nd={}) ", self.dihedrals.len());
        for (i, d) in self.dihedrals.iter().enumerate() {
            println!("dihedral[{:3}] a({:3},{:3},{:3},{:3})", i, d.x, d.y, d.z, d.w);
        }
    }

    pub fn print_inversions(&self) {
        println!(" # MM::Builder.printInversions(ni={}) ", self.inversions.len());
        for (i, inv) in self.inversions.iter().enumerate() {
            println!("inversion[{:3}] a({:3},{:3},{:3},{:3})", i, inv.x, inv.y, inv.z, inv.w);
        }
    }
}

pub fn build_bonds_by_cutoff(apos: &[Vec3d], rcut: f64) -> Vec<[i32; 2]> {
    let mut bonds = Vec::new();
    let r2cut = rcut * rcut;
    for i in 0..apos.len() {
        for j in (i + 1)..apos.len() {
            let d = Vec3d::set_sub(apos[j], apos[i]);
            if d.norm2() < r2cut { bonds.push([i as i32, j as i32]); }
        }
    }
    bonds
}

pub fn build_angles_from_bonds(natoms: i32, bonds: &[[i32; 2]]) -> Vec<[i32; 3]> {
    let mut neigh: Vec<Vec<i32>> = vec![Vec::new(); natoms as usize];
    for b in bonds {
        neigh[b[0] as usize].push(b[1]);
        neigh[b[1] as usize].push(b[0]);
    }
    let mut angles = Vec::new();
    for j in 0..natoms {
        let ns = &neigh[j as usize];
        for a in 0..ns.len() {
            for b in (a + 1)..ns.len() {
                angles.push([ns[a], j, ns[b]]);
            }
        }
    }
    angles
}

pub fn build_dihedrals_from_bonds(bonds: &[[i32; 2]]) -> Vec<Quat4i> {
    // placeholder enumerator for now; the dynamic Builder will eventually provide a canonical + incremental implementation
    use std::collections::{HashMap, HashSet};
    let mut adj: HashMap<i32, Vec<i32>> = HashMap::new();
    for b in bonds {
        adj.entry(b[0]).or_default().push(b[1]);
        adj.entry(b[1]).or_default().push(b[0]);
    }
    let mut set = HashSet::<(i32, i32, i32, i32)>::new();
    for (&j, js) in &adj {
        for &k in js {
            if let Some(is) = adj.get(&j) {
                for &i in is {
                    if i == k { continue; }
                    if let Some(ls) = adj.get(&k) {
                        for &l in ls {
                            if l == j { continue; }
                            set.insert((i, j, k, l));
                        }
                    }
                }
            }
        }
    }
    set.into_iter().map(|(i, j, k, l)| Quat4i::new(i, j, k, l)).collect()
}

pub fn build_inversions_from_bonds(natoms: i32, bonds: &[[i32; 2]]) -> Vec<Quat4i> {
    // placeholder: for atoms with 3 neighbors pick one triple
    let mut neigh: Vec<Vec<i32>> = vec![Vec::new(); natoms as usize];
    for b in bonds {
        neigh[b[0] as usize].push(b[1]);
        neigh[b[1] as usize].push(b[0]);
    }
    let mut invs = Vec::new();
    for i in 0..natoms {
        let ns = &neigh[i as usize];
        if ns.len() == 3 {
            invs.push(Quat4i::new(i, ns[0], ns[1], ns[2]));
        }
    }
    invs
}
