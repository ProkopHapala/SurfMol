use std::collections::{HashMap, HashSet};
use numcore::math::vec3::Vec3d;
use crate::topology::{Topology, build_angles_from_bonds, build_dihedrals_from_bonds, build_inversions_from_bonds};

#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, Debug)]
pub struct AtomH { pub idx: u32, pub gen: u32 }

#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, Debug)]
pub struct BondH { pub idx: u32, pub gen: u32 }

struct Slot<T> { gen: u32, val: Option<T> }

#[derive(Clone)]
pub struct AtomData {
    pub pos: Vec3d,
    pub element: String,
    pub atype: i32,
    pub neigh_bonds: [BondH; 4],
    pub nbond: u8,
    pub pin: Option<(i32, i32)>,
    pub parent: Option<AtomH>,
    pub is_h_cap: bool,
    pub hybridization: u8,
    pub alive: bool,
}

#[derive(Clone, Copy)]
pub struct BondData {
    pub a: AtomH,
    pub b: AtomH,
    pub order: u8,
    pub alive: bool,
}

pub struct Builder {
    atoms: Vec<Slot<AtomData>>,
    bonds: Vec<Slot<BondData>>,
    free_atoms: Vec<u32>,
    free_bonds: Vec<u32>,
    pub dirty_derived: bool,
    pub pin_to_atom: HashMap<(i32, i32), AtomH>,
    pub hex_tiles: HashSet<(i32, i32)>,
}

impl Builder {
    pub fn new() -> Self {
        Self { atoms: Vec::new(), bonds: Vec::new(), free_atoms: Vec::new(), free_bonds: Vec::new(), dirty_derived: true, pin_to_atom: HashMap::new(), hex_tiles: HashSet::new() }
    }

    /// Lookup atomic number from element symbol.
    pub fn element_to_z(element: &str) -> i32 {
        match element {
            "H" => 1, "He" => 2, "Li" => 3, "Be" => 4, "B" => 5, "C" => 6, "N" => 7, "O" => 8, "F" => 9, "Ne" => 10,
            "Na" => 11, "Mg" => 12, "Al" => 13, "Si" => 14, "P" => 15, "S" => 16, "Cl" => 17, "Ar" => 18,
            "K" => 19, "Ca" => 20, "Sc" => 21, "Ti" => 22, "V" => 23, "Cr" => 24, "Mn" => 25, "Fe" => 26,
            "Co" => 27, "Ni" => 28, "Cu" => 29, "Zn" => 30, "Ga" => 31, "Ge" => 32, "As" => 33, "Se" => 34, "Br" => 35,
            _ => 0,
        }
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

    pub fn add_atom(&mut self, pos: Vec3d, element: &str) -> AtomH {
        let atype = Self::element_to_z(element);
        let data = AtomData { pos, element: element.to_string(), atype, neigh_bonds: [BondH::default(); 4], nbond: 0, pin: None, parent: None, is_h_cap: false, hybridization: 0, alive: true };
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

    /// Hard remove: immediately frees the slot. Use with caution — invalidates all handles to this atom.
    pub fn remove_atom_hard(&mut self, a: AtomH) {
        let alive = {
            let s = &self.atoms[a.idx as usize];
            s.gen == a.gen && s.val.is_some()
        };
        assert!(alive, "remove_atom_hard: stale handle {:?}", a);
        // remove incident bonds (iterate local copy because we will mutate nbond)
        let neighs: [BondH; 4] = self.atom(a).neigh_bonds;
        for bh in neighs {
            if bh.gen == 0 { continue; }
            self.remove_bond_hard(bh);
        }
        if let Some(pin) = self.atom(a).pin {
            self.pin_to_atom.remove(&pin);
        }
        let s = &mut self.atoms[a.idx as usize];
        s.val = None;
        s.gen += 1;
        self.free_atoms.push(a.idx);
        self.dirty_derived = true;
    }

    /// Soft remove: marks atom and its bonds as dead. Slot is NOT freed until `cleanup_dead()`.
    /// Safe to call while holding other handles; stale handles will be detected by `is_atom_alive`.
    pub fn remove_atom(&mut self, a: AtomH) {
        assert!(self.is_atom_alive(a), "remove_atom: stale or dead handle {:?}", a);
        let neighs: [BondH; 4] = self.atom(a).neigh_bonds;
        for bh in neighs {
            if bh.gen == 0 { continue; }
            if let Some(bd) = self.bonds[bh.idx as usize].val.as_mut() {
                if bd.alive { bd.alive = false; }
            }
        }
        if let Some(pin) = self.atom(a).pin {
            self.pin_to_atom.remove(&pin);
        }
        self.atom_mut(a).alive = false;
        self.dirty_derived = true;
    }

    #[inline(always)] pub fn atom(&self, a: AtomH) -> &AtomData { Self::slot_get(&self.atoms, a.idx, a.gen) }
    #[inline(always)] pub fn atom_mut(&mut self, a: AtomH) -> &mut AtomData { Self::slot_get_mut(&mut self.atoms, a.idx, a.gen) }
    #[inline(always)] pub fn bond(&self, b: BondH) -> &BondData { Self::slot_get(&self.bonds, b.idx, b.gen) }
    #[inline(always)] pub fn bond_mut(&mut self, b: BondH) -> &mut BondData { Self::slot_get_mut(&mut self.bonds, b.idx, b.gen) }

    /// Find the nearest live atom to `pos` within `radius`. Returns (handle, distance²) or None.
    pub fn find_nearest_atom(&self, pos: Vec3d, radius: f64) -> Option<(AtomH, f64)> {
        let r2 = radius * radius;
        let mut best = None;
        let mut best_d2 = r2;
        for (ah, ad) in self.iter_atoms() {
            let d2 = Vec3d::set_sub(ad.pos, pos).norm2();
            if d2 < best_d2 { best_d2 = d2; best = Some((ah, d2)); }
        }
        best
    }

    /// Change the element of an existing atom.
    pub fn set_atom_element(&mut self, a: AtomH, element: &str) {
        assert!(self.is_atom_alive(a), "set_atom_element: stale or dead handle {:?}", a);
        let ad = self.atom_mut(a);
        ad.element = element.to_string();
        ad.atype = Self::element_to_z(element);
    }

    /// Iterate over all live atoms (handle, data).
    pub fn iter_atoms(&self) -> impl Iterator<Item = (AtomH, &AtomData)> {
        self.atoms.iter().enumerate().filter_map(|(i, slot)| {
            slot.val.as_ref().filter(|ad| ad.alive).map(|ad| (AtomH { idx: i as u32, gen: slot.gen }, ad))
        })
    }

    /// Iterate over all live bonds (handle, data).
    pub fn iter_bonds(&self) -> impl Iterator<Item = (BondH, &BondData)> {
        self.bonds.iter().enumerate().filter_map(|(i, slot)| {
            slot.val.as_ref().filter(|bd| bd.alive).map(|bd| (BondH { idx: i as u32, gen: slot.gen }, bd))
        })
    }

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

    /// Add bond with optional order (default 1). Checks for duplicate bonds between the same pair of atoms.
    pub fn add_bond(&mut self, a: AtomH, b: AtomH, order: u8) -> BondH {
        assert!(self.is_atom_alive(a), "add_bond: dead atom a {:?}", a);
        assert!(self.is_atom_alive(b), "add_bond: dead atom b {:?}", b);
        assert!(a != b, "add_bond: cannot bond atom to itself");
        // check for duplicate bond
        for i in 0..self.atom(a).nbond as usize {
            let bh = self.atom(a).neigh_bonds[i];
            let bd = self.bond(bh);
            if bd.a == b || bd.b == b { return bh; }
        }
        let data = BondData { a, b, order, alive: true };
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

    /// Hard remove bond: immediately frees the slot.
    pub fn remove_bond_hard(&mut self, bh: BondH) {
        let alive = {
            let s = &self.bonds[bh.idx as usize];
            s.gen == bh.gen && s.val.is_some()
        };
        if !alive { return; }
        let (a, b) = {
            let bd = self.bond(bh);
            (bd.a, bd.b)
        };
        if self.is_atom_alive(a) { self.remove_bond_from_atom(a, bh); }
        if self.is_atom_alive(b) { self.remove_bond_from_atom(b, bh); }
        let s = &mut self.bonds[bh.idx as usize];
        s.val = None;
        s.gen += 1;
        self.free_bonds.push(bh.idx);
        self.dirty_derived = true;
    }

    /// Soft remove bond: marks bond as dead. Slot is NOT freed until `cleanup_dead()`.
    pub fn remove_bond(&mut self, bh: BondH) {
        if !self.is_bond_alive(bh) { return; }
        self.bond_mut(bh).alive = false;
        self.dirty_derived = true;
    }

    /// Batch cleanup: frees all dead atom/bond slots, rebuilds neighbor lists, and reclaims memory.
    pub fn cleanup_dead(&mut self) {
        // clean dead bonds from atom neighbor lists
        for slot in &mut self.atoms {
            if let Some(ad) = slot.val.as_mut() {
                if !ad.alive { continue; }
                let mut n_alive = 0;
                for i in 0..ad.nbond as usize {
                    let bh = ad.neigh_bonds[i];
                    if bh.gen == 0 { continue; }
                    if let Some(bd) = self.bonds[bh.idx as usize].val.as_ref() {
                        if bd.alive {
                            ad.neigh_bonds[n_alive] = bh;
                            n_alive += 1;
                        }
                    }
                }
                for i in n_alive..4 {
                    ad.neigh_bonds[i] = BondH::default();
                }
                ad.nbond = n_alive as u8;
            }
        }
        // free dead bond slots
        for (i, slot) in self.bonds.iter_mut().enumerate() {
            if slot.val.is_some() && !slot.val.as_ref().unwrap().alive {
                slot.val = None;
                slot.gen += 1;
                self.free_bonds.push(i as u32);
            }
        }
        // free dead atom slots
        for (i, slot) in self.atoms.iter_mut().enumerate() {
            if slot.val.is_some() && !slot.val.as_ref().unwrap().alive {
                slot.val = None;
                slot.gen += 1;
                self.free_atoms.push(i as u32);
            }
        }
        self.dirty_derived = true;
    }

    #[inline(always)] pub fn is_atom_alive(&self, a: AtomH) -> bool {
        let s = &self.atoms[a.idx as usize];
        s.gen == a.gen && s.val.is_some() && s.val.as_ref().unwrap().alive
    }

    #[inline(always)] pub fn is_bond_alive(&self, bh: BondH) -> bool {
        let s = &self.bonds[bh.idx as usize];
        s.gen == bh.gen && s.val.is_some() && s.val.as_ref().unwrap().alive
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

    /// Return elements for live atoms in the same order as `bake()` produces topology atoms.
    pub fn bake_elements(&self) -> Vec<String> {
        let mut elems = Vec::new();
        for s in &self.atoms {
            if let Some(a) = &s.val {
                if a.alive { elems.push(a.element.clone()); }
            }
        }
        elems
    }

    pub fn bake(&mut self) -> Topology {
        // map live atoms to dense indices
        let mut map: Vec<i32> = vec![-1; self.atoms.len()];
        let mut apos: Vec<Vec3d> = Vec::new();
        for (i, s) in self.atoms.iter().enumerate() {
            if let Some(a) = &s.val {
                if !a.alive { continue; }
                map[i] = apos.len() as i32;
                apos.push(a.pos);
            }
        }
        // export live bonds (dense atom indices)
        let mut bonds: Vec<[i32; 2]> = Vec::new();
        for s in &self.bonds {
            if let Some(bd) = &s.val {
                if !bd.alive { continue; }
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

    pub fn from_positions_cutoff(apos: &[Vec3d], elems: &[String], rcut: f64) -> Self {
        let mut b = Self::new();
        let mut hs: Vec<AtomH> = Vec::with_capacity(apos.len());
        for i in 0..apos.len() {
            let el = elems.get(i).map(|s| s.as_str()).unwrap_or("?");
            hs.push(b.add_atom(apos[i], el));
        }
        let r2cut = rcut * rcut;
        for i in 0..apos.len() {
            for j in (i + 1)..apos.len() {
                let d = Vec3d::set_sub(apos[j], apos[i]);
                if d.norm2() < r2cut { b.add_bond(hs[i], hs[j], 1); }
            }
        }
        b
    }

    /// Build bonds using element-specific covalent radii with tolerance.
    /// radii[i] is the covalent radius of atom i (in same units as apos).
    pub fn from_positions_and_radii(apos: &[Vec3d], elems: &[String], radii: &[f64], tol: f64) -> Self {
        assert_eq!(apos.len(), radii.len(), "apos and radii must have same length");
        let mut b = Self::new();
        let mut hs: Vec<AtomH> = Vec::with_capacity(apos.len());
        for i in 0..apos.len() {
            let el = elems.get(i).map(|s| s.as_str()).unwrap_or("?");
            hs.push(b.add_atom(apos[i], el));
        }
        for i in 0..apos.len() {
            for j in (i + 1)..apos.len() {
                let d = Vec3d::set_sub(apos[j], apos[i]);
                let rcut = radii[i] + radii[j] + tol;
                if d.norm2() < rcut * rcut { b.add_bond(hs[i], hs[j], 1); }
            }
        }
        b
    }

    // ================== Hex grid editing (ported from KekuleBackend.py) ==================

    /// Return the 6 node positions of a hexagonal ring at axial coords (q,r).
    /// Pointy-top orientation. Circumradius = a_cc.
    pub fn honeycomb_ring_nodes(q: i32, r: i32, a_cc: f64) -> [(f64, f64); 6] {
        let s3 = 3.0_f64.sqrt();
        let cx = a_cc * s3 * (q as f64 + r as f64 * 0.5);
        let cy = a_cc * 1.5 * r as f64;
        let mut nodes = [(0.0, 0.0); 6];
        for i in 0..6 {
            let angle = (i as f64) * (std::f64::consts::PI / 3.0) + std::f64::consts::PI / 6.0;
            nodes[i] = (cx + a_cc * angle.cos(), cy + a_cc * angle.sin());
        }
        nodes
    }

    /// Snap Cartesian position to grid key (rounded to 4 decimal places as i32).
    pub fn snap_to_grid(x: f64, y: f64) -> (i32, i32) {
        let rx = (x * 10000.0).round() as i32;
        let ry = (y * 10000.0).round() as i32;
        (rx, ry)
    }

    /// Find axial coordinates (q,r) of the hexagon whose center is closest to (x,y).
    pub fn snap_to_ring(x: f64, y: f64, a_cc: f64) -> (i32, i32) {
        let s3 = 3.0_f64.sqrt();
        let r_exact = y / (1.5 * a_cc);
        let q_exact = x / (s3 * a_cc) - r_exact * 0.5;
        (q_exact.round() as i32, r_exact.round() as i32)
    }

    /// Find the nearest honeycomb node to (x,y) within tolerance. Returns grid key or None.
    pub fn snap_to_node(x: f64, y: f64, a_cc: f64, tol: f64) -> Option<(i32, i32)> {
        let (q, r) = Self::snap_to_ring(x, y, a_cc);
        let tol2 = tol * tol;
        let mut best = None;
        let mut best_d2 = f64::MAX;
        for dq in -1..=1 {
            for dr in -1..=1 {
                if i32::abs(dq - dr) > 1 { continue; } // only adjacent hexes in axial coords
                let nodes = Self::honeycomb_ring_nodes(q + dq, r + dr, a_cc);
                for node in nodes {
                    let dx = node.0 - x;
                    let dy = node.1 - y;
                    let d2 = dx * dx + dy * dy;
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = Some(Self::snap_to_grid(node.0, node.1));
                    }
                }
            }
        }
        if best_d2 < tol2 { best } else { None }
    }

    /// Add a benzene ring at axial position (q,r). Idempotent for existing atoms.
    pub fn add_hex_ring(&mut self, q: i32, r: i32, a_cc: f64) {
        let nodes = Self::honeycomb_ring_nodes(q, r, a_cc);
        let mut ring_atoms: Vec<AtomH> = Vec::with_capacity(6);
        for node in nodes {
            let pin = Self::snap_to_grid(node.0, node.1);
            if let Some(&ah) = self.pin_to_atom.get(&pin) {
                if self.is_atom_alive(ah) {
                    ring_atoms.push(ah);
                    continue;
                }
            }
            let pos = Vec3d::new(node.0, node.1, 0.0);
            let ah = self.add_atom(pos, "C");
            {
                let ad = self.atom_mut(ah);
                ad.pin = Some(pin);
                ad.hybridization = 2; // sp2
            }
            self.pin_to_atom.insert(pin, ah);
            ring_atoms.push(ah);
        }
        // create bonds between ring atoms at proper C-C distance
        self.create_bonds_for_ring_atoms(&ring_atoms, a_cc);
        self.hex_tiles.insert((q, r));
        self.dirty_derived = true;
    }

    /// Remove a hex ring. In paint mode removes all 6 nodes; in toggle mode preserves shared atoms.
    pub fn remove_hex_ring(&mut self, q: i32, r: i32, _toggle_mode: bool, a_cc: f64) {
        let nodes = Self::honeycomb_ring_nodes(q, r, a_cc);
        let mut to_remove = Vec::new();
        for node in nodes {
            let pin = Self::snap_to_grid(node.0, node.1);
            if let Some(&ah) = self.pin_to_atom.get(&pin) {
                if self.is_atom_alive(ah) {
                    to_remove.push(ah);
                }
            }
        }
        for ah in to_remove {
            // remove H children first
            for h in self.h_children(ah) {
                self.remove_atom(h);
            }
            self.remove_atom(ah);
        }
        self.hex_tiles.remove(&(q, r));
        self.dirty_derived = true;
    }

    /// Toggle a hex ring: add if absent, remove if present.
    pub fn toggle_hex_ring(&mut self, q: i32, r: i32, a_cc: f64) {
        if self.hex_tiles.contains(&(q, r)) {
            self.remove_hex_ring(q, r, false, a_cc);
        } else {
            self.add_hex_ring(q, r, a_cc);
        }
    }

    fn create_bonds_for_ring_atoms(&mut self, atoms: &[AtomH], a_cc: f64) {
        let cc_bond_sq = (a_cc * 1.1).powi(2);
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let a = atoms[i];
                let b = atoms[j];
                let pa = self.atom(a).pos;
                let pb = self.atom(b).pos;
                let d2 = Vec3d::set_sub(pb, pa).norm2();
                if d2 < cc_bond_sq {
                    self.add_bond(a, b, 1);
                }
            }
        }
    }

    // ================== H cap management ==================

    /// Return all H_cap atoms whose parent is the given atom.
    pub fn h_children(&self, parent: AtomH) -> Vec<AtomH> {
        let mut out = Vec::new();
        for (i, slot) in self.atoms.iter().enumerate() {
            if let Some(ad) = &slot.val {
                if ad.alive && ad.is_h_cap {
                    if let Some(p) = ad.parent {
                        if p == parent { out.push(AtomH { idx: i as u32, gen: slot.gen }); }
                    }
                }
            }
        }
        out
    }

    // ================== Picking helpers ==================

    /// Find atom within radius of position. Returns handle or None.
    pub fn pick_atom(&self, pos: Vec3d, radius: f64) -> Option<AtomH> {
        let r2 = radius * radius;
        let mut best = None;
        let mut best_d2 = f64::MAX;
        for (i, slot) in self.atoms.iter().enumerate() {
            if let Some(ad) = &slot.val {
                if !ad.alive { continue; }
                let d2 = Vec3d::set_sub(ad.pos, pos).norm2();
                if d2 < r2 && d2 < best_d2 {
                    best_d2 = d2;
                    best = Some(AtomH { idx: i as u32, gen: slot.gen });
                }
            }
        }
        best
    }

    /// Find bond whose center is within radius of position.
    pub fn pick_bond(&self, pos: Vec3d, radius: f64) -> Option<BondH> {
        let r2 = radius * radius;
        let mut best = None;
        let mut best_d2 = f64::MAX;
        for (i, slot) in self.bonds.iter().enumerate() {
            if let Some(bd) = &slot.val {
                if !bd.alive { continue; }
                if !self.is_atom_alive(bd.a) || !self.is_atom_alive(bd.b) { continue; }
                let pa = self.atom(bd.a).pos;
                let pb = self.atom(bd.b).pos;
                let center = Vec3d::new((pa.x + pb.x) * 0.5, (pa.y + pb.y) * 0.5, (pa.z + pb.z) * 0.5);
                let d2 = Vec3d::set_sub(center, pos).norm2();
                if d2 < r2 && d2 < best_d2 {
                    best_d2 = d2;
                    best = Some(BondH { idx: i as u32, gen: slot.gen });
                }
            }
        }
        best
    }
}
