use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::Path;

/// Element parameters from ElementTypes.dat
#[derive(Clone, Debug)]
pub struct ElementType {
    pub name: String,
    pub i_z: u8,
    pub n_val: u8,
    pub valence: u8,
    pub n_pi: u8,
    pub color: u32,
    pub r_cov: f64,
    pub r_vdw: f64,
    pub e_vdw: f64,
    pub q_uff: f64,
    pub u_uff: f64,
    pub v_uff: f64,
    pub b_qeq: bool,
    pub e_aff: f64,
    pub e_hard: f64,
    pub r_a: f64,
    pub eta: f64,
}

impl ElementType {
    pub fn print(&self, i: usize, b_params: bool) {
        print!("ElementType[{:3},{}] {:3}({:2},{:2},{:2}) {:08X} ", i, self.name, self.i_z, self.n_val, self.valence, self.n_pi, self.color);
        if b_params {
            print!("REQ({:7.3},{:7.3},{:7.3},{:7.3}) QEq({:7.3},{:7.3},{:7.3},{:7.3})", self.r_cov, self.r_vdw, self.e_vdw, self.q_uff, self.e_aff, self.e_hard, self.r_a, self.eta);
        }
        println!();
    }
}

/// Atom type parameters from AtomTypes.dat
#[derive(Clone, Debug)]
pub struct AtomType {
    pub name: String,
    pub parent: String,
    pub element: String,
    pub epair: String,
    pub nv: u8,
    pub ne: i8,
    pub n_pi: i8,
    pub sym: u8,
    pub r_uff: f64,
    pub r_vdw: f64,
    pub e_vdw: f64,
    pub q_base: f64,
    pub h_b: f64,
    pub b_mmff: bool,
    pub a_ss: f64,
    pub a_sp: f64,
    pub k_ss: f64,
    pub k_sp: f64,
    pub k_ep: f64,
    pub k_pp: f64,
}

impl AtomType {
    pub fn print(&self, i: usize, b_params: bool) {
        print!("AtomType[{:3},{}] ({:2},{:2},{:2},{:2}) ", i, self.name, self.nv, self.ne, self.n_pi, self.sym);
        if b_params {
            print!("REQH({:7.3},{:7.3},{:7.3},{:7.3}) MMFF({:7.3},{:7.3},{:7.3},{:7.3},{:7.3},{:7.3})",
                   self.r_vdw, self.e_vdw, self.q_base, self.h_b, self.a_ss, self.a_sp, self.k_ss, self.k_sp, self.k_ep, self.k_pp);
        }
        println!();
    }
}

/// Bond parameters from BondTypes.dat
#[derive(Clone, Debug)]
pub struct BondParam {
    pub a: String,
    pub b: String,
    pub order: u8,
    pub l0: f64,
    pub k: f64,
}

/// Angle parameters from AngleTypes.dat
#[derive(Clone, Debug)]
pub struct AngleParam {
    pub a: String,
    pub b: String,
    pub c: String,
    pub a0: f64,
    pub k: f64,
}

/// Dihedral parameters from DihedralTypes.dat
#[derive(Clone, Debug)]
pub struct DihedralParam {
    pub a: String,
    pub b: String,
    pub c: String,
    pub d: String,
    pub order: u8,
    pub k: f64,
    pub a0: f64,
    pub n: i32,
}

pub struct Params {
    pub elements: Vec<ElementType>,
    pub element_dict: HashMap<String, usize>,
    pub atom_types: Vec<AtomType>,
    pub atom_type_dict: HashMap<String, usize>,
    pub bonds: Vec<BondParam>,
    pub bond_dict: HashMap<(String, String, u8), usize>, // sorted(a,b), order
    pub angles: Vec<AngleParam>,
    pub dihedrals: Vec<DihedralParam>,
}

impl Params {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            element_dict: HashMap::new(),
            atom_types: Vec::new(),
            atom_type_dict: HashMap::new(),
            bonds: Vec::new(),
            bond_dict: HashMap::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        }
    }

    fn parse_element_line(line: &str) -> Option<ElementType> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 10 { return None; }
        let name = parts[0].to_string();
        let i_z: u8 = parts[1].parse().ok()?;
        let n_val: u8 = parts[2].parse().ok()?;
        let valence: u8 = parts[3].parse().ok()?;
        let n_pi: u8 = parts[4].parse().ok()?;
        let color = u32::from_str_radix(parts[5].trim_start_matches("0x"), 16).unwrap_or(0);
        let r_cov: f64 = parts[6].parse().ok()?;
        let r_vdw: f64 = parts[7].parse().ok()?;
        let e_vdw: f64 = parts[8].parse().ok()?;
        let q_uff: f64 = parts[9].parse().ok()?;
        let mut u_uff = 0.0;
        let mut v_uff = 0.0;
        let mut b_qeq = false;
        let mut e_aff = 0.0;
        let mut e_hard = 0.0;
        let mut r_a = 0.0;
        let mut eta = 0.0;
        if parts.len() >= 11 { u_uff = parts[10].parse().ok()?; }
        if parts.len() >= 12 { v_uff = parts[11].parse().ok()?; }
        if parts.len() >= 16 {
            b_qeq = true;
            e_aff = parts[12].parse().ok()?;
            e_hard = parts[13].parse().ok()?;
            r_a = parts[14].parse().ok()?;
            eta = parts[15].parse().ok()?;
        }
        Some(ElementType { name, i_z, n_val, valence, n_pi, color, r_cov, r_vdw, e_vdw, q_uff, u_uff, v_uff, b_qeq, e_aff, e_hard, r_a, eta })
    }

    fn parse_atom_type_line(line: &str) -> Option<AtomType> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 13 { return None; }
        let name = parts[0].to_string();
        let parent = parts[1].to_string();
        let element = parts[2].to_string();
        let epair = parts[3].to_string();
        let nv: u8 = parts[4].parse().ok()?;
        let ne: i8 = parts[5].parse().ok()?;
        let n_pi: i8 = parts[6].parse().ok()?;
        let sym: u8 = parts[7].parse().ok()?;
        let r_uff: f64 = parts[8].parse().ok()?;
        let r_vdw: f64 = parts[9].parse().ok()?;
        let e_vdw: f64 = parts[10].parse().ok()?;
        let q_base: f64 = parts[11].parse().ok()?;
        let h_b: f64 = parts[12].parse().ok()?;
        let mut b_mmff = false;
        let mut a_ss = 0.0;
        let mut a_sp = 0.0;
        let mut k_ss = 0.0;
        let mut k_sp = 0.0;
        let mut k_ep = 0.0;
        let mut k_pp = 0.0;
        if parts.len() >= 19 {
            b_mmff = true;
            a_ss = parts[13].parse().ok()?;
            a_sp = parts[14].parse().ok()?;
            k_ss = parts[15].parse().ok()?;
            k_sp = parts[16].parse().ok()?;
            k_ep = parts[17].parse().ok()?;
            k_pp = parts[18].parse().ok()?;
        }
        Some(AtomType { name, parent, element, epair, nv, ne, n_pi, sym, r_uff, r_vdw, e_vdw, q_base, h_b, b_mmff, a_ss, a_sp, k_ss, k_sp, k_ep, k_pp })
    }

    fn parse_bond_line(line: &str) -> Option<BondParam> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 { return None; }
        let a = parts[0].to_string();
        let b = parts[1].to_string();
        let order: u8 = parts[2].parse().ok()?;
        let l0: f64 = parts[3].parse().ok()?;
        let k: f64 = parts[4].parse().ok()?;
        Some(BondParam { a, b, order, l0, k })
    }

    fn parse_angle_line(line: &str) -> Option<AngleParam> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 { return None; }
        let a = parts[0].to_string();
        let b = parts[1].to_string();
        let c = parts[2].to_string();
        let a0: f64 = parts[3].parse().ok()?;
        let k: f64 = parts[4].parse().ok()?;
        Some(AngleParam { a, b, c, a0, k })
    }

    fn parse_dihedral_line(line: &str) -> Option<DihedralParam> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 8 { return None; }
        let a = parts[0].to_string();
        let b = parts[1].to_string();
        let c = parts[2].to_string();
        let d = parts[3].to_string();
        let order: u8 = parts[4].parse().ok()?;
        let k: f64 = parts[5].parse().ok()?;
        let a0: f64 = parts[6].parse().ok()?;
        let n: i32 = parts[7].parse().ok()?;
        Some(DihedralParam { a, b, c, d, order, k, a0, n })
    }

    pub fn load_element_types<P: AsRef<Path>>(&mut self, path: P) {
        let text = read_to_string(path).expect("cannot read ElementTypes.dat");
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some(et) = Self::parse_element_line(line) {
                self.element_dict.insert(et.name.clone(), self.elements.len());
                self.elements.push(et);
            }
        }
    }

    pub fn load_atom_types<P: AsRef<Path>>(&mut self, path: P) {
        let text = read_to_string(path).expect("cannot read AtomTypes.dat");
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some(at) = Self::parse_atom_type_line(line) {
                self.atom_type_dict.insert(at.name.clone(), self.atom_types.len());
                self.atom_types.push(at);
            }
        }
    }

    pub fn load_bond_types<P: AsRef<Path>>(&mut self, path: P) {
        let text = read_to_string(path).expect("cannot read BondTypes.dat");
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some(bp) = Self::parse_bond_line(line) {
                let (a, b) = if bp.a <= bp.b { (bp.a.clone(), bp.b.clone()) } else { (bp.b.clone(), bp.a.clone()) };
                self.bond_dict.insert((a.clone(), b.clone(), bp.order), self.bonds.len());
                self.bonds.push(bp);
            }
        }
    }

    pub fn load_angle_types<P: AsRef<Path>>(&mut self, path: P) {
        let text = read_to_string(path).expect("cannot read AngleTypes.dat");
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some(ap) = Self::parse_angle_line(line) {
                self.angles.push(ap);
            }
        }
    }

    pub fn load_dihedral_types<P: AsRef<Path>>(&mut self, path: P) {
        let text = read_to_string(path).expect("cannot read DihedralTypes.dat");
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some(dp) = Self::parse_dihedral_line(line) {
                self.dihedrals.push(dp);
            }
        }
    }

    pub fn get_element_type(&self, name: &str) -> Option<&ElementType> {
        self.element_dict.get(name).map(|&i| &self.elements[i])
    }

    /// RGB color as [f32; 3] from the element's packed u32 color field.
    pub fn element_color_f32(&self, elem: &str) -> [f32; 3] {
        self.get_element_type(elem)
            .map(|et| {
                let c = et.color;
                [((c >> 16) & 0xFF) as f32 / 255.0, ((c >> 8) & 0xFF) as f32 / 255.0, (c & 0xFF) as f32 / 255.0]
            })
            .unwrap_or([0.78, 0.78, 0.78])
    }

    /// Van-der-Waals radius as f32, default 1.0 Å.
    pub fn element_radius_vdw(&self, elem: &str) -> f32 {
        self.get_element_type(elem).map(|et| et.r_vdw as f32).unwrap_or(1.0)
    }

    pub fn get_atom_type(&self, name: &str) -> Option<&AtomType> {
        self.atom_type_dict.get(name).map(|&i| &self.atom_types[i])
    }

    /// Look up the ElementType for a given atom type name (via AtomType.element field).
    /// Parity with FireCore MMFFparams::elementOfAtomType.
    pub fn element_of_atom_type(&self, atype_name: &str) -> Option<&ElementType> {
        let at = self.get_atom_type(atype_name)?;
        self.get_element_type(&at.element)
    }

    pub fn get_bond_param(&self, a: &str, b: &str, order: u8) -> Option<&BondParam> {
        let (a_sorted, b_sorted) = if a <= b { (a, b) } else { (b, a) };
        self.bond_dict.get(&(a_sorted.to_string(), b_sorted.to_string(), order)).map(|&i| &self.bonds[i])
    }

    /// Find angle parameter by matching patterns. Supports wildcards '*'
    pub fn get_angle_param(&self, a: &str, b: &str, c: &str) -> Option<&AngleParam> {
        let (a_rev, c_rev) = (c, a);
        for ap in &self.angles {
            if !match_wildcard(&ap.b, b) { continue; }
            let fwd = match_wildcard(&ap.a, a) && match_wildcard(&ap.c, c);
            let rev = match_wildcard(&ap.a, a_rev) && match_wildcard(&ap.c, c_rev);
            if fwd || rev {
                return Some(ap);
            }
        }
        None
    }

    /// Find dihedral parameter by matching patterns. Supports wildcards '*'
    pub fn get_dihedral_param(&self, a: &str, b: &str, c: &str, d: &str, order: u8) -> Option<&DihedralParam> {
        let (a_rev, b_rev, c_rev, d_rev) = (d, c, b, a);
        for dp in &self.dihedrals {
            if dp.order != order { continue; }
            let fwd = match_wildcard(&dp.a, a) && match_wildcard(&dp.b, b) && match_wildcard(&dp.c, c) && match_wildcard(&dp.d, d);
            let rev = match_wildcard(&dp.a, a_rev) && match_wildcard(&dp.b, b_rev) && match_wildcard(&dp.c, c_rev) && match_wildcard(&dp.d, d_rev);
            if fwd || rev {
                return Some(dp);
            }
        }
        None
    }

    // ================== MMFFparams diagnostic prints ==================

    pub fn print_bond_type(&self, i: usize) {
        let t = &self.bonds[i];
        let ta = self.get_atom_type(&t.a).map(|a| a.name.as_str()).unwrap_or("?");
        let tb = self.get_atom_type(&t.b).map(|a| a.name.as_str()).unwrap_or("?");
        println!("bondType[{:3}] {}-{} l0({:7.3}) k({:7.3})", i, ta, tb, t.l0, t.k);
    }

    pub fn print_angle_type(&self, i: usize) {
        let t = &self.angles[i];
        let ta = self.get_atom_type(&t.a).map(|a| a.name.as_str()).unwrap_or("?");
        let tb = self.get_atom_type(&t.b).map(|a| a.name.as_str()).unwrap_or("?");
        let tc = self.get_atom_type(&t.c).map(|a| a.name.as_str()).unwrap_or("?");
        println!("angleType[{:3}] {}-{}-{} ang0({:7.3}) k({:7.3})", i, ta, tb, tc, t.a0, t.k);
    }

    pub fn print_dihedral_type(&self, i: usize) {
        let t = &self.dihedrals[i];
        let ta = self.get_atom_type(&t.a).map(|a| a.name.as_str()).unwrap_or("?");
        let tb = self.get_atom_type(&t.b).map(|a| a.name.as_str()).unwrap_or("?");
        let tc = self.get_atom_type(&t.c).map(|a| a.name.as_str()).unwrap_or("?");
        let td = self.get_atom_type(&t.d).map(|a| a.name.as_str()).unwrap_or("?");
        println!("dihedralType[{:3}] {}-{}-{}-{} ang0({:7.3}) k({:7.3}) n({})", i, ta, tb, tc, td, t.a0, t.k, t.n);
    }

    pub fn print_element_types(&self, b_params: bool) {
        println!("MMFFparams::printElementTypes()");
        for (i, et) in self.elements.iter().enumerate() { et.print(i, b_params); }
    }

    pub fn print_atom_types(&self, b_params: bool) {
        println!("MMFFparams::printAtomTypes()");
        for (i, at) in self.atom_types.iter().enumerate() { at.print(i, b_params); }
    }

    pub fn print_bond_types(&self) {
        println!("MMFFparams::printBondTypes()");
        for i in 0..self.bonds.len() { self.print_bond_type(i); }
    }

    pub fn print_angle_types(&self) {
        println!("MMFFparams::printAngleTypes()");
        for i in 0..self.angles.len() { self.print_angle_type(i); }
    }

    pub fn print_dihedral_types(&self) {
        println!("MMFFparams::printDihedralTypes()");
        for i in 0..self.dihedrals.len() { self.print_dihedral_type(i); }
    }

    pub fn print_atom_type_dict(&self) {
        for (i, at) in self.atom_types.iter().enumerate() {
            if let Some(&idx) = self.atom_type_dict.get(&at.name) {
                println!("AtomType[{:3}] {} {}", i, at.name, idx);
            }
        }
    }

    pub fn print_element_type_dict(&self) {
        for (i, et) in self.elements.iter().enumerate() {
            if let Some(&idx) = self.element_dict.get(&et.name) {
                println!("ElementType[{:3}] {} {}", i, et.name, idx);
            }
        }
    }

    pub fn print_types_of_atoms(&self, itypes: &[usize], b_name_only: bool, b_params: bool) {
        println!("MMFFparams::printTypesOfAtoms({}) ", itypes.len());
        for (i, &ityp) in itypes.iter().enumerate() {
            if b_name_only {
                let name = self.atom_types.get(ityp).map(|a| a.name.as_str()).unwrap_or("?");
                println!("atom {:3} t: {:3} {:8}", i, ityp, name);
            } else if let Some(at) = self.atom_types.get(ityp) {
                at.print(ityp, b_params);
            }
        }
    }
}

fn match_wildcard(pattern: &str, value: &str) -> bool {
    if pattern == "*" || pattern == value { return true; }
    // Fallback: element prefix (e.g. "C" matches "C_3", "C_R"; "H" matches "H_")
    let pat_elem = element_prefix(pattern);
    let val_elem = element_prefix(value);
    pat_elem == val_elem
}

fn element_prefix(s: &str) -> &str {
    // Extract element symbol: everything before '_' or first digit
    if let Some(i) = s.find('_') {
        &s[..i]
    } else if let Some(i) = s.find(|c: char| c.is_ascii_digit()) {
        &s[..i]
    } else {
        s
    }
}

/// Assign UFF atom types based on topology (simplified from UFFbuilder)
pub fn assign_uff_types(elems: &[String], neighs: &[[i32; 4]]) -> Vec<String> {
    let natoms = elems.len();
    let mut types: Vec<String> = elems.iter().map(|e| e.clone()).collect();
    let mut bond_orders: Vec<i32> = vec![-1; natoms * 4]; // per-neighbor bond order
    let mut set_atom: Vec<bool> = vec![false; natoms];
    let mut set_bond: Vec<bool> = vec![false; natoms * 4];

    // --- Trivial rules
    for ia in 0..natoms {
        let name = &elems[ia];
        let nbond = neighs[ia].iter().take_while(|&&n| n >= 0).count();
        if name == "H" {
            types[ia] = "H_".to_string();
            set_atom[ia] = true;
            if nbond > 0 {
                bond_orders[ia * 4] = 1;
                set_bond[ia * 4] = true;
            }
        } else if nbond == 4 {
            // generic sp3: try Element_3, then Element3
            let tname = format!("{}_3", name);
            if type_exists(&tname) {
                types[ia] = tname;
                set_atom[ia] = true;
                for in_ in 0..4 {
                    if neighs[ia][in_] < 0 { break; }
                    bond_orders[ia * 4 + in_] = 1;
                    set_bond[ia * 4 + in_] = true;
                }
            } else {
                let tname2 = format!("{}3", name);
                if type_exists(&tname2) {
                    types[ia] = tname2;
                    set_atom[ia] = true;
                    for in_ in 0..4 {
                        if neighs[ia][in_] < 0 { break; }
                        bond_orders[ia * 4 + in_] = 1;
                        set_bond[ia * 4 + in_] = true;
                    }
                }
            }
        } else if name == "N" && nbond == 1 {
            types[ia] = "N_1".to_string();
            set_atom[ia] = true;
            if nbond > 0 {
                bond_orders[ia * 4] = 3;
                set_bond[ia * 4] = true;
            }
        } else if name == "N" && nbond == 3 {
            // check for nitro-like (2 O neighbors)
            let mut n_o = 0;
            for in_ in 0..3 {
                let ja = neighs[ia][in_];
                if ja >= 0 && elems[ja as usize] == "O" { n_o += 1; }
            }
            if n_o == 2 {
                types[ia] = "N_R".to_string();
                set_atom[ia] = true;
                for in_ in 0..3 {
                    let ja = neighs[ia][in_];
                    if ja >= 0 {
                        if elems[ja as usize] == "O" {
                            types[ja as usize] = "O_R".to_string();
                            set_atom[ja as usize] = true;
                            bond_orders[ia * 4 + in_] = 2; // resonant
                            set_bond[ia * 4 + in_] = true;
                        } else {
                            bond_orders[ia * 4 + in_] = 1;
                            set_bond[ia * 4 + in_] = true;
                        }
                    }
                }
            } else {
                types[ia] = "N_3".to_string();
                set_atom[ia] = true;
                for in_ in 0..3 {
                    if neighs[ia][in_] >= 0 {
                        bond_orders[ia * 4 + in_] = 1;
                        set_bond[ia * 4 + in_] = true;
                    }
                }
            }
        } else if name == "O" && nbond == 2 {
            types[ia] = "O_3".to_string();
            set_atom[ia] = true;
            for in_ in 0..2 {
                if neighs[ia][in_] >= 0 {
                    bond_orders[ia * 4 + in_] = 1;
                    set_bond[ia * 4 + in_] = true;
                }
            }
        } else if name == "O" && nbond == 1 {
            types[ia] = "O_2".to_string();
            set_atom[ia] = true;
            if nbond > 0 {
                bond_orders[ia * 4] = 2;
                set_bond[ia * 4] = true;
            }
        } else if name == "C" && nbond == 3 {
            // sp2 carbon
            types[ia] = "C_R".to_string();
            set_atom[ia] = true;
            for in_ in 0..3 {
                if neighs[ia][in_] >= 0 {
                    bond_orders[ia * 4 + in_] = 1;
                    set_bond[ia * 4 + in_] = true;
                }
            }
        } else if name == "C" && nbond == 2 {
            // sp1 carbon
            types[ia] = "C_1".to_string();
            set_atom[ia] = true;
            for in_ in 0..2 {
                if neighs[ia][in_] >= 0 {
                    bond_orders[ia * 4 + in_] = 3;
                    set_bond[ia * 4 + in_] = true;
                }
            }
        }
    }

    // --- Fall back to element name for unassigned atoms
    for ia in 0..natoms {
        if !set_atom[ia] {
            // Keep element symbol as type if no specific UFF type assigned
            // (will be looked up in atom_types dict)
        }
    }

    types
}

/// Quick check if a UFF atom type exists in the standard set (hardcoded minimal set)
fn type_exists(name: &str) -> bool {
    let known = ["H_", "C_3", "C_R", "C_2", "C_1", "N_3", "N_R", "N_2", "N_1", "O_3", "O_R", "O_2", "O_1", "F", "Cl", "Si3", "P", "S"];
    known.contains(&name)
}

/// Look up atom type parameters. Returns (RvdW, sqrtEvdW, Qbase, Hb)
pub fn get_reqh(params: &Params, atype_name: &str) -> [f64; 4] {
    if let Some(at) = params.get_atom_type(atype_name) {
        return [at.r_vdw, at.e_vdw.sqrt(), at.q_base, at.h_b];
    }
    // Fallback to element
    if let Some(et) = params.get_element_type(atype_name) {
        return [et.r_vdw, et.e_vdw.sqrt(), et.q_uff, 0.0];
    }
    [1.5, 0.0, 0.0, 0.0]
}

/// 1 kcal/mol in eV. Used by UFF dihedral/inversion parameter assignment.
pub const KCAL_TO_EV: f64 = 4.1840 / 60.2214076 / 1.602176634;

/// Determine bond order from assigned UFF atom type hybridization suffix.
/// Ported from FireCore UFFbuilder type assignment logic:
///   - sp1 (_1) atoms → triple bond (BO=3)
///   - sp2 (_2) atoms → double bond (BO=2), but ONLY if the other atom is also _2
///   - aromatic (_R) atoms → single bond (BO=1) — UFF treats aromatic bonds as single
///   - H_ → always single bond
/// Note: FireCore sets bond order explicitly during type assignment, not by suffix inference.
/// This function approximates that logic: _R is aromatic (BO=1), _2 is true double bond.
pub fn bond_order_from_types(ta: &str, tb: &str) -> f64 {
    let is_sp1 = |t: &str| t.ends_with("_1");
    let is_double = |t: &str| t.ends_with("_2"); // true sp2 double bond, NOT aromatic _R
    if is_sp1(ta) || is_sp1(tb) { 3.0 }
    else if is_double(ta) && is_double(tb) { 2.0 }
    else { 1.0 } // aromatic _R, sp3, H, etc. → single bond
}

/// UFF bond length from atom type radii and element electronegativities.
/// Ported from FireCore UFFbuilder.h:1043 assignUFFparams_calcrij.
pub fn uff_bond_length(ti: &AtomType, tj: &AtomType, ei: &ElementType, ej: &ElementType, bo: f64) -> f64 {
    let r_bo = -0.1332 * (ti.r_uff + tj.r_uff) * bo.ln();
    let r_en = if ei.e_aff < 0.0 && ej.e_aff < 0.0 {
        let s = (-ei.e_aff).sqrt() - (-ej.e_aff).sqrt();
        ti.r_uff * tj.r_uff * s * s / (-ei.e_aff * ti.r_uff - ej.e_aff * tj.r_uff)
    } else { 0.0 };
    ti.r_uff + tj.r_uff + r_bo - r_en
}

/// UFF bond force constant. Ported from FireCore UFFbuilder.h:1075.
pub fn uff_bond_k(ei: &ElementType, ej: &ElementType, l0: f64) -> f64 {
    0.5 * 28.7989689090648 * ei.q_uff * ej.q_uff / (l0 * l0 * l0)
}

/// UFF angle Fourier coefficients for sp3. Ported from FireCore UFFbuilder.h:1122-1127.
pub fn uff_angle_sp3(ct: f64, st2: f64) -> (f64, f64, f64, f64) {
    let c2 = 1.0 / (4.0 * st2);
    let c1 = -4.0 * c2 * ct;
    let c0 = c2 * (2.0 * ct * ct + 1.0);
    (c0, c1, c2, 0.0)
}

/// UFF angle Fourier coefficients for sp2/R. Ported from FireCore UFFbuilder.h:1115-1120.
pub fn uff_angle_sp2() -> (f64, f64, f64, f64) {
    (1.0, 0.0, 0.0, -1.0)
}

/// UFF angle Fourier coefficients for sp1. Ported from FireCore UFFbuilder.h:1108-1114.
pub fn uff_angle_sp1() -> (f64, f64, f64, f64) {
    (1.0, 1.0, 0.0, 0.0)
}

/// Hybridization character from UFF atom type name
fn hybridization(tname: &str) -> char {
    if let Some(i) = tname.rfind('_') {
        if i + 1 < tname.len() { return tname.chars().nth(i + 1).unwrap_or('3'); }
    }
    // fallback: element types without suffix default to generic
    if tname == "H" || tname == "H_" { return '3'; }
    '3'
}

// TODO: setup_forcefield depends on UFF and NonBondedFF which belong in mol_engine
// This function should be moved to mol_engine or refactored to be forcefield-agnostic
/*
pub fn setup_forcefield(
    ff: &mut crate::uff::Uff,
    nb: &mut crate::nonbonded::NonBondedFF,
    params: &Params,
    elems: &[String],
) -> Vec<String> {
    // ... implementation moved to mol_engine
}
*/
