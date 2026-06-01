use crate::params::Params;
use crate::topology::hybridization;

/// Assign UFF atom types based on topology neighbor list and octet-rule hybridization.
/// `neighs[ia]` are up to 4 neighbor atom indices, -1 padded.
/// Hybridization is derived from: 4 = nepair + nsigma + npi, where nsigma = neighbor count.
/// sp3 (npi=0) → _3,  sp2 (npi=1) → _2 / _R,  sp (npi=2) → _1.
pub fn assign_uff_types(elems: &[String], neighs: &[[i32; 4]]) -> Vec<String> {
    let natoms = elems.len();
    let mut types: Vec<String> = elems.iter().map(|e| e.clone()).collect();
    let mut bond_orders: Vec<i32> = vec![-1; natoms * 4];
    let mut set_atom: Vec<bool> = vec![false; natoms];
    let mut set_bond: Vec<bool> = vec![false; natoms * 4];

    for ia in 0..natoms {
        if set_atom[ia] { continue; }
        let name = &elems[ia];
        let nbond = neighs[ia].iter().take_while(|&&n| n >= 0).count() as i32;
        let h = hybridization(name, nbond);

        // --- Special cases (functional groups with known bond orders) ---
        // Hydrogen: always H_ in UFF, regardless of neighbor count
        if name == "H" {
            types[ia] = "H_".to_string();
            set_atom[ia] = true;
            if nbond > 0 {
                bond_orders[ia * 4] = 1;
                set_bond[ia * 4] = true;
            }
            continue;
        }

        // N in nitro group (NO2) — trigonal planar, both N and O are sp2 aromatic-like
        if name == "N" && nbond == 3 {
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
                            bond_orders[ia * 4 + in_] = 2;
                            set_bond[ia * 4 + in_] = true;
                        } else {
                            bond_orders[ia * 4 + in_] = 1;
                            set_bond[ia * 4 + in_] = true;
                        }
                    }
                }
                continue;
            }
        }

        // O with 1 neighbor (double bond, e.g. C=O) — sp2
        if name == "O" && nbond == 1 {
            types[ia] = "O_2".to_string();
            set_atom[ia] = true;
            bond_orders[ia * 4] = 2;
            set_bond[ia * 4] = true;
            continue;
        }

        // C with 2 neighbors (e.g. alkyne, allene center) — sp
        if name == "C" && nbond == 2 {
            types[ia] = "C_1".to_string();
            set_atom[ia] = true;
            bond_orders[ia * 4] = 3;
            set_bond[ia * 4] = true;
            bond_orders[ia * 4 + 1] = 3;
            set_bond[ia * 4 + 1] = true;
            continue;
        }

        // --- General case: map hybridization to UFF suffix ---
        let suffix = match h {
            3 => "3",   // sp3
            2 => "2",   // sp2 (non-aromatic default; aromatic override below)
            1 => "1",   // sp
            _ => "3",
        };

        let tname = format!("{}_{}", name, suffix);
        if type_exists(&tname) {
            types[ia] = tname.clone();
            set_atom[ia] = true;
            for in_ in 0..4 {
                let j = neighs[ia][in_];
                if j < 0 { break; }
                bond_orders[ia * 4 + in_] = 1;
                set_bond[ia * 4 + in_] = true;
            }
            // For sp2 C and N in all-carbon/sp2 rings, aromatic _R is more accurate.
            // Simple heuristic: if all neighbors are also sp2 C/N/O, treat as aromatic.
            if h == 2 && (name == "C" || name == "N") {
                let all_sp2 = (0..nbond as usize).all(|in_| {
                    let ja = neighs[ia][in_];
                    if ja < 0 { return false; }
                    let je = &elems[ja as usize];
                    let jbond = neighs[ja as usize].iter().take_while(|&&n| n >= 0).count() as i32;
                    hybridization(je, jbond) == 2
                });
                if all_sp2 {
                    let rname = format!("{}_R", name);
                    if type_exists(&rname) {
                        types[ia] = rname;
                    }
                }
            }
        } else {
            // Fallback: try unprefixed element name (e.g. "F", "Cl")
            if type_exists(name) {
                types[ia] = name.clone();
                set_atom[ia] = true;
            }
        }
    }

    // Warn about any atoms that could not be typed
    for ia in 0..natoms {
        if !set_atom[ia] {
            eprintln!("WARNING: atom {} ({}) could not be assigned a UFF type (hybridization={})", ia, elems[ia], hybridization(&elems[ia], neighs[ia].iter().take_while(|&&n| n >= 0).count() as i32));
        }
    }

    types
}

fn type_exists(name: &str) -> bool {
    let known = ["H_", "C_3", "C_R", "C_2", "C_1", "N_3", "N_R", "N_2", "N_1", "O_3", "O_R", "O_2", "O_1", "F", "Cl", "Si3", "P", "S"];
    known.contains(&name)
}

pub fn get_reqh(params: &Params, atype_name: &str) -> [f64; 4] { crate::params::get_reqh(params, atype_name) }
