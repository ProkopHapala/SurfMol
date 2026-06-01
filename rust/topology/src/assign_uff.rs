use crate::params::Params;

/// Assign UFF atom types based on topology neighbor list (simplified from FireCore C++ UFFBuilder).
/// `neighs[ia]` are up to 4 neighbor atom indices, -1 padded.
pub fn assign_uff_types(elems: &[String], neighs: &[[i32; 4]]) -> Vec<String> {
    let natoms = elems.len();
    let mut types: Vec<String> = elems.iter().map(|e| e.clone()).collect();
    let mut bond_orders: Vec<i32> = vec![-1; natoms * 4];
    let mut set_atom: Vec<bool> = vec![false; natoms];
    let mut set_bond: Vec<bool> = vec![false; natoms * 4];

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
            types[ia] = "C_R".to_string();
            set_atom[ia] = true;
            for in_ in 0..3 {
                if neighs[ia][in_] >= 0 {
                    bond_orders[ia * 4 + in_] = 1;
                    set_bond[ia * 4 + in_] = true;
                }
            }
        } else if name == "C" && nbond == 2 {
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

    for ia in 0..natoms {
        if !set_atom[ia] {
            let _ = &types[ia];
        }
    }

    types
}

fn type_exists(name: &str) -> bool {
    let known = ["H_", "C_3", "C_R", "C_2", "C_1", "N_3", "N_R", "N_2", "N_1", "O_3", "O_R", "O_2", "O_1", "F", "Cl", "Si3", "P", "S"];
    known.contains(&name)
}

pub fn get_reqh(params: &Params, atype_name: &str) -> [f64; 4] { crate::params::get_reqh(params, atype_name) }
