//! Host-side data preparation for the RRsp3 cluster-sorted OpenCL layout.
//!
//! Ports the Python helpers from FireCore `pyBall/RigidAtomFF/RRsp3/RRsp3.py`
//! (`build_neighs_bk_from_bonds`, `make_bk_slots_clustered`, `make_exclusions_1st_2nd`,
//! `make_revSlot_clustered`) and `RRsp3_utils.py` (`make_ports_from_neighs`),
//! plus `XPTB_utils.py::pack_molecules_contiguous`.
//!
//! Cluster-sorted layout (Axis 4b): atoms are grouped into workgroups of
//! `GROUP_SIZE` (64). Within each group: node atoms first, then cap atoms,
//! then padding (invM=0, NaN pos) to fill the group. Only nodes have rotational
//! DOFs; caps are translated by their host node's quaternion via `bkSlots`.

use numtypes::Vec3d;

// ------------------------------------------------------------------
//  Types
// ------------------------------------------------------------------

/// A molecule to be packed: elements, positions, bonds (local indices), node count.
/// `nnode` = number of node atoms (heavy atoms with rotational DOF); caps (H) follow.
#[derive(Clone)]
pub struct MolInput {
    pub elems: Vec<String>,
    pub pos: Vec<[f32; 3]>,
    pub bonds: Vec<(usize, usize)>,
    pub nnode: usize,
}

/// Packed system ready for upload to the GPU. All arrays are flat and padded
/// to `natoms_total` (a multiple of `group_size`).
#[derive(Clone)]
pub struct PackedSystem {
    pub natoms: usize,
    pub group_size: usize,
    pub num_groups: usize,
    pub elems: Vec<String>,          // len natoms; "X" for padding
    pub pos: Vec<[f32; 3]>,          // len natoms; [0,0,0] for padding
    pub inv_mass: Vec<f32>,          // len natoms; 0 for padding
    pub is_padding: Vec<bool>,       // len natoms
    pub bonds: Vec<(usize, usize)>,  // global indices within packed array
    pub nnode_per_group: Vec<i32>,   // len num_groups
}

// ------------------------------------------------------------------
//  pack_molecules — port of XPTB_utils.py::pack_molecules_contiguous
// ------------------------------------------------------------------

/// Pack molecules into contiguous workgroups. Nodes first within each molecule,
/// then padding to fill each group to `group_size`.
///
/// Nodes are inferred from bond degree > 1 (matches the Python `nodes_first`
/// heuristic). If a molecule provides `nnode` that disagrees with the inferred
/// count, we panic (fail-loud — the Python version raises).
pub fn pack_molecules(mols: &[MolInput], group_size: usize) -> PackedSystem {
    assert!(group_size > 0, "pack_molecules: group_size must be > 0, got {group_size}");
    let mut elems_all: Vec<String> = Vec::new();
    let mut pos_all: Vec<[f32; 3]> = Vec::new();
    let mut inv_mass_all: Vec<f32> = Vec::new();
    let mut is_pad_all: Vec<bool> = Vec::new();
    let mut bonds_all: Vec<(usize, usize)> = Vec::new();
    let mut nnode_group: Vec<i32> = Vec::new();
    let mut base = 0usize;
    for (imol, mol) in mols.iter().enumerate() {
        let n = mol.pos.len();
        assert!(mol.elems.len() == n, "pack_molecules: mol[{imol}] elems.len()={} != n={n}", mol.elems.len());
        assert!(mol.nnode <= n, "pack_molecules: mol[{imol}] nnode={} > n={n}", mol.nnode);
        if n > group_size {
            panic!("pack_molecules: mol[{imol}] n={n} > group_size={group_size}; increase group_size or split molecule");
        }
        // Infer node mask from bond degree > 1 (matches Python nodes_first heuristic)
        let mut deg = vec![0i32; n];
        for &(i, j) in &mol.bonds {
            assert!(i < n && j < n, "pack_molecules: mol[{imol}] bond ({i},{j}) out of range [0,{n})");
            if i == j { continue; }
            deg[i] += 1; deg[j] += 1;
        }
        let node_mask: Vec<bool> = deg.iter().map(|&d| d > 1).collect();
        let nnode_inferred = node_mask.iter().filter(|&&b| b).count();
        assert!(nnode_inferred == mol.nnode, "pack_molecules: mol[{imol}] inferred nnode={nnode_inferred} from degree>1 but provided nnode={}; provide explicit perm if custom node/cap split needed", mol.nnode);
        // Permutation: nodes first, then caps
        let mut perm: Vec<usize> = (0..n).filter(|&i| node_mask[i]).collect();
        perm.extend((0..n).filter(|&i| !node_mask[i]));
        // perm_inv[perm[k]] = k
        let mut perm_inv = vec![0usize; n];
        for (k, &p) in perm.iter().enumerate() { perm_inv[p] = k; }
        // Append real atoms (reordered)
        let ng = base / group_size;
        for &p in &perm {
            elems_all.push(mol.elems[p].clone());
            pos_all.push(mol.pos[p]);
            inv_mass_all.push(0.0); // filled later from masses
            is_pad_all.push(false);
        }
        // Bonds in packed global indexing
        for &(i, j) in &mol.bonds {
            bonds_all.push((base + perm_inv[i], base + perm_inv[j]));
        }
        // Padding
        let n_pad = group_size - n;
        for _ in 0..n_pad {
            elems_all.push("X".to_string());
            pos_all.push([0.0, 0.0, 0.0]);
            inv_mass_all.push(0.0);
            is_pad_all.push(true);
        }
        nnode_group.push(mol.nnode as i32);
        base += n + n_pad;
        let _ = ng; // group id implicit from position
    }
    let natoms = pos_all.len();
    assert!(natoms % group_size == 0, "pack_molecules: natoms={natoms} not divisible by group_size={group_size}");
    let num_groups = natoms / group_size;
    PackedSystem { natoms, group_size, num_groups, elems: elems_all, pos: pos_all, inv_mass: inv_mass_all, is_padding: is_pad_all, bonds: bonds_all, nnode_per_group: nnode_group }
}

/// Fill `inv_mass` in the packed system from element masses (in atomic units).
/// Padding atoms keep inv_mass = 0. Real atoms get 1/mass.
pub fn set_masses_from_elements(packed: &mut PackedSystem, masses: &[f32]) {
    assert!(masses.len() == packed.natoms, "set_masses_from_elements: masses.len()={} != natoms={}", masses.len(), packed.natoms);
    for i in 0..packed.natoms {
        if packed.is_padding[i] { packed.inv_mass[i] = 0.0; }
        else {
            let m = masses[i];
            assert!(m > 0.0, "set_masses_from_elements: atom {i} has non-positive mass {m}");
            packed.inv_mass[i] = 1.0 / m;
        }
    }
}

/// Common element masses (atomic units). H=1.008, C=12.011, N=14.007, O=15.999, etc.
/// Padding ("X") = 0 (handled separately).
pub fn element_mass(elem: &str) -> f32 {
    match elem {
        "H" => 1.008, "C" => 12.011, "N" => 14.007, "O" => 15.999,
        "F" => 18.998, "P" => 30.974, "S" => 32.06, "Cl" => 35.45,
        "Br" => 79.904, "I" => 126.904, "Na" => 22.990, "K" => 39.098,
        "Mg" => 24.305, "Ca" => 40.078, "Fe" => 55.845, "Cu" => 63.546,
        "Zn" => 65.38, "Al" => 26.982, "Si" => 28.085, "B" => 10.81,
        "Li" => 6.94, "Be" => 9.012, "Ne" => 20.180, "Ar" => 39.948,
        _ => 12.0, // fallback for unknown
    }
}

/// Build the masses array for a packed system from its element labels.
pub fn masses_from_elems(packed: &PackedSystem) -> Vec<f32> {
    (0..packed.natoms).map(|i| if packed.is_padding[i] { 0.0 } else { element_mass(&packed.elems[i]) }).collect()
}

// ------------------------------------------------------------------
//  build_neighs_from_bonds — port of RRsp3.py::build_neighs_bk_from_bonds
// ------------------------------------------------------------------

/// Build neighbor lists (max degree 4) from bonds. Returns `neighs[natoms*4]`
/// (flattened, -1 = unused). Panics if any atom has degree > 4.
pub fn build_neighs_from_bonds(natoms: usize, bonds: &[(usize, usize)]) -> Vec<i32> {
    let mut neighs = vec![-1i32; natoms * 4];
    let mut deg = vec![0i32; natoms];
    for &(i, j) in bonds {
        assert!(i < natoms && j < natoms, "build_neighs_from_bonds: bond ({i},{j}) out of range [0,{natoms})");
        if deg[i] >= 4 { panic!("build_neighs_from_bonds: atom {i} has degree >= 4 (bond {i}-{j})"); }
        if deg[j] >= 4 { panic!("build_neighs_from_bonds: atom {j} has degree >= 4 (bond {i}-{j})"); }
        neighs[i * 4 + deg[i] as usize] = j as i32;
        neighs[j * 4 + deg[j] as usize] = i as i32;
        deg[i] += 1; deg[j] += 1;
    }
    neighs
}

// ------------------------------------------------------------------
//  make_exclusions_1st_2nd — port of RRsp3.py::make_exclusions_1st_2nd
// ------------------------------------------------------------------

/// Build 1st and 2nd neighbor exclusion lists (each up to 4 entries, -1 = unused).
/// `excl1` = first neighbors (from neighs), `excl2` = second neighbors
/// (neighbors of neighbors, excluding self and first neighbors).
pub fn make_exclusions_1st_2nd(neighs: &[i32], natoms: usize) -> (Vec<i32>, Vec<i32>) {
    assert!(neighs.len() == natoms * 4, "make_exclusions_1st_2nd: neighs.len()={} != natoms*4={}", neighs.len(), natoms * 4);
    let mut excl1 = vec![-1i32; natoms * 4];
    let mut excl2 = vec![-1i32; natoms * 4];
    for i in 0..natoms {
        // first neighbors
        let mut n1: Vec<i32> = Vec::new();
        for k in 0..4 {
            let j = neighs[i * 4 + k];
            if j >= 0 { n1.push(j); }
        }
        let n1_trunc: Vec<i32> = n1.iter().take(4).copied().collect();
        for (k, &j) in n1_trunc.iter().enumerate() { excl1[i * 4 + k] = j; }
        // second neighbors
        let s1set: std::collections::HashSet<i32> = n1_trunc.iter().copied().collect();
        let mut s2: Vec<i32> = Vec::new();
        'outer: for &j in &n1 {
            for k in 0..4 {
                let t = neighs[j as usize * 4 + k];
                if t < 0 || t == i as i32 { continue; }
                if s1set.contains(&t) { continue; }
                if s2.contains(&t) { continue; }
                s2.push(t);
                if s2.len() >= 4 { break 'outer; }
            }
        }
        for (k, &t) in s2.iter().take(4).enumerate() { excl2[i * 4 + k] = t; }
    }
    (excl1, excl2)
}

// ------------------------------------------------------------------
//  make_bk_slots_clustered — port of RRsp3.py::make_bk_slots_clustered
// ------------------------------------------------------------------

/// Build back-slots for recoil gathering. `bkSlots[natoms*4]` maps each atom's
/// port slot k to the node-port index `inode*4 + k` that will recoil it.
///
/// For each node `ia` (in group `ig`, local index `il`), its port k points to
/// neighbor `ja`. The slot `bkSlots[ja, s]` (s = next free slot for ja) is set
/// to `inode*4 + k`, so that when corrections are applied, atom `ja` gathers
/// the recoil impulse from that node-port.
pub fn make_bk_slots_clustered(neighs: &[i32], group_size: usize, nnode_per_group: &[i32], natoms: usize) -> Vec<i32> {
    assert!(natoms % group_size == 0, "make_bk_slots_clustered: natoms={natoms} not multiple of group_size={group_size}");
    assert!(neighs.len() == natoms * 4, "make_bk_slots_clustered: neighs.len()={} != natoms*4", neighs.len());
    let ng = natoms / group_size;
    let mut bk_slots = vec![-1i32; natoms * 4];
    let mut bk_count = vec![0i32; natoms];
    for ig in 0..ng {
        let abase = ig * group_size;
        let inode_base = ig * nnode_per_group[ig] as usize;
        for il in 0..nnode_per_group[ig] as usize {
            let ia = abase + il;
            let inode = inode_base + il;
            for k in 0..4 {
                let ja = neighs[ia * 4 + k];
                if ja < 0 { continue; }
                let s = bk_count[ja as usize];
                if s >= 4 { panic!("make_bk_slots_clustered: bkSlots overflow: atom {ja} has >4 back slots (from group {ig} node {il})"); }
                bk_slots[ja as usize * 4 + s as usize] = (inode as i32) * 4 + k as i32;
                bk_count[ja as usize] += 1;
            }
        }
    }
    bk_slots
}

// ------------------------------------------------------------------
//  make_rev_slot_clustered — port of RRsp3.py::make_revSlot_clustered
// ------------------------------------------------------------------

/// Build reverse-slot mapping `revSlot[nnode_tot*4]`: for each node-port (inode,k),
/// the reciprocal node-port (jnode,kk) such that neighs[ja,kk]==ia.
/// Only valid for node-node bonds (both endpoints are nodes). -1 for node-cap bonds.
pub fn make_rev_slot_clustered(neighs: &[i32], group_size: usize, nnode_per_group: &[i32], natoms: usize) -> Vec<i32> {
    assert!(natoms % group_size == 0, "make_rev_slot_clustered: natoms={natoms} not multiple of group_size={group_size}");
    assert!(neighs.len() == natoms * 4, "make_rev_slot_clustered: neighs.len()={} != natoms*4", neighs.len());
    let ng = natoms / group_size;
    let nnode_tot: usize = nnode_per_group.iter().map(|&n| n as usize).sum();
    let mut rev_slot = vec![-1i32; nnode_tot * 4];
    for ig in 0..ng {
        let abase = ig * group_size;
        let inode_base = ig * nnode_per_group[ig] as usize;
        for il in 0..nnode_per_group[ig] as usize {
            let ia = abase + il;
            let inode = inode_base + il;
            let i4 = inode * 4;
            for k in 0..4 {
                let ja = neighs[ia * 4 + k];
                if ja < 0 { continue; }
                let jl = (ja as usize) % group_size;
                if jl >= nnode_per_group[ig] as usize { continue; } // neighbor is a cap, skip
                // find reciprocal slot kk where neighs[ja, kk] == ia
                let mut kk = -1i32;
                for t in 0..4 {
                    if neighs[ja as usize * 4 + t] == ia as i32 { kk = t as i32; break; }
                }
                if kk < 0 { panic!("make_rev_slot_clustered: missing reciprocal neigh: {ia} -> {ja} (slot {k})"); }
                let jnode = (ja as usize / group_size) * nnode_per_group[ig] as usize + jl;
                rev_slot[i4 + k] = (jnode as i32) * 4 + kk;
            }
        }
    }
    rev_slot
}

// ------------------------------------------------------------------
//  make_ports_from_neighs — port of RRsp3_utils.py::make_ports_from_neighs
// ------------------------------------------------------------------

/// Build per-atom port local vectors and stiffness from neighbor positions.
/// `port_local[natoms*4*4]` (flattened [natoms,4,4], .xyz = direction, .w = 0),
/// `kflat[natoms*4]` = stiffness per port. Only the first `nnode_per_group`
/// atoms per group are used by the kernel, but we compute for all (harmless).
///
/// Port direction = `pos[j] - pos[i]` (the initial neighbor displacement).
/// This is the per-atom ARAP convention: identity rotation aligns all ports.
pub fn make_ports_from_neighs(pos: &[[f32; 3]], neighs: &[i32], natoms: usize, k_stiff: f32) -> (Vec<f32>, Vec<f32>) {
    assert!(pos.len() == natoms, "make_ports_from_neighs: pos.len()={} != natoms={natoms}", pos.len());
    assert!(neighs.len() == natoms * 4, "make_ports_from_neighs: neighs.len()={} != natoms*4", neighs.len());
    let mut port_local = vec![0.0f32; natoms * 4 * 4];
    let mut kflat = vec![0.0f32; natoms * 4];
    for i in 0..natoms {
        for k in 0..4 {
            let j = neighs[i * 4 + k];
            if j < 0 { continue; }
            let ju = j as usize;
            port_local[(i * 4 + k) * 4 + 0] = pos[ju][0] - pos[i][0];
            port_local[(i * 4 + k) * 4 + 1] = pos[ju][1] - pos[i][1];
            port_local[(i * 4 + k) * 4 + 2] = pos[ju][2] - pos[i][2];
            // .w stays 0
            kflat[i * 4 + k] = k_stiff;
        }
    }
    (port_local, kflat)
}

// ------------------------------------------------------------------
//  Geometry helpers for building test molecules
// ------------------------------------------------------------------

/// Water geometry: O at origin, two H's at ~0.96 Å, angle ~104.5°.
/// Atom order: 0=O, 1=H1, 2=H2. Bonds: (0,1),(0,2). nnode=1 (O is the node).
pub fn make_h2o_geometry() -> (Vec<[f32; 3]>, Vec<(usize, usize)>, usize, Vec<String>) {
    let angle = 104.5f32.to_radians();
    let r_oh = 0.96;
    let h1 = [r_oh * (angle * 0.5).cos(), r_oh * (angle * 0.5).sin(), 0.0];
    let h2 = [r_oh * (angle * 0.5).cos(), -r_oh * (angle * 0.5).sin(), 0.0];
    let pos = vec![[0.0, 0.0, 0.0], h1, h2];
    let bonds = vec![(0, 1), (0, 2)];
    let elems = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    (pos, bonds, 1, elems)
}

/// Methanol CH3OH geometry. Atom order: 0=C, 1=O, 2-4=H(methyl), 5=H(hydroxyl).
/// Nodes: C(0), O(1). Bonds: C-O, C-Hx3, O-H.
pub fn make_ch3oh_geometry() -> (Vec<[f32; 3]>, Vec<(usize, usize)>, usize, Vec<String>) {
    let c = [0.0, 0.0, 0.0];
    let o = [1.43, 0.0, 0.0];
    let tet = 109.5f32.to_radians();
    let cos_tet = tet.cos();
    let sin_tet = tet.sin();
    let h_c1 = [-1.09, 0.0, 0.0];
    let h_c2 = [1.09 * cos_tet, 1.09 * sin_tet, 0.0];
    let h_c3 = [1.09 * cos_tet, 1.09 * sin_tet * 120f32.to_radians().cos(), 1.09 * sin_tet * 120f32.to_radians().sin()];
    let oh_ang = 108.5f32.to_radians();
    let h_o = [o[0] + 0.96 * (std::f32::consts::PI - oh_ang).cos(), o[1] + 0.96 * (std::f32::consts::PI - oh_ang).sin(), 0.0];
    let pos = vec![c, o, h_c1, h_c2, h_c3, h_o];
    let bonds = vec![(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)];
    let elems = vec!["C".to_string(), "O".to_string(), "H".to_string(), "H".to_string(), "H".to_string(), "H".to_string()];
    (pos, bonds, 2, elems)
}

/// Convert a `Vec3d` position array to `[[f32;3]]` (f32 for GPU upload).
pub fn pos3d_to_f32(pos: &[Vec3d]) -> Vec<[f32; 3]> {
    pos.iter().map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect()
}

// ------------------------------------------------------------------
//  Tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_two_water() {
        let (pos, bonds, nnode, elems) = make_h2o_geometry();
        let m1 = MolInput { elems: elems.clone(), pos: pos.clone(), bonds: bonds.clone(), nnode };
        let m2 = MolInput { elems, pos: pos.iter().map(|p| [p[0] + 4.0, p[1], p[2]]).collect(), bonds, nnode };
        let packed = pack_molecules(&[m1, m2], 64);
        assert_eq!(packed.natoms, 128);
        assert_eq!(packed.num_groups, 2);
        assert_eq!(packed.nnode_per_group, vec![1, 1]);
        // O is node (degree 2), H's are caps (degree 1)
        assert_eq!(packed.elems[0], "O"); // group 0 node
        assert_eq!(packed.elems[1], "H");
        assert_eq!(packed.elems[2], "H");
        assert!(packed.is_padding[3]);
        assert_eq!(packed.elems[64], "O"); // group 1 node
    }

    #[test]
    fn test_build_neighs_water() {
        let neighs = build_neighs_from_bonds(3, &[(0, 1), (0, 2)]);
        // atom 0 (O) has neighbors 1, 2
        assert_eq!(neighs[0], 1); assert_eq!(neighs[1], 2); assert_eq!(neighs[2], -1); assert_eq!(neighs[3], -1);
        // atom 1 (H) has neighbor 0
        assert_eq!(neighs[4], 0); assert_eq!(neighs[5], -1);
        // atom 2 (H) has neighbor 0
        assert_eq!(neighs[8], 0); assert_eq!(neighs[9], -1);
    }

    #[test]
    fn test_exclusions_water() {
        let neighs = build_neighs_from_bonds(3, &[(0, 1), (0, 2)]);
        let (excl1, excl2) = make_exclusions_1st_2nd(&neighs, 3);
        // O (atom 0): 1st neighs = [1, 2], 2nd neighs = [] (H's have no other neighbors)
        assert_eq!(excl1[0], 1); assert_eq!(excl1[1], 2);
        assert_eq!(excl2[0], -1);
        // H1 (atom 1): 1st neigh = [0], 2nd neigh = [2] (neighbor of O, not self, not 1st)
        assert_eq!(excl1[4], 0);
        assert_eq!(excl2[4], 2);
    }

    #[test]
    fn test_bk_slots_water_packed() {
        let (pos, bonds, nnode, elems) = make_h2o_geometry();
        let m1 = MolInput { elems: elems.clone(), pos: pos.clone(), bonds: bonds.clone(), nnode };
        let packed = pack_molecules(&[m1], 64);
        let neighs = build_neighs_from_bonds(packed.natoms, &packed.bonds);
        let bk = make_bk_slots_clustered(&neighs, 64, &packed.nnode_per_group, packed.natoms);
        // O (atom 0) is node, inode=0. Its port 0 -> H1 (atom 1), port 1 -> H2 (atom 2).
        // H1 (atom 1) should have bkSlots[1*4+0] = 0*4+0 = 0 (inode 0, port 0)
        // H2 (atom 2) should have bkSlots[2*4+0] = 0*4+1 = 1 (inode 0, port 1)
        assert_eq!(bk[1 * 4 + 0], 0, "H1 bkSlot should be inode0*4+0=0");
        assert_eq!(bk[2 * 4 + 0], 1, "H2 bkSlot should be inode0*4+1=1");
    }

    #[test]
    fn test_ports_from_neighs_water() {
        let (pos, bonds, nnode, elems) = make_h2o_geometry();
        let neighs = build_neighs_from_bonds(3, &bonds);
        let (port_local, kflat) = make_ports_from_neighs(&pos, &neighs, 3, 200.0);
        // O (atom 0), port 0 -> H1 (atom 1): direction = pos[1] - pos[0]
        let dx = port_local[(0 * 4 + 0) * 4 + 0];
        let dy = port_local[(0 * 4 + 0) * 4 + 1];
        assert!((dx - pos[1][0]).abs() < 1e-5, "port dir x = {dx}, expected {}", pos[1][0]);
        assert!((dy - pos[1][1]).abs() < 1e-5, "port dir y = {dy}, expected {}", pos[1][1]);
        assert_eq!(kflat[0], 200.0);
    }
}
