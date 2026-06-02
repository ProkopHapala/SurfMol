use std::env;
use std::fs;
use std::path::PathBuf;

use serde::{Serialize, Deserialize};

use surfmol_common::xyz;
use surfmol_topology::builder::Builder;
use surfmol_topology::topology::hybridization;
use surfmol_topology::assign_uff::assign_uff_types;

/// Human-readable JSON output of topology + UFF assignment.
#[derive(Serialize, Deserialize, Debug)]
struct AtomInfo {
    index: usize,
    element: String,
    position: [f64; 3],
    uff_type: String,
    hybridization: i32,  // 1=sp, 2=sp2, 3=sp3
    neighbors: Vec<i32>,
}

#[derive(Serialize, Deserialize, Debug)]
struct BondInfo {
    index: usize,
    atoms: [i32; 2],
    order: i32,
}

#[derive(Serialize, Deserialize, Debug)]
struct AngleInfo {
    index: usize,
    atoms: [i32; 3],
}

#[derive(Serialize, Deserialize, Debug)]
struct DihedralInfo {
    index: usize,
    atoms: [i32; 4],
}

#[derive(Serialize, Deserialize, Debug)]
struct InversionInfo {
    index: usize,
    atoms: [i32; 4],
}

#[derive(Serialize, Deserialize, Debug)]
struct TopologyJson {
    natoms: usize,
    nbonds: usize,
    nangles: usize,
    ndihedrals: usize,
    ninversions: usize,
    atoms: Vec<AtomInfo>,
    bonds: Vec<BondInfo>,
    angles: Vec<AngleInfo>,
    dihedrals: Vec<DihedralInfo>,
    inversions: Vec<InversionInfo>,
}

/// Simple binary format for dense array ingestion by MD engine.
/// Header: magic "UFFTOPO" + u8 version(1) + 5xi32 counts + 1xi32 flags
/// Then sequential flat arrays:
///   apos:     natoms * 3 * f64
///   atypes:   natoms * i32  (type index into type_table)
///   bonds:    nbonds * 2 * i32
///   angles:   nangles * 3 * i32
///   dihedrals:ndihedrals * 4 * i32
///   inversions:ninversions * 4 * i32
///   type_table: N * 8 bytes (fixed-width type names, padded)
struct BinaryHeader {
    magic: [u8; 7],      // b"UFFTOPO"
    version: u8,         // 1
    natoms: i32,
    nbonds: i32,
    nangles: i32,
    ndihedrals: i32,
    ninversions: i32,
    ntypes: i32,
    flags: i32,          // reserved
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: assign-uff <xyz_file> [options]");
        println!("Options:");
        println!("  --json <path>       Write human-readable topology JSON");
        println!("  --bin <path>        Write flat binary arrays for MD ingestion");
        println!("  --tol <f>           Covalent radius tolerance (default 0.4 A)");
        println!("  --rcut <f>          Override with global cutoff (ignores radii)");
        return;
    }

    let xyz_path = PathBuf::from(&args[1]);
    let mut json_path: Option<PathBuf> = None;
    let mut bin_path: Option<PathBuf> = None;
    let mut tol = 0.4;
    let mut rcut: Option<f64> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => { i += 1; json_path = Some(PathBuf::from(&args[i])); }
            "--bin"  => { i += 1; bin_path  = Some(PathBuf::from(&args[i])); }
            "--tol"  => { i += 1; tol = args[i].parse().unwrap_or(0.4); }
            "--rcut" => { i += 1; rcut = Some(args[i].parse().unwrap_or(1.5)); }
            _ => {}
        }
        i += 1;
    }

    // 1. Read XYZ
    let sys = xyz::read_xyz(&xyz_path).expect("read_xyz failed");
    println!("Loaded {} atoms from {:?}", sys.elems.len(), xyz_path);

    // 2. Build topology
    let radii: Vec<f64> = sys.elems.iter().map(|el| {
        match el.as_str() {
            "H" => 0.31, "C" => 0.76, "N" => 0.71, "O" => 0.66,
            "F" => 0.57, "Si" => 1.11, "P" => 1.07, "S" => 1.05,
            "Cl" => 1.02, _ => 1.0,
        }
    }).collect();

    let top = if let Some(r) = rcut {
        Builder::from_positions_cutoff(&sys.apos, &sys.elems, r).bake()
    } else {
        Builder::from_positions_and_radii(&sys.apos, &sys.elems, &radii, tol).bake()
    };

    println!("Topology: {} atoms, {} bonds, {} angles, {} dihedrals, {} inversions",
             top.natoms(), top.bonds.len(), top.angles.len(), top.dihedrals.len(), top.inversions.len());

    // 3. Build neighbor list for type assignment
    let mut neighs = vec![[-1i32; 4]; top.natoms() as usize];
    let mut nneigh = vec![0usize; top.natoms() as usize];
    for b in &top.bonds {
        let i = b[0] as usize;
        let j = b[1] as usize;
        if nneigh[i] < 4 { neighs[i][nneigh[i]] = j as i32; nneigh[i] += 1; }
        if nneigh[j] < 4 { neighs[j][nneigh[j]] = i as i32; nneigh[j] += 1; }
    }

    // 4. Assign UFF types
    let uff_types = assign_uff_types(&sys.elems, &neighs);

    // 5. Print summary
    let mut counts = std::collections::HashMap::new();
    for t in &uff_types { *counts.entry(t.clone()).or_insert(0usize) += 1; }
    let mut kv: Vec<(String, usize)> = counts.into_iter().collect();
    kv.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    println!("\n=== UFF type histogram ===");
    for (t, c) in kv.iter() { println!("{:6}  {}", t, c); }

    // 6. JSON output
    if let Some(ref path) = json_path {
        let mut atoms = Vec::with_capacity(top.natoms() as usize);
        for i in 0..top.natoms() as usize {
            let p = top.apos[i];
            let n: Vec<i32> = neighs[i].iter().take_while(|&&n| n >= 0).copied().collect();
            atoms.push(AtomInfo {
                index: i,
                element: sys.elems[i].clone(),
                position: [p.x, p.y, p.z],
                uff_type: uff_types[i].clone(),
                hybridization: hybridization(&sys.elems[i], n.len() as i32),
                neighbors: n,
            });
        }

        let mut bonds = Vec::with_capacity(top.bonds.len());
        for (ib, b) in top.bonds.iter().enumerate() {
            bonds.push(BondInfo { index: ib, atoms: *b, order: 1 });
        }

        let mut angles = Vec::with_capacity(top.angles.len());
        for (ia, a) in top.angles.iter().enumerate() {
            angles.push(AngleInfo { index: ia, atoms: *a });
        }

        let mut dihedrals = Vec::with_capacity(top.dihedrals.len());
        for (id, d) in top.dihedrals.iter().enumerate() {
            dihedrals.push(DihedralInfo { index: id, atoms: [d.x, d.y, d.z, d.w] });
        }

        let mut inversions = Vec::with_capacity(top.inversions.len());
        for (ii, inv) in top.inversions.iter().enumerate() {
            inversions.push(InversionInfo { index: ii, atoms: [inv.x, inv.y, inv.z, inv.w] });
        }

        let out = TopologyJson {
            natoms: top.natoms() as usize,
            nbonds: top.bonds.len(),
            nangles: top.angles.len(),
            ndihedrals: top.dihedrals.len(),
            ninversions: top.inversions.len(),
            atoms, bonds, angles, dihedrals, inversions,
        };

        let json = serde_json::to_string_pretty(&out).expect("serialize JSON");
        fs::write(path, json).expect("write JSON");
        println!("Wrote JSON to {:?}", path);
    }

    // 7. Binary output
    if let Some(ref path) = bin_path {
        // Build type table
        let mut type_set: Vec<String> = uff_types.iter().cloned().collect();
        type_set.sort_unstable();
        type_set.dedup();
        let ntypes = type_set.len() as i32;

        // Map each atom to its type index
        let atype_indices: Vec<i32> = uff_types.iter().map(|t| {
            type_set.iter().position(|s| s == t).unwrap_or(0) as i32
        }).collect();

        let h = BinaryHeader {
            magic: *b"UFFTOPO",
            version: 1,
            natoms: top.natoms(),
            nbonds: top.bonds.len() as i32,
            nangles: top.angles.len() as i32,
            ndihedrals: top.dihedrals.len() as i32,
            ninversions: top.inversions.len() as i32,
            ntypes,
            flags: 0,
        };

        let mut buf: Vec<u8> = Vec::new();
        // Write header as raw bytes
        buf.extend_from_slice(&h.magic);
        buf.push(h.version);
        buf.extend_from_slice(&h.natoms.to_le_bytes());
        buf.extend_from_slice(&h.nbonds.to_le_bytes());
        buf.extend_from_slice(&h.nangles.to_le_bytes());
        buf.extend_from_slice(&h.ndihedrals.to_le_bytes());
        buf.extend_from_slice(&h.ninversions.to_le_bytes());
        buf.extend_from_slice(&h.ntypes.to_le_bytes());
        buf.extend_from_slice(&h.flags.to_le_bytes());

        // apos (natoms * 3 * f64)
        for p in &top.apos {
            buf.extend_from_slice(&p.x.to_le_bytes());
            buf.extend_from_slice(&p.y.to_le_bytes());
            buf.extend_from_slice(&p.z.to_le_bytes());
        }
        // atype indices (natoms * i32)
        for &t in &atype_indices {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        // bonds (nbonds * 2 * i32)
        for b in &top.bonds {
            buf.extend_from_slice(&b[0].to_le_bytes());
            buf.extend_from_slice(&b[1].to_le_bytes());
        }
        // angles (nangles * 3 * i32)
        for a in &top.angles {
            for j in 0..3 { buf.extend_from_slice(&a[j].to_le_bytes()); }
        }
        // dihedrals (ndihedrals * 4 * i32)
        for d in &top.dihedrals {
            buf.extend_from_slice(&d.x.to_le_bytes());
            buf.extend_from_slice(&d.y.to_le_bytes());
            buf.extend_from_slice(&d.z.to_le_bytes());
            buf.extend_from_slice(&d.w.to_le_bytes());
        }
        // inversions (ninversions * 4 * i32)
        for inv in &top.inversions {
            buf.extend_from_slice(&inv.x.to_le_bytes());
            buf.extend_from_slice(&inv.y.to_le_bytes());
            buf.extend_from_slice(&inv.z.to_le_bytes());
            buf.extend_from_slice(&inv.w.to_le_bytes());
        }
        // type table (ntypes * 8 bytes fixed-width, padded with spaces)
        for t in &type_set {
            let mut bytes = [b' '; 8];
            let tbytes = t.as_bytes();
            let n = tbytes.len().min(8);
            bytes[..n].copy_from_slice(&tbytes[..n]);
            buf.extend_from_slice(&bytes);
        }

        fs::write(path, buf).expect("write binary");
        println!("Wrote binary to {:?} ({} bytes)", path, fs::metadata(path).map(|m| m.len()).unwrap_or(0));
        println!("  Type table: {:?}", type_set);
    }
}
