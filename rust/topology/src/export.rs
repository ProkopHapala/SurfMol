use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use surfmol_common::math::quat4::Quat4i;
use surfmol_common::math::vec3::Vec3d;

use crate::topology::Topology;

/// Serializable topology data for export
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TopologyData {
    pub natoms: usize,
    pub elements: Vec<String>,
    pub positions: Vec<[f64; 3]>,
    pub bonds: Vec<[i32; 2]>,
    pub angles: Vec<[i32; 3]>,
    pub dihedrals: Vec<[i32; 4]>,
    pub inversions: Vec<[i32; 4]>,
    // Forcefield parameters (optional, can be empty)
    pub bond_params: Vec<[f64; 2]>,
    pub angle_params: Vec<[f64; 5]>,
    pub dihedral_params: Vec<[f64; 3]>,
    pub inversion_params: Vec<[f64; 4]>,
    pub atom_params: Vec<[f64; 4]>,
}

impl Topology {
    /// Export topology to JSON format
    pub fn export_json<P: AsRef<Path>>(&self, path: P, elements: &[String]) -> Result<(), Box<dyn std::error::Error>> {
        let data = TopologyData {
            natoms: self.apos.len(),
            elements: elements.to_vec(),
            positions: self.apos.iter().map(|v| [v.x, v.y, v.z]).collect(),
            bonds: self.bonds.clone(),
            angles: self.angles.clone(),
            dihedrals: self.dihedrals.iter().map(|q| [q.x, q.y, q.z, q.w]).collect(),
            inversions: self.inversions.iter().map(|q| [q.x, q.y, q.z, q.w]).collect(),
            bond_params: vec![], // TODO: add when parameters are computed
            angle_params: vec![],
            dihedral_params: vec![],
            inversion_params: vec![],
            atom_params: vec![],
        };

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &data)?;
        Ok(())
    }

    // TODO: Add .npy export once we determine correct npy crate API
    // For now, JSON export is sufficient for the initial implementation
}

/// Import topology from JSON format
pub fn import_json<P: AsRef<Path>>(path: P) -> Result<(Topology, Vec<String>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let data: TopologyData = serde_json::from_reader(file)?;

    let apos: Vec<Vec3d> = data.positions.iter().map(|p| Vec3d::new(p[0], p[1], p[2])).collect();
    let dihedrals: Vec<Quat4i> = data.dihedrals.iter().map(|d| Quat4i::new(d[0], d[1], d[2], d[3])).collect();
    let inversions: Vec<Quat4i> = data.inversions.iter().map(|i| Quat4i::new(i[0], i[1], i[2], i[3])).collect();

    let topology = Topology {
        apos,
        bonds: data.bonds,
        angles: data.angles,
        dihedrals,
        inversions,
    };

    Ok((topology, data.elements))
}
