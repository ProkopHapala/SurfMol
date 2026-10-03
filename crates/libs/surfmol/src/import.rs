use std::path::Path;

use moltopo::export::import_json;
use moltopo::topology::Topology;
use numtypes::Vec3d;

use molff::uff::Uff;

/// Load topology from JSON file and create UFF engine.
/// Returns (uff, elements, apos) — Uff does not own positions, so the caller
/// must copy apos into MolWorld::dyn_atoms (from_uff does not see them).
pub fn load_topology_from_json<P: AsRef<Path>>(path: P) -> Result<(Uff, Vec<String>, Vec<Vec3d>), Box<dyn std::error::Error>> {
    let (topology, elements) = import_json(path)?;
    let apos = topology.apos.clone();
    let ff = Uff::from_topology(&topology);
    Ok((ff, elements, apos))
}
