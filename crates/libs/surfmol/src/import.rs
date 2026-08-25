use std::path::Path;

use moltopo::export::import_json;
use moltopo::topology::Topology;

use molff::uff::Uff;

/// Load topology from JSON file and create UFF engine
pub fn load_topology_from_json<P: AsRef<Path>>(path: P) -> Result<(Uff, Vec<String>), Box<dyn std::error::Error>> {
    let (topology, elements) = import_json(path)?;
    let ff = Uff::from_topology(&topology);
    Ok((ff, elements))
}
