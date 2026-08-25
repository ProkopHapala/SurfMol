use std::fs;
use std::path::Path;

use numcore::math::vec3::Vec3d;

pub struct XyzSystem {
    pub apos: Vec<Vec3d>,
    pub elems: Vec<String>,
    pub charges: Vec<f64>,
}

pub fn read_xyz(path: &Path) -> Result<XyzSystem, String> {
    let s = fs::read_to_string(path).map_err(|e| format!("read_to_string({:?}) failed: {e}", path))?;
    let mut it = s.lines();
    let natoms: usize = it.next().ok_or("missing natoms")?.trim().parse().map_err(|e| format!("parse natoms failed: {e}"))?;
    let _comment = it.next().unwrap_or("");

    let mut apos = Vec::with_capacity(natoms);
    let mut elems = Vec::with_capacity(natoms);
    let mut charges = Vec::with_capacity(natoms);
    for (iline, line) in it.take(natoms).enumerate() {
        let mut w = line.split_whitespace();
        let el = w.next().ok_or_else(|| format!("line {} missing element", iline + 3))?.to_string();
        let x: f64 = w.next().ok_or_else(|| format!("line {} missing x", iline + 3))?.parse().map_err(|e| format!("line {} x parse: {e}", iline + 3))?;
        let y: f64 = w.next().ok_or_else(|| format!("line {} missing y", iline + 3))?.parse().map_err(|e| format!("line {} y parse: {e}", iline + 3))?;
        let z: f64 = w.next().ok_or_else(|| format!("line {} missing z", iline + 3))?.parse().map_err(|e| format!("line {} z parse: {e}", iline + 3))?;
        let q: f64 = w.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        elems.push(el);
        apos.push(Vec3d::new(x, y, z));
        charges.push(q);
    }

    if apos.len() != natoms { return Err(format!("expected {} atoms, got {}", natoms, apos.len())); }
    Ok(XyzSystem { apos, elems, charges })
}

pub fn write_xyz_frame(path: &Path, elems: &[String], apos: &[Vec3d], comment: &str, append: bool) -> Result<(), String> {
    let mut s = format!("{}\n{}\n", apos.len(), comment);
    for i in 0..apos.len() {
        s.push_str(&format!("{}  {:12.6} {:12.6} {:12.6}\n", elems[i], apos[i].x, apos[i].y, apos[i].z));
    }
    use std::fs::OpenOptions;
    use std::io::Write;
    let mut file = OpenOptions::new().create(true).append(append).write(true).truncate(!append).open(path)
        .map_err(|e| format!("open {:?} failed: {}", path, e))?;
    file.write_all(s.as_bytes()).map_err(|e| format!("write failed: {}", e))?;
    Ok(())
}
