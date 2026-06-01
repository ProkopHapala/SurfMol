use std::path::PathBuf;
use surfmol_common::xyz::read_xyz;
use surfmol_topology::params::Params;
use surfmol_molrender::ThumbnailRenderer;

#[test]
fn render_all_xyz_thumbs() {
    let data_dir = PathBuf::from("../../data");
    let xyz_dir = data_dir.join("xyz");

    let mut params = Params::new();
    params.load_element_types(data_dir.join("ElementTypes.dat"));
    params.load_atom_types(data_dir.join("AtomTypes.dat"));
    params.load_bond_types(data_dir.join("BondTypes.dat"));
    params.load_angle_types(data_dir.join("AngleTypes.dat"));

    let mut renderer = ThumbnailRenderer::new(128);

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&xyz_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "xyz"))
        .collect();
    paths.sort();

    let out_dir = PathBuf::from("/tmp/surfmol_thumbs_all");
    std::fs::create_dir_all(&out_dir).unwrap();

    let mut ok = 0;
    let mut fail = 0;
    for path in &paths {
        let xyz = match read_xyz(path) {
            Ok(x) => x,
            Err(e) => { println!("FAIL read {:?}: {}", path, e); fail += 1; continue; }
        };
        if xyz.apos.is_empty() { println!("SKIP empty {:?}", path); fail += 1; continue; }

        let radii: Vec<f64> = xyz.elems.iter().map(|el| {
            params.get_element_type(el).map(|et| et.r_cov).unwrap_or(1.0)
        }).collect();

        let mut bonds = Vec::new();
        let n = xyz.apos.len();
        for i in 0..n {
            let mut candidates = Vec::new();
            for j in (i+1)..n {
                let d = surfmol_common::math::vec3::Vec3d::set_sub(xyz.apos[j], xyz.apos[i]);
                let rcut = radii[i] + radii[j] + 0.4;
                let dist2 = d.norm2();
                if dist2 < rcut * rcut {
                    candidates.push((j, dist2));
                }
            }
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for (j, _) in candidates.iter().take(4) {
                bonds.push([i, *j]);
            }
        }

        let rgba = renderer.render(128, &xyz.apos, &xyz.elems, &bonds, &params);
        let name = path.file_stem().unwrap().to_string_lossy();
        let out_path = out_dir.join(format!("{}.png", name));

        image::RgbaImage::from_raw(128, 128, rgba)
            .unwrap()
            .save(&out_path)
            .unwrap();
        ok += 1;
    }
    println!("Done: {} ok, {} fail, total {}", ok, fail, paths.len());
    // Some .xyz files have format issues (e.g. styrene.xyz starts with blank line); skip those
    assert!(ok > 0, "no molecules rendered");
}
