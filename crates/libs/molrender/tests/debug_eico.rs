use std::path::PathBuf;
use moltopo::xyz::read_xyz;
use moltopo::params::Params;
use molrender::ThumbnailRenderer;

#[test]
fn debug_eicosanediol() {
    let data_dir = PathBuf::from("../../../data");
    let path = data_dir.join("xyz/1,20-eicosanediol.xyz");

    let mut params = Params::new();
    params.load_element_types(data_dir.join("ElementTypes.dat"));
    params.load_atom_types(data_dir.join("AtomTypes.dat"));
    params.load_bond_types(data_dir.join("BondTypes.dat"));
    params.load_angle_types(data_dir.join("AngleTypes.dat"));

    let mut renderer = ThumbnailRenderer::new(512);
    let xyz = read_xyz(&path).unwrap();

    println!("natoms: {}", xyz.apos.len());
    println!("elems: {:?}", xyz.elems);

    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    for p in &xyz.apos {
        let coords = [p.x as f32, p.y as f32, p.z as f32];
        for i in 0..3 {
            mn[i] = mn[i].min(coords[i]);
            mx[i] = mx[i].max(coords[i]);
        }
    }
    println!("bbox min: {:?}", mn);
    println!("bbox max: {:?}", mx);

    let radii: Vec<f64> = xyz.elems.iter().map(|el| {
        params.get_element_type(el).map(|et| et.r_cov).unwrap_or(1.0)
    }).collect();

    let mut bonds = Vec::new();
    let n = xyz.apos.len();
    for i in 0..n {
        let mut candidates = Vec::new();
        for j in (i+1)..n {
            let d = numcore::math::vec3::Vec3d::set_sub(xyz.apos[j], xyz.apos[i]);
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

    let rgba = renderer.render(512, &xyz.apos, &xyz.elems, &bonds, &params);
    let mut non_bg = 0;
    for i in (0..rgba.len()).step_by(4) {
        if rgba[i] != 20 || rgba[i+1] != 20 || rgba[i+2] != 31 {
            non_bg += 1;
        }
    }
    println!("non-bg pixels: {} / {}", non_bg, rgba.len()/4);

    let out = PathBuf::from("/tmp/eico_debug.png");
    image::RgbaImage::from_raw(512, 512, rgba).unwrap().save(&out).unwrap();
    println!("saved to {}", out.display());
}
