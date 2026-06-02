use std::path::PathBuf;
use surfmol_common::math::vec3::Vec3d;
use surfmol_common::xyz::read_xyz;
use surfmol_topology::params::Params;
use surfmol_apps::gui::thumbnailer::MolThumbnailer;

fn save_png(path: &std::path::Path, w: u32, h: u32, rgba: &[u8]) {
    let img = image::RgbaImage::from_raw(w, h, rgba.to_vec()).expect("bad image buffer");
    img.save(path).unwrap();
}

fn main() {
    let root = PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut params = Params::new();
    params.load_element_types(root.join("data/ElementTypes.dat"));
    params.load_atom_types(root.join("data/AtomTypes.dat"));
    params.load_bond_types(root.join("data/BondTypes.dat"));
    params.load_angle_types(root.join("data/AngleTypes.dat"));

    let mut thumb = MolThumbnailer::new();

    let out_dir = std::path::PathBuf::from("/tmp/thumb_test");
    std::fs::create_dir_all(&out_dir).unwrap();

    let size = 256u32;
    for name in ["1,20-eicosanediol", "BPBA", "Benzene_deriv", "NaCl_1x1_L3"] {
        let path = root.join(format!("data/xyz/{}.xyz", name));
        let xyz = read_xyz(&path).unwrap();

        // NOTE: bonds are optional for now; focus is ortho impostor correctness
        let rgba = thumb.render(size, &xyz.apos, &xyz.elems, &[], &params);
        let clear = [80u8, 80, 97, 255];
        let mut non_clear = 0usize;
        for i in (0..rgba.len()).step_by(4) {
            if rgba[i] != clear[0] || rgba[i+1] != clear[1] || rgba[i+2] != clear[2] { non_clear += 1; }
        }

        let out_path = out_dir.join(format!("{}.png", name));
        save_png(&out_path, size, size, &rgba);
        println!("saved {} non_clear={}/{}", out_path.display(), non_clear, (size*size) as usize);
    }
}
