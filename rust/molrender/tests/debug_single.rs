use surfmol_topology::params::Params;
use surfmol_molrender::ThumbnailRenderer;

#[test]
fn debug_single_atom() {
    let mut params = Params::new();
    params.load_element_types("../../data/ElementTypes.dat");
    params.load_atom_types("../../data/AtomTypes.dat");
    params.load_bond_types("../../data/BondTypes.dat");
    params.load_angle_types("../../data/AngleTypes.dat");

    let mut renderer = ThumbnailRenderer::new(128);
    let apos = vec![surfmol_common::math::vec3::Vec3d { x: 0.0, y: 0.0, z: 0.0 }];
    let elems = vec!["C".to_string()];
    let bonds: Vec<[usize; 2]> = vec![];

    let rgba = renderer.render(128, &apos, &elems, &bonds, &params);
    let mut non_bg = 0;
    for i in (0..rgba.len()).step_by(4) {
        if rgba[i] != 20 || rgba[i+1] != 20 || rgba[i+2] != 31 {
            non_bg += 1;
        }
    }
    println!("single atom non-bg pixels: {} / {}", non_bg, rgba.len()/4);
    assert!(non_bg > 0, "single atom at origin should be visible");
}
