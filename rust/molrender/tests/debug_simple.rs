use surfmol_topology::params::Params;
use surfmol_molrender::ThumbnailRenderer;

#[test]
fn debug_simple_atoms() {
    let mut params = Params::new();
    params.load_element_types("../../data/ElementTypes.dat");

    let mut renderer = ThumbnailRenderer::new(128);

    // Single atom at origin
    let apos = vec![surfmol_common::math::vec3::Vec3d { x: 0.0, y: 0.0, z: 0.0 }];
    let elems = vec!["C".to_string()];
    let bonds: Vec<[usize; 2]> = vec![];

    let rgba = renderer.render(128, &apos, &elems, &bonds, &params);
    let clear = [80u8, 80, 97, 255]; // sRGB clear color
    let mut non_clear = 0;
    for i in (0..rgba.len()).step_by(4) {
        if rgba[i] != clear[0] || rgba[i+1] != clear[1] || rgba[i+2] != clear[2] {
            non_clear += 1;
        }
    }
    println!("single atom: non-clear pixels: {} / {}", non_clear, rgba.len()/4);
    assert!(non_clear > 0, "single atom at origin should render");
}
