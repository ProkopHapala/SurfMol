//! Test that the new fragment files parse correctly with ClAssembler.

use oclff::assemble::{ClAssembler, ClLibrary, Substitutions};

#[test]
fn parse_gridff_build_fragments() {
    let src = include_str!("../../../../opencl/gridff_build.cl");
    let lib = ClLibrary::parse(src);
    // 21 build+utility kernels
    assert!(lib.functions.contains_key("BsplineConv3D"), "missing BsplineConv3D");
    assert!(lib.functions.contains_key("make_MorseFF"), "missing make_MorseFF");
    assert!(lib.functions.contains_key("make_GridFF"), "missing make_GridFF");
    assert!(lib.functions.contains_key("poissonW"), "missing poissonW");
    assert!(lib.functions.contains_key("project_atom_on_grid_cubic_pbc"), "missing project_atom_on_grid_cubic_pbc");
    assert_eq!(lib.functions.len(), 21, "expected 21 function blocks, got {}", lib.functions.len());
    assert!(lib.macros.is_empty(), "build file should have no macros");
}

#[test]
fn parse_gridff_eval_macros() {
    let src = include_str!("../../../../opencl/gridff_eval.cl");
    let lib = ClLibrary::parse(src);
    assert!(lib.macros.contains_key("SAMPLE_3D"), "missing SAMPLE_3D");
    assert!(lib.macros.contains_key("SAMPLE_3D_GRID"), "missing SAMPLE_3D_GRID");
    assert!(lib.macros.contains_key("SAMPLE_3D_COMB"), "missing SAMPLE_3D_COMB");
    assert!(lib.macros.contains_key("SAMPLE_3D_COMB2"), "missing SAMPLE_3D_COMB2");
    assert!(lib.macros.contains_key("SAMPLE_1D_PBC"), "missing SAMPLE_1D_PBC");
    assert!(lib.macros.contains_key("SAMPLE_GRIDFF_BSPLINE_POINTS"), "missing SAMPLE_GRIDFF_BSPLINE_POINTS");
    assert!(lib.macros.contains_key("SAMPLE_GRIDFF"), "missing SAMPLE_GRIDFF");
    assert_eq!(lib.macros.len(), 7, "expected 7 macro blocks, got {}", lib.macros.len());
    // Each macro body should contain the __kernel definition
    for (_name, body) in &lib.macros {
        assert!(body.contains("__kernel void"), "macro body missing __kernel: {}", _name);
    }
}

#[test]
fn parse_faf_build_fragments() {
    let src = include_str!("../../../../opencl/faf_build.cl");
    let lib = ClLibrary::parse(src);
    assert!(lib.functions.contains_key("getSurfMorse"), "missing getSurfMorse");
    assert!(lib.functions.contains_key("compute_ewald_coefficients"), "missing compute_ewald_coefficients");
    assert!(lib.functions.contains_key("eval_potential_brute"), "missing eval_potential_brute");
    assert!(lib.functions.contains_key("getSurfaceIsoGridFF"), "missing getSurfaceIsoGridFF");
    assert!(lib.functions.contains_key("addDipoleField"), "missing addDipoleField");
    assert_eq!(lib.functions.len(), 10, "expected 10 function blocks, got {}", lib.functions.len());
    assert!(lib.macros.is_empty(), "build file should have no macros");
}

#[test]
fn parse_faf_eval_macros() {
    let src = include_str!("../../../../opencl/faf_eval.cl");
    let lib = ClLibrary::parse(src);
    assert!(lib.macros.contains_key("GET_SURF_FOLDED"), "missing GET_SURF_FOLDED");
    assert!(lib.macros.contains_key("GET_SURF_FOLDED_WORKGROUP"), "missing GET_SURF_FOLDED_WORKGROUP");
    assert!(lib.macros.contains_key("GET_SURF_FOLDED_HARMONICS"), "missing GET_SURF_FOLDED_HARMONICS");
    assert!(lib.macros.contains_key("GET_SURF_FOLDED_TENSOR_EXP"), "missing GET_SURF_FOLDED_TENSOR_EXP");
    assert!(lib.macros.contains_key("GET_SURF_FOLDED_TENSOR_POLY"), "missing GET_SURF_FOLDED_TENSOR_POLY");
    assert_eq!(lib.macros.len(), 5, "expected 5 macro blocks, got {}", lib.macros.len());
    for (_name, body) in &lib.macros {
        assert!(body.contains("__kernel void"), "macro body missing __kernel: {}", _name);
    }
}

#[test]
fn assemble_gridff_eval_injection() {
    // Test that we can assemble a template that injects a GridFF eval macro
    let mut asm = ClAssembler::new();
    asm.add_fragment("common.cl", "#define EXCL_MAX 32\ninline float3 modulo(float3 a, float3 b){ return a; }\n");
    asm.add_fragment("Forces.cl", "//>>>function getLJQH\ninline float4 getLJQH(float3 dp, float4 REQ, float R2damp){ return (float4)(0.0f); }\n");
    asm.add_fragment("gridff_eval.cl", include_str!("../../../../opencl/gridff_eval.cl"));

    let template = r#"//<<<file common.cl
//<<<file Forces.cl
//<<<file gridff_eval.cl

__kernel void testGetNonBonded(){
    //<<<macro SAMPLE_3D
}
"#;

    let eval_lib = asm.library("gridff_eval.cl").expect("gridff_eval.cl not registered");
    let mut subs = Substitutions::new();
    subs.macros.insert("SAMPLE_3D".to_string(), eval_lib.macros["SAMPLE_3D"].clone());

    let out = asm.assemble(template, &subs).expect("assembly failed");
    assert!(out.contains("#define EXCL_MAX 32"), "common.cl not injected");
    assert!(out.contains("__kernel void sample3D("), "SAMPLE_3D macro not injected");
    assert!(out.contains("__kernel void testGetNonBonded"), "host kernel not in output");
}
