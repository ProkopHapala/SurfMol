//! Test assembly of getNonBond_generic.cl with different axis-variant combinations.
//!
//! Verifies that the 3-axis macro-assembler produces valid OpenCL source for
//! all 4 combinations: {LJQH} × {NEIGHS4} × {NONE, GRIDFF_BSPLINE, FAF}.

use oclff::assemble::{ClAssembler, Substitutions};

const COMMON_CL: &str = include_str!("../../../../opencl/common.cl");
const FORCES_CL: &str = include_str!("../../../../opencl/Forces.cl");
const NB_COMMON_CL: &str = include_str!("../../../../opencl/nb_common.cl");
const GRIDFF_EVAL_CL: &str = include_str!("../../../../opencl/gridff_eval.cl");
const FAF_EVAL_CL: &str = include_str!("../../../../opencl/faf_eval.cl");
const TEMPLATE: &str = include_str!("../../../../opencl/getNonBond_generic.cl");

fn make_assembler() -> ClAssembler {
    let mut asm = ClAssembler::new();
    asm.add_fragment("common.cl", COMMON_CL);
    asm.add_fragment("Forces.cl", FORCES_CL);
    asm.add_fragment("nb_common.cl", NB_COMMON_CL);
    asm.add_fragment("gridff_eval.cl", GRIDFF_EVAL_CL);
    asm.add_fragment("faf_eval.cl", FAF_EVAL_CL);
    asm
}

/// Build the NB_VARIANT_DEFINES macro body for a given (pair, excl, surf) combo.
fn variant_defines(pair: &str, excl: &str, surf: &str) -> String {
    format!(
        "#define NB_PAIR_FORCE(dp,REQK,R2damp)  {pair}(dp,REQK,R2damp)\n\
         #define NB_EXCL_ARGS                   NB_EXCL_ARGS_{excl}\n\
         #define NB_EXCL_SETUP(iaa)             NB_EXCL_SETUP_{excl}(iaa)\n\
         #define NB_EXCL_TEST(ja)               NB_EXCL_TEST_{excl}(ja)\n\
         #define NB_EXCL_PBC_TEST(ja,ipbc)      NB_EXCL_PBC_TEST_{excl}(ja,ipbc)\n\
         #define SURF_ARGS                      SURF_ARGS_{surf}\n\
         #define SURF_INJECT(posi,REQKi,fe)     SURF_INJECT_{surf}(posi,REQKi,fe)"
    )
}

fn assemble_variant(pair: &str, excl: &str, surf: &str) -> String {
    let asm = make_assembler();
    let mut subs = Substitutions::new();
    subs.macros.insert(
        "NB_VARIANT_DEFINES".to_string(),
        variant_defines(pair, excl, surf),
    );
    asm.assemble(TEMPLATE, &subs).expect("assembly failed")
}

#[test]
fn assemble_nb_only_ljqh_neighs4() {
    let out = assemble_variant("NB_PAIR_LJQH", "NEIGHS4", "NONE");
    assert!(out.contains("__kernel void getNonBond_generic"), "kernel missing");
    assert!(out.contains("getLJQH"), "LJQH pairwise missing");
    assert!(out.contains("neighs"), "neighs4 exclusion missing");
    assert!(out.contains("NB_EXCL_TEST_NEIGHS4"), "neighs4 test macro missing");
    // SURF_INJECT_NONE expands to empty — the alias is present but the kernel
    // body doesn't call fe3d_pbc_comb/folded_eval_basis. The eval files are
    // always included as utility libraries (inline functions), so we check
    // that the SURF alias maps to NONE, not that the functions are absent.
    assert!(out.contains("#define SURF_INJECT(posi,REQKi,fe)     SURF_INJECT_NONE(posi,REQKi,fe)"), "SURF_INJECT_NONE alias missing");
    // The kernel body should not contain a call to fe3d_pbc_comb inside the
    // getNonBond_generic function (it may appear in the library definitions).
    let kernel_body = out.split("__kernel void getNonBond_generic").nth(1).unwrap_or("");
    assert!(!kernel_body.contains("fe3d_pbc_comb("), "GridFF called in kernel body");
    assert!(!kernel_body.contains("folded_eval_basis("), "FAF called in kernel body");
}

#[test]
fn assemble_nb_gridff_ljqh_neighs4() {
    let out = assemble_variant("NB_PAIR_LJQH", "NEIGHS4", "GRIDFF_BSPLINE");
    assert!(out.contains("__kernel void getNonBond_generic"), "kernel missing");
    assert!(out.contains("getLJQH"), "LJQH pairwise missing");
    assert!(out.contains("fe3d_pbc_comb"), "GridFF Bspline injection missing");
    assert!(out.contains("BsplinePLQ"), "GridFF args missing");
    assert!(out.contains("make_inds_pbc"), "GridFF pbc indices missing");
    assert!(out.contains("SURF_ARGS_GRIDFF_BSPLINE"), "SURF_ARGS alias missing");
}

#[test]
fn assemble_nb_faf_ljqh_neighs4() {
    let out = assemble_variant("NB_PAIR_LJQH", "NEIGHS4", "FAF");
    assert!(out.contains("__kernel void getNonBond_generic"), "kernel missing");
    assert!(out.contains("getLJQH"), "LJQH pairwise missing");
    assert!(out.contains("folded_eval_basis"), "FAF basis injection missing");
    assert!(out.contains("folded_eval_grad"), "FAF grad injection missing");
    assert!(out.contains("folded_coeffs"), "FAF args missing");
    assert!(out.contains("folded_lvec2d"), "FAF lattice args missing");
}

#[test]
fn assemble_all_variants_have_common_helpers() {
    for surf in &["NONE", "GRIDFF_BSPLINE", "FAF"] {
        let out = assemble_variant("NB_PAIR_LJQH", "NEIGHS4", surf);
        assert!(out.contains("mixREQ_arithmetic"), "mixREQ missing for surf={}", surf);
        assert!(out.contains("float4Zero"), "float4Zero missing for surf={}", surf);
        assert!(out.contains("cl_Mat3"), "cl_Mat3 missing for surf={}", surf);
        assert!(out.contains("NB_PAIR_LJQH"), "NB_PAIR_LJQH alias missing for surf={}", surf);
    }
}

#[test]
fn assemble_variant_defines_correct() {
    let defs = variant_defines("NB_PAIR_LJQH", "NEIGHS4", "GRIDFF_BSPLINE");
    assert!(defs.contains("#define NB_PAIR_FORCE(dp,REQK,R2damp)  NB_PAIR_LJQH(dp,REQK,R2damp)"));
    assert!(defs.contains("#define NB_EXCL_ARGS                   NB_EXCL_ARGS_NEIGHS4"));
    assert!(defs.contains("#define SURF_ARGS                      SURF_ARGS_GRIDFF_BSPLINE"));
    assert!(defs.contains("#define SURF_INJECT(posi,REQKi,fe)     SURF_INJECT_GRIDFF_BSPLINE(posi,REQKi,fe)"));
}
