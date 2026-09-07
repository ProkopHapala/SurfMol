//! OpenCL GPU harness for GridFF and Folded Atomic Forcefield (FAF).
//!
//! Uses the SurfMol OpenCL macro assembler (`assemble.rs`) — a Rust port of
//! SPAMMM/FireCore `OpenCLBase.preprocess_opencl_source` — to compose build and
//! evaluator programs from `//<<<-marked` templates and `//>>>`-fragment libraries.
//!
//! CPU references (authoritative for correctness):
//!   - `surfff::SurfaceFolded` for the folded-basis evaluator,
//!   - FireCore `cpp/common/molecular/GridFF.h` / SPAMMM `surfaces/GridFF.py` for GridFF.
//!
//! See `opencl/README.md` for kernel conventions and
//! `notes/designs/2026-08-29_gridff_faf_porting_notes.md` for equations, memory
//! layout, strides, boundary conditions, and basis-set hierarchy.

use ocl::{ProQue, Kernel};
use crate::assemble::{ClAssembler, Substitutions};

// OpenCL fragment library. All paths are relative to the workspace `opencl/`.
fn make_assembler() -> ClAssembler {
    let mut asm = ClAssembler::new();
    asm.add_fragment("common.cl",       include_str!("../../../../opencl/common.cl"));
    asm.add_fragment("Forces.cl",       include_str!("../../../../opencl/Forces.cl"));
    asm
}

fn assemble_program(template: &str) -> ocl::Result<String> {
    let asm = make_assembler();
    let subs = Substitutions::new();
    let src = asm.assemble(template, &subs)
        .map_err(|e| ocl::Error::from(format!("OpenCL assembly failed: {e}")))?;
    Ok(src)
}

/// GPU GridFF builder (from `opencl/gridff_build.cl`).
pub struct GridFFBuildOcl { pro_que: ProQue }

impl GridFFBuildOcl {
    const TEMPLATE: &str = include_str!("../../../../opencl/gridff_build.cl");

    /// Assemble and compile the GridFF builder program.
    pub fn new() -> ocl::Result<Self> {
        let src = assemble_program(Self::TEMPLATE)?;
        Ok(Self { pro_que: crate::nvidia_proque(src)? })
    }

    pub fn pro_que(&self) -> &ProQue { &self.pro_que }

    pub fn kernel(&self, name: &str) -> ocl::Result<Kernel> {
        self.pro_que.kernel_builder(name).build()
    }
}

/// GPU GridFF evaluator (from `opencl/gridff_eval.cl`).
pub struct GridFFEvalOcl { pro_que: ProQue }

impl GridFFEvalOcl {
    const TEMPLATE: &str = include_str!("../../../../opencl/gridff_eval.cl");

    pub fn new() -> ocl::Result<Self> {
        let src = assemble_program(Self::TEMPLATE)?;
        Ok(Self { pro_que: crate::nvidia_proque(src)? })
    }

    pub fn pro_que(&self) -> &ProQue { &self.pro_que }

    pub fn kernel(&self, name: &str) -> ocl::Result<Kernel> {
        self.pro_que.kernel_builder(name).build()
    }
}

/// GPU FAF builder (from `opencl/faf_build.cl`).
pub struct FafBuildOcl { pro_que: ProQue }

impl FafBuildOcl {
    const TEMPLATE: &str = include_str!("../../../../opencl/faf_build.cl");

    pub fn new() -> ocl::Result<Self> {
        let src = assemble_program(Self::TEMPLATE)?;
        Ok(Self { pro_que: crate::nvidia_proque(src)? })
    }

    pub fn pro_que(&self) -> &ProQue { &self.pro_que }

    pub fn kernel(&self, name: &str) -> ocl::Result<Kernel> {
        self.pro_que.kernel_builder(name).build()
    }
}

/// GPU FAF evaluator (from `opencl/faf_eval.cl`).
pub struct FafEvalOcl { pro_que: ProQue }

impl FafEvalOcl {
    const TEMPLATE: &str = include_str!("../../../../opencl/faf_eval.cl");

    pub fn new() -> ocl::Result<Self> {
        let src = assemble_program(Self::TEMPLATE)?;
        Ok(Self { pro_que: crate::nvidia_proque(src)? })
    }

    pub fn pro_que(&self) -> &ProQue { &self.pro_que }

    pub fn kernel(&self, name: &str) -> ocl::Result<Kernel> {
        self.pro_que.kernel_builder(name).build()
    }
}
