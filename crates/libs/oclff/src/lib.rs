//! OpenCL GPU harness for forcefields: UFF, SPFFsp3, RAFF/RRsp3, GridFF, FAF.
//!
//! CPU references (authoritative for correctness):
//!   - `molff::uff::Uff` for UFF,
//!   - `molff::raff` for RAFF/RRsp3,
//!   - `surfff::SurfaceFolded` for FAF,
//!   - SPAMMM `SPFF_cl.py` / FireCore C++ `MMFFsp3_lib.cpp` for SPFFsp3 (Rust ref TODO).
//!
//! See `opencl/README.md` for kernel conventions (FireCore self-contained vs SPAMMM modular concat).

use ocl::{ProQue, Program, Platform, Device};

/// Build a `ProQue` on the NVIDIA OpenCL platform, using the first GPU device.
/// Fails loud if no NVIDIA platform is present (per AGENTS: never report PoCL/CPU as GPU).
pub(crate) fn nvidia_proque(src: impl Into<String>) -> ocl::Result<ProQue> {
    let platform = Platform::list().into_iter()
        .find(|p| p.vendor().map(|v| v.to_lowercase().contains("nvidia")).unwrap_or(false))
        .ok_or("NVIDIA OpenCL platform not found")?;
    let device = Device::first(&platform)?;
    let mut pb = Program::builder();
    pb.src(src).cmplr_opt("-w").cmplr_opt("-cl-std=CL1.2");
    ProQue::builder().platform(platform).device(device).prog_bldr(pb).build()
}

pub mod assemble;
pub mod pack;
pub mod uff;
pub mod spff;
pub mod rrsp3;
pub mod surfff;

pub use assemble::{ClAssembler, ClLibrary, Substitutions};
pub use pack::{pack_molecules, PackedSystem, build_neighs_from_bonds, make_exclusions_1st_2nd, make_bk_slots_clustered, make_rev_slot_clustered, make_ports_from_neighs};
pub use uff::{UffOcl};
pub use spff::{SpffOcl};
pub use rrsp3::{RRsp3, RRsp3Multi, PortKernel, StepConfig};
pub use surfff::{GridFFBuildOcl, GridFFEvalOcl, FafBuildOcl, FafEvalOcl};
