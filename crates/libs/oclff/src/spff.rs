//! OpenCL GPU harness for SPFFsp3 (SPAMMM `SPFF.cl`) = FireCore MMFFsp3 GPU forcefield.
//!
//! SPFF.cl is **not** self-contained; the program source must concatenate
//! `common.cl` first, then `Forces.cl`, then `SPFF.cl` (this matches the SPAMMM
//! Python `SPFF_cl.py` kernel list for bonded-only work). Non-bonded also needs
//! `nonbonded.cl` + `gridFF.cl`/`surface.cl`.
//!
//! CPU reference (authoritative for correctness): none in Rust yet — verify
//! against SPAMMM `SPFF_cl.py` or FireCore C++ `MMFFsp3_lib.cpp` for now.

use ocl::{ProQue, Kernel};

const SPFF_SRC: &str = concat!(
    include_str!("../../../../opencl/common.cl"),
    include_str!("../../../../opencl/Forces.cl"),
    include_str!("../../../../opencl/SPFF.cl"),
);

/// GPU SPFFsp3 harness. Wraps a `ProQue` compiled from `common.cl` + `Forces.cl` + `SPFF.cl`.
pub struct SpffOcl { pro_que: ProQue }

impl SpffOcl {
    /// Compile the concatenated SPFFsp3 source on the first available NVIDIA GPU.
    pub fn new() -> ocl::Result<Self> {
        Ok(Self { pro_que: crate::nvidia_proque(SPFF_SRC)? })
    }

    pub fn pro_que(&self) -> &ProQue { &self.pro_que }

    /// Builder for a named SPFF kernel (`getSPFFf4`, `updateAtomsSPFFf4`, `cleanForceSPFFf4`, ...).
    /// Caller supplies all `arg(...)` in the exact order declared in `SPFF.cl`.
    pub fn kernel(&self, name: &str) -> ocl::Result<Kernel> {
        self.pro_que.kernel_builder(name).build()
    }
}
