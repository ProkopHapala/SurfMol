//! SPFFsp3 OpenCL compile smoke test.
//!
//! A full CPU Rust SPFF reference does not yet exist in `molff`; once it is
//! ported (from SPAMMM `SPFF_cl.py` / FireCore `MMFFsp3_loc.h`), this test
//! should be extended to dispatch `getSPFFf4` and compare per-atom forces.

use oclff::SpffOcl;

#[test]
fn spff_cl_compiles() {
    let _spff = SpffOcl::new().expect("OpenCL device/context init for SPFF");
}
