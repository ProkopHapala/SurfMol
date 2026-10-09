pub mod uff;
pub mod nonbonded;
pub mod rigid_sp3;
pub mod raff;
pub mod raff_reactive;
pub mod rarff2d;
pub mod multigrid;  // TEMP: pre-existing compile errors in multigrid.rs — uncomment when fixed
pub mod ffi_rarff2d;  // ctypes API (cdylib → libmolff.so) — contract: invPPAFM rarff2d_ffi.py
pub mod ffi_raff_reactive;  // ctypes API (cdylib → libmolff.so) — contract: SurfMol scripts/raff3d_ffi.py
pub mod ffi_uff;      // ctypes API (cdylib → libmolff.so) — contract: invPPAFM uff_ffi.py
