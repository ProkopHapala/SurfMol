//! numtypes — low-level memory/data-layout vocabulary for SurfMol.
//!
//! Defines the project's common low-level data vocabulary and the tiny intrinsic
//! operations needed to manipulate those values efficiently. Mathematical/domain
//! interpretation is expressed by explicit functions rather than additional wrapper types.
//!
//! ## Boundary rule
//!
//! - **`numtypes`**: data layouts + tiny intrinsic O(1) operations. No algorithms whose
//!   cost scales with dataset size; no numerical approximation/fitting/iteration policy.
//! - **`numcore`**: generic numerical algorithms acting on `numtypes` or slices. Fast
//!   approximations belong here even when inline, because approximation accuracy is an
//!   algorithmic choice.
//! - **`pgraph`**: algorithms whose defining input is connectivity.
//! - **`spacc`**: algorithms whose defining purpose is spatial acceleration.
//!
//! ## `unsafe` policy
//!
//! `unsafe` is allowed here when needed for allocation, alignment, zero-copy views,
//! SIMD/GPU interoperability, etc. Keep unsafe small, documented and tested.

#![allow(clippy::too_many_arguments)]

pub mod vec;
pub mod mat;
pub mod alloc;
pub mod graph;
pub mod spatial;

pub use vec::*;
pub use mat::*;
pub use alloc::AlignedVec;
pub use graph::*;
pub use spatial::*;
