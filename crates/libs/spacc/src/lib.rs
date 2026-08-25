//! spacc — spatial acceleration: AABB, Buckets, grids.
//! See `notes/designs/topology_builder.md` §7 for design.
//! Depends only on `numtypes`. Bounds are invalidated by geometry changes — caller must rebuild.

pub mod aabb;
pub mod buckets;
