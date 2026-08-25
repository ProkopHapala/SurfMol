//! spacc — spatial acceleration: AABB, Buckets, grids.
//! See `notes/designs/topology_builder.md` §7 for design.
//! Depends only on `numcore`. Bounds are invalidated by geometry changes — caller must rebuild.

pub mod aabb;
pub mod buckets;
