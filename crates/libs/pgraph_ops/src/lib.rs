//! pgraph_ops — reusable graph algorithms: adjacency, components, bridges, reorder, geometry.
//! See `notes/designs/topology_builder.md` §10 for design.
//!
//! Algorithms accept slices / `PGraphView` / `CsrAdj` — no ownership of graph data.
//! Scratch is local; allocation-free overloads can take caller buffers when needed.

pub mod adjacency;
pub mod components;
pub mod bridges;
pub mod reorder;
pub mod geometry;
