//! pgraph — positioned graph: minimal data contract (positions + edges + index containers).
//! See `notes/designs/topology_builder.md` for design rationale.
//!
//! Invariants: dense vertex/edge ids; valid edge endpoints. Atom types, materials, flags,
//! charges, colors are sidecar arrays indexed by the same ids — NOT fields of PGraph.
//! Adjacency is a selectable representation/cache, not the graph identity.

use numcore::math::vec3::Vec3d;

/// Dense vertex/edge id. u32 is sufficient for all SurfMol use cases (molecules < 1M atoms).
pub type Index = u32;

/// Sentinel for empty slots in FixedRows / FixedAdj. Uses i32 because ELLPACK/ELL
/// adjacency is conventionally i32 (GPU-friendly, matches FireCore `neighs[natom]`).
pub const INVALID: i32 = -1;

// =======================================================================================
// Core graph: positions + edges
// =======================================================================================

/// Positioned graph: vertex positions + edge list. The SSOT for connectivity.
/// Higher-order elements (angles, dihedrals, triangles) are sidecars, not fields here.
#[derive(Debug)]
pub struct PGraph {
    pub pos:   Vec<Vec3d>,
    pub edges: Vec<[Index; 2]>,
}

impl PGraph {
    #[inline] pub fn nverts(&self) -> usize { self.pos.len() }
    #[inline] pub fn nedges(&self) -> usize { self.edges.len() }

    /// Borrow as a lightweight view (no allocation).
    pub fn view(&self) -> PGraphView<'_> { PGraphView { pos: &self.pos, edges: &self.edges } }

    /// Validate edge endpoints are in range. Call after construction from untrusted sources.
    pub fn validate(&self) -> Result<(), String> {
        let nv = self.nverts();
        for (e, &[a, b]) in self.edges.iter().enumerate() {
            if (a as usize) >= nv || (b as usize) >= nv {
                return Err(format!("edge {e}: endpoint {a} or {b} >= nverts {nv}"));
            }
        }
        Ok(())
    }
}

/// Borrowed positioned graph — zero-allocation view (safe Rust analogue of SSE CMesh).
pub struct PGraphView<'a> {
    pub pos:   &'a [Vec3d],
    pub edges: &'a [[Index; 2]],
}

// =======================================================================================
// Fixed-size element collections (triangles, angle triples, tetrahedra, dihedrals)
// =======================================================================================

/// Fixed-size element collection: each element is N vertex indices.
/// `Elements<3>` = triangles OR molecular angle triples; `Elements<4>` = tetrahedra OR dihedrals.
/// Share storage and algorithms, not semantics — the owning layer decides meaning.
#[derive(Debug)]
pub struct Elements<const N: usize> {
    pub verts: Vec<[Index; N]>,
}

impl<const N: usize> Elements<N> {
    #[inline] pub fn len(&self) -> usize { self.verts.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.verts.is_empty() }
    pub fn from_vec(v: Vec<[Index; N]>) -> Self { Self { verts: v } }
}

// =======================================================================================
// Ragged: variable-length sets (polygon loops, ring/cycle lists, arbitrary groups)
// =======================================================================================

/// Ragged array: count/offset/packed-items primitive behind polygon loops, ring lists,
/// arbitrary groups. `offsets[i]..offsets[i+1]` slices into `items`.
#[derive(Debug)]
pub struct Ragged {
    pub offsets: Vec<Index>,  // len = ngroups + 1, offsets[0] = 0
    pub items:   Vec<Index>,
}

impl Ragged {
    /// Build from per-group item counts. Allocates offsets and items, items left zeroed.
    pub fn from_counts(counts: &[Index]) -> Self {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0);
        let mut total = 0;
        for &c in counts { total += c; offsets.push(total); }
        Self { offsets, items: vec![0; total as usize] }
    }

    #[inline] pub fn ngroups(&self) -> usize { self.offsets.len() - 1 }
    #[inline] pub fn group_range(&self, g: usize) -> (usize, usize) {
        (self.offsets[g] as usize, self.offsets[g + 1] as usize)
    }
    #[inline] pub fn group(&self, g: usize) -> &[Index] {
        let (i0, i1) = self.group_range(g);
        &self.items[i0..i1]
    }
    #[inline] pub fn group_mut(&mut self, g: usize) -> &mut [Index] {
        let (i0, i1) = self.group_range(g);
        &mut self.items[i0..i1]
    }
}

// =======================================================================================
// Permutation: bidirectional index remapping
// =======================================================================================

/// Bidirectional permutation for compaction/reordering.
/// `old2new[old] = new`; `new2old[new] = old`.
#[derive(Debug)]
pub struct Permutation {
    pub old2new: Vec<Index>,
    pub new2old: Vec<Index>,
}

impl Permutation {
    /// Identity permutation of length n.
    pub fn identity(n: usize) -> Self {
        let old2new: Vec<Index> = (0..n as u32).collect();
        let new2old = old2new.clone();
        Self { old2new, new2old }
    }

    /// Build old2new from a new2old mapping (e.g. from sorting or group packing).
    pub fn from_new2old(new2old: Vec<Index>) -> Self {
        let n = new2old.len();
        let mut old2new = vec![0; n];
        for (new_i, &old_i) in new2old.iter().enumerate() {
            old2new[old_i as usize] = new_i as Index;
        }
        Self { old2new, new2old }
    }

    #[inline] pub fn len(&self) -> usize { self.old2new.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.old2new.is_empty() }
}

// =======================================================================================
// Fixed-stride padded adjacency (ELLPACK/ELL-like)
// =======================================================================================

/// Fixed-stride rows with -1 sentinel for empty slots. Valid entries packed first.
/// `FixedRows<4>` is the layout behind FireCore `neighs[natom]` (Quat4i).
#[derive(Debug)]
pub struct FixedRows<const K: usize> {
    pub data: Vec<[i32; K]>,
}

impl<const K: usize> FixedRows<K> {
    /// All slots initialized to INVALID (-1).
    pub fn new(nrows: usize) -> Self {
        Self { data: vec!([INVALID; K]; nrows) }
    }

    #[inline] pub fn nrows(&self) -> usize { self.data.len() }
    #[inline] pub fn row(&self, r: usize) -> &[i32; K] { &self.data[r] }
    #[inline] pub fn row_mut(&mut self, r: usize) -> &mut [i32; K] { &mut self.data[r] }

    /// Append a value to row r in the first INVALID slot. Returns slot index or error if row full.
    pub fn push(&mut self, r: usize, val: i32) -> Result<usize, String> {
        for s in 0..K {
            if self.data[r][s] == INVALID {
                self.data[r][s] = val;
                return Ok(s);
            }
        }
        Err(format!("FixedRows<{K}>::push: row {r} full (degree >= {K})"))
    }

    /// Count valid (non-INVALID) entries in row r.
    #[inline] pub fn degree(&self, r: usize) -> usize {
        self.data[r].iter().take_while(|&&v| v != INVALID).count()
    }
}

/// Fixed-stride adjacency: neighbor vertex ids + corresponding edge ids, both ELL-like.
/// Keeping both tables is intentional — a hot kernel should not search again for the bond id.
/// Builder must fail loud on degree > K; never truncate.
#[derive(Debug)]
pub struct FixedAdj<const K: usize> {
    pub neigh: FixedRows<K>,  // neighbor vertex/atom index per slot
    pub edge:  FixedRows<K>,  // corresponding edge/bond index per slot
}

impl<const K: usize> FixedAdj<K> {
    pub fn new(nverts: usize) -> Self {
        Self { neigh: FixedRows::new(nverts), edge: FixedRows::new(nverts) }
    }

    #[inline] pub fn nverts(&self) -> usize { self.neigh.nrows() }
    #[inline] pub fn degree(&self, v: usize) -> usize { self.neigh.degree(v) }
}

// =======================================================================================
// CSR adjacency (compact, arbitrary degree)
// =======================================================================================

/// Compact CSR adjacency for arbitrary degree. Handles graphs that FixedAdj<K> cannot.
/// Built by count → prefix → scatter (same pattern as FireCore MolecularGraph::makeNeighbors).
#[derive(Debug)]
pub struct CsrAdj {
    pub offsets: Vec<Index>,  // nvert + 1, offsets[0] = 0
    pub neigh:   Vec<Index>,  // 2*nedges for undirected graph
    pub edge:    Vec<Index>,  // matching edge ids, parallel to neigh
}

impl CsrAdj {
    #[inline] pub fn nverts(&self) -> usize { self.offsets.len() - 1 }
    #[inline] pub fn degree(&self, v: usize) -> usize {
        (self.offsets[v + 1] - self.offsets[v]) as usize
    }
    #[inline] pub fn neighbors(&self, v: usize) -> (&[Index], &[Index]) {
        let i0 = self.offsets[v] as usize;
        let i1 = self.offsets[v + 1] as usize;
        (&self.neigh[i0..i1], &self.edge[i0..i1])
    }
}

// =======================================================================================
// Groups: partition + packed representations
// =======================================================================================

/// Disjoint partition: item → group id (-1 unassigned).
/// Flexible during editing; convert to IndexGroups or RangeGroups for packed access.
#[derive(Debug)]
pub struct Partition {
    pub item_group: Vec<i32>,  // -1 = unassigned
}

impl Partition {
    pub fn new(nitems: usize) -> Self { Self { item_group: vec![-1; nitems] } }
    #[inline] pub fn nitems(&self) -> usize { self.item_group.len() }
    #[inline] pub fn group_of(&self, item: usize) -> i32 { self.item_group[item] }
    #[inline] pub fn assign(&mut self, item: usize, group: i32) { self.item_group[item] = group; }

    /// Number of distinct groups (excluding -1).
    pub fn ngroups(&self) -> usize {
        let max_g = self.item_group.iter().copied().filter(|&g| g >= 0).max().unwrap_or(-1);
        (max_g + 1) as usize
    }
}

/// Packed index groups: offsets + items (same layout as Ragged but for group membership).
/// `group(g) = items[offsets[g]..offsets[g+1]]`.
#[derive(Debug)]
pub struct IndexGroups {
    pub offsets: Vec<Index>,
    pub items:   Vec<Index>,
}

impl IndexGroups {
    #[inline] pub fn ngroups(&self) -> usize { self.offsets.len() - 1 }
    #[inline] pub fn group(&self, g: usize) -> &[Index] {
        let i0 = self.offsets[g] as usize;
        let i1 = self.offsets[g + 1] as usize;
        &self.items[i0..i1]
    }
}

/// Contiguous range groups: each group is a [start, end) range after packing.
/// Produced by group-aware permutation that reorders items so each group is contiguous.
#[derive(Debug)]
pub struct RangeGroups {
    pub ranges: Vec<[Index; 2]>,  // [start, end) — end exclusive
}

impl RangeGroups {
    #[inline] pub fn ngroups(&self) -> usize { self.ranges.len() }
    #[inline] pub fn group(&self, g: usize) -> (Index, Index) {
        let r = self.ranges[g];
        (r[0], r[1])
    }
    #[inline] pub fn group_len(&self, g: usize) -> usize {
        (self.ranges[g][1] - self.ranges[g][0]) as usize
    }
}
