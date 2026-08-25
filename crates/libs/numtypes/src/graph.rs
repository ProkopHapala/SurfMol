//! Positioned graph + generic index containers. Data layouts only — builders live in `pgraph`.

use crate::alloc::AlignedVec;
use crate::vec::Vec3d;

/// Dense vertex/edge id. u32 is sufficient for all SurfMol use cases (molecules < 1M atoms).
pub type Index = u32;

/// Sentinel for empty slots in FixedRows / FixedAdj. i32 because ELLPACK/ELL adjacency
/// conventionally uses i32 and matches FireCore `neighs[natom]`.
pub const INVALID: i32 = -1;

// =======================================================================================
// Core graph: positions + edges
// =======================================================================================

/// Positioned graph: vertex positions + edge list. SSOT for connectivity.
#[derive(Debug)]
pub struct PGraph {
    pub pos: Vec<Vec3d>,
    pub edges: Vec<[Index; 2]>,
}

impl PGraph {
    #[inline(always)] pub fn nverts(&self) -> usize { self.pos.len() }
    #[inline(always)] pub fn nedges(&self) -> usize { self.edges.len() }
    #[inline(always)] pub fn view(&self) -> PGraphView<'_> { PGraphView { pos: &self.pos, edges: &self.edges } }

    /// Validate edge endpoints are in range and positions are finite. Fails loud on first problem.
    pub fn validate(&self) -> Result<(), String> {
        let nv = self.nverts();
        for (e, &[a, b]) in self.edges.iter().enumerate() {
            if (a as usize) >= nv || (b as usize) >= nv {
                return Err(format!("PGraph::validate: edge {e}: endpoint {a} or {b} >= nverts {nv}"));
            }
        }
        for (v, p) in self.pos.iter().enumerate() {
            if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
                return Err(format!("PGraph::validate: vertex {v}: position not finite: {p:?}"));
            }
        }
        Ok(())
    }
}

/// Borrowed positioned graph — zero-allocation view.
#[derive(Debug)]
pub struct PGraphView<'a> {
    pub pos: &'a [Vec3d],
    pub edges: &'a [[Index; 2]],
}

impl<'a> PGraphView<'a> {
    #[inline(always)] pub fn nverts(&self) -> usize { self.pos.len() }
    #[inline(always)] pub fn nedges(&self) -> usize { self.edges.len() }
}

// =======================================================================================
// Fixed-size element collections (triangles, angles, tetrahedra, dihedrals)
// =======================================================================================

/// Fixed-size element collection: each element is N vertex indices.
/// `Elements<3>` = triangles OR angle triples; `Elements<4>` = tetrahedra OR dihedrals.
/// Semantics live in the owning layer.
#[derive(Debug)]
pub struct Elements<const N: usize> {
    pub verts: Vec<[Index; N]>,
}

impl<const N: usize> Elements<N> {
    #[inline(always)] pub fn new() -> Self { Self { verts: Vec::new() } }
    #[inline(always)] pub fn from_vec(v: Vec<[Index; N]>) -> Self { Self { verts: v } }
    #[inline(always)] pub fn len(&self) -> usize { self.verts.len() }
    #[inline(always)] pub fn is_empty(&self) -> bool { self.verts.is_empty() }
}

// =======================================================================================
// Ragged: variable-length sets (polygon loops, rings, arbitrary groups)
// Replaces the previous duplicated `Ragged` and `IndexGroups` with a single primitive.
// =======================================================================================

/// Ragged index array: count/offset/packed-items primitive.
/// `group(g) = items[offsets[g]..offsets[g+1]]`.
#[derive(Debug)]
pub struct RaggedIndex {
    pub offsets: Vec<Index>,  // len = ngroups + 1, offsets[0] = 0
    pub items: Vec<Index>,
}

impl RaggedIndex {
    #[inline(always)] pub fn new() -> Self { Self { offsets: vec![0], items: Vec::new() } }
    #[inline(always)] pub fn ngroups(&self) -> usize { self.offsets.len().saturating_sub(1) }
    #[inline(always)] pub fn group_range(&self, g: usize) -> (usize, usize) { (self.offsets[g] as usize, self.offsets[g + 1] as usize) }
    #[inline(always)] pub fn group(&self, g: usize) -> &[Index] { let (i0, i1) = self.group_range(g); &self.items[i0..i1] }
    #[inline(always)] pub fn group_mut(&mut self, g: usize) -> &mut [Index] { let (i0, i1) = self.group_range(g); &mut self.items[i0..i1] }
    #[inline(always)] pub fn len(&self) -> usize { self.items.len() }
    #[inline(always)] pub fn is_empty(&self) -> bool { self.items.is_empty() }

    /// Build from per-group item counts. Offsets and items allocated; items zeroed.
    pub fn from_counts(counts: &[Index]) -> Self {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0);
        let mut total = 0u32;
        for &c in counts { total += c; offsets.push(total); }
        Self { offsets, items: vec![0; total as usize] }
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

    /// Build old2new from a new2old mapping. Fails loud on duplicate or missing old indices.
    pub fn from_new2old(new2old: Vec<Index>) -> Self {
        let n = new2old.len();
        let mut old2new = vec![0u32; n];
        for (new_i, &old_i) in new2old.iter().enumerate() {
            if (old_i as usize) >= n { panic!("Permutation::from_new2old: old index {old_i} out of range [0,{n})"); }
            old2new[old_i as usize] = new_i as Index;
        }
        // Verify every old index appears exactly once
        let mut seen = vec![false; n];
        for &old in &new2old {
            if seen[old as usize] { panic!("Permutation::from_new2old: duplicate old index {old}"); }
            seen[old as usize] = true;
        }
        for (i, s) in seen.iter().enumerate() { assert!(*s, "Permutation::from_new2old: missing old index {i}"); }
        Self { old2new, new2old }
    }

    #[inline(always)] pub fn len(&self) -> usize { self.old2new.len() }
    #[inline(always)] pub fn is_empty(&self) -> bool { self.old2new.is_empty() }
}

// =======================================================================================
// Fixed-stride padded adjacency (ELLPACK/ELL-like), 64-byte aligned.
// =======================================================================================

/// Fixed-stride rows with -1 sentinel for empty slots. Valid entries packed first.
/// `FixedRows<4>` is the layout behind FireCore `neighs[natom]` (Quat4i).
#[derive(Debug)]
pub struct FixedRows<const K: usize> {
    pub data: AlignedVec<[i32; K], 64>,
}

impl<const K: usize> FixedRows<K> {
    /// All slots initialized to INVALID (-1).
    pub fn new(nrows: usize) -> Self {
        Self { data: AlignedVec::<[i32; K], 64>::with_len_fill(nrows, [INVALID; K]) }
    }

    #[inline(always)] pub fn nrows(&self) -> usize { self.data.len() }
    #[inline(always)] pub fn row(&self, r: usize) -> &[i32; K] { &self.data[r] }
    #[inline(always)] pub fn row_mut(&mut self, r: usize) -> &mut [i32; K] { &mut self.data[r] }

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
    #[inline(always)] pub fn degree(&self, r: usize) -> usize {
        self.data[r].iter().take_while(|&&v| v != INVALID).count()
    }
}

/// Fixed-stride adjacency: neighbor vertex ids + corresponding edge ids, both ELL-like.
#[derive(Debug)]
pub struct FixedAdj<const K: usize> {
    pub neigh: FixedRows<K>,
    pub edge:  FixedRows<K>,
}

impl<const K: usize> FixedAdj<K> {
    pub fn new(nverts: usize) -> Self {
        Self { neigh: FixedRows::new(nverts), edge: FixedRows::new(nverts) }
    }
    #[inline(always)] pub fn nverts(&self) -> usize { self.neigh.nrows() }
    #[inline(always)] pub fn degree(&self, v: usize) -> usize { self.neigh.degree(v) }
}

// =======================================================================================
// CSR adjacency (compact, arbitrary degree)
// =======================================================================================

/// Compact CSR adjacency for arbitrary degree.
#[derive(Debug)]
pub struct CsrAdj {
    pub offsets: Vec<Index>,  // nvert + 1, offsets[0] = 0
    pub neigh:   Vec<Index>,  // 2*nedges for undirected graph
    pub edge:    Vec<Index>,  // matching edge ids, parallel to neigh
}

impl CsrAdj {
    #[inline(always)] pub fn nverts(&self) -> usize { self.offsets.len().saturating_sub(1) }
    #[inline(always)] pub fn degree(&self, v: usize) -> usize { (self.offsets[v + 1] - self.offsets[v]) as usize }
    #[inline(always)] pub fn neighbors(&self, v: usize) -> (&[Index], &[Index]) {
        let i0 = self.offsets[v] as usize;
        let i1 = self.offsets[v + 1] as usize;
        (&self.neigh[i0..i1], &self.edge[i0..i1])
    }
}

// =======================================================================================
// Groups: partition + packed representations
// =======================================================================================

/// Disjoint partition: item → group id (-1 unassigned).
#[derive(Debug)]
pub struct Partition {
    pub item_group: Vec<i32>,
}

impl Partition {
    #[inline(always)] pub fn new(nitems: usize) -> Self { Self { item_group: vec![-1; nitems] } }
    #[inline(always)] pub fn nitems(&self) -> usize { self.item_group.len() }
    #[inline(always)] pub fn group_of(&self, item: usize) -> i32 { self.item_group[item] }
    #[inline(always)] pub fn assign(&mut self, item: usize, group: i32) { self.item_group[item] = group; }
    /// Number of distinct groups (excluding -1).
    pub fn ngroups(&self) -> usize {
        let max_g = self.item_group.iter().copied().filter(|&g| g >= 0).max().unwrap_or(-1);
        (max_g + 1) as usize
    }
}

/// Contiguous range groups: each group is a [start, end) range after packing.
#[derive(Debug)]
pub struct RangeGroups {
    pub ranges: Vec<[Index; 2]>,
}

impl RangeGroups {
    #[inline(always)] pub fn new() -> Self { Self { ranges: Vec::new() } }
    #[inline(always)] pub fn ngroups(&self) -> usize { self.ranges.len() }
    #[inline(always)] pub fn group(&self, g: usize) -> (Index, Index) { let r = self.ranges[g]; (r[0], r[1]) }
    #[inline(always)] pub fn group_len(&self, g: usize) -> usize { (self.ranges[g][1] - self.ranges[g][0]) as usize }
    #[inline(always)] pub fn add(&mut self, start: Index, end: Index) { self.ranges.push([start, end]); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_index_from_counts() {
        let r = RaggedIndex::from_counts(&[1, 2, 0]);
        assert_eq!(r.ngroups(), 3);
        assert_eq!(r.group(0), &[0]);     // zeroed items for now
        assert_eq!(r.group(1), &[0, 0]);
        assert_eq!(r.group(2), &[]);
    }

    #[test]
    fn fixed_rows_push_and_degree() {
        let mut fr = FixedRows::<4>::new(2);
        fr.push(0, 1).unwrap();
        fr.push(0, 3).unwrap();
        assert_eq!(fr.degree(0), 2);
        assert_eq!(fr.row(0)[2], INVALID);
        assert_eq!(fr.row(0)[3], INVALID);
        fr.push(0, 5).unwrap();
        fr.push(0, 7).unwrap();
        assert!(fr.push(0, 9).is_err());
    }

    #[test]
    fn permutation_validates() {
        let p = Permutation::from_new2old(vec![2, 0, 1]);
        assert_eq!(p.old2new, vec![1, 2, 0]);
    }

    #[test]
    #[should_panic]
    fn permutation_detects_duplicate() {
        Permutation::from_new2old(vec![0, 0, 2]);
    }

    #[test]
    fn pgraph_validate_finite() {
        let g = PGraph { pos: vec![Vec3d::new(0.0, 1.0, 2.0)], edges: vec![] };
        assert!(g.validate().is_ok());
        let g_bad = PGraph { pos: vec![Vec3d::new(0.0, f64::NAN, 0.0)], edges: vec![] };
        assert!(g_bad.validate().is_err());
    }
}
