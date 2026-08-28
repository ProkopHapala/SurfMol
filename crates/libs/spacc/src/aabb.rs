//! Axis-aligned bounding boxes. Rebuildable cache — invalidated by geometry changes.
//!
//! AABB is `numtypes::Aabb3d` = `Vec6d` (`a` = lo, `b` = hi). Intrinsic operations live in
//! `numtypes::spatial` (`aabb_include`, `aabb_contains`, ...). This module provides the
//! algorithms that build/fit AABBs over datasets.

use numtypes::{Aabb3d, Index, RaggedIndex, Vec3d, aabb_empty, aabb_include, aabb_overlap_margin, aabb_is_valid};

/// Fit an AABB to a set of positions selected by `ids`.
pub fn fit_aabb(pos: &[Vec3d], ids: &[Index]) -> Aabb3d {
    assert!(!ids.is_empty(), "fit_aabb: empty id list");
    let mut bb = aabb_empty();
    for &id in ids { aabb_include(&mut bb, pos[id as usize]); }
    bb
}

/// Fit AABBs for multiple groups defined by `RaggedIndex` (offsets + items).
/// Writes to `out` which must have length >= groups.ngroups().
pub fn fit_group_aabbs(pos: &[Vec3d], groups: &RaggedIndex, out: &mut [Aabb3d]) {
    let ng = groups.ngroups();
    assert!(out.len() >= ng, "fit_group_aabbs: out.len() {} < ngroups {}", out.len(), ng);
    for g in 0..ng {
        let items = groups.group(g);
        if items.is_empty() { out[g] = aabb_empty(); }
        else                { out[g] = fit_aabb(pos, items); }
    }
}

/// Fit AABBs for contiguous ranges [i0, i1) per group. Cache-optimal for packed fragments.
pub fn fit_range_aabbs(pos: &[Vec3d], ranges: &[[Index; 2]], out: &mut [Aabb3d]) {
    assert!(out.len() >= ranges.len(), "fit_range_aabbs: out.len() {} < ranges {}", out.len(), ranges.len());
    for (g, &[i0, i1]) in ranges.iter().enumerate() {
        assert!((i1 as usize) <= pos.len(), "fit_range_aabbs: range [{i0},{i1}) exceeds pos.len() {}", pos.len());
        let mut bb = aabb_empty();
        for p in &pos[i0 as usize..i1 as usize] { aabb_include(&mut bb, *p); }
        out[g] = bb;
    }
}

/// Broad-phase collision: find all overlapping cluster-pair indices `(i, j)` with `i < j`.
/// Each cluster has an AABB; two clusters "overlap" when their AABBs overlap after expanding
/// by `margin` on each side. Returns sorted pairs for deterministic iteration order.
/// Mirrors FireCore `NBFF.h:evalSortRange_BBs` bucket-pair loop (CPU O(N²) over clusters).
pub fn broad_phase_pairs(cluster_aabbs: &[Aabb3d], margin: f64) -> Vec<(u32, u32)> {
    let n = cluster_aabbs.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        assert!(aabb_is_valid(cluster_aabbs[i]), "broad_phase_pairs: cluster {} has invalid AABB", i);
        for j in (i+1)..n {
            if aabb_overlap_margin(cluster_aabbs[i], cluster_aabbs[j], margin) {
                pairs.push((i as u32, j as u32));
            }
        }
    }
    pairs
}

/// Generate the 12 edge segments of an AABB as `(p0, p1)` pairs for line rendering.
/// Mirrors FireCore `VispyUtils.py:900-926` bbox edge drawing.
pub fn aabb_edges(bb: Aabb3d) -> [[f32; 3]; 24] {
    let mn = bb.a; let mx = bb.b;
    // 8 corners
    let c = [
        [mn.x as f32, mn.y as f32, mn.z as f32], // 0
        [mx.x as f32, mn.y as f32, mn.z as f32], // 1
        [mx.x as f32, mx.y as f32, mn.z as f32], // 2
        [mn.x as f32, mx.y as f32, mn.z as f32], // 3
        [mn.x as f32, mn.y as f32, mx.z as f32], // 4
        [mx.x as f32, mn.y as f32, mx.z as f32], // 5
        [mx.x as f32, mx.y as f32, mx.z as f32], // 6
        [mn.x as f32, mx.y as f32, mx.z as f32], // 7
    ];
    // 12 edges as pairs of corner indices
    let e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)];
    let mut out = [[0.0f32; 3]; 24];
    for (k, &(a, b)) in e.iter().enumerate() {
        out[k*2] = c[a];
        out[k*2+1] = c[b];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtypes::aabb_contains;

    #[test]
    fn test_fit_aabb() {
        let pos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(2.0, 0.0, 0.0), Vec3d::new(0.0, 3.0, 0.0)];
        let ids = vec![0, 1, 2];
        let bb = fit_aabb(&pos, &ids);
        assert!((bb.a.x - 0.0).abs() < 1e-12);
        assert!((bb.b.x - 2.0).abs() < 1e-12);
        assert!((bb.b.y - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_fit_group_aabbs() {
        let pos = vec![
            Vec3d::new(0.0, 0.0, 0.0),  // group 0
            Vec3d::new(1.0, 0.0, 0.0),  // group 0
            Vec3d::new(10.0, 10.0, 0.0), // group 1
            Vec3d::new(12.0, 10.0, 0.0), // group 1
        ];
        let groups = RaggedIndex { offsets: vec![0, 2, 4], items: vec![0, 1, 2, 3] };
        let mut out = vec![aabb_empty(); 2];
        fit_group_aabbs(&pos, &groups, &mut out);
        assert!((out[0].b.x - 1.0).abs() < 1e-12, "group 0 max x: {}", out[0].b.x);
        assert!((out[1].a.x - 10.0).abs() < 1e-12, "group 1 min x: {}", out[1].a.x);
        assert!((out[1].b.x - 12.0).abs() < 1e-12, "group 1 max x: {}", out[1].b.x);
    }

    #[test]
    fn test_fit_range_aabbs() {
        let pos = vec![
            Vec3d::new(0.0, 0.0, 0.0),
            Vec3d::new(1.0, 0.0, 0.0),
            Vec3d::new(10.0, 10.0, 0.0),
            Vec3d::new(12.0, 10.0, 0.0),
        ];
        let ranges: Vec<[Index; 2]> = vec![[0, 2], [2, 4]];
        let mut out = vec![aabb_empty(); 2];
        fit_range_aabbs(&pos, &ranges, &mut out);
        assert!((out[0].b.x - 1.0).abs() < 1e-12);
        assert!((out[1].b.x - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_contains() {
        let mut bb = aabb_empty();
        aabb_include(&mut bb, Vec3d::new(0.0, 0.0, 0.0));
        aabb_include(&mut bb, Vec3d::new(2.0, 2.0, 2.0));
        assert!(aabb_contains(bb, Vec3d::new(1.0, 1.0, 1.0)));
        assert!(aabb_contains(bb, Vec3d::new(0.0, 0.0, 0.0)));
        assert!(aabb_contains(bb, Vec3d::new(2.0, 2.0, 2.0)));
        assert!(!aabb_contains(bb, Vec3d::new(3.0, 1.0, 1.0)));
    }

    #[test]
    fn test_broad_phase_pairs() {
        // 3 clusters: 0 and 1 overlap (margin 0.5), 2 is far away
        let aabbs = vec![
            Aabb3d::new(Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(2.0, 2.0, 2.0)),   // cluster 0
            Aabb3d::new(Vec3d::new(2.5, 0.0, 0.0), Vec3d::new(4.5, 2.0, 2.0)),   // cluster 1 (gap=0.5 in x)
            Aabb3d::new(Vec3d::new(20.0, 20.0, 0.0), Vec3d::new(22.0, 22.0, 2.0)), // cluster 2 (far)
        ];
        // margin 0.0: no overlap (gap = 0.5)
        let pairs0 = broad_phase_pairs(&aabbs, 0.0);
        assert!(pairs0.is_empty(), "margin 0.0 should find no pairs, got {:?}", pairs0);
        // margin 0.5: touching → overlap (0,1)
        let pairs5 = broad_phase_pairs(&aabbs, 0.5);
        assert_eq!(pairs5, vec![(0, 1)], "margin 0.5 should find pair (0,1), got {:?}", pairs5);
        // margin 2.0: still only (0,1), cluster 2 is far
        let pairs2 = broad_phase_pairs(&aabbs, 2.0);
        assert_eq!(pairs2, vec![(0, 1)], "margin 2.0 should find only (0,1), got {:?}", pairs2);
    }
}
