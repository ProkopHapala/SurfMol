//! Axis-aligned bounding boxes. Rebuildable cache — invalidated by geometry changes.
//!
//! AABB is `numtypes::Aabb3d` = `Vec6d` (`a` = lo, `b` = hi). Intrinsic operations live in
//! `numtypes::spatial` (`aabb_include`, `aabb_contains`, ...). This module provides the
//! algorithms that build/fit AABBs over datasets.

use numtypes::{Aabb3d, Index, RaggedIndex, Vec3d, aabb_empty, aabb_include};

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
}
