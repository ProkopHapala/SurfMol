//! Axis-aligned bounding boxes. Rebuildable cache — invalidated by geometry changes.

use numcore::math::vec3::Vec3d;
use pgraph::Index;

/// Axis-aligned bounding box.
#[derive(Clone, Debug)]
pub struct Aabb {
    pub lo: Vec3d,
    pub hi: Vec3d,
}

impl Aabb {
    pub fn empty() -> Self { Self { lo: Vec3d::new(f64::INFINITY, f64::INFINITY, f64::INFINITY), hi: Vec3d::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY) } }

    /// Expand to include a point.
    #[inline(always)]
    pub fn include_point(&mut self, p: Vec3d) {
        if p.x < self.lo.x { self.lo.x = p.x; }
        if p.x > self.hi.x { self.hi.x = p.x; }
        if p.y < self.lo.y { self.lo.y = p.y; }
        if p.y > self.hi.y { self.hi.y = p.y; }
        if p.z < self.lo.z { self.lo.z = p.z; }
        if p.z > self.hi.z { self.hi.z = p.z; }
    }

    /// Expand to include another AABB.
    #[inline(always)]
    pub fn include_aabb(&mut self, other: &Aabb) {
        self.include_point(other.lo);
        self.include_point(other.hi);
    }

    #[inline] pub fn center(&self) -> Vec3d {
        Vec3d::new((self.lo.x + self.hi.x) * 0.5, (self.lo.y + self.hi.y) * 0.5, (self.lo.z + self.hi.z) * 0.5)
    }

    #[inline] pub fn size(&self) -> Vec3d {
        Vec3d::new(self.hi.x - self.lo.x, self.hi.y - self.lo.y, self.hi.z - self.lo.z)
    }

    #[inline] pub fn max_extent(&self) -> f64 {
        let s = self.size();
        s.x.max(s.y).max(s.z)
    }

    /// Test if a point is inside the AABB (inclusive bounds).
    #[inline] pub fn contains(&self, p: Vec3d) -> bool {
        p.x >= self.lo.x && p.x <= self.hi.x &&
        p.y >= self.lo.y && p.y <= self.hi.y &&
        p.z >= self.lo.z && p.z <= self.hi.z
    }
}

/// Fit an AABB to a set of positions selected by `ids`.
pub fn fit_aabb(pos: &[Vec3d], ids: &[Index]) -> Aabb {
    assert!(!ids.is_empty(), "fit_aabb: empty id list");
    let mut bb = Aabb::empty();
    for &id in ids {
        bb.include_point(pos[id as usize]);
    }
    bb
}

/// Fit AABBs for multiple groups defined by IndexGroups (offsets + items).
/// Writes to `out` which must have length >= groups.ngroups().
pub fn fit_group_aabbs(pos: &[Vec3d], groups: &pgraph::IndexGroups, out: &mut [Aabb]) {
    let ng = groups.ngroups();
    assert!(out.len() >= ng, "fit_group_aabbs: out.len() {} < ngroups {}", out.len(), ng);
    for g in 0..ng {
        let items = groups.group(g);
        if items.is_empty() {
            out[g] = Aabb::empty();
        } else {
            out[g] = fit_aabb(pos, items);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_include() {
        let mut bb = Aabb::empty();
        bb.include_point(Vec3d::new(1.0, 2.0, 3.0));
        bb.include_point(Vec3d::new(-1.0, 5.0, 0.0));
        bb.include_point(Vec3d::new(0.0, 0.0, 10.0));
        assert!((bb.lo.x + 1.0).abs() < 1e-12);
        assert!((bb.hi.x - 1.0).abs() < 1e-12);
        assert!((bb.hi.z - 10.0).abs() < 1e-12);
        assert!((bb.max_extent() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_fit_aabb() {
        let pos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(2.0, 0.0, 0.0), Vec3d::new(0.0, 3.0, 0.0)];
        let ids = vec![0, 1, 2];
        let bb = fit_aabb(&pos, &ids);
        assert!((bb.lo.x - 0.0).abs() < 1e-12);
        assert!((bb.hi.x - 2.0).abs() < 1e-12);
        assert!((bb.hi.y - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_fit_group_aabbs() {
        use pgraph::IndexGroups;
        let pos = vec![
            Vec3d::new(0.0, 0.0, 0.0),  // group 0
            Vec3d::new(1.0, 0.0, 0.0),  // group 0
            Vec3d::new(10.0, 10.0, 0.0), // group 1
            Vec3d::new(12.0, 10.0, 0.0), // group 1
        ];
        let groups = IndexGroups { offsets: vec![0, 2, 4], items: vec![0, 1, 2, 3] };
        let mut out = vec![Aabb::empty(); 2];
        fit_group_aabbs(&pos, &groups, &mut out);
        assert!((out[0].hi.x - 1.0).abs() < 1e-12, "group 0 max x: {}", out[0].hi.x);
        assert!((out[1].lo.x - 10.0).abs() < 1e-12, "group 1 min x: {}", out[1].lo.x);
        assert!((out[1].hi.x - 12.0).abs() < 1e-12, "group 1 max x: {}", out[1].hi.x);
    }

    #[test]
    fn test_contains() {
        let mut bb = Aabb::empty();
        bb.include_point(Vec3d::new(0.0, 0.0, 0.0));
        bb.include_point(Vec3d::new(2.0, 2.0, 2.0));
        assert!(bb.contains(Vec3d::new(1.0, 1.0, 1.0)));
        assert!(bb.contains(Vec3d::new(0.0, 0.0, 0.0)));  // inclusive
        assert!(bb.contains(Vec3d::new(2.0, 2.0, 2.0)));  // inclusive
        assert!(!bb.contains(Vec3d::new(3.0, 1.0, 1.0)));
    }
}
