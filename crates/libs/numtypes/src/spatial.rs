//! Spatial data layouts: AABB as `Vec6` and symmetric 3×3 tensor as `Vec6`.
//!
//! Both are `2 × Vec3`; the same physical layout, different semantic interpretations.
//! Intrinsic operations are standalone functions (type aliases cannot carry inherent impls).

use crate::vec::{Vec3d, Vec6d, Vec6f};

// =======================================================================================
// Vec6 semantic aliases
// =======================================================================================

/// AABB in 3D: `a` = lower bound, `b` = upper bound.
pub type Aabb3d = Vec6d;
pub type Aabb3f = Vec6f;

/// Symmetric 3×3 matrix: `a` = diagonal (xx, yy, zz), `b` = off-diagonal pairs (yz, xz, xy).
pub type SymMat3d = Vec6d;
pub type SymMat3f = Vec6f;

// =======================================================================================
// AABB intrinsic functions
// =======================================================================================

/// Empty AABB: lo = +inf, hi = -inf.
#[inline(always)] pub fn aabb_empty() -> Aabb3d { Aabb3d::new(Vec3d::new(f64::INFINITY, f64::INFINITY, f64::INFINITY), Vec3d::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY)) }

/// Expand AABB to include a point.
#[inline(always)] pub fn aabb_include(bb: &mut Aabb3d, p: Vec3d) {
    if p.x < bb.a.x { bb.a.x = p.x; } if p.x > bb.b.x { bb.b.x = p.x; }
    if p.y < bb.a.y { bb.a.y = p.y; } if p.y > bb.b.y { bb.b.y = p.y; }
    if p.z < bb.a.z { bb.a.z = p.z; } if p.z > bb.b.z { bb.b.z = p.z; }
}

/// Expand AABB to include another AABB.
#[inline(always)] pub fn aabb_include_aabb(bb: &mut Aabb3d, other: Aabb3d) { aabb_include(bb, other.a); aabb_include(bb, other.b); }

/// Test if a point is inside the AABB (inclusive bounds).
#[inline(always)] pub fn aabb_contains(bb: Aabb3d, p: Vec3d) -> bool {
    p.x >= bb.a.x && p.x <= bb.b.x && p.y >= bb.a.y && p.y <= bb.b.y && p.z >= bb.a.z && p.z <= bb.b.z
}

/// Test if two AABBs overlap.
#[inline(always)] pub fn aabb_overlap(a: Aabb3d, b: Aabb3d) -> bool {
    a.a.x <= b.b.x && a.b.x >= b.a.x && a.a.y <= b.b.y && a.b.y >= b.a.y && a.a.z <= b.b.z && a.b.z >= b.a.z
}

/// Merge two AABBs.
#[inline(always)] pub fn aabb_merge(a: Aabb3d, b: Aabb3d) -> Aabb3d {
    Aabb3d::new(Vec3d::new(a.a.x.min(b.a.x), a.a.y.min(b.a.y), a.a.z.min(b.a.z)), Vec3d::new(a.b.x.max(b.b.x), a.b.y.max(b.b.y), a.b.z.max(b.b.z)))
}

/// AABB center. Returns NaN for empty AABBs (caller must guard or detect).
#[inline(always)] pub fn aabb_center(bb: Aabb3d) -> Vec3d { Vec3d::new((bb.a.x + bb.b.x) * 0.5, (bb.a.y + bb.b.y) * 0.5, (bb.a.z + bb.b.z) * 0.5) }

/// AABB size (hi - lo).
#[inline(always)] pub fn aabb_size(bb: Aabb3d) -> Vec3d { Vec3d::new(bb.b.x - bb.a.x, bb.b.y - bb.a.y, bb.b.z - bb.a.z) }

/// AABB maximum extent.
#[inline(always)] pub fn aabb_max_extent(bb: Aabb3d) -> f64 { aabb_size(bb).x.max(aabb_size(bb).y).max(aabb_size(bb).z) }

/// True if the AABB has valid (non-empty) bounds: lo <= hi in all axes.
#[inline(always)] pub fn aabb_is_valid(bb: Aabb3d) -> bool { bb.a.x <= bb.b.x && bb.a.y <= bb.b.y && bb.a.z <= bb.b.z }

// =======================================================================================
// Symmetric 3×3 intrinsic functions
// =======================================================================================

/// Determinant of a symmetric 3×3 matrix with the `(diag, off)` packing.
#[inline(always)] pub fn sym3_det(m: SymMat3d) -> f64 {
    m.a.x*m.a.y*m.a.z + 2.0*m.b.x*m.b.y*m.b.z - m.a.x*m.b.x*m.b.x - m.a.y*m.b.y*m.b.y - m.a.z*m.b.z*m.b.z
}

/// Symmetric 3×3 matrix times vector.
#[inline(always)] pub fn sym3_dot(m: SymMat3d, v: Vec3d) -> Vec3d {
    Vec3d::new(
        m.a.x*v.x + m.b.z*v.y + m.b.y*v.z,
        m.b.z*v.x + m.a.y*v.y + m.b.x*v.z,
        m.b.y*v.x + m.b.x*v.y + m.a.z*v.z,
    )
}

/// Outer product of a vector with itself as a symmetric 3×3 matrix.
#[inline(always)] pub fn sym3_outer(v: Vec3d) -> SymMat3d {
    SymMat3d::new(Vec3d::new(v.x*v.x, v.y*v.y, v.z*v.z), Vec3d::new(v.y*v.z, v.x*v.z, v.x*v.y))
}

/// Quadratic form v^T M v.
#[inline(always)] pub fn sym3_quadratic(m: SymMat3d, v: Vec3d) -> f64 {
    m.a.x*v.x*v.x + m.a.y*v.y*v.y + m.a.z*v.z*v.z
    + 2.0*(m.b.z*v.x*v.y + m.b.y*v.x*v.z + m.b.x*v.y*v.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_basic() {
        let mut bb = aabb_empty();
        aabb_include(&mut bb, Vec3d::new(1.0, 2.0, 3.0));
        aabb_include(&mut bb, Vec3d::new(-1.0, 5.0, 0.0));
        aabb_include(&mut bb, Vec3d::new(0.0, 0.0, 10.0));
        assert!(bb.a == Vec3d::new(-1.0, 0.0, 0.0));
        assert!(bb.b == Vec3d::new(1.0, 5.0, 10.0));
        assert!(aabb_contains(bb, Vec3d::new(0.0, 2.5, 5.0)));
        assert!(!aabb_contains(bb, Vec3d::new(2.0, 2.5, 5.0)));
    }

    #[test]
    fn sym3_det_and_dot() {
        // M = diag(1,2,3)
        let m = SymMat3d::new(Vec3d::new(1.0, 2.0, 3.0), Vec3d::new(0.0, 0.0, 0.0));
        assert_eq!(sym3_det(m), 6.0);
        assert_eq!(sym3_dot(m, Vec3d::new(1.0, 1.0, 1.0)), Vec3d::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn aabb_overlap_test() {
        let a = Aabb3d::new(Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(2.0, 2.0, 2.0));
        let b = Aabb3d::new(Vec3d::new(1.5, 1.5, 1.5), Vec3d::new(3.5, 3.5, 3.5));
        assert!(aabb_overlap(a, b));
        let c = Aabb3d::new(Vec3d::new(3.0, 0.0, 0.0), Vec3d::new(4.0, 1.0, 1.0));
        assert!(!aabb_overlap(a, c));
    }
}
