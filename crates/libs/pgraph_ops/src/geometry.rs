//! Geometry helpers: edge vectors, lengths, normals from positioned graph data.
//! These are shared by picking, selection, and rendering — no molecular semantics.

use numcore::math::vec3::Vec3d;
use pgraph::Index;

/// Edge vector from vertex a to vertex b: `pos[b] - pos[a]`.
#[inline(always)]
pub fn edge_vec(pos: &[Vec3d], a: Index, b: Index) -> Vec3d {
    Vec3d::set_sub(pos[b as usize], pos[a as usize])
}

/// Edge length: `|pos[b] - pos[a]|`.
#[inline(always)]
pub fn edge_length(pos: &[Vec3d], a: Index, b: Index) -> f64 {
    edge_vec(pos, a, b).norm()
}

/// Compute all edge lengths as a Vec<f64>.
pub fn edge_lengths(pos: &[Vec3d], edges: &[[Index; 2]]) -> Vec<f64> {
    edges.iter().map(|&[a, b]| edge_length(pos, a, b)).collect()
}

/// Bounding box of a set of positions. Returns (min, max).
pub fn bounding_box(pos: &[Vec3d]) -> (Vec3d, Vec3d) {
    assert!(!pos.is_empty(), "bounding_box: empty position slice");
    let mut lo = pos[0];
    let mut hi = pos[0];
    for &p in &pos[1..] {
        if p.x < lo.x { lo.x = p.x; } else if p.x > hi.x { hi.x = p.x; }
        if p.y < lo.y { lo.y = p.y; } else if p.y > hi.y { hi.y = p.y; }
        if p.z < lo.z { lo.z = p.z; } else if p.z > hi.z { hi.z = p.z; }
    }
    (lo, hi)
}

/// Bounding box center: `(min + max) / 2`.
pub fn bounding_box_center(pos: &[Vec3d]) -> Vec3d {
    let (lo, hi) = bounding_box(pos);
    Vec3d::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, (lo.z + hi.z) * 0.5)
}

/// Bounding box span (max dimension): `max(hi - lo)`.
pub fn bounding_box_span(pos: &[Vec3d]) -> f64 {
    let (lo, hi) = bounding_box(pos);
    let dx = hi.x - lo.x;
    let dy = hi.y - lo.y;
    let dz = hi.z - lo.z;
    dx.max(dy).max(dz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_vec_length() {
        let pos = vec![Vec3d::new(0.0, 0.0, 0.0), Vec3d::new(3.0, 4.0, 0.0)];
        let v = edge_vec(&pos, 0, 1);
        assert!((v.x - 3.0).abs() < 1e-12);
        assert!((v.y - 4.0).abs() < 1e-12);
        assert!((v.z - 0.0).abs() < 1e-12);
        let l = edge_length(&pos, 0, 1);
        assert!((l - 5.0).abs() < 1e-12, "3-4-5 triangle length should be 5, got {}", l);
    }

    #[test]
    fn test_bounding_box() {
        let pos = vec![
            Vec3d::new(1.0, 2.0, 3.0),
            Vec3d::new(-1.0, 5.0, 0.0),
            Vec3d::new(0.0, 0.0, 10.0),
        ];
        let (lo, hi) = bounding_box(&pos);
        assert!((lo.x + 1.0).abs() < 1e-12);
        assert!((lo.y - 0.0).abs() < 1e-12);
        assert!((lo.z - 0.0).abs() < 1e-12);
        assert!((hi.x - 1.0).abs() < 1e-12);
        assert!((hi.y - 5.0).abs() < 1e-12);
        assert!((hi.z - 10.0).abs() < 1e-12);
        let center = bounding_box_center(&pos);
        assert!((center.x - 0.0).abs() < 1e-12);
        assert!((center.y - 2.5).abs() < 1e-12);
        assert!((center.z - 5.0).abs() < 1e-12);
        assert!((bounding_box_span(&pos) - 10.0).abs() < 1e-12);
    }
}
