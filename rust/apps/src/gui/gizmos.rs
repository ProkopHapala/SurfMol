use glam::Vec3;
use surfmol_molrender::line_renderer::LineVertex;

/// Generate line segments for a multi-segment bond between two points.
pub fn make_bond_segments(p0: Vec3, p1: Vec3, n_seg: i32, col: [f32; 4]) -> Vec<LineVertex> {
    let mut verts = Vec::with_capacity(n_seg as usize * 2);
    let mut prev = p0;
    for s in 1..=n_seg {
        let t = s as f32 / n_seg as f32;
        let curr = p0 + (p1 - p0) * t;
        verts.push(LineVertex { pos: [prev.x, prev.y, prev.z], col });
        verts.push(LineVertex { pos: [curr.x, curr.y, curr.z], col });
        prev = curr;
    }
    verts
}

/// Generate a ring (circle) of line segments in the XY plane.
pub fn make_ring(center: Vec3, radius: f32, n_seg: i32, col: [f32; 4]) -> Vec<LineVertex> {
    let mut verts = Vec::with_capacity(n_seg as usize * 2);
    for k in 0..n_seg {
        let t0 = (k as f32 / n_seg as f32) * std::f32::consts::TAU;
        let t1 = ((k + 1) as f32 / n_seg as f32) * std::f32::consts::TAU;
        let p0r = center + Vec3::new(t0.cos(), t0.sin(), 0.0) * radius;
        let p1r = center + Vec3::new(t1.cos(), t1.sin(), 0.0) * radius;
        verts.push(LineVertex { pos: [p0r.x, p0r.y, p0r.z], col });
        verts.push(LineVertex { pos: [p1r.x, p1r.y, p1r.z], col });
    }
    verts
}

/// Generate RGB XYZ axes at the origin, scaled by `scale`.
pub fn make_axes(origin: [f32; 3], scale: f32) -> Vec<LineVertex> {
    let mut verts = Vec::with_capacity(6);
    let o = origin;
    verts.push(LineVertex { pos: o, col: [1.0, 0.0, 0.0, 1.0] });
    verts.push(LineVertex { pos: [o[0] + scale, o[1], o[2]], col: [1.0, 0.0, 0.0, 1.0] });
    verts.push(LineVertex { pos: o, col: [0.0, 1.0, 0.0, 1.0] });
    verts.push(LineVertex { pos: [o[0], o[1] + scale, o[2]], col: [0.0, 1.0, 0.0, 1.0] });
    verts.push(LineVertex { pos: o, col: [0.0, 0.0, 1.0, 1.0] });
    verts.push(LineVertex { pos: [o[0], o[1], o[2] + scale], col: [0.0, 0.0, 1.0, 1.0] });
    verts
}

/// Generate a 3D crosshair at `center` with arm length `sz`.
pub fn make_crosshair(center: Vec3, sz: f32, col: [f32; 4]) -> Vec<LineVertex> {
    let mut verts = Vec::with_capacity(12);
    let c = center;
    verts.push(LineVertex { pos: [c.x - sz, c.y, c.z], col });
    verts.push(LineVertex { pos: [c.x + sz, c.y, c.z], col });
    verts.push(LineVertex { pos: [c.x, c.y - sz, c.z], col });
    verts.push(LineVertex { pos: [c.x, c.y + sz, c.z], col });
    verts.push(LineVertex { pos: [c.x, c.y, c.z - sz], col });
    verts.push(LineVertex { pos: [c.x, c.y, c.z + sz], col });
    verts
}
