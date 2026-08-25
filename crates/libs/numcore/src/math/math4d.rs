use crate::math::math3d::*;

/// Build a right-handed look-at view matrix (row-major, row-vector convention:
/// clip = point * M). Upload directly to WGSL — the column-major byte
/// interpretation in WGSL transposes it, so M_wgsl = M^T and M_wgsl * v = v * M.
/// Do NOT additionally transpose before upload (that double-transposes).
#[inline(always)] pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let z = normalize3(sub3(eye, target));
    let x = normalize3(cross3(up, z));
    let y = cross3(z, x);
    [
        [x[0], y[0], z[0], 0.0],
        [x[1], y[1], z[1], 0.0],
        [x[2], y[2], z[2], 0.0],
        [-dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1.0],
    ]
}

/// Orthographic projection for Vulkan/DirectX NDC Z in [0,1].
#[inline(always)] pub fn ortho(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> [[f32; 4]; 4] {
    [
        [2.0/(r-l), 0.0,        0.0,         -(r+l)/(r-l)],
        [0.0,       2.0/(t-b),  0.0,         -(t+b)/(t-b)],
        [0.0,       0.0,        -1.0/(f-n),  -n/(f-n)],
        [0.0,       0.0,        0.0,         1.0],
    ]
}

#[inline(always)] pub fn mul4x4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut c = [[0.0f32; 4]; 4];
    for i in 0..4 { for j in 0..4 { for k in 0..4 { c[i][j] += a[i][k] * b[k][j]; } } }
    c
}

#[inline(always)] pub fn transpose4x4(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut t = [[0.0f32; 4]; 4];
    for i in 0..4 { for j in 0..4 { t[j][i] = m[i][j]; } }
    t
}
