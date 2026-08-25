/// f32 3D vector helpers

#[inline(always)] pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if n > 1e-8 { [v[0]/n, v[1]/n, v[2]/n] } else { v }
}

#[inline(always)] pub fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

#[inline(always)] pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

#[inline(always)] pub fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}

#[inline(always)] pub fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
}

#[inline(always)] pub fn mul3s(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0]*s, a[1]*s, a[2]*s]
}
