#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Vec3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3d {
    #[inline(always)] pub const fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    #[inline(always)] pub fn set(&mut self, x: f64, y: f64, z: f64) { self.x = x; self.y = y; self.z = z; }
    #[inline(always)] pub fn add(&mut self, b: Self) { self.x += b.x; self.y += b.y; self.z += b.z; }
    #[inline(always)] pub fn sub(&mut self, b: Self) { self.x -= b.x; self.y -= b.y; self.z -= b.z; }
    #[inline(always)] pub fn mul(&mut self, f: f64) { self.x *= f; self.y *= f; self.z *= f; }
    #[inline(always)] pub fn add_mul(&mut self, a: Self, f: f64) { self.x += a.x * f; self.y += a.y * f; self.z += a.z * f; }
    #[inline(always)] pub fn dot(self, b: Self) -> f64 { self.x * b.x + self.y * b.y + self.z * b.z }
    #[inline(always)] pub fn norm2(self) -> f64 { self.dot(self) }
    #[inline(always)] pub fn norm(self) -> f64 { self.norm2().sqrt() }
    #[inline(always)] pub fn set_sub(a: Self, b: Self) -> Self { Self { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z } }
    #[inline(always)] pub fn cross(a: Self, b: Self) -> Self { Self { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x } }
    #[inline(always)] pub fn set_mul(a: Self, s: f64) -> Self { Self { x: a.x * s, y: a.y * s, z: a.z * s } }
    #[inline(always)] pub fn set_add(a: Self, b: Self) -> Self { Self { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z } }
    #[inline(always)] pub fn set_add_mul(a: Self, b: Self, s: f64) -> Self { Self { x: a.x + b.x * s, y: a.y + b.y * s, z: a.z + b.z * s } }
    #[inline(always)] pub fn set_lincomb(s1: f64, a1: Self, s2: f64, a2: Self) -> Self { Self { x: s1 * a1.x + s2 * a2.x, y: s1 * a1.y + s2 * a2.y, z: s1 * a1.z + s2 * a2.z } }
    #[inline(always)] pub fn add_lincomb(&mut self, s1: f64, a1: Self, s2: f64, a2: Self) { self.x += s1 * a1.x + s2 * a2.x; self.y += s1 * a1.y + s2 * a2.y; self.z += s1 * a1.z + s2 * a2.z; }
    #[inline(always)] pub fn normalize(&mut self) -> f64 { let n = self.norm(); if n > 1e-14 { let inv = 1.0 / n; self.x *= inv; self.y *= inv; self.z *= inv; } n }
}

impl std::ops::Add<Vec3d> for Vec3d {
    type Output = Vec3d;
    #[inline(always)] fn add(self, rhs: Vec3d) -> Vec3d { Vec3d { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z } }
}
impl std::ops::Sub<Vec3d> for Vec3d {
    type Output = Vec3d;
    #[inline(always)] fn sub(self, rhs: Vec3d) -> Vec3d { Vec3d { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z } }
}
impl std::ops::Mul<f64> for Vec3d {
    type Output = Vec3d;
    #[inline(always)] fn mul(self, rhs: f64) -> Vec3d { Vec3d { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs } }
}

pub const VEC3_NAN: Vec3d = Vec3d { x: f64::NAN, y: f64::NAN, z: f64::NAN };
pub const VEC3_ZERO: Vec3d = Vec3d { x: 0.0, y: 0.0, z: 0.0 };
