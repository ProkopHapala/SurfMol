//! Low-level vector types: Vec2/3/4/6 in f32 and f64.
//!
//! * Vec2 = 2 numbers, sometimes interpreted as complex.
//! * Vec4 = 4 numbers, sometimes interpreted as quaternion.
//! * Vec6 = 2 × Vec3, used for AABB and symmetric 3×3 tensors.
//!
//! Operators (`+`, `-`, `*`, `/`) are component-wise vector arithmetic.
//! Quaternion and complex multiplication are explicit free functions (`qmul`, `cmul`).

use bytemuck::{Pod, Zeroable};

// =======================================================================================
// Vec2
// =======================================================================================

macro_rules! impl_vec2 {
    ($Name:ident, $T:ty) => {
        #[repr(C)]
        #[derive(Copy, Clone, Default, Debug, PartialEq, Pod, Zeroable)]
        pub struct $Name { pub x: $T, pub y: $T }

        impl $Name {
            #[inline(always)] pub const fn new(x: $T, y: $T) -> Self { Self { x, y } }
            #[inline(always)] pub fn set(&mut self, x: $T, y: $T) { self.x = x; self.y = y; }
            #[inline(always)] pub fn array(&self) -> &[$T; 2] { bytemuck::cast_ref(self) }
            #[inline(always)] pub fn array_mut(&mut self) -> &mut [$T; 2] { bytemuck::cast_mut(self) }
            #[inline(always)] pub fn dot(self, b: Self) -> $T { self.x * b.x + self.y * b.y }
            #[inline(always)] pub fn norm2(self) -> $T { self.dot(self) }
            #[inline(always)] pub fn norm(self) -> $T { self.norm2().sqrt() }
            #[inline(always)] pub fn add(&mut self, b: Self) { self.x += b.x; self.y += b.y; }
            #[inline(always)] pub fn sub(&mut self, b: Self) { self.x -= b.x; self.y -= b.y; }
            #[inline(always)] pub fn mul(&mut self, s: $T) { self.x *= s; self.y *= s; }
            /// Complex multiply in-place. Component-wise `*` is not overloaded for this.
            #[inline(always)] pub fn mul_cmplx(&mut self, rhs: Self) { let nx = self.x * rhs.x - self.y * rhs.y; let ny = self.x * rhs.y + self.y * rhs.x; self.x = nx; self.y = ny; }
            /// Complex divide in-place.
            #[inline(always)] pub fn udiv_cmplx(&mut self, rhs: Self) { let den = rhs.x * rhs.x + rhs.y * rhs.y; let nx = (self.x * rhs.x + self.y * rhs.y) / den; let ny = (self.y * rhs.x - self.x * rhs.y) / den; self.x = nx; self.y = ny; }
        }

        impl std::ops::Index<usize> for $Name {
            type Output = $T;
            #[inline(always)] fn index(&self, i: usize) -> &$T { &self.array()[i] }
        }
        impl std::ops::IndexMut<usize> for $Name {
            #[inline(always)] fn index_mut(&mut self, i: usize) -> &mut $T { &mut self.array_mut()[i] }
        }
        impl std::ops::Add for $Name { type Output = Self; #[inline(always)] fn add(self, rhs: Self) -> Self { Self { x: self.x + rhs.x, y: self.y + rhs.y } } }
        impl std::ops::Sub for $Name { type Output = Self; #[inline(always)] fn sub(self, rhs: Self) -> Self { Self { x: self.x - rhs.x, y: self.y - rhs.y } } }
        impl std::ops::Mul for $Name { type Output = Self; #[inline(always)] fn mul(self, rhs: Self) -> Self { Self { x: self.x * rhs.x, y: self.y * rhs.y } } }
        impl std::ops::Div for $Name { type Output = Self; #[inline(always)] fn div(self, rhs: Self) -> Self { Self { x: self.x / rhs.x, y: self.y / rhs.y } } }
        impl std::ops::Mul<$T> for $Name { type Output = Self; #[inline(always)] fn mul(self, rhs: $T) -> Self { Self { x: self.x * rhs, y: self.y * rhs } } }
        impl std::ops::Div<$T> for $Name { type Output = Self; #[inline(always)] fn div(self, rhs: $T) -> Self { Self { x: self.x / rhs, y: self.y / rhs } } }
    };
}

impl_vec2!(Vec2d, f64);
impl_vec2!(Vec2f, f32);

/// Complex multiply (a+ib)*(c+id). Explicit — never overload `*` for this.
#[inline(always)] pub fn cmul(a: Vec2d, b: Vec2d) -> Vec2d { Vec2d::new(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x) }
/// Complex conjugate.
#[inline(always)] pub fn cconj(a: Vec2d) -> Vec2d { Vec2d::new(a.x, -a.y) }

// =======================================================================================
// Vec3
// =======================================================================================

macro_rules! impl_vec3 {
    ($Name:ident, $T:ty) => {
        #[repr(C)]
        #[derive(Copy, Clone, Default, Debug, PartialEq, Pod, Zeroable)]
        pub struct $Name { pub x: $T, pub y: $T, pub z: $T }

        impl $Name {
            #[inline(always)] pub const fn new(x: $T, y: $T, z: $T) -> Self { Self { x, y, z } }
            #[inline(always)] pub fn set(&mut self, x: $T, y: $T, z: $T) { self.x = x; self.y = y; self.z = z; }
            #[inline(always)] pub fn array(&self) -> &[$T; 3] { bytemuck::cast_ref(self) }
            #[inline(always)] pub fn array_mut(&mut self) -> &mut [$T; 3] { bytemuck::cast_mut(self) }
            #[inline(always)] pub fn dot(self, b: Self) -> $T { self.x * b.x + self.y * b.y + self.z * b.z }
            #[inline(always)] pub fn norm2(self) -> $T { self.dot(self) }
            #[inline(always)] pub fn norm(self) -> $T { self.norm2().sqrt() }
            #[inline(always)] pub fn normalize(&mut self) -> $T { let n = self.norm(); if n > 1e-14 as $T { let inv = 1.0 as $T / n; self.x *= inv; self.y *= inv; self.z *= inv; } n }
            #[inline(always)] pub fn add(&mut self, b: Self) { self.x += b.x; self.y += b.y; self.z += b.z; }
            #[inline(always)] pub fn sub(&mut self, b: Self) { self.x -= b.x; self.y -= b.y; self.z -= b.z; }
            #[inline(always)] pub fn mul(&mut self, s: $T) { self.x *= s; self.y *= s; self.z *= s; }
            #[inline(always)] pub fn add_mul(&mut self, b: Self, s: $T) { self.x += b.x * s; self.y += b.y * s; self.z += b.z * s; }
            #[inline(always)] pub fn add_lincomb(&mut self, s1: $T, a1: Self, s2: $T, a2: Self) { self.x += s1 * a1.x + s2 * a2.x; self.y += s1 * a1.y + s2 * a2.y; self.z += s1 * a1.z + s2 * a2.z; }
            #[inline(always)] pub fn cross(self, b: Self) -> Self { Self { x: self.y * b.z - self.z * b.y, y: self.z * b.x - self.x * b.z, z: self.x * b.y - self.y * b.x } }
            #[inline(always)] pub fn set_add(a: Self, b: Self) -> Self { Self { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z } }
            #[inline(always)] pub fn set_sub(a: Self, b: Self) -> Self { Self { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z } }
            #[inline(always)] pub fn set_mul(a: Self, s: $T) -> Self { Self { x: a.x * s, y: a.y * s, z: a.z * s } }
            #[inline(always)] pub fn set_add_mul(a: Self, b: Self, s: $T) -> Self { Self { x: a.x + b.x * s, y: a.y + b.y * s, z: a.z + b.z * s } }
            #[inline(always)] pub fn set_lincomb(s1: $T, a1: Self, s2: $T, a2: Self) -> Self { Self { x: s1 * a1.x + s2 * a2.x, y: s1 * a1.y + s2 * a2.y, z: s1 * a1.z + s2 * a2.z } }
        }

        impl std::ops::Index<usize> for $Name {
            type Output = $T;
            #[inline(always)] fn index(&self, i: usize) -> &$T { &self.array()[i] }
        }
        impl std::ops::IndexMut<usize> for $Name {
            #[inline(always)] fn index_mut(&mut self, i: usize) -> &mut $T { &mut self.array_mut()[i] }
        }
        impl std::ops::Add for $Name { type Output = Self; #[inline(always)] fn add(self, rhs: Self) -> Self { Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z } } }
        impl std::ops::Sub for $Name { type Output = Self; #[inline(always)] fn sub(self, rhs: Self) -> Self { Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z } } }
        impl std::ops::Mul for $Name { type Output = Self; #[inline(always)] fn mul(self, rhs: Self) -> Self { Self { x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z } } }
        impl std::ops::Div for $Name { type Output = Self; #[inline(always)] fn div(self, rhs: Self) -> Self { Self { x: self.x / rhs.x, y: self.y / rhs.y, z: self.z / rhs.z } } }
        impl std::ops::Mul<$T> for $Name { type Output = Self; #[inline(always)] fn mul(self, rhs: $T) -> Self { Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs } } }
        impl std::ops::Div<$T> for $Name { type Output = Self; #[inline(always)] fn div(self, rhs: $T) -> Self { Self { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs } } }
    };
}

impl_vec3!(Vec3d, f64);
impl_vec3!(Vec3f, f32);

#[inline(always)] pub fn cross(a: Vec3d, b: Vec3d) -> Vec3d { Vec3d::new(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x) }
#[inline(always)] pub fn cross_f(a: Vec3f, b: Vec3f) -> Vec3f { Vec3f::new(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x) }

// =======================================================================================
// Vec4
// =======================================================================================

macro_rules! impl_vec4 {
    ($Name:ident, $T:ty, $V3:ty) => {
        #[repr(C)]
        #[derive(Copy, Clone, Default, Debug, PartialEq, Pod, Zeroable)]
        pub struct $Name { pub x: $T, pub y: $T, pub z: $T, pub w: $T }

        impl $Name {
            #[inline(always)] pub const fn new(x: $T, y: $T, z: $T, w: $T) -> Self { Self { x, y, z, w } }
            #[inline(always)] pub fn set(&mut self, x: $T, y: $T, z: $T, w: $T) { self.x = x; self.y = y; self.z = z; self.w = w; }
            #[inline(always)] pub fn array(&self) -> &[$T; 4] { bytemuck::cast_ref(self) }
            #[inline(always)] pub fn array_mut(&mut self) -> &mut [$T; 4] { bytemuck::cast_mut(self) }
            #[inline(always)] pub fn dot(self, b: Self) -> $T { self.x*b.x + self.y*b.y + self.z*b.z + self.w*b.w }
            #[inline(always)] pub fn norm2(self) -> $T { self.dot(self) }
            #[inline(always)] pub fn norm(self) -> $T { self.norm2().sqrt() }
            #[inline(always)] pub fn add(&mut self, b: Self) { self.x += b.x; self.y += b.y; self.z += b.z; self.w += b.w; }
            #[inline(always)] pub fn sub(&mut self, b: Self) { self.x -= b.x; self.y -= b.y; self.z -= b.z; self.w -= b.w; }
            #[inline(always)] pub fn mul(&mut self, s: $T) { self.x *= s; self.y *= s; self.z *= s; self.w *= s; }
            #[inline(always)] pub fn xyz(&self) -> $V3 { <$V3>::new(self.x, self.y, self.z) }
            /// Alias for `xyz()` — vector part. Keeps the FireCore `q.f()` convention.
            #[inline(always)] pub fn f(&self) -> $V3 { self.xyz() }
        }

        impl std::ops::Index<usize> for $Name {
            type Output = $T;
            #[inline(always)] fn index(&self, i: usize) -> &$T { &self.array()[i] }
        }
        impl std::ops::IndexMut<usize> for $Name {
            #[inline(always)] fn index_mut(&mut self, i: usize) -> &mut $T { &mut self.array_mut()[i] }
        }
        impl std::ops::Add for $Name { type Output = Self; #[inline(always)] fn add(self, rhs: Self) -> Self { Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w } } }
        impl std::ops::Sub for $Name { type Output = Self; #[inline(always)] fn sub(self, rhs: Self) -> Self { Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w } } }
        impl std::ops::Mul for $Name { type Output = Self; #[inline(always)] fn mul(self, rhs: Self) -> Self { Self { x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z, w: self.w * rhs.w } } }
        impl std::ops::Div for $Name { type Output = Self; #[inline(always)] fn div(self, rhs: Self) -> Self { Self { x: self.x / rhs.x, y: self.y / rhs.y, z: self.z / rhs.z, w: self.w / rhs.w } } }
        impl std::ops::Mul<$T> for $Name { type Output = Self; #[inline(always)] fn mul(self, rhs: $T) -> Self { Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs, w: self.w * rhs } } }
        impl std::ops::Div<$T> for $Name { type Output = Self; #[inline(always)] fn div(self, rhs: $T) -> Self { Self { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs, w: self.w / rhs } } }
    };
}

impl_vec4!(Vec4d, f64, Vec3d);
impl_vec4!(Vec4f, f32, Vec3f);

/// Quaternion multiplication with canonical layout (x,y,z) = vector part, w = scalar.
#[inline(always)] pub fn qmul(a: Vec4d, b: Vec4d) -> Vec4d {
    Vec4d::new(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    )
}
/// Quaternion conjugate.
#[inline(always)] pub fn qconj(q: Vec4d) -> Vec4d { Vec4d::new(-q.x, -q.y, -q.z, q.w) }
/// Rotate a 3-vector by a unit quaternion.
#[inline(always)] pub fn qrotate(q: Vec4d, v: Vec3d) -> Vec3d {
    let qv = Vec4d::new(v.x, v.y, v.z, 0.0);
    qmul(q, qmul(qv, qconj(q))).xyz()
}

// =======================================================================================
// Vec4i / Quat4i — 4-int pack for neighbor indices and 4-atom interaction tuples.
// Kept concrete and un-aliased at type level: Quat4i is a compatibility alias.
// =======================================================================================

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Vec4i { pub x: i32, pub y: i32, pub z: i32, pub w: i32 }

impl Vec4i {
    #[inline(always)] pub const fn new(x: i32, y: i32, z: i32, w: i32) -> Self { Self { x, y, z, w } }
    #[inline(always)] pub fn set(&mut self, x: i32, y: i32, z: i32, w: i32) { self.x = x; self.y = y; self.z = z; self.w = w; }
    #[inline(always)] pub fn array(&self) -> &[i32; 4] { bytemuck::cast_ref(self) }
    #[inline(always)] pub fn array_mut(&mut self) -> &mut [i32; 4] { bytemuck::cast_mut(self) }
    #[inline(always)] pub fn as_array(&self) -> [i32; 4] { [self.x, self.y, self.z, self.w] }
    #[inline(always)] pub fn xyz(&self) -> [i32; 3] { [self.x, self.y, self.z] }
}

impl std::ops::Index<usize> for Vec4i {
    type Output = i32;
    #[inline(always)] fn index(&self, i: usize) -> &i32 { &self.array()[i] }
}
impl std::ops::IndexMut<usize> for Vec4i {
    #[inline(always)] fn index_mut(&mut self, i: usize) -> &mut i32 { &mut self.array_mut()[i] }
}

/// Backward-compatible quaternion/4-int aliases.
pub type Quat4d = Vec4d;
pub type Quat4i = Vec4i;

pub const QUAT4I_MINUS_ONES: Quat4i = Quat4i { x: -1, y: -1, z: -1, w: -1 };

// =======================================================================================
// Vec6 = 2 × Vec3
// Used for AABB (a=lo, b=hi) and symmetric 3×3 matrices (a=diag, b=yz/xz/xy off-pairs).
// =======================================================================================

macro_rules! impl_vec6 {
    ($Name:ident, $V3:ty, $T:ty) => {
        #[repr(C)]
        #[derive(Copy, Clone, Default, Debug, PartialEq, Pod, Zeroable)]
        pub struct $Name { pub a: $V3, pub b: $V3 }

        impl $Name {
            #[inline(always)] pub const fn new(a: $V3, b: $V3) -> Self { Self { a, b } }
            #[inline(always)] pub fn array(&self) -> &[$T; 6] { bytemuck::cast_ref(self) }
            #[inline(always)] pub fn array_mut(&mut self) -> &mut [$T; 6] { bytemuck::cast_mut(self) }
            #[inline(always)] pub fn vecs(&self) -> &[$V3; 2] { bytemuck::cast_ref(self) }
            #[inline(always)] pub fn vecs_mut(&mut self) -> &mut [$V3; 2] { bytemuck::cast_mut(self) }
            #[inline(always)] pub fn add(&mut self, b: Self) { self.a.add(b.a); self.b.add(b.b); }
            #[inline(always)] pub fn sub(&mut self, b: Self) { self.a.sub(b.a); self.b.sub(b.b); }
            #[inline(always)] pub fn mul(&mut self, s: $T) { self.a.mul(s); self.b.mul(s); }
        }

        impl std::ops::Index<usize> for $Name {
            type Output = $T;
            #[inline(always)] fn index(&self, i: usize) -> &$T { &self.array()[i] }
        }
        impl std::ops::IndexMut<usize> for $Name {
            #[inline(always)] fn index_mut(&mut self, i: usize) -> &mut $T { &mut self.array_mut()[i] }
        }
    };
}

impl_vec6!(Vec6d, Vec3d, f64);
impl_vec6!(Vec6f, Vec3f, f32);

pub const VEC2D_ZERO: Vec2d = Vec2d { x: 0.0, y: 0.0 };
pub const VEC3D_ZERO: Vec3d = Vec3d { x: 0.0, y: 0.0, z: 0.0 };
pub const VEC4D_ZERO: Vec4d = Vec4d { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
pub const VEC6D_ZERO: Vec6d = Vec6d { a: VEC3D_ZERO, b: VEC3D_ZERO };
pub const VEC2F_ZERO: Vec2f = Vec2f { x: 0.0, y: 0.0 };
pub const VEC3F_ZERO: Vec3f = Vec3f { x: 0.0, y: 0.0, z: 0.0 };
pub const VEC4F_ZERO: Vec4f = Vec4f { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
pub const VEC6F_ZERO: Vec6f = Vec6f { a: VEC3F_ZERO, b: VEC3F_ZERO };
pub const VEC3D_NAN: Vec3d = Vec3d { x: f64::NAN, y: f64::NAN, z: f64::NAN };

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_ops() {
        let a = Vec3d::new(1.0, 2.0, 3.0);
        let b = Vec3d::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0);
        assert_eq!((a * 2.0).x, 2.0);
        let c = a + b;
        assert_eq!(c, Vec3d::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn vec4_qmul() {
        let q = qmul(Vec4d::new(0.0, 0.0, 0.0, 1.0), Vec4d::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(q, Vec4d::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn vec6_array() {
        let v = Vec6d::new(Vec3d::new(1.0, 2.0, 3.0), Vec3d::new(4.0, 5.0, 6.0));
        assert_eq!(v.array(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
