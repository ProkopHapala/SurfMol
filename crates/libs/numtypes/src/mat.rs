//! 3×3 and 4×4 matrices as rows of Vec3/Vec4. Zero-copy `rows()` and `array()` views.

use bytemuck::{Pod, Zeroable};
use crate::vec::*;

// =======================================================================================
// Mat3
// =======================================================================================

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Pod, Zeroable)]
pub struct Mat3d { pub a: Vec3d, pub b: Vec3d, pub c: Vec3d }

impl Mat3d {
    #[inline(always)] pub const fn new(a: Vec3d, b: Vec3d, c: Vec3d) -> Self { Self { a, b, c } }
    #[inline(always)] pub fn zero() -> Self { Self::new(VEC3D_ZERO, VEC3D_ZERO, VEC3D_ZERO) }
    #[inline(always)] pub fn identity() -> Self { Self::new(Vec3d::new(1.0,0.0,0.0), Vec3d::new(0.0,1.0,0.0), Vec3d::new(0.0,0.0,1.0)) }
    #[inline(always)] pub fn rows(&self) -> &[Vec3d; 3] { bytemuck::cast_ref(self) }
    #[inline(always)] pub fn rows_mut(&mut self) -> &mut [Vec3d; 3] { bytemuck::cast_mut(self) }
    #[inline(always)] pub fn array(&self) -> &[f64; 9] { bytemuck::cast_ref(self) }
    #[inline(always)] pub fn array_mut(&mut self) -> &mut [f64; 9] { bytemuck::cast_mut(self) }
    #[inline(always)] pub fn set(&mut self, a: Vec3d, b: Vec3d, c: Vec3d) { self.a = a; self.b = b; self.c = c; }
    #[inline(always)] pub fn transpose(&mut self) { std::mem::swap(&mut self.a.y, &mut self.b.x); std::mem::swap(&mut self.a.z, &mut self.c.x); std::mem::swap(&mut self.b.z, &mut self.c.y); }
    #[inline(always)] pub fn transposed(self) -> Self { let mut m = self; m.transpose(); m }
    #[inline(always)] pub fn dot(self, v: Vec3d) -> Vec3d { Vec3d::new(self.a.dot(v), self.b.dot(v), self.c.dot(v)) }
    #[inline(always)] pub fn dot_t(self, v: Vec3d) -> Vec3d { Vec3d::new(self.a.x*v.x + self.b.x*v.y + self.c.x*v.z, self.a.y*v.x + self.b.y*v.y + self.c.y*v.z, self.a.z*v.x + self.b.z*v.y + self.c.z*v.z) }
    #[inline(always)] pub fn add(&mut self, b: Self) { self.a.add(b.a); self.b.add(b.b); self.c.add(b.c); }
    #[inline(always)] pub fn sub(&mut self, b: Self) { self.a.sub(b.a); self.b.sub(b.b); self.c.sub(b.c); }
    #[inline(always)] pub fn scale(&mut self, s: f64) { self.a.mul(s); self.b.mul(s); self.c.mul(s); }
    #[inline(always)] pub fn det(self) -> f64 {
        self.a.x*(self.b.y*self.c.z - self.b.z*self.c.y)
      - self.a.y*(self.b.x*self.c.z - self.b.z*self.c.x)
      + self.a.z*(self.b.x*self.c.y - self.b.y*self.c.x)
    }
    #[inline(always)] pub fn outer(v1: Vec3d, v2: Vec3d) -> Self {
        Self::new(
            Vec3d::new(v1.x*v2.x, v1.x*v2.y, v1.x*v2.z),
            Vec3d::new(v1.y*v2.x, v1.y*v2.y, v1.y*v2.z),
            Vec3d::new(v1.z*v2.x, v1.z*v2.y, v1.z*v2.z),
        )
    }
    #[inline(always)] pub fn add_outer(&mut self, v1: Vec3d, v2: Vec3d) { self.a.add_mul(Vec3d::new(v2.x, v2.y, v2.z), v1.x); self.b.add_mul(Vec3d::new(v2.x, v2.y, v2.z), v1.y); self.c.add_mul(Vec3d::new(v2.x, v2.y, v2.z), v1.z); }
    /// Explicit 3×3 inverse. Panics if singular (det == 0) — caller must guard.
    #[inline(always)] pub fn inverse(self) -> Self {
        let d = self.det();
        assert!(d != 0.0, "Mat3d::inverse: singular matrix");
        let inv_d = 1.0 / d;
        Self::new(
            Vec3d::new((self.b.y*self.c.z - self.b.z*self.c.y)*inv_d, (self.a.z*self.c.y - self.a.y*self.c.z)*inv_d, (self.a.y*self.b.z - self.a.z*self.b.y)*inv_d),
            Vec3d::new((self.b.z*self.c.x - self.b.x*self.c.z)*inv_d, (self.a.x*self.c.z - self.a.z*self.c.x)*inv_d, (self.a.z*self.b.x - self.a.x*self.b.z)*inv_d),
            Vec3d::new((self.b.x*self.c.y - self.b.y*self.c.x)*inv_d, (self.a.y*self.c.x - self.a.x*self.c.y)*inv_d, (self.a.x*self.b.y - self.a.y*self.b.x)*inv_d),
        )
    }
}

#[inline(always)] pub fn mmul3(a: Mat3d, b: Mat3d) -> Mat3d {
    let bt = b.transposed();
    Mat3d::new(
        Vec3d::new(a.a.dot(bt.a), a.a.dot(bt.b), a.a.dot(bt.c)),
        Vec3d::new(a.b.dot(bt.a), a.b.dot(bt.b), a.b.dot(bt.c)),
        Vec3d::new(a.c.dot(bt.a), a.c.dot(bt.b), a.c.dot(bt.c)),
    )
}

// =======================================================================================
// Mat4
// =======================================================================================

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Pod, Zeroable)]
pub struct Mat4d { pub a: Vec4d, pub b: Vec4d, pub c: Vec4d, pub d: Vec4d }

impl Mat4d {
    #[inline(always)] pub const fn new(a: Vec4d, b: Vec4d, c: Vec4d, d: Vec4d) -> Self { Self { a, b, c, d } }
    #[inline(always)] pub fn zero() -> Self { Self::new(VEC4D_ZERO, VEC4D_ZERO, VEC4D_ZERO, VEC4D_ZERO) }
    #[inline(always)] pub fn identity() -> Self { Self::new(Vec4d::new(1.0,0.0,0.0,0.0), Vec4d::new(0.0,1.0,0.0,0.0), Vec4d::new(0.0,0.0,1.0,0.0), Vec4d::new(0.0,0.0,0.0,1.0)) }
    #[inline(always)] pub fn rows(&self) -> &[Vec4d; 4] { bytemuck::cast_ref(self) }
    #[inline(always)] pub fn rows_mut(&mut self) -> &mut [Vec4d; 4] { bytemuck::cast_mut(self) }
    #[inline(always)] pub fn array(&self) -> &[f64; 16] { bytemuck::cast_ref(self) }
    #[inline(always)] pub fn array_mut(&mut self) -> &mut [f64; 16] { bytemuck::cast_mut(self) }
    #[inline(always)] pub fn dot(self, v: Vec4d) -> Vec4d { Vec4d::new(self.a.dot(v), self.b.dot(v), self.c.dot(v), self.d.dot(v)) }
    #[inline(always)] pub fn dot_t(self, v: Vec4d) -> Vec4d { Vec4d::new(self.a.x*v.x + self.b.x*v.y + self.c.x*v.z + self.d.x*v.w, self.a.y*v.x + self.b.y*v.y + self.c.y*v.z + self.d.y*v.w, self.a.z*v.x + self.b.z*v.y + self.c.z*v.z + self.d.z*v.w, self.a.w*v.x + self.b.w*v.y + self.c.w*v.z + self.d.w*v.w) }
    #[inline(always)] pub fn transpose(&mut self) { std::mem::swap(&mut self.a.y, &mut self.b.x); std::mem::swap(&mut self.a.z, &mut self.c.x); std::mem::swap(&mut self.a.w, &mut self.d.x); std::mem::swap(&mut self.b.z, &mut self.c.y); std::mem::swap(&mut self.b.w, &mut self.d.y); std::mem::swap(&mut self.c.w, &mut self.d.z); }
    #[inline(always)] pub fn transposed(self) -> Self { let mut m = self; m.transpose(); m }
}

#[inline(always)] pub fn mmul4(a: Mat4d, b: Mat4d) -> Mat4d {
    let bt = b.transposed();
    Mat4d::new(
        Vec4d::new(a.a.dot(bt.a), a.a.dot(bt.b), a.a.dot(bt.c), a.a.dot(bt.d)),
        Vec4d::new(a.b.dot(bt.a), a.b.dot(bt.b), a.b.dot(bt.c), a.b.dot(bt.d)),
        Vec4d::new(a.c.dot(bt.a), a.c.dot(bt.b), a.c.dot(bt.c), a.c.dot(bt.d)),
        Vec4d::new(a.d.dot(bt.a), a.d.dot(bt.b), a.d.dot(bt.c), a.d.dot(bt.d)),
    )
}

// =======================================================================================
// Mat4f (f32) — graphics pipeline matrices
// =======================================================================================

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Pod, Zeroable)]
pub struct Mat4f { pub a: Vec4f, pub b: Vec4f, pub c: Vec4f, pub d: Vec4f }

impl Mat4f {
    #[inline(always)] pub const fn new(a: Vec4f, b: Vec4f, c: Vec4f, d: Vec4f) -> Self { Self { a, b, c, d } }
    #[inline(always)] pub fn zero() -> Self { Self::new(VEC4F_ZERO, VEC4F_ZERO, VEC4F_ZERO, VEC4F_ZERO) }
    #[inline(always)] pub fn identity() -> Self { Self::new(Vec4f::new(1.0,0.0,0.0,0.0), Vec4f::new(0.0,1.0,0.0,0.0), Vec4f::new(0.0,0.0,1.0,0.0), Vec4f::new(0.0,0.0,0.0,1.0)) }
    #[inline(always)] pub fn rows(&self) -> &[Vec4f; 4] { bytemuck::cast_ref(self) }
    #[inline(always)] pub fn rows_mut(&mut self) -> &mut [Vec4f; 4] { bytemuck::cast_mut(self) }
    #[inline(always)] pub fn array(&self) -> &[f32; 16] { bytemuck::cast_ref(self) }
    #[inline(always)] pub fn array_mut(&mut self) -> &mut [f32; 16] { bytemuck::cast_mut(self) }

    /// Row-major 4×4 as nested arrays, matching `CameraData.view_proj` layout.
    #[inline(always)] pub fn to_arr4x4(self) -> [[f32; 4]; 4] {
        [*self.a.array(), *self.b.array(), *self.c.array(), *self.d.array()]
    }

    #[inline(always)] pub fn dot(self, v: Vec4f) -> Vec4f { Vec4f::new(self.a.dot(v), self.b.dot(v), self.c.dot(v), self.d.dot(v)) }
    #[inline(always)] pub fn dot_t(self, v: Vec4f) -> Vec4f { Vec4f::new(self.a.x*v.x + self.b.x*v.y + self.c.x*v.z + self.d.x*v.w, self.a.y*v.x + self.b.y*v.y + self.c.y*v.z + self.d.y*v.w, self.a.z*v.x + self.b.z*v.y + self.c.z*v.z + self.d.z*v.w, self.a.w*v.x + self.b.w*v.y + self.c.w*v.z + self.d.w*v.w) }

    #[inline(always)] pub fn transpose(&mut self) { std::mem::swap(&mut self.a.y, &mut self.b.x); std::mem::swap(&mut self.a.z, &mut self.c.x); std::mem::swap(&mut self.a.w, &mut self.d.x); std::mem::swap(&mut self.b.z, &mut self.c.y); std::mem::swap(&mut self.b.w, &mut self.d.y); std::mem::swap(&mut self.c.w, &mut self.d.z); }
    #[inline(always)] pub fn transposed(self) -> Self { let mut m = self; m.transpose(); m }

    /// Right-handed look-at view matrix (row-major, row-vector convention:
    /// clip = point * M). Upload directly to WGSL — the column-major byte
    /// interpretation in WGSL transposes it, so M_wgsl = M^T and M_wgsl * v = v * M.
    /// Do NOT additionally transpose before upload (that double-transposes).
    #[inline(always)] pub fn look_at(eye: Vec3f, target: Vec3f, up: Vec3f) -> Self {
        let mut z = eye - target;
        z.normalize();
        let mut x = up.cross(z);
        x.normalize();
        let y = z.cross(x);
        Self::new(
            Vec4f::new(x.x, y.x, z.x, 0.0),
            Vec4f::new(x.y, y.y, z.y, 0.0),
            Vec4f::new(x.z, y.z, z.z, 0.0),
            Vec4f::new(-x.dot(eye), -y.dot(eye), -z.dot(eye), 1.0),
        )
    }

    /// Orthographic projection for Vulkan/DirectX NDC Z in [0,1].
    #[inline(always)] pub fn ortho(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Self {
        Self::new(
            Vec4f::new(2.0/(r-l), 0.0,        0.0,         -(r+l)/(r-l)),
            Vec4f::new(0.0,       2.0/(t-b),  0.0,         -(t+b)/(t-b)),
            Vec4f::new(0.0,       0.0,        -1.0/(f-n),  -n/(f-n)),
            Vec4f::new(0.0,       0.0,        0.0,         1.0),
        )
    }
}

#[inline(always)] pub fn mmul4f(a: Mat4f, b: Mat4f) -> Mat4f {
    let bt = b.transposed();
    Mat4f::new(
        Vec4f::new(a.a.dot(bt.a), a.a.dot(bt.b), a.a.dot(bt.c), a.a.dot(bt.d)),
        Vec4f::new(a.b.dot(bt.a), a.b.dot(bt.b), a.b.dot(bt.c), a.b.dot(bt.d)),
        Vec4f::new(a.c.dot(bt.a), a.c.dot(bt.b), a.c.dot(bt.c), a.c.dot(bt.d)),
        Vec4f::new(a.d.dot(bt.a), a.d.dot(bt.b), a.d.dot(bt.c), a.d.dot(bt.d)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat3_identity() {
        let m = Mat3d::identity();
        let v = Vec3d::new(1.0, 2.0, 3.0);
        assert_eq!(m.dot(v), v);
    }

    #[test]
    fn mat3_det() {
        let m = Mat3d::new(Vec3d::new(1.0,2.0,3.0), Vec3d::new(4.0,5.0,6.0), Vec3d::new(7.0,8.0,10.0));
        assert!((m.det() + 3.0).abs() < 1e-12);
    }

    #[test]
    fn mat3_inverse() {
        let m = Mat3d::new(Vec3d::new(2.0,0.0,0.0), Vec3d::new(0.0,2.0,0.0), Vec3d::new(0.0,0.0,2.0));
        let inv = m.inverse();
        assert_eq!(mmul3(m, inv).a, Vec3d::new(1.0, 0.0, 0.0));
        assert_eq!(mmul3(m, inv).b, Vec3d::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn mat4_rows_view() {
        let m = Mat4d::identity();
        assert_eq!(m.rows()[0], Vec4d::new(1.0, 0.0, 0.0, 0.0));
    }
}
