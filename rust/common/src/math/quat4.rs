use crate::math::vec3::Vec3d;

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Quat4d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Quat4d {
    #[inline(always)] pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self { Self { x, y, z, w } }
    #[inline(always)] pub fn f(self) -> Vec3d { Vec3d::new(self.x, self.y, self.z) }
}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Quat4i {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
}

impl Quat4i {
    #[inline(always)] pub const fn new(x: i32, y: i32, z: i32, w: i32) -> Self { Self { x, y, z, w } }
    #[inline(always)] pub fn as_array(&self) -> [i32; 4] { [self.x, self.y, self.z, self.w] }
}

pub const QUAT4I_MINUS_ONES: Quat4i = Quat4i { x: -1, y: -1, z: -1, w: -1 };
