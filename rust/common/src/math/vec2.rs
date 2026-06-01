/// 2D vector used for complex-number operations in angle/dihedral Fourier expansions.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Vec2d {
    pub x: f64,
    pub y: f64,
}

impl Vec2d {
    #[inline(always)] pub const fn new(x: f64, y: f64) -> Self { Self { x, y } }
    /// Complex multiply: self *= rhs interpreted as (a+ib)*(c+id)
    #[inline(always)] pub fn mul_cmplx(&mut self, rhs: Self) {
        let nx = self.x * rhs.x - self.y * rhs.y;
        let ny = self.x * rhs.y + self.y * rhs.x;
        self.x = nx; self.y = ny;
    }
    /// Complex division: self /= rhs
    #[inline(always)] pub fn udiv_cmplx(&mut self, rhs: Self) {
        let den = rhs.x * rhs.x + rhs.y * rhs.y;
        let nx = (self.x * rhs.x + self.y * rhs.y) / den;
        let ny = (self.y * rhs.x - self.x * rhs.y) / den;
        self.x = nx; self.y = ny;
    }
}

pub const VEC2_ZERO: Vec2d = Vec2d { x: 0.0, y: 0.0 };
