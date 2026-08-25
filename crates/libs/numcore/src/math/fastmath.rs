#[inline(always)]
pub fn sq(a: f64) -> f64 { a * a }

#[inline(always)]
pub fn dangle(da: f64) -> f64 {
    if da > std::f64::consts::PI { da - 2.0 * std::f64::consts::PI } else if da < -std::f64::consts::PI { da + 2.0 * std::f64::consts::PI } else { da }
}

#[inline(always)]
pub fn clamp_abs(x: f64, xmax: f64) -> f64 {
    if x > 0.0 { if x > xmax { xmax } else { x } } else { if x < -xmax { -xmax } else { x } }
}

#[inline(always)]
pub fn sincos_taylor2(a: f64) -> (f64, f64) {
    const C2: f64 = 1.0 / 2.0;
    const C3: f64 = 1.0 / 6.0;
    const C4: f64 = 1.0 / 24.0;
    const C5: f64 = 1.0 / 120.0;
    let a2 = a * a;
    let sa = a * (1.0 - a2 * (C3 - C5 * a2));
    let ca = 1.0 - a2 * (C2 - C4 * a2);
    (sa, ca)
}

#[inline(always)]
pub fn sincos_r2_taylor(r2: f64) -> (f64, f64) {
    const C2: f64 = -1.0 / 2.0;
    const C3: f64 = -1.0 / 6.0;
    const C4: f64 = 1.0 / 24.0;
    const C5: f64 = 1.0 / 120.0;
    const C6: f64 = -1.0 / 720.0;
    let sa = 1.0 + r2 * (C3 + C5 * r2);
    let ca = C2 + r2 * (C4 + C6 * r2);
    (sa, ca)
}
