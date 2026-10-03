// rarff2d_dense — ~200 atoms on a hexagonal lattice (a=1.3, 70% fill, jitter)
// Realistic molecular-density stress test for the broad-phase accel.
// Field OFF by default (slow for many atoms) — press F to toggle on.
field(false);

let a  = 1.3;
let ny = 15;
let nx = 17;
for iy in 0..ny {
    let y = (iy - ny / 2.0) * a * 0.866;
    let xoff = if iy % 2 == 0 { 0.0 } else { 0.5 * a };
    for ix in 0..nx {
        if (ix * 3 + iy * 7) % 10 >= 7 { continue; }   // ~70% fill (deterministic)
        let x = (ix - nx / 2.0) * a + xoff;
        // deterministic pseudo-jitter ~±0.25 A
        let jx = 0.25 * sin((ix * 7 + iy * 13) * 1.0);
        let jy = 0.25 * cos((ix * 11 + iy * 5) * 1.0);
        atom(x + jx, y + jy, ((ix * 37 + iy * 61) % 120) * 1.0);
    }
}
arena(11.0, 0.5);
grid(6.0);
