//! Uniform 3D cell grid over `Buckets` (count→prefix→scatter) — the generic
//! geometric half of the `Grid2d` cell-pair pattern (molff/rarff2d.rs) ported
//! to 3D. The forward half-stencil is FireCore `Buckets3D::getForwardNeighbors`
//! (RARFF_SR.h:interEF_buckets): for stencil radius s it enumerates each
//! unordered pair of cells with Chebyshev distance <= s exactly once.
//! Atoms outside the domain are clamped to the nearest boundary cell so every
//! object is always gridded.

use numtypes::Vec3d;
use crate::buckets::Buckets;

/// Uniform 3D neighbor grid: `cell = ix + nx*(iy + ny*iz)`, row-major.
/// Cell size dx >= rcut gives the classic 3x3x3 stencil (s=1, <=13 forward cells).
pub struct Grid3 {
    pub origin: [f64; 3],  // world coords of the (0,0,0) cell corner
    pub n: [usize; 3],     // cells per axis (nx, ny, nz)
    pub dx: f64,           // cell size [A]
    pub buckets: Buckets,
    pub cell_of: Vec<i32>, // object -> cell id (clamped into domain)
}

impl Grid3 {
    /// Allocate once for `nobj` objects over `n` cells of size `dx` from `origin`.
    pub fn new(origin: [f64; 3], n: [usize; 3], dx: f64, nobj: usize) -> Self {
        assert!(n[0] > 0 && n[1] > 0 && n[2] > 0, "Grid3::new: degenerate grid dims n={n:?}");
        assert!(dx > 0.0, "Grid3::new: dx={dx} must be > 0");
        Self { origin, n, dx, buckets: Buckets::new(n[0] * n[1] * n[2]), cell_of: vec![0; nobj] }
    }

    /// Object position -> clamped cell index (every object is always gridded).
    #[inline] pub fn cell_at(&self, p: Vec3d) -> i32 {
        let ix = ((p.x - self.origin[0]) / self.dx).floor().clamp(0.0, self.n[0] as f64 - 1.0) as i32;
        let iy = ((p.y - self.origin[1]) / self.dx).floor().clamp(0.0, self.n[1] as f64 - 1.0) as i32;
        let iz = ((p.z - self.origin[2]) / self.dx).floor().clamp(0.0, self.n[2] as f64 - 1.0) as i32;
        ix + self.n[0] as i32 * (iy + self.n[1] as i32 * iz)
    }

    /// Rebuild CSR for `pos` (len must equal cell_of.len()). O(N + ncells).
    pub fn rebuild(&mut self, pos: &[Vec3d]) {
        assert_eq!(pos.len(), self.cell_of.len(), "Grid3::rebuild: pos.len()={} != nobj={} — recreate the grid", pos.len(), self.cell_of.len());
        for (i, p) in pos.iter().enumerate() { self.cell_of[i] = self.cell_at(*p); }
        self.buckets.build(&self.cell_of);
    }

    /// 1D cell index -> 3D cell coords (ix, iy, iz).
    #[inline] pub fn unravel(&self, c: usize) -> [i64; 3] {
        let nx = self.n[0] as i64;
        let ny = self.n[1] as i64;
        let c = c as i64;
        [c % nx, (c / nx) % ny, c / (nx * ny)]
    }

    /// FORWARD half-stencil of cell `c` within Chebyshev radius `s`, clipped to
    /// the domain: same z-layer (dz=0) → same row dx>0 plus rows dy>0 (all dx
    /// in ±s); upper layers dz>0 → all (dx,dy) in ±s. For s=1 yields <=13 cells.
    /// Together with intra-cell i<j pairs this enumerates every unordered pair
    /// of cells with Chebyshev distance <= s exactly once.
    /// Ported from FireCore Buckets3D::getForwardNeighbors (via Grid2d eval).
    pub fn forward_cells(&self, c: usize, s: i64, out: &mut Vec<usize>) {
        out.clear();
        let [cx, cy, cz] = self.unravel(c);
        let (nx, ny, nz) = (self.n[0] as i64, self.n[1] as i64, self.n[2] as i64);
        for dz in 0..=s {
            let z = cz + dz;
            if z >= nz { break; }
            for dy in -s..=s {
                if dz == 0 && dy < 0 { continue; }
                let y = cy + dy;
                if y < 0 { continue; }
                if y >= ny { break; }
                let dx0 = if dz == 0 && dy == 0 { 1 } else { -s };
                for dx in dx0..=s {
                    let x = cx + dx;
                    if x < 0 { continue; }
                    if x >= nx { break; }
                    out.push((x + nx * (y + ny * z)) as usize);
                }
            }
        }
    }

    /// All unordered pairs (i,j), i<j, with |pos_j - pos_i| < rc.
    /// Rebuilds the buckets for `pos` first (call again per new geometry).
    /// Stencil radius s = ceil(rc/dx) covers any rc — for rc <= dx this is the
    /// classic 3x3x3 stencil; intra-cell i<j + forward-stencil cell pairs each
    /// enumerated exactly once. `out` is cleared first. Query fn — may allocate.
    pub fn pairs_within(&mut self, pos: &[Vec3d], rc: f64, out: &mut Vec<(u32, u32)>) {
        out.clear();
        assert!(rc.is_finite() && rc > 0.0, "Grid3::pairs_within: rc={rc} must be finite > 0");
        self.rebuild(pos);
        let s = (rc / self.dx).ceil() as i64;
        let rc2 = rc * rc;
        let mut fwd = Vec::new();
        for c in 0..self.buckets.ncells {
            let objs = self.buckets.cell_objects(c);
            for a in 0..objs.len() {                          // intra-cell i<j pairs
                let i = objs[a];
                for &j in &objs[a + 1..] {
                    if (pos[j as usize] - pos[i as usize]).norm2() < rc2 { out.push((i.min(j), i.max(j))); }
                }
            }
            self.forward_cells(c, s, &mut fwd);               // forward-stencil pairs
            for &hc in &fwd {
                for &j in self.buckets.cell_objects(hc) {
                    for &i in objs {
                        if (pos[j as usize] - pos[i as usize]).norm2() < rc2 { out.push((i.min(j), i.max(j))); }
                    }
                }
            }
        }
    }
}

/// One-shot auto-fit neighbor search: AABB of `pos` (+1e-6 margin), cells of
/// size dx=rc, n = ceil(extent/rc)+1 per axis (min 1), then Grid3::pairs_within.
/// Atoms landing outside the box are clamped into boundary cells (still found).
/// rc <= 0 or empty pos → empty out. Query fn — allocates a throwaway Grid3.
pub fn neighbors_within(pos: &[Vec3d], rc: f64, out: &mut Vec<(u32, u32)>) {
    out.clear();
    if pos.is_empty() || !(rc.is_finite() && rc > 0.0) { return; }
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in pos { for k in 0..3 { if p[k] < lo[k] { lo[k] = p[k]; } if p[k] > hi[k] { hi[k] = p[k]; } } }
    let mut n = [0usize; 3];
    for k in 0..3 { n[k] = (((hi[k] - lo[k] + 2e-6) / rc).ceil() as usize + 1).max(1); lo[k] -= 1e-6; }
    Grid3::new(lo, n, rc, pos.len()).pairs_within(pos, rc, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// THE correctness property: every unordered pair of distinct cells with
    /// Chebyshev distance <= s appears exactly once as (c, forward(c)).
    fn check_forward_stencil(n: [usize; 3], s: i64) {
        let g = Grid3::new([0.0; 3], n, 1.0, 0);
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut out = Vec::new();
        let ncells = n[0] * n[1] * n[2];
        for c in 0..ncells {
            g.forward_cells(c, s, &mut out);
            let cc = g.unravel(c);
            for &h in &out {
                assert_ne!(h, c, "forward_cells({c}) contains self");
                let hh = g.unravel(h);
                let cheb = (0..3).map(|k| (cc[k] - hh[k]).abs()).max().unwrap();
                assert!(cheb <= s && cheb > 0, "forward_cells({c}) -> {h}: cheb={cheb} not in 1..={s}");
                assert!(seen.insert((c, h)), "pair ({c},{h}) duplicated");
            }
        }
        // completeness: enumerate all unordered pairs independently
        let mut expect = 0usize;
        for a in 0..ncells {
            for b in (a + 1)..ncells {
                let (ca, cb) = (g.unravel(a), g.unravel(b));
                let cheb = (0..3).map(|k| (ca[k] - cb[k]).abs()).max().unwrap();
                if cheb <= s { expect += 1; assert!(seen.contains(&(a, b)) || seen.contains(&(b, a)), "pair ({a},{b}) missing"); }
            }
        }
        assert_eq!(seen.len(), expect, "stencil pair count {} != expected {}", seen.len(), expect);
    }

    #[test]
    fn test_forward_stencil_3x3x3() { check_forward_stencil([3, 3, 3], 1); }
    #[test]
    fn test_forward_stencil_4x3x2() { check_forward_stencil([4, 3, 2], 1); }
    #[test]
    fn test_forward_stencil_s2() { check_forward_stencil([5, 4, 3], 2); }

    #[test]
    fn test_forward_count_interior() {
        let g = Grid3::new([0.0; 3], [9, 9, 9], 1.0, 0);
        let mut out = Vec::new();
        g.forward_cells(4 + 9 * (4 + 9 * 4), 1, &mut out); // interior cell (4,4,4)
        assert_eq!(out.len(), 13, "interior forward stencil should be 13 cells, got {}", out.len());
    }

    #[test]
    fn test_cell_at_clamping() {
        let g = Grid3::new([0.0, 0.0, 0.0], [4, 3, 2], 1.0, 0);
        assert_eq!(g.cell_at(Vec3d::new(-5.0, -1.0, -9.0)), 0);
        assert_eq!(g.cell_at(Vec3d::new(99.0, 99.0, 99.0)), (4 * 3 * 2 - 1) as i32);
        assert_eq!(g.cell_at(Vec3d::new(1.5, 0.5, 0.5)), 1);
        assert_eq!(g.cell_at(Vec3d::new(0.5, 2.9, 1.9)), 0 + 4 * (2 + 3 * 1));
    }

    /// Deterministic LCG (no rand dep) — same as molff tests.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 11) as f64) * (1.0 / 9007199254740992.0) }
    }

    /// Brute-force O(N^2) reference: all i<j with dist < rc.
    fn brute_pairs(pos: &[Vec3d], rc: f64) -> HashSet<(u32, u32)> {
        let mut s = HashSet::new();
        let rc2 = rc * rc;
        for i in 0..pos.len() {
            for j in (i + 1)..pos.len() {
                if (pos[j] - pos[i]).norm2() < rc2 { s.insert((i as u32, j as u32)); }
            }
        }
        s
    }

    #[test]
    fn test_pairs_within() {
        let mut rng = Lcg(0xB5297A4D);
        let pos: Vec<Vec3d> = (0..200).map(|_| Vec3d::new(rng.next() * 8.0, rng.next() * 8.0, rng.next() * 8.0)).collect();
        // fixed grid dx=1.0, [0,8)^3; rc below / at / above dx
        for &rc in &[0.5, 1.0, 2.3] {
            let mut g = Grid3::new([0.0; 3], [8, 8, 8], 1.0, pos.len());
            let mut out = Vec::new();
            g.pairs_within(&pos, rc, &mut out);
            let got: HashSet<(u32, u32)> = out.iter().copied().collect();
            let want = brute_pairs(&pos, rc);
            assert_eq!(out.len(), got.len(), "rc={rc}: duplicate pairs emitted");
            assert_eq!(got, want, "rc={rc}: grid pairs != brute-force pairs (got {} want {})", got.len(), want.len());
        }
        // auto-fit free fn — same parity, atoms at box edges must still be found
        for &rc in &[0.5, 1.0, 2.3] {
            let mut out = Vec::new();
            neighbors_within(&pos, rc, &mut out);
            let got: HashSet<(u32, u32)> = out.iter().copied().collect();
            assert_eq!(got, brute_pairs(&pos, rc), "neighbors_within rc={rc}: != brute force");
        }
    }

    #[test]
    fn test_rebuild_groups() {
        let mut g = Grid3::new([0.0; 3], [2, 2, 2], 1.0, 4);
        let pos = vec![
            Vec3d::new(0.2, 0.2, 0.2), Vec3d::new(1.7, 0.3, 0.4),
            Vec3d::new(0.3, 1.6, 1.7), Vec3d::new(0.4, 0.4, 0.4),
        ];
        g.rebuild(&pos);
        let c0: HashSet<u32> = g.buckets.cell_objects(0).iter().copied().collect();
        assert_eq!(c0, [0, 3].into_iter().collect());
        assert_eq!(g.buckets.cell_objects(g.cell_at(pos[1]) as usize), &[1]);
        assert_eq!(g.buckets.cell_objects(g.cell_at(pos[2]) as usize), &[2]);
    }
}
