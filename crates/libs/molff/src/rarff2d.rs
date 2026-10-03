//! RARFF-2D — rod-free reactive angular-gated Morse forcefield (2D, f64).
//!
//! Concept ported from FireCore `cpp/common/molecular/RARFF_SR.h::pairEF`,
//! simplified to 2D and made rod-free: each atom carries orientation as a
//! normalized complex number `zc = (cos φ, sin φ)` instead of a quaternion +
//! port-direction template. The n-fold Lorentzian angular gate replaces the
//! port-pair alignment product (ci·cj·cij)^4 — bonds emerge from geometry,
//! no bond list is needed.
//!
//! Pair energy (d = x_j - x_i, r = |d|, θ = atan2(d.y, d.x)):
//!   e      = exp(-b·(r - r0)),  r0 = r_i + r_j  (covalent radii)
//!   Y      = g_i(Δ_i)·g_j(Δ_j),  Δ_i = θ - φ_i,  Δ_j = θ + π - φ_j
//!   g(Δ)   = 1/(1 + (sin(n·Δ/2)/w)²)            — peaks at port dirs
//!   E      = a·(e² - 2·e·Y)                      — repulsive wall ungated,
//!            attractive well gated by BOTH orientations
//!
//! Combine rules (per RARFF_SR::RigidAtomType::combine): r0 = r_i + r_j,
//! b = b_i + b_j, a = a_i·a_j.
//!
//! Consistency readout: `eatom[i]` accumulates half of each pair energy —
//! unsatisfied atoms stay high → per-atom inconsistency map for structure
//! repair (see invPPAFM export_invAFM/surfmol_forcefields.md).

use numtypes::{cmul, cconj, Vec2d};
use spacc::buckets::Buckets;

/// Per-atom type: fold symmetry + covalent radius + Morse params + gate width.
#[derive(Copy, Clone, Debug)]
pub struct Rarff2dType { pub nfold: u32, pub r_cov: f64, pub a: f64, pub b: f64, pub w: f64 }

/// sp2-like trigonal atom, r_cov=0.65 A → C_ar–C_ar bond = 1.3 A, pair b = 1.8.
pub const TYPE_SP2: Rarff2dType = Rarff2dType { nfold: 3, r_cov: 0.65, a: 1.0, b: 0.9, w: 0.3 };
/// sp3-like tetragonal atom (2D projection).
pub const TYPE_SP3: Rarff2dType = Rarff2dType { nfold: 4, r_cov: 0.77, a: 1.0, b: 0.9, w: 0.3 };

/// Ring entity — external annular field for ring repair (n=6 hexagon, n=5 pentagon).
/// E = -a·B(|x_k - p_c| - R)·(1 + cos(n·(φ_k - φ_c)))/2 + a_core·pex(-b_core·(r-r0_core))²
/// Sites: radial bump (1-ρ²/h²)⁴ x n-fold site gate. Core: poly-Morse repulsive
/// wall centered on the ring hole (atoms can't sit at ring center); a_core=0 off.
/// Ring center/orientation are DOFs too — reaction force/torque accumulated into
/// ring_f/ring_t so the ring can be relaxed or scored like an atom.
#[derive(Copy, Clone, Debug)]
pub struct Ring2d { pub pos: Vec2d, pub zc: Vec2d, pub nfold: u32, pub radius: f64, pub a: f64, pub h: f64, pub a_core: f64, pub r0_core: f64, pub b_core: f64 }

/// Compact polynomial bump B(ρ) = (1 - ρ²/h²)⁴ for |ρ| ≤ h — polynomial "Gaussian".
/// C³ at edges (vanishes with derivatives to order 3), peak B=1 at ρ=0.
/// Evaluated by squaring: u = (ρ/h)², q = 1-u, B = q²·q² — ~4 multiplies.
/// Returns (B, dB/dρ) with dB = -(8ρ/h²)·q³; zero outside support.
#[inline(always)]
pub fn bump(rho: f64, h: f64) -> (f64, f64) {
    let ih2 = 1.0 / (h * h);
    let q = 1.0 - rho * rho * ih2;
    if q <= 0.0 { return (0.0, 0.0); }
    let q2 = q * q;
    (q2 * q2, -8.0 * rho * ih2 * q * q2)
}

/// Fast polynomial exp: pex(x) = (1 + x/N)^N → e^x as N→∞ (here N=2^5=32).
/// Evaluated by M squarings; compact support: returns (0,0) for 1+x/N ≤ 0.
/// Returns (e, de/dx) with de/dx = e/s, s = 1+x/N.
/// Rel. error vs exp ~ x²/(2N): ~3% at |x|=1.4, ~14% at |x|=3 (acceptable for FF).
/// Polynomial-exp squaring count m — THE CUTOFF KNOB.
/// `pex(x) = (1 + x/2^m)^(2^m)` computed as m repeated squarings:
///   `u = 1 + x/2^m; for k in 0..m { u *= u }`   → u^(2^m), no pow(), no exp().
/// Exact zero at x = −2^m i.e. support ends at **r = r0 + 2^m/b** — lower m
/// gives shorter TRUE compact support AND cheaper eval (m muls). This is the
/// point of the pexp design: finite support by construction, no taper.
/// m=3 → 2^3=8 → 5.7 A; m=4 → 10.2 A; m=5 → 19.1 A (r0=1.3, pair b=1.8).
/// Smaller m makes the well more abrupt (needs smaller dt); well near r0 is
/// preserved — approx error only reshapes the deep tail where |E| is tiny.
/// See debug/rarff2d_cutoff/cutoff.png (scripts/plot_rarff2d_cutoff.py).
pub const PEXP_M: u32 = 3;
#[inline(always)]
pub fn pex(x: f64, m: u32) -> (f64, f64) {
    let n = (1u64 << m) as f64;
    let s = 1.0 + x / n;
    if s <= 0.0 { return (0.0, 0.0); }
    let mut e = s;
    for _ in 0..m { e = e * e; }        // m squarings → s^(2^m)
    (e, e / s)                          // de/dx = s^(2^m - 1) = e/s
}

/// Lorentzian angular gate and its derivative dg/dΔ.
/// g = 1/(1+u²), u = sin(n·Δ/2)/w → zeros of u at ports → g=1 there;
/// u ≈ (n/2)Δ near ports → g ≈ 1 - (n·Δ/2w)² quadratic (not quartic like
/// the (1-cos) form) → finite restoring torque near alignment.
/// dg = -2 g²·u·(n/2·cos(nΔ/2))/w. 3-fold: u(Δ+2π/n) = -u(Δ) → g same.
#[inline(always)]
pub fn gate(nfold: u32, w: f64, d: f64) -> (f64, f64) {
    let n = nfold as f64;
    let u = (0.5 * n * d).sin() / w;
    let g = 1.0 / (1.0 + u * u);
    (g, -2.0 * g * g * u * (0.5 * n * (0.5 * n * d).cos()) / w)
}

/// Complex square root of a unit complex c = e^{iΔ} → e^{iΔ/2}, principal
/// branch (√ at Δ=±π maps to +i, matching atan2's upper half-plane cut).
/// Exact algebra: sqrt((1+x)/2), sign(y)·sqrt((1-x)/2).
#[inline(always)]
fn csqrt(c: Vec2d) -> Vec2d {
    // clamp: |c| can drift past 1 by ~1e-16 (fp error in unit-complex product)
    // → sqrt(<0) → NaN seeds the whole system
    let re = (0.5 * (1.0 + c.x)).max(0.0).sqrt();
    let im = (0.5 * (1.0 - c.x)).max(0.0).sqrt().copysign(c.y);
    Vec2d::new(re, im)
}

/// (sin, cos) of n·Δ/2 from unit complex c = e^{iΔ} — NO atan2/sin/cos calls.
/// e^{i·nΔ/2} = c^{n/2}: n=1→√c, 2→c, 3→c·√c, 4→c², 6→c³.
#[inline(always)]
fn phase_half(c: Vec2d, nfold: u32) -> (f64, f64) {
    let q = match nfold {
        1 => csqrt(c),
        3 => cmul(c, csqrt(c)),
        4 => cmul(c, c),
        6 => cmul(cmul(c, c), c),
        _ => c,                                    // nfold=2 (default sp3-like)
    };
    (q.y, q.x)
}

/// Gate from precomputed (sin, cos) of n·Δ/2 — same math as `gate`, no trig.
#[inline(always)]
fn gate_cs(nfold: u32, w: f64, sn: f64, cn: f64) -> (f64, f64) {
    let n = nfold as f64;
    let u = sn / w;
    let g = 1.0 / (1.0 + u * u);
    (g, -2.0 * g * g * u * (0.5 * n * cn) / w)
}

/// Uniform 2D neighbor grid over `Buckets` (count→prefix→scatter).
/// Cell size dx: ~1 A cells match the U-Net pixel map (finer than needed);
/// dx >= rcut gives the classic 3x3-cell stencil (fewer segments, more atoms
/// scanned). Atoms outside the domain are clamped to the nearest boundary
/// cell so every atom is always gridded.
pub struct Grid2d {
    pub origin: [f64; 2],   // world coords of cell (0,0) corner
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,            // cell size [A]
    pub buckets: Buckets,
    pub cell_of: Vec<i32>,  // atom -> cell id (clamped into domain)
    // Halo gather scratch — the RRsp3 ghost-list pattern: contiguous copies of
    // halo atoms so the inner loop streams instead of random-gathering 6 SoA
    // arrays. Allocated ONCE in set_grid (never in the hot loop); len=halo.
    pub h_idx: Vec<u32>, pub h_pos: Vec<Vec2d>, pub h_zc: Vec<Vec2d>, pub h_ty: Vec<Rarff2dType>,
    pub h_f: Vec<Vec2d>, pub h_t: Vec<f64>, pub h_e: Vec<f64>,
}

impl Grid2d {
    /// Atom position -> clamped cell index (row-major cx + nx*cy).
    #[inline] pub fn cell_at(&self, p: Vec2d) -> i32 {
        let cx = ((p.x - self.origin[0]) / self.dx).floor().clamp(0.0, self.nx as f64 - 1.0) as i32;
        let cy = ((p.y - self.origin[1]) / self.dx).floor().clamp(0.0, self.ny as f64 - 1.0) as i32;
        cx + self.nx as i32 * cy
    }

    /// Rebuild CSR for `pos` (len must equal cell_of.len()). O(N + nx*ny).
    pub fn rebuild(&mut self, pos: &[Vec2d]) {
        assert_eq!(pos.len(), self.cell_of.len(), "Grid2d::rebuild: natom changed — call set_grid() again");
        for (i, p) in pos.iter().enumerate() { self.cell_of[i] = self.cell_at(*p); }
        self.buckets.build(&self.cell_of);
    }
}

/// Dynamic collision groups — topology-free alternative to FireCore's
/// molecule-cluster AABBs. Topology here is REACTIVE (no fixed bonds), so
/// groups are rebuilt from positions alone: each atom joins the nearest group
/// COG within `r_join`; otherwise it spawns a new group. Then each group gets
/// a position-fitted AABB and group pairs are pruned by AABB+rcut overlap
/// (`broad_phase_pairs` pattern from spacc, mirrored inline).
/// Rebuild is O(N·G) — call on a period, not every eval; stale groups NEVER
/// break correctness (AABBs refit each rebuild), only culling efficiency.
pub struct Groups2d {
    pub r_join: f64,        // max distance to join an existing group's COG
    pub g_of: Vec<i32>,     // atom -> group id
    pub cog: Vec<Vec2d>,    // group center of gravity (last rebuild)
    pub buckets: Buckets,
    pub aabb: Vec<[f64; 4]>,// per-group [xmin, ymin, xmax, ymax]
}

impl Groups2d {
    /// Assign atoms to nearest-COG groups (r_join) and refit AABBs. O(N·G).
    pub fn rebuild(&mut self, pos: &[Vec2d]) {
        assert_eq!(pos.len(), self.g_of.len(), "Groups2d::rebuild: natom changed — call set_groups() again");
        self.cog.clear();
        self.g_of.iter_mut().for_each(|g| *g = -1);
        for (i, p) in pos.iter().enumerate() {
            let mut best = -1i32; let mut best_d2 = self.r_join * self.r_join;
            for (g, c) in self.cog.iter().enumerate() {
                let d2 = (*p - *c).norm2();
                if d2 < best_d2 { best_d2 = d2; best = g as i32; }
            }
            if best < 0 {
                best = self.cog.len() as i32;
                assert!((best as usize) < self.buckets.ncells, "Groups2d::rebuild: >max_groups ({}) needed", self.buckets.ncells);
                self.cog.push(*p);
            } else {
                self.cog[best as usize] = self.cog[best as usize] * 0.9 + *p * 0.1;  // running avg keeps COG stable
            }
            self.g_of[i] = best;
        }
        self.buckets.build(&self.g_of);
        for a in &mut self.aabb { *a = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY]; }
        for (i, p) in pos.iter().enumerate() {
            let a = &mut self.aabb[self.g_of[i] as usize];
            a[0] = a[0].min(p.x); a[1] = a[1].min(p.y); a[2] = a[2].max(p.x); a[3] = a[3].max(p.y);
        }
    }

    /// AABB overlap with margin (same test as numtypes::aabb_overlap_margin).
    #[inline] pub fn overlap(&self, g1: usize, g2: usize, margin: f64) -> bool {
        let (a, b) = (&self.aabb[g1], &self.aabb[g2]);
        a[0] <= b[2] + margin && a[2] + margin >= b[0] && a[1] <= b[3] + margin && a[3] + margin >= b[1]
    }

    #[inline] pub fn ngroup(&self) -> usize { self.cog.len() }
}

/// 2D reactive-atom system. SoA layout; `force`/`torq`/`eatom` are scratch
/// recomputed by `eval()` each call.
pub struct Rarff2d {
    pub types: Vec<Rarff2dType>,
    pub pos: Vec<Vec2d>,     // x_i
    pub zc: Vec<Vec2d>,      // unit complex orientation (cos φ_i, sin φ_i)
    pub vel: Vec<Vec2d>,
    pub om: Vec<f64>,        // angular velocity (scalar in 2D)
    pub force: Vec<Vec2d>,   // scratch
    pub torq: Vec<f64>,      // scratch
    pub eatom: Vec<f64>,     // scratch: per-atom accumulated energy (consistency map)
    pub inv_m: Vec<f64>,     // inverse translational mass (1 = relaxation mode)
    pub inv_i: Vec<f64>,     // inverse scalar inertia
    pub rcut: f64,
    pub rings: Vec<Ring2d>,  // ring entities (hexagon/pentagon constraints)
    pub ring_f: Vec<Vec2d>,  // scratch: reaction force on ring centers
    pub ring_t: Vec<f64>,    // scratch: torque on ring orientations
    /// Optional confinement arena (r_arena [A], k [eV/A²]): soft quadratic wall
    /// E = k·(r - R)² for |x_i| > R — keeps unbound atoms from evaporating
    /// in self-assembly demos (a "beaker"), zero effect inside.
    pub arena: Option<(f64, f64)>,
    /// Optional neighbor grid; when `Some`, `eval()` uses the grid pair loop
    /// (must produce identical forces/energy as the O(N²) loop — parity-tested).
    pub grid: Option<Grid2d>,
    /// Optional coarse-cell AABB groups; when `Some` (and no `grid`), `eval()`
    /// uses the group-AABB pruned pair loop (same parity guarantee).
    pub groups: Option<Groups2d>,
    /// Debug counters: pair_ef calls and in-cutoff evals since last eval()
    /// start. Cheap instrumentation — proves where the time actually goes.
    pub dbg_npair: usize,
    pub dbg_ninside: usize,
    /// Debug per-pass timing [intra, gather, inner, scatter] (sec), when
    /// dbg_timing is set. Reset each eval.
    pub dbg_t: [f64; 4],
    pub dbg_timing: bool,
    /// Squaring count of pex — controls BOTH approximation quality and the
    /// true compact support (r0 + 2^m/b). Change via set_pex_m() (recomputes
    /// rcut); default PEXP_M.
    pub pex_m: u32,
}

/// Max pair support over the unique (r_cov, b) type pairs: r0 + 2^m/(b_i+b_j).
/// This IS the interaction cutoff — the potential is identically 0 beyond it.
fn pair_support_max(types: &[Rarff2dType], m: u32) -> f64 {
    let nsup = (1u64 << m) as f64;
    let mut ubt: Vec<(f64, f64)> = Vec::new();              // unique (r_cov, b)
    for t in types { let k = (t.r_cov, t.b); if !ubt.contains(&k) { ubt.push(k); } }
    let mut rcut: f64 = 0.0;
    for &(ri, bi) in &ubt { for &(rj, bj) in &ubt { rcut = rcut.max(ri + rj + nsup / (bi + bj)); } }
    assert!(rcut > 0.0, "pair_support_max: no types / degenerate b");
    rcut
}

impl Rarff2d {
    /// `rcut` is auto-derived: max pair support r0 + 2^pex_m/(b_i+b_j).
    pub fn new(types: Vec<Rarff2dType>, pos: Vec<Vec2d>) -> Self {
        let n = pos.len();
        assert_eq!(types.len(), n, "Rarff2d::new: types.len()={} != natoms={}", types.len(), n);
        let rcut = pair_support_max(&types, PEXP_M);
        Self { types, pos,
            zc: vec![Vec2d::new(1.0, 0.0); n], vel: vec![Vec2d::new(0.0, 0.0); n], om: vec![0.0; n],
            force: vec![Vec2d::new(0.0, 0.0); n], torq: vec![0.0; n], eatom: vec![0.0; n],
            inv_m: vec![1.0; n], inv_i: vec![1.0; n], rcut,
            rings: Vec::new(), ring_f: Vec::new(), ring_t: Vec::new(), arena: None, grid: None, groups: None,
            dbg_npair: 0, dbg_ninside: 0, dbg_t: [0.0; 4], dbg_timing: false, pex_m: PEXP_M }
    }

    /// Change the pexp squaring count — recomputes rcut (support r0+2^m/b).
    pub fn set_pex_m(&mut self, m: u32) {
        assert!(m >= 1 && m <= 10, "set_pex_m: m={m} out of sane range 1..10");
        self.pex_m = m;
        self.rcut = pair_support_max(&self.types, m);
    }

    #[inline] pub fn natom(&self) -> usize { self.pos.len() }
    #[inline] pub fn phi(&self, i: usize) -> f64 { self.zc[i].y.atan2(self.zc[i].x) }
    #[inline] pub fn set_orient(&mut self, i: usize, phi: f64) { self.zc[i].set(phi.cos(), phi.sin()); }

    /// Pair energy/force/torque for atoms (i,j); accumulates into scratch arrays.
    /// Returns pair energy (0 outside rcut).
    pub fn pair_ef(&mut self, i: usize, j: usize) -> f64 {
        let (e_pair, fx, fy, taui, tauj) = self.pair_math(self.types[i], self.zc[i], self.pos[i], self.types[j], self.zc[j], self.pos[j]);
        self.force[j].x -= fx; self.force[j].y -= fy;
        self.force[i].x += fx; self.force[i].y += fy;
        self.torq[i] += taui; self.torq[j] += tauj;
        self.eatom[i] += 0.5 * e_pair; self.eatom[j] += 0.5 * e_pair;
        e_pair
    }

    /// Pure pair math — the single source of truth for the pair interaction.
    /// Returns (e_pair, f_on_i.x, f_on_i.y, tau_i, tau_j); all zero outside rcut.
    /// Orientations passed as complex z; φ recovered by atan2 (same as phi()).
    #[inline]
    pub fn pair_math(&mut self, ti: Rarff2dType, zi: Vec2d, pi: Vec2d, tj: Rarff2dType, zj: Vec2d, pj: Vec2d) -> (f64, f64, f64, f64, f64) {
        if self.dbg_timing { self.dbg_npair += 1; }
        let d = pj - pi;
        let r2 = d.norm2();
        if r2 > self.rcut * self.rcut || r2 < 1e-12 { return (0.0, 0.0, 0.0, 0.0, 0.0); }
        if self.dbg_timing { self.dbg_ninside += 1; }
        let r = r2.sqrt();
        let hij = d * (1.0 / r);
        let r0 = ti.r_cov + tj.r_cov;
        let b = ti.b + tj.b;
        let a = ti.a * tj.a;
        let (e, dedx) = pex(-b * (r - r0), self.pex_m);   // e = pexp(-b·Δr), de/dr = -b·(e/s) = -b·dedx
        // atan2-free gates: e^{iΔi} = hij·conj(zi), e^{iΔj} = -hij·conj(zj)
        let ci = cmul(hij, cconj(zi));
        let cj = cmul(hij, Vec2d::new(-zj.x, zj.y));
        let (si, cci) = phase_half(ci, ti.nfold);
        let (sj, ccj) = phase_half(cj, tj.nfold);
        let (gi, dgi) = gate_cs(ti.nfold, ti.w, si, cci);
        let (gj, dgj) = gate_cs(tj.nfold, tj.w, sj, ccj);
        let y = gi * gj;
        let e_pair = a * (e * e - 2.0 * e * y);
        // radial: dE/dr = a·(2e - 2Y)·de/dr, de/dr = -b·dedx  →  dedr = 2ab·dedx·(Y - e)
        let dedr = 2.0 * a * b * dedx * (y - e);
        // angular: dE/dθ = -2a·e·(g_i'·g_j + g_i·g_j')  (dΔ_i/dθ = dΔ_j/dθ = +1)
        let dedth = -2.0 * a * e * (dgi * gj + gi * dgj);
        // ∇_j E = dedr·hij + dedth·perp/r,  perp = (-sinθ, cosθ) = (-hij.y, hij.x)
        let fx = dedr * hij.x + dedth * (-hij.y) / r;
        let fy = dedr * hij.y + dedth * (hij.x) / r;
        // τ_i = -dE/dφ_i = -2a·e·g_i'·g_j ;  τ_j = -2a·e·g_i·g_j'   (dΔ/dφ = -1)
        (e_pair, fx, fy, -2.0 * a * e * dgi * gj, -2.0 * a * e * gi * dgj)
    }

    /// Ring-atom energy/force for ring `ir` vs atom `i`; accumulates into
    /// force[i], ring_f[ir], ring_t[ir], eatom[i]. Returns energy.
    /// Δ = atan2(d.y,d.x) - φ_c ; E = -a·B(r-R)·(1+cos(n·Δ))/2
    pub fn ring_ef(&mut self, ir: usize, i: usize) -> f64 {
        let ring = self.rings[ir];
        let d = self.pos[i] - ring.pos;
        let r2 = d.norm2();
        // support: max(site annulus R+h, poly-exp core tail r0_core + N/b_core)
        let nsup = (1u64 << self.pex_m) as f64;
        let rmax = if ring.a_core != 0.0 { (ring.r0_core + nsup / ring.b_core).max(ring.radius + ring.h) } else { ring.radius + ring.h };
        if r2 > rmax * rmax || r2 < 1e-12 { return 0.0; }
        let r = r2.sqrt();
        let hij = d * (1.0 / r);
        let mut e = 0.0f64;
        let mut dedr = 0.0f64;
        let mut dedph = 0.0f64;
        // core repulsion: E_c = a_core·e², e = pex(-b_core·(r-r0_core))  (poly-Morse wall)
        if ring.a_core != 0.0 {
            let (ec, dec) = pex(-ring.b_core * (r - ring.r0_core), self.pex_m);
            if ec > 0.0 {
                e += ring.a_core * ec * ec;
                dedr += -2.0 * ring.a_core * ring.b_core * dec * ec;   // <0 → pushes atom out
            }
        }
        // site attraction: E_s = -a·B(r-R)·C(φ)
        let (bb, db) = bump(r - ring.radius, ring.h);
        if bb != 0.0 || db != 0.0 {
            let phi_c = ring.zc.y.atan2(ring.zc.x);
            let dl = d.y.atan2(d.x) - phi_c;
            let n = ring.nfold as f64;
            let c = 0.5 * (1.0 + (n * dl).cos());
            let dc = -0.5 * n * (n * dl).sin();
            e += -ring.a * bb * c;
            dedr += -ring.a * db * c;
            dedph += -ring.a * bb * dc;
            // τ_c = -dE/dφ_c ; dφ/dφ_c = -1 → τ_c = +dE/dφ  (core has no φ dep)
            self.ring_t[ir] += dedph;
        } else if e == 0.0 { return 0.0; }
        let fx = dedr * hij.x + dedph * (-hij.y) / r;
        let fy = dedr * hij.y + dedph * (hij.x) / r;
        self.force[i].x -= fx; self.force[i].y -= fy;
        self.ring_f[ir].x += fx; self.ring_f[ir].y += fy;  // reaction on ring center
        self.eatom[i] += e;
        e
    }

    /// Configure ring entities; allocates ring scratch ONCE (reconfigure step, not hot path).
    pub fn set_rings(&mut self, rings: Vec<Ring2d>) {
        let nr = rings.len();
        self.rings = rings;
        self.ring_f = vec![Vec2d::new(0.0, 0.0); nr];
        self.ring_t = vec![0.0; nr];
    }

    /// Configure the uniform neighbor grid (reconfigure step — allocates once).
    /// `dx` = cell size [A]; dx>=rcut → classic 3x3 stencil (skips most pairs);
    /// dx~1 A → U-Net pixel map. Domain covers origin + nx*dx × ny*dx.
    /// Call again whenever natom changes.
    pub fn set_grid(&mut self, dx: f64, origin: [f64; 2], nx: usize, ny: usize) {
        let n = self.natom();
        self.grid = Some(Grid2d {
            origin, nx, ny, dx,
            buckets: Buckets::new(nx * ny), cell_of: vec![0; n],
            h_idx: Vec::with_capacity(n), h_pos: Vec::with_capacity(n), h_zc: Vec::with_capacity(n), h_ty: Vec::with_capacity(n),
            h_f: Vec::with_capacity(n), h_t: Vec::with_capacity(n), h_e: Vec::with_capacity(n),
        });
        self.groups = None;
    }

    /// Configure dynamic COG collision groups (reconfigure step — allocates once).
    /// `r_join` = max atom→group-COG distance (use ~rcut); `max_groups` = capacity.
    /// Call again whenever natom changes.
    pub fn set_groups(&mut self, r_join: f64, max_groups: usize) {
        let n = self.natom();
        self.groups = Some(Groups2d { r_join, g_of: vec![-1; n], cog: Vec::with_capacity(max_groups),
            buckets: Buckets::new(max_groups), aabb: vec![[0.0; 4]; max_groups] });
        self.grid = None;
    }

    /// Zero scratch, evaluate pairs + ring-atom terms + arena, return total energy.
    /// With `grid` set: CSR neighbor loop (stencil ⌈rcut/dx⌉ cells, pairs i<j);
    /// without: O(N²) all-pairs. Both paths must give identical results.
    pub fn eval(&mut self) -> f64 {
        assert_eq!(self.ring_f.len(), self.rings.len(), "eval: ring scratch stale — call set_rings() after modifying rings");
        self.dbg_npair = 0; self.dbg_ninside = 0; self.dbg_t = [0.0; 4];
        for i in 0..self.natom() { self.force[i].set(0.0, 0.0); self.torq[i] = 0.0; self.eatom[i] = 0.0; }
        let nr = self.rings.len();
        for ir in 0..nr { self.ring_f[ir].set(0.0, 0.0); self.ring_t[ir] = 0.0; }
        let mut e = 0.0;
        if let Some(mut grid) = self.grid.take() {
            // CELL-PAIR loop with halo gather (RRsp3 ghost-list pattern):
            // per non-empty cell c — intra-cell pairs direct, then FORWARD-stencil
            // halo atoms copied into contiguous scratch so the hot loop streams
            // instead of random-gathering 6 SoA arrays; j-contributions are
            // accumulated in scratch and scattered back once per cell.
            grid.rebuild(&self.pos);
            let s = (self.rcut / grid.dx).ceil() as i64;
            let nx = grid.nx as i64;
            let ny = grid.ny as i64;
            let mut t0 = std::time::Instant::now();
            for c in 0..grid.buckets.ncells as i64 {
                let i0 = grid.buckets.offsets[c as usize] as usize;
                let i1 = grid.buckets.offsets[c as usize + 1] as usize;
                if i0 == i1 { continue; }
                let (ci, cj) = (c % nx, c / nx);
                // pass 1: intra-cell pairs (i<j within c)
                for a in i0..i1 {
                    let ia = grid.buckets.items[a] as usize;
                    for b in (a + 1)..i1 {
                        let ib = grid.buckets.items[b] as usize;
                        let (ep, fx, fy, taui, tauj) = self.pair_math(self.types[ia], self.zc[ia], self.pos[ia], self.types[ib], self.zc[ib], self.pos[ib]);
                        self.force[ia].x += fx; self.force[ia].y += fy;
                        self.force[ib].x -= fx; self.force[ib].y -= fy;
                        self.torq[ia] += taui; self.torq[ib] += tauj;
                        self.eatom[ia] += 0.5 * ep; self.eatom[ib] += 0.5 * ep;
                        e += ep;
                    }
                }
                if self.dbg_timing { self.dbg_t[0] += t0.elapsed().as_secs_f64(); t0 = std::time::Instant::now(); }
                // pass 2: gather forward-stencil halo cells (h>c) into scratch
                grid.h_idx.clear(); grid.h_pos.clear(); grid.h_zc.clear(); grid.h_ty.clear();
                grid.h_f.clear(); grid.h_t.clear(); grid.h_e.clear();
                for cx in (ci + 1)..=(ci + s).min(nx - 1) {   // same row, forward
                    let h = (cx + nx * cj) as usize;
                    for &j in grid.buckets.cell_objects(h) {
                        grid.h_idx.push(j); grid.h_pos.push(self.pos[j as usize]); grid.h_zc.push(self.zc[j as usize]); grid.h_ty.push(self.types[j as usize]);
                        grid.h_f.push(Vec2d::new(0.0, 0.0)); grid.h_t.push(0.0); grid.h_e.push(0.0);
                    }
                }
                for cy in (cj + 1)..=(cj + s).min(ny - 1) {   // upper rows, ±s
                    for cx in (ci - s).max(0)..=(ci + s).min(nx - 1) {
                        let h = (cx + nx * cy) as usize;
                        for &j in grid.buckets.cell_objects(h) {
                            grid.h_idx.push(j); grid.h_pos.push(self.pos[j as usize]); grid.h_zc.push(self.zc[j as usize]); grid.h_ty.push(self.types[j as usize]);
                            grid.h_f.push(Vec2d::new(0.0, 0.0)); grid.h_t.push(0.0); grid.h_e.push(0.0);
                        }
                    }
                }
                if self.dbg_timing { self.dbg_t[1] += t0.elapsed().as_secs_f64(); t0 = std::time::Instant::now(); }
                // pass 3: i∈c × contiguous halo scratch (the hot streaming loop)
                for a in i0..i1 {
                    let i = grid.buckets.items[a] as usize;
                    let (ti, zi, pi) = (self.types[i], self.zc[i], self.pos[i]);
                    let (mut fx_i, mut fy_i, mut t_i, mut e_i) = (0.0, 0.0, 0.0, 0.0);
                    for k in 0..grid.h_idx.len() {
                        let (ep, fx, fy, taui, tauj) = self.pair_math(ti, zi, pi, grid.h_ty[k], grid.h_zc[k], grid.h_pos[k]);
                        fx_i += fx; fy_i += fy; t_i += taui; e_i += 0.5 * ep; e += ep;
                        grid.h_f[k].x -= fx; grid.h_f[k].y -= fy; grid.h_t[k] += tauj; grid.h_e[k] += 0.5 * ep;
                    }
                    self.force[i].x += fx_i; self.force[i].y += fy_i;
                    self.torq[i] += t_i; self.eatom[i] += e_i;
                }
                if self.dbg_timing { self.dbg_t[2] += t0.elapsed().as_secs_f64(); t0 = std::time::Instant::now(); }
                // pass 4: scatter halo contributions back — skip zero entries
                // (most halo atoms never interact; random SoA writes are costly)
                for k in 0..grid.h_idx.len() {
                    if grid.h_f[k].x == 0.0 && grid.h_f[k].y == 0.0 && grid.h_t[k] == 0.0 && grid.h_e[k] == 0.0 { continue; }
                    let j = grid.h_idx[k] as usize;
                    self.force[j].x += grid.h_f[k].x; self.force[j].y += grid.h_f[k].y;
                    self.torq[j] += grid.h_t[k]; self.eatom[j] += grid.h_e[k];
                }
                if self.dbg_timing { self.dbg_t[3] += t0.elapsed().as_secs_f64(); t0 = std::time::Instant::now(); }
            }
            self.grid = Some(grid);
        } else if let Some(mut groups) = self.groups.take() {
            // COG-group loop: intra-group pairs + AABB-overlapping inter-group
            // pairs only (O(G²) over groups — G small). Groups are COG-assigned
            // (topology-free); stale groups still correct since AABBs refit.
            groups.rebuild(&self.pos);
            let ng = groups.ngroup();
            for g in 0..ng {
                if groups.buckets.offsets[g] == groups.buckets.offsets[g + 1] { continue; }
                for &i in groups.buckets.cell_objects(g) {
                    for &j in groups.buckets.cell_objects(g) {
                        if j > i { e += self.pair_ef(i as usize, j as usize); }
                    }
                }
                for h in (g + 1)..ng {
                    if groups.buckets.offsets[h] == groups.buckets.offsets[h + 1] { continue; }
                    if !groups.overlap(g, h, self.rcut) { continue; }
                    for &i in groups.buckets.cell_objects(g) {
                        for &j in groups.buckets.cell_objects(h) {
                            e += self.pair_ef(i as usize, j as usize);
                        }
                    }
                }
            }
            self.groups = Some(groups);
        } else {
            for i in 0..self.natom() { for j in (i + 1)..self.natom() { e += self.pair_ef(i, j); } }
        }
        for ir in 0..nr { for i in 0..self.natom() { e += self.ring_ef(ir, i); } }
        if let Some((ra, k)) = self.arena {
            for i in 0..self.natom() {
                let r = self.pos[i].norm();
                let dr = r - ra;
                if dr > 0.0 {
                    e += k * dr * dr;
                    self.eatom[i] += k * dr * dr;
                    let f = -2.0 * k * dr / r;                    // inward radial force
                    self.force[i].x += f * self.pos[i].x;
                    self.force[i].y += f * self.pos[i].y;
                }
            }
        }
        e
    }

    /// One damped-MD step (symplectic Euler): v = v·cdamp + F·inv_m·dt, x += v·dt;
    /// ω = ω·cdamp + τ·inv_I·dt, z *= exp(i·ω·dt) (complex multiply + renormalize).
    pub fn step_md(&mut self, dt: f64, cdamp: f64) {
        for i in 0..self.natom() {
            self.vel[i] = self.vel[i] * cdamp + self.force[i] * (self.inv_m[i] * dt);
            self.pos[i] = self.pos[i] + self.vel[i] * dt;
            self.om[i] = self.om[i] * cdamp + self.torq[i] * self.inv_i[i] * dt;
            let (s, c) = (self.om[i] * dt).sin_cos();
            self.zc[i] = cmul(self.zc[i], Vec2d::new(c, s));
            let n = self.zc[i].norm();
            assert!(n.is_finite() && n > 1e-12, "rarff2d::step_md: degenerate orientation atom {i}: |z|={n}");
            self.zc[i].mul(1.0 / n);
        }
    }

    /// Overdamped gradient step (repair/minimization — no inertia):
    /// x += F·inv_m·dt capped at |dx| ≤ dmax (trust region);
    /// φ += τ·inv_I·dt capped at |dφ| ≤ dphi_max, applied via complex multiply.
    /// Use this (not step_md) when the goal is "slide down E", not MD dynamics.
    pub fn step_gd(&mut self, dt: f64, dmax: f64, dphi_max: f64) {
        for i in 0..self.natom() {
            let mut dx = self.force[i] * (self.inv_m[i] * dt);
            let l = dx.norm();
            if l > dmax { dx.mul(dmax / l); }
            self.pos[i].add(dx);
            let dph = (self.torq[i] * self.inv_i[i] * dt).clamp(-dphi_max, dphi_max);
            let (s, c) = dph.sin_cos();
            self.zc[i] = cmul(self.zc[i], Vec2d::new(c, s));
            let n = self.zc[i].norm();
            assert!(n.is_finite() && n > 1e-12, "rarff2d::step_gd: degenerate orientation atom {i}: |z|={n}");
            self.zc[i].mul(1.0 / n);
        }
    }

    /// max|F|² over atoms and max|τ|² over orientations — convergence criteria.
    pub fn max_f2(&self) -> f64 { self.force.iter().map(|f| f.norm2()).fold(0.0, f64::max) }
    pub fn max_t2(&self) -> f64 { self.torq.iter().map(|t| t * t).fold(0.0, f64::max) }

    /// Probe-atom potential field at `p`: energy an optimally-oriented probe
    /// (probe gate ≡ 1 → shows attraction basins wherever SOURCE ports point)
    /// would feel from all atoms except `skip`, plus all ring entities.
    /// Read-only, no force/torque side effects. `tp` = probe atom type.
    pub fn field_at(&self, p: Vec2d, skip: usize, tp: Rarff2dType) -> f64 {
        let mut e = 0.0;
        for i in 0..self.natom() {
            if i == skip { continue; }
            let ti = &self.types[i];
            let dx = p - self.pos[i];
            let r2 = dx.norm2();
            if r2 > self.rcut * self.rcut || r2 < 1e-12 { continue; }
            let r = r2.sqrt();
            let x = -(ti.b + tp.b) * (r - (ti.r_cov + tp.r_cov));
            let (ee, _s) = pex(x, self.pex_m);
            let th = dx.y.atan2(dx.x);
            let (gi, _dgi) = gate(ti.nfold, ti.w, th - self.phi(i));
            e += ti.a * ee * ee - 2.0 * ti.a * ee * gi;   // rep wall + gated attraction
        }
        for rg in &self.rings {
            let dx = p - rg.pos;
            let rho = dx.norm2().sqrt();
            let (bb, _db) = bump(rho - rg.radius, rg.h);  // site band attraction
            if bb != 0.0 {
                let phi_c = rg.zc.y.atan2(rg.zc.x);
                let dl = dx.y.atan2(dx.x) - phi_c;
                e -= rg.a * bb * 0.5 * (1.0 + (rg.nfold as f64 * dl).cos());
            }
            let (ec, _) = pex(-rg.b_core * (rho - rg.r0_core), self.pex_m);  // core repulsion
            e += rg.a_core * ec * ec;
        }
        e
    }

    /// Relax by damped MD until force+torque < tol or nsteps exhausted.
    /// Returns (E_final, n_steps, converged).
    pub fn relax(&mut self, nsteps: usize, dt: f64, cdamp: f64, tol: f64, verbose: bool) -> (f64, usize, bool) {
        let mut e = self.eval();
        for step in 0..nsteps {
            self.step_md(dt, cdamp);
            e = self.eval();
            if verbose && step % 20 == 0 { eprintln!("[rarff2d] step {step:5} E={e:10.6} max|F|={:.6} max|t|={:.6}", self.max_f2().sqrt(), self.max_t2().sqrt()); }
            if self.max_f2() < tol * tol && self.max_t2() < tol * tol {
                if verbose { eprintln!("[rarff2d] converged step {step} E={e:.6}"); }
                return (e, step, true);
            }
        }
        (e, nsteps, false)
    }

    /// Finite-difference check of analytic forces/torques vs -dE/dx and -dE/dφ.
    /// Returns (max force err, max torque err). Diagnostic, not pass/fail.
    pub fn fd_check(&mut self, eps: f64) -> (f64, f64) {
        self.eval();
        let (f0, t0) = (self.force.clone(), self.torq.clone());
        let (rf0, rt0) = (self.ring_f.clone(), self.ring_t.clone());
        let (pos0, zc0) = (self.pos.clone(), self.zc.clone());
        let (mut f_err, mut t_err) = (0.0f64, 0.0f64);
        for i in 0..self.natom() {
            for c in 0..2 {
                self.pos = pos0.clone();
                self.pos[i][c] += eps; let ep = self.eval();
                self.pos[i][c] -= 2.0 * eps; let em = self.eval();
                self.pos[i][c] = pos0[i][c]; // restore — else perturbs next FD evals
                let f_fd = -(ep - em) / (2.0 * eps);
                let err = (f_fd - f0[i][c]).abs();
                if err > f_err { f_err = err; eprintln!("fd_check force: atom {i} comp {c}: F_an={:.6} F_fd={f_fd:.6} err={err:.3e}", f0[i][c]); }
            }
            self.pos = pos0.clone(); self.zc = zc0.clone();
            let dp = Vec2d::new(eps.cos(), eps.sin());
            self.zc[i] = cmul(self.zc[i], dp); let ep = self.eval();
            self.zc[i] = zc0[i]; self.zc[i] = cmul(self.zc[i], cconj(dp)); let em = self.eval();
            self.zc[i] = zc0[i]; // restore — else perturbs next FD evals
            let t_fd = -(ep - em) / (2.0 * eps);
            let err = (t_fd - t0[i]).abs();
            if err > t_err { t_err = err; eprintln!("fd_check torque: atom {i}: T_an={:.6} T_fd={t_fd:.6} err={err:.3e}", t0[i]); }
        }
        // ring DOFs: center pos + orientation → ring_f, ring_t
        let rings0 = self.rings.clone();
        for ir in 0..self.rings.len() {
            for c in 0..2 {
                self.rings[ir].pos[c] += eps; let ep = self.eval();
                self.rings[ir].pos[c] -= 2.0 * eps; let em = self.eval();
                self.rings[ir].pos = rings0[ir].pos;
                let f_fd = -(ep - em) / (2.0 * eps);
                let err = (f_fd - rf0[ir][c]).abs();
                if err > f_err { f_err = err; eprintln!("fd_check ring_f: ring {ir} comp {c}: F_an={:.6} F_fd={f_fd:.6} err={err:.3e}", rf0[ir][c]); }
            }
            let dp = Vec2d::new(eps.cos(), eps.sin());
            self.rings[ir].zc = cmul(self.rings[ir].zc, dp); let ep = self.eval();
            self.rings[ir].zc = rings0[ir].zc; self.rings[ir].zc = cmul(self.rings[ir].zc, cconj(dp)); let em = self.eval();
            self.rings[ir].zc = rings0[ir].zc;
            let t_fd = -(ep - em) / (2.0 * eps);
            let err = (t_fd - rt0[ir]).abs();
            if err > t_err { t_err = err; eprintln!("fd_check ring_t: ring {ir}: T_an={:.6} T_fd={t_fd:.6} err={err:.3e}", rt0[ir]); }
        }
        self.pos = pos0; self.zc = zc0; self.rings = rings0;
        self.eval();
        (f_err, t_err)
    }
}
