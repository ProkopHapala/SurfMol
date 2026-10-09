//! Reactive RAFF — 3D rigid-atom reactive forcefield on quaternion state.
//!
//! RARFF-2D radial Morse (pex compact support) × FireCore port-pair angular
//! gate `(ci·cj·cij)^4`, evaluated on `RaffState`/`RaffTopology` (quaternion
//! rigid atoms with up to 4 ports). Bonds EMERGE from geometry — no bond list.
//! Ported from FireCore `cpp/common/molecular/RARFF_SR.h` (`pairEF` line 441,
//! `interEF_buckets` 599, `evalTorques` 718, `projectBonds` 667); the pair
//! equations implement `notes/designs/2026-10-09_raff_reactive_3d_inventory.md`
//! §3.1 (authoritative).
//!
//! Pair term (d = x_j - x_i, r, ĥ = d/r; r0 = r_i+r_j, b = b_i+b_j, a = a_i·a_j):
//!   e,dedx = pex(-b(r-r0), m)          — compact support r0 + 2^m/b
//!   Y      = Σ_αβ (ci·cj·cij)^4        — ci=h_iα·ĥ>0, cj=h_jβ·ĥ<0, cij=h_iα·h_jβ<0
//!   E      = a·(e² - 2·e·Y)            — ungated repulsive wall + gated well
//!   point partner j:  Y = Σ_α ci_α⁴ (ci>0);  both point → Y=0 (pure wall).
//! Torques: f_iα = 2ae·∂Y/∂h_iα, τ_i += h_iα × f_iα (FireCore evalTorques inlined).
//!
//! H is a 1-PORT atom (RaffTopology::set_1port, nport=1): the two-sided gate
//! (ci·cj·cij)^4 needs BOTH ports facing the partner, so H binds to exactly
//! one atom along its port direction — emergent valence saturation; in all
//! other directions it sees only the ungated e² wall and repels. This is the
//! explicit-particle analogue of FireCore's cap sites (`apos + h*lcap`).
//! A separate steric/vdW wall at non-bonded radius is deferred (needs a
//! gate-blend or second radial term — left as an open design question).

use numtypes::{Vec3d, VEC3D_ZERO};
use spacc::uniform_grid::Grid3;
use crate::raff::{RaffState, RaffTopology, RaffConfig, quat_rotate, eval_box_forces, integrate_md};
use crate::rarff2d::{pex, support_max, PEXP_M};

/// Per-atom reactive params: covalent radius + Morse a (depth) and b (stiffness).
#[derive(Copy, Clone, Debug)]
pub struct ReactParams { pub r_cov: f64, pub a: f64, pub b: f64 }

/// Reactive evaluator config. `use_cij=false` drops the h_i·h_j factor from the
/// port-pair gate (debug/ablation only — physics is cij ON).
#[derive(Copy, Clone, Debug)]
pub struct ReactConfig { pub pex_m: u32, pub rcut: f64, pub use_cij: bool }

/// Reactive system: per-atom params + scratch (world port dirs, per-atom energy,
/// optional Grid3 with halo-gather buffers). All buffers allocated once in
/// `new`/`set_grid` — the eval hot loops never allocate.
pub struct Reactive {
    pub params: Vec<ReactParams>,
    pub cfg: ReactConfig,
    pub h_world: Vec<Vec3d>,       // [natoms*4] world-frame port dirs (projectBonds)
    pub eatom: Vec<f64>,           // per-atom pair energy (consistency map)
    pub grid: Option<Grid3>,
    // Halo gather scratch (Grid2d ghost-list pattern) + forward-stencil scratch
    pub fwd_cells: Vec<usize>,
    pub h_idx: Vec<u32>, pub h_pos: Vec<Vec3d>, pub h_dir: Vec<[Vec3d; 4]>,
    pub h_np: Vec<u8>, pub h_par: Vec<ReactParams>,
    pub h_f: Vec<Vec3d>, pub h_t: Vec<Vec3d>, pub h_e: Vec<f64>,
    pub dbg_npair: usize,          // pair_math calls since last eval
    pub dbg_ninside: usize,        // calls inside rcut
}

/// FireCore `projectBonds`: rotate body-frame port dirs into world frame —
/// `h_world[i*4+s] = R_i · port_local[i*4+s]` for s < nport[i]. Other slots
/// untouched. Fresh projection from `state.quat` (callers never see stale dirs).
pub fn project_ports(state: &RaffState, topo: &RaffTopology, h_world: &mut [Vec3d]) {
    for i in 0..topo.natoms {
        let np = topo.nport[i] as usize;
        for s in 0..np { h_world[i * 4 + s] = quat_rotate(state.quat[i], topo.port_local[i * 4 + s]); }
    }
}

/// Pure pair math — the single source of truth for the pair interaction
/// (both the O(N²) and the grid loop call this). Implements §3.1 of the notes
/// verbatim; mirrors `RARFF_SR.h::pairEF` 441–564.
/// `hi`/`hj` are the world port dirs of i/j (len = nport, 0 for point atoms).
/// Returns None outside rcut; else (E_pair, F on j, τ_i, τ_j).
#[inline]
pub fn pair_math(
    cfg: &ReactConfig,
    pi: &ReactParams, hi: &[Vec3d], pos_i: Vec3d,
    pj: &ReactParams, hj: &[Vec3d], pos_j: Vec3d,
) -> Option<(f64, Vec3d, Vec3d, Vec3d)> {
    let d = pos_j - pos_i;
    let r2 = d.norm2();
    if r2 > cfg.rcut * cfg.rcut || r2 < 1e-12 { return None; }
    let r = r2.sqrt();
    let h = d * (1.0 / r);
    let r0 = pi.r_cov + pj.r_cov;
    let b = pi.b + pj.b;
    let a = pi.a * pj.a;
    let (e, dedx) = pex(-b * (r - r0), cfg.pex_m);   // de/dr = -b·dedx
    if e == 0.0 { return None; }                     // outside compact support
    let npi = hi.len();
    let npj = hj.len();
    let mut y = 0.0f64;
    let mut gpos = VEC3D_ZERO;          // angular ∇_jE accumulator (pre −2ae/r)
    let mut fi = [VEC3D_ZERO; 4];       // Σ ∂Y/∂h_iα per port
    let mut fj = [VEC3D_ZERO; 4];
    if npi > 0 && npj > 0 {
        for al in 0..npi {
            let ha = hi[al];
            let ci = ha.dot(h);
            if ci <= 0.0 { continue; }              // i-port must aim at j
            for be in 0..npj {
                let hb = hj[be];
                let cj = hb.dot(h);
                if cj >= 0.0 { continue; }          // j-port must aim at i
                let cij = if cfg.use_cij { ha.dot(hb) } else { -1.0 };
                if cfg.use_cij && cij >= 0.0 { continue; }   // ports anti-parallel
                let cc = ci * cj * cij;             // > 0 at alignment
                let cc2 = cc * cc;
                y += cc2 * cc2;
                let dk = 4.0 * cc2 * cc;            // dY/dcc = 4cc³
                gpos.add_mul(ha - h * ci, dk * cj * cij);   // ∂ci/∂x_j = (ha - ci·ĥ)/r
                fi[al].add_mul(h, dk * cj * cij);            // ∂cc/∂h_iα via ci
                fj[be].add_mul(h, dk * ci * cij);            // ∂cc/∂h_jβ via cj
                if cfg.use_cij {
                    gpos.add_mul(hb - h * cj, dk * ci * cij); // ∂cj/∂x_j
                    fi[al].add_mul(hb, dk * ci * cj);         // ∂cc/∂h_iα via cij
                    fj[be].add_mul(ha, dk * ci * cj);         // ∂cc/∂h_jβ via cij
                }
            }
        }
    } else if npi > 0 {
        for al in 0..npi {                          // point partner j: Y = Σ ci⁴
            let ha = hi[al];
            let ci = ha.dot(h);
            if ci <= 0.0 { continue; }
            let c2 = ci * ci;
            y += c2 * c2;
            let dk = 4.0 * c2 * ci;
            gpos.add_mul(ha - h * ci, dk);
            fi[al].add_mul(h, dk);
        }
    } else if npj > 0 {
        for be in 0..npj {                          // point partner i: Y = Σ cj⁴ (cj<0)
            let hb = hj[be];
            let cj = hb.dot(h);
            if cj >= 0.0 { continue; }
            let c2 = cj * cj;
            y += c2 * c2;
            let dk = 4.0 * c2 * cj;                 // d(cj⁴)/dcj (sign carried)
            gpos.add_mul(hb - h * cj, dk);
            fj[be].add_mul(h, dk);
        }
    }
    let e_pair = a * (e * e - 2.0 * e * y);
    let dedr = 2.0 * a * b * dedx * (y - e);        // dE/dr (radial part)
    let k = -2.0 * a * e / r;                       // angular prefactor
    let grad_j = h * dedr + gpos * k;               // ∇_jE
    let f2ae = -2.0 * a * e;                        // f_port = 2ae·Σ → ×(-f2ae)
    let mut taui = VEC3D_ZERO;
    let mut tauj = VEC3D_ZERO;
    for al in 0..npi { taui.add(Vec3d::cross(hi[al], fi[al] * (-f2ae))); }
    for be in 0..npj { tauj.add(Vec3d::cross(hj[be], fj[be] * (-f2ae))); }
    Some((e_pair, grad_j * (-1.0), taui, tauj))     // F_j = −∇_jE ; F_i = +∇_jE
}

impl Reactive {
    /// `pex_m` squarings → rcut auto-derived as max pair support r0 + 2^m/(b_i+b_j).
    /// Asserts params.len() == topo.natoms. All scratch allocated here.
    pub fn new(topo: &RaffTopology, params: Vec<ReactParams>, pex_m: u32) -> Self {
        let n = topo.natoms;
        assert_eq!(params.len(), n, "Reactive::new: params.len()={} != natoms={}", params.len(), n);
        let mut ubt: Vec<(f64, f64)> = Vec::new();
        for p in &params { let k = (p.r_cov, p.b); if !ubt.contains(&k) { ubt.push(k); } }
        let rcut = support_max(&ubt, pex_m);
        Self { params, cfg: ReactConfig { pex_m, rcut, use_cij: true },
            h_world: vec![VEC3D_ZERO; n * 4], eatom: vec![0.0; n], grid: None,
            fwd_cells: Vec::with_capacity(64),
            h_idx: Vec::with_capacity(n), h_pos: Vec::with_capacity(n), h_dir: Vec::with_capacity(n),
            h_np: Vec::with_capacity(n), h_par: Vec::with_capacity(n),
            h_f: Vec::with_capacity(n), h_t: Vec::with_capacity(n), h_e: Vec::with_capacity(n),
            dbg_npair: 0, dbg_ninside: 0 }
    }

    /// Live re-parameterization (interactive viewer): replace params, recompute
    /// pex_m + auto rcut (same unique-(r_cov,b) support scan as `new`), drop the
    /// grid — the forward-stencil radius depends on rcut, caller re-runs
    /// `set_grid` if still wanted. Keeps h_world/eatom/scratch allocations;
    /// natoms must not change (adding atoms → build a new `Reactive`).
    pub fn reconfigure(&mut self, params: Vec<ReactParams>, pex_m: u32) {
        assert_eq!(params.len(), self.params.len(), "Reactive::reconfigure: params.len()={} != natoms={}", params.len(), self.params.len());
        let mut ubt: Vec<(f64, f64)> = Vec::new();
        for p in &params { let k = (p.r_cov, p.b); if !ubt.contains(&k) { ubt.push(k); } }
        self.params = params;
        self.cfg.pex_m = pex_m;
        self.cfg.rcut = support_max(&ubt, pex_m);
        self.grid = None;
    }

    /// Configure the 3D cell grid (reconfigure step — allocates once).
    /// dx >= rcut → strict 3x3x3 stencil (s=1). Call again if natoms changes.
    pub fn set_grid(&mut self, origin: [f64; 3], n: [usize; 3], dx: f64) {
        let s = (self.cfg.rcut / dx).ceil() as usize;
        self.fwd_cells = Vec::with_capacity((2 * s + 1).pow(3));
        self.grid = Some(Grid3::new(origin, n, dx, self.params.len()));
    }

    /// Zero fapos/tau/eatom, project ports to world frame (projectBonds), then
    /// evaluate all pairs — O(N²) reference loop without grid, cell-pair loop
    /// with halo gather when `set_grid` was called. Both paths agree to ~1e-12.
    pub fn eval_reactive(&mut self, state: &RaffState, topo: &RaffTopology, fapos: &mut [Vec3d], tau: &mut [Vec3d]) -> f64 {
        let n = topo.natoms;
        self.dbg_npair = 0;
        self.dbg_ninside = 0;
        for i in 0..n { fapos[i] = VEC3D_ZERO; tau[i] = VEC3D_ZERO; self.eatom[i] = 0.0; }
        project_ports(state, topo, &mut self.h_world);
        let mut e = 0.0;
        let cfg = self.cfg;
        if let Some(mut grid) = self.grid.take() {
            // Cell-pair loop (Grid2d eval pattern → 3D; FireCore interEF_buckets):
            // intra-cell i<j pairs + forward-stencil halo gathered contiguous.
            grid.rebuild(&state.pos);
            let s = (cfg.rcut / grid.dx).ceil() as i64;
            for c in 0..grid.buckets.ncells {
                let objs = grid.buckets.cell_objects(c);
                let (i0, i1) = (grid.buckets.offsets[c] as usize, grid.buckets.offsets[c + 1] as usize);
                if i0 == i1 { continue; }
                for a in i0..i1 {                           // pass 1: intra-cell pairs
                    let ia = objs[a - i0] as usize;
                    let npa = topo.nport[ia] as usize;
                    for bb in (a + 1)..i1 {
                        let ib = objs[bb - i0] as usize;
                        let npb = topo.nport[ib] as usize;
                        self.dbg_npair += 1;
                        if let Some((ep, fjf, ti, tj)) = pair_math(&cfg, &self.params[ia], &self.h_world[ia * 4..ia * 4 + npa], state.pos[ia], &self.params[ib], &self.h_world[ib * 4..ib * 4 + npb], state.pos[ib]) {
                            self.dbg_ninside += 1;
                            fapos[ib].add(fjf); fapos[ia].sub(fjf);
                            tau[ia].add(ti); tau[ib].add(tj);
                            self.eatom[ia] += 0.5 * ep; self.eatom[ib] += 0.5 * ep;
                            e += ep;
                        }
                    }
                }
                // pass 2: gather forward-stencil halo into contiguous scratch
                grid.forward_cells(c, s, &mut self.fwd_cells);
                self.h_idx.clear(); self.h_pos.clear(); self.h_dir.clear(); self.h_np.clear(); self.h_par.clear();
                self.h_f.clear(); self.h_t.clear(); self.h_e.clear();
                for &hc in &self.fwd_cells {
                    for &j in grid.buckets.cell_objects(hc) {
                        let ju = j as usize;
                        self.h_idx.push(j); self.h_pos.push(state.pos[ju]);
                        self.h_np.push(topo.nport[ju]); self.h_par.push(self.params[ju]);
                        self.h_dir.push(self.h_world[ju * 4..ju * 4 + 4].try_into().unwrap());
                        self.h_f.push(VEC3D_ZERO); self.h_t.push(VEC3D_ZERO); self.h_e.push(0.0);
                    }
                }
                // pass 3: i∈cell × contiguous halo scratch (streaming loop)
                for a in i0..i1 {
                    let i = objs[a - i0] as usize;
                    let npi = topo.nport[i] as usize;
                    let pi = self.params[i];
                    let pos_i = state.pos[i];
                    let (mut f_i, mut t_i, mut e_i) = (VEC3D_ZERO, VEC3D_ZERO, 0.0f64);
                    for k in 0..self.h_idx.len() {
                        self.dbg_npair += 1;
                        let npk = self.h_np[k] as usize;
                        if let Some((ep, fjf, ti, tj)) = pair_math(&cfg, &pi, &self.h_world[i * 4..i * 4 + npi], pos_i, &self.h_par[k], &self.h_dir[k][..npk], self.h_pos[k]) {
                            self.dbg_ninside += 1;
                            f_i.sub(fjf); t_i.add(ti); e_i += 0.5 * ep; e += ep;
                            self.h_f[k].add(fjf); self.h_t[k].add(tj); self.h_e[k] += 0.5 * ep;
                        }
                    }
                    fapos[i].add(f_i); tau[i].add(t_i); self.eatom[i] += e_i;
                }
                // pass 4: scatter halo contributions back — skip all-zero entries
                for k in 0..self.h_idx.len() {
                    if self.h_e[k] == 0.0 { continue; }
                    let j = self.h_idx[k] as usize;
                    fapos[j].add(self.h_f[k]); tau[j].add(self.h_t[k]); self.eatom[j] += self.h_e[k];
                }
            }
            self.grid = Some(grid);
        } else {
            for i in 0..n {                                 // O(N²) reference loop
                let npi = topo.nport[i] as usize;
                for j in (i + 1)..n {
                    let npj = topo.nport[j] as usize;
                    self.dbg_npair += 1;
                    if let Some((ep, fjf, ti, tj)) = pair_math(&cfg, &self.params[i], &self.h_world[i * 4..i * 4 + npi], state.pos[i], &self.params[j], &self.h_world[j * 4..j * 4 + npj], state.pos[j]) {
                        self.dbg_ninside += 1;
                        fapos[j].add(fjf); fapos[i].sub(fjf);
                        tau[i].add(ti); tau[j].add(tj);
                        self.eatom[i] += 0.5 * ep; self.eatom[j] += 0.5 * ep;
                        e += ep;
                    }
                }
            }
        }
        e
    }

    /// One damped-MD step: eval_reactive + box forces + shared integrator.
    /// Same return convention as `step_force_md`: (total_E, max|F|, max|τ|).
    /// eval_port_forces is NOT called — this is the reactive-only mode.
    pub fn step_reactive_md(&mut self, state: &mut RaffState, topo: &RaffTopology, cfg: &RaffConfig, fapos: &mut [Vec3d], tau: &mut [Vec3d]) -> (f64, f64, f64) {
        let e_rx = self.eval_reactive(state, topo, fapos, tau);
        let e_box = eval_box_forces(state, &cfg.box_cfg, fapos);
        let (max_f, max_t) = integrate_md(state, topo, cfg, fapos, tau);
        (e_rx + e_box, max_f, max_t)
    }

    /// Diagnostic: all pairs with E_pair < e_thr = "emerged bonds". O(N²).
    /// Returns (i, j, E_pair) sorted by energy.
    pub fn emerged_bonds(&mut self, state: &RaffState, topo: &RaffTopology, e_thr: f64) -> Vec<(u32, u32, f64)> {
        let n = topo.natoms;
        project_ports(state, topo, &mut self.h_world);
        let mut out = Vec::new();
        for i in 0..n {
            let npi = topo.nport[i] as usize;
            for j in (i + 1)..n {
                let npj = topo.nport[j] as usize;
                if let Some((ep, _, _, _)) = pair_math(&self.cfg, &self.params[i], &self.h_world[i * 4..i * 4 + npi], state.pos[i], &self.params[j], &self.h_world[j * 4..j * 4 + npj], state.pos[j]) {
                    if ep < e_thr { out.push((i as u32, j as u32, ep)); }
                }
            }
        }
        out.sort_by(|a, b| a.2.partial_cmp(&b.2).expect("emerged_bonds: NaN pair energy"));
        out
    }
}

/// Multi-frame XYZ trajectory frame: `{n}\nstep {s} E={e}\n` + `El x y z`
/// per atom. Element label from nport: 4→C, 3→N, 2→O, 1→H, else X (same as
/// raff_reactive_demo). Generic over fmt::Write — String or a file adapter.
pub fn write_xyz_frame<W: std::fmt::Write>(buf: &mut W, state: &RaffState, topo: &RaffTopology, step: usize, e: f64) -> std::fmt::Result {
    writeln!(buf, "{}", state.natoms)?;
    writeln!(buf, "step {step} E={e:.6}")?;
    for i in 0..state.natoms {
        let el = match topo.nport[i] { 1 => "H", 2 => "O", 3 => "N", 4 => "C", _ => "X" };
        let p = state.pos[i];
        writeln!(buf, "{el}  {:.6} {:.6} {:.6}", p.x, p.y, p.z)?;
    }
    Ok(())
}

/// Reactive-mode inertia: `compute_inertia` is unusable (reads bond_params via
/// neigh_bs), so use the same formula with l0 = 2·r_cov: I = 0.4·(2 r_cov)².
/// nport=0 atoms get inv_inertia = 0 (true points don't rotate); 1-port H
/// rotates — its port is torqued onto the partner by the same Y gate.
pub fn set_reactive_inertia(topo: &mut RaffTopology, params: &[ReactParams]) {
    assert_eq!(params.len(), topo.natoms, "set_reactive_inertia: params.len()={} != natoms={}", params.len(), topo.natoms);
    for i in 0..topo.natoms {
        topo.inv_inertia[i] = if topo.nport[i] > 0 { 1.0 / (0.4 * (2.0 * params[i].r_cov).powi(2) + 1e-18) } else { 0.0 };
    }
}

/// Default pex squaring count re-export for callers (same knob as RARFF-2D).
pub const REACT_PEX_M: u32 = PEXP_M;
