//! RAFF — Rigid-Atom port-based Force Field computational core.
//!
//! Implements the corrected equations from `notes/designs/raff_theory_equations.md`:
//! - §1: Port energy with clean convention (E = k_p/2 |e|^2, F = k_p·e)
//! - §2.1: Dynamic rotation (symplectic Euler with physical inertia)
//! - §2.2: Analytical rotation via Wahba problem (rotation-only, no centroid subtraction)
//! - §3.1: Force-based MD
//! - §3.2: XPBD port constraints (C = |x_j - tip| = 0, compliance α̃ = 1/(K·dt²))
//! - §11: Common proximal problem (IMEX: soft force → target y, hard solve implicit)
//! - §11.7: Finite-difference validation utilities
//!
//! Design: RaffState (positions, quaternions, velocities) + RaffTopology (ports, neighbors, params)
//! + RaffConfig (solver mode, dt, damping). Scratch values (forces, torques) computed on the fly.
//! This separation maps cleanly to the eventual OpenCL version.

use numtypes::{Mat3d, Quat4d, Quat4i, Vec3d, VEC3D_ZERO};

// ==================================================================
//  Data structures (§11 architecture: State + Topology + Config)
// ==================================================================

/// Per-port stiffness and equilibrium length. k_p is the **per-port** stiffness;
/// for reciprocal ports (two per bond), use k_p = K_bond / 2.
#[derive(Copy, Clone, Debug)]
pub struct PortParam { pub k_p: f64, pub l0: f64 }

/// Orientation mode: dynamic DOF (physical inertia) or adiabatic (memoryless, solved each step).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OrientMode { Dynamic, Adiabatic }

/// Dynamics strategy: force-based MD or position-based (XPBD).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DynMode { ForceMD, Xpbd }

/// Solver configuration. Separated from state and topology (§11).
#[derive(Copy, Clone, Debug)]
pub struct RaffConfig {
    pub orient_mode: OrientMode,
    pub dyn_mode: DynMode,
    pub dt: f64,
    pub cdamp: f64,       // linear damping (1 = no damping, 0 = full)
    pub rot_damp: f64,    // rotational damping
    pub flim: f64,        // force clamp limit (0 = no clamp)
    pub xpbd_iters: usize,
    pub xpbd_over_relax: f64, // >1 for over-relaxation Jacobi
}

impl Default for RaffConfig {
    fn default() -> Self {
        Self { orient_mode: OrientMode::Dynamic, dyn_mode: DynMode::ForceMD,
            dt: 0.01, cdamp: 0.95, rot_damp: 0.95, flim: 100.0,
            xpbd_iters: 16, xpbd_over_relax: 1.0 }
    }
}

/// Rigid-atom state: positions, orientations, velocities, angular velocities.
/// No forces/torques stored — those are scratch (computed on the fly).
#[derive(Clone)]
pub struct RaffState {
    pub natoms: usize,
    pub pos: Vec<Vec3d>,    // x_i
    pub quat: Vec<Quat4d>,  // q_i (xyzw convention: w = scalar)
    pub vel: Vec<Vec3d>,    // v_i
    pub omega: Vec<Vec3d>,  // ω_i (angular velocity, body frame)
}

/// Per-atom non-bonded parameters: Lennard-Jones (σ, ε) + Coulomb charge + collision radius.
/// LJ uses Lorentz-Berthelot mixing: σ_ij = (σ_i + σ_j)/2, ε_ij = sqrt(ε_i · ε_j).
/// Collision uses soft-sphere repulsion when r < r_i + r_j (hard-sphere overlap).
#[derive(Copy, Clone, Debug, Default)]
pub struct NbParams {
    pub sigma: f64,   // LJ σ (Å)
    pub epsilon: f64, // LJ ε (eV)
    pub charge: f64,  // Coulomb charge (e)
    pub radius: f64,  // collision radius (Å) — 0 = no collision
}

/// Non-bonded configuration: cutoff, exclusion depth, damping.
#[derive(Copy, Clone, Debug)]
pub struct NbConfig {
    pub enabled: bool,
    pub rcut: f64,           // cutoff distance (Å), 0 = no cutoff
    pub r_damp: f64,         // distance damping for 1/r singularity (Å²)
    pub f_max: f64,          // force clamp (eV/Å), 0 = no clamp
    pub k_coll: f64,         // collision stiffness (eV/Å²)
    pub excl_12: bool,       // exclude 1-2 (bonded) pairs
    pub excl_13: bool,       // exclude 1-3 (angle) pairs
}

impl Default for NbConfig {
    fn default() -> Self { Self { enabled: false, rcut: 10.0, r_damp: 0.1, f_max: 50.0, k_coll: 100.0, excl_12: true, excl_13: true } }
}

const EXCL_MAX: usize = 16;  // max exclusions per atom (1-2 + 1-3 neighbors)

/// Topology: ports, neighbors, bond params, inertia. Constant during simulation.
#[derive(Clone)]
pub struct RaffTopology {
    pub natoms: usize,
    pub nport: Vec<u8>,           // number of ports per atom (0-4)
    pub port_local: Vec<Vec3d>,   // body-frame port directions [natoms*4], unit vectors
    pub neighs: Vec<Quat4i>,      // neighbor atom index per port slot (-1 = unused)
    pub neigh_bs: Vec<Quat4i>,    // bond parameter index per port slot (-1 = unused)
    pub bond_params: Vec<PortParam>,
    pub mass: Vec<f64>,           // per-atom mass
    pub inv_inertia: Vec<f64>,    // per-atom inverse scalar inertia (isotropic)
    // Non-bonded
    pub nb_params: Vec<NbParams>,           // per-atom LJ/Coulomb/collision params
    pub excl: Vec<i32>,                     // [natoms * EXCL_MAX] sorted exclusion list (-1 = unused)
}

// ==================================================================
//  Quaternion utilities
// ==================================================================

#[inline(always)] pub fn quat_normalize(mut q: Quat4d) -> Quat4d {
    let n2 = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    if n2 > 0.0 { let inv = 1.0 / n2.sqrt(); q.x *= inv; q.y *= inv; q.z *= inv; q.w *= inv; }
    q
}

#[inline(always)] pub fn quat_mul(a: Quat4d, b: Quat4d) -> Quat4d {
    Quat4d::new(a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
                a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
                a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
                a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z)
}

#[inline(always)] pub fn quat_conj(q: Quat4d) -> Quat4d { Quat4d::new(-q.x, -q.y, -q.z, q.w) }

#[inline(always)] fn quat_rotate(q: Quat4d, v: Vec3d) -> Vec3d {
    let qv = Quat4d::new(v.x, v.y, v.z, 0.0);
    let r = quat_mul(quat_mul(q, qv), quat_conj(q));
    Vec3d::new(r.x, r.y, r.z)
}

#[inline(always)] pub fn quat_from_omega_dt(omega: Vec3d, dt: f64) -> Quat4d {
    let w = omega.norm();
    if w < 1e-12 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); }
    let half = 0.5 * w * dt;
    let s = half.sin();
    let c = half.cos();
    let invw = 1.0 / w;
    Quat4d::new(omega.x*invw*s, omega.y*invw*s, omega.z*invw*s, c)
}

/// Find quaternion that rotates body-frame vector `a` to align with world-frame vector `b`.
/// Uses the half-vector method: q rotates a → b via the shortest arc.
#[inline(always)] fn quat_align_vectors(a: Vec3d, b: Vec3d) -> Quat4d {
    let dot = a.dot(b);
    if dot > 1.0 - 1e-12 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); } // already aligned
    if dot < -1.0 + 1e-12 { // anti-parallel: 180° rotation around any perpendicular axis
        // Find a perpendicular axis
        let axis = if a.x.abs() < 0.9 { Vec3d::new(1.0, 0.0, 0.0) } else { Vec3d::new(0.0, 1.0, 0.0) };
        let perp = Vec3d::cross(a, axis);
        let n = perp.norm();
        if n < 1e-12 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); }
        let axis = perp * (1.0 / n);
        return Quat4d::new(axis.x, axis.y, axis.z, 0.0); // 180° rotation
    }
    // General case: half-vector quaternion
    let half = Vec3d::cross(a, b);
    let w = 1.0 + dot;
    let n = (half.norm2() + w * w).sqrt();
    if n < 1e-12 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); }
    let inv = 1.0 / n;
    Quat4d::new(half.x * inv, half.y * inv, half.z * inv, w * inv)
}

/// Convert quaternion to 3×3 rotation matrix.
#[inline(always)] fn quat_to_mat3(q: Quat4d) -> Mat3d {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    Mat3d::new(
        Vec3d::new(1.0 - 2.0*(y*y+z*z), 2.0*(x*y-w*z), 2.0*(x*z+w*y)),
        Vec3d::new(2.0*(x*y+w*z), 1.0 - 2.0*(x*x+z*z), 2.0*(y*z-w*x)),
        Vec3d::new(2.0*(x*z-w*y), 2.0*(y*z+w*x), 1.0 - 2.0*(x*x+y*y)),
    )
}

// ==================================================================
//  Port geometry setup
// ==================================================================

const INV_SQRT3: f64 = 0.57735026918962576451;

impl RaffTopology {
    pub fn new(natoms: usize) -> Self {
        Self {
            natoms,
            nport: vec![4u8; natoms],
            port_local: vec![VEC3D_ZERO; natoms * 4],
            neighs: vec![Quat4i::new(-1,-1,-1,-1); natoms],
            neigh_bs: vec![Quat4i::new(-1,-1,-1,-1); natoms],
            bond_params: Vec::new(),
            mass: vec![1.0; natoms],
            inv_inertia: vec![0.0; natoms],
            nb_params: vec![NbParams::default(); natoms],
            excl: vec![-1; natoms * EXCL_MAX],
        }
    }

    /// Set sp3 tetrahedral port geometry for atom i.
    pub fn set_sp3(&mut self, i: usize) {
        self.nport[i] = 4;
        let o = i * 4;
        self.port_local[o]   = Vec3d::new( INV_SQRT3,  INV_SQRT3,  INV_SQRT3);
        self.port_local[o+1] = Vec3d::new( INV_SQRT3, -INV_SQRT3, -INV_SQRT3);
        self.port_local[o+2] = Vec3d::new(-INV_SQRT3,  INV_SQRT3, -INV_SQRT3);
        self.port_local[o+3] = Vec3d::new(-INV_SQRT3, -INV_SQRT3,  INV_SQRT3);
    }

    /// Set sp2 trigonal port geometry for atom i.
    pub fn set_sp2(&mut self, i: usize) {
        self.nport[i] = 3;
        let o = i * 4;
        self.port_local[o]   = Vec3d::new( 1.0, 0.0, 0.0);
        self.port_local[o+1] = Vec3d::new(-0.5, 0.8660254037844386, 0.0);
        self.port_local[o+2] = Vec3d::new(-0.5, -0.8660254037844386, 0.0);
    }

    /// Set sp1 linear port geometry for atom i.
    pub fn set_sp1(&mut self, i: usize) {
        self.nport[i] = 2;
        let o = i * 4;
        self.port_local[o]   = Vec3d::new( 1.0, 0.0, 0.0);
        self.port_local[o+1] = Vec3d::new(-1.0, 0.0, 0.0);
    }

    /// Set point atom (no ports, e.g. terminal H).
    pub fn set_point(&mut self, i: usize) {
        self.nport[i] = 0;
    }

    /// Build neighbor/bond-param index arrays from bond list.
    /// Each bond [i,j] with param index ib creates a port on both atoms.
    /// Fills neighs/neigh_bs slots sequentially (x, y, z, w) per atom.
    pub fn build_neighs_from_bonds(&mut self, bonds: &[[i32; 2]]) {
        // Count ports per atom
        let mut port_count = vec![0u8; self.natoms];
        for b in bonds {
            port_count[b[0] as usize] += 1;
            port_count[b[1] as usize] += 1;
        }
        for i in 0..self.natoms {
            if port_count[i] > 4 { panic!("Atom {} has {} bonds (>4), port model supports max 4", i, port_count[i]); }
            self.nport[i] = port_count[i];
        }
        // Reset neighbor slots
        for i in 0..self.natoms {
            self.neighs[i] = Quat4i::new(-1,-1,-1,-1);
            self.neigh_bs[i] = Quat4i::new(-1,-1,-1,-1);
        }
        // Fill neighbor slots sequentially
        let mut slot = vec![0u8; self.natoms];
        for (ib, b) in bonds.iter().enumerate() {
            let (i, j) = (b[0] as usize, b[1] as usize);
            let si = slot[i] as usize;
            let sj = slot[j] as usize;
            // Write into the correct component of Quat4i
            match si {
                0 => { self.neighs[i].x = j as i32; self.neigh_bs[i].x = ib as i32; }
                1 => { self.neighs[i].y = j as i32; self.neigh_bs[i].y = ib as i32; }
                2 => { self.neighs[i].z = j as i32; self.neigh_bs[i].z = ib as i32; }
                _ => { self.neighs[i].w = j as i32; self.neigh_bs[i].w = ib as i32; }
            }
            match sj {
                0 => { self.neighs[j].x = i as i32; self.neigh_bs[j].x = ib as i32; }
                1 => { self.neighs[j].y = i as i32; self.neigh_bs[j].y = ib as i32; }
                2 => { self.neighs[j].z = i as i32; self.neigh_bs[j].z = ib as i32; }
                _ => { self.neighs[j].w = i as i32; self.neigh_bs[j].w = ib as i32; }
            }
            slot[i] += 1;
            slot[j] += 1;
        }
        // Set port geometry based on number of ports
        for i in 0..self.natoms {
            match self.nport[i] {
                0 => self.set_point(i),
                1 => { self.port_local[i*4] = Vec3d::new(1.0,0.0,0.0); }
                2 => self.set_sp1(i),
                3 => self.set_sp2(i),
                _ => self.set_sp3(i),
            }
        }
        self.compute_inertia();
        self.build_exclusions();
    }

    /// Build sorted exclusion list of 1-2 (bonded) and 1-3 (angle) neighbors for each atom.
    /// Ported from nonbonded.rs:make_second_neighs. Stored in `excl` as [natoms * EXCL_MAX].
    pub fn build_exclusions(&mut self) {
        self.excl.fill(-1);
        for ia in 0..self.natoms {
            let i0 = ia * EXCL_MAX;
            let mut n = 0usize;
            let mut add = |jb: i32, excl: &mut [i32]| {
                if jb < 0 || jb == ia as i32 { return; }
                for m in 0..n { if excl[i0 + m] == jb { return; } }
                if n < EXCL_MAX { excl[i0 + n] = jb; n += 1; }
                else { panic!("RaffTopology::build_exclusions: atom {} has >={} exclusions", ia, EXCL_MAX); }
            };
            let ng = self.neighs[ia].as_array();
            // 1-2: direct neighbors
            for s in 0..4 { add(ng[s], &mut self.excl); }
            // 1-3: neighbors of neighbors
            for s in 0..4 {
                let ja = ng[s];
                if ja < 0 { continue; }
                let nj = self.neighs[ja as usize].as_array();
                for t in 0..4 { add(nj[t], &mut self.excl); }
            }
            // Sort the filled portion
            self.excl[i0..i0 + n].sort_unstable();
        }
    }

    /// Check if atom `j` is in the exclusion list of atom `i`.
    #[inline(always)]
    pub fn is_excluded(&self, i: usize, j: i32) -> bool {
        if j < 0 || j == i as i32 { return true; }
        let i0 = i * EXCL_MAX;
        // Linear search in exclusion list (only filled entries, i.e. >= 0)
        // The list is sorted but padded with -1, so binary search on the full slice fails.
        for k in 0..EXCL_MAX {
            let v = self.excl[i0 + k];
            if v < 0 { break; }  // end of filled entries
            if v == j { return true; }
            if v > j { break; }  // sorted → no match possible
        }
        false
    }

    /// Compute isotropic scalar inertia from bond lengths (same as rigid_sp3.rs:202-218).
    pub fn compute_inertia(&mut self) {
        for i in 0..self.natoms {
            let np = self.nport[i] as usize;
            if np == 0 { self.inv_inertia[i] = 0.0; continue; }
            let mut sum_l2 = 0.0;
            let mut n = 0.0;
            let bs = self.neigh_bs[i].as_array();
            for s in 0..np {
                let ib = bs[s];
                if ib < 0 { continue; }
                let l0 = self.bond_params[ib as usize].l0;
                sum_l2 += l0 * l0;
                n += 1.0;
            }
            if n > 0.0 {
                let l2 = sum_l2 / n;
                let i_mom = 0.4 * l2 + 1e-18; // approximate moment of inertia
                self.inv_inertia[i] = 1.0 / i_mom;
            }
        }
    }

    /// Get port tip in world frame: tip = x_i + R_i · (l0 · a_α)
    #[inline(always)]
    pub fn port_tip(&self, state: &RaffState, i: usize, slot: usize) -> Vec3d {
        let ib = self.neigh_bs[i].as_array()[slot] as usize;
        let l0 = self.bond_params[ib].l0;
        let r0 = self.port_local[i * 4 + slot] * l0;
        state.pos[i] + quat_rotate(state.quat[i], r0)
    }
}

impl RaffState {
    pub fn new(natoms: usize) -> Self {
        Self {
            natoms,
            pos: vec![VEC3D_ZERO; natoms],
            quat: vec![Quat4d::new(0.0, 0.0, 0.0, 1.0); natoms],
            vel: vec![VEC3D_ZERO; natoms],
            omega: vec![VEC3D_ZERO; natoms],
        }
    }

    pub fn set_positions(&mut self, apos: &[Vec3d]) {
        assert_eq!(apos.len(), self.natoms);
        self.pos.copy_from_slice(apos);
    }
}

// ==================================================================
//  §1: Port energy and forces (corrected convention)
// ==================================================================

/// Evaluate port forces and energy. Returns total port energy.
/// Fills fapos and tau (scratch arrays passed by caller).
///
/// Convention (§1 corrected): E = k_p/2 |e|², F = k_p · e
/// where e = x_j - tip_i (port error vector).
/// For reciprocal ports (two per bond), k_p = K_bond/2, so the pair gives F_total = K_bond · e.
pub fn eval_port_forces(
    state: &RaffState, topo: &RaffTopology,
    fapos: &mut [Vec3d], tau: &mut [Vec3d],
) -> f64 {
    for f in fapos.iter_mut() { *f = VEC3D_ZERO; }
    for t in tau.iter_mut() { *t = VEC3D_ZERO; }
    let mut e_total = 0.0;
    for i in 0..topo.natoms {
        let np = topo.nport[i] as usize;
        if np == 0 { continue; }
        let xi = state.pos[i];
        let q = state.quat[i];
        let ns = topo.neighs[i].as_array();
        let bs = topo.neigh_bs[i].as_array();
        for s in 0..np {
            let j = ns[s];
            if j < 0 { continue; }
            let ib = bs[s];
            if ib < 0 { continue; }
            let par = topo.bond_params[ib as usize];
            if par.k_p <= 0.0 { continue; }
            let r0 = topo.port_local[i * 4 + s] * par.l0;
            let r_arm = quat_rotate(q, r0);           // rotated port arm (world frame)
            let tip = xi + r_arm;
            let e_vec = state.pos[j as usize] - tip;   // port error e = x_j - tip
            let f = e_vec * par.k_p;                   // F = k_p · e (corrected: no 0.5 factor)
            fapos[i].add(f);
            fapos[j as usize].sub(f);
            tau[i].add(Vec3d::cross(r_arm, f));
            e_total += 0.5 * par.k_p * e_vec.norm2();  // E = k_p/2 |e|²
        }
    }
    e_total
}

// ==================================================================
//  §2.2: Analytical rotation — Wahba problem (corrected)
// ==================================================================

/// Solve for optimal rotation R_i* = argmin_R Σ k_α |d_α - R r_α|²
/// using the cross-covariance H = Σ k_α d_α r_α^T (NO centroid subtraction).
/// Returns the optimal quaternion.
///
/// Uses Newton-Schulz polar decomposition: R* = polar(H), then converts to quaternion.
/// This is more robust than Horn K-matrix power iteration (which can converge to the
/// wrong eigenvector when the largest-magnitude eigenvalue is negative).
/// d_α = x_j - x_i (neighbor direction), r_α = l_α · a_α (body-frame port arm).
pub fn solve_rotation_wahba(
    i: usize, state: &RaffState, topo: &RaffTopology,
) -> Quat4d {
    let np = topo.nport[i] as usize;
    if np == 0 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); }
    let xi = state.pos[i];
    let ns = topo.neighs[i].as_array();
    let bs = topo.neigh_bs[i].as_array();

    // Special case: single port — find rotation aligning port direction with neighbor direction.
    // The Wahba problem is rank-1 here, so polar decomposition is unreliable.
    if np == 1 {
        let j = ns[0];
        if j < 0 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); }
        let ib = bs[0];
        if ib < 0 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); }
        let _l0 = topo.bond_params[ib as usize].l0;
        let a_body = topo.port_local[i * 4];     // body-frame port direction (unit)
        let d_world = state.pos[j as usize] - xi; // world-frame neighbor direction
        let d_norm = d_world.norm();
        if d_norm < 1e-12 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); }
        let d_unit = d_world * (1.0 / d_norm);
        let a_unit = a_body * (1.0 / a_body.norm());
        return quat_align_vectors(a_unit, d_unit);
    }

    // Build cross-covariance M = Σ k_α r_α d_α^T (3×3, no centering).
    // NOTE: Wahba minimizes Σ |d_α - R r_α|², optimal R = polar(M) where M = Σ k r d^T.
    // (NOT H = Σ k d r^T — that gives the inverse rotation.)
    let mut h = Mat3d::zero();
    for s in 0..np {
        let j = ns[s];
        if j < 0 { continue; }
        let ib = bs[s];
        if ib < 0 { continue; }
        let k = topo.bond_params[ib as usize].k_p;
        if k <= 0.0 { continue; }
        let d = state.pos[j as usize] - xi;           // neighbor direction (no centering)
        let r = topo.port_local[i * 4 + s] * topo.bond_params[ib as usize].l0; // body-frame arm
        h.add_outer(r * k, d);                         // M += k * r * d^T
    }

    // Horn K-matrix method with shifted power iteration.
    // The K-matrix has eigenvalues summing to 0, so the largest-magnitude eigenvalue
    // can be negative. We shift K' = K + cI to make all eigenvalues positive,
    // ensuring power iteration converges to the largest (optimal rotation).
    let (hxx, hxy, hxz) = (h.a.x, h.a.y, h.a.z);
    let (hyx, hyy, hyz) = (h.b.x, h.b.y, h.b.z);
    let (hzx, hzy, hzz) = (h.c.x, h.c.y, h.c.z);
    let tr = hxx + hyy + hzz;
    let mut k = [0.0f64; 16];
    k[0] = tr;                     k[1] = hyz - hzy;          k[2] = hzx - hxz;          k[3] = hxy - hyx;
    k[4] = hyz - hzy;              k[5] = hxx - hyy - hzz;    k[6] = hxy + hyx;          k[7] = hzx + hxz;
    k[8] = hzx - hxz;              k[9] = hxy + hyx;          k[10] = hyy - hxx - hzz;   k[11] = hyz + hzy;
    k[12] = hxy - hyx;             k[13] = hzx + hxz;         k[14] = hyz + hzy;         k[15] = hzz - hxx - hyy;
    // Shift: c = 2 * Frobenius norm of K (upper bound on |eigenvalues|)
    let k_frob = k.iter().map(|x| x * x).sum::<f64>().sqrt();
    let shift = 2.0 * k_frob;
    for i in 0..4 { k[i*5] += shift; }  // add shift to diagonal

    // Power iteration on shifted K. K-matrix uses [w, x, y, z] ordering (scalar first).
    let q_warm = state.quat[i];
    let mut q = [q_warm.w, q_warm.x, q_warm.y, q_warm.z];  // [w, x, y, z]
    { let n = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]).sqrt(); if n > 1e-30 { for x in q.iter_mut() { *x /= n; } } }
    for _ in 0..64 {
        let mut qnew = [0.0f64; 4];
        for row in 0..4 {
            for col in 0..4 { qnew[row] += k[row*4+col] * q[col]; }
        }
        let n = (qnew[0]*qnew[0]+qnew[1]*qnew[1]+qnew[2]*qnew[2]+qnew[3]*qnew[3]).sqrt();
        if n < 1e-30 { break; }
        let inv = 1.0 / n;
        let mut max_delta: f64 = 0.0;
        for idx in 0..4 {
            qnew[idx] *= inv;
            max_delta = max_delta.max((qnew[idx] - q[idx]).abs());
            q[idx] = qnew[idx];
        }
        if max_delta < 1e-14 { break; }
    }
    // Horn K-matrix eigenvector is [w, x, y, z] (scalar first), but Quat4d is (x, y, z, w)
    Quat4d::new(q[1], q[2], q[3], q[0])
}

/// Convert 3×3 rotation matrix to quaternion (x, y, z, w).
/// Uses the standard algorithm: find largest diagonal element, compute others from it.
fn mat3_to_quat(m: Mat3d) -> Quat4d {
    let trace = m.a.x + m.b.y + m.c.z;
    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        let w = 0.25 / s;
        let x = (m.b.z - m.c.y) * s;
        let y = (m.c.x - m.a.z) * s;
        let z = (m.a.y - m.b.x) * s;
        Quat4d::new(x, y, z, w)
    } else if m.a.x > m.b.y && m.a.x > m.c.z {
        let s = 0.5 / (1.0 + m.a.x - m.b.y - m.c.z).sqrt();
        let x = 0.25 / s;
        let y = (m.b.x + m.a.y) * s;
        let z = (m.c.x + m.a.z) * s;
        let w = (m.b.z - m.c.y) * s;
        Quat4d::new(x, y, z, w)
    } else if m.b.y > m.c.z {
        let s = 0.5 / (1.0 + m.b.y - m.a.x - m.c.z).sqrt();
        let x = (m.b.x + m.a.y) * s;
        let y = 0.25 / s;
        let z = (m.c.y + m.b.z) * s;
        let w = (m.c.x - m.a.z) * s;
        Quat4d::new(x, y, z, w)
    } else {
        let s = 0.5 / (1.0 + m.c.z - m.a.x - m.b.y).sqrt();
        let x = (m.c.x + m.a.z) * s;
        let y = (m.c.y + m.b.z) * s;
        let z = 0.25 / s;
        let w = (m.a.y - m.b.x) * s;
        Quat4d::new(x, y, z, w)
    }
}

/// Solve rotation for all atoms (adiabatic mode).
pub fn solve_all_rotations(state: &mut RaffState, topo: &RaffTopology) {
    for i in 0..topo.natoms {
        if topo.nport[i] == 0 { continue; }
        state.quat[i] = solve_rotation_wahba(i, state, topo);
    }
}

// ==================================================================
//  §4: Non-bonded interactions (LJ + Coulomb + collision repulsion)
// ==================================================================

const COULOMB_CONST: f64 = 14.3996448915; // eV·Å / e²

/// Evaluate non-bonded forces and energy. O(N²) pair loop with exclusion check.
/// Accumulates forces into `fapos` (does NOT zero — caller decides).
/// Returns total non-bonded energy.
///
/// Physics:
/// - Lennard-Jones: E = 4ε[(σ/r)^12 - (σ/r)^6], F = -dE/dr · r̂
/// - Coulomb: E = k·q_i·q_j/r, F = k·q_i·q_j/r² · r̂
/// - Collision (soft sphere): when r < R_i + R_j, E = k_coll/2 · (R_i+R_j - r)²
///   (only if radius > 0 for both atoms; separate from LJ)
///
/// Exclusions: 1-2 and 1-3 pairs are skipped (controlled by NbConfig flags).
/// Ported from nonbonded.rs:get_ljqh + RRsp3.cl:compute_collision_cluster_rigid.
pub fn eval_nonbonded(
    state: &RaffState, topo: &RaffTopology, nbcfg: &NbConfig,
    fapos: &mut [Vec3d],
) -> f64 {
    if !nbcfg.enabled { return 0.0; }
    let rcut2 = nbcfg.rcut * nbcfg.rcut;
    let r_damp2 = nbcfg.r_damp * nbcfg.r_damp;
    let f2max = nbcfg.f_max * nbcfg.f_max;
    let mut e_total = 0.0;
    for i in 0..topo.natoms {
        let pi = state.pos[i];
        let nbi = topo.nb_params[i];
        let mi = topo.mass[i];
        for j in (i+1)..topo.natoms {
            // Exclusion check
            if topo.is_excluded(i, j as i32) { continue; }
            let dp = state.pos[j] - pi;
            let r2 = dp.norm2();
            if r2 > rcut2 { continue; }
            if r2 < 1e-16 { continue; }  // skip coincident atoms
            let r = r2.sqrt();
            let inv_r = 1.0 / r;
            let n_hat = dp * inv_r;  // unit vector i→j
            let nbj = topo.nb_params[j];
            let mut e_ij = 0.0;
            let mut f_scalar = 0.0;  // positive = repulsive (pushes i in -n_hat, j in +n_hat)

            // Lennard-Jones (Lorentz-Berthelot mixing)
            if nbi.epsilon > 0.0 && nbj.epsilon > 0.0 {
                let sigma = 0.5 * (nbi.sigma + nbj.sigma);
                let eps = (nbi.epsilon * nbj.epsilon).sqrt();
                let sr6 = (sigma * inv_r).powi(6);
                let sr12 = sr6 * sr6;
                // E = 4ε(σ^12/r^12 - σ^6/r^6)
                e_ij += 4.0 * eps * (sr12 - sr6);
                // F = -dE/dr = 24ε/r · (2σ^12/r^12 - σ^6/r^6)  (positive = repulsive)
                f_scalar += 24.0 * eps * inv_r * (2.0 * sr12 - sr6);
            }

            // Coulomb
            if nbi.charge != 0.0 && nbj.charge != 0.0 {
                let r_damped = (r2 + r_damp2).sqrt();
                let e_coul = COULOMB_CONST * nbi.charge * nbj.charge / r_damped;
                e_ij += e_coul;
                // F = -dE/dr = k·q_i·q_j/r² (positive = repulsive for same-sign charges)
                f_scalar += COULOMB_CONST * nbi.charge * nbj.charge / (r_damped * r_damped);
            }

            // Soft-sphere collision (separate from LJ, for hard-sphere overlap)
            if nbi.radius > 0.0 && nbj.radius > 0.0 {
                let rsum = nbi.radius + nbj.radius;
                if r < rsum {
                    let overlap = rsum - r;
                    e_ij += 0.5 * nbcfg.k_coll * overlap * overlap;
                    // F = -dE/dr = k_coll · (rsum - r)  (positive = repulsive)
                    f_scalar += nbcfg.k_coll * overlap;
                }
            }

            if f_scalar.abs() > 1e-30 {
                // Force on i is -f_scalar · n_hat (repulsive pushes i away from j)
                let mut fi = n_hat * (-f_scalar);
                // Clamp force magnitude
                if nbcfg.f_max > 0.0 {
                    let f2 = fi.norm2();
                    if f2 > f2max { fi.mul(f2max.sqrt() / f2.sqrt()); }
                }
                fapos[i].add(fi);
                fapos[j].sub(fi);  // Newton's 3rd law
            }
            e_total += e_ij;
            let _ = mi;  // mass not needed for force eval, only for integration
        }
    }
    e_total
}

// ==================================================================
//  §3.1: Force-based MD step
// ==================================================================

/// One force-based MD step for all atoms. Returns (total_energy, max_force, max_torque).
/// Uses symplectic Euler: v ← cdamp·v + F/m·dt; x ← x + v·dt.
/// Rotation: ω ← rot_damp·ω + I⁻¹·τ·dt; q ← normalize(dq(ω·dt) ⊗ q).
/// If `nbcfg.enabled`, non-bonded forces (LJ + Coulomb + collision) are added to fapos.
pub fn step_force_md(
    state: &mut RaffState, topo: &RaffTopology, cfg: &RaffConfig,
    fapos: &mut [Vec3d], tau: &mut [Vec3d],
    nbcfg: &NbConfig,
) -> (f64, f64, f64) {
    let e_port = eval_port_forces(state, topo, fapos, tau);
    let e_nb = eval_nonbonded(state, topo, nbcfg, fapos);
    let e = e_port + e_nb;
    let mut max_f: f64 = 0.0;
    let mut max_t: f64 = 0.0;
    for i in 0..topo.natoms {
        // Translation
        let mut f = fapos[i];
        let f2 = f.norm2();
        if cfg.flim > 0.0 && f2 > cfg.flim * cfg.flim {
            f.mul(cfg.flim / f2.sqrt());
        }
        max_f = max_f.max(f2.sqrt());
        let mut v = state.vel[i];
        v.mul(cfg.cdamp);
        v.add_mul(f, cfg.dt / topo.mass[i]);
        state.vel[i] = v;
        state.pos[i].add_mul(v, cfg.dt);

        // Rotation (only if atom has ports and inertia)
        if topo.inv_inertia[i] > 0.0 && topo.nport[i] > 0 {
            let t = tau[i];
            max_t = max_t.max(t.norm());
            let mut w = state.omega[i];
            w.mul(cfg.rot_damp);
            w.add_mul(t, topo.inv_inertia[i] * cfg.dt);
            state.omega[i] = w;
            let dq = quat_from_omega_dt(w, cfg.dt);
            state.quat[i] = quat_normalize(quat_mul(dq, state.quat[i]));
        }
    }
    (e, max_f, max_t)
}

// ==================================================================
//  §3.2: XPBD port constraint solver (corrected)
// ==================================================================

/// One XPBD step for all port constraints. Constraint: C = |x_j - tip_i| = 0.
/// Compliance: α̃ = 1/(k_p · dt²). Iterates xpbd_iters times (Gauss-Seidel).
///
/// For dynamic orientation: distributes impulse into linear (Δx) and angular (Δθ).
/// For adiabatic orientation: re-solves R_i* after position updates.
///
/// Solve collision constraints (position-based, PBD style).
/// For each overlapping non-excluded pair (i,j): push apart proportional to inv_mass.
/// Constraint: C = r_i + r_j - |x_j - x_i| > 0 → resolve by moving along n = (x_j - x_i)/|·|.
/// Ported from RRsp3.cl:compute_collision_cluster_rigid (CPU version, no ghost atoms).
pub fn solve_collisions(
    state: &mut RaffState, topo: &RaffTopology, nbcfg: &NbConfig,
) {
    for i in 0..topo.natoms {
        let ri = topo.nb_params[i].radius;
        if ri <= 0.0 { continue; }
        let inv_mi = 1.0 / topo.mass[i];
        for j in (i+1)..topo.natoms {
            if topo.is_excluded(i, j as i32) { continue; }
            let rj = topo.nb_params[j].radius;
            if rj <= 0.0 { continue; }
            let dp = state.pos[j] - state.pos[i];
            let d2 = dp.norm2();
            let rsum = ri + rj;
            if d2 < rsum * rsum && d2 > 1e-16 {
                let dist = d2.sqrt();
                let n = dp * (1.0 / dist);
                let inv_mj = 1.0 / topo.mass[j];
                let w_tot = inv_mi + inv_mj + 1e-12;
                // PBD: push apart by overlap / w_tot, split by inverse mass
                let overlap = rsum - dist;
                let lambda = overlap / w_tot * 0.5;  // 0.5 = relaxation factor
                // i moves -n (away from j), j moves +n (away from i)
                state.pos[i].add_mul(n, -lambda * inv_mi);
                state.pos[j].add_mul(n,  lambda * inv_mj);
            }
        }
    }
}

/// Full XPBD step: (1) predict x ← x + v·dt, (2) solve port constraints, (3) solve collisions,
/// (4) v ← cdamp·(x_new - x_old)/dt. For relaxation (cdamp=0), pure constraint projection.
pub fn step_xpbd(
    state: &mut RaffState, topo: &RaffTopology, cfg: &RaffConfig,
    nbcfg: &NbConfig,
) -> f64 {
    let dt2 = cfg.dt * cfg.dt;
    let np = topo.natoms;

    // For adiabatic mode: solve rotations first (before any position changes)
    if cfg.orient_mode == OrientMode::Adiabatic {
        solve_all_rotations(state, topo);
    }

    // Save old positions for velocity update
    let pos_old = state.pos.clone();

    // (1) Predict: x_pred = x + v·dt (skip if cdamp=0 for pure relaxation)
    if cfg.cdamp > 0.0 {
        for i in 0..np {
            state.pos[i].add_mul(state.vel[i], cfg.dt);
        }
    }

    // (2) Solve constraints iteratively
    for iter in 0..cfg.xpbd_iters {
        for i in 0..topo.natoms {
            let npi = topo.nport[i] as usize;
            if npi == 0 { continue; }
            let ns = topo.neighs[i].as_array();
            let bs = topo.neigh_bs[i].as_array();
            for s in 0..npi {
                let j = ns[s];
                if j < 0 { continue; }
                let ib = bs[s];
                if ib < 0 { continue; }
                let par = topo.bond_params[ib as usize];
                if par.k_p <= 0.0 { continue; }

                let r0 = topo.port_local[i * 4 + s] * par.l0;
                let r_arm = quat_rotate(state.quat[i], r0);
                let tip = state.pos[i] + r_arm;
                let diff = state.pos[j as usize] - tip;  // e = x_j - tip
                let r = diff.norm();
                if r < 1e-12 { continue; }
                let n = diff * (1.0 / r);  // unit direction

                // XPBD denominator (§3.2 corrected): w_total = 1/m_i + 1/m_j + w_ang + α̃
                let inv_mi = 1.0 / topo.mass[i];
                let inv_mj = 1.0 / topo.mass[j as usize];
                let rxn = Vec3d::cross(r_arm, n);
                let w_ang = if cfg.orient_mode == OrientMode::Dynamic { rxn.norm2() * topo.inv_inertia[i] } else { 0.0 };
                let alpha_tilde = 1.0 / (par.k_p * dt2);
                let w_total = inv_mi + inv_mj + w_ang + alpha_tilde + 1e-12;

                let c = r;  // C = |x_j - tip| = 0
                let lambda = c / w_total * cfg.xpbd_over_relax;

                // Position corrections: n points tip→x_j, so to reduce C:
                // x_i moves +n (tip follows, toward x_j), x_j moves -n (toward tip)
                state.pos[i].add_mul(n, lambda * inv_mi);
                state.pos[j as usize].add_mul(n, -lambda * inv_mj);

                // Rotation correction (dynamic mode only): rotate tip toward x_j
                if cfg.orient_mode == OrientMode::Dynamic && topo.inv_inertia[i] > 0.0 {
                    let dtheta = Vec3d::cross(rxn, n) * (-lambda * topo.inv_inertia[i]);
                    let dq = quat_from_omega_dt(dtheta, 1.0);
                    state.quat[i] = quat_normalize(quat_mul(dq, state.quat[i]));
                }
            }
        }
        // Adiabatic: re-solve rotations after position updates
        if cfg.orient_mode == OrientMode::Adiabatic {
            solve_all_rotations(state, topo);
        }

        // Collision constraints (position-based, same iteration as ports)
        if nbcfg.enabled && nbcfg.k_coll > 0.0 {
            solve_collisions(state, topo, nbcfg);
        }
        let _ = iter;
    }

    // (3) Velocity update: v = cdamp * (x_new - x_old) / dt
    // Damping prevents velocity buildup from repeated constraint corrections.
    for i in 0..np {
        state.vel[i] = (state.pos[i] - pos_old[i]) * (cfg.cdamp / cfg.dt);
    }

    // Report final energy
    let mut fapos = vec![VEC3D_ZERO; np];
    let mut tau = vec![VEC3D_ZERO; np];
    eval_port_forces(state, topo, &mut fapos, &mut tau)
}

// ==================================================================
//  §11: Common proximal problem (IMEX step)
// ==================================================================

/// One IMEX macrostep: compute soft force → target y, then solve hard problem implicitly.
///
/// 1. F_s = -∇U_s(x^n)  [soft force, e.g. nonbonded — here just port forces for testing]
/// 2. y = x^n + H·v^n + H²·M⁻¹·F_s^n  [inertial target]
/// 3. Solve x^{n+1} = argmin_x [1/(2H²)(x-y)^T M (x-y) + U_h(x,R)]  [via XPBD or PD]
///
/// Currently U_s = U_h = port energy (no split yet). This is the framework for future
/// nonbonded splitting: soft = long-range Coulomb + attractive tail, hard = ports + repulsion.
pub fn step_proximal(
    state: &mut RaffState, topo: &RaffTopology, cfg: &RaffConfig,
    fapos: &mut [Vec3d], tau: &mut [Vec3d],
    nbcfg: &NbConfig,
    macro_H: f64,
) -> f64 {
    // 1. Compute soft forces (currently port forces — will be split later)
    let e_soft = eval_port_forces(state, topo, fapos, tau);

    // 2. Compute inertial target y = x + H·v + H²·F_s/m
    let h2 = macro_H * macro_H;
    let mut y = vec![VEC3D_ZERO; topo.natoms];
    for i in 0..topo.natoms {
        y[i] = state.pos[i];
        y[i].add_mul(state.vel[i], macro_H);
        y[i].add_mul(fapos[i], h2 / topo.mass[i]);
    }

    // 3. Save old positions, then solve hard problem via XPBD
    //    The target y enters as a "spring" pulling toward the inertial prediction.
    //    For now, we use XPBD on port constraints with the target as anchor.
    let pos_old = state.pos.clone();

    // Apply inertial target as velocity: v ← (y - x) / H, then XPBD corrects
    for i in 0..topo.natoms {
        state.vel[i] = (y[i] - state.pos[i]) * (1.0 / macro_H);
        state.pos[i].add_mul(state.vel[i], macro_H);
    }

    // XPBD correction on port constraints
    let e_hard = step_xpbd(state, topo, cfg, nbcfg);

    // Update velocity from total position change
    for i in 0..topo.natoms {
        state.vel[i] = (state.pos[i] - pos_old[i]) * (1.0 / macro_H);
    }

    e_soft + e_hard * 0.0 // energy reporting (avoid double-count)
}

// ==================================================================
//  §11.7: Finite-difference validation utilities
// ==================================================================

/// Finite-difference check: compare analytic force to (E(x+ε) - E(x-ε)) / (2ε).
/// Returns max relative error over all atoms and directions.
/// Tests §1 corrected force convention: F = k_p · e.
pub fn fd_check_forces(
    state: &RaffState, topo: &RaffTopology, eps: f64,
) -> (f64, Vec<(usize, Vec3d, Vec3d)>) {
    let np = topo.natoms;
    let mut fapos = vec![VEC3D_ZERO; np];
    let mut tau = vec![VEC3D_ZERO; np];
    let e0 = eval_port_forces(state, topo, &mut fapos, &mut tau);

    let mut max_err: f64 = 0.0;
    let mut details: Vec<(usize, Vec3d, Vec3d)> = Vec::new();

    for i in 0..np {
        for d in 0..3 {
            let mut pos_plus = state.pos.clone();
            let mut pos_minus = state.pos.clone();
            match d {
                0 => { pos_plus[i].x += eps; pos_minus[i].x -= eps; }
                1 => { pos_plus[i].y += eps; pos_minus[i].y -= eps; }
                2 => { pos_plus[i].z += eps; pos_minus[i].z -= eps; }
                _ => unreachable!(),
            }
            let st_plus = RaffState { natoms: np, pos: pos_plus, quat: state.quat.clone(), vel: state.vel.clone(), omega: state.omega.clone() };
            let st_minus = RaffState { natoms: np, pos: pos_minus, quat: state.quat.clone(), vel: state.vel.clone(), omega: state.omega.clone() };
            let mut fp = vec![VEC3D_ZERO; np];
            let mut tp = vec![VEC3D_ZERO; np];
            let mut fm = vec![VEC3D_ZERO; np];
            let mut tm = vec![VEC3D_ZERO; np];
            let e_plus = eval_port_forces(&st_plus, topo, &mut fp, &mut tp);
            let e_minus = eval_port_forces(&st_minus, topo, &mut fm, &mut tm);
            let fd_force = -(e_plus - e_minus) / (2.0 * eps);  // F = -dE/dx
            let analytic = match d { 0 => fapos[i].x, 1 => fapos[i].y, _ => fapos[i].z };
            let rel_err = if analytic.abs() > 1e-10 { (fd_force - analytic).abs() / analytic.abs() }
                          else { (fd_force - analytic).abs() };
            if rel_err > 1e-6 {
                details.push((i, Vec3d::new(fd_force, analytic, rel_err), Vec3d::new(0.0,0.0,0.0)));
            }
            max_err = max_err.max(rel_err);
        }
    }
    let _ = e0;
    (max_err, details)
}

/// Finite-difference check: compare analytic torque to rotation derivative.
/// Tests §1 torque convention: τ = r × F, and dE/dθ = -τ.
/// Perturbs by small rotation angle δθ (not quaternion component directly).
pub fn fd_check_torques(
    state: &RaffState, topo: &RaffTopology, eps: f64,
) -> f64 {
    let np = topo.natoms;
    let mut fapos = vec![VEC3D_ZERO; np];
    let mut tau = vec![VEC3D_ZERO; np];
    eval_port_forces(state, topo, &mut fapos, &mut tau);

    let mut max_err: f64 = 0.0;
    for i in 0..np {
        if topo.nport[i] == 0 { continue; }
        // Perturb by small rotation δθ around each axis, then check dE/dθ vs -τ
        for axis in 0..3 {
            let dtheta = match axis { 0 => Vec3d::new(eps, 0.0, 0.0), 1 => Vec3d::new(0.0, eps, 0.0), _ => Vec3d::new(0.0, 0.0, eps) };
            let dq = quat_from_omega_dt(dtheta, 1.0); // δθ as quaternion
            let q_plus = quat_normalize(quat_mul(dq, state.quat[i]));
            let q_minus = quat_normalize(quat_mul(quat_conj(dq), state.quat[i]));
            let mut qp = state.quat.clone(); qp[i] = q_plus;
            let mut qm = state.quat.clone(); qm[i] = q_minus;
            let st_plus = RaffState { natoms: np, pos: state.pos.clone(), quat: qp, vel: state.vel.clone(), omega: state.omega.clone() };
            let st_minus = RaffState { natoms: np, pos: state.pos.clone(), quat: qm, vel: state.vel.clone(), omega: state.omega.clone() };
            let mut fp = vec![VEC3D_ZERO; np];
            let mut tp = vec![VEC3D_ZERO; np];
            let mut fm = vec![VEC3D_ZERO; np];
            let mut tm = vec![VEC3D_ZERO; np];
            let e_plus = eval_port_forces(&st_plus, topo, &mut fp, &mut tp);
            let e_minus = eval_port_forces(&st_minus, topo, &mut fm, &mut tm);
            // dE/dθ = -τ (torque drives rotation, energy decreases). FD: -dE/dθ = +τ.
            let fd_dedtheta = -(e_plus - e_minus) / (2.0 * eps);
            let analytic = match axis { 0 => tau[i].x, 1 => tau[i].y, _ => tau[i].z };
            // fd_dedtheta = -dE/dθ = +τ, so check fd_dedtheta - τ ≈ 0
            let rel_err = if analytic.abs() > 1e-10 { (fd_dedtheta - analytic).abs() / analytic.abs() }
                          else { (fd_dedtheta - analytic).abs() };
            max_err = max_err.max(rel_err);
        }
    }
    max_err
}

/// Check global translation invariance: Σ F_i should be 0.
pub fn check_translation_invariance(
    state: &RaffState, topo: &RaffTopology,
) -> f64 {
    let np = topo.natoms;
    let mut fapos = vec![VEC3D_ZERO; np];
    let mut tau = vec![VEC3D_ZERO; np];
    eval_port_forces(state, topo, &mut fapos, &mut tau);
    let mut sum = VEC3D_ZERO;
    for f in &fapos { sum.add(*f); }
    sum.norm()
}

/// Check global rotation invariance: Σ x_i × F_i + τ_i should be 0.
pub fn check_rotation_invariance(
    state: &RaffState, topo: &RaffTopology,
) -> f64 {
    let np = topo.natoms;
    let mut fapos = vec![VEC3D_ZERO; np];
    let mut tau = vec![VEC3D_ZERO; np];
    eval_port_forces(state, topo, &mut fapos, &mut tau);
    let mut sum = VEC3D_ZERO;
    for i in 0..np {
        sum.add(Vec3d::cross(state.pos[i], fapos[i]));
        sum.add(tau[i]);
    }
    sum.norm()
}

/// Check adiabatic rotation torque residual: at exact R_i*, τ_i should be ~0 per atom.
/// (§2.3 corrected: envelope theorem gives Σ r_α × F_α = 0 at convergence)
pub fn check_adiabatic_torque_residual(
    state: &mut RaffState, topo: &RaffTopology,
) -> Vec<f64> {
    let np = topo.natoms;
    let mut fapos = vec![VEC3D_ZERO; np];
    let mut tau = vec![VEC3D_ZERO; np];
    // Solve rotations to convergence, tracking energy
    for iter in 0..20 {
        solve_all_rotations(state, topo);
        let e = eval_port_forces(state, topo, &mut fapos, &mut tau);
        let max_t = tau.iter().map(|t| t.norm()).fold(0.0f64, f64::max);
        if iter < 5 || iter % 5 == 0 {
            eprintln!("[adiabatic] iter {}: E = {:.6e}, max|τ| = {:.4e}", iter, e, max_t);
        }
    }
    eval_port_forces(state, topo, &mut fapos, &mut tau);
    tau.iter().map(|t| t.norm()).collect()
}

// ==================================================================
//  Convenience: full relaxation loop
// ==================================================================

/// Relax using force-based MD with damping. Returns (final_energy, n_steps, converged).
pub fn relax_force_md(
    state: &mut RaffState, topo: &RaffTopology, cfg: &RaffConfig, nbcfg: &NbConfig,
    max_steps: usize, f_tol: f64, t_tol: f64,
) -> (f64, usize, bool) {
    let np = topo.natoms;
    let mut fapos = vec![VEC3D_ZERO; np];
    let mut tau = vec![VEC3D_ZERO; np];
    let mut last_e = f64::INFINITY;
    for step in 0..max_steps {
        let (e, max_f, max_t) = step_force_md(state, topo, cfg, &mut fapos, &mut tau, nbcfg);
        if step % 100 == 0 {
            eprintln!("[relax_force_md] step {} E={:.6} max|F|={:.6} max|τ|={:.6}", step, e, max_f, max_t);
        }
        if max_f < f_tol && max_t < t_tol {
            return (e, step + 1, true);
        }
        last_e = e;
    }
    (last_e, max_steps, false)
}

/// Relax using XPBD. Returns (final_energy, n_steps, converged).
pub fn relax_xpbd(
    state: &mut RaffState, topo: &RaffTopology, cfg: &RaffConfig, nbcfg: &NbConfig,
    max_steps: usize, e_tol: f64,
) -> (f64, usize, bool) {
    let mut last_e = f64::INFINITY;
    for step in 0..max_steps {
        let e = step_xpbd(state, topo, cfg, nbcfg);
        if step % 100 == 0 {
            eprintln!("[relax_xpbd] step {} E={:.6}", step, e);
        }
        if (last_e - e).abs() < e_tol && step > 10 {
            return (e, step + 1, true);
        }
        last_e = e;
    }
    (last_e, max_steps, false)
}
