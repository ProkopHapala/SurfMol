//! Multigrid solver for linearized molecular elasticity.
//!
//! Ported from NumericalMathPlayground/topics/LinarElasticity/{MultiGrid.py, TrussSolver.py}.
//! Solves the SPD linear system  A·x = b  where  A = diag(M/Δt²) + K  and K is the
//! stiffness matrix. Two operators are provided:
//! - `TrussOp`: bond-only axial-spring stiffness (k_eff·n⊗n per bond). Fast, matrix-free.
//! - `UffHessianOp`: full UFF Hessian (bonds + angles + dihedrals + inversions) via finite
//!   differences. Captures aromatic bending/torsion stiffness — required for physically
//!   meaningful multigrid on aromatic molecules like pentacene.
//!
//! Phase 1 (this module): UFF / point-based, 3 DOF/atom. See
//! `notes/designs/2026-08-29_modal_relaxation_design_spec.md` for the full spec.
//!
//! V-cycle: pre-smooth (damped block Jacobi) → restrict residual (r_c = Pᵀr) →
//! coarse solve (dense Cholesky on A_c = PᵀAP) → prolongate (x += P·e_c) → post-smooth.
//!
//! Prolongation strategies: geometric pivots (maximin BFS + inverse-distance), first.
//! Spectral (Lanczos lowest modes) deferred — needs a sparse eigensolver not yet in SurfMol.

use numtypes::{Vec3d, Quat4i};
use numcore::math::linalg::{cholesky_factor_f64, cholesky_solve_f64};
use crate::uff::Uff;

const DIM: usize = 3;

// ============================================================================
// LinearOp trait — abstract operator interface for multigrid solvers
// ============================================================================

/// Linear operator interface: A = K + diag(mass_dt2) where K is the stiffness matrix.
/// Implemented by `TrussOp` (bond-only) and `UffHessianOp` (full UFF Hessian).
/// All multigrid solver functions accept `&impl LinearOp`.
pub trait LinearOp {
    fn natoms(&self) -> usize;
    /// A·x for flat DOF vector x (natoms*DIM).
    fn matvec(&self, x: &[f64]) -> Vec<f64>;
    /// Diagonal 3×3 blocks D_i = (m_i/Δt²)·I + K_ii. Returns natoms×9 row-major.
    fn diagonal_blocks(&self) -> Vec<[f64; 9]>;
    /// Assemble dense A (natoms*DIM × natoms*DIM) for direct solve / parity tests.
    fn assemble_dense(&self) -> Vec<f64>;
}

// ============================================================================
// TrussOp — matrix-free operator A = diag(M/Δt²) + K (axial-spring stiffness)
// ============================================================================

/// Matrix-free bond-stretch operator. Ported from TrussSolver.compute_edge_data + matvec_A.
/// A = diag(mass_dt2) + K, where each bond e=(i,j) contributes k_eff·n⊗n to K_ii, K_jj
/// and -k_eff·n⊗n to K_ij. Fixed nodes get huge mass_dt2 (Dirichlet via penalty).
pub struct TrussOp {
    pub natoms: usize,
    pub ei: Vec<i32>,          // bond endpoint a (nedges)
    pub ej: Vec<i32>,          // bond endpoint b (nedges)
    pub k_eff: Vec<f64>,       // stiffness per edge (nedges) — for UFF: 2*k_bond
    pub n_dirs: Vec<Vec3d>,    // unit bond direction (x_j − x_i)/L (nedges)
    pub mass_dt2: Vec<f64>,    // M/Δt² per node (natoms); fixed → huge
}

impl TrussOp {
    /// Build from UFF bonds + positions. k_eff = 2·bon_params[k] (Hessian of E=k·(l−l0)²).
    /// `mass_dt2` is the per-node mass/Δt²; fixed nodes should already be set to a huge value.
    pub fn from_uff_bonds(uff: &Uff, apos: &[Vec3d], mass_dt2: &[f64]) -> Self {
        let natoms = uff.natoms as usize;
        assert_eq!(apos.len(), natoms, "TrussOp::from_uff_bonds: apos.len={} != natoms={}", apos.len(), natoms);
        assert_eq!(mass_dt2.len(), natoms, "TrussOp::from_uff_bonds: mass_dt2.len={} != natoms={}", mass_dt2.len(), natoms);
        let bonds = uff.bon_atoms.as_slice();
        let bon_params = uff.bon_params.as_slice();
        let nedges = bonds.len();
        let mut ei = Vec::with_capacity(nedges);
        let mut ej = Vec::with_capacity(nedges);
        let mut k_eff = Vec::with_capacity(nedges);
        let mut n_dirs = Vec::with_capacity(nedges);
        for b in bonds {
            let i = b[0] as usize;
            let j = b[1] as usize;
            let d = apos[j] - apos[i];
            let l = d.norm();
            let n = if l > 1e-12 { d * (1.0 / l) } else { Vec3d::new(1.0, 0.0, 0.0) };
            // Find bond param index: UFF stores params per bond in order matching bon_atoms.
            // bon_params index = position in bon_atoms. We iterate in order, so idx = current len.
            let ib = ei.len();  // bond param index = edge index (UFF keeps them aligned)
            let k_bond = bon_params[ib][0];
            ei.push(b[0]);
            ej.push(b[1]);
            k_eff.push(2.0 * k_bond);  // Hessian of E = k·(l−l0)² → 2k·n⊗n
            n_dirs.push(n);
        }
        Self { natoms, ei, ej, k_eff, n_dirs, mass_dt2: mass_dt2.to_vec() }
    }

    /// Build from raw bond lists + positions (for tests / non-UFF forcefields).
    pub fn from_bonds(bonds: &[[i32; 2]], k_bond: &[f64], apos: &[Vec3d], mass_dt2: &[f64]) -> Self {
        let natoms = apos.len();
        let nedges = bonds.len();
        assert_eq!(k_bond.len(), nedges, "TrussOp::from_bonds: k_bond.len={} != nedges={}", k_bond.len(), nedges);
        let mut ei = Vec::with_capacity(nedges);
        let mut ej = Vec::with_capacity(nedges);
        let mut k_eff = Vec::with_capacity(nedges);
        let mut n_dirs = Vec::with_capacity(nedges);
        for (b, &k) in bonds.iter().zip(k_bond) {
            let i = b[0] as usize;
            let j = b[1] as usize;
            let d = apos[j] - apos[i];
            let l = d.norm();
            let n = if l > 1e-12 { d * (1.0 / l) } else { Vec3d::new(1.0, 0.0, 0.0) };
            ei.push(b[0]);
            ej.push(b[1]);
            k_eff.push(2.0 * k);
            n_dirs.push(n);
        }
        Self { natoms, ei, ej, k_eff, n_dirs, mass_dt2: mass_dt2.to_vec() }
    }

    /// A·x for flat DOF vector x (natoms*DIM). Parity: TrussSolver.matvec_A_flat.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let n = self.natoms * DIM;
        assert_eq!(x.len(), n, "TrussOp::matvec: x.len={} != natoms*DIM={}", x.len(), n);
        let mut ax = vec![0.0f64; n];
        // Mass part: diag(mass_dt2) · x
        for i in 0..self.natoms {
            let m = self.mass_dt2[i];
            for d in 0..DIM { ax[i*DIM + d] = m * x[i*DIM + d]; }
        }
        // Stiffness part: for each edge, contrib = k_eff · (diff·n) · n
        for e in 0..self.ei.len() {
            let i = self.ei[e] as usize;
            let j = self.ej[e] as usize;
            let n = self.n_dirs[e];
            let ke = self.k_eff[e];
            let mut diff = [0.0f64; DIM];
            let mut dot = 0.0;
            for d in 0..DIM { diff[d] = x[i*DIM + d] - x[j*DIM + d]; dot += diff[d] * n.array()[d]; }
            let c = ke * dot;
            for d in 0..DIM {
                let nd = n.array()[d];
                ax[i*DIM + d] += c * nd;
                ax[j*DIM + d] -= c * nd;
            }
        }
        ax
    }

    /// Diagonal 3×3 blocks D_i = (m_i/Δt²)·I + Σ_e k_eff·n⊗n. Returns natoms×9 row-major.
    /// Parity: TrussSolver.compute_diagonal_3x3.
    pub fn diagonal_blocks(&self) -> Vec<[f64; 9]> {
        let mut d = vec![[0.0f64; 9]; self.natoms];
        for i in 0..self.natoms {
            let m = self.mass_dt2[i];
            d[i][0] = m; d[i][4] = m; d[i][8] = m;  // diagonal
        }
        for e in 0..self.ei.len() {
            let i = self.ei[e] as usize;
            let j = self.ej[e] as usize;
            let n = self.n_dirs[e].array();
            let ke = self.k_eff[e];
            // n⊗n outer product, add to both i and j
            for a in 0..DIM {
                for b in 0..DIM {
                    let nn = ke * n[a] * n[b];
                    d[i][a*DIM + b] += nn;
                    d[j][a*DIM + b] += nn;
                }
            }
        }
        d
    }

    /// Assemble dense A (natoms*DIM × natoms*DIM) for parity tests / direct solve.
    /// Fixed nodes (mass_dt2 huge) are NOT identity-rowed here — the penalty handles it.
    /// Parity: TrussSolver.assemble_dense_A (without fixed_nodes identity rows).
    pub fn assemble_dense(&self) -> Vec<f64> {
        let ndof = self.natoms * DIM;
        let mut mat = vec![0.0f64; ndof * ndof];
        for i in 0..self.natoms {
            let m = self.mass_dt2[i];
            for d in 0..DIM { mat[(i*DIM + d) * ndof + (i*DIM + d)] = m; }
        }
        for e in 0..self.ei.len() {
            let i = self.ei[e] as usize;
            let j = self.ej[e] as usize;
            let n = self.n_dirs[e].array();
            let ke = self.k_eff[e];
            for aa in 0..DIM {
                for bb in 0..DIM {
                    let nn = ke * n[aa] * n[bb];
                    let ii = i*DIM + aa;
                    let jj = i*DIM + bb;
                    mat[ii * ndof + jj] += nn;
                    let ji = j*DIM + aa;
                    let jjb = j*DIM + bb;
                    mat[ji * ndof + jjb] += nn;
                    mat[(i*DIM + aa) * ndof + (j*DIM + bb)] -= nn;
                    mat[(j*DIM + aa) * ndof + (i*DIM + bb)] -= nn;
                }
            }
        }
        mat
    }
}

impl LinearOp for TrussOp {
    #[inline] fn natoms(&self) -> usize { self.natoms }
    #[inline] fn matvec(&self, x: &[f64]) -> Vec<f64> { TrussOp::matvec(self, x) }
    #[inline] fn diagonal_blocks(&self) -> Vec<[f64; 9]> { TrussOp::diagonal_blocks(self) }
    #[inline] fn assemble_dense(&self) -> Vec<f64> { TrussOp::assemble_dense(self) }
}

// ============================================================================
// UffHessianOp — full UFF Hessian (bonds + angles + dihedrals + inversions)
// ============================================================================

/// Dense linear operator from a finite-difference UFF Hessian + mass regularization.
/// Captures ALL stiffness terms: bonds, angles, dihedrals, inversions.
/// Required for physically meaningful multigrid on aromatic molecules — the bond-only
/// `TrussOp` has zero out-of-plane bending stiffness at a planar equilibrium.
/// Cost: 2·n_dof force evaluations to build (once); then matvec is O(n_dof²).
pub struct UffHessianOp {
    pub natoms: usize,
    /// Dense (n*3)×(n*3) matrix, already includes mass_dt2 on the diagonal.
    pub mat: Vec<f64>,
    pub mass_dt2: Vec<f64>,
}

impl UffHessianOp {
    /// Build the full UFF stiffness Hessian via central differences, then add mass regularization.
    /// K[:,q] = -(F(x+ε·e_q) - F(x-ε·e_q)) / (2ε)  where F = -∇E (force).
    /// A = K + diag(mass_dt2).  `eps` is the finite-difference step (Å).
    pub fn from_uff(uff: &mut Uff, apos: &[Vec3d], neighs: &[Quat4i], neigh_bs: &[Quat4i],
                    mass_dt2: &[f64], eps: f64) -> Self {
        let n = uff.natoms as usize;
        assert_eq!(apos.len(), n, "UffHessianOp::from_uff: apos.len={} != natoms={}", apos.len(), n);
        assert_eq!(mass_dt2.len(), n, "UffHessianOp::from_uff: mass_dt2.len={} != natoms={}", mass_dt2.len(), n);
        assert!(eps.is_finite() && eps > 0.0, "UffHessianOp::from_uff: invalid eps={eps}");
        let ndof = n * DIM;
        let mut k = vec![0.0f64; ndof * ndof];
        let mut fpos = vec![Vec3d::new(0.0,0.0,0.0); n];
        let mut fneg = vec![Vec3d::new(0.0,0.0,0.0); n];
        let mut apos_mut = apos.to_vec();
        for q in 0..ndof {
            let ia = q / DIM;
            let id = q % DIM;
            let orig = apos_mut[ia].array()[id];
            apos_mut[ia].array_mut()[id] = orig + eps;
            uff.eval_forces(&apos_mut, &mut fpos, neighs, neigh_bs);
            apos_mut[ia].array_mut()[id] = orig - eps;
            uff.eval_forces(&apos_mut, &mut fneg, neighs, neigh_bs);
            apos_mut[ia].array_mut()[id] = orig;
            // K[:,q] = -(F+ - F-) / (2ε)  (F = -∇E → K = ∂²E/∂x² = -∂F/∂x)
            let inv_2eps = 1.0 / (2.0 * eps);
            for i in 0..n {
                let fp = fpos[i].array();
                let fn_ = fneg[i].array();
                for d in 0..DIM {
                    let row = i * DIM + d;
                    k[row * ndof + q] = -(fp[d] - fn_[d]) * inv_2eps;
                }
            }
        }
        // Symmetrize (finite-difference noise breaks exact symmetry)
        for i in 0..ndof {
            for j in (i+1)..ndof {
                let avg = 0.5 * (k[i*ndof+j] + k[j*ndof+i]);
                k[i*ndof+j] = avg;
                k[j*ndof+i] = avg;
            }
        }
        // Add mass regularization to diagonal
        for i in 0..n {
            for d in 0..DIM {
                let q = i * DIM + d;
                k[q * ndof + q] += mass_dt2[i];
            }
        }
        assert!(k.iter().all(|v| v.is_finite()), "UffHessianOp::from_uff: non-finite Hessian entry");
        Self { natoms: n, mat: k, mass_dt2: mass_dt2.to_vec() }
    }
}

impl LinearOp for UffHessianOp {
    #[inline] fn natoms(&self) -> usize { self.natoms }
    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let ndof = self.natoms * DIM;
        assert_eq!(x.len(), ndof, "UffHessianOp::matvec: x.len={} != ndof={}", x.len(), ndof);
        let mut ax = vec![0.0f64; ndof];
        for i in 0..ndof {
            let row = i * ndof;
            let mut s = 0.0;
            for j in 0..ndof { s += self.mat[row + j] * x[j]; }
            ax[i] = s;
        }
        ax
    }
    fn diagonal_blocks(&self) -> Vec<[f64; 9]> {
        let ndof = self.natoms * DIM;
        let mut d = vec![[0.0f64; 9]; self.natoms];
        for i in 0..self.natoms {
            for a in 0..DIM {
                for b in 0..DIM {
                    d[i][a*DIM+b] = self.mat[(i*DIM+a)*ndof + (i*DIM+b)];
                }
            }
        }
        d
    }
    #[inline] fn assemble_dense(&self) -> Vec<f64> { self.mat.clone() }
}

// ============================================================================
// Smoother — damped block Jacobi
// ============================================================================

/// Invert natoms×9 batch of 3×3 SPD blocks. Singular → zero block (matches TrussSolver.invert_3x3_blocks).
pub fn invert_3x3_blocks(d: &[[f64; 9]]) -> Vec<[f64; 9]> {
    let n = d.len();
    let mut dinv = vec![[0.0f64; 9]; n];
    for i in 0..n {
        let m = &d[i];
        // 3×3 inverse via cofactors / det
        let det = m[0]*(m[4]*m[8] - m[5]*m[7]) - m[1]*(m[3]*m[8] - m[5]*m[6]) + m[2]*(m[3]*m[7] - m[4]*m[6]);
        if det.abs() < 1e-30 { continue; }  // singular → leave as zeros
        let inv_det = 1.0 / det;
        dinv[i][0] = (m[4]*m[8] - m[5]*m[7]) * inv_det;
        dinv[i][1] = (m[2]*m[7] - m[1]*m[8]) * inv_det;
        dinv[i][2] = (m[1]*m[5] - m[2]*m[4]) * inv_det;
        dinv[i][3] = (m[5]*m[6] - m[3]*m[8]) * inv_det;
        dinv[i][4] = (m[0]*m[8] - m[2]*m[6]) * inv_det;
        dinv[i][5] = (m[2]*m[3] - m[0]*m[5]) * inv_det;
        dinv[i][6] = (m[3]*m[7] - m[4]*m[6]) * inv_det;
        dinv[i][7] = (m[1]*m[6] - m[0]*m[7]) * inv_det;
        dinv[i][8] = (m[0]*m[4] - m[1]*m[3]) * inv_det;
    }
    dinv
}

/// n_steps of damped block Jacobi: x += ω·D⁻¹·(b − A·x), fixed nodes pinned to x0.
/// Ported from MultiGrid._jacobi_smooth. Mutates x in place. beta=0 (no heavy-ball).
pub fn jacobi_smooth(op: &impl LinearOp, dinv: &[[f64; 9]], b: &[f64], x: &mut [f64],
                     free_mask: &[bool], omega: f64, n_steps: usize) {
    jacobi_smooth_momentum(op, dinv, b, x, free_mask, omega, 0.0, &mut vec![0.0f64; op.natoms() * DIM], n_steps);
}

/// Damped block Jacobi with heavy-ball momentum: v = β·v_prev + ω·D⁻¹·r; x += v.
/// Ported from TrussSolver.solve_global_jacobi (beta=0.5 there). The reference's
/// _jacobi_smooth (inside the V-cycle) uses beta=0, but heavy-ball in the smoother
/// accelerates high-frequency error reduction → fewer pre/post-smooth steps needed.
/// `vel` is the persistent velocity buffer (caller allocates, must be len = natoms*DIM).
pub fn jacobi_smooth_momentum(op: &impl LinearOp, dinv: &[[f64; 9]], b: &[f64], x: &mut [f64],
                              free_mask: &[bool], omega: f64, beta: f64,
                              vel: &mut [f64], n_steps: usize) {
    let natoms = op.natoms();
    let n = natoms * DIM;
    assert_eq!(vel.len(), n, "jacobi_smooth_momentum: vel.len={} != natoms*DIM={}", vel.len(), n);
    for _ in 0..n_steps {
        let ax = op.matvec(x);
        // dx_i = Dinv_i · r_i, then v = β·v_prev + ω·dx, x += v
        for i in 0..natoms {
            if !free_mask[i] {
                for d in 0..DIM { vel[i*DIM + d] = 0.0; }
                continue;
            }
            let di = &dinv[i];
            let r = [b[i*DIM] - ax[i*DIM], b[i*DIM+1] - ax[i*DIM+1], b[i*DIM+2] - ax[i*DIM+2]];
            for a in 0..DIM {
                let mut s = 0.0;
                for bb in 0..DIM { s += di[a*DIM + bb] * r[bb]; }
                // heavy-ball: v = β·v_prev + ω·D⁻¹·r
                vel[i*DIM + a] = beta * vel[i*DIM + a] + omega * s;
                x[i*DIM + a] += vel[i*DIM + a];
            }
        }
    }
}

// ============================================================================
// Prolongation — geometric pivots (Strategy B)
// ============================================================================

/// BFS distances from `source` over the bond adjacency graph. Returns -1 for unreachable.
/// Parity: TrussSolver.bfs_distances.
pub fn bfs_distances(adj: &[Vec<i32>], source: usize, n_nodes: usize) -> Vec<i32> {
    let mut dist = vec![-1i32; n_nodes];
    dist[source] = 0;
    let mut queue = vec![source];
    let mut head = 0;
    while head < queue.len() {
        let u = queue[head]; head += 1;
        for &v in &adj[u] {
            let vi = v as usize;
            if dist[vi] == -1 {
                dist[vi] = dist[u] + 1;
                queue.push(vi);
            }
        }
    }
    dist
}

/// Build adjacency list from bond edges.
pub fn build_adjacency(bonds: &[[i32; 2]], n_nodes: usize) -> Vec<Vec<i32>> {
    let mut adj = vec![Vec::new(); n_nodes];
    for b in bonds {
        let i = b[0] as usize;
        let j = b[1] as usize;
        adj[i].push(j as i32);
        adj[j].push(i as i32);
    }
    adj
}

/// Farthest-point (maximin) pivot selection via BFS. Greedy phase only (no swaps).
/// Parity: MultiGrid.select_pivots_maximin (Phase 1: greedy, n_swap_iter=0).
pub fn select_pivots_maximin(bonds: &[[i32; 2]], n_nodes: usize, n_pivots: usize,
                             free_mask: &[bool]) -> Vec<usize> {
    let adj = build_adjacency(bonds, n_nodes);
    let free_nodes: Vec<usize> = (0..n_nodes).filter(|&i| free_mask[i]).collect();
    assert!(!free_nodes.is_empty(), "select_pivots_maximin: no free nodes");
    let first = free_nodes[0];
    let mut pivots = vec![first];
    let mut min_dist = bfs_distances(&adj, first, n_nodes);
    for i in 0..n_nodes { if !free_mask[i] { min_dist[i] = -1; } }
    while pivots.len() < n_pivots {
        // Pick node with max min_dist (farthest from current pivot set)
        let mut best = -1i32;
        let mut best_d = -1i32;
        for &i in &free_nodes {
            if min_dist[i] > best_d { best_d = min_dist[i]; best = i as i32; }
        }
        if best < 0 || best_d <= 0 { break; }
        let nxt = best as usize;
        pivots.push(nxt);
        let d = bfs_distances(&adj, nxt, n_nodes);
        for i in 0..n_nodes { if d[i] >= 0 && (min_dist[i] < 0 || d[i] < min_dist[i]) { min_dist[i] = d[i]; } }
        for i in 0..n_nodes { if !free_mask[i] { min_dist[i] = -1; } }
    }
    pivots
}

/// Inverse-distance interpolation prolongation P: (N*DIM, k*DIM), row-major.
/// P[i*DIM+d, p*DIM+d] = w_{i,p} (same weight per spatial component).
/// Parity: MultiGrid.build_pivot_prolongation.
pub fn build_pivot_prolongation(apos: &[Vec3d], pivots: &[usize], power: f64,
                                free_mask: &[bool]) -> Vec<f64> {
    let n = apos.len();
    let k = pivots.len();
    let n_dof = n * DIM;
    let n_coarse = k * DIM;
    let mut p = vec![0.0f64; n_dof * n_coarse];
    let pivot_pos: Vec<Vec3d> = pivots.iter().map(|&i| apos[i]).collect();
    for i in 0..n {
        if !free_mask[i] { continue; }
        let mut d = vec![0.0f64; k];
        for (pi, &pp) in pivot_pos.iter().enumerate() { d[pi] = (apos[i] - pp).norm(); }
        let mut w = vec![0.0f64; k];
        for pi in 0..k {
            let dd = if d[pi] > 1e-12 { d[pi] } else { 1e-12 };
            w[pi] = 1.0 / dd.powf(power);
        }
        let w_sum: f64 = w.iter().sum();
        if w_sum > 1e-30 {
            for pi in 0..k { w[pi] /= w_sum; }
        } else {
            // Fallback: nearest pivot gets weight 1
            let nn = (0..k).min_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap()).unwrap();
            for pi in 0..k { w[pi] = if pi == nn { 1.0 } else { 0.0 }; }
        }
        for pi in 0..k {
            for dd in 0..DIM {
                p[(i*DIM + dd) * n_coarse + (pi*DIM + dd)] = w[pi];
            }
        }
    }
    p
}

// ============================================================================
// Galerkin coarse operator + coarse solve
// ============================================================================

/// A_c = Pᵀ·A·P via matvec on each column of P. Symmetrized + regularized.
/// Parity: MultiGrid.galerkin_coarse_operator.
/// P: (n_dof, n_coarse) row-major. Returns (n_coarse, n_coarse) row-major.
pub fn galerkin_coarse(op: &impl LinearOp, p: &[f64], n_coarse: usize) -> Vec<f64> {
    let n_dof = op.natoms() * DIM;
    assert_eq!(p.len(), n_dof * n_coarse, "galerkin_coarse: P.len={} != n_dof*n_coarse={}", p.len(), n_dof * n_coarse);
    // AP[:, j] = A · P[:, j]
    let mut ap = vec![0.0f64; n_dof * n_coarse];
    for j in 0..n_coarse {
        let mut col = vec![0.0f64; n_dof];
        for i in 0..n_dof { col[i] = p[i * n_coarse + j]; }
        let acol = op.matvec(&col);
        for i in 0..n_dof { ap[i * n_coarse + j] = acol[i]; }
    }
    // A_c = Pᵀ · AP
    let mut a_c = vec![0.0f64; n_coarse * n_coarse];
    for i in 0..n_coarse {
        for j in 0..n_coarse {
            let mut s = 0.0;
            for r in 0..n_dof { s += p[r * n_coarse + i] * ap[r * n_coarse + j]; }
            a_c[i * n_coarse + j] = s;
        }
    }
    // Symmetrize
    for i in 0..n_coarse {
        for j in (i+1)..n_coarse {
            let avg = 0.5 * (a_c[i*n_coarse + j] + a_c[j*n_coarse + i]);
            a_c[i*n_coarse + j] = avg;
            a_c[j*n_coarse + i] = avg;
        }
    }
    // Regularize: add small diagonal to prevent singularity from null-space modes
    let diag_max = (0..n_coarse).map(|i| a_c[i*n_coarse + i].abs()).fold(0.0f64, f64::max) + 1e-30;
    for i in 0..n_coarse { a_c[i*n_coarse + i] += 1e-6 * diag_max; }
    a_c
}

// ============================================================================
// V-cycle solver
// ============================================================================

/// Cached Galerkin level. Building it costs `n_coarse` fine operator applications;
/// subsequent coarse force steps use only dense Pᵀ/P contractions and a small Cholesky solve.
pub struct GalerkinLevel {
    pub n_dof: usize,
    pub n_coarse: usize,
    pub p: Vec<f64>,
    pub a_c: Vec<f64>,
    pub a_c_chol: Vec<f64>,
}

impl GalerkinLevel {
    pub fn new(op: &impl LinearOp, p: &[f64], n_coarse: usize) -> Self {
        let n_dof = op.natoms() * DIM;
        assert_eq!(p.len(), n_dof*n_coarse, "GalerkinLevel::new: P.len={} != n_dof*n_coarse={}", p.len(), n_dof*n_coarse);
        let a_c = galerkin_coarse(op, p, n_coarse);
        let a_c_chol = cholesky_factor_f64(&a_c, n_coarse);
        Self { n_dof, n_coarse, p: p.to_vec(), a_c, a_c_chol }
    }

    /// Solve Δx=P·A_c⁻¹·PᵀF without applying the fine operator. Returns sqrt((PᵀF)ᵀA_c⁻¹(PᵀF)).
    pub fn solve_force(&self, force: &[f64], free_mask: &[bool], dx: &mut [f64]) -> f64 {
        assert_eq!(force.len(), self.n_dof, "GalerkinLevel::solve_force: force.len={} != n_dof={}", force.len(), self.n_dof);
        assert_eq!(dx.len(), self.n_dof, "GalerkinLevel::solve_force: dx.len={} != n_dof={}", dx.len(), self.n_dof);
        assert_eq!(free_mask.len()*DIM, self.n_dof, "GalerkinLevel::solve_force: free_mask.len*DIM={} != n_dof={}", free_mask.len()*DIM, self.n_dof);
        let mut g = vec![0.0f64; self.n_coarse];
        for j in 0..self.n_coarse {
            let mut s = 0.0;
            for i in 0..free_mask.len() {
                if free_mask[i] { for d in 0..DIM { let q=i*DIM+d; s += self.p[q*self.n_coarse+j]*force[q]; } }
            }
            g[j] = s;
        }
        let dq = cholesky_solve_f64(&self.a_c_chol, &g, self.n_coarse);
        dx.fill(0.0);
        for i in 0..free_mask.len() {
            if free_mask[i] {
                for d in 0..DIM {
                    let q=i*DIM+d;
                    for j in 0..self.n_coarse { dx[q] += self.p[q*self.n_coarse+j]*dq[j]; }
                }
            }
        }
        assert!(dx.iter().all(|v| v.is_finite()), "GalerkinLevel::solve_force: non-finite displacement for n_dof={} n_coarse={}", self.n_dof, self.n_coarse);
        g.iter().zip(dq.iter()).map(|(f, x)| f*x).sum::<f64>().max(0.0).sqrt()
    }

    /// Apply a scaled coarse-preconditioned nonlinear force step to atom positions.
    /// The caller controls `scale` and decides when the frozen hierarchy must be rebuilt.
    pub fn apply_force_step(&self, apos: &mut [Vec3d], force: &[Vec3d], free_mask: &[bool], scale: f64) -> (f64, f64) {
        assert!(scale.is_finite() && scale >= 0.0, "GalerkinLevel::apply_force_step: invalid scale={scale}");
        assert_eq!(apos.len(), free_mask.len(), "GalerkinLevel::apply_force_step: apos.len={} != free_mask.len={}", apos.len(), free_mask.len());
        assert_eq!(force.len(), apos.len(), "GalerkinLevel::apply_force_step: force.len={} != apos.len={}", force.len(), apos.len());
        let mut f = vec![0.0f64; self.n_dof];
        for i in 0..apos.len() { f[i*DIM]=force[i].x; f[i*DIM+1]=force[i].y; f[i*DIM+2]=force[i].z; }
        let mut dx = vec![0.0f64; self.n_dof];
        let coarse_energy = self.solve_force(&f, free_mask, &mut dx);
        let mut max_step = 0.0f64;
        for i in 0..apos.len() {
            if free_mask[i] {
                let dr = Vec3d::new(scale*dx[i*DIM], scale*dx[i*DIM+1], scale*dx[i*DIM+2]);
                max_step = max_step.max(dr.norm());
                apos[i].add(dr);
            }
        }
        (coarse_energy, max_step)
    }
}

/// Build two orthonormal modes for an elongated planar molecule: out-of-plane bend and axial twist.
/// Returns row-major Φ with shape (3*natoms,2). `axis` is the long axis and `normal` the plane normal.
pub fn build_bend_twist_modes(apos: &[Vec3d], axis: Vec3d, normal: Vec3d) -> Vec<f64> {
    assert!(apos.len() >= 2, "build_bend_twist_modes: need at least 2 atoms, got {}", apos.len());
    let u = axis * (1.0/axis.norm());
    let mut n = normal - u*u.dot(normal);
    n = n * (1.0/n.norm());
    assert!(u.array().iter().chain(n.array().iter()).all(|v| v.is_finite()), "build_bend_twist_modes: invalid axis={axis:?} normal={normal:?}");
    let mut center = Vec3d::new(0.0,0.0,0.0);
    for &p in apos { center.add(p); }
    center = center * (1.0/apos.len() as f64);
    let s: Vec<f64> = apos.iter().map(|p| (*p-center).dot(u)).collect();
    let smin=s.iter().copied().fold(f64::INFINITY,f64::min);
    let smax=s.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    let span=smax-smin;
    assert!(span > 1e-12, "build_bend_twist_modes: zero extent along axis={axis:?}: smin={smin} smax={smax}");
    let mut phi = vec![0.0f64; apos.len()*3*2];
    let mut mean_b=Vec3d::new(0.0,0.0,0.0);
    let mut mean_t=Vec3d::new(0.0,0.0,0.0);
    for i in 0..apos.len() {
        let t=(s[i]-smin)/span;
        let bend=n*(std::f64::consts::PI*t).sin();
        let radial=apos[i]-center-u*s[i];
        let twist=u.cross(radial)*(2.0*t-1.0);
        mean_b.add(bend); mean_t.add(twist);
        for d in 0..3 { phi[(i*3+d)*2]=bend.array()[d]; phi[(i*3+d)*2+1]=twist.array()[d]; }
    }
    mean_b = mean_b * (1.0/apos.len() as f64); mean_t = mean_t * (1.0/apos.len() as f64);
    for i in 0..apos.len() { for d in 0..3 { phi[(i*3+d)*2]-=mean_b.array()[d]; phi[(i*3+d)*2+1]-=mean_t.array()[d]; } }
    let nb=(0..apos.len()*3).map(|i| phi[i*2]*phi[i*2]).sum::<f64>().sqrt();
    assert!(nb > 1e-12, "build_bend_twist_modes: degenerate bend mode, norm={nb}");
    for i in 0..apos.len()*3 { phi[i*2]/=nb; }
    let bt=(0..apos.len()*3).map(|i| phi[i*2]*phi[i*2+1]).sum::<f64>();
    for i in 0..apos.len()*3 { phi[i*2+1]-=bt*phi[i*2]; }
    let nt=(0..apos.len()*3).map(|i| phi[i*2+1]*phi[i*2+1]).sum::<f64>().sqrt();
    assert!(nt > 1e-12, "build_bend_twist_modes: degenerate twist mode, norm={nt}; molecule needs finite width around axis");
    for i in 0..apos.len()*3 { phi[i*2+1]/=nt; }
    phi
}

/// Quadratic elastic model in a small orthonormal modal basis Φ.
/// Central-difference fitting costs two full-force evaluations per mode; all later solves are modal-only.
pub struct ModalQuadratic {
    pub n_dof: usize,
    pub n_modes: usize,
    pub phi: Vec<f64>,
    pub k: Vec<f64>,
    pub k_chol: Vec<f64>,
    pub fit_radius: f64,
}

impl ModalQuadratic {
    /// `phi` is row-major (n_dof,n_modes); force samples are sample-major (n_modes,n_dof)
    /// at x_ref ± fit_radius*phi[:,mode]. Force is -∇E, hence K=-∂(ΦᵀF)/∂q.
    pub fn fit_central(phi: &[f64], n_dof: usize, n_modes: usize, fit_radius: f64,
                       force_minus: &[f64], force_plus: &[f64]) -> Self {
        assert!(fit_radius.is_finite() && fit_radius > 0.0, "ModalQuadratic::fit_central: invalid fit_radius={fit_radius}");
        assert_eq!(phi.len(), n_dof*n_modes, "ModalQuadratic::fit_central: phi.len={} != n_dof*n_modes={}", phi.len(), n_dof*n_modes);
        assert_eq!(force_minus.len(), n_dof*n_modes, "ModalQuadratic::fit_central: force_minus.len={} != n_dof*n_modes={}", force_minus.len(), n_dof*n_modes);
        assert_eq!(force_plus.len(), n_dof*n_modes, "ModalQuadratic::fit_central: force_plus.len={} != n_dof*n_modes={}", force_plus.len(), n_dof*n_modes);
        for a in 0..n_modes {
            for b in 0..n_modes {
                let gram = (0..n_dof).map(|i| phi[i*n_modes+a]*phi[i*n_modes+b]).sum::<f64>();
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((gram-expected).abs() < 1e-6, "ModalQuadratic::fit_central: modes not orthonormal at ({a},{b}): gram={gram:.15e} expected={expected}");
            }
        }
        let mut k = vec![0.0f64; n_modes*n_modes];
        for a in 0..n_modes {
            for b in 0..n_modes {
                let dg = (0..n_dof).map(|i| phi[i*n_modes+a]*(force_plus[b*n_dof+i]-force_minus[b*n_dof+i])).sum::<f64>()/(2.0*fit_radius);
                k[a*n_modes+b] = -dg;
            }
        }
        for a in 0..n_modes { for b in (a+1)..n_modes { let s=0.5*(k[a*n_modes+b]+k[b*n_modes+a]); k[a*n_modes+b]=s; k[b*n_modes+a]=s; } }
        assert!(k.iter().all(|v| v.is_finite()), "ModalQuadratic::fit_central: non-finite fitted stiffness K={k:?}");
        let k_chol = cholesky_factor_f64(&k, n_modes);
        Self { n_dof, n_modes, phi: phi.to_vec(), k, k_chol, fit_radius }
    }

    pub fn project_force(&self, force: &[f64]) -> Vec<f64> {
        assert_eq!(force.len(), self.n_dof, "ModalQuadratic::project_force: force.len={} != n_dof={}", force.len(), self.n_dof);
        (0..self.n_modes).map(|a| (0..self.n_dof).map(|i| self.phi[i*self.n_modes+a]*force[i]).sum()).collect()
    }

    /// Modal equilibrium/response K·dq=g, followed by reconstruction dx=Φ·dq.
    pub fn solve_force(&self, force: &[f64], dx: &mut [f64]) -> Vec<f64> {
        assert_eq!(dx.len(), self.n_dof, "ModalQuadratic::solve_force: dx.len={} != n_dof={}", dx.len(), self.n_dof);
        let g = self.project_force(force);
        let dq = cholesky_solve_f64(&self.k_chol, &g, self.n_modes);
        for i in 0..self.n_dof { dx[i]=(0..self.n_modes).map(|a| self.phi[i*self.n_modes+a]*dq[a]).sum(); }
        dq
    }
}

/// Relative Euclidean residual over free atom DOFs.
pub fn relative_residual(op: &impl LinearOp, b: &[f64], x: &[f64], free_mask: &[bool]) -> f64 {
    let ax = op.matvec(x);
    let mut r2 = 0.0;
    let mut b2 = 0.0;
    for i in 0..op.natoms() {
        if free_mask[i] {
            for d in 0..DIM { let q = i*DIM+d; let r = b[q]-ax[q]; r2 += r*r; b2 += b[q]*b[q]; }
        }
    }
    r2.sqrt() / b2.sqrt().max(1e-30)
}

/// Apply one exact Galerkin coarse correction to `x` and return sqrt(r_c^T A_c^-1 r_c),
/// the energy-scaled coarse residual before correction. Costs one fine operator application;
/// no fine smoothing is performed. This is the primitive for coarse-first relaxation.
pub fn coarse_correct(op: &impl LinearOp, p: &[f64], a_c_chol: &[f64], n_coarse: usize,
                      b: &[f64], x: &mut [f64], free_mask: &[bool]) -> f64 {
    let natoms = op.natoms();
    let n_dof = natoms * DIM;
    assert_eq!(p.len(), n_dof * n_coarse, "coarse_correct: P.len={} != n_dof*n_coarse={}", p.len(), n_dof * n_coarse);
    assert_eq!(b.len(), n_dof, "coarse_correct: b.len={} != n_dof={}", b.len(), n_dof);
    assert_eq!(x.len(), n_dof, "coarse_correct: x.len={} != n_dof={}", x.len(), n_dof);
    assert_eq!(free_mask.len(), natoms, "coarse_correct: free_mask.len={} != natoms={}", free_mask.len(), natoms);
    let ax = op.matvec(x);
    let mut r_c = vec![0.0f64; n_coarse];
    for j in 0..n_coarse {
        let mut s = 0.0;
        for i in 0..natoms {
            if free_mask[i] { for d in 0..DIM { let q = i*DIM+d; s += p[q*n_coarse+j] * (b[q]-ax[q]); } }
        }
        r_c[j] = s;
    }
    let e_c = cholesky_solve_f64(a_c_chol, &r_c, n_coarse);
    let coarse_energy = r_c.iter().zip(e_c.iter()).map(|(r, e)| r*e).sum::<f64>().max(0.0).sqrt();
    for i in 0..natoms {
        if free_mask[i] {
            for d in 0..DIM {
                let q = i*DIM+d;
                let mut s = 0.0;
                for j in 0..n_coarse { s += p[q*n_coarse+j] * e_c[j]; }
                x[q] += s;
            }
        }
    }
    coarse_energy
}

/// One exact coarse correction followed by only enough fine smoothing to meet `tol`.
/// Returns (x, residual after coarse and after each fine step, coarse energy, A_c).
pub fn solve_coarse_first(op: &impl LinearOp, p: &[f64], n_coarse: usize, b: &[f64], x0: &[f64],
                          free_mask: &[bool], omega: f64, beta: f64,
                          n_fine_max: usize, tol: f64) -> (Vec<f64>, Vec<f64>, f64, Vec<f64>) {
    let a_c = galerkin_coarse(op, p, n_coarse);
    let a_c_chol = cholesky_factor_f64(&a_c, n_coarse);
    let d = op.diagonal_blocks();
    let dinv = invert_3x3_blocks(&d);
    let mut x = x0.to_vec();
    let coarse_energy = coarse_correct(op, p, &a_c_chol, n_coarse, b, &mut x, free_mask);
    let mut residuals = vec![relative_residual(op, b, &x, free_mask)];
    let mut vel = vec![0.0f64; op.natoms() * DIM];
    for _ in 0..n_fine_max {
        if residuals[residuals.len()-1] < tol { break; }
        jacobi_smooth_momentum(op, &dinv, b, &mut x, free_mask, omega, beta, &mut vel, 1);
        residuals.push(relative_residual(op, b, &x, free_mask));
    }
    (x, residuals, coarse_energy, a_c)
}

/// Two-grid V-cycle with heavy-ball smoother momentum.
/// Ported from MultiGrid.solve_two_grid + TrussSolver.solve_global_jacobi (heavy-ball).
/// Returns (x, total_residuals, coarse_residuals) where:
///   total_residuals[k]   = |r|/|b| over all free DOFs after k-th V-cycle
///   coarse_residuals[k]  = |Pᵀ·r|/|Pᵀ·b| — the low-frequency residual the coarse solver handles
/// The coarse residual isolates the coarse solver's performance from the smoother's.
pub fn solve_two_grid(op: &impl LinearOp, p: &[f64], a_c_chol: &[f64], n_coarse: usize,
                      b: &[f64], x0: &[f64], free_mask: &[bool],
                      omega: f64, beta: f64, n_pre: usize, n_post: usize,
                      n_outer: usize, tol: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let natoms = op.natoms();
    let n_dof = natoms * DIM;
    let d = op.diagonal_blocks();
    let dinv = invert_3x3_blocks(&d);
    let mut x = x0.to_vec();
    let mut vel = vec![0.0f64; n_dof];  // persistent velocity for heavy-ball
    let mut residuals = Vec::with_capacity(n_outer + 1);
    let mut coarse_res = Vec::with_capacity(n_outer + 1);
    let b_norm = (0..natoms).filter(|&i| free_mask[i]).flat_map(|i| (0..DIM).map(move |d| b[i*DIM+d]*b[i*DIM+d])).sum::<f64>().sqrt().max(1e-30);
    // Coarse RHS norm: |Pᵀ·b|
    let mut b_c = vec![0.0f64; n_coarse];
    for j in 0..n_coarse {
        let mut s = 0.0;
        for i in 0..n_dof { s += p[i * n_coarse + j] * b[i]; }
        b_c[j] = s;
    }
    let b_c_norm: f64 = b_c.iter().map(|x| x*x).sum::<f64>().sqrt().max(1e-30);

    let compute_residuals = |x: &[f64]| -> (f64, f64) {
        let ax = op.matvec(x);
        let mut r = vec![0.0f64; n_dof];
        let mut rn = 0.0;
        for i in 0..natoms {
            if free_mask[i] {
                for d in 0..DIM { let dd = b[i*DIM+d] - ax[i*DIM+d]; r[i*DIM+d] = dd; rn += dd*dd; }
            }
        }
        let total = rn.sqrt() / b_norm;
        // Coarse-projected residual: |Pᵀ·r|
        let mut rcn = 0.0;
        for j in 0..n_coarse {
            let mut s = 0.0;
            for i in 0..n_dof { s += p[i * n_coarse + j] * r[i]; }
            rcn += s*s;
        }
        (total, rcn.sqrt() / b_c_norm)
    };

    let (r0t, r0c) = compute_residuals(&x);
    residuals.push(r0t);
    coarse_res.push(r0c);

    for _ in 0..n_outer {
        // 1. Pre-smooth (with heavy-ball momentum)
        jacobi_smooth_momentum(op, &dinv, b, &mut x, free_mask, omega, beta, &mut vel, n_pre);
        // 2–4. Restrict, solve coarse correction, prolongate.
        coarse_correct(op, p, a_c_chol, n_coarse, b, &mut x, free_mask);
        // The coarse correction is a discontinuous update; old fine-level momentum is stale.
        vel.fill(0.0);
        // 5. Post-smooth (with heavy-ball momentum)
        jacobi_smooth_momentum(op, &dinv, b, &mut x, free_mask, omega, beta, &mut vel, n_post);
        // 6. Residual (total + coarse-projected)
        let (rt, rc) = compute_residuals(&x);
        residuals.push(rt);
        coarse_res.push(rc);
        if rt < tol { break; }
    }
    (x, residuals, coarse_res)
}

/// Convenience: build Galerkin + Cholesky factorize + V-cycle.
/// Parity: MultiGrid.solve_multigrid. Returns (x, total_residuals, coarse_residuals, A_c).
pub fn solve_multigrid(op: &impl LinearOp, p: &[f64], n_coarse: usize, b: &[f64], x0: &[f64],
                       free_mask: &[bool], omega: f64, beta: f64, n_pre: usize, n_post: usize,
                       n_outer: usize, tol: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let a_c = galerkin_coarse(op, p, n_coarse);
    let a_c_chol = cholesky_factor_f64(&a_c, n_coarse);
    let (x, res, cres) = solve_two_grid(op, p, &a_c_chol, n_coarse, b, x0, free_mask,
                                        omega, beta, n_pre, n_post, n_outer, tol);
    (x, res, cres, a_c)
}

// ============================================================================
// Dense direct solve (for parity tests) — Gaussian elimination with partial pivoting
// ============================================================================

/// Solve A·x = b for dense n×n A via Gaussian elimination with partial pivoting.
/// Used only by tests as a reference (parity check vs multigrid).
pub fn dense_solve(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n*n);
    assert_eq!(b.len(), n);
    let mut lu = a.to_vec();
    let mut bb = b.to_vec();
    let mut piv = (0..n).collect::<Vec<usize>>();
    for k in 0..n {
        // Partial pivot
        let mut pmax = lu[k*n + k].abs();
        let mut pmax_i = k;
        for i in (k+1)..n {
            let v = lu[i*n + k].abs();
            if v > pmax { pmax = v; pmax_i = i; }
        }
        assert!(pmax > 1e-30, "dense_solve: singular matrix at pivot {k}, max|A[.,{k}]|={pmax:.3e}");
        if pmax_i != k {
            for j in 0..n { lu.swap(k*n + j, pmax_i*n + j); }
            bb.swap(k, pmax_i);
            piv.swap(k, pmax_i);
        }
        let inv_pk = 1.0 / lu[k*n + k];
        for i in (k+1)..n {
            let f = lu[i*n + k] * inv_pk;
            lu[i*n + k] = f;
            for j in (k+1)..n { lu[i*n + j] -= f * lu[k*n + j]; }
            bb[i] -= f * bb[k];
        }
    }
    // Back-substitution
    let mut x = vec![0.0f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut s = bb[i];
        for j in (i+1)..n { s -= lu[i*n + j] * x[j]; }
        x[i] = s / lu[i*n + i];
    }
    x
}
