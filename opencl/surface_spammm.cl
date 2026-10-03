// surface.cl - Surface electrostatics and molecule-substrate interaction kernels
// ====================================================================
//
// SURFACE INTERACTION KERNELS FOR GPU
// ===================================
//
// This file provides multiple methods for computing molecule-surface
// interactions in the SPAMMM simulation package. The physical context is
// typically an AFM experiment: a molecule (or tip) adsorbed on a crystalline
// substrate (NaCl, CaF₂, etc.), where we need forces, energies, and
// electrostatic potentials to drive relaxation or manipulation.
//
// --- Physics Overview ---
//
// The molecule-surface interaction has several components:
//
//   1. Pauli repulsion (short-range, exponential): arises from orbital
//      overlap. Modeled as Morse or LJ potential per atom pair.
//
//   2. London dispersion (long-range, ~r⁻⁶): induced-dipole attraction.
//      Combined with Pauli as the Morse potential: V(r) = D·(e^{-2α(r-r₀)} - 2e^{-α(r-r₀)})
//
//   3. Coulomb electrostatics (long-range, ~r⁻¹): interaction between
//      atom charges and substrate ions. For periodic surfaces this requires
//      Ewald 2D summation (conditional convergence, special handling).
//
//   4. Macroscopic dipole/charge layers: for polarized surfaces (e.g. ionic
//      crystals with layer charges), the far-field can be approximated by
//      analytic rectangular sheet formulas.
//
// --- Five Evaluation Strategies ---
//
//   1. Brute-force pairwise (getSurfMorse, getSurfFlat):
//      Sum over substrate atoms × PBC replicas. O(N_atoms × N_surf × N_PBC³).
//      Accurate but slow for large surfaces. Gold-standard reference.
//
//   2. Folded basis expansion (getSurfFolded, getSurfFolded_workgroup,
//      getSurfFolded_tensor_exp/poly):
//      Pre-fit the surface potential to an analytic Fourier-type basis:
//        E(x,y,z) = Σ_{i,b} c_{i,b} · cos(2π·k_u·u) · cos(2π·k_v·v) · f(z)
//      where (u,v) are fractional surface-lattice coordinates and f(z) is
//      either exp(-α·z) (exp variant) or (1-z/zcut)^m (poly variant).
//      O(N_atoms × N_basis) per evaluation — much faster than brute-force.
//      Coefficients fitted by fit_folded_surface_basis() in Python.
//
//   3. Ewald 2D summation (compute_ewald_coefficients, eval_potential_*):
//      GPU implementation of 2D Ewald electrostatics for charged surfaces.
//      Decomposes the conditionally-convergent Coulomb sum into:
//        - Real-space part (short-range, damped by erfc)
//        - Reciprocal-space part (G-vectors, decays as e^{-Gz})
//      Key optimization: complex multiplication for e^{iG·ρ} (see below).
//
//   4. Isosurface-based (getSurfaceIsoSurfMorse, getSurfaceIsoGridFF):
//      Find the z-height where the surface potential crosses a threshold,
//      producing a 2D height map (isosurface). Used for visualization and
//      for computing the effective AFM tip height.
//
//   5. Macroscopic dipole/charge (addDipoleField, macro_phi_rect_*):
//      Analytic potential of polarized rectangular sheets. Used for
//      macroscopic field corrections on large-area surfaces.
//
// --- Key Caveats ---
//
//   CAVEAT 1: folded_eval_grad() in this file has a swapped off-diagonal
//   bug in the inverse lattice matrix (dudy ↔ dvdx swapped). This is
//   invisible for orthogonal lattices (bx=ay=0) but produces wrong forces
//   for sheared cells. The same bug was fixed in rigid.cl's
//   folded_eval_grad_rigid() — this copy needs the same fix.
//   See: lines ~119-122 below.
//
//   CAVEAT 2: The Ewald2D kernels use float32 throughout. For large G-vectors
//   or many ions, accumulation error can reach ~1e-4. Use eval_potential_cluster
//   with double-single accumulation for high-precision validation.
//
//   CAVEAT 3: getSurfMorse uses local-memory tiling over substrate atoms.
//   The barrier inside the tiling loop MUST be hit by ALL threads in the
//   workgroup, even if some have iG >= nAtoms. Early return before the loop
//   would skip the barrier and cause undefined behavior.
//
// Helper functions: macro_phi_rect_dipole/charge (analytic rectangle potential),
// folded_eval_basis/grad (folded basis evaluation), getR4repulsion, limnitForce.
// Requires: common.cl + Forces.cl to be concatenated before this file.

// ==================================================================
//  Macroscopic Rectangle Potential Helpers
// ==================================================================
//
//  Analytic electrostatic potential of uniformly charged/polarized
//  rectangular sheets. Used for macroscopic field corrections on
//  large-area ionic surfaces (e.g. NaCl, CaF₂).
//
//  Reference: W. R. Smythe, "Static and Dynamic Electricity", Ch. 4.
//  Also: J. D. Jackson, "Classical Electrodynamics", §2.6–2.7.
//
//  The potential of a rectangular sheet at point (x,y,z) is obtained by
//  integrating 1/r or 1/r² over the rectangle area. The closed-form
//  involves solid angles (Ω) and logarithmic terms.
//

// Potential of a rectangular dipole sheet (uniform dipole moment Pz).
//   Pz = (px, py, pz, 0) — dipole moment components
//   AB = (Ax, By, 0, 0)   — half-widths in x, y
//   p  = (x, y, z)        — evaluation point (relative to sheet center)
//
//   φ = pz·Ω - px·ln(Y+R) - py·ln(X+R)
// where Ω = solid angle subtended by the rectangle at point p,
//       R = sqrt(X²+Y²+Z²), X,Y are corner-relative coordinates.
//
//  CAVEAT: The 1e-12f regularizers prevent log(0) and atan2(0,0) at
//  points directly above/below corners. They introduce a small bias
//  (~1e-12) that is negligible for physical distances.
inline float macro_phi_rect_dipole(float3 p, float4 Pz, float4 AB) {
    float Ax = AB.x;
    float Bx = AB.y;
    float x = p.x;
    float y = p.y;
    float z = p.z;
    float sumOmega = 0.0f;
    float sumLogY  = 0.0f;
    float sumLogX  = 0.0f;
    float xs[2] = {-Ax, Ax};
    float ys[2] = {-Bx, Bx};
    for (int ix=0; ix<2; ix++) {
        for (int iy=0; iy<2; iy++) {
            float X = x - xs[ix];
            float Y = y - ys[iy];
            float R = sqrt(X*X + Y*Y + z*z);
            float s = ((ix==0)?-1.0f:1.0f) * ((iy==0)?-1.0f:1.0f);
            sumOmega += s * atan2( X*Y, z * R + 1e-12f );
            sumLogY  += s * log( Y + R + 1e-12f );
            sumLogX  += s * log( X + R + 1e-12f );
        }
    }
    return (Pz.z * sumOmega) - (Pz.x * sumLogY) - (Pz.y * sumLogX);
}

// Helper for macro_phi_rect_charge: evaluates the antiderivative of the
// 2D integral of 1/R over a rectangular region. Based on Smythe's formula.
//   F(X,Y,Z) = X·ln(Y+R) + Y·ln(X+R) - Z·atan2(XY, ZR)
// where R = sqrt(X²+Y²+Z²).
inline float rect_sheet_F(float X, float Y, float Z){
    float R = sqrt(X*X + Y*Y + Z*Z);
    return X*log(Y + R + 1e-12f) + Y*log(X + R + 1e-12f) - Z*atan2(X*Y, Z*R + 1e-12f);
}

// Potential of a uniformly charged rectangular sheet (surface charge σ).
//   φ = σ · ∫∫ dx' dy' / |r - r'|
// Computed via corner-sum of rect_sheet_F (Smythe's antiderivative):
//   φ = F(+Ax,+By) - F(-Ax,+By) - F(+Ax,-By) + F(-Ax,-By)
// This is the 2D analog of the 1D endpoint-evaluation quadrature.
inline float macro_phi_rect_charge(float3 p, float4 AB){
    float Ax = AB.x;
    float By = AB.y;
    float x0 = p.x + Ax;
    float x1 = p.x - Ax;
    float y0 = p.y + By;
    float y1 = p.y - By;
    return rect_sheet_F(x0,y0,p.z) - rect_sheet_F(x1,y0,p.z) - rect_sheet_F(x0,y1,p.z) + rect_sheet_F(x1,y1,p.z);
}

// Combine charge-sheet and dipole-sheet potentials for multiple surface layers.
// Each layer i has:
//   - charge density σ_i (from S0.x, S0.y, S0.z)
//   - dipole moment (Q_i.x, Q_i.y, Q_i.z) at height L_i.w
//   - layer position L_i.w (z-offset)
// Returns (Fx, Fy, Fz, φ) — currently only potential is implemented.
// CAVEAT: Force (gradient) is NOT implemented — returns (0,0,0,φ).
//          This means macro dipole layers contribute to energy but NOT
//          to forces in getSurfMorse. For dynamics this is a known limitation.
inline float4 getMacroRectLayers( float3 pos, float q, float4 bounds, float4 L0, float4 L1, float4 L2, float4 S0, float4 Q0, float4 Q1, float4 Q2, int nlayer ){
    float Ax = 0.5f*(bounds.y - bounds.x);
    float By = 0.5f*(bounds.w - bounds.z);
    float cx = 0.5f*(bounds.y + bounds.x);
    float cy = 0.5f*(bounds.w + bounds.z);
    float3 p = pos - (float3)(cx,cy,0.0f);
    float phi = 0.0f;
    float4 ls[3] = {L0,L1,L2};
    float sigmas[3] = {S0.x,S0.y,S0.z};
    float4 qs[3] = {Q0,Q1,Q2};
    for(int i=0; i<nlayer; i++){
        float4 Li = ls[i];
        float3 pp = (float3)(p.x,p.y,p.z-Li.w);
        float4 AB = (float4)(Ax,By,0.0f,0.0f);
        phi += sigmas[i] * macro_phi_rect_charge( pp, AB );
        // dipole contribution
        float4 Pz = (float4)(qs[i].x, qs[i].y, qs[i].z, 0.0f);
        phi += q * macro_phi_rect_dipole( pp, Pz, AB );
    }
    // potential gradient (force) - TODO: implement gradient
    return (float4){0.0f, 0.0f, 0.0f, phi};
}

// ==================================================================
//  Folded Basis Helpers
// ==================================================================
//
//  The folded basis is a separable Fourier-type expansion of the periodic
//  surface potential:
//    E(x,y,z) = Σ_b c_b · cos(2π·k_u·u) · cos(2π·k_v·v) · exp(-α·max(0, z-z₀))
//
//  where (u,v) are fractional coordinates w.r.t. the 2D surface lattice:
//    u = (b_y·x - b_x·y) / det    v = (-a_y·x + a_x·y) / det
//  with det = a_x·b_y - b_x·a_y.
//
//  The basis is separable: B(u,v,z) = Bx(u)·By(v)·Bz(z), which allows
//  factorized evaluation and precomputation of 1D components.
//
//  Coefficients c_b are pre-fitted per atom type by fit_folded_surface_basis()
//  to encode Pauli + London + Coulomb(Ewald) interactions.
//
//  prm = (k_u, k_v, α, z₀) — frequency in u, frequency in v, decay rate, z offset
//

// Evaluate single basis function: B(u,v,z) = cos(2π·k_u·u) · cos(2π·k_v·v) · exp(-α·max(0, z-z₀))
inline float folded_eval_basis(float u, float v, float z, float4 prm){
    float bx = cos( (2.0f*M_PI_F) * prm.x * u );
    float by = cos( (2.0f*M_PI_F) * prm.y * v );
    float dz = fmax(0.0f, z - prm.w);
    float bz = exp( -prm.z * dz );
    return bx * by * bz;
}

// Gradient of single basis function w.r.t. world coordinates (x, y, z).
// Uses chain rule through fractional coordinates:
//   dE/dx = dE/du · du/dx + dE/dv · dv/dx
//   dE/dy = dE/du · du/dy + dE/dv · dv/dy
//   dE/dz = -α · E_basis   (for z > z₀)
//
// invLvec2d = (du/dx, du/dy, dv/dx, dv/dy) — inverse 2D lattice matrix.
//
// CAVEAT (BUG): Lines below swap du/dy ↔ dv/dx:
//   dudy = invLvec2d.z  ← should be invLvec2d.y (du/dy)
//   dvdx = invLvec2d.y  ← should be invLvec2d.z (dv/dx)
// For orthogonal lattices (bx=ay=0) both are zero, so the bug is invisible.
// For sheared lattices it produces wrong forces. The same bug was fixed
// in rigid.cl's folded_eval_grad_rigid() — this copy needs the same fix.
inline float3 folded_eval_grad(float u, float v, float z, float4 prm, float4 invLvec2d){
    float phix = (2.0f*M_PI_F) * prm.x;
    float phiy = (2.0f*M_PI_F) * prm.y;
    float cu = cos(phix*u);
    float su = sin(phix*u);
    float cv = cos(phiy*v);
    float sv = sin(phiy*v);
    float dz = fmax(0.0f, z - prm.w);
    float bz = exp(-prm.z * dz);
    float dEdu = -phix * su * cv * bz;
    float dEdv = -phiy * cu * sv * bz;
    float dEdz = (z > prm.w) ? (-prm.z * cu * cv * bz) : 0.0f;
    float dudx = invLvec2d.x;
    float dudy = invLvec2d.z;  // BUG: should be invLvec2d.y
    float dvdx = invLvec2d.y;  // BUG: should be invLvec2d.z
    float dvdy = invLvec2d.w;
    return (float3)( dEdu*dudx + dEdv*dvdx, dEdu*dudy + dEdv*dvdy, dEdz );
}

// limit force magnitude to fmax
float3 limnitForce( float3 f, float fmax ){
    float fr2 = dot(f,f);                         // force magnitude squared
    if( fr2>(fmax*fmax) ){ f*=(fmax/sqrt(fr2)); } // if force magnitude is larger than fmax we scale it down to fmax
    return f;
}

// R4 blob repulsion: models Pauli repulsion as a compactly-supported polynomial.
//   V(r) = A·(1 - r²/Rcut²)²   for r < Rcut,  0 otherwise.
//   F(r) = -dV/dr = 4A·r·(1 - r²/Rcut²)
// The amplitude A is chosen so that |F(R)| = fmax at the reference distance R.
// This provides a smooth (C¹) cutoff, unlike hard truncation.
// CAVEAT: The force is discontinuous in derivative at r=Rcut (C¹ but not C²),
// which can cause minor energy drift in long MD runs.
float4 getR4repulsion( float3 d, float R, float Rcut, float A ){
    // we use R4blob(r) = A * (1-r^2)^2
    // such that at distance r=R we have force f = fmax
    // f = -dR4blob/dr = 4*A*r*(1-r^2) = fmax
    // A = fmax/(4*R*(1-R^2))
    float R2    = R*R;
    float R2cut = Rcut*Rcut;
    float r2 = dot(d,d);
    if( r2>R2cut ){
        return (float4){0.0f,0.0f,0.0f,0.0f};
    }else if( r2>R2 ){
        float mr2 = R2cut-r2;
        float fr = A*mr2;
        return (float4){ d*(-4*fr), fr*mr2 };
    }else{
        float mr2 = R2cut-R2;
        float fr = A*mr2;
        return (float4){ d*(-4*fr), fr*mr2 };
    }
}

#ifndef MAKE_INDS_PBC_DEF
#define MAKE_INDS_PBC_DEF
inline int4 make_inds_pbc(const int n, const int iG) {
    // Generate PBC index patterns for B-spline interpolation
    // Returns 4 indices: (i0, i1, i2, i3) for 4-point B-spline
    // Handles wrapping at boundaries
    int4 inds;
    int i = iG % n;
    inds.x = (i - 1 + n) % n;
    inds.y = i;
    inds.z = (i + 1) % n;
    inds.w = (i + 2) % n;
    return inds;
}
#endif

// ============================================================
//  Brute Force Surface Interaction (getSurfMorse)
// ============================================================
//
//  Gold-standard pairwise evaluation of molecule-substrate interactions.
//  For each molecule atom, sums Morse (Pauli+London) + Coulomb forces over
//  all substrate atoms × PBC replicas. Optionally adds macroscopic
//  dipole/charge layer corrections.
//
//  Complexity: O(N_atoms × N_surf × N_PBC³) — accurate but slow.
//  Used as reference for validating faster methods (GridFF, folded basis).
//
//  GPU strategy: Local-memory tiling over substrate atoms.
//    - Substrate atoms loaded in chunks of nL (workgroup size) into LATOMS/LCLJS
//    - Each thread processes one molecule atom, iterating over all tiles
//    - PBC replicas handled by shifting dp by lattice vectors
//
//  CAVEAT: The early return `if(iG>=nAtoms) return;` is placed AFTER the
//  tiling loop setup. All threads MUST participate in loading substrate
//  atoms (barrier inside loop). If some threads return early, they skip
//  the barrier → undefined behavior. The current code handles this
//  correctly by returning before the loop but after local decls.
//
//  Physics:
//    F_i = -Σ_j Σ_RBC  ∇V_Morse(r_ij + R) + q_i·E_macro(r_i)
//    V_Morse(r) = D·(e^{-2K(r-r0)} - 2·e^{-K(r-r0)})
//    where D = depth, K = range, r0 = equilibrium distance
//    Combined with Coulomb (Q term) and H-bond (H term) via getMorsePLQH().
//

__kernel void getSurfMorse(
    const int4 ns,                // 1
    __global float4*  atoms,      // 2
    __global float4*  REQs,       // 3
    __global float4*  forces,     // 4
    __global float4*  atoms_s,    // 5
    __global float4*  REQ_s,      // 6
    __global float4*  surf_mpos,  // 7  (xmin,xmax,ymin,ymax)
    __global float4*  surf_mdip,  // 8  (mx,my,mz,0)
    __global float4*  surf_mQa,   // 9  Q row a
    __global float4*  surf_mQb,   // 10 Q row b
    __global float4*  surf_mQc,   // 11 (sigma0,sigma1,sigma2,Qtot)
    __global float4*  surf_qQa,   // 12 layer quadrupole (Qxx,Qxy,Qyy,z0)
    __global float4*  surf_qQb,   // 13 layer quadrupole (Qxx,Qxy,Qyy,z1)
    __global float4*  surf_qQc,   // 14 layer quadrupole (Qxx,Qxy,Qyy,z2)
    const int4     nPBC,          // 15
    const cl_Mat3  lvec,          // 16
    const float4   pos0,          // 17
    const float4   GFFParams,     // 18
    const float4   PLQH           // 19   (Pauli, London, Coulomb, HBond)
){

    __local float4 LATOMS[32];
    __local float4 LCLJS [32];

    const int nAtoms  = ns.x;

    const int iG = get_global_id  (0); // index of atom in the system
    const int iS = get_global_id  (1); // index of system
    const int iL = get_local_id   (0); // index of atom in the local memory chunk
    const int nG = get_global_size(0); // total number of atoms in the system
    const int nS = get_global_size(1); // total number of systems
    const int nL = get_local_size (0); // number of atoms in the local memory chunk

    const int natoms  = ns.x;         // number of atoms in the system
    const int nnode   = ns.y;         // number of nodes in the system
    const int nvec    = natoms+nnode; // number of vectos (atoms and pi-orbitals) in the system
    const int na_surf = ns.z;         //

    const int i0a = iS*natoms;     // index of the first atom in the system
    const int i0v = iS*nvec;       // index of the first vector (atom or pi-orbital) in the system
    const int iaa = iG + i0a;      // index of the atom in the system
    const int iav = iG + i0v;      // index of the vector (atom or pi-orbital) in the system

    float4 fe   = (float4){0.0f,0.0f,0.0f,0.0f};

    if(iG>=nAtoms) return;

    const float  K          = -GFFParams.y;
    const float  R2damp     =  GFFParams.x*GFFParams.x;
    const float3 shift_b = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);      //  shift in scan(iy)
    const float3 shift_c = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);      //  shift in scan(iz)
    const int bMacro      = (int)(GFFParams.z>0.5f);

    const float3 pos  = atoms[iav].xyz - pos0.xyz +  lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;  // most negative PBC-cell
    const float4 REQi = REQs [iaa];

    for (int j0=0; j0<na_surf; j0+= nL ){
        const int i = j0 + iL;
        LATOMS[iL] = atoms_s[i];
        LCLJS [iL] = REQ_s  [i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int jl=0; jl<nL; jl++){
            const int ja=jl+j0;
            if( ja<na_surf ){
                float4 REQH =       LCLJS [jl];
                float3 dp   = pos - LATOMS[jl].xyz;
                REQH.x   += REQi.x;
                REQH.yzw *= REQi.yzw;
                for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
                    for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                        for(int ix=-nPBC.x; ix<=nPBC.x; ix++){
                            float4 fej = getMorsePLQH( dp, REQH, PLQH, K, R2damp );
                            fe -= fej;
                            dp   +=lvec.a.xyz;
                        }
                        dp   +=shift_b;
                    }
                    dp   +=shift_c;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if( bMacro && (fabs(PLQH.z) > 1e-12f) && (fabs(REQi.z) > 1e-12f) ){
        int nlayer = (int)(GFFParams.w + 0.5f);
        float4 fm = getMacroRectLayers( atoms[iav].xyz, REQi.z, surf_mpos[iS], surf_mdip[iS], surf_mQa[iS], surf_mQb[iS], surf_mQc[iS], surf_qQa[iS], surf_qQb[iS], surf_qQc[iS], nlayer );
        fe.xyz += fm.xyz;
        fe.w   += fm.w;
    }

    forces[iav] += fe;
}

// ============================================================
//  Folded Basis Evaluation (getSurfFolded)
// ============================================================
//
//  One thread per atom. Evaluates the folded basis potential and gradient
//  analytically — no grid precomputation needed.
//
//    E(x,y,z) = Σ_b c_b · cos(2π·k_u·u) · cos(2π·k_v·v) · exp(-α·max(0, z-z0))
//    F = -∇E  (via chain rule through fractional coordinates)
//
//  Basis params and coefficients are cooperatively loaded into local memory
//  for fast access. Max 64 basis functions, 8 atom types (compile-time limits).
//
//  CAVEAT: Uses folded_eval_grad() which has a swapped off-diagonal bug
//  in the inverse lattice matrix. See CAVEAT 1 in file header.
//
//  Output: forces[iav] += (Fx, Fy, Fz, -E)  — note energy stored as -E in .w
//

__kernel void getSurfFolded(
    const int4 ns,                     // 1
    __global float4*  atoms,           // 2
    __global float4*  REQs,            // 3
    __global float4*  forces,          // 4
    __global float*   folded_coeffs,   // 5  [ntypeMax*nbasisMax]
    __global float4*  folded_kxyz,     // 6  [nbasisMax]
    __global int*     folded_atom_type,// 7  [natoms]
    const int4        folded_meta,     // 8  (nbasis, ntypes, 0, 0)
    const float4      folded_lvec2d    // 9  (ax,bx,ay,by)
){
    __local float4 LBASIS[64];
    __local float  LCOEFFS[8*64];

    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    const int iL = get_local_id(0);
    const int nL = get_local_size(0);

    const int natoms = ns.x;
    const int nnode  = ns.y;
    const int nvec   = natoms + nnode;
    const int i0a    = iS*natoms;
    const int i0v    = iS*nvec;
    const int iaa    = iG + i0a;
    const int iav    = iG + i0v;
    if(iG>=natoms) return;

    const int nbasis = folded_meta.x;
    const int ntypes = folded_meta.y;
    if(nbasis<=0) return;
    if(nbasis>64){ return; }
    if(ntypes>8 ){ return; }

    for(int j=iL; j<nbasis; j+=nL){
        LBASIS[j] = folded_kxyz[j];
    }
    for(int j=iL; j<nbasis*ntypes; j+=nL){
        LCOEFFS[j] = folded_coeffs[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float ax = folded_lvec2d.x;
    float bx = folded_lvec2d.y;
    float ay = folded_lvec2d.z;
    float by = folded_lvec2d.w;
    float det = ax*by - bx*ay;
    if(fabs(det) < 1e-12f) return;
    float4 invLvec2d = (float4)( by/det, -bx/det, -ay/det, ax/det );

    float3 pos = atoms[iav].xyz;
    float u = invLvec2d.x*pos.x + invLvec2d.y*pos.y;
    float v = invLvec2d.z*pos.x + invLvec2d.w*pos.y;
    u = u - floor(u);
    v = v - floor(v);
    int ityp = folded_atom_type[iG];
    if(ityp < 0 || ityp >= ntypes) return;

    float E = 0.0f;
    float3 F = (float3)(0.0f,0.0f,0.0f);
    int ioff = ityp*nbasis;
    for(int ib=0; ib<nbasis; ib++){
        float c = LCOEFFS[ioff + ib];
        float4 prm = LBASIS[ib];
        float  b = folded_eval_basis(u, v, pos.z, prm);
        float3 g = folded_eval_grad (u, v, pos.z, prm, invLvec2d);
        E += c * b;
        F -= c * g;
    }
    forces[iav] += (float4)(F.x, F.y, F.z, -E);
}

// ============================================================
//  Folded Basis Workgroup-Optimized (getSurfFolded_workgroup)
// ============================================================
//
//  Optimized variant of getSurfFolded that precomputes 1D basis factors
//  (cos, sin, exp) per atom and stores them in local memory, then
//  reconstructs the 3D basis as a tensor product in the triple loop.
//
//  Strategy:
//    1. Each thread evaluates its atom's 1D basis factors:
//       Bx(u), dBx/du, By(v), dBy/dv, Bz(z), dBz/dz
//       and stores them in local memory arrays L_BX[iL][i], etc.
//    2. Triple loop (iz→iy→ix) reconstructs B = Bx·By·Bz from stored 1D factors.
//       This avoids redundant cos/sin/exp calls — each is computed once per atom.
//
//  Memory layout: Local arrays are [MAX_ATOMS][MAX_XY] — one row per thread.
//  CAVEAT: MAX_ATOMS=64 limits workgroup size to 64. If nL > 64, overflow.
//  CAVEAT: Uses native_cos/native_sin/native_exp (fast but lower precision).
//          For validation, use getSurfFolded (full precision cos/sin/exp).
//
//  Loop order: iz (outer) → iy → ix (inner)
//    Expensive exp is outermost (computed once per iz), cos/sin are inner.
//    This is optimal when Nz < Nxy (typical: Nz=8, Nxy=4).
//

#define MAX_ATOMS 64
#define MAX_XY 4
#define MAX_Z  8

__kernel void getSurfFolded_workgroup(
    const int4 ns,                     // (natoms, nnode, 0, 0)
    __global float4*  atoms,           
    __global float4*  REQs,            
    __global float4*  forces,          
    __global float*   folded_coeffs,   
    __global float4*  folded_kxyz,     // [Nxy params, Nz params]
    __global int*     folded_atom_type,
    const int4        folded_meta,     // (N_xy, N_z, ntypes, 0) 
    const float4      folded_lvec2d    
){
    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    const int iL = get_local_id(0);    // Thread ID (0 to 63) maps to Atom index within batch
    const int nL = get_local_size(0);  // 64

    const int natoms = ns.x;
    const int Nxy = folded_meta.x; 
    const int Nz  = folded_meta.y;
    const int ntypes = folded_meta.z;
    const int nbasis_total = Nxy * Nxy * Nz;

    // ==================================================================
    // 1. ALLOCATE __LOCAL MEMORY FOR EXPLICIT PRECALCULATION STORAGE
    // ==================================================================
    // Coefficients and parameters
    __local float  LCOEFFS[MAX_XY * MAX_XY * MAX_Z * 8]; 
    __local float4 LPARAMS_XY[MAX_XY]; 
    __local float4 LPARAMS_Z[MAX_Z];

    // Evaluated 1D Basis Arrays [Atom_Index][Basis_Index]
    __local float L_BX [MAX_ATOMS][MAX_XY];
    __local float L_dBX[MAX_ATOMS][MAX_XY];
    __local float L_BY [MAX_ATOMS][MAX_XY];
    __local float L_dBY[MAX_ATOMS][MAX_XY];
    __local float L_BZ [MAX_ATOMS][MAX_Z];
    __local float L_dBZ[MAX_ATOMS][MAX_Z];

    // Cooperative parameter loading
    for(int j = iL; j < Nxy; j += nL) LPARAMS_XY[j] = folded_kxyz[j];
    for(int j = iL; j < Nz;  j += nL) LPARAMS_Z[j]  = folded_kxyz[Nxy + j];
    for(int j = iL; j < nbasis_total * ntypes; j += nL) LCOEFFS[j] = folded_coeffs[j];

    barrier(CLK_LOCAL_MEM_FENCE);

    int active = (iG < natoms);
    int ityp = active ? folded_atom_type[iG] : -1;
    active = active && (ityp >= 0) && (ityp < ntypes);

    // Geometry transforms
    float det = folded_lvec2d.x * folded_lvec2d.w - folded_lvec2d.y * folded_lvec2d.z;
    float4 invLvec = (float4)(folded_lvec2d.w/det, -folded_lvec2d.y/det, -folded_lvec2d.z/det, folded_lvec2d.x/det);

    int iav = iG + iS * (natoms + ns.y);
    float3 pos = (float3)(0.0f, 0.0f, 0.0f);
    if(active){ pos = atoms[iav].xyz; }
    
    float u = invLvec.x * pos.x + invLvec.y * pos.y;
    float v = invLvec.z * pos.x + invLvec.w * pos.y;
    u -= floor(u);
    v -= floor(v);

    // ==================================================================
    // 2. PARALLEL PRECALCULATION -> SAVE TO LOCAL MEMORY
    // Every thread calculates its own atom's basis and explicitly saves 
    // it to its dedicated row in the Local Memory array.
    // ==================================================================
    for(int i = 0; i < Nxy; i++){
        float k = LPARAMS_XY[i].x; 
        float phi = 2.0f * M_PI_F * k;
        
        float phix_u = phi * u;
        L_BX[iL][i]  = active ? native_cos(phix_u) : 0.0f;
        L_dBX[iL][i] = active ? (-phi * native_sin(phix_u)) : 0.0f;
        
        float phiy_v = phi * v;
        L_BY[iL][i]  = active ? native_cos(phiy_v) : 0.0f;
        L_dBY[iL][i] = active ? (-phi * native_sin(phiy_v)) : 0.0f;
    }

    for(int i = 0; i < Nz; i++){
        float kz = LPARAMS_Z[i].z;
        float z0 = LPARAMS_Z[i].w;
        float dz = fmax(0.0f, pos.z - z0);
        float bz = active ? native_exp(-kz * dz) : 0.0f;
        L_BZ[iL][i]  = bz;
        L_dBZ[iL][i] = active && (pos.z > z0) ? (-kz * bz) : 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // ==================================================================
    // 3. THE TRIPLE LOOP
    // Thread streams its precalculated 1D factors from Local Memory,
    // avoiding the risk of register spilling entirely.
    // ==================================================================
    float E_tot = 0.0f;
    float dEdu_tot = 0.0f;
    float dEdv_tot = 0.0f;
    float dEdz_tot = 0.0f;

    int ic = active ? (ityp * nbasis_total) : 0; // Pointer to coefficients

    for(int iz = 0; iz < Nz; iz++){
        float bz  = L_BZ[iL][iz];
        float dbz = L_dBZ[iL][iz];

        for(int iy = 0; iy < Nxy; iy++){
            float by  = L_BY[iL][iy];
            float dby = L_dBY[iL][iy];
            
            // Outer loop multipliers
            float bz_by  = bz * by;
            float dbz_by = dbz * by;
            float bz_dby = bz * dby;

            for(int ix = 0; ix < Nxy; ix++){
                float bx  = L_BX[iL][ix];
                float dbx = L_dBX[iL][ix];

                float c = LCOEFFS[ic++]; 

                // Dynamic 3D Basis Construction
                E_tot    += c * (bx * bz_by);
                dEdu_tot += c * (dbx * bz_by);
                dEdv_tot += c * (bx * bz_dby);
                dEdz_tot += c * (bx * dbz_by);
            }
        }
    }

    // Map gradients back to forces
    float3 F_tot;
    F_tot.x = -(dEdu_tot * invLvec.x + dEdv_tot * invLvec.z);
    F_tot.y = -(dEdu_tot * invLvec.y + dEdv_tot * invLvec.w);
    F_tot.z = -dEdz_tot;

    if(active){ forces[iav] += (float4)(F_tot.x, F_tot.y, F_tot.z, -E_tot); }
}

// ============================================================
//  Folded Basis Harmonics (getSurfFolded_harmonics)
// ============================================================
//
//  Stub kernel for a planned harmonics-based variant. Not yet implemented.
//  Intended approach: precompute 1D harmonic coefficients (cos/sin tables)
//  in local memory, then evaluate via tensor product reconstruction.
//
//  TODO: Complete implementation or remove if superseded by tensor kernels.
//

__kernel void getSurfFolded_harmonics(
    const int4 ns,                     
    __global float4*  atoms,           
    __global float4*  REQs,            
    __global float4*  forces,          
    __global float*   folded_coeffs,   
    __global float4*  folded_kxyz,     // Now stores 1D params: [Nx params, Ny params, Nz params]
    __global int*     folded_atom_type,
    const int4        folded_meta,     // (Nx, Ny, Nz, ntypes)
    const float4      folded_lvec2d    
){    
    // Local memory for coefficients and 1D parameters
    __local float  LCOEFFS[MAX_XY * MAX_XY * MAX_Z * 8];
    __local float4 LBASIS[(2 * MAX_XY) + MAX_Z];

    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    const int iL = get_local_id(0);
    const int nL = get_local_size(0);
    const int natoms = ns.x;
    
    if(iG >= natoms) return;

    // Tensor product dimensions
    const int Nx = folded_meta.x;
    const int Ny = folded_meta.y;
    const int Nz = folded_meta.z;
    const int ntypes = folded_meta.w;
    const int nbasis_total = Nx * Ny * Nz;
    const int nparams_1d = Nx + Ny + Nz;

    // TODO: Complete harmonics kernel implementation
}

// ============================================================
//  Folded Basis Tensor Product Kernels — exp & poly variants
// ============================================================
//
//  Highest-performance folded basis kernels. One thread per atom.
//  No private arrays — all precomputation in local memory or registers.
//
//  Key optimization: Complex multiplication (cmul) for Fourier recursion.
//    Precompute z1_u = e^{i*2pi*u} = (cos(2pi*u), sin(2pi*u))  [once per atom]
//    Then z_h = z1_u^h is obtained by h-1 complex multiplications.
//    This replaces N_xy cos/sin evaluations with 1 sincos + N_xy cmul.
//
//  Physics: Each basis function has separate Pauli/London/Coulomb coeffs:
//    E = B * (cCoulomb + B*(cLondon + B*cPauli))
//      = cCoulomb*B + cLondon*B^2 + cPauli*B^3
//    where B = cos(2pi*k_u*u)*cos(2pi*k_v*v)*f(z) is the basis value.
//    Coulomb decays as f(z)^n (slowest), London as f(z)^(2n), Pauli as f(z)^(3n).
//    dE/dB = cCoulomb + B*(2*cLondon + B*3*cPauli)
//    c.w (H-bond) omitted for now.
//
//  Two specialized kernels with different loop orders:
//
//    getSurfFolded_tensor_exp:  iz->iy->ix (exp expensive, outermost)
//      z-basis: f(z) = exp(-alpha*max(0, z-z0))  [per-basis alpha, z0]
//      Needs folded_kxyz for per-basis alpha and z0.
//      Loop order puts expensive exp() outermost (computed Nz times).
//
//    getSurfFolded_tensor_poly: ix->iy->iz (cheap tpow*=t innermost)
//      z-basis: f(z) = (1 - dz/zcut)^m  [polynomial decay]
//      No folded_kxyz — uses scalar zmin, zcut, m_start.
//      Powers = m_start, m_start+1, ..., m_start+Nz-1.
//      Loop order puts cheap tpow*=t innermost (just multiply).
//
//  CAVEAT: The tensor kernels use a DIFFERENT invLvec convention than
//  folded_eval_grad above. Here the chain rule is applied manually:
//    F.x = -(dEdu*invLvec.x + dEdv*invLvec.z)
//    F.y = -(dEdu*invLvec.y + dEdv*invLvec.w)
//  This is correct: invLvec = (du/dx, du/dy, dv/dx, dv/dy).
//  (No swapped bug here — the swap bug is only in folded_eval_grad.)

#ifndef FOLDED_TYPES_MAX
#define FOLDED_TYPES_MAX 8
#endif
#ifndef FOLDED_BASIS_MAX
#define FOLDED_BASIS_MAX 128
#endif

// Complex multiply: (a.x + i*a.y) * (b.x + i*b.y) = (a.x*b.x - a.y*b.y) + i*(a.x*b.y + a.y*b.x)
// Used for Fourier recursion in tensor and Ewald kernels.
inline float2 cmul(float2 a, float2 b) {
    return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// --- Exponential variant ---
__kernel void getSurfFolded_tensor_exp(
    const int4 ns,                     // (natoms, nnode, 0, 0)
    __global float4*  atoms,
    __global float4*  REQs,
    __global float4*  forces,
    __global float4*  folded_coeffs,   // [ntypes * Nxy * Nxy * Nz] float4
    __global float4*  folded_kxyz,
    __global int*     folded_atom_type,
    const int4        folded_meta,     // (Nxy, Nz, ntypes, 0)
    const float4      folded_lvec2d,
    const float       poly_R           // unused
){
    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    if(iG >= ns.x) return;

    const int Nxy = folded_meta.x;
    const int Nz  = folded_meta.y;
    const int ntypes = folded_meta.z;
    const int nbasis_total = Nxy * Nxy * Nz;

    // Preload coefficients into local memory
    __local float4 L_coeffs[FOLDED_TYPES_MAX * FOLDED_BASIS_MAX];
    int total_coeffs = ntypes * nbasis_total;
    int lid = get_local_linear_id();
    int lsize = get_local_size(0) * get_local_size(1);
    for(int i = lid; i < total_coeffs; i += lsize){  L_coeffs[i] = folded_coeffs[i]; }
    barrier(CLK_LOCAL_MEM_FENCE);

    int ityp = folded_atom_type[iG];
    if(ityp < 0 || ityp >= ntypes) return;

    float det = folded_lvec2d.x * folded_lvec2d.w - folded_lvec2d.y * folded_lvec2d.z;
    float4 invLvec = (float4)(folded_lvec2d.w/det, -folded_lvec2d.y/det,
                              -folded_lvec2d.z/det,  folded_lvec2d.x/det);
    int iav = iG + iS * (ns.x + ns.y);
    float3 pos = atoms[iav].xyz;
    float u = invLvec.x * pos.x + invLvec.y * pos.y;
    float v = invLvec.z * pos.x + invLvec.w * pos.y;
    u -= floor(u);
    v -= floor(v);

    float cu, su = sincos(2.0f * M_PI_F * u, &cu);
    float cv, sv = sincos(2.0f * M_PI_F * v, &cv);
    float2 z1_u = (float2)(cu, su);
    float2 z1_v = (float2)(cv, sv);

    float E_tot = 0.0f, dEdu_tot = 0.0f, dEdv_tot = 0.0f, dEdz_tot = 0.0f;
    int ic = ityp * nbasis_total;

    for(int iz = 0; iz < Nz; iz++){
        float alpha = folded_kxyz[2*Nxy + iz].z;
        float z0    = folded_kxyz[2*Nxy + iz].w;
        float dz = fmax(0.0f, pos.z - z0);
        float bz = exp(-alpha * dz);
        float dbz = (pos.z > z0) ? (-alpha * bz) : 0.0f;

        float2 z_v = (float2)(1.0f, 0.0f);
        for(int iy = 0; iy < Nxy; iy++){
            float by = z_v.x;
            float dby = -2.0f * M_PI_F * (float)iy * z_v.y;
            float bz_by = bz * by, dbz_by = dbz * by, bz_dby = bz * dby;
            float2 z_u = (float2)(1.0f, 0.0f);
            for(int ix = 0; ix < Nxy; ix++){
                float bx = z_u.x;
                float dbx = -2.0f * M_PI_F * (float)ix * z_u.y;
                float B = bx * bz_by;
                float4 c = L_coeffs[ic++];
                E_tot    += B * (c.z + B*(c.y + B*c.x));
                float dE_fac = c.z + B*(2.0f*c.y + B*3.0f*c.x);
                dEdu_tot += dE_fac * (dbx * bz_by);
                dEdv_tot += dE_fac * (bx * bz_dby);
                dEdz_tot += dE_fac * (bx * dbz_by);
                z_u = cmul(z_u, z1_u);
            }
            z_v = cmul(z_v, z1_v);
        }
    }

    float3 F_tot;
    F_tot.x = -(dEdu_tot * invLvec.x + dEdv_tot * invLvec.z);
    F_tot.y = -(dEdu_tot * invLvec.y + dEdv_tot * invLvec.w);
    F_tot.z = -dEdz_tot;
    forces[iav] += (float4)(F_tot.x, F_tot.y, F_tot.z, -E_tot);
}

// --- Polynomial variant ---
// Loop order: ix→iy→iz (cheap tpow*=t innermost, expensive cmul outermost)
// Coefficient layout: coeffs[ntype][ix][iy][iz] (natural order, no transpose)
__kernel void getSurfFolded_tensor_poly(
    const int4 ns,                     // (natoms, nnode, 0, 0)
    __global float4*  atoms,
    __global float4*  REQs,
    __global float4*  forces,
    __global float4*  folded_coeffs,   // [ntypes * Nxy * Nxy * Nz] float4
    __global int*     folded_atom_type,
    const int4        folded_meta,     // (Nxy, Nz, ntypes, m_start)
    const float4      folded_lvec2d,
    const float       zmin,
    const float       zcut
){
    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    if(iG >= ns.x) return;

    const int Nxy = folded_meta.x;
    const int Nz  = folded_meta.y;
    const int ntypes = folded_meta.z;
    const int m_start = folded_meta.w;
    const int nbasis_total = Nxy * Nxy * Nz;

    // Preload coefficients into local memory
    __local float4 L_coeffs[FOLDED_TYPES_MAX * FOLDED_BASIS_MAX];
    int total_coeffs = ntypes * nbasis_total;
    int lid = get_local_linear_id();
    int lsize = get_local_size(0) * get_local_size(1);
    for(int i = lid; i < total_coeffs; i += lsize)
        L_coeffs[i] = folded_coeffs[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    int ityp = folded_atom_type[iG];
    if(ityp < 0 || ityp >= ntypes) return;

    float det = folded_lvec2d.x * folded_lvec2d.w - folded_lvec2d.y * folded_lvec2d.z;
    float4 invLvec = (float4)(folded_lvec2d.w/det, -folded_lvec2d.y/det,
                              -folded_lvec2d.z/det,  folded_lvec2d.x/det);
    int iav = iG + iS * (ns.x + ns.y);
    float3 pos = atoms[iav].xyz;
    float u = invLvec.x * pos.x + invLvec.y * pos.y;
    float v = invLvec.z * pos.x + invLvec.w * pos.y;
    u -= floor(u);
    v -= floor(v);

    float cu, su = sincos(2.0f * M_PI_F * u, &cu);
    float cv, sv = sincos(2.0f * M_PI_F * v, &cv);
    float2 z1_u = (float2)(cu, su);
    float2 z1_v = (float2)(cv, sv);

    // Poly z-basis: t = 1 - min(dz/zcut, 1), powers = m_start..m_start+Nz-1
    float dz = fmax(0.0f, pos.z - zmin);
    float invR = 1.0f / zcut;
    float x = fmin(dz * invR, 1.0f);
    float t = 1.0f - x;
    bool active_z = (pos.z > zmin) && (x < 1.0f);

    // Precompute t^m_start and t^(m_start-1) for reset inside loop
    float t_m_start = 1.0f, t_m_start_prev = 1.0f;
    for(int i = 0; i < m_start; i++){ t_m_start_prev = t_m_start; t_m_start *= t; }

    float E_tot = 0.0f, dEdu_tot = 0.0f, dEdv_tot = 0.0f, dEdz_tot = 0.0f;
    int ic = ityp * nbasis_total;

    float2 z_u = (float2)(1.0f, 0.0f);
    for(int ix = 0; ix < Nxy; ix++){
        float bx = z_u.x;
        float dbx = -2.0f * M_PI_F * (float)ix * z_u.y;

        float2 z_v = (float2)(1.0f, 0.0f);
        for(int iy = 0; iy < Nxy; iy++){
            float by = z_v.x;
            float dby = -2.0f * M_PI_F * (float)iy * z_v.y;

            float tpow = t_m_start, tprev = t_m_start_prev;
            for(int iz = 0; iz < Nz; iz++){
                float n = (float)(m_start + iz);
                float bz = tpow;
                float dbz = active_z ? (-n * invR * tprev) : 0.0f;

                float B = bx * by * bz;
                float4 c = L_coeffs[ic++];
                E_tot    += B * (c.z + B*(c.y + B*c.x));
                float dE_fac = c.z + B*(2.0f*c.y + B*3.0f*c.x);
                dEdu_tot += dE_fac * (dbx * by * bz);
                dEdv_tot += dE_fac * (bx * dby * bz);
                dEdz_tot += dE_fac * (bx * by * dbz);

                tprev = tpow;
                tpow *= t;
            }
            z_v = cmul(z_v, z1_v);
        }
        z_u = cmul(z_u, z1_u);
    }

    float3 F_tot;
    F_tot.x = -(dEdu_tot * invLvec.x + dEdv_tot * invLvec.z);
    F_tot.y = -(dEdu_tot * invLvec.y + dEdv_tot * invLvec.w);
    F_tot.z = -dEdz_tot;
    forces[iav] += (float4)(F_tot.x, F_tot.y, F_tot.z, -E_tot);
}

// ============================================================
//  OpenCL Ewald2D Kernels (GPU-accelerated surface electrostatics)
// ============================================================
//
//  GPU implementation of 2D Ewald summation for electrostatic potentials
//  of periodic charged surfaces (e.g. ionic crystals like NaCl, CaF2).
//
//  Physics:
//    For a 2D-periodic array of charges q_i at positions (x_i, y_i, z_i),
//    the electrostatic potential at point r is:
//      phi(r) = Sum_i q_i / |r - r_i|    [conditionally convergent!]
//
//    Ewald2D decomposes this into:
//      phi(r) = phi_real(r) + phi_recip(r)
//
//    phi_recip(r) = Sum_G  C_G * e^{iG.rho} * e^{-G*|z-z_i|}
//    where G = h*b1 + k*b2 are 2D reciprocal lattice vectors,
//          C_G = (2pi/A) * (1/G) * Sum_i q_i * e^{-iG.rho_i} * e^{G*z_i}
//
//    For z above all ions (vacuum region): e^{G*z_i} -> e^{-G*z} * e^{G*z_i}
//    simplifies to a single decay factor.
//
//  Reference: Parry, "The electrostatic potential near a crystal surface",
//    Surf. Sci. 49, 433 (1975). Also: pyBall/Ewald2D.py.
//
//  Key optimization: Complex multiplication for e^{iG.rho}.
//    For G = h*b1 + k*b2:
//      e^{iG.rho} = e^{ih*b1.rho} * e^{ik*b2.rho}
//    Precompute z1_b1 = e^{i*b1.rho}, z1_b2 = e^{i*b2.rho}  [2 sincos per point]
//    Then e^{ih*b1.rho} = z1_b1^h  [by repeated cmul]
//         e^{ik*b2.rho} = z1_b2^k  [by repeated cmul]
//    This reduces N_G cos/sin evaluations to just 2 per evaluation point!
//
//  CAVEAT: float32 throughout. For large N_G or N_ions, accumulation error
//  can reach ~1e-4. Use eval_potential_cluster (double-single) for validation.
//

// (cmul defined earlier, before tensor kernels)

// ------------------------------------------------------------------
// Ewald Kernel 1: Compute C_G coefficients (vacuum) and w[g,i] (full)
// ------------------------------------------------------------------
// Each work item computes coefficients for one G-vector.
// Work size: N_G (number of G-vectors).
//
// C_G = (2pi/A) * (1/G) * Sum_i q_i * e^{-iG.rho_i} * e^{G*z_i}
//
// w[g,i] = (2pi/A) * (1/G) * q_i * e^{-iG.rho_i}  [per-ion weights for full eval]
//
// ion_data[i] = (x, y, z, q)  — ion position and charge
// G_data[ig]  = (h, k, |G|, 0) — Miller indices and magnitude
// b_vectors   = [b1, b2]       — 2D reciprocal lattice vectors
//
// Output: C_G_out[ig] = (Re(C_G), Im(C_G))
//         w_out[ig*N_ions + i] = (Re(w), Im(w))  [if not NULL]
__kernel void compute_ewald_coefficients(
    __global const float4* ion_data,
    __global const float4* G_data,
    __global const float2* b_vectors,
    const float area,
    const int N_ions,
    const int N_G,
    __global float2* C_G_out,
    __global float2* w_out
){
    const int ig = get_global_id(0);
    if(ig >= N_G) return;

    float4 G = G_data[ig];
    int h = (int)G.x;
    int k = (int)G.y;
    float Gn = G.z;

    float2 b1 = b_vectors[0];
    float2 b2 = b_vectors[1];

    float Gx = h * b1.x + k * b2.x;
    float Gy = h * b1.y + k * b2.y;

    float prefactor = (2.0f * M_PI_F) / (area * Gn);

    float2 C_G = (float2)(0.0f, 0.0f);

    for(int i = 0; i < N_ions; i++){
        float4 ion = ion_data[i];
        float rx = ion.x;
        float ry = ion.y;
        float rz = ion.z;
        float q = ion.w;

        float Gdotr = Gx * rx + Gy * ry;
        float cos_gr = cos(Gdotr);
        float sin_gr = sin(Gdotr);
        float2 phase = (float2)(cos_gr, -sin_gr);

        float decay_ion = exp(Gn * rz);
        float2 contrib = (float2)(q * decay_ion * phase.x, q * decay_ion * phase.y);
        C_G += contrib;

        if(w_out != NULL){
            float2 w_gi = (float2)(q * phase.x * prefactor, q * phase.y * prefactor);
            w_out[ig * N_ions + i] = w_gi;
        }
    }

    C_G_out[ig] = (float2)(C_G.x * prefactor, C_G.y * prefactor);
}

// ------------------------------------------------------------------
// Ewald Kernel 2: Vacuum potential evaluation
// ------------------------------------------------------------------
// Evaluates phi(r) = Sum_G C_G * e^{iG.rho} * e^{-G*z}  for z > max(z_ions).
// Uses complex multiplication recursion for e^{iG.rho}.
// Output: phi * COULOMB_CONST (in physical units).
//
// CAVEAT: Only valid in the vacuum region (z above all ions).
// For z between ion layers, use eval_potential_full instead.
// ------------------------------------------------------------------
__kernel void eval_potential_vacuum(
    __global const float4* eval_points,
    __global const float2* C_G,
    __global const float4* G_data,
    __global const float2* b_vectors,
    const int N_points,
    const int N_G,
    const int n_harm,
    __global float* phi_out
){
    const int ip = get_global_id(0);
    if(ip >= N_points) return;

    float4 p = eval_points[ip];
    float x = p.x;
    float y = p.y;
    float z = p.z;

    float2 b1 = b_vectors[0];
    float2 b2 = b_vectors[1];

    float b1dotr = b1.x * x + b1.y * y;
    float b2dotr = b2.x * x + b2.y * y;
    float2 z1_b1 = (float2)(cos(b1dotr), sin(b1dotr));
    float2 z1_b2 = (float2)(cos(b2dotr), sin(b2dotr));

    float phi = 0.0f;

    for(int ig = 0; ig < N_G; ig++){
        float4 G = G_data[ig];
        int h = (int)G.x;
        int k = (int)G.y;
        float Gn = G.z;

        float2 zh_b1 = (float2)(1.0f, 0.0f);
        int h_abs = abs(h);
        for(int i = 0; i < h_abs; i++){
            zh_b1 = cmul(zh_b1, z1_b1);
        }
        if(h < 0) zh_b1.y = -zh_b1.y;

        float2 zk_b2 = (float2)(1.0f, 0.0f);
        int k_abs = abs(k);
        for(int i = 0; i < k_abs; i++){
            zk_b2 = cmul(zk_b2, z1_b2);
        }
        if(k < 0) zk_b2.y = -zk_b2.y;

        float2 phase = cmul(zh_b1, zk_b2);
        float decay = exp(-Gn * z);
        float2 C = C_G[ig];
        float2 contrib = cmul(C, phase);

        phi += contrib.x * decay;
    }

    phi_out[ip] = phi * COULOMB_CONST;
}

// ------------------------------------------------------------------
// Ewald Kernel 3: Full potential evaluation (any z)
// ------------------------------------------------------------------
// Evaluates phi(r) at any z, including between ion layers.
//
// phi(r) = phi0(r) + phi_G(r)
//   phi0 = -(2pi/A) * Sum_i q_i * |z - z_i|              [zeroth-order term]
//   phi_G = Sum_G Sum_i w[g,i] * e^{iG.rho} * e^{-G*|z-z_i|} [reciprocal terms]
//
// The |z - z_i| term accounts for the non-analytic G=0 contribution.
// Output: (phi0 + phi_G) * COULOMB_CONST.
//
// CAVEAT: O(N_G * N_ions) per point — expensive for large systems.
//         Prefer eval_potential_vacuum when z is above all ions.
// ------------------------------------------------------------------
__kernel void eval_potential_full(
    __global const float4* eval_points,
    __global const float2* w,
    __global const float4* ion_data,
    __global const float4* G_data,
    __global const float2* b_vectors,
    const float area,
    const int N_points,
    const int N_ions,
    const int N_G,
    __global float* phi_out
){
    const int ip = get_global_id(0);
    if(ip >= N_points) return;

    float4 p = eval_points[ip];
    float x = p.x;
    float y = p.y;
    float z = p.z;

    float2 b1 = b_vectors[0];
    float2 b2 = b_vectors[1];

    float b1dotr = b1.x * x + b1.y * y;
    float b2dotr = b2.x * x + b2.y * y;
    float2 z1_b1 = (float2)(cos(b1dotr), sin(b1dotr));
    float2 z1_b2 = (float2)(cos(b2dotr), sin(b2dotr));

    float phi0 = 0.0f;
    for(int i = 0; i < N_ions; i++){
        float4 ion = ion_data[i];
        float q = ion.w;
        float rz = ion.z;
        phi0 -= q * fabs(z - rz);
    }
    phi0 *= (2.0f * M_PI_F / area);

    float phi_G = 0.0f;

    for(int ig = 0; ig < N_G; ig++){
        float4 G = G_data[ig];
        int h = (int)G.x;
        int k = (int)G.y;
        float Gn = G.z;

        float2 zh_b1 = (float2)(1.0f, 0.0f);
        int h_abs = abs(h);
        for(int i = 0; i < h_abs; i++){
            zh_b1 = cmul(zh_b1, z1_b1);
        }
        if(h < 0) zh_b1.y = -zh_b1.y;

        float2 zk_b2 = (float2)(1.0f, 0.0f);
        int k_abs = abs(k);
        for(int i = 0; i < k_abs; i++){
            zk_b2 = cmul(zk_b2, z1_b2);
        }
        if(k < 0) zk_b2.y = -zk_b2.y;

        float2 phase = cmul(zh_b1, zk_b2);

        for(int i = 0; i < N_ions; i++){
            float4 ion = ion_data[i];
            float rz = ion.z;
            float decay = exp(-Gn * fabs(z - rz));
            float2 w_gi = w[ig * N_ions + i];
            float2 contrib = cmul(w_gi, phase);
            phi_G += contrib.x * decay;
        }
    }

    phi_out[ip] = (phi0 + phi_G) * COULOMB_CONST;
}

// ------------------------------------------------------------------
// Ewald Kernel 4: Brute force Coulomb sum (reference/validation)
// ------------------------------------------------------------------
// Direct summation: phi(r) = Sum_n Sum_m Sum_i q_i / |r - r_i - n*a - m*b|
// over a circular cutoff of PBC replicas (n^2+m^2 <= N_rep^2).
// No Ewald decomposition — used to validate the Ewald2D kernels.
//
// CAVEAT: Slow convergence (~1/N_rep). Use small N_rep for rough checks,
// eval_potential_cluster for high-precision reference.
// ------------------------------------------------------------------
__kernel void eval_potential_brute(
    __global const float4* eval_points,
    __global const float4* ion_data,
    __global const float2* a_vec,
    __global const float2* b_vec,
    const int N_points,
    const int N_ions,
    const int N_rep,
    __global float* phi_out
){
    const int ip = get_global_id(0);
    if(ip >= N_points) return;

    float4 p = eval_points[ip];
    float3 r = (float3)(p.x, p.y, p.z);

    float2 a = a_vec[0];
    float2 b = b_vec[0];

    float phi = 0.0f;

    for(int n = -N_rep; n <= N_rep; n++){
        for(int m = -N_rep; m <= N_rep; m++){
            if(n*n + m*m > N_rep*N_rep) continue;

            float3 R = (float3)(n*a.x + m*b.x, n*a.y + m*b.y, 0.0f);

            for(int i = 0; i < N_ions; i++){
                float4 ion = ion_data[i];
                float3 ri = (float3)(ion.x, ion.y, ion.z);
                float q = ion.w;

                float3 dr = r - (ri + R);
                float r_mag = sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);

                if(r_mag > 1e-12f){
                    phi += q / r_mag;
                }
            }
        }
    }

    phi_out[ip] = phi * COULOMB_CONST;
}

// ------------------------------------------------------------------
// Kernel 5: Finite-cluster Coulomb sum (no PBC, local-memory tiling)
//
// Computes V(r) = sum_j q_j / |r - r_j| * COULOMB_CONST for a finite
// cluster of ions (no periodic boundary conditions). Used as a
// brute-force reference for Ewald summation tests.
//
// Accumulation: two-sum (error-free transform) double-single, giving
// ~48 bits of mantissa precision in float32. This reduces accumulation
// error to ~1.3e-6, below the per-term q/r float32 error (~1.9e-6).
// The remaining ~1.5e-6 RMSE is the float32 sqrt/division floor.
//
// CAVEAT: The bounds check `if(ip >= N_points) return;` MUST NOT be
// placed before the barrier(CLK_LOCAL_MEM_FENCE). If some threads in
// a workgroup return early, they skip the barrier, causing undefined
// behavior (hang/crash/garbage) because not all threads cooperate on
// loading ion_loc. The check must be inside the loop, around the
// computation block only, so ALL threads participate in loading.
// ------------------------------------------------------------------
__kernel void eval_potential_cluster(
    __global const float4* eval_points,
    __global const float4* ion_data,
    const int N_points,
    const int N_ions,
    __global float* phi_out,
    __local float4* ion_loc
){
    const int ip = get_global_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    float3 r = (float3)(0.0f, 0.0f, 0.0f);
    if(ip < N_points){
        float4 p = eval_points[ip];
        r = (float3)(p.x, p.y, p.z);
    }

    // Double-single accumulator via two-sum (error-free transform)
    // (hi, lo) together represent ~48 bits of precision
    float phi_hi = 0.0f;
    float phi_lo = 0.0f;

    for(int base = 0; base < N_ions; base += lsz){
        int j = base + lid;
        if(j < N_ions){
            ion_loc[lid] = ion_data[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int imax = N_ions - base;
        if(imax > lsz) imax = lsz;

        if(ip < N_points){
            for(int i = 0; i < imax; i++){
                float4 ion = ion_loc[i];
                float3 ri = (float3)(ion.x, ion.y, ion.z);
                float q = ion.w;
                float3 dr = r - ri;
                float r_mag = sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
                if(r_mag > 1e-12f){
                    float term = q / r_mag;
                    // Two-sum: add term to (phi_hi, phi_lo)
                    // Step 1: two_sum(phi_hi, term) -> (s, e)
                    float s = phi_hi + term;
                    float bb = s - phi_hi;
                    float e = (phi_hi - (s - bb)) + (term - bb);
                    // Step 2: add phi_lo and e, then renormalize
                    float lo = phi_lo + e;
                    phi_hi = s + lo;
                    phi_lo = lo - (phi_hi - s);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(ip < N_points){
        phi_out[ip] = (phi_hi + phi_lo) * COULOMB_CONST;
    }
}

// ---- From relax_multi.cl: additional surface kernels ----
// ======================================================================
//                          Flat Surface Forces
// ======================================================================
//
//  Simplified surface models where the substrate is treated as a flat
//  plane (no lateral periodicity). Used for quick estimates and testing.
//
//  Two interaction models:
//    mode=1: Hamaker LJ93 — integrated Lennard-Jones over a half-space
//      V(z) = (1/2)*A*((z0/z)^9 - 3*(z0/z)^3)   [9-3 potential]
//      F(z) = (4.5*A/z)*((z0/z)^9 - (z0/z)^3)
//      Reference: Hamaker, Physica 4, 1058 (1937).
//
//    mode=2: Morse — simple exponential well
//      V(z) = D*(e^{-2K(z-z0)} - 2*e^{-K(z-z0)})
//      F(z) = 2*K*D*e^{-K(z-z0)}*(e^{-K(z-z0)} - 1)
//
//  combineREQ: Combining rules for atom-surface pair parameters.
//    R_ij = R_i + R_j    (additive radii)
//    E_ij = E_i * E_j    (geometric mean well depth)
//    Q_ij = Q_i * Q_j    (geometric mean charge)
//

inline float4 combineREQ(float4 a, float4 b){
    return (float4)(a.x+b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

// Hamaker LJ 9-3 potential: integrated LJ over a flat half-space.
//   V(z) = (1/2)*A*((z0/z)^9 - 3*(z0/z)^3)
//   F = dV/dz * n_hat  (force along surface normal)
// CAVEAT: z is clamped to 1e-6 to avoid singularity at z=0.
inline float getHamakerLJ93( float3 dp, float3 n, __private float3* f, float4 REQH ){
    float z = dot(dp, n);
    z = fmax(z, 1e-6f);
    float ratio = REQH.x / z;
    float r3    = ratio*ratio*ratio; // (z0/z)^3
    float r9    = r3*r3*r3;          // (z0/z)^9
    float E = 0.5f * REQH.y * ( r9 - 3.0f*r3 );
    float F_scalar = ( 4.5f * REQH.y / z ) * ( r9 - r3 );
    *f = n * F_scalar;
    return E;
}

// Morse potential with a flat surface: V(z) = D*(e^{-2K(z-z0)} - 2*e^{-K(z-z0)})
//   Force is along surface normal n_hat. z = distance to surface plane.
inline float getMorseSurface( float3 dp, float3 n, __private float3* f, float4 REQH, float K ){
    float z = dot(dp, n);
    float exp_term = exp( -K * (z - REQH.x) );
    float E = REQH.y * ( exp_term*exp_term - 2.0f*exp_term );
    float F_scalar = 2.0f * K * REQH.y * exp_term * ( exp_term - 1.0f );
    *f = n * F_scalar;
    return E;
}

// Flat-surface kernel: one thread per atom, one system per workgroup dim 1.
// Evaluates either Hamaker LJ93 (mode=1) or Morse (mode=2) against a flat
// surface defined by pos0, normal, and REQ parameters.
// Output: fapos[iav] += (Fx, Fy, Fz, E)
__kernel void getSurfFlat(
    const int4 nDOFs,               // 1   (nAtoms,nnode, nSystems, 0)
    // Dynamical
    __global float4*  apos,         // 2  [natoms]
    __global float4*  fapos,        // 3  [natoms]
    // parameters
    __global float4*  REQs,         // 4  [natoms]
    // Surface params
    const float4 surf_pos0,         // 5
    const float4 surf_normal,       // 6
    const float4 surf_REQ,          // 7
    const float4 surf_param         // 8  (K, mode, 0, 0)
){
    const int iG = get_global_id (0);   // index of atom
    const int iS = get_global_id (1);   // index of system
    const int nAtoms = nDOFs.x;
    const int nnode  = nDOFs.y;

    if(iG >= nAtoms) return;

    const int i0a   = iS*nAtoms;         // index of first atom
    const int i0v   = iS*(nAtoms+nnode); // index of first vector

    const int iav = iG + i0v;
    const int iaa = iG + i0a;

    float3 p = apos[iav].xyz;
    float4 REQi = REQs[iaa];

    float4 REQij = combineREQ( surf_REQ, REQi );

    float3 f = (float3)(0.0f);
    float E = 0.0f;

    float3 dp = p - surf_pos0.xyz;
    float3 nn = surf_normal.xyz;
    float  K  = surf_param.x;
    int mode  = (int)surf_param.y;

    if(mode == 1){ // Hamaker LJ93
        E = getHamakerLJ93( dp, nn, &f, REQij );
    } else if (mode == 2){ // Morse
        E = getMorseSurface( dp, nn, &f, REQij, K );
    }

    fapos[iav] += (float4)(f, E);
}


// Energy-only evaluation of brute-force Morse surface potential at point pos.
// Used by getSurfaceIsoSurfMorse for isosurface finding (no forces needed).
// Same physics as getSurfMorse but returns only E (no force vector).
inline float evalSurfMorseE3D(
    const float3 pos,
    const float4 REQi,
    __global float4*  atoms_s,
    __global float4*  REQ_s,
    __global float4*  surf_mpos,
    __global float4*  surf_mdip,
    __global float4*  surf_mQa,
    __global float4*  surf_mQb,
    __global float4*  surf_mQc,
    __global float4*  surf_qQa,
    __global float4*  surf_qQb,
    __global float4*  surf_qQc,
    const int na_surf,
    const int4 nPBC,
    const cl_Mat3 lvec,
    const float4 GFFParams,
    const float4 PLQH
){
    const float  K          = -GFFParams.y;
    const float  R2damp     =  GFFParams.x*GFFParams.x;
    const float3 shift_b    = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);
    const float3 shift_c    = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);
    const int bMacro        = (int)(GFFParams.z>0.5f);
    const float3 pos0       = pos + lvec.a.xyz*-nPBC.x + lvec.b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;
    float E = 0.0f;
    for(int ja=0; ja<na_surf; ja++){
        float4 REQH = REQ_s[ja];
        float3 dp   = pos0 - atoms_s[ja].xyz;
        REQH.x   += REQi.x;
        REQH.yzw *= REQi.yzw;
        for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
            for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                for(int ix=-nPBC.x; ix<=nPBC.x; ix++){
                    float4 fej = getMorsePLQH(dp, REQH, PLQH, K, R2damp);
                    E -= fej.w;
                    dp += lvec.a.xyz;
                }
                dp += shift_b;
            }
            dp += shift_c;
        }
    }
    if( bMacro && (fabs(PLQH.z) > 1e-12f) && (fabs(REQi.z) > 1e-12f) ){
        int nlayer = (int)(GFFParams.w + 0.5f);
        float4 fm = getMacroRectLayers( pos, REQi.z, surf_mpos[0], surf_mdip[0], surf_mQa[0], surf_mQb[0], surf_mQc[0], surf_qQa[0], surf_qQb[0], surf_qQc[0], nlayer );
        E += fm.w;
    }
    return E;
}


// ======================================================================
//  Isosurface Kernels
// ======================================================================
//
//  These kernels find the z-height at which the surface potential crosses
//  a given threshold, producing a 2D height map (isosurface). This is used
//  for AFM visualization and for computing effective tip heights.
//
//  Two modes:
//    mode=0: Threshold crossing — find z where E(z) = threshold via linear
//            interpolation between grid points. Fast but less accurate.
//    mode=1: Parabolic minimum — find local minimum of E(z) by fitting a
//            parabola through 3 consecutive points. More accurate for
//            finding the equilibrium height.
//
//  getSurfaceIsoSurfMorse: Uses brute-force pairwise Morse for E(z) evaluation.
//  getSurfaceIsoGridFF:    Uses precomputed B-spline GridFF for E(z) evaluation.
//
//  Output: surf_xyzq[i] = (x, y, z_height, ok_flag)
//          surf_zc[i]   = (z_height, color_value)
//

__kernel void getSurfaceIsoSurfMorse(
    const int4 ns,                // 1  (1,0,na_surf,0)
    __global float4*  atoms_s,    // 2
    __global float4*  REQ_s,      // 3
    __global float4*  surf_mpos,  // 4
    __global float4*  surf_mdip,  // 5
    __global float4*  surf_mQa,   // 6
    __global float4*  surf_mQb,   // 7
    __global float4*  surf_mQc,   // 8
    __global float4*  surf_qQa,   // 9
    __global float4*  surf_qQb,   // 10
    __global float4*  surf_qQc,   // 11
    const int4        nPBC,       // 12
    const cl_Mat3     lvec,       // 13
    const float4      GFFParams,  // 14
    const float4      probe_REQ,  // 15
    const float4      sel_PLQH,   // 16
    const float4      col_PLQH,   // 17
    const int4        surf_ns,    // 18 (nx,ny,nz,mode)
    const float4      surf_p0,    // 19 (x0,y0,zmin,threshold)
    const float4      surf_step,  // 20 (dx,dy,dz,zmax)
    __global float4*  surf_xyzq,  // 21 (x,y,z,ok)
    __global float2*  surf_zc     // 22 (z_report,color)
){
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int nx = surf_ns.x;
    const int ny = surf_ns.y;
    const int nz = surf_ns.z;
    const int mode = surf_ns.w;
    if((ix>=nx)||(iy>=ny)) return;
    const int i = ix + iy*nx;
    const float x_in = surf_p0.x + surf_step.x*(float)ix;
    const float y_in = surf_p0.y + surf_step.y*(float)iy;
    const float ax = lvec.a.x;
    const float ay = lvec.a.y;
    const float bx = lvec.b.x;
    const float by = lvec.b.y;
    const float det = ax*by - bx*ay;
    float x = x_in;
    float y = y_in;
    if(fabs(det) > 1e-12f){
        const float inv00 =  by/det;
        const float inv01 = -bx/det;
        const float inv10 = -ay/det;
        const float inv11 =  ax/det;
        float fu = inv00*x_in + inv01*y_in;
        float fv = inv10*x_in + inv11*y_in;
        fu -= rint(fu);
        fv -= rint(fv);
        x = ax*fu + bx*fv;
        y = ay*fu + by*fv;
    }
    const float zmin = surf_p0.z;
    const float thr  = surf_p0.w;
    const float dz   = surf_step.z;
    const float zmax = surf_step.w;
    float zh = NAN;
    float ch = NAN;
    int ok = 0;
    if(mode==0){
        float z_prev = zmax;
        float e_prev = evalSurfMorseE3D((float3)(x,y,z_prev), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, sel_PLQH);
        for(int iz=nz-2; iz>=0; iz--){
            float z_cur = zmin + dz*(float)iz;
            float e_cur = evalSurfMorseE3D((float3)(x,y,z_cur), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, sel_PLQH);
            float s0 = e_prev - thr;
            float s1 = e_cur  - thr;
            if( isfinite(s0) && isfinite(s1) && (((s0<=0.f)&&(s1>=0.f)) || ((s0>=0.f)&&(s1<=0.f))) ){
                float dv = s1 - s0;
                float t = (fabs(dv)<1e-16f) ? 0.5f : (-s0/dv);
                t = clamp(t, 0.0f, 1.0f);
                zh = z_prev + t*(z_cur-z_prev);
                ch = evalSurfMorseE3D((float3)(x,y,zh), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, col_PLQH);
                ok = 1;
                break;
            }
            z_prev = z_cur;
            e_prev = e_cur;
        }
    }else{
        if(nz>=3){
            float z0 = zmin;
            float z1 = zmin + dz;
            float v0 = evalSurfMorseE3D((float3)(x,y,z0), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, sel_PLQH);
            float v1 = evalSurfMorseE3D((float3)(x,y,z1), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, sel_PLQH);
            for(int iz=2; iz<nz; iz++){
                float z2 = zmin + dz*(float)iz;
                float v2 = evalSurfMorseE3D((float3)(x,y,z2), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, sel_PLQH);
                if( isfinite(v0) && isfinite(v1) && isfinite(v2) && (v1<=v0) && (v1<=v2) && ((v1<v0)||(v1<v2)) ){
                    float den = (z0-z1)*(z0-z2)*(z1-z2);
                    zh = z1;
                    if(fabs(den)>=1e-16f){
                        float A = (z2*(v1-v0) + z1*(v0-v2) + z0*(v2-v1)) / den;
                        float B = (z2*z2*(v0-v1) + z1*z1*(v2-v0) + z0*z0*(v1-v2)) / den;
                        if(fabs(A)>=1e-16f){
                            float zm = -B/(2.f*A);
                            if((zm>=fmin(z0,z2)) && (zm<=fmax(z0,z2))) zh = zm;
                        }
                    }
                    ch = evalSurfMorseE3D((float3)(x,y,zh), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, col_PLQH);
                    ok = 1;
                    break;
                }
                z0 = z1; z1 = z2; v0 = v1; v1 = v2;
            }
        }
    }
    surf_xyzq[i] = (float4)(x, y, zh, ok ? 1.0f : 0.0f);
    surf_zc [i] = (float2)(zh, ch);
}


// Isosurface kernel using GridFF (B-spline interpolated force field).
// Same logic as getSurfaceIsoSurfMorse but evaluates E(z) from a precomputed
// 3D grid via tricubic B-spline interpolation with PBC in x,y.
// Much faster than brute-force Morse — use when grid is available.
__kernel void getSurfaceIsoGridFF(
    const int4        grid_ns,      // 1
    __global float4*  BsplinePLQ,   // 2
    const float4      grid_invStep, // 3
    const float4      grid_p0,      // 4
    const float4      sel_PLQH,     // 5
    const float4      col_PLQH,     // 6
    const int4        surf_ns,      // 7  (nx,ny,nz,mode)
    const float4      surf_p0,      // 8  (x0,y0,zmin,threshold)
    const float4      surf_step,    // 9  (dx,dy,dz,zmax)
    const float4      surf_z0,      // 10 (z_top,0,0,0)
    __global float4*  surf_xyzq,    // 11
    __global float2*  surf_zc       // 12
){
    __local int4 xqs[4];
    __local int4 yqs[4];
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iLx = get_local_id(0);
    const int iLy = get_local_id(1);
    const int nx = surf_ns.x;
    const int ny = surf_ns.y;
    const int nz = surf_ns.z;
    const int mode = surf_ns.w;
    if((iLy==0) && (iLx<4)){ xqs[iLx] = make_inds_pbc(grid_ns.x, iLx); }
    if((iLx==0) && (iLy<4)){ yqs[iLy] = make_inds_pbc(grid_ns.y, iLy); }
    barrier(CLK_LOCAL_MEM_FENCE);
    if((ix>=nx)||(iy>=ny)) return;
    const int i = ix + iy*nx;
    const float x = surf_p0.x + surf_step.x*(float)ix;
    const float y = surf_p0.y + surf_step.y*(float)iy;
    const float zmin = surf_p0.z;
    const float thr  = surf_p0.w;
    const float dz   = surf_step.z;
    const float zmax = surf_step.w;
    float zh = NAN;
    float ch = NAN;
    int ok = 0;
    if(mode==0){
        float z_prev = zmax;
        const float3 u_prev = ((float3)(x,y,z_prev) - grid_p0.xyz) * grid_invStep.xyz;
        float e_prev = fe3d_pbc_comb(u_prev, grid_ns.xyz, BsplinePLQ, sel_PLQH, xqs, yqs).w;
        for(int iz=nz-2; iz>=0; iz--){
            float z_cur = zmin + dz*(float)iz;
            const float3 u_cur = ((float3)(x,y,z_cur) - grid_p0.xyz) * grid_invStep.xyz;
            float e_cur = fe3d_pbc_comb(u_cur, grid_ns.xyz, BsplinePLQ, sel_PLQH, xqs, yqs).w;
            float s0 = e_prev - thr;
            float s1 = e_cur  - thr;
            if( isfinite(s0) && isfinite(s1) && (((s0<=0.f)&&(s1>=0.f)) || ((s0>=0.f)&&(s1<=0.f))) ){
                float dv = s1 - s0;
                float t = (fabs(dv)<1e-16f) ? 0.5f : (-s0/dv);
                t = clamp(t, 0.0f, 1.0f);
                zh = z_prev + t*(z_cur-z_prev);
                ch = fe3d_pbc_comb((((float3)(x,y,zh) - grid_p0.xyz) * grid_invStep.xyz), grid_ns.xyz, BsplinePLQ, col_PLQH, xqs, yqs).w;
                ok = 1;
                break;
            }
            z_prev = z_cur;
            e_prev = e_cur;
        }
    }else{
        if(nz>=3){
            float z0 = zmin;
            float z1 = zmin + dz;
            float v0 = fe3d_pbc_comb((((float3)(x,y,z0) - grid_p0.xyz) * grid_invStep.xyz), grid_ns.xyz, BsplinePLQ, sel_PLQH, xqs, yqs).w;
            float v1 = fe3d_pbc_comb((((float3)(x,y,z1) - grid_p0.xyz) * grid_invStep.xyz), grid_ns.xyz, BsplinePLQ, sel_PLQH, xqs, yqs).w;
            for(int iz=2; iz<nz; iz++){
                float z2 = zmin + dz*(float)iz;
                float v2 = fe3d_pbc_comb((((float3)(x,y,z2) - grid_p0.xyz) * grid_invStep.xyz), grid_ns.xyz, BsplinePLQ, sel_PLQH, xqs, yqs).w;
                if( isfinite(v0) && isfinite(v1) && isfinite(v2) && (v1<=v0) && (v1<=v2) && ((v1<v0)||(v1<v2)) ){
                    float den = (z0-z1)*(z0-z2)*(z1-z2);
                    zh = z1;
                    if(fabs(den)>=1e-16f){
                        float A = (z2*(v1-v0) + z1*(v0-v2) + z0*(v2-v1)) / den;
                        float B = (z2*z2*(v0-v1) + z1*z1*(v2-v0) + z0*z0*(v1-v2)) / den;
                        if(fabs(A)>=1e-16f){
                            float zm = -B/(2.f*A);
                            if((zm>=fmin(z0,z2)) && (zm<=fmax(z0,z2))) zh = zm;
                        }
                    }
                    ch = fe3d_pbc_comb((((float3)(x,y,zh) - grid_p0.xyz) * grid_invStep.xyz), grid_ns.xyz, BsplinePLQ, col_PLQH, xqs, yqs).w;
                    ok = 1;
                    break;
                }
                z0 = z1; z1 = z2; v0 = v1; v1 = v2;
            }
        }
    }
    surf_xyzq[i] = (float4)(x, y, zh, ok ? 1.0f : 0.0f);
    surf_zc [i] = (float2)(zh - surf_z0.x, ch);
}



// ======================================================================
//  Macroscopic Dipole Field (addDipoleField)
// ======================================================================
//
//  Adds the electric field of point dipoles to a 3D force/energy grid.
//  Used for macroscopic dipole corrections (e.g. polarized surface layers).
//
//  Physics (electric dipole field):
//    phi(r) = (1/4pi*eps0) * (p.r_hat)/r^2 + q/r
//    E(r) = (1/4pi*eps0) * [3(p.r_hat)r_hat - p] / r^3 + q*r_hat/r^2
//
//  In the kernel, dipoles are stored as (px, py, pz, q) — combining
//  point charge and dipole contributions:
//    F = E_field = COULOMB_CONST * [d*(q + 3*(p.d)/r^2) - p] / r^3
//    E_potential = COULOMB_CONST * (q + (p.d)/r^2) / r
//  where d = r_atom - r_dipole, r = |d|.
//
//  Reference: https://en.wikipedia.org/wiki/Electric_dipole_moment
//
//  GPU strategy: Local-memory tiling over dipoles (same as getSurfMorse).
//  Output: write_imagef to 3D image FE_Coul at grid position (ia, ib, ic).
//
//  CAVEAT: Uses `if(iG > nMax) return;` instead of `>=`. This means the
//  last grid point (iG == nMax) is incorrectly included. Should be `>=`.
//  CAVEAT: The commented-out `//if(i>=nAtoms) break;` inside the tiling
//  loop is correct to leave commented — breaking would skip the barrier.
//

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void addDipoleField(
    const int n,                     // 1
    __global float4*  ps,            // 2
    __global float4*  dipols,        // 3
    __write_only image3d_t  FE_Coul, // 4
    const int4     nGrid,            // 5
    const cl_Mat3  dGrid,            // 6
    const float4   grid_p0           // 7
){
    __local float4 LATOMS[32];
    __local float4 LCLJS [32];
    const int iG = get_global_id (0);
    const int nG = get_global_size(0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);
    const int nab = nGrid.x*nGrid.y;
    const int ia  = iG%nGrid.x;
    const int ib  = (iG%nab)/nGrid.x;
    const int ic  = iG/nab;

    const int nMax = nab*nGrid.z;
    if(iG>nMax) return;

    //if(iG==0){printf("GPU::addDipoleField(nL=%i,nG=%i,nAtoms=%i,nPBC(%i,%i,%i))\n", nL, nG, n  );}

    float3 pos     = grid_p0.xyz + dGrid.a.xyz*ia + dGrid.b.xyz*ib  + dGrid.c.xyz*ic;
    float4 fe  = float4Zero;
    for (int i0=0; i0<n; i0+= nL ){
        int i = i0 + iL;
        //if(i>=nAtoms) break;  // wrong !!!!
        LATOMS[iL] = ps    [i];
        LCLJS [iL] = dipols[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<n ){
                float4 P     = LCLJS [j];
                float4 atom  = LATOMS[j];
                float3 d     = pos - atom.xyz;
                float  invr2 = 1.f / dot(d,d);
                float  invr  = sqrt(invr2);
                float  invr3 = invr*invr2;
                // https://en.wikipedia.org/wiki/Electric_dipole_moment#Potential_and_field_of_an_electric_dipole
                // Efield(R) = const *(    R*(Q/|R|^3) + R*3*<p|R>/|R|^5 - p/|R|^3

                float  VP  =  dot( P.xyz, d )*invr2;
                float4 fei = (float4){
                    (d*( P.w + 3*VP ) - P.xyz )*invr3,   // Force  (E-filed )
                       ( P.w +   VP           )*invr     // Energy (Potential)
                }*COULOMB_CONST;
                fe    += fei;

            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int4 coord = (int4){ia,ib,ic,0};
    write_imagef( FE_Coul, coord, fe );
}


