// faf_eval.cl — FAF evaluation macros (macro library for injection).
//
// Extracted from surface_spammm.cl. Contains folded-atomic forcefield (FAF)
// evaluation kernels as //>>>macro blocks, plus helper inline functions at the
// top (always emitted). The macros are injected into getNonBonded of
// UFF.cl/SPFF.cl/RAFF.cl/RigidMolFF.cl via //<<<macro NAME.
// See doc/topical_audit/gridff_faf.md. Requires common.cl + Forces.cl first.

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

inline float2 cmul(float2 a, float2 b) {
    return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline float4 combineREQ(float4 a, float4 b){
    return (float4)(a.x+b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

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

inline float getMorseSurface( float3 dp, float3 n, __private float3* f, float4 REQH, float K ){
    float z = dot(dp, n);
    float exp_term = exp( -K * (z - REQH.x) );
    float E = REQH.y * ( exp_term*exp_term - 2.0f*exp_term );
    float F_scalar = 2.0f * K * REQH.y * exp_term * ( exp_term - 1.0f );
    *f = n * F_scalar;
    return E;
}

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

//>>>macro GET_SURF_FOLDED
// Original kernel: __kernel void getSurfFolded(...)
// Converted to macro for injection into getNonBonded.
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

//>>>macro GET_SURF_FOLDED_WORKGROUP
// Original kernel: __kernel void getSurfFolded_workgroup(...)
// Converted to macro for injection into getNonBonded.
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

//>>>macro GET_SURF_FOLDED_HARMONICS
// Original kernel: __kernel void getSurfFolded_harmonics(...)
// Converted to macro for injection into getNonBonded.
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

//>>>macro GET_SURF_FOLDED_TENSOR_EXP
// Original kernel: __kernel void getSurfFolded_tensor_exp(...)
// Converted to macro for injection into getNonBonded.
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

//>>>macro GET_SURF_FOLDED_TENSOR_POLY
// Original kernel: __kernel void getSurfFolded_tensor_poly(...)
// Converted to macro for injection into getNonBonded.
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

