// ported from SPAMMM/kernels/SPFF.cl (SPFFsp3 = FireCore MMFFsp3 GPU forcefield)
// spff.cl - SPFFsp3 force field: bonding interactions + MD integrator
// ====================================================================
//
// SPFFsp3 FORCE FIELD FOR GPU MOLECULAR DYNAMICS
// ==============================================
//
// Implements the SPFFsp3 (Simple Pauli Force Field for sp3 atoms) force
// field for molecular dynamics with pi-orbital degrees of freedom.
// Designed for covalent systems (organic molecules, sp3/sp2 hybridization)
// where each atom has up to 4 sigma-bonded neighbors plus an optional
// pi-orbital vector representing the p-orbital orientation.
//
// --- Physics Overview ---
//
// The SPFFsp3 force field includes the following interaction terms:
//
//   1. Bond stretching (harmonic):
//      V_bond = (1/2) * k * (r - r0)^2
//      F_bond = -k * (r - r0) * r_hat
//
//   2. Angle bending (cos-half formulation):
//      V_angle = k * (1 - cos((theta - theta0)/2))
//      This quasi-harmonic form is better than simple cos(theta) because
//      it remains harmonic for angles > 90 degrees.
//      Reference: see evalAngleCosHalf() comments below.
//
//   3. Pi-pi alignment (conjugation):
//      V_pipi = -K * cos(angle between pi-orbitals)
//      Favors parallel pi-orbital orientation on adjacent atoms,
//      modeling pi-conjugation in molecules with double bonds.
//
//   4. Pi-sigma orthogonalization (planarity):
//      V_pisigma = K * (cos(angle between pi-orbital and sigma-bond) - c0)^2
//      Keeps pi-orbitals perpendicular to the sigma-bond plane,
//      enforcing molecular planarity at sp2 centers.
//
//   5. Non-bonded (LJ + Coulomb, in nonbonded.cl):
//      V_LJ = 4*eps*((sigma/r)^12 - (sigma/r)^6)
//      V_Coul = q_i*q_j / (4*pi*eps0*r)
//      Excluded for 1-2 and 1-3 pairs (bonded neighbors).
//
// --- Pi-Orbital Degrees of Freedom ---
//
// Each node atom has an associated pi-orbital vector (stored as an extra
// float4 in the apos array at index iav+nAtoms). This vector:
//   - Is normalized to unit length after each integration step
//   - Experiences forces from pi-pi and pi-sigma interactions
//   - In the _rot variant, is integrated as a rotational DOF (torque ->
//     angular velocity -> rotation via Rodrigues' formula)
//   - In the standard variant, is integrated as a linear vector with
//     radial component projected out (tangential motion only)
//
// --- Recoil Force Mechanism ---
//
// When getSPFFf4 computes forces on a node atom, it also generates
// "recoil" forces on its neighbors (Newton's 3rd law). These are stored
// in the fneigh array (indexed by node * 4 + neighbor_slot). During
// integration, updateAtomsSPFFf4 gathers these recoil forces from
// back-neighbors (bkNeighs) and adds them to the atom's total force.
//
// This two-pass approach (scatter recoil -> gather recoil) avoids
// race conditions: multiple threads writing to the same atom's force
// would require atomic operations.
//
// --- Execution Flow (one MD step) ---
//
//   1. getSPFFf4 — evaluate bonding forces (bonds, angles, pi-pi, pi-sigma)
//      Stores: atom forces in fapos, recoil forces in fneigh
//   2. getNonBond_ex2 (in nonbonded.cl) — evaluate non-bonded LJ/Coulomb
//      Adds to fapos (accumulates with bonding forces)
//   3. cleanForceSPFFf4 — zero/clear force buffers for next step
//   4. updateAtomsSPFFf4 — gather recoil forces, integrate positions/velocities,
//      normalize pi-orbitals, apply constraints and bounding boxes
//
//   Alternative: relax_nsteps_serial — runs all steps in local memory
//   within a single workgroup, eliminating Python dispatch overhead.
//
// --- Key Caveats ---
//
//   CAVEAT 1: The bond evaluation uses `if(iG < ing)` to avoid double-counting
//   bonds (each bond is evaluated once, by the lower-indexed atom). This
//   means the recoil force on the higher-indexed neighbor is stored in
//   fneigh, not applied directly. The gather step in updateAtomsSPFFf4
//   collects these recoil forces.
//
//   CAVEAT 2: Pi-orbital normalization after each step means the pi-DOF
//   lives on a sphere (S^2), not in R^3. The standard integrator projects
//   out the radial component of force and velocity, keeping motion tangent
//   to the sphere. The _rot variant uses proper rotational dynamics.
//
//   CAVEAT 3: Force limiting (Flimit) scales down any force exceeding the
//   threshold. This prevents numerical instabilities from close contacts
//   but introduces energy drift (non-conservative). Use sparingly.
//
//   CAVEAT 4: The bSubtractVdW flag subtracts non-bonded LJ/Coulomb
//   interactions between 1-3 pairs (atoms sharing a common neighbor).
//   This is needed because the non-bonded kernel does not exclude 1-3
//   pairs by default. The subtraction is done inside the angle loop.
//
// Kernels:
//   - getSPFFf4: Bonding forces (bonds, angles, pi-pi, pi-sigma). 1 thread = 1 node.
//   - getSPFFf4_rot: Same but with torque-based pi interactions for rotational dynamics.
//   - updateGroups: Compute group CoM, forward, and up vectors for rigid groups.
//   - groupForce: Distribute external group forces/torques back to individual atoms.
//   - updateAtomsSPFFf4: MD integrator (leap-frog with damping) + recoil gather.
//   - updateAtomsSPFFf4_rot: MD integrator with rotational pi-orbital dynamics.
//   - cleanForceSPFFf4: Zero force arrays between MD steps.
//   - relax_nsteps_serial: Single-workgroup in-local-memory relaxation loop.
//
// Helper functions: evalAngCos, evalAngleCosHalf (angular force/energy),
// evalPiAling (pi-pi alignment), evalBond (harmonic bond stretching),
// evalPiSigma_tq, evalPiAlign_tq (torque-based pi variants),
// sinc_div_r2_taylor, rotate_by_omega_taylor (rotational helpers),
// KvaziFIREdamp (FIRE damping), hash_wang/hashf_wang (RNG for thermal noise).
// Requires: common.cl + Forces.cl to be concatenated before this file.

// ==================================================================
//  SPFF Bonding Helper Functions
// ==================================================================
//
//  These inline functions compute energy and force for individual
//  interaction terms. All return energy E and write forces via pointers.
//  The hr vectors are (direction, inverse_length) — direction is normalized,
//  hr.w = 1/|r| is used to scale forces to physical units.
//

// Angular force using cos(theta) formulation.
//   V = K * (cos(theta) - c0)^2
//   F1 = -2K*(cos(theta)-c0) * (hr2 - hr1*cos(theta)) / |hr1|
//   F2 = -2K*(cos(theta)-c0) * (hr1 - hr2*cos(theta)) / |hr2|
//
// CAVEAT: This formulation is NOT quasi-harmonic for angles > 90 deg.
// The force vanishes at theta=180 deg (cos=-1) even if c0 != -1.
// For sp3 angles near 109.5 deg this is fine, but for angles near 180 deg
// use evalAngleCosHalf instead.
inline float evalAngCos( const float4 hr1, const float4 hr2, float K, float c0, __private float3* f1, __private float3* f2 ){
    float  c = dot(hr1.xyz,hr2.xyz);
    float3 hf1,hf2;
    hf1 = hr2.xyz - hr1.xyz*c;
    hf2 = hr1.xyz - hr2.xyz*c;
    float c_   = c-c0;
    float E    = K*c_*c_;
    float fang = -K*c_*2;
    hf1 *= fang*hr1.w;
    hf2 *= fang*hr2.w;
    *f1=hf1;
    *f2=hf2;
    return E;
}

// Angular force using cos(theta/2) formulation — quasi-harmonic.
//
//   V = k * (1 - cos((theta - theta0)/2))
//
// This is MUCH better than evalAngCos for angles > 90 deg because it
// remains harmonic (restoring force increases monotonically) up to 180 deg.
// The cost is 2 extra sqrt() calls.
//
// Math:
//   cos(theta/2) = |hr1 + hr2| / 2   (for normalized hr1, hr2)
//   sin(theta/2) = sqrt(1 - cos^2(theta/2))
//   The force is derived by differentiating V w.r.t. atom positions.
//
// cs0 = (cos(theta0/2), sin(theta0/2)) — equilibrium half-angle
// k = angular stiffness
//
// CAVEAT: The 1e-7 regularizer in s2 = 1-c2+1e-7 prevents NaN from
// sqrt(0) when theta=0 (collapsed angle). This introduces a tiny bias.
inline float evalAngleCosHalf( const float4 hr1, const float4 hr2, const float2 cs0, float k, __private float3* f1, __private float3* f2 ){
    // This is much better angular function than evalAngleCos() with just a little higher computational cost ( 2x sqrt )
    // the main advantage is that it is quasi-harmonic beyond angles > 90 deg
    float3 h  = hr1.xyz + hr2.xyz;  // h = a+b
    float  c2 = dot(h,h)*0.25f;     // cos(a/2) = |ha+hb|  (after normalization)
    float  s2 = 1.f-c2 + 1e-7;      // sin(a/2) = sqrt(1-cos(a/2)^2) ;  s^2 must be positive (otherwise we get NaNs)
    float2 cso = (float2){ sqrt(c2), sqrt(s2) }; // cso = cos(a/2) + i*sin(a/2)
    float2 cs = udiv_cmplx( cs0, cso );          // rotate back by equilibrium angle
    float  E         =  k*( 1 - cs.x );          // E = k*( 1 - cos(a/2) )  ; Do we need Energy? Just for debugging ?
    float  fr        = -k*(     cs.y );          // fr = k*( sin(a/2) )     ; force magnitude
    c2 *= -2.f;
    fr /=  4.f*cso.x*cso.y;   //    |h - 2*c2*a| =  1/(2*s*c) = 1/sin(a)
    float  fr1    = fr*hr1.w; // magnitude of force on atom a
    float  fr2    = fr*hr2.w; // magnitude of force on atom b
    *f1 =  h*fr1  + hr1.xyz*(fr1*c2);  //fa = (h - 2*c2*a)*fr / ( la* |h - 2*c2*a| ); force on atom a
    *f2 =  h*fr2  + hr2.xyz*(fr2*c2);  //fb = (h - 2*c2*b)*fr / ( lb* |h - 2*c2*b| ); force on atom b
    return E;
}

// Pi-pi alignment (conjugation) interaction.
//   V = -K * cos(angle between pi-orbitals h1, h2)
//   F1 = K * (h2 - h1*cos)   [perpendicular component of h2]
//   F2 = K * (h1 - h2*cos)   [perpendicular component of h1]
// If angle > 90 deg (cos < 0), force sign is flipped to maintain restoring force.
// h1, h2 must be normalized.
inline float evalPiAling( const float3 h1, const float3 h2,  float K, __private float3* f1, __private float3* f2 ){
    float  c = dot(h1,h2); // cos(a) (assumes that h1 and h2 are normalized)
    float3 hf1,hf2;        // working forces or direction vectors
    hf1 = h2 - h1*c;       // component of h2 perpendicular to h1
    hf2 = h1 - h2*c;       // component of h1 perpendicular to h2
    bool sign = c<0; if(sign) c=-c; // if angle is > 90 deg we need to flip the sign of force
    float E    = -K*c;     // energy is -K*cos(a)
    float fang =  K;       // force magnitude
    if(sign)fang=-fang;    // flip the sign of force if angle is > 90 deg
    hf1 *= fang;           // force on atom a
    hf2 *= fang;           // force on atom b
    *f1=hf1;
    *f2=hf2;
    return E;
}

// Harmonic bond stretching.
//   V = (1/2) * k * dl^2    where dl = (r - r0)
//   F = -k * dl * h_hat     (force along bond direction)
// h = normalized bond direction, dl = length deviation, k = stiffness.
inline float evalBond( float3 h, float dl, float k, __private float3* f ){
    float fr = dl*k;   // force magnitude
    *f = h * fr;       // force on atom a
    return fr*dl*0.5;  // energy
}

// ==================================================================
//  Torque-Based Pi Interaction Helpers (for rotational pi dynamics)
// ==================================================================
//
//  These variants return torques instead of linear forces, for use with
//  the _rot kernels that integrate pi-orbitals as rotational DOFs.
//  The torque is computed as tau = hpi x F_linear, which is the correct
//  transformation from linear force on a unit vector to torque on that vector.
//

// Pi-sigma orthogonalization (torque variant):
//   V = K * (cos(angle) - c0)^2
//   tau_pi = cross(hpi, h) * (-2K*(cos-c0))   [torque on pi-orbital]
//   F_recoil = (hpi - h*cos) * (-2K*(cos-c0)) * h.w  [recoil on neighbor]
inline float evalPiSigma_tq(const float3 hpi, const float4 h, const float K, const float c0, __private float3 *tqi, __private float3 *fj){
    const float c    = dot(hpi, h.xyz);
    const float c_   = c - c0;
    const float E    = K * c_ * c_;
    const float fang = -2.0f * K * c_;
    *tqi = cross(hpi, h.xyz) * fang;          // torque on pi
    const float s2   = fang * h.w;            // recoil scaling
    *fj  = (hpi - h.xyz * c) * s2;            // recoil force perpendicular to bond
    return E;
}

// Pi-pi alignment (torque variant):
//   V = -K * cos(angle)
//   tau = cross(h1, h2) * K   [torque on pi-orbital 1]
// Returns (tau_x, tau_y, tau_z, E) as float4.
inline float4 evalPiAlign_tq(const float3 h1, const float3 h2, const float K){
    const float c = dot(h1,h2);
    const float E = -K * c;
    return (float4){ cross(h1, h2) * K, E };
}

// ==================================================================
//  Rotation Helpers for Pi-Orbital Rotational Dynamics
// ==================================================================
//
//  These functions implement small-angle rotation of a unit vector p
//  by a rotation vector w (where |w| = rotation angle, w_hat = axis).
//  Uses Rodrigues' formula with Taylor series for sinc/sinc-half:
//    p' = p + (w x p) * sinc(|w|) + (w x (w x p)) * (1-cos(|w|))/|w|^2
//  where sinc(x) = sin(x)/x and (1-cos(x))/x^2 = (1-cos(x))/x^2.
//
//  The Taylor series avoids division by zero for small |w| and is
//  accurate to ~1e-7 for |w| < 0.1 rad (one MD step with small dt).
//
//  Reference: Rodrigues' rotation formula.
//  Also used in rigid.cl for quaternion-based rotation.
//

// Taylor series for sin(r)/r and (1-cos(r))/r^2, accurate for small r.
//   sin(r)/r     = 1 - r^2/6 + r^4/120 - r^6/5040 + ...
//   (1-cos(r))/r^2 = 1/2 - r^2/24 + r^4/720 - r^6/40320 + ...
float2 sinc_div_r2_taylor(float r2){
    float s = 1.0f + r2 * ( (-1.0f/6.0f)  + r2 * ( (1.0f/120.0f) + r2 * (-1.0f/5040.0f  ) ) );
    float c = 0.5f + r2 * ( (-1.0f/24.0f) + r2 * ( (1.0f/720.0f) + r2 * (-1.0f/40320.0f ) ) );
    return (float2){s, c};
}

// Rotate unit vector p by rotation vector w using Rodrigues' formula
// with Taylor series for sinc terms. Accurate for small |w| (< 0.1 rad).
//   p' = p + (w x p)*sinc(|w|) + (w x (w x p))*(1-cos(|w|))/|w|^2
float3 rotate_by_omega_taylor(float3 p, float3 w){
    float r2    = dot(w,w);
    float2 sc   = sinc_div_r2_taylor(r2);
    float3 wxp  = cross(w, p);
    float3 wwxp = cross(w, wxp);
    return p + wxp*sc.x + wwxp*sc.y;
}

// ======================================================================
//                          getSPFFf4()
// ======================================================================
//
//  Computes all bonding interactions for one node atom per thread.
//  GPU parallelization: dim 0 = atom index, dim 1 = system index.
//
//  Per node atom (up to 4 neighbors), evaluates:
//    1. Bond stretching (harmonic) — each bond once (iG < ing guard)
//    2. Pi-pi alignment (conjugation) — only between node atoms
//    3. Pi-sigma orthogonalization (planarity) — pi vs each sigma bond
//    4. Angle bending (cos-half) — all unique neighbor pairs (i < j)
//    5. Optional: subtract non-bonded VdW for 1-3 pairs (bSubtractVdW)
//
//  Force storage:
//    fapos[iav]       += (F_center, E)   — force + energy on center atom
//    fapos[iav+nAtoms] = (F_pi, 0)       — force on pi-orbital
//    fneigh[i4+i]      = (F_recoil_i, 0) — recoil force on i-th neighbor
//    fneigh[i4p+i]     = (F_pi_recoil_i, 0) — pi recoil on i-th neighbor's pi
//
//  where i4  = (iG + iS*nnode*2)*4         (sigma recoil slot)
//        i4p = i4 + nnode*4                (pi recoil slot)
//
//  CAVEAT: The `if(iG < ing)` guard ensures each bond is evaluated once.
//  The neighbor with higher index receives its force via recoil, not directly.
//
//  CAVEAT: PBC shifts are applied to bond vectors via neighCell indices
//  into the pbc_shifts array. This handles bonds across cell boundaries.
//
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void getSPFFf4(
    const int4 nDOFs,               // 1   (nAtoms,nnode) dimensions of the system
    // Dynamical
    __global float4*  apos,         // 2  [natoms]     positions of atoms (including node atoms [0:nnode] and capping atoms [nnode:natoms] and pi-orbitals [natoms:natoms+nnode] )
    __global float4*  fapos,        // 3  [natoms]     forces on    atoms (just node atoms are evaluated)
    __global float4*  fneigh,       // 4  [nnode*4*2]  recoil forces on neighbors (and pi-orbitals)
    // parameters
    __global int4*    neighs,       // 5  [nnode]  neighboring atoms
    __global int4*    neighCell,    // 5  [nnode]  neighboring atom  cell index
    __global float4*  REQKs,        // 6  [natoms] non-boding parametes {R0,E0,Q} i.e. R0: van der Waals radii, E0: well depth and partial charge, Q: partial charge
    __global float4*  apars,        // 7  [nnode]  per atom forcefield parametrs {c0ss,Kss,c0sp}, i.e. c0ss: cos(equlibrium angle/2) for sigma-sigma; Kss: stiffness of sigma-sigma angle; c0sp: is cos(equlibrium angle) for sigma-pi
    __global float4*  bLs,          // 8  [nnode]  bond length    between node and each neighbor
    __global float4*  bKs,          // 9  [nnode]  bond stiffness between node and each neighbor
    __global float4*  Ksp,          // 10 [nnode]  stiffness of pi-alignment for each neighbor     (only node atoms have pi-pi alignemnt interaction)
    __global float4*  Kpp,          // 11 [nnode]  stiffness of pi-planarization for each neighbor (only node atoms have pi-pi alignemnt interaction)
    __global cl_Mat3* lvecs,        // 12 lattice vectors         for each system
    __global cl_Mat3* ilvecs,       // 13 inverse lattice vectors for each system
    __global float4*  pbc_shifts,   // 14 pbc shifts for each system
    const int npbc,                 // 15 number of pbc shifts
    const int bSubtractVdW          // 16 subtract vdW energy
){

    const int iG = get_global_id (0);   // intex of atom   (iG<nAtoms)
    const int iS = get_global_id (1);   // index of system (iS<nS)
    //const int nG = get_global_size(0);
    //const int nS = get_global_size(1);  // number of systems
    //const int iL = get_local_id  (0);
    //const int nL = get_local_size(0);
    const int nAtoms=nDOFs.x;  // number of atoms in the system
    const int nnode =nDOFs.y;  // number of nodes in the system
    //const int nvec  = nAtoms+nnode;

    if(iG>=nnode) return;

    const int i0a   = iS*nAtoms;         // index of first atom      in the system
    const int i0n   = iS*nnode;          // index of first node atom in the system
    const int i0v   = iS*(nAtoms+nnode); // index of first vector    in the system ( either atom or pi-orbital )

    const int iaa = iG + i0a;  // index of current atom (either node or capping atom)
    const int ian = iG + i0n;  // index of current node atom
    const int iav = iG + i0v;  // index of current vector ( either atom or pi-orbital )

    #define NNEIGH 4

    // ---- Dynamical
    float4  hs [4];              // direction vectors of bonds (h.xyz) and inverse bond lengths (h.w)
    float3  fbs[4];              // force on neighbor sigma    (fbs[i] is sigma recoil force on i-th neighbor)
    float3  fps[4];              // force on neighbor pi       (fps[i] is pi    recoil force on i-th neighbor)
    float3  fa  = float3Zero;    // force on center atom positon

    float E=0;                   // Total Energy of this atom
    // ---- Params
    const int4   ng  = neighs[iaa];    // neighboring atoms
    const float3 pa  = apos[iav].xyz;  // position of current atom
    const float4 par = apars[ian];     // (xy=s0_ss,z=ssK,w=piC0 ) forcefield parameters for current atom


    // Temp Arrays
    const int*   ings  = (int*  )&ng; // neighboring atoms, we cast it to int[] to be index it in for loop


    const float   ssC0   = par.x*par.x - par.y*par.y;                      // cos(2) = cos(x)^2 - sin(x)^2, because we store cos(ang0/2) to use in  evalAngleCosHalf , where ang0 is equilibrium angle
    for(int i=0; i<NNEIGH; i++){ fbs[i]=float3Zero; fps[i]=float3Zero; }   // clear recoil forces on neighbors

    float3 f1,f2;         // working forces

    #if DBG_UFF
    if((iG==0)&&(iS==0)){
        printf( "getSPFFf4() iG %i, iS %i, iaa %i bSubtractVdW %i\n", iG, iS, iaa, bSubtractVdW );
    }
    #endif

    { // ========= BONDS - here we evaluate pairwise interactions of node atoms with its 4 neighbors

        float3  fpi = float3Zero;                // force on pi-orbital
        const int4   ngC = neighCell[iaa];       // neighboring atom cell index
        const float3 hpi = apos[iav+nAtoms].xyz; // direction of pi-orbital
        const float4 vbL = bLs[ian];             // bond lengths
        const float4 vbK = bKs[ian];             // bond stiffness
        const float4 vKs = Ksp[ian];             // stiffness of sigma-pi othogonalization
        const float4 vKp = Kpp[ian];             // stiffness of pi-pi    alignment

        const int*   ingC  = (int*  )&ngC;   // neighboring atom cell index (we cast it to int[] to be index it in for loop)
        const float* bL    = (float*)&vbL;   // bond lengths
        const float* bK    = (float*)&vbK;   // bond stiffness
        const float* Kspi  = (float*)&vKs;   // stiffness of sigma-pi othogonalization
        const float* Kppi  = (float*)&vKp;   // stiffness of pi-pi    alignment

        const int ipbc0 = iS*npbc;  // index of first PBC shift for current system

        for(int i=0; i<NNEIGH; i++){  // loop over 4 neighbors
            float4 h;                 // direction vector of bond
            const int ing  = ings[i]; // index of i-th neighbor node atom
            const int ingv = ing+i0v; // index of i-th neighbor vector
            const int inga = ing+i0a; // index of i-th neighbor atom
            if(ing<0) break;

            // --- Compute bond direction vector and inverse bond length
            h.xyz    = apos[ingv].xyz - pa;  // direction vector of bond
            { // shift bond to the proper PBC cell
                int ic  = ingC[i];                  // index of i-th neighbor cell
                h.xyz  += pbc_shifts[ipbc0+ic].xyz; // shift bond to the proper PBC cell
            }
            float  l = length(h.xyz);  // compute bond length
            h.w      = 1./l;           // store ivnerse bond length
            h.xyz   *= h.w;            // normalize bond direction vector
            hs[i]    = h;              // store bond direction vector and inverse bond length

            float epp = 0; // pi-pi    energy
            float esp = 0; // pi-sigma energy

            // --- Evaluate bond-length stretching energy and forces
            if(iG<ing){
                E+= evalBond( h.xyz, l-bL[i], bK[i], &f1 );  fbs[i]-=f1;  fa+=f1;   // harmonic bond stretching, fa is force on center atom, fbs[i] is recoil force on i-th neighbor,

                // pi-pi alignment interaction
                float kpp = Kppi[i];
                if( (ing<nnode) && (kpp>1.e-6) ){   // Only node atoms have pi-pi alignemnt interaction
                    epp += evalPiAling( hpi, apos[ingv+nAtoms].xyz, kpp,  &f1, &f2 );   fpi+=f1;  fps[i]+=f2;    //   pi-alignment(konjugation), fpi is force on pi-orbital, fps[i] is recoil force on i-th neighbor's pi-orbital
                    E+=epp;
                }
            }

            // pi-sigma othogonalization interaction
            float ksp = Kspi[i];
            if(ksp>1.e-6){
                esp += evalAngCos( (float4){hpi,1.}, h, ksp, par.w, &f1, &f2 );   fpi+=f1; fa-=f2;  fbs[i]+=f2;    //   pi-planarization (orthogonality), fpi is force on pi-orbital, fbs[i] is recoil force on i-th neighbor
                E+=esp;
            }
        }

        // --- Store Pi-forces                      we store pi-forces here because we don't use them in the angular force evaluation
        const int i4p=(iG + iS*nnode*2 )*4 + nnode*4; // index of first pi-force for current atom
        for(int i=0; i<NNEIGH; i++){
            fneigh[i4p+i] = (float4){fps[i],0}; // store recoil pi-force on i-th neighbor
        }
        fapos[iav+nAtoms]  = (float4){fpi,0};  // store pi-force on pi-orbital of current atom

    }

    { //  ============== Angles   - here we evaluate angular interactions between pair of sigma-bonds of node atoms with its 4 neighbors

        for(int i=0; i<NNEIGH; i++){ // loop over first bond
            int ing = ings[i];
            if(ing<0) break;         // if there is no i-th neighbor we break the loop
            const float4 hi = hs[i];
            const int ingv = ing+i0v;
            const int inga = ing+i0a;
            for(int j=i+1; j<NNEIGH; j++){ // loop over second bond
                int jng  = ings[j];
                if(jng<0) break;           // if there is no j-th neighbor we break the loop
                const int jngv = jng+i0v;
                const int jnga = jng+i0a;
                const float4 hj = hs[j];

                E += evalAngleCosHalf( hi, hj, par.xy, par.z, &f1, &f2 );    // evaluate angular force and energy using cos(angle/2) formulation
                fa    -= f1+f2;

                if(bSubtractVdW)
                { // Remove non-bonded interactions from atoms that are bonded to common neighbor
                    float4 REQi=REQKs[inga];   // non-bonding parameters of i-th neighbor
                    float4 REQj=REQKs[jnga];   // non-bonding parameters of j-th neighbor
                    // combine non-bonding parameters of i-th and j-th neighbors using mixing rules
                    float4 REQij;
                    REQij.x  = REQi.x  + REQj.x;
                    REQij.yz = REQi.yz * REQj.yz;

                    float3 dp = (hj.xyz/hj.w) - (hi.xyz/hi.w);   // recover vector between i-th and j-th neighbors using stored vectos and inverse bond lengths, this should be faster than dp=apos[jngv].xyz-apos[ingv].xyz; from global memory
                    float4 fij = getLJQH( dp, REQij, 1.0f );     // compute non-bonded interaction between i-th and j-th neighbors using Lennard-Jones and Coulomb interactions and Hydrogen bond correction
                    f1 -=  fij.xyz;
                    f2 +=  fij.xyz;
                }

                fbs[i]+= f1;
                fbs[j]+= f2;
            }
        }

    }

    // ========= Save results - store forces on atoms and recoil on its neighbors  (pi-forces are already done)
    const int i4 =(iG + iS*nnode*2 )*4;
    //const int i4p=i4+nnode*4;
    for(int i=0; i<NNEIGH; i++){
        fneigh[i4 +i] = (float4){fbs[i],0};
        //fneigh[i4p+i] = (float4){fps[i],0};
    }
    //fapos[iav     ] = (float4){fa ,0}; // If we do  run it as first forcefield
    fapos[iav       ] += (float4){fa ,E};  // If we not run it as first forcefield, store energy in .w
    //fapos[iav+nAtoms]  = (float4){fpi,0};

}

// ======================================================================
//                          getSPFFf4_rot()
// ======================================================================
//
//  Torque-based variant of getSPFFf4 for rotational pi-orbital dynamics.
//  Instead of linear forces on pi-orbitals, computes torques:
//    - Pi-pi alignment:  tau = cross(h1, h2) * K
//    - Pi-sigma ortho:   tau = cross(hpi, h) * (-2K*(cos-c0))
//  Torques are stored in aforce[iav+nAtoms] and integrated by
//  updateAtomsSPFFf4_rot using Rodrigues' rotation formula.
//
//  The sigma (bond + angle) forces are identical to getSPFFf4.
//  Only the pi interaction helpers differ (evalPiAlign_tq, evalPiSigma_tq).
//
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void getSPFFf4_rot(
    const int4 nDOFs,               // 1   (nAtoms,nnode) dimensions of the system
    __global float4*  apos,         // 2  [natoms]     positions of atoms
    __global float4*  aforce,       // 3  [natoms]     forces on atoms
    __global float4*  fneigh,       // 4  [nnode*4*2]  recoil forces on neighbors
    __global int4*    neighs,       // 5  [nnode]  neighboring atoms
    __global int4*    neighCell,    // 5  [nnode]  neighboring atom cell index
    __global float4*  REQs,         // 6  [natoms] non-bonding parameters
    __global float4*  apars,        // 7  [nnode]  per atom forcefield parameters
    __global float4*  bLs,          // 8  [nnode]  bond lengths
    __global float4*  bKs,          // 9  [nnode]  bond stiffness
    __global float4*  Ksp,          // 10 [nnode]  stiffness of pi-sigma orthogonalization
    __global float4*  Kpp,          // 11 [nnode]  stiffness of pi-pi alignment
    __global cl_Mat3* lvecs,        // 12 lattice vectors
    __global cl_Mat3* ilvecs,       // 13 inverse lattice vectors
    __global float4*  pbc_shifts,
    const int npbc,
    const int bSubtractVdW
){
    const int iG = get_global_id (0);
    const int iS = get_global_id (1);
    const int nAtoms=nDOFs.x;
    const int nnode =nDOFs.y;
    if(iG>=nnode) return;

    const int i0a   = iS*nAtoms;
    const int i0n   = iS*nnode;
    const int i0v   = iS*(nAtoms+nnode);
    const int iaa = iG + i0a;
    const int ian = iG + i0n;
    const int iav = iG + i0v;

    #define NNEIGH 4
    float4  hs [4];
    float3  fbs[4];
    float3  fa  = float3Zero;
    float E=0;
    const int4   ng  = neighs[iaa];
    const float3 pa  = apos[iav].xyz;
    const float4 par = apars[ian];
    const int*   ings  = (int*)&ng;
    const float   ssC0   = par.x*par.x - par.y*par.y;
    for(int i=0; i<NNEIGH; i++){ fbs[i]=float3Zero; }
    float3 f1,f2;

    {
        float3  fpi = float3Zero;
        const int4   ngC = neighCell[iaa];
        const float3 hpi = apos[iav+nAtoms].xyz;
        const float4 vbL = bLs[ian];
        const float4 vbK = bKs[ian];
        const float4 vKs = Ksp[ian];
        const float4 vKp = Kpp[ian];
        const int*   ingC  = (int*)&ngC;
        const float* bL    = (float*)&vbL;
        const float* bK    = (float*)&vbK;
        const float* Kspi  = (float*)&vKs;
        const float* Kppi  = (float*)&vKp;
        const int ipbc0 = iS*npbc;

        for(int i=0; i<NNEIGH; i++){
            float4 h;
            const int ing  = ings[i];
            const int ingv = ing+i0v;
            if(ing<0) break;
            h.xyz    = apos[ingv].xyz - pa;
            { int ic = ingC[i]; h.xyz += pbc_shifts[ipbc0+ic].xyz; }
            float  l = length(h.xyz);
            h.w      = 1.f/l;
            h.xyz   *= h.w;
            hs[i]    = h;

            if(iG<ing){
                float elb = evalBond( h.xyz, l-bL[i], bK[i], &f1 );  fbs[i]-=f1;  fa+=f1; E+=elb;
            }

            float kpp = Kppi[i];
            if( (ing<nnode) && (kpp>1.e-6f) ){
                float3 hpj = apos[ingv+nAtoms].xyz;
                float4 fepi = evalPiAlign_tq( hpi, hpj, kpp );
                E  += fepi.w;
                fpi += fepi.xyz;
            }

            float ksp = Kspi[i];
            if(ksp>1.e-6f){
                float esp = evalPiSigma_tq( hpi, h, ksp, par.w, &f1, &f2 );
                E  += esp; fa-=f2;  fbs[i]+=f2; fpi+=f1;
            }
        }

        aforce[iav+nAtoms]  = (float4){fpi,0};
    }

    {
        for(int i=0; i<NNEIGH; i++){
            int ing = ings[i];
            if(ing<0) break;
            const float4 hi = hs[i];
            for(int j=i+1; j<NNEIGH; j++){
                int jng  = ings[j];
                if(jng<0) break;
                const float4 hj = hs[j];
                float ea = evalAngleCosHalf( hi, hj, par.xy, par.z, &f1, &f2 );
                fa  -= f1+f2;
                E   += ea;
                fbs[i]+= f1;
                fbs[j]+= f2;
            }
        }
    }

    const int i4 =(iG + iS*nnode*2 )*4;
    for(int i=0; i<NNEIGH; i++){
        fneigh[i4 +i] = (float4){fbs[i],0};
    }
    aforce[iav ] += (float4){fa.x,fa.y,fa.z,E};
}

// ======================================================================
//                     updateGroups()
// ======================================================================
//
//  Computes geometric properties of rigid atom groups for constrained dynamics.
//  One thread per group. For each group:
//    1. Center of geometry (CoG) = weighted average of member positions
//    2. Forward vector (fw) = weighted principal direction (1st eigenvector)
//    3. Up vector (up) = weighted secondary direction, orthogonalized to fw
//
//  The fw/up vectors define a local coordinate frame for the group, used
//  by groupForce to decompose applied torques into Cartesian components.
//
//  Weights: gweights[ia] = (w_com, w_fw, w_up, 0) — separate weights for
//  CoM, forward, and up computation. This allows e.g. using only terminal
//  atoms for direction vectors while all atoms contribute to CoM.
//
//  Orthonormalization: Gram-Schmidt — fw normalized, then up projected
//  perpendicular to fw and normalized.
//
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void updateGroups(
    int               ngroup,      // 1 // number of groups (total, for all systems)
    __global int2*    granges,     // 2 // (i0,n) range of indexes specifying the group
    __global int*     g2a,         // 3 // indexes of atoms corresponding to groups defined by granges
    __global float4*  apos,        // 4 // positions of atoms  (including node atoms [0:nnode] and capping atoms [nnode:natoms] and pi-orbitals [natoms:natoms+nnode] )
    __global float4*  gcenters,    // 5 // centers of each groups (CoGs)
    __global float4*  gfws,        // 6 // forwad  orietantian vector for each group
    __global float4*  gups,        // 7 // up      orietantian vector for each group
    __global float4*  gweights     // 8 // up      orietantian vector for each group
){
    const int iG = get_global_id  (0); // index of atom
    if(iG>=ngroup) return; // make sure we are not out of bounds of current system

    // if(iG==0){
    //     printf( "GPU ngroup=%i \n", ngroup );
    //     for(int i=0; i<ngroup; i++){
    //         const int2 grange = granges[i];
    //         printf("GPU granges[%i] i0=%i n=%i \n", i, grange.x, grange.y  );
    //         for(int j=0; j<grange.y; j++){
    //             int ia = g2a[ grange.x + j ];
    //             //printf( "[%i] %i \n", j, ia );
    //             printf( "GPU gweights[%i](%g,%g,%g,%g)\n", ia, gweights[ia].x,gweights[ia].y,gweights[ia].z,gweights[ia].w );
    //         }
    //         printf("\n");
    //     }
    // }

    const int2 grange = granges[iG];

    float3 cog = (float3){0.0f,0.0f,0.0f};

    float wsum = 0.f;
    for(int i=0; i<grange.y; i++){
        int ia = g2a[ grange.x + i ];
        //const float4 pe = apos[ia];
        const float4 w = gweights[ia];
        cog    += apos[ia].xyz * w.x;
        wsum   += w.x;
    }
    cog *= ( 1.f/wsum );
    gcenters[iG] = (float4){cog,0.0f};

    float3 up  = (float3){0.0f,0.0f,0.0f};
    float3 fw  = (float3){0.0f,0.0f,0.0f};
    for(int i=0; i<grange.y; i++){
        int ia = g2a[ grange.x + i ];
        //const float4 pe = apos[ia];
        const float4 w = gweights[ia];
        const float3 d = apos[ia].xyz - cog.xyz;
        fw.xyz += d * w.y;
        up.xyz += d * w.z;
    }
    {  // Orthonormalize
        fw  = normalize( fw );
        up += fw * -dot( fw, up );
        up  = normalize( up );
    }

    //printf( "GPU[iG=%i] cog(%g,%g,%g) fw(%g,%g,%g) up(%g,%g,%g) \n", iG, cog.x,cog.y,cog.z,   fw.x,fw.y,fw.z,  up.x,up.y,up.z );
    gfws[iG] = (float4){fw,0.0f};
    gups[iG] = (float4){up,0.0f};
}

// ======================================================================
//                     groupForce()
// ======================================================================
//
//  Distributes external forces and torques applied to rigid groups back
//  to individual member atoms. One thread per atom.
//
//  For each atom belonging to a group:
//    F_atom += F_group * w_linear + cross(r_atom - r_group, tau_group) * w_linear
//
//  where tau_group is decomposed into the group's local frame (fw, up, lf):
//    tau_cartesian = fw * tau.x + up * tau.y + lf * tau.z
//  and lf = normalize(cross(fw, up)) is the left/right direction.
//
//  This is the rigid-body force distribution: a force on the group center
//  translates all atoms equally, while a torque rotates the group and
//  produces position-dependent forces on each atom.
//
//  CAVEAT: gfweights stores only a single weight (w.x) used for both
//  linear and torque contributions. If different weighting is needed,
//  the kernel must be modified.
//
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void groupForce(
    const int4        n,            // 1 // (natoms,nnode) dimensions of the system
    __global float4*  apos,         // 2 // positions of atoms  (including node atoms [0:nnode] and capping atoms [nnode:natoms] and pi-orbitals [natoms:natoms+nnode] )
    __global float4*  aforce,       // 3 // forces on atoms
    __global int*     a2g,          // 4 // atom to group maping (index)
    __global float4*  gforces,      // 5 // linar forces appliaed to atoms of the group
    __global float4*  gtorqs,       // 6 // {hx,hy,hz,t} torques applied to atoms of the group
    __global float4*  gcenters,     // 7 // centers of rotation (for evaluation of the torque
    __global float4*  gfws,         // 8 // forward vector of group orientation
    __global float4*  gups,         // 9 // up      vector of group orientation
    __global float2*  gfweights    // 10 // weights for application of forces on atoms
){
    const int natoms = n.x;           // number of atoms
    const int nnode  = n.y;           // number of node atoms
    const int nGrpup = n.w;           // number of node atoms
    const int nvec   = natoms+nnode; // number of vectors (atoms+node atoms)
    const int iG = get_global_id  (0); // index of atom

    if(iG>=natoms) return; // make sure we are not out of bounds of current system

    const int iS = get_global_id  (1); // index of system
    const int nG = get_global_size(0); // number of atoms
    const int nS = get_global_size(1); // number of systems

    // if( (iG==0) && (iS==0) ){
    //     printf( "GPU::groupForce() natom=%i nnode=%i nvec=%i \n", natoms, nnode, nvec );
    // //     int ig_sel = 0;
    // //     int is = 0;
    // //     // for(int ia=0; ia<natoms; ia++){
    // //     //      int iav = ia + is*nvec;
    // //     //     printf( "%i ", a2g[iav] );
    // //     // }
    // //     // printf("\n");

    //     for(int is=0; is<nS; is++){
    //         // printf( "sys[%i] ", is );
    //         // for(int ia=0; ia<natoms; ia++){
    //         //     int iav = ia + is*nvec;
    //         //     printf( "%i ", a2g[iav] );
    //         // }
    //         // printf("\n");
    //         for(int ia=0; ia<natoms; ia++){
    //             int iav = ia + is*nvec;
    //             const int ig = a2g[iav];
    //             if(ig>=0){
    //                 //printf( "GPU:atom[%i|%i,%i] ig=%i(%i/%i) gforces(%10.6f,%10.6f,%10.6f)\n", is, ia, iav, ig, ig-is*nGrpup,nGrpup, gforces[ig].x, gforces[ig].y, gforces[ig].z  );
    //                 printf( "GPU:atom[isys=%i|ia=%i] gfweights[iav=%i](%10.6f,%10.6f) gtorqs[ig=%i](%10.6f,%10.6f,%10.6f,%10.6f)\n", is, ia,     iav,  gfweights[iav].x,gfweights[iav].y,    ig, gtorqs[ig].x, gtorqs[ig].y, gtorqs[ig].z, gtorqs[ig].w  );
    //             }
    //         }
    //     }
    // }

    //const int ian = iG + iS*nnode;
    const int iaa = iG + iS*natoms;  // index of atom in atoms array
    const int iav = iG + iS*nvec;    // index of atom in vectors array

    float4 fe    = aforce[iav]; // position of atom or pi-orbital
    const int ig = a2g[iav];  // index of the group to which this atom belongs

    float2  w = gfweights[ig];

    // --- apply linear forece from the group
    fe.xyz += gforces[ig].xyz * w.x;

    // ToDo: group vectors may be stored in Local Memory ?
    const float3 torq = gtorqs[ig].xyz;
    const float3 fw   = gfws  [ig].xyz;
    const float3 up   = gups  [ig].xyz;
    const float3 lf   = normalize( cross(fw,up) );
    const float3 tq   = fw * torq.x   +  up * torq.y    +   lf * torq.z;

    // --- apply torque from the group
    const float3 dp  = apos[iav].xyz - gcenters[ig].xyz;
    fe.xyz          += cross( dp, tq.xyz ) * w.x;

    // --- store results
    aforce[iav] = fe;

}

// ======================================================================
//                     updateAtomsSPFFf4()
// ======================================================================
//
//  MD integrator for SPFF with pi-orbital linear dynamics.
//  One thread per atom (or pi-orbital), one system per dim-1.
//
//  Steps per atom:
//    1. Gather recoil forces from back-neighbors (bkNeighs -> fneigh)
//    2. Apply force limiting (Flimit) if enabled
//    3. Apply constraints (harmonic springs to fixed positions)
//    4. Apply bounding box (z-direction only; x,y commented out)
//    5. Apply inter-system bonds (compression/tension between replicas)
//    6. For pi-orbitals: project out radial component of force & velocity
//    7. Integrate: v *= damp; v += F*dt/m; r += v*dt
//    8. For pi-orbitals: normalize to unit length
//
//  Integration scheme: Damped leap-frog (semi-implicit Euler):
//    v_new = damp * v_old + F * dt / m
//    r_new = r_old + v_new * dt
//  When damp=1.0, this is pure leap-frog (symplectic for conservative forces).
//  When damp<1.0, velocity is scaled each step (non-conservative, energy dissipating).
//
//  Pi-orbital dynamics (standard variant):
//    The pi-orbital is a unit vector. Forces are projected tangent to the
//    sphere: F_tangent = F - (F.p)p, same for velocity. After integration,
//    the vector is renormalized. This is a first-order approximation to
//    motion on S^2. For proper rotational dynamics, use updateAtomsSPFFf4_rot.
//
//  CAVEAT: Bounding box is only applied in z-direction (x,y lines commented
//  out at lines ~801-802). This is intentional for surface simulations where
//  atoms should be free laterally but confined vertically.
//
//  CAVEAT: Force limiting (Flimit) is non-conservative and can cause energy
//  drift. It should only be used during initial relaxation, not production.
//


/*
float2 KvaziFIREdamp( float c, float2 damp_lims, float2 clim ){
    float2 cvf;
    if      (c < clim.x ){   //-- force against veloctiy
        cvf.x = damp_lims.x; // v    // like 0.5 (strong damping)
        cvf.y = 0;           // f
    }else if(c > clim.y ){   //-- force alingned to velocity
        cvf.x = 1-damping;   // v    // like 0.99 (weak dampong damping)
        cvf.y =   damping;   // f
    }else{                   // -- force ~ perpendicular to velocity
        float f = (c-clim.x )/( clim.y - clim.x  );
        cvf.x = (1.-damping)*f;
        cvf.y =     damping *f;
    }
    return cvf;
}
*/

/*
def KvaziFIREdamp( c, clim, damps ):
    # ----- velocity & force ~ perpendicular
    t = (c-clim[0] )/( clim[1] - clim[0]  )
    cv = damps[0] + (damps[1]-damps[0])*t
    #cf =     damps[1] *t*(1-t)*4
    cf =     damps[1]*t*(1-t)*2
    # ----- velocity & force ~ against each other
    mask_lo     =  c < clim[0]
    cv[mask_lo] = damps[0]  # v    // like 0.5 (strong damping)
    cf[mask_lo] = 0             # f
    # ----- velocity & force ~ alligned
    mask_hi     =  c > clim[1]
    cv[mask_hi] = damps[1]  # v    // like 0.99 (weak dampong damping)
    cf[mask_hi] = 0           # f
    return cv,cf
*/


// FIRE (Fast Inertial Relaxation Engine) damping function.
// Returns (velocity_factor, force_factor) based on the alignment c = F.v / (|F||v|):
//   c < clim.x: F and v anti-aligned -> strong damping (damps.x), no force boost
//   c > clim.y: F and v aligned -> weak damping (damps.y), no force boost
//   clim.x <= c <= clim.y: intermediate -> linear interpolation + parabolic boost
//
// Reference: Bitzek et al., PRL 97, 170201 (2006).
// CAVEAT: This is a modified FIRE for GPU parallelization — the force boost
// term (cvf.y) is applied per-atom rather than globally, which differs from
// the original algorithm. This may affect convergence behavior.
float2 KvaziFIREdamp( float c, float2 clim, float2 damps ){
    float2 cvf;
    if      (c < clim.x ){   //-- force against veloctiy
        cvf.x = damps.x;     // v    // like 0.5 (strong damping)
        cvf.y = 0;           // f
    }else if(c > clim.y ){   //-- force alingned to velocity
        cvf.x = damps.y;     // v    // like 0.99 (weak dampong damping)
        cvf.y = 0;           // f
    }else{                   // -- force ~ perpendicular to velocity
        float t = (c-clim.x )/( clim.y - clim.x );
        cvf.x = damps.x + (damps.y-damps.x)*t;
        cvf.y = damps.y*t*(1.f-t)*2.f;
    }
    return cvf;
}

// Wang hash: fast integer hash for thermal noise generation.
// Reference: https://www.reishin.org/pseud-random-number-generator-wang/
unsigned int hash_wang(unsigned int bits) {
    //unsigned int bits = __float_as_int(value);
    bits = (bits ^ 61) ^ (bits >> 16);
    bits *= 9;
    bits = bits ^ (bits >> 4);
    bits *= 0x27d4eb2d;
    bits = bits ^ (bits >> 15);
    return bits;
}

// Hash a float to a uniform random float in [xmin, xmax].
// Uses Wang hash on the bit representation of val.
float hashf_wang( float val, float xmin, float xmax) {
    //return ( (float)(bits)*(2147483647.0f );
    // ported: __float_as_int is CUDA/PTX; use OpenCL as_int for bitcast.
    return (((float)( hash_wang(  as_int(val) ) )) * 4.6566129e-10 )  *(xmax-xmin)+ xmin;
}

//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void updateAtomsSPFFf4(
    const int4        n,            // 1 // (natoms,nnode) dimensions of the system
    __global float4*  apos,         // 2 // positions of atoms  (including node atoms [0:nnode] and capping atoms [nnode:natoms] and pi-orbitals [natoms:natoms+nnode] )
    __global float4*  avel,         // 3 // velocities of atoms
    __global float4*  aforce,       // 4 // forces on atoms
    __global float4*  cvf,          // 5 // damping coefficients for velocity and force
    __global float4*  fneigh,       // 6 // recoil forces on neighbors (and pi-orbitals)
    __global int4*    bkNeighs,     // 7 // back neighbors indices (for recoil forces)
    __global float4*  constr,       // 8 // constraints (x,y,z,K) for each atom
    __global float4*  constrK,      // 9 // constraints stiffness (kx,ky,kz,?) for each atom
    __global float4*  MDparams,     // 10 // MD parameters (dt,damp,Flimit)
    __global float4*  TDrives,      // 11 // Thermal driving (T,gamma_damp,seed,?)
    __global cl_Mat3* bboxes,       // 12 // bounding box (xmin,ymin,zmin)(xmax,ymax,zmax)(kx,ky,kz)
    __global int*     sysneighs,    // 13 // // for each system contains array int[nMaxSysNeighs] of nearby other systems
    __global float4*  sysbonds      // 14 // // contains parameters of bonds (constrains) with neighbor systems   {Lmin,Lmax,Kpres,Ktens}
){
    const int natoms=n.x;           // number of atoms
    const int nnode =n.y;           // number of node atoms
    const int nMaxSysNeighs = n.w;  // max number of inter-system interactions; if <0 shwitch inter system interactions off
    const int nvec  = natoms+nnode; // number of vectors (atoms+node atoms)
    const int iG = get_global_id  (0); // index of atom

    if(iG>=nvec) return;

    const int iS = get_global_id  (1); // index of system
    const int nG = get_global_size(0); // number of atoms
    const int nS = get_global_size(1); // number of systems

    //const int ian = iG + iS*nnode;
    const int iaa = iG + iS*natoms;  // index of atom in atoms array
    const int iav = iG + iS*nvec;    // index of atom in vectors array

    const float4 MDpars  = MDparams[iS]; // (dt,damp,Flimit)
    const float4 TDrive = TDrives[iS];

    // if((iS==0)&&(iG==0)){
    //     //printf("MDpars[%i] (%g,%g,%g,%g) \n", iS, MDpars.x,MDpars.y,MDpars.z,MDpars.w);
    //     for(int is=0; is<nS; is++){
    //         //printf( "GPU::TDrives[%i](%g,%g,%g,%g)\n", i, TDrives[i].x,TDrives[i].y,TDrives[i].z,TDrives[i].w );
    //         //printf( "GPU::bboxes[%i](%g,%g,%g)(%g,%g,%g)(%g,%g,%g)\n", is, bboxes[is].a.x,bboxes[is].a.y,bboxes[is].a.z,   bboxes[is].b.x,bboxes[is].b.y,bboxes[is].b.z,   bboxes[is].c.x,bboxes[is].c.y,bboxes[is].c.z );
    //         for(int ia=0; ia<natoms; ia++){
    //             int ic = ia+is*natoms;
    //             if(constr[ia+is*natoms].w>0) printf( "GPU:sys[%i]atom[%i] constr(%g,%g,%g|%g) constrK(%g,%g,%g|%g)\n", is, ia, constr[ic].x,constr[ic].y,constr[ic].z,constr[ic].w,   constrK[ic].x,constrK[ic].y,constrK[ic].z,constrK[ic].w  );
    //         }
    //     }
    // }

    const int iS_DBG = 5; // debug system
    //const int iG_DBG = 0;
    const int iG_DBG = 1; // debug atom

    //if((iG==iG_DBG)&&(iS==iS_DBG))printf( "updateAtomsSPFFf4() natoms=%i nnode=%i nvec=%i nG %i iS %i/%i  dt=%g damp=%g Flimit=%g \n", natoms,nnode, nvec, iS, nG, nS, MDpars.x, MDpars.y, MDpars.z );
    // if((iG==iG_DBG)&&(iS==iS_DBG)){
    //     int i0a = iS*natoms;
    //     for(int i=0; i<natoms; i++){
    //         printf( "GPU:constr[%i](%7.3f,%7.3f,%7.3f |K= %7.3f) \n", i, constr[i0a+i].x,constr[i0a+i].y,constr[i0a+i].z,  constr[i0a+i].w   );
    //     }
    // }
    if(iG>=(natoms+nnode)) return; // make sure we are not out of bounds of current system

    //aforce[iav] = float4Zero;

    const float4 fe0     = aforce[iav]; // force on atom or pi-orbital (before recoil)
    float4 fe      = fe0;
    const bool bPi = iG>=natoms;  // is it pi-orbital ?

    // ------ Gather Forces from back-neighbors

    int4 ngs = bkNeighs[ iav ]; // back neighbors indices

    //if(iS==5)printf( "iG,iS %i %i ngs %i,%i,%i,%i \n", iG, iS, ngs.x,ngs.y,ngs.z,ngs.w );
    //if( (iS==0)&&(iG==0) ){ printf( "GPU:fe.1[iS=%i,iG=%i](%g,%g,%g,%g) \n", fe.x,fe.y,fe.z,fe.w ); }

    // sum all recoil forces from back neighbors   - WARRNING : bkNeighs must be properly shifted on CPU by adding offset of system iS*nvec*4
    {
    float4 frec = float4Zero;
    if(ngs.x>=0){ frec += fneigh[ngs.x]; } // if neighbor index is negative it means that there is no neighbor, so we skip it
    if(ngs.y>=0){ frec += fneigh[ngs.y]; }
    if(ngs.z>=0){ frec += fneigh[ngs.z]; }
    if(ngs.w>=0){ frec += fneigh[ngs.w]; }
    fe += frec;

    #if DBG_UFF
    if((iG==iGdbg)&&(iS==iSdbg)){
        printf("DBG updateAtomsSPFFf4(relax_multi.cl) iS=%i iG=%i iav=%i bPi=%i fe0=(%g,%g,%g|%g) frec=(%g,%g,%g|%g) fe=(%g,%g,%g|%g) ngs=(%i,%i,%i,%i)\n",
            iS,iG,iav,(int)bPi, fe0.x,fe0.y,fe0.z,fe0.w, frec.x,frec.y,frec.z,frec.w, fe.x,fe.y,fe.z,fe.w, ngs.x,ngs.y,ngs.z,ngs.w );
        if(!bPi){
            if(ngs.x>=0){ float4 t=fneigh[ngs.x]; printf("DBG updateAtomsSPFFf4(relax_multi.cl) recoil0 idx=%i fneigh=(%g,%g,%g|%g)\n", ngs.x, t.x,t.y,t.z,t.w ); }
            if(ngs.y>=0){ float4 t=fneigh[ngs.y]; printf("DBG updateAtomsSPFFf4(relax_multi.cl) recoil1 idx=%i fneigh=(%g,%g,%g|%g)\n", ngs.y, t.x,t.y,t.z,t.w ); }
            if(ngs.z>=0){ float4 t=fneigh[ngs.z]; printf("DBG updateAtomsSPFFf4(relax_multi.cl) recoil2 idx=%i fneigh=(%g,%g,%g|%g)\n", ngs.z, t.x,t.y,t.z,t.w ); }
            if(ngs.w>=0){ float4 t=fneigh[ngs.w]; printf("DBG updateAtomsSPFFf4(relax_multi.cl) recoil3 idx=%i fneigh=(%g,%g,%g|%g)\n", ngs.w, t.x,t.y,t.z,t.w ); }
        }
    }
    #endif
    }
 // ---- Limit Forces - WARNING: this can lead to drift; prefer limiting in forcefield kernels when possible
    float Flimit = MDpars.z;
    if(Flimit>0){
        float fr2 = dot(fe.xyz,fe.xyz);  // squared force
        if( fr2 > (Flimit*Flimit) ){  fe.xyz*=(Flimit/sqrt(fr2)); }  // if force is too big, we scale it down to Flimit
    }

    // =============== FORCE DONE
    aforce[iav] = fe;             // store force before limit
    //aforce[iav] = float4Zero;   // clean force   : This can be done in the first forcefield run (best is NBFF)

    // =============== DYNAMICS

    float4 ve = avel[iav]; // velocity of atom or pi-orbital
    float4 pe = apos[iav]; // position of atom or pi-orbital

    // -------- Fixed Atoms and Bounding Box
    if(iG<natoms){                  // only atoms have constraints, not pi-orbitals
        // ------- bboxes
        const cl_Mat3 B = bboxes[iS];
        // if(B.c.x>0.0f){ if(pe.x<B.a.x){ fe.x+=(B.a.x-pe.x)*B.c.x; }else if(pe.x>B.b.x){ fe.x+=(B.b.x-pe.x)*B.c.x; }; }
        // if(B.c.y>0.0f){ if(pe.y<B.a.y){ fe.y+=(B.a.y-pe.y)*B.c.y; }else if(pe.y>B.b.y){ fe.y+=(B.b.y-pe.y)*B.c.y; }; }
        if(B.c.z>0.0f){ if(pe.z<B.a.z){ fe.z+=(B.a.z-pe.z)*B.c.z; }else if(pe.z>B.b.z){ fe.z+=(B.b.z-pe.z)*B.c.z; }; }
        // ------- constrains
        float4 cons = constr[ iaa ]; // constraints (x,y,z,K)
        if( cons.w>0.f ){            // if stiffness is positive, we have constraint
            float4 cK = constrK[ iaa ];
            cK = max( cK, (float4){0.0f,0.0f,0.0f,0.0f} );
            const float3 fc = (cons.xyz - pe.xyz)*cK.xyz;
            fe.xyz += fc; // add constraint force
            // if(iS==0){printf( "GPU::constr[ia=%i|iS=%i] (%g,%g,%g|K=%g) fc(%g,%g,%g) cK(%g,%g,%g)\n", iG, iS, cons.x,cons.y,cons.z,cons.w, fc.x,fc.y,fc.z , cK.x, cK.y, cK.z ); }
        }
    }

    // -------- Inter system interactions
    if( nMaxSysNeighs>0 ){
        for(int i=0; i<nMaxSysNeighs; i++){
            const int j     = iS*nMaxSysNeighs + i;
            const int    jS = sysneighs[j];
            const float4 bj = sysbonds [j];
            const float4 pj = apos[jS*nvec + iG];
            float3 d        = pj.xyz - pe.xyz;
            float  l = length( d );
            if      (l<bj.x){
                d*=(l-bj.x)*bj.z/l;  // f = dx*kPress
            }else if(l>bj.y){
                d*=(bj.y-l)*bj.w/l;  // f = dx*kTens
            }
            fe.xyz += d;
        }
    }

    // ------ Simple damped MD (leap-frog when damp=1.0)
    if(bPi){
        fe.xyz += pe.xyz * -dot( pe.xyz, fe.xyz );   // project out radial component for pi-orbitals
        ve.xyz += pe.xyz * -dot( pe.xyz, ve.xyz );
    }
    const float dt   = MDpars.x;
    const float damp = MDpars.y;
    float inv_mass = (pe.w > 1e-8f) ? (1.0f / pe.w) : 1.0f;
    ve.xyz *= damp;
    ve.xyz += fe.xyz * dt * inv_mass;
    pe.xyz += ve.xyz * dt;
    if(bPi){
        pe.xyz=normalize(pe.xyz);                   // normalize pi-orbitals
    }
    ve.w=0;
    avel[iav] = ve;
    apos[iav] = pe;   // pe.w still holds mass
}
// ======================================================================
//                     cleanForceSPFFf4()
// ======================================================================
//
//  Zeros force arrays between MD steps. One thread per atom per system.
//  Clears: aforce[iav] and fneigh[ian*4..ian*4+3] (4 recoil slots per node).
//  Must be called before getSPFFf4/getNonBond_ex2 at the start of each step.
//
//  CAVEAT: Only clears fneigh for node atoms (iG < nnode). Pi-orbital
//  recoil slots (i4p) are cleared in the force evaluation kernels, not here.
//
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void cleanForceSPFFf4(
    const int4        n,           // 2
    __global float4*  aforce,      // 5
    __global float4*  fneigh       // 6
){
    const int natoms=n.x;
    const int nnode =n.y;
    const int iG = get_global_id  (0);
    const int iS = get_global_id  (1);
    const int nG = get_global_size(0);
    const int nS = get_global_size(1);
    const int nvec = natoms+nnode;

    const int iav = iG + iS*nvec;
    const int ian = iG + iS*nnode;

    aforce[iav]=float4Zero;
    //aforce[iav]=(float4){iG,iS,iav,0.0};

    //if(iav==0){ printf("GPU::cleanForceSPFFf4() iS %i nG %i nS %i \n", iS, nG, nS );}
    //if(iG==0){ for(int i=0;i<(natoms+nnode);i++ ){printf("cleanForceSPFFf4[%i](%g,%g,%g)\n",i,aforce[i].x,aforce[i].y,aforce[i].z);} }
    if(iG<nnode){
        const int i4 = ian*4;
        fneigh[i4+0]=float4Zero;
        fneigh[i4+1]=float4Zero;
        fneigh[i4+2]=float4Zero;
        fneigh[i4+3]=float4Zero;
    }
    //if(iG==0){ printf( "GPU::updateAtomsSPFFf4() END\n" ); }
}

// ======================================================================
//                     updateAtomsSPFFf4_rot()
// ======================================================================
//
//  MD integrator for SPFF with rotational pi-orbital dynamics.
//  Atoms use damped leap-frog (same as updateAtomsSPFFf4).
//  Pi-orbitals use proper rotational dynamics:
//    1. angular_velocity *= damp
//    2. angular_velocity += torque * dt  (inv_I = 1 for unit sphere)
//    3. pi_orbital = rotate_by_omega_taylor(pi, angular_velocity * dt)
//    4. pi_orbital = normalize(pi_orbital)
//
//  The rotation uses Rodrigues' formula with Taylor series for small angles.
//  This is more physically correct than the linear projection approach in
//  updateAtomsSPFFf4, which only approximates motion on S^2.
//
//  CAVEAT: inv_I = 1.0 hardcoded for pi-orbitals. If different moments of
//  inertia are needed for different atom types, this must be parameterized.
//
//  CAVEAT: Recoil forces are NOT gathered for pi-orbitals in this kernel
//  (the `if(!bPi)` guard at line ~932 skips the gather). Pi recoil is
//  handled differently in the _rot variant — torques are applied directly.
//
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void updateAtomsSPFFf4_rot(
    const int4        nDOFs,            // 1 // (natoms,nnode) dimensions of the system
    __global float4*  apos,         // 2 // positions of atoms
    __global float4*  avel,         // 3 // velocities of atoms (angular velocity for pi)
    __global float4*  aforce,       // 4 // forces on atoms (torques on pi)
    __global float4*  cvf,          // 5 // damping coefficients
    __global float4*  fneigh,       // 6 // recoil forces on neighbors
    __global int4*    bkNeighs,     // 7 // back neighbors indices
    __global float4*  constr,       // 8 // constraints (x,y,z,K) for each atom
    __global float4*  constrK,      // 9 // constraints stiffness (kx,ky,kz,?) for each atom
    __global float4*  MDparams,     // 10 // MD parameters (dt,damp,Flimit)
    __global float4*  TDrives,      // 11 // Thermal driving (T,gamma_damp,seed,?)
    __global cl_Mat3* bboxes,       // 12 // bounding box
    __global int*     sysneighs,    // 13 // inter-system neighbor indices
    __global float4*  sysbonds,     // 14 // inter-system bond parameters
    __global float4*  aforce_old    // 15 // previous step forces
){
    const int natoms=nDOFs.x;
    const int nnode =nDOFs.y;
    const int nMaxSysNeighs = nDOFs.z;
    const int nvec  = natoms+nnode;
    const int iG = get_global_id  (0);
    if(iG>=nvec) return;
    const int iS = get_global_id  (1);

    const int iaa = iG + iS*natoms;
    const int iav = iG + iS*nvec;

    const float4 MDpars  = MDparams[iS]; // (dt,damp,Flimit)
    const float4 TDrive = TDrives[iS];

    if(iG>=(natoms+nnode)) return;

    float4 fe      = aforce[iav];
    const bool bPi = iG>=natoms;

    int4 ngs = bkNeighs[ iav ];

    if(!bPi){
        if(ngs.x>=0){ fe += fneigh[ngs.x]; }
        if(ngs.y>=0){ fe += fneigh[ngs.y]; }
        if(ngs.z>=0){ fe += fneigh[ngs.z]; }
        if(ngs.w>=0){ fe += fneigh[ngs.w]; }
    }

    float Flimit = MDpars.z;
    if(Flimit>0){
        float fr2 = dot(fe.xyz,fe.xyz);
        if( fr2 > (Flimit*Flimit) ){  fe.xyz*=(Flimit/sqrt(fr2)); }
    }

    aforce[iav] = fe;

    float4 ve = avel[iav];
    float4 pe = apos[iav];

    // Constraints and bounding box (atoms only)
    if(iG<natoms){
        const cl_Mat3 B = bboxes[iS];
        if(B.c.z>0.0f){ if(pe.z<B.a.z){ fe.z+=(B.a.z-pe.z)*B.c.z; }else if(pe.z>B.b.z){ fe.z+=(B.b.z-pe.z)*B.c.z; }; }
        float4 cons = constr[ iaa ];
        if( cons.w>0.f ){
            float4 cK = constrK[ iaa ];
            cK = max( cK, (float4){0.0f,0.0f,0.0f,0.0f} );
            const float3 fc = (cons.xyz - pe.xyz)*cK.xyz;
            fe.xyz += fc;
        }
    }

    // Inter-system interactions
    if( nMaxSysNeighs>0 ){
        for(int i=0; i<nMaxSysNeighs; i++){
            const int j     = iS*nMaxSysNeighs + i;
            const int    jS = sysneighs[j];
            const float4 bj = sysbonds [j];
            const float4 pj = apos[jS*nvec + iG];
            float3 d        = pj.xyz - pe.xyz;
            float  l = length( d );
            if      (l<bj.x){ d*=(l-bj.x)*bj.z/l; }
            else if (l>bj.y){ d*=(bj.y-l)*bj.w/l; }
            fe.xyz += d;
        }
    }

    const float dt   = MDpars.x;
    const float damp = MDpars.y;

    if (bPi){
        // ROTATIONAL DYNAMICS FOR PI-ORBITAL
        float inv_I  = 1.0f;
        ve.xyz *= damp;
        ve.xyz += (fe.xyz * inv_I) * dt;
        pe.xyz  = rotate_by_omega_taylor( pe.xyz, ve.xyz*dt );
        pe.xyz  = normalize(pe.xyz);
    } else {
        // LEAP-FROG FOR ATOMS with damping
        float inv_mass = (pe.w > 1e-8f) ? (1.0f / pe.w) : 1.0f;
        ve.xyz *= damp;
        ve.xyz += fe.xyz * dt * inv_mass;
        pe.xyz += ve.xyz * dt;
    }
    pe.w = 0.0f; ve.w = 0.0f;
    avel[iav] = ve;
    apos[iav] = (float4){ pe.xyz, 0.0f };
}


// ======================================================================
//                     relax_nsteps_serial()
// ======================================================================
//
//  Fused relaxation kernel: runs nsteps MD steps entirely in local memory
//  within a single workgroup. Eliminates Python dispatch overhead (3 kernel
//  calls/step -> 1 call total). No non-bonded interactions (slot reserved).
//
//  GPU strategy: One workgroup per molecule. All data (positions, velocities,
//  forces, neighbor lists, FF params) is loaded into __local memory once,
//  then the relaxation loop runs entirely on-chip with barriers between phases.
//
//  Workgroup size: WG_SIZE=192 threads. Covers molecules up to ~96 atoms
//  with pi nodes (nvec = natoms + nnode <= MAX_NVEC=192, nnode <= MAX_NNODE=96).
//
//  Local arrays are sized to MAX_NVEC / MAX_NATOM / MAX_NNODE (not all to WG_SIZE)
//  so WG=192 fits in ~38 KB local memory (NVIDIA ~48 KB limit). Naive WG=256 with
//  12*WG float4 arrays exceeds ~64 KB and fails to compile on many GPUs.
//
//  Data flow per step:
//    Phase 1: Zero aforce + fneigh (all threads cooperate) -> barrier
//    Phase 2: Compute SPFF bonded forces (threads 0..nnode-1) -> barrier
//    Phase 3: Gather recoil + integrate (threads 0..nvec-1) -> barrier
//  All data stays in __local memory between steps — no global memory traffic.
//
//  CAVEAT: No non-bonded interactions. Only bonded (bond, angle, pi) forces
//  are computed. Suitable for intramolecular relaxation where non-bonded
//  forces are negligible or handled separately.
//
//  CAVEAT: PBC is NOT handled in this kernel (bonds assumed within one cell).
//  For periodic systems, use the multi-kernel path with getSPFFf4 instead.
//
//  CAVEAT: The fneigh array is sized MAX_NNODE*8 = 96*8 = 768 float4s.
//  This is the maximum for nnode=96 with 4 neighbors * 2 (sigma+pi) slots.
//
//  CAVEAT: Phase 2/3 require WG_SIZE >= nnode and WG_SIZE >= nvec (one thread
//  per DOF). Enforce in Python before launch.
//
// ======================================================================
#ifndef WG_SIZE
#define WG_SIZE 192
#endif
#ifndef MAX_NVEC
#define MAX_NVEC 192
#endif
#ifndef MAX_NATOM
#define MAX_NATOM 128
#endif
#ifndef MAX_NNODE
#define MAX_NNODE 96
#endif

#ifndef FAF_BASIS_MAX
#define FAF_BASIS_MAX 128
#endif
#ifndef FAF_TYPES_MAX
#define FAF_TYPES_MAX 8
#endif

inline float folded_eval_basis_s(float u, float v, float z, float4 prm){
    const float twopi = 6.283185307179586f;
    float bx = native_cos(twopi * prm.x * u);
    float by = native_cos(twopi * prm.y * v);
    float bz = native_exp(-prm.z * fmax(0.0f, z - prm.w));
    return bx * by * bz;
}

// Correct chain rule (dudx,dudy,dvdx,dvdy) packed in invL — matches rigid.cl, not buggy surface.cl
inline float3 folded_eval_grad_s(float u, float v, float z, float4 prm, float4 invL){
    const float twopi = 6.283185307179586f;
    float ku = prm.x, kv = prm.y, az = prm.z, z0 = prm.w;
    float bx = native_cos(twopi * ku * u);
    float by = native_cos(twopi * kv * v);
    float bz = native_exp(-az * fmax(0.0f, z - z0));
    float dE_du = (-twopi * ku * native_sin(twopi * ku * u)) * by * bz;
    float dE_dv = bx * (-twopi * kv * native_sin(twopi * kv * v)) * bz;
    float dE_dz = (z >= z0) ? (bx * by * (-az * bz)) : 0.0f;
    return (float3)(dE_du*invL.x + dE_dv*invL.z, dE_du*invL.y + dE_dv*invL.w, dE_dz);
}

__kernel void relax_nsteps_serial(
    const int4  nDOFs,              // (natoms, nnode, nsteps, do_faf)
    __global       float4*  g_apos,     // [nvec] positions (atoms + pi)
    __global       float4*  g_avel,     // [nvec] velocities
    __global       float4*  g_aforce,   // [nvec] forces (output)
    __global const int4*    g_neighs,   // [natoms] neighbor indices
    __global const int4*    g_bkNeighs, // [nvec] back-neighbor indices into fneigh
    __global const float4*  g_apars,    // [nnode] FF params {c0ss, Kss, c0sp}
    __global const float4*  g_bLs,      // [nnode] bond lengths (4 per node)
    __global const float4*  g_bKs,      // [nnode] bond stiffness
    __global const float4*  g_Ksp,      // [nnode] sigma-pi stiffness
    __global const float4*  g_Kpp,      // [nnode] pi-pi stiffness
    __global const float4*  g_constr,   // [natoms] constraints (xyz, K_flag)
    __global const float4*  g_constrK,  // [natoms] constraint stiffness
    __global const float4*  g_MDparams, // (dt, damp, Flimit, 0)
    // --- FAF substrate (used when do_faf!=0); same contract as getSurfFolded ---
    __global const float*   g_folded_coeffs,   // [ntypes*nbasis]
    __global const float4*  g_folded_kxyz,     // [nbasis]
    __global const int*     g_folded_atom_type,// [natoms]
    const int4              folded_meta,       // (nbasis, ntypes, 0, 0)
    const float4            folded_lvec2d      // (ax,bx,ay,by)
){
    const int natoms = nDOFs.x;
    const int nnode  = nDOFs.y;
    const int nsteps = nDOFs.z;
    const int do_faf = nDOFs.w;
    const int nvec   = natoms + nnode;
    const int iL     = get_local_id(0);

    // ---- Local memory buffers (sized by role, not all to WG_SIZE) ----
    __local float4  s_apos   [MAX_NVEC];      // positions (atoms + pi)
    __local float4  s_avel   [MAX_NVEC];      // velocities
    __local float4  s_aforce [MAX_NVEC];      // forces
    __local float4  s_fneigh [MAX_NNODE*8];   // recoil forces (nnode*4*2 max)
    __local int4    s_neighs [MAX_NATOM];     // neighbor indices
    __local int4    s_bkNeighs[MAX_NVEC];     // back-neighbor indices
    __local float4  s_apars  [MAX_NNODE];     // FF params
    __local float4  s_bLs    [MAX_NNODE];     // bond lengths
    __local float4  s_bKs    [MAX_NNODE];     // bond stiffness
    __local float4  s_Ksp    [MAX_NNODE];     // sigma-pi stiffness
    __local float4  s_Kpp    [MAX_NNODE];     // pi-pi stiffness
    __local float4  s_constr [MAX_NATOM];     // constraints
    __local float4  s_constrK[MAX_NATOM];     // constraint stiffness
    __local int     s_atype  [MAX_NATOM];     // FAF atom type
    __local float4  LBASIS   [FAF_BASIS_MAX];
    __local float   LCOEFFS  [FAF_TYPES_MAX * FAF_BASIS_MAX];

    // ---- Cooperative load from global to local ----
    for(int i = iL; i < nvec;    i += WG_SIZE) s_apos[i]      = g_apos[i];
    for(int i = iL; i < nvec;    i += WG_SIZE) s_avel[i]      = g_avel[i];
    for(int i = iL; i < natoms;  i += WG_SIZE) s_neighs[i]    = g_neighs[i];
    for(int i = iL; i < nvec;    i += WG_SIZE) s_bkNeighs[i]  = g_bkNeighs[i];
    for(int i = iL; i < nnode;   i += WG_SIZE) s_apars[i]     = g_apars[i];
    for(int i = iL; i < nnode;   i += WG_SIZE) s_bLs[i]       = g_bLs[i];
    for(int i = iL; i < nnode;   i += WG_SIZE) s_bKs[i]       = g_bKs[i];
    for(int i = iL; i < nnode;   i += WG_SIZE) s_Ksp[i]       = g_Ksp[i];
    for(int i = iL; i < nnode;   i += WG_SIZE) s_Kpp[i]       = g_Kpp[i];
    for(int i = iL; i < natoms;  i += WG_SIZE) s_constr[i]    = g_constr[i];
    for(int i = iL; i < natoms;  i += WG_SIZE) s_constrK[i]   = g_constrK[i];
    for(int i = iL; i < nnode*8; i += WG_SIZE) s_fneigh[i]   = float4Zero;

    // Load MD params
    const float4 MDpars = g_MDparams[0];
    const float dt   = MDpars.x;
    const float damp = MDpars.y;
    const float Flimit = MDpars.z;

    // FAF cache (once)
    float4 invLvec2d = (float4)(0,0,0,0);
    int nbasis = 0, ntypes = 0;
    if(do_faf){
        nbasis = folded_meta.x;
        ntypes = folded_meta.y;
        if(nbasis>0 && nbasis<=FAF_BASIS_MAX && ntypes>0 && ntypes<=FAF_TYPES_MAX){
            for(int j=iL; j<nbasis; j+=WG_SIZE) LBASIS[j] = g_folded_kxyz[j];
            for(int j=iL; j<nbasis*ntypes; j+=WG_SIZE) LCOEFFS[j] = g_folded_coeffs[j];
            for(int j=iL; j<natoms; j+=WG_SIZE) s_atype[j] = g_folded_atom_type[j];
            float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
            float det = ax*by - bx*ay;
            if(fabs(det) > 1e-12f) invLvec2d = (float4)(by/det, -bx/det, -ay/det, ax/det);
            else nbasis = 0;
        } else { nbasis = 0; }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    #define NNEIGH 4

    // ---- Main relaxation loop ----
    for(int step = 0; step < nsteps; step++){

        // === Phase 1: Zero forces ===
        for(int i = iL; i < nvec;    i += WG_SIZE) s_aforce[i] = float4Zero;
        for(int i = iL; i < nnode*8; i += WG_SIZE) s_fneigh[i] = float4Zero;
        barrier(CLK_LOCAL_MEM_FENCE);

        // === Phase 2: Compute SPFF bonded forces (threads 0..nnode-1) ===
        if(iL < nnode){
            const int iG = iL;
            float4  hs[4];
            float3  fbs[4];
            float3  fps[4];
            float3  fa  = float3Zero;
            float   E   = 0;

            const int4  ng  = s_neighs[iG];
            const float3 pa = s_apos[iG].xyz;
            const float4 par = s_apars[iG];
            const int*  ings = (int*)&ng;

            for(int i=0; i<NNEIGH; i++){ fbs[i]=float3Zero; fps[i]=float3Zero; }

            float3 fpi = float3Zero;

            // --- Bonds ---
            {
                const float4 vbL = s_bLs[iG];
                const float4 vbK = s_bKs[iG];
                const float4 vKs = s_Ksp[iG];
                const float4 vKp = s_Kpp[iG];
                const float* bL   = (float*)&vbL;
                const float* bK   = (float*)&vbK;
                const float* Kspi = (float*)&vKs;
                const float* Kppi = (float*)&vKp;
                const float3 hpi  = s_apos[natoms + iG].xyz;

                for(int i=0; i<NNEIGH; i++){
                    float4 h;
                    const int ing = ings[i];
                    if(ing<0) break;
                    const int ingv = ing;

                    h.xyz = s_apos[ingv].xyz - pa;
                    float l = length(h.xyz);
                    h.w = 1.f/l;
                    h.xyz *= h.w;
                    hs[i] = h;

                    if(iG < ing){
                        float3 f1;
                        E += evalBond(h.xyz, l-bL[i], bK[i], &f1);
                        fbs[i] -= f1; fa += f1;

                        float kpp = Kppi[i];
                        if((ing < nnode) && (kpp > 1e-6f)){
                            float3 f1p, f2p;
                            const float3 hpi_j = s_apos[natoms + ing].xyz;
                            E += evalPiAling(hpi, hpi_j, kpp, &f1p, &f2p);
                            fpi += f1p; fps[i] += f2p;
                        }
                    }

                    float ksp = Kspi[i];
                    if(ksp > 1e-6f){
                        float3 f1, f2;
                        E += evalAngCos((float4){hpi,1.f}, h, ksp, par.w, &f1, &f2);
                        fpi += f1; fa -= f2; fbs[i] += f2;
                    }
                }

                const int i4p = iG*4 + nnode*4;
                for(int i=0; i<NNEIGH; i++) s_fneigh[i4p+i] = (float4){fps[i], 0};
                s_aforce[natoms + iG] = (float4){fpi, 0};
            }

            // --- Angles ---
            for(int i=0; i<NNEIGH; i++){
                int ing = ings[i];
                if(ing<0) break;
                const float4 hi = hs[i];
                for(int j=i+1; j<NNEIGH; j++){
                    int jng = ings[j];
                    if(jng<0) break;
                    const float4 hj = hs[j];
                    float3 f1, f2;
                    E += evalAngleCosHalf(hi, hj, par.xy, par.z, &f1, &f2);
                    fa -= f1 + f2;
                    fbs[i] += f1;
                    fbs[j] += f2;
                }
            }

            // --- Store forces ---
            const int i4 = iG*4;
            for(int i=0; i<NNEIGH; i++) s_fneigh[i4+i] = (float4){fbs[i], 0};
            s_aforce[iG] += (float4){fa, E};
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // === Phase 2b: FAF substrate (atoms only; parallel over atoms) ===
        if(do_faf && nbasis>0){
            for(int ia = iL; ia < natoms; ia += WG_SIZE){
                float3 pos = s_apos[ia].xyz;
                float u = invLvec2d.x*pos.x + invLvec2d.y*pos.y;
                float v = invLvec2d.z*pos.x + invLvec2d.w*pos.y;
                u = u - floor(u);
                v = v - floor(v);
                int ityp = s_atype[ia];
                if(ityp < 0 || ityp >= ntypes) continue;
                float E = 0.0f;
                float3 F = (float3)(0.0f,0.0f,0.0f);
                int ioff = ityp*nbasis;
                for(int ib=0; ib<nbasis; ib++){
                    float c = LCOEFFS[ioff + ib];
                    float4 prm = LBASIS[ib];
                    float  b = folded_eval_basis_s(u, v, pos.z, prm);
                    float3 g = folded_eval_grad_s(u, v, pos.z, prm, invLvec2d);
                    E += c * b;
                    F -= c * g;
                }
                s_aforce[ia] += (float4)(F.x, F.y, F.z, -E);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // === Phase 3: Gather recoil + integrate (threads 0..nvec-1) ===
        if(iL < nvec){
            const int iG = iL;
            float4 fe = s_aforce[iG];
            const bool bPi = (iG >= natoms);

            // Gather recoil from back-neighbors
            const int4 ngs = s_bkNeighs[iG];
            float4 frec = float4Zero;
            if(ngs.x>=0) frec += s_fneigh[ngs.x];
            if(ngs.y>=0) frec += s_fneigh[ngs.y];
            if(ngs.z>=0) frec += s_fneigh[ngs.z];
            if(ngs.w>=0) frec += s_fneigh[ngs.w];
            fe += frec;

            // Force limiting
            if(Flimit > 0){
                float fr2 = dot(fe.xyz, fe.xyz);
                if(fr2 > Flimit*Flimit) fe.xyz *= Flimit / sqrt(fr2);
            }

            s_aforce[iG] = fe;

            // Constraints (only for atoms, not pi)
            if(iG < natoms){
                float4 cons = s_constr[iG];
                if(cons.w > 0){
                    float4 cK = s_constrK[iG];
                    cK = max(cK, (float4){0,0,0,0});
                    float4 pe_c = s_apos[iG];
                    fe.xyz += (cons.xyz - pe_c.xyz) * cK.xyz;
                }
            }

            // Damped MD integration
            float4 ve = s_avel[iG];
            float4 pe = s_apos[iG];

            if(bPi){
                fe.xyz += pe.xyz * (-dot(pe.xyz, fe.xyz));
                ve.xyz += pe.xyz * (-dot(pe.xyz, ve.xyz));
            }

            float inv_mass = (pe.w > 1e-8f) ? (1.f / pe.w) : 1.f;
            ve.xyz *= damp;
            ve.xyz += fe.xyz * dt * inv_mass;
            pe.xyz += ve.xyz * dt;

            if(bPi) pe.xyz = normalize(pe.xyz);

            ve.w = 0;
            s_avel[iG] = ve;
            s_apos[iG] = pe;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }  // end step loop

    // ---- Write results back to global memory ----
    for(int i = iL; i < nvec;    i += WG_SIZE) g_apos[i]   = s_apos[i];
    for(int i = iL; i < nvec;    i += WG_SIZE) g_avel[i]   = s_avel[i];
    for(int i = iL; i < nvec;    i += WG_SIZE) g_aforce[i] = s_aforce[i];
}


// ======================================================================
//                     relax_nsteps_global()
// ======================================================================
//
//  Same fused nsteps MD loop as relax_nsteps_serial, but dynamics and
//  topology stay in GLOBAL memory. Overcomes local-memory size limits so
//  workgroup size can be 256/512 and nvec can exceed ~192.
//
//  Strategy: one workgroup, strided loops over nnode/nvec (WG need not
//  equal nvec). Uses global fneigh buffer for recoil. Barriers between
//  phases. Optional FAF substrate (do_faf!=0) adds folded-basis forces
//  after bonded forces (coeffs/kxyz cached in local — tiny).
//
// ======================================================================
#ifndef WG_GLOBAL
#define WG_GLOBAL 256
#endif

inline float folded_eval_basis_g(float u, float v, float z, float4 prm){
    const float twopi = 6.283185307179586f;
    float bx = native_cos(twopi * prm.x * u);
    float by = native_cos(twopi * prm.y * v);
    float bz = native_exp(-prm.z * fmax(0.0f, z - prm.w));
    return bx * by * bz;
}

inline float3 folded_eval_grad_g(float u, float v, float z, float4 prm, float4 invL){
    const float twopi = 6.283185307179586f;
    float ku = prm.x, kv = prm.y, az = prm.z, z0 = prm.w;
    float bx = native_cos(twopi * ku * u);
    float by = native_cos(twopi * kv * v);
    float bz = native_exp(-az * fmax(0.0f, z - z0));
    float dbx_du = -twopi * ku * native_sin(twopi * ku * u);
    float dby_dv = -twopi * kv * native_sin(twopi * kv * v);
    float dbz_dz = (z >= z0) ? (-az * bz) : 0.0f;
    float dE_du = dbx_du * by * bz;
    float dE_dv = bx * dby_dv * bz;
    float dE_dz = bx * by * dbz_dz;
    float3 g;
    g.x = dE_du * invL.x + dE_dv * invL.z;
    g.y = dE_du * invL.y + dE_dv * invL.w;
    g.z = dE_dz;
    return g;
}

__kernel void relax_nsteps_global(
    const int4  nDOFs,              // (natoms, nnode, nsteps, do_faf)
    __global       float4*  g_apos,
    __global       float4*  g_avel,
    __global       float4*  g_aforce,
    __global       float4*  g_fneigh,   // [nnode*8]
    __global const int4*    g_neighs,
    __global const int4*    g_bkNeighs,
    __global const float4*  g_apars,
    __global const float4*  g_bLs,
    __global const float4*  g_bKs,
    __global const float4*  g_Ksp,
    __global const float4*  g_Kpp,
    __global const float4*  g_constr,
    __global const float4*  g_constrK,
    __global const float4*  g_MDparams,
    // --- FAF (used when do_faf!=0) ---
    __global const float*   g_folded_coeffs,   // [ntypes*nbasis] flat
    __global const float4*  g_folded_kxyz,     // [nbasis]
    __global const int*     g_folded_atom_type,// [natoms]
    const int4              folded_meta,       // (nbasis, ntypes, 0, 0)
    const float4            folded_lvec2d      // (ax,bx,ay,by)
){
    const int natoms = nDOFs.x;
    const int nnode  = nDOFs.y;
    const int nsteps = nDOFs.z;
    const int do_faf = nDOFs.w;
    const int nvec   = natoms + nnode;
    const int iL     = get_local_id(0);
    const int nL     = get_local_size(0);

    const float4 MDpars = g_MDparams[0];
    const float dt   = MDpars.x;
    const float damp = MDpars.y;
    const float Flimit = MDpars.z;

    // FAF local cache
    __local float4 LBASIS[FAF_BASIS_MAX];
    __local float  LCOEFFS[FAF_TYPES_MAX * FAF_BASIS_MAX];
    float4 invLvec2d = (float4)(0,0,0,0);
    int nbasis = 0, ntypes = 0;
    if(do_faf){
        nbasis = folded_meta.x;
        ntypes = folded_meta.y;
        if(nbasis>0 && nbasis<=FAF_BASIS_MAX && ntypes>0 && ntypes<=FAF_TYPES_MAX){
            for(int j=iL; j<nbasis; j+=nL) LBASIS[j] = g_folded_kxyz[j];
            for(int j=iL; j<nbasis*ntypes; j+=nL) LCOEFFS[j] = g_folded_coeffs[j];
            float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
            float det = ax*by - bx*ay;
            if(fabs(det) > 1e-12f) invLvec2d = (float4)(by/det, -bx/det, -ay/det, ax/det);
            else nbasis = 0;
        } else { nbasis = 0; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #define NNEIGH 4

    for(int step = 0; step < nsteps; step++){
        // Phase 1: zero forces
        for(int i = iL; i < nvec;    i += nL) g_aforce[i] = float4Zero;
        for(int i = iL; i < nnode*8; i += nL) g_fneigh[i] = float4Zero;
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Phase 2: bonded SPFF (strided over nodes)
        for(int iG = iL; iG < nnode; iG += nL){
            float4  hs[4];
            float3  fbs[4];
            float3  fps[4];
            float3  fa  = float3Zero;
            float   E   = 0;
            const int4  ng  = g_neighs[iG];
            const float3 pa = g_apos[iG].xyz;
            const float4 par = g_apars[iG];
            const int*  ings = (int*)&ng;
            for(int i=0; i<NNEIGH; i++){ fbs[i]=float3Zero; fps[i]=float3Zero; }
            float3 fpi = float3Zero;
            {
                const float4 vbL = g_bLs[iG];
                const float4 vbK = g_bKs[iG];
                const float4 vKs = g_Ksp[iG];
                const float4 vKp = g_Kpp[iG];
                const float* bL   = (float*)&vbL;
                const float* bK   = (float*)&vbK;
                const float* Kspi = (float*)&vKs;
                const float* Kppi = (float*)&vKp;
                const float3 hpi  = g_apos[natoms + iG].xyz;
                for(int i=0; i<NNEIGH; i++){
                    float4 h;
                    const int ing = ings[i];
                    if(ing<0) break;
                    h.xyz = g_apos[ing].xyz - pa;
                    float l = length(h.xyz);
                    h.w = 1.f/l;
                    h.xyz *= h.w;
                    hs[i] = h;
                    if(iG < ing){
                        float3 f1;
                        E += evalBond(h.xyz, l-bL[i], bK[i], &f1);
                        fbs[i] -= f1; fa += f1;
                        float kpp = Kppi[i];
                        if((ing < nnode) && (kpp > 1e-6f)){
                            float3 f1p, f2p;
                            const float3 hpi_j = g_apos[natoms + ing].xyz;
                            E += evalPiAling(hpi, hpi_j, kpp, &f1p, &f2p);
                            fpi += f1p; fps[i] += f2p;
                        }
                    }
                    float ksp = Kspi[i];
                    if(ksp > 1e-6f){
                        float3 f1, f2;
                        E += evalAngCos((float4){hpi,1.f}, h, ksp, par.w, &f1, &f2);
                        fpi += f1; fa -= f2; fbs[i] += f2;
                    }
                }
                const int i4p = iG*4 + nnode*4;
                for(int i=0; i<NNEIGH; i++) g_fneigh[i4p+i] = (float4){fps[i], 0};
                g_aforce[natoms + iG] = (float4){fpi, 0};
            }
            for(int i=0; i<NNEIGH; i++){
                int ing = ings[i];
                if(ing<0) break;
                const float4 hi = hs[i];
                for(int j=i+1; j<NNEIGH; j++){
                    int jng = ings[j];
                    if(jng<0) break;
                    const float4 hj = hs[j];
                    float3 f1, f2;
                    E += evalAngleCosHalf(hi, hj, par.xy, par.z, &f1, &f2);
                    fa -= f1 + f2;
                    fbs[i] += f1;
                    fbs[j] += f2;
                }
            }
            const int i4 = iG*4;
            for(int i=0; i<NNEIGH; i++) g_fneigh[i4+i] = (float4){fbs[i], 0};
            g_aforce[iG] += (float4){fa, E};
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Phase 2b: optional FAF substrate
        if(do_faf && nbasis>0){
            for(int iG = iL; iG < natoms; iG += nL){
                float3 pos = g_apos[iG].xyz;
                float u = invLvec2d.x*pos.x + invLvec2d.y*pos.y;
                float v = invLvec2d.z*pos.x + invLvec2d.w*pos.y;
                u = u - floor(u);
                v = v - floor(v);
                int ityp = g_folded_atom_type[iG];
                if(ityp < 0 || ityp >= ntypes) continue;
                float E = 0.0f;
                float3 F = (float3)(0.0f,0.0f,0.0f);
                int ioff = ityp*nbasis;
                for(int ib=0; ib<nbasis; ib++){
                    float c = LCOEFFS[ioff + ib];
                    float4 prm = LBASIS[ib];
                    float  b = folded_eval_basis_g(u, v, pos.z, prm);
                    float3 g = folded_eval_grad_g(u, v, pos.z, prm, invLvec2d);
                    E += c * b;
                    F -= c * g;
                }
                g_aforce[iG] += (float4)(F.x, F.y, F.z, -E);
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }

        // Phase 3: gather + integrate (strided over nvec)
        for(int iG = iL; iG < nvec; iG += nL){
            float4 fe = g_aforce[iG];
            const bool bPi = (iG >= natoms);
            const int4 ngs = g_bkNeighs[iG];
            float4 frec = float4Zero;
            if(ngs.x>=0) frec += g_fneigh[ngs.x];
            if(ngs.y>=0) frec += g_fneigh[ngs.y];
            if(ngs.z>=0) frec += g_fneigh[ngs.z];
            if(ngs.w>=0) frec += g_fneigh[ngs.w];
            fe += frec;
            if(Flimit > 0){
                float fr2 = dot(fe.xyz, fe.xyz);
                if(fr2 > Flimit*Flimit) fe.xyz *= Flimit / sqrt(fr2);
            }
            g_aforce[iG] = fe;
            if(iG < natoms){
                float4 cons = g_constr[iG];
                if(cons.w > 0){
                    float4 cK = g_constrK[iG];
                    cK = max(cK, (float4){0,0,0,0});
                    float4 pe_c = g_apos[iG];
                    fe.xyz += (cons.xyz - pe_c.xyz) * cK.xyz;
                }
            }
            float4 ve = g_avel[iG];
            float4 pe = g_apos[iG];
            if(bPi){
                fe.xyz += pe.xyz * (-dot(pe.xyz, fe.xyz));
                ve.xyz += pe.xyz * (-dot(pe.xyz, ve.xyz));
            }
            float inv_mass = (pe.w > 1e-8f) ? (1.f / pe.w) : 1.f;
            ve.xyz *= damp;
            ve.xyz += fe.xyz * dt * inv_mass;
            pe.xyz += ve.xyz * dt;
            if(bPi) pe.xyz = normalize(pe.xyz);
            ve.w = 0;
            g_avel[iG] = ve;
            g_apos[iG] = pe;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
