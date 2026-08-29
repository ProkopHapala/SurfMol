// getNonBond_generic.cl — 3-axis macro-assembler template for non-bonded kernels.
//
// This template is assembled by ClAssembler. The three axes are:
//   Axis 1 — NB_PAIR_FORCE(dp, REQK, R2damp)  : pairwise potential
//   Axis 2 — NB_EXCL_ARGS / NB_EXCL_SETUP / NB_EXCL_TEST / NB_EXCL_PBC_TEST
//                                            : exclusion strategy
//   Axis 3 — SURF_ARGS / SURF_INJECT(posi, REQKi, fe) : surface interaction
//
// The assembler injects #define aliases (via //<<<macro NB_VARIANT_DEFINES)
// that map the generic names to the chosen variant, e.g.:
//   #define NB_PAIR_FORCE(dp,REQK,R2damp)  NB_PAIR_LJQH(dp,REQK,R2damp)
//   #define NB_EXCL_ARGS                   NB_EXCL_ARGS_NEIGHS4
//   #define SURF_ARGS                      SURF_ARGS_GRIDFF_BSPLINE
//   #define SURF_INJECT(posi,REQKi,fe)     SURF_INJECT_GRIDFF_BSPLINE(posi,REQKi,fe)
//
// Fragment files (nb_common.cl, gridff_eval.cl, faf_eval.cl) define the
// variant macros. The template includes the needed fragments via //<<<file.
//
// Reference: FireCore UFF.cl:getNonBond (neighs4, no surface)
//            FireCore UFF.cl:getNonBond_GridFF_Bspline (neighs4 + GridFF)
// This template reproduces both by choosing the appropriate axis variants.

//<<<file common.cl
//<<<file Forces.cl
//<<<file nb_common.cl
//<<<file gridff_eval.cl
//<<<file faf_eval.cl

// --- Variant aliases (injected by assembler) ---
//<<<macro NB_VARIANT_DEFINES

// ======================================================================
//  getNonBond_generic — assembled non-bonded + surface kernel
// ======================================================================
__attribute__((reqd_work_group_size(32,1,1)))
__kernel void getNonBond_generic(
    const int4 ns,                  // 1 // (natoms,nnode) dimensions
    // Dynamical
    __global float4*  atoms,        // 2 // positions of atoms
    __global float4*  forces,       // 3 // forces on atoms
    // Parameters
    __global float4*  REQKs,        // 4 // non-bonded parameters (RvdW,EvdW,Q,H)
    NB_EXCL_ARGS,                   // 5+ // exclusion-strategy-specific args
    __global cl_Mat3* lvecs,        //     lattice vectors for each system
    const int4        nPBC,         //     number of PBC images in each direction
    const float4      GFFParams     //     forcefield params (R2damp, alphaMorse, ...)
    SURF_ARGS                       //     surface-strategy-specific args (can be empty)
){
    __local float4 LATOMS[32];   // local buffer for atom positions
    __local float4 LCLJS [32];   // local buffer for atom parameters

    const int iG = get_global_id  (0);
    const int nG = get_global_size(0);
    const int iS = get_global_id  (1);
    const int nS = get_global_size(1);
    const int iL = get_local_id   (0);
    const int nL = get_local_size (0);

    const int natoms = ns.x;
    const int i0a = iS*natoms;
    const int iaa = iG + i0a;

    if( (DBG_UFF>0) && (iG==0) && (iS==0) ){
        printf("GPU ENTER getNonBond_generic() natoms=%d nS=%d nG=%d\n", natoms, nS, nG);
    }

    // --- Exclusion setup (Axis 2) ---
    NB_EXCL_SETUP(iaa);

    const bool   bPBC  = (nPBC.x+nPBC.y+nPBC.z)>0;
    const float4 REQKi = REQKs    [iaa];
    const float3 posi  = atoms    [iaa].xyz;
    const float  R2damp = GFFParams.x*GFFParams.x;
    float4 fe          = float4Zero;

    const cl_Mat3 lvec = lvecs[iS];

    const float3 shift0  = lvec.a.xyz*-nPBC.x + lvec.b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;
    const float3 shift_a = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);
    const float3 shift_b = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);

    // ========= Atom-to-Atom interaction (N-body, chunked by local memory)
    for (int j0=0; j0<nG; j0+=nL){
        const int i=j0+iL;
        if(i<natoms){
            LATOMS[iL] = atoms [i+i0a];
            LCLJS [iL] = REQKs [i+i0a];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int jl=0; jl<nL; jl++){
            const int ja=j0+jl;
            if( (ja!=iG) && (ja<natoms) ){
                const float4 aj = LATOMS[jl];
                float4 REQK = mixREQ_arithmetic( REQKi, LCLJS [jl] );
                float3 dp   = aj.xyz - posi;

                const bool bBonded = NB_EXCL_TEST(ja);

                if(bPBC){
                    int ipbc=0;
                    dp += shift0;
                    for(int iy=0; iy<3; iy++){
                        for(int ix=0; ix<3; ix++){
                            if( !( bBonded && NB_EXCL_PBC_TEST(ja,ipbc) )){
                                float4 fij = NB_PAIR_FORCE( dp, REQK, R2damp );
                                if((DBG_UFF>3) && (iG==IDBG_ATOM) && (iS==IDBG_SYS)){ printf("GPU fij(% .6e,% .6e,% .6e) E % .6e REQij(% .6e,% .6e,% .6e) bBonded %i bPBC %i\n", fij.x, fij.y, fij.z, fij.w, REQK.x, REQK.y, REQK.z, bBonded, bPBC); }
                                fe += fij;
                            }
                            ipbc++;
                            dp += lvec.a.xyz;
                        }
                        dp += shift_a;
                    }
                }else
                if( !bBonded ){
                    float4 fij = NB_PAIR_FORCE( dp, REQK, R2damp );
                    if((DBG_UFF>2) && (iG==IDBG_ATOM) && (iS==IDBG_SYS)){
                        printf("GPU fij [i:%3i,j:%3i|isys:%i] dp(% .6e,% .6e,% .6e) r % .6e REQi(% .6e,% .6e,% .6e) REQij(% .6e,% .6e,% .6e) fij(% .6e,% .6e,% .6e) E % .6e bBonded %i bPBC %i\n", iG, ja, iS, dp.x, dp.y, dp.z, length(dp),  REQKi.x, REQKi.y, REQKi.z, REQK.x, REQK.y, REQK.z,  fij.x, fij.y, fij.z, fij.w, bBonded, bPBC );
                    }
                    fe += fij;
                }
            }
        }
        //barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(iG>=natoms) return;

    // ========= Surface injection (Axis 3)
    SURF_INJECT(posi, REQKi, fe);

    forces[iaa] += fe;
}
