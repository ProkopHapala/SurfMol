// ======================================================================
// getNonBond_reference.cl
// ======================================================================
// VERBATIM copies of FireCore kernels, kept as reference for the
// macro-assembler template in getNonBond_generic.cl.
//
// Source: FireCore cpp/common_resources/cl/UFF.cl
//   - getNonBond()               (UFF.cl:1023-1204)  — NB only, neighs4 excl
//   - getNonBond_GridFF_Bspline() (UFF.cl:1523-1717) — NB + GridFF surface
//
// These are NOT compiled directly. They exist so we can diff against the
// macro-assembled output to verify correctness.
// ======================================================================

// ======================================================================
//                           getNonBond()
// ======================================================================
// Calculate non-bonded forces on atoms (icluding both node atoms and capping atoms), cosidering periodic boundary conditions
// It calculate Lenard-Jones, Coulomb and Hydrogen-bond forces between all atoms in the system
// it can be run in parallel for multiple systems, in order to efficiently use number of GPU cores (for small systems with <100 this is essential to get good performance)
// This is the most time consuming part of the forcefield evaluation, especially for large systems when nPBC>1
__attribute__((reqd_work_group_size(32,1,1)))
__kernel void getNonBond(
    const int4 ns,                  // 1 // (natoms,nnode) dimensions of the system
    // Dynamical
    __global float4*  atoms,        // 2 // positions of atoms  (including node atoms [0:nnode] and capping atoms [nnode:natoms] and pi-orbitals [natoms:natoms+nnode] )
    __global float4*  forces,       // 3 // forces on atoms
    // Parameters
    __global float4*  REQKs,        // 4 // non-bonded parameters (RvdW,EvdW,QvdW,Hbond)
    __global int4*    neighs,       // 5 // neighbors indices      ( to ignore interactions between bonded atoms )
    __global int4*    neighCell,    // 6 // neighbors cell indices ( to know which PBC image should be ignored  due to bond )
    __global cl_Mat3* lvecs,        // 7 // lattice vectors for each system
    const int4        nPBC,         // 8 // number of PBC images in each direction (x,y,z)
    const float4      GFFParams     // 9 // Grid-Force-Field parameters
){

    // we use local memory to store atomic position and parameters to speed up calculation, the size of local buffers should be equal to local workgroup size
    //__local float4 LATOMS[2];
    //__local float4 LCLJS [2];
    //__local float4 LATOMS[4];
    //__local float4 LCLJS [4];
    //__local float4 LATOMS[8];
    //__local float4 LCLJS [8];
    //__local float4 LATOMS[16];
    //__local float4 LCLJS [16];
    __local float4 LATOMS[32];   // local buffer for atom positions
    __local float4 LCLJS [32];   // local buffer for atom parameters
    //__local float4 LATOMS[64];
    //__local float4 LCLJS [64];
    //__local float4 LATOMS[128];
    //__local float4 LCLJS [128];

    const int iG = get_global_id  (0); // index of atom
    const int nG = get_global_size(0); // number of atoms
    const int iS = get_global_id  (1); // index of system
    const int nS = get_global_size(1); // number of systems
    const int iL = get_local_id   (0); // index of atom in local memory
    const int nL = get_local_size (0); // number of atoms in local memory

    const int natoms=ns.x;  // number of atoms
    const int i0a = iS*natoms;  // index of first atom in atoms array
    const int iaa = iG + i0a; // index of atom in atoms array

    if( (DBG_UFF>0) && (iG==0) && (iS==0) ){
        printf("GPU ENTER getNonBond() natoms=%d nS=%d nG=%d\n", natoms, nS, nG);
    }

    if((DBG_UFF>1) && (iG==IDBG_ATOM)&&(iS==IDBG_SYS)){  printf( "GPU::getNonBond() natoms(%i) nS,nG,nL(%i,%i,%i) \n", natoms, nS,nG,nL ); }
    if((DBG_UFF>1) && (iG==IDBG_ATOM)&&(iS==IDBG_SYS)){
       printf( "GPU::getNonBond() natoms(%i) nS,nG,nL(%i,%i,%i) \n", natoms, nS,nG,nL );
       for(int is=0; is<nS; is++){
            cl_Mat3 lvec = lvecs[is];
            printf( "GPU[is:%i]   lvec(%6.3f,%6.3f,%6.3f)(%6.3f,%6.3f,%6.3f)(%6.3f,%6.3f,%6.3f) \n", is, lvec.a.x,lvec.a.y,lvec.a.z,  lvec.b.x,lvec.b.y,lvec.b.z,   lvec.c.x,lvec.c.y,lvec.c.z  );
            for(int ia=0; ia<natoms; ia++){
                int i = ia+is*natoms;
                printf( "GPU atom [is:%i,ia:%i] p(% .3f,% .3f,% .3f,% .3f) REQ(% .3f,% .3f,% .3f,% .3f) neighs{%3i,%3i,%3i,%3i} \n", is,ia, atoms[i].x,atoms[i].y,atoms[i].z,atoms[i].w, REQKs[i].x,REQKs[i].y,REQKs[i].z,REQKs[i].w, neighs[i].x,neighs[i].y,neighs[i].z,neighs[i].w );
            }
       }
    }


    //if( iL==0 ){ for(int i=0; i<nL; i++){  LATOMS[i]=(float4){ 10000.0, (float)i,(float)iG,(float)iS }; LCLJS[i]=(float4){ 20000.0, (float)i,(float)iG,(float)iS }; } }


    // NOTE: if(iG>=natoms) we are reading from invalid adress => last few processors produce crap, but that is not a problem
    //       importaint is that we do not write this crap to invalid address, so we put   if(iG<natoms){forces[iav]+=fe;} at the end
    //       we may also put these if(iG<natoms){ .. } around more things, but that will unnecessarily slow down other processors
    //       we need these processors with (iG>=natoms) to read remaining atoms to the local memory.

    //if(iG<natoms){
    //const bool   bNode = iG<nnode;   // All atoms need to have neighbors !!!!
    const bool   bPBC  = (nPBC.x+nPBC.y+nPBC.z)>0;  // PBC is used if any of the PBC dimensions is >0
    //const bool bPBC=false;

    const int4   ng    = neighs   [iaa];  // neighbors indices
    const int4   ngC   = neighCell[iaa];  // neighbors cell indices
    const float4 REQKi = REQKs    [iaa];  // non-bonded parameters
    const float3 posi  = atoms    [iaa].xyz; // position of atom
    const float  R2damp = GFFParams.x*GFFParams.x; // squared damping radius
    float4 fe          = float4Zero;  // force on atom

    const cl_Mat3 lvec = lvecs[iS]; // lattice vectors for this system

    //if(iG==0){ printf("GPU[iS=%i] lvec{%6.3f,%6.3f,%6.3f}{%6.3f,%6.3f,%6.3f}{%6.3f,%6.3f,%6.3f} \n", iS, lvec.a.x,lvec.a.y,lvec.a.z,  lvec.b.x,lvec.b.y,lvec.b.z,   lvec.c.x,lvec.c.y,lvec.c.z );  }

    //if(iG==0){ for(int i=0; i<natoms; i++)printf( "GPU[%i] ng(%i,%i,%i,%i) REQ(%g,%g,%g) \n", i, neighs[i].x,neighs[i].y,neighs[i].z,neighs[i].w, REQKs[i].x,REQKs[i].y,REQKs[i].z ); }

    const float3 shift0  = lvec.a.xyz*-nPBC.x + lvec.b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;   // shift of PBC image 0
    const float3 shift_a = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);                      // shift of PBC image in the inner loop
    const float3 shift_b = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);                      // shift of PBC image in the outer loop
    //}

    /*
    if((iG==iG_DBG)&&(iS==iS_DBG)){
        printf( "GPU::getNonBond() natoms,nnode,nvec(%i,%i,%i) nS,nG,nL(%i,%i,%i) bPBC=%i nPBC(%i,%i,%i)\n", natoms,nnode,nvec, nS,nG,nL, bPBC, nPBC.x,nPBC.y,nPBC.z );
        for(int i=0; i<natoms; i++){
            printf( "GPU a[%i] ", i);
            printf( "p{%6.3f,%6.3f,%6.3f} ", atoms[i0v+i].x,atoms[i0v+i].y,atoms[i0v+i].z  );
            printf( "ng{%i,%i,%i,%i} ", neighs[i0a+i].x,neighs[i0a+i].y,neighs[i0a+i].z,neighs[i0a+i].w );
            printf( "ngC{%i,%i,%i,%i} ", neighCell[i0a+i].x,neighCell[i0a+i].y,neighCell[i0a+i].z,neighCell[i0a+i].w );
            printf( "\n");
        }
    }
    */

    // ========= Atom-to-Atom interaction ( N-body problem ), we do it in chunks of size of local memory, in order to reuse data and reduce number of reads from global memory
    //barrier(CLK_LOCAL_MEM_FENCE);
    for (int j0=0; j0<nG; j0+=nL){      // loop over all atoms in the system, by chunks of size of local memory
        const int i=j0+iL;              // index of atom in local memory
        if(i<natoms){                   // j0*nL may be larger than natoms, so we need to check if we are not reading from invalid address
            LATOMS[iL] = atoms [i+i0a]; // read atom position to local memory
            LCLJS [iL] = REQKs [i+i0a]; // read atom parameters to local memory
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // wait until all atoms are read to local memory
        for (int jl=0; jl<nL; jl++){    // loop over all atoms in local memory (like 32 atoms)
            const int ja=j0+jl;         // index of atom in global memory
            if( (ja!=iG) && (ja<natoms) ){   // if atom is not the same as current atom and it is not out of range,  // ToDo: Should atom interact with himself in PBC ?
                const float4 aj = LATOMS[jl];    // read atom position   from local memory
                float4 REQK = mixREQ_arithmetic( REQKi, LCLJS [jl] );   // read atom parameters from local memory
                float3 dp       = aj.xyz - posi; // vector between atoms
                //if((iG==44)&&(iS==0))printf( "[i=%i,ja=%i/%i,j0=%i,jl=%i/%i][iG/nG/na %i/%i/%i] aj(%g,%g,%g,%g) REQ(%g,%g,%g,%g)\n", i,ja,nG,j0,jl,nL,   iG,nG,natoms,   aj.x,aj.y,aj.z,aj.w,  REQK.x,REQK.y,REQK.z,REQK.w  );

                // Debug print for pairwise interaction - print all interactions for atom 0
                //if((DBG_UFF!=0) && (iG==IDBG_ATOM) && (iS==IDBG_SYS)){ printf("GPU pair [%i,%i] dp(% .6e,% .6e,% .6e) r % .6e REQi(% .6e,% .6e,% .6e) REQij(% .6e,% .6e,% .6e) \n", iG, ja, dp.x, dp.y, dp.z, length(dp), REQKi.x,REQKi.y,REQKi.z,  REQK.x,REQK.y,REQK.z); }

                //REQK.x  +=REQKi.x;   // mixing rules for vdW Radius
                //REQK.yz *=REQKi.yz;  // mixing rules for vdW Energy
                //REQK = mixREQ_arithmetic( REQKi, REQK );

                const bool bBonded = ((ja==ng.x)||(ja==ng.y)||(ja==ng.z)||(ja==ng.w));
                //if( (j==0)&&(iG==0) )printf( "pbc NONE dp(%g,%g,%g)\n", dp.x,dp.y,dp.z );
                //if( (ji==1)&&(iG==0) )printf( "2 non-bond[%i,%i] bBonded %i\n",iG,ji,bBonded );

                if(bPBC){         // ===== if PBC is used, we need to loop over all PBC images of the atom
                    int ipbc=0;   // index of PBC image
                    dp += shift0; // shift to PBC image 0
                    // Fixed PBC size
                    for(int iy=0; iy<3; iy++){
                        for(int ix=0; ix<3; ix++){
                            //if( (ji==1)&&(iG==0)&&(iS==0) )printf( "GPU ipbc %i(%i,%i) shift(%7.3g,%7.3g,%7.3g)\n", ipbc,ix,iy, shift.x,shift.y,shift.z );
                            // Without these IF conditions if(bBonded) time of evaluation reduced from 61 [ms] to 51[ms]
                            if( !( bBonded &&(                     // if atoms are bonded, we do not want to calculate non-covalent interaction between them
                                    ((ja==ng.x)&&(ipbc==ngC.x))    // check if this PBC image is not the same as one of the bonded atoms
                                    ||((ja==ng.y)&&(ipbc==ngC.y))  // i.e. if ja is neighbor of iG, and ipbc is its neighbor cell index then we skip this interaction
                                    ||((ja==ng.z)&&(ipbc==ngC.z))
                                    ||((ja==ng.w)&&(ipbc==ngC.w))
                            ))){
                                //fe += getMorseQ( dp+shifts, REQK, R2damp );
                                //if( (ji==1)&&(iG==0)&&(iS==0) )printf( "ipbc %i(%i,%i) shift(%g,%g,%g)\n", ipbc,ix,iy, shift.x,shift.y,shift.z );
                                float4 fij = getLJQH( dp, REQK, R2damp );  // calculate non-bonded force between atoms using LJQH potential
                                //if((iG==iG_DBG)&&(iS==iS_DBG)){  printf( "GPU_LJQ[%i,%i|%i] fj(%g,%g,%g) R2damp %g REQ(%g,%g,%g) r %g \n", iG,ji,ipbc, fij.x,fij.y,fij.z, R2damp, REQK.x,REQK.y,REQK.z, length(dp+shift)  ); }

                                // Debug print for force calculation - print all interactions for atom 0
                                if((DBG_UFF>3) && (iG==IDBG_ATOM) && (iS==IDBG_SYS)){ printf("GPU fij(% .6e,% .6e,% .6e) E % .6e REQij(% .6e,% .6e,% .6e) bBonded %i bPBC %i\n", fij.x, fij.y, fij.z, fij.w, REQK.x, REQK.y, REQK.z, bBonded, bPBC);  }

                                fe += fij;
                            }
                            ipbc++;
                            dp    += lvec.a.xyz;
                        }
                        dp    += shift_a;
                    }
                }else                                                  // ===== if PBC is not used, it is much simpler
                if( !bBonded ){
                    float4 fij = getLJQH( dp, REQK, R2damp );  // calculate non-bonded force between atoms using LJQH potential

                    // Debug print for force calculation (non-PBC case) - print all interactions for atom 0
                    if((DBG_UFF>2) && (iG==IDBG_ATOM) && (iS==IDBG_SYS)){
                        printf("GPU fij [i:%3i,j:%3i|isys:%i] dp(% .6e,% .6e,% .6e) r % .6e REQi(% .6e,% .6e,% .6e) REQij(% .6e,% .6e,% .6e) fij(% .6e,% .6e,% .6e) E % .6e bBonded %i bPBC %i\n", iG, ja, iS, dp.x, dp.y, dp.z, length(dp),  REQKi.x, REQKi.y, REQKi.z, REQK.x, REQK.y, REQK.z,  fij.x, fij.y, fij.z, fij.w, bBonded, bPBC );
                    }

                    fe += fij;
                }  // if atoms are not bonded, we calculate non-bonded interaction between them
            }
        }
        //barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(iG<natoms){
        //if(iS==0){ printf( "GPU::getNonBond(iG=%i) fe(%g,%g,%g,%g)\n", iG, fe.x,fe.y,fe.z,fe.w ); }
        forces[iaa] += fe;        // If we don't run it as first forcefield, we need to add force to existing force
        //forces[iav] += fe;        // If we don't run it as first forcefield, we need to add force to existing force
        //forces[iav] = fe*(-1.f);
    }
}


// ======================================================================
//                           getNonBond_GridFF_Bspline()
// ======================================================================
// VERBATIM from FireCore UFF.cl:1523-1717
// Non-bonded + GridFF (tricubic B-spline) surface interaction.
// This is the reference for the SURF_INJECT_GRIDFF_BSPLINE macro.
// ======================================================================
__attribute__((reqd_work_group_size(32,1,1)))
__kernel void getNonBond_GridFF_Bspline(
    const int4 ns,                  // 1 // dimensions of the system (natoms,nnode,nvec)
    // Dynamical
    __global float4*  atoms,        // 2 // positions of atoms
    __global float4*  forces,       // 3 // forces on atoms
    // Parameters
    __global float4*  REQKs,        // 4 // parameters of Lenard-Jones potential, Coulomb and Hydrogen Bond (RvdW,EvdW,Q,H)
    __global int4*    neighs,       // 5 // indexes neighbors of atoms
    __global int4*    neighCell,    // 6 // indexes of cells of neighbor atoms
    __global cl_Mat3* lvecs,        // 7 // lattice vectors of the system
    const int4 nPBC,                // 8 // number of PBC images in each direction
    const float4  GFFParams,        // 9 // parameters of Grid-Force-Field (GFF) (RvdW,EvdW,Q,H)
    // GridFF
    __global float4*  BsplinePLQ,   // 10 // Grid-Force-Field (GFF) for Pauli repulsion
    const int4     grid_ns,         // 11 // origin of the grid
    const float4   grid_invStep,    // 12 // origin of the grid
    const float4   grid_p0          // 13 // origin of the grid
){
    __local float4 LATOMS[32];         // local memory chumk of positions of atoms
    __local float4 LCLJS [32];         // local memory chumk of atom parameters
    const int iG = get_global_id  (0); // index of atom in the system
    const int iS = get_global_id  (1); // index of system
    const int iL = get_local_id   (0); // index of atom in the local memory chunk
    const int nG = get_global_size(0); // total number of atoms in the system
    const int nS = get_global_size(1); // total number of systems
    const int nL = get_local_size (0); // number of atoms in the local memory chunk

    const int natoms=ns.x;         // number of atoms in the system

    //const int i0n = iS*nnode;    // index of the first node in the system
    const int i0a = iS*natoms;     // index of the first atom in the system
    //const int ian = iG + i0n;    // index of the atom in the system
    const int iaa = iG + i0a;      // index of the atom in the system

    const float4 REQKi = REQKs    [iaa];           // parameters of Lenard-Jones potential, Coulomb and Hydrogen Bond (RvdW,EvdW,Q,H) of the atom
    const float3 posi  = atoms    [iaa].xyz;       // position of the atom
    float4 fe          = float4Zero;              // forces on the atom

    if( (DBG_UFF>0) && (iG==0) && (iS==0) ){
        printf("GPU ENTER getNonBond_GridFF_Bspline() natoms=%d nS=%d nG=%d\n", natoms, nS, nG);
    }

    const int iS_DBG = 0;
    const int iG_DBG = 0;

    // =================== Non-Bonded interaction ( molecule-molecule )

    if(ns.w>=0)
    { // insulate nbff

    const cl_Mat3 lvec = lvecs[iS]; // lattice vectors of the system

    if((DBG_UFF>1) && (iG==IDBG_ATOM)&&(iS==IDBG_SYS)){  printf( "GPU::getNonBond_GridFF_Bspline() natoms(%i) nS,nG,nL(%i,%i,%i) \n", natoms, nS,nG,nL ); }
    //if((iG==iG_DBG)&&(iS==iS_DBG)){  printf( "GPU::getNonBond_GridFF_Bspline() natoms,nnode,nvec(%i,%i,%i) nS,nG,nL(%i,%i,%i) \n", natoms,nnode,nvec, nS,nG,nL ); }
    //if((iG==iG_DBG)&&(iS==iS_DBG)) printf( "GPU::getNonBond_GridFF_Bspline() nPBC_(%i,%i,%i) lvec (%g,%g,%g) (%g,%g,%g) (%g,%g,%g)\n", nPBC.x,nPBC.y,nPBC.z, lvec.a.x,lvec.a.y,lvec.a.z,  lvec.b.x,lvec.b.y,lvec.b.z,   lvec.c.x,lvec.c.y,lvec.c.z );
    // if((iG==iG_DBG)&&(iS==iS_DBG)){
    //     printf( "GPU::getNonBond_GridFF_Bspline() natoms,nnode,nvec(%i,%i,%i) nS,nG,nL(%i,%i,%i) \n", natoms,nnode,nvec, nS,nG,nL );
    //     for(int i=0; i<nS*nG; i++){
    //         int ia = i%nS;
    //         int is = i/nS;
    //         if(ia==0){ cl_Mat3 lvec = lvecs[is];  printf( "GPU[%i] lvec(%6.3f,%6.3f,%6.3f)(%6.3f,%6.3f,%6.3f)(%6.3f,%6.3f,%6.3f) \n", is, lvec.a.x,lvec.a.y,lvec.a.z,  lvec.b.x,lvec.b.y,lvec.b.z,   lvec.c.x,lvec.c.y,lvec.c.z  ); }
    //         //printf( "GPU[%i,%i] \n", is,ia,  );
    //     }
    // }

    //if(iG>=natoms) return;

    //const bool   bNode = iG<nnode;   // All atoms need to have neighbors !!!!
    const bool   bPBC  = (nPBC.x+nPBC.y+nPBC.z)>0; // Periodic boundary conditions if any of nPBC.x,nPBC.y,nPBC.z is non-zero
    const int4   ng    = neighs   [iaa];           // indexes of neighbors of the atom
    const int4   ngC   = neighCell[iaa];           // indexes of cells of neighbors of the atom

    const float  R2damp = GFFParams.x*GFFParams.x; // damping radius for Lenard-Jones potential

    //if(iG==0){ for(int i=0; i<natoms; i++)printf( "GPU[%i] ng(%i,%i,%i,%i) REQ(%g,%g,%g) \n", i, neighs[i].x,neighs[i].y,neighs[i].z,neighs[i].w, REQKs[i].x,REQKs[i].y,REQKs[i].z ); }

    const float3 shift0  = lvec.a.xyz*nPBC.x + lvec.b.xyz*nPBC.y + lvec.c.xyz*nPBC.z;  // shift of the first PBC image
    const float3 shift_a = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);                  // shift of lattice vector in the inner loop
    const float3 shift_b = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);                  // shift of lattice vector in the outer loop

    // ========= Atom-to-Atom interaction ( N-body problem )     - we do it by chunks of nL atoms in order to reuse data and reduce number of global memory reads
    for (int j0=0; j0<natoms; j0+= nL ){ // loop over atoms in the system by chunks of nL atoms which fit into local memory
        const int i = j0 + iL;           // global index of atom in the system
        LATOMS[iL] = atoms [i+i0a];      // load positions  of atoms into local memory
        LCLJS [iL] = REQKs [i+i0a];      // load parameters of atoms into local memory
        barrier(CLK_LOCAL_MEM_FENCE);    // wait until all atoms are loaded into local memory
        for (int jl=0; jl<nL; jl++){     // loop over atoms in the local memory chunk
            const int ja=jl+j0;          // global index of atom in the system
            if( (ja!=iG) && (ja<natoms) ){ // atom should not interact with himself, and should be in the system ( j0*nL+iL may be out of range of natoms )
                const float4 aj   = LATOMS[jl]; // position of the atom
                float4       REQK = LCLJS [jl]; // parameters of the atom
                float3 dp   = aj.xyz - posi;    // vector between atoms
                REQK = mixREQ_arithmetic( REQKi, REQK ); // Correct mixing
                
                // Debug print for pairwise interaction - print all interactions for atom 0
                if((DBG_UFF!=0) && (iG==IDBG_ATOM) && (iS==IDBG_SYS)){ printf("GPU pair [%i,%i] dp(% .6e,% .6e,% .6e) r % .6e REQi(% .6e,% .6e,% .6e) REQij(% .6e,% .6e,% .6e) \n", iG, ja, dp.x, dp.y, dp.z, length(dp), REQKi.x,REQKi.y,REQKi.z,  REQK.x,REQK.y,REQK.z); }
                
                const bool bBonded = ((ja==ng.x)||(ja==ng.y)||(ja==ng.z)||(ja==ng.w));   // atom is bonded if it is one of the neighbors
                if(bPBC){       // ==== with periodic boundary conditions we need to consider all PBC images of the atom
                    int ipbc=0; // index of PBC image
                    //if( (i0==0)&&(j==0)&&(iG==0) )printf( "pbc NONE dp(%g,%g,%g)\n", dp.x,dp.y,dp.z );
                    dp -= shift0;  // shift to the first PBC image
                    for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
                        for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                            for(int ix=-nPBC.x; ix<=nPBC.x; ix++){
                                if( !( bBonded &&(  // if bonded in any of PBC images, then we have to check both index of atom and index of PBC image to decide if we should skip this interaction
                                          ((ja==ng.x)&&(ipbc==ngC.x)) // check 1. neighbor and its PBC cell
                                        ||((ja==ng.y)&&(ipbc==ngC.y)) // check 2. neighbor and its PBC cell
                                        ||((ja==ng.z)&&(ipbc==ngC.z)) // ...
                                        ||((ja==ng.w)&&(ipbc==ngC.w)) // ...
                                ))){
                                    //fe += getMorseQ( dp+shifts, REQK, R2damp );
                                    float4 fij = getLJQH( dp, REQK, R2damp );  // calculate Lenard-Jones, Coulomb and Hydrogen-bond forces between atoms
                                    //if((iG==iG_DBG)&&(iS==iS_DBG)){  printf( "GPU_LJQ[%i,%i|%i] fj(%g,%g,%g) R2damp %g REQ(%g,%g,%g) r %g \n", iG,ji,ipbc, fij.x,fij.y,fij.z, R2damp, REQK.x,REQK.y,REQK.z, length(dp+shift)  ); }
                                    
                                    // Debug print for force calculation - print all interactions for atom 0
                                    if((DBG_UFF!=0) && (iG==IDBG_ATOM) && (iS==IDBG_SYS)){ printf("GPU fij(% .6e,% .6e,% .6e) E % .6e REQij(% .6e,% .6e,% .6e) bBonded %i bPBC %i\n", fij.x, fij.y, fij.z, fij.w, REQK.x, REQK.y, REQK.z, bBonded, bPBC);  }
                                    
                                    fe += fij; // accumulate forces
                                }
                                ipbc++;         // increment index of PBC image
                                dp+=lvec.a.xyz; // shift to the next PBC image
                            }
                            dp+=shift_a;        // shift to the next PBC image
                        }
                        dp+=shift_b;            // shift to the next PBC image
                    }
                }else{ //  ==== without periodic boundary it is much simpler, not need to care about PBC images
                    if(bBonded) continue;  // Bonded ?
                    float4 fij = getLJQH( dp, REQK, R2damp ); // calculate Lenard-Jones, Coulomb and Hydrogen-bond forces between atoms
                    
                    // Debug print for force calculation (non-PBC case) - print all interactions for atom 0
                    if((DBG_UFF!=0) && (iG==IDBG_ATOM) && (iS==IDBG_SYS)){
                        printf("GPU fij [i:%3i,j:%3i|isys:%i] dp(% .6e,% .6e,% .6e) r % .6e REQi(% .6e,% .6e,% .6e) REQij(% .6e,% .6e,% .6e) fij(% .6e,% .6e,% .6e) E % .6e bBonded %i bPBC %i\n", iG, ja, iS, dp.x, dp.y, dp.z, length(dp),  REQKi.x, REQKi.y, REQKi.z, REQK.x, REQK.y, REQK.z,  fij.x, fij.y, fij.z, fij.w, bBonded, bPBC );
                    }
                    
                    fe += fij;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // wait until all atoms are processed, ToDo: not sure if it is needed here ?
    }

    } // insulate nbff

    if(iG>=natoms) return; // natoms <= nG, because nG must be multiple of nL (loccal kernel size). We cannot put this check at the beginning of the kernel, because it will break reading of atoms to local memory

    // ========== Molecule-Grid interaction with GridFF using tricubic Bspline ================== (see. kernel sample3D_comb() in GridFF.cl

    __local int4 xqs[4];
    __local int4 yqs[4];
    { // insulate gridff
        if      (iL<4){             xqs[iL]=make_inds_pbc(grid_ns.x,iL); }
        else if (iL<8){ int i=iL-4; yqs[i ]=make_inds_pbc(grid_ns.y,i ); };
        //const float3 inv_dg = 1.0f / grid_d.xyz;
        barrier(CLK_LOCAL_MEM_FENCE);

        const float ej = exp( GFFParams.y * REQKi.x ); // exp(-alphaMorse*RvdW) pre-factor for factorized Morse potential
        const float4 PLQH = (float4){
            ej*ej*REQKi.y,                   // prefactor London dispersion (attractive part of Morse potential)
            ej*   REQKi.y,                   // prefactor Pauli repulsion   (repulsive part of Morse potential)
            REQKi.z,
            0.0f
        };
        //const float3 p = ps[iG].xyz;
        const float3 u = (posi - grid_p0.xyz) * grid_invStep.xyz;

        float4 fg = fe3d_pbc_comb(u, grid_ns.xyz, BsplinePLQ, PLQH, xqs, yqs);

        // GridFF-specific debug prints for test atom
        if((DBG_UFF>1) && (iG==IDBG_ATOM) && (iS==IDBG_SYS)){
            printf("GPU GridFF [i:%3i|isys:%i] posi(% .6e,% .6e,% .6e) u(% .6e,% .6e,% .6e) PLQH(% .6e,% .6e,% .6e,% .6e) fg_raw(% .6e,% .6e,% .6e,% .6e)\n",
                   iG, iS, posi.x, posi.y, posi.z, u.x, u.y, u.z, PLQH.x, PLQH.y, PLQH.z, PLQH.w, fg.x, fg.y, fg.z, fg.w);
        }

        //if((iG==iG_DBG)&&(iS==iS_DBG)){  printf( "GPU::getNonBond_GridFF_Bspline() fg(%g,%g,%g|%g) u(%g,%g,%g) posi(%g,%g,%g) grid_invStep(%g,%g,%g)\n", fg.x,fg.y,fg.z,fg.w,  u.x,u.y,u.z, posi.x,posi.y,posi.z, grid_invStep.x,grid_invStep.y,grid_invStep.z  ); }

        fg.xyz *= -grid_invStep.xyz;
        
        // Debug print for final GridFF force after scaling
        if((DBG_UFF>1) && (iG==IDBG_ATOM) && (iS==IDBG_SYS)){
            printf("GPU GridFF scaled [i:%3i|isys:%i] fg_scaled(% .6e,% .6e,% .6e,% .6e) grid_invStep(% .6e,% .6e,% .6e)\n",
                   iG, iS, fg.x, fg.y, fg.z, fg.w, grid_invStep.x, grid_invStep.y, grid_invStep.z);
        }
        
        fe += fg;
        //fes[iG] = fe;
    }  // insulate gridff

    forces[iaa] += fe;        // If we don't run it as first forcefield, we need to add force to existing force
    //forces[iav] += fe;     // If we don't run it as first forcefield, we need to add forces to the forces calculated by previous forcefields
    //forces[iav] = fe*(-1.f);


}
