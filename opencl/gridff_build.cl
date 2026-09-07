// gridff_build.cl — GridFF construction kernels (macro/fragment library).
//
// Extracted from gridff_spammm.cl. Contains build and utility kernels as
// //>>>function blocks. Requires common.cl + Forces.cl concatenated first.
// Assembled by oclff::ClAssembler. See doc/topical_audit/gridff_faf.md.

//>>>function BsplineConv3D (__kernel void BsplineConv3D()
__kernel void BsplineConv3D(
    const int4 ns,
    __global const float* Gs,
    __global const float* G0,
    __global       float* out,
    const float2 coefs
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);

    //if( (ix==0)&&(iy==0)&&(iz==0) ){ printf("GPU BsplineConv3D() ns{%i,%i,%i,%i}\n", ns.x,ns.y,ns.z,ns.w); }
    if( (ix>=ns.x) || (iy>=ns.y) || (iz>=ns.z) ) return;

    const float  B0 = 2.0f/3.0f;
    const float  B1 = 1.0f/6.0f;
    const float3 Bs = (float3){B0*B0, B0*B1, B1*B1 };

    // if( (ix==0) && (iy==0) && (iz==0) ) {  
    //     int4 ls=(int4){get_local_size(0), get_local_size(1), get_local_size(2),0};
    //     int4 gs=(int4){get_global_size(0), get_global_size(1), get_global_size(2),0};
    //     //printf("GPU BsplineConv3D() weights{%g,%g}  ns{%i,%i,%i,%i} coefs{%f,%f} \n", ns.x,ns.y,ns.z,ns.w, coefs.x, coefs.y, ); 
    //     printf("GPU BsplineConv3D ns{%i,%i,%i,%i} weights{%f,%f,%f,%f} coefs{%f,%f} G0=%p local_size{%i,%i,%i} global_size{%i,%i,%i}\n",
    //         ns.x, ns.y, ns.z, ns.w,
    //         B0*B0*B0, B0*B0*B1, B0*B1*B1, B1*B1*B1,
    //         coefs.x, coefs.y,
    //         G0, 
    //         ls.x, ls.y, ls.z,
    //         gs.x, gs.y, gs.z
    //     );
    // }
    
    const int3 ixs =  (int3){ modulo(ix-1,ns.x),  ix,   modulo(ix+1,ns.x)  };
    const int3 iys = ((int3){ modulo(iy-1,ns.y),  iy,   modulo(iy+1,ns.y)  })*ns.x;

    const int nxy = ns.x*ns.y;

    float val=0;
    const int iiz =iz*nxy;  val += conv3x3_pbc( Gs, Bs, iiz                    , ixs, iys ) * B0;
    if(iz>0     ){          val += conv3x3_pbc( Gs, Bs, modulo(iz-1, ns.z)*nxy , ixs, iys ) * B1; }
    if(iz<ns.z-1){          val += conv3x3_pbc( Gs, Bs, modulo(iz+1, ns.z)*nxy , ixs, iys ) * B1; }
    
    const int i = iiz + iys.y + ixs.y;
    val*=coefs.x;
    if (G0 != NULL) { val+=G0[i]*coefs.y; }
    out[i] =  val;

    // const int i = ix + ns.x*( iy + ns.y*iz);
    // // out[i] =  Gs[i];
    // // out[i] =  G0[i];
    // out[i] =  G0[i] - Gs[i];


}

//>>>function BsplineConv3D_tex (__kernel void BsplineConv3D_tex()
__kernel void BsplineConv3D_tex(
    const int4 ns,
    __read_only image3d_t Gs,
    __global const float* G0,
    __global       float* out    
) {

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    
    //if( (ix==0)&&(iy==0)&&(iz==0) ){ printf("GPU BsplineConv3D_tex() ns{%i,%i,%i,%i}\n", ns.x,ns.y,ns.z,ns.w); }
    if( (ix>=ns.x) || (iy>=ns.y) || (iz>=ns.z) ) return;

    const float  B0 = 2.0/3.0;
    const float  B1 = 1.0/6.0;
    const float3 Bs = (float3){B0*B0, B0*B1, B1*B1 };

    int4 coord = (int4){ix, iy, iz, 0};

    float          val  = conv_3x3_tex( samp_pbc, Gs, Bs, coord                  ) * B0;
    if(iz>0     ){ val += conv_3x3_tex( samp_pbc, Gs, Bs, coord-(int4){0,0,0,-1} ) * B1; } 
    if(iz<ns.z-1){ val += conv_3x3_tex( samp_pbc, Gs, Bs, coord-(int4){0,0,0, 1} ) * B1; }

    const int i = ix + ns.x * ( iy* + iz*ns.y );
    
    if (G0 != NULL) { val-=G0[i]; }
    out[i] =  val;

}

//>>>function Convolution3D_General (__kernel void Convolution3D_General()
__kernel void Convolution3D_General(
    const int4 ns,            // {nx, ny, nz, 0}

//>>>function addMul (__kernel void addMul()
__kernel void addMul(
    const int ntot,
    __global       float* a,
    __global const float* b,
    
    const float c
){
    const int i = get_global_id(0);
    if(i>=ntot) return;
    a[i]+=b[i]*c;
}

//>>>function dot_wg (__kernel void dot_wg()
__kernel void dot_wg(
    const int ntot,
    __global const float* a,
    __global const float* b,
    __global       float* partial
){
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    float acc = 0.0f;
    // just in case we want to run nG<ntot, that would decrease paralelism, but also less work for CPU to do the final reduction of partial sum
    for(int i=gid; i<ntot; i+=get_global_size(0)){
        acc += a[i]*b[i];
    }
    __local float s[64];
    s[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int step=lsz>>1; step>0; step>>=1){
        if(lid<step){ s[lid]+=s[lid+step]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid==0){
        partial[get_group_id(0)] = s[0];
    }
}

//>>>function setLinear (__kernel void setLinear()
__kernel void setLinear(
    const int ntot,
    __global       float* out,
    const float c1,
    __global const float* a1,
    const float c2,
    __global const float* a2
) {
    const int i = get_global_id(0);
    if( i >= ntot ) return;
    out[i] = c1 * a1[i] + c2 * a2[i];
}

//>>>function move (__kernel void move()
__kernel void move(
    const int  ntot,
    __global float* p,
    __global float* v,
    __global float* f,  
    const float4 MDpar
) {

    const int i = get_global_id(0);
    //if( i==0 ){ printf("GPU move() ntot=%i MDpar{%g,%g,%g,%g}\n", ntot,  MDpar.x, MDpar.y, MDpar.z,MDpar.w); }
    if (i > ntot ) return;

    // leap frog
    float vi =  v[i];
    float pi =  p[i];
    float fi  = f[i];

    vi *=    MDpar.z;
    vi += fi*MDpar.x;
    pi += vi*MDpar.y;

    v[i]=vi;
    p[i]=pi;
}

//>>>function setMul (__kernel void setMul()
__kernel void setMul(
    const int  ntot,
    __global float* v,
    __global float* out,  
    float c
) {
    const int i = get_global_id(0);
    //if( i==0 ){ printf("GPU move() ntot=%i MDpar{%g,%g,%g,%g}\n", ntot,  MDpar.x, MDpar.y, MDpar.z,MDpar.w); }
    if (i > ntot ) return;
    out[i] = v[i]*c;
}

//>>>function setCMul (__kernel void setCMul()
__kernel void setCMul(
    const int  ntot,
    __global float2* v,
    __global float* out,  
    float2 c
) {
    const int i = get_global_id(0);
    //if( i==0 ){ printf("GPU move() ntot=%i MDpar{%g,%g,%g,%g}\n", ntot,  MDpar.x, MDpar.y, MDpar.z,MDpar.w); }
    if (i > ntot ) return;
    out[i] = v[i].x*c.x + v[i].y*c.y;
}

//>>>function set (__kernel void set()
__kernel void set(
    const int  ntot,
    __global float* out,  
    float c
) {
    const int i = get_global_id(0);
    if (i > ntot ) return;
    out[i] = c;
}

//>>>function make_MorseFF (__kernel void make_MorseFF()
__kernel void make_MorseFF(
    const int nAtoms,                // 1
    __global const float4*  atoms,         // 2
    __global const float4*  REQs,          // 3
    __global float* E_Paul,         // 4
    __global float* E_Lond,         // 5
    //__global * FE_Coul,
    const int4     nPBC,             // 6
    const int4     nGrid,            // 7
    //const cl_Mat3  lvec,           
    const float4  lvec_a,            // 8
    const float4  lvec_b,            // 9
    const float4  lvec_c,            // 10
    const float4  grid_p0,           // 11
    const float4  GFFParams          // 12
){
    __local float4 LATOMS[32];
    __local float4 LCLJS [32];
    const int iG = get_global_id (0);
    const int nG = get_global_size(0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);
    const int nab = nGrid.x*nGrid.y;
    const int ia  =  iG%nGrid.x; 
    const int ib  = (iG%nab)/nGrid.x;
    const int ic  =  iG/nab; 

    const float  alphaMorse = GFFParams.y;
    const float  R2damp     = GFFParams.x*GFFParams.x;
    const float3 dGrid_a = lvec_a.xyz*(1.f/(float)nGrid.x);
    const float3 dGrid_b = lvec_b.xyz*(1.f/(float)nGrid.y);
    const float3 dGrid_c = lvec_c.xyz*(1.f/(float)nGrid.z); 
    const float3 shift_b = lvec_b.xyz + lvec_a.xyz*(nPBC.x*-2.f-1.f);      //  shift in scan(iy)
    const float3 shift_c = lvec_c.xyz + lvec_b.xyz*(nPBC.y*-2.f-1.f);      //  shift in scan(iz) 
    
    //if( (ia==0)&&(ib==0)&&(ic==0) ){  
    //     printf(  "GPU nAtoms %i alphaMorse(%g) R2damp(%g) \n", nAtoms, alphaMorse, R2damp );
    //       for(int ia=0; ia<nAtoms; ia++){printf(  "GPU atom[%i] pos(%8.4f,%8.4f,%8.4f|%8.4f) REQs (%16.8f,%16.8f,%16.8f,%16.8f) R2damp(%g) \n", ic,    atoms[ia].x, atoms[ia].y, atoms[ia].z, atoms[ia].w,    REQs[ia].x, REQs[ia].y, REQs[ia].z, REQs[ia].w );}
    //     for (int iz=0; iz<nGrid.z; iz++ ){
    //         const float3 pos    = grid_p0.xyz  + dGrid_a.xyz*ia      + dGrid_b.xyz*ib      + dGrid_c.xyz*iz;          // +  lvec_a.xyz*-nPBC.x + lvec_b.xyz*-nPBC.y + lvec_c.xyz*-nPBC.z;  // most negative PBC-cell
    //         int    ia   = 0;
    //         float4 REQK = REQs[ia];
    //         float3 dp   = pos - atoms[ia].xyz;
    //         float  r2  = dot(dp,dp);
    //         float  r   = sqrt(r2+1e-32 );
    //         // ---- Morse ( Pauli + Dispersion )
    //         float    e = exp( -alphaMorse*(r-REQK.x) );
    //         float   eM = REQK.y*e;
    //         //fe_Paul += eM * e;
    //         //fe_Lond += eM * -2.0f;
    //         printf( "GPU pos(%8.4f,%8.4f,%8.4f) iz=%i dp(%8.4f,%8.4f,%8.4f|r=%8.4f) e=%g EPaul=%g ELond=%g alphaMorse=%g R0=%g E0=%g \n", pos.x,pos.y,pos.z,  iz, dp.x,dp.y,dp.z, r, e, eM*e, eM*-2.0f,  alphaMorse, REQK.x, REQK.y );
    //     }
    //}
    //if( (ia==0)&&(ib==0) ){  printf(  "GPU ic %i nGrid(%i,%i,%i)\n", ic, nGrid.x,nGrid.y,nGrid.z );}

    const int nMax = nab*nGrid.z;
    if(iG>=nMax) return;

    const float3 pos    = grid_p0.xyz  + dGrid_a.xyz*ia      + dGrid_b.xyz*ib      + dGrid_c.xyz*ic       // grid point within cell
                                       +  lvec_a.xyz*-nPBC.x + lvec_b.xyz*-nPBC.y + lvec_c.xyz*-nPBC.z;  // most negative PBC-cell

    //const float3  shift0 = lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;
    float Paul = 0.0f;
    float Lond = 0.0f;
    //float4 fe_Coul = float4Zero;
    for (int j0=0; j0<nAtoms; j0+= nL ){
        const int i = j0 + iL;
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = REQs [i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int jl=0; jl<nL; jl++){
            const int ja=jl+j0;
            if( ja<nAtoms ){ 
                const float4 REQK =       LCLJS [jl];
                float3       dp   = pos - LATOMS[jl].xyz;
            
                //if( (i0==0)&&(j==0)&&(iG==0) )printf( "pbc NONE dp(%g,%g,%g)\n", dp.x,dp.y,dp.z ); 
                //dp+=lvec.a.xyz*-nPBC.x + lvec.b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;

                //float3 shift=shift0; 
                for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
                    for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                        for(int ix=-nPBC.x; ix<=nPBC.x; ix++){

                            //if( (i0==0)&&(j==0)&&(iG==0) )printf( "pbc[%i,%i,%i] dp(%g,%g,%g)\n", ix,iy,iz, dp.x,dp.y,dp.z );   
                            float  r2  = dot(dp,dp);
                            float  r   = sqrt(r2+1e-32f );
                            // ---- Electrostatic
                            //float ir2  = 1.f/(r2+R2damp); 
                            //float   E  = COULOMB_CONST*REQK.z*sqrt(ir2);
                            //fe_Coul   += (float4)(dp*(E*ir2), E );
                            // ---- Morse ( Pauli + Dispersion )
                            float    e = exp( -alphaMorse*(r-REQK.x) );
                            float   eM = REQK.y*e;
                            Paul += eM * e;
                            Lond += eM * -2.0f;

                            // if((iG==0)&&(j==0)){
                            //     //float3 sh = dp - pos + LCLJS[j].xyz + lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;
                            //     float3 sh = shift;
                            //     printf( "GPU(%2i,%2i,%2i) sh(%7.3f,%7.3f,%7.3f)\n", ix,iy,iz, sh.x,sh.y,sh.z  );
                            // }
                            //ipbc++; 
                            
                            dp   +=lvec_a.xyz;
                            //shift+=lvec.a.xyz;
                        }
                        dp   +=shift_b;
                        //shift+=shift_b;
                        //dp+=lvec.a.xyz*(nPBC.x*-2.f-1.f);
                        //dp+=lvec.b.xyz;
                    }
                    dp   +=shift_c;
                    //shift+=shift_c;
                    //dp+=lvec.b.xyz*(nPBC.y*-2.f-1.f);
                    //dp+=lvec.c.xyz;
                }

            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    E_Paul[iG] = Paul;
    E_Lond[iG] = Lond;
    //FE_Coul[iG] = fe_Coul;
    //int4 coord = (int4){ia,ib,ic,0};
    //write_imagef( FE_Paul, coord, (float4){pos,(float)iG} );
    //write_imagef( FE_Paul, coord, fe_Paul );
    //write_imagef( FE_Lond, coord, fe_Lond );
    //write_imagef( FE_Coul, coord, fe_Coul );
}

//>>>function make_MorseFF_f4 (__kernel void make_MorseFF_f4()
__kernel void make_MorseFF_f4(
    const int nAtoms,                // 1
    __global const float4*  atoms,         // 2
    __global const float4*  REQs,          // 3
    __global float4* FE_Paul,        // 4
    __global float4* FE_Lond,        // 5
    // __global float4* FE_Coul,
    const int4     nPBC,             // 6
    const int4     nGrid,            // 7
    const float4  lvec_a,            // 8
    const float4  lvec_b,            // 9
    const float4  lvec_c,            // 10
    const float4   grid_p0,          // 9
    const float4   GFFParams         // 10
){
 __local float4 LATOMS[32];
    __local float4 LCLJS [32];
    const int iG = get_global_id (0);
    const int nG = get_global_size(0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);
    const int nab = nGrid.x*nGrid.y;
    const int ia  =  iG%nGrid.x; 
    const int ib  = (iG%nab)/nGrid.x;
    const int ic  =  iG/nab; 

    const float  alphaMorse = GFFParams.y;
    const float  R2damp     = GFFParams.x*GFFParams.x;
    const float3 dGrid_a = lvec_a.xyz*(1.f/(float)nGrid.x);
    const float3 dGrid_b = lvec_b.xyz*(1.f/(float)nGrid.y);
    const float3 dGrid_c = lvec_c.xyz*(1.f/(float)nGrid.z); 
    const float3 shift_b = lvec_b.xyz + lvec_a.xyz*(nPBC.x*-2.f-1.f);      //  shift in scan(iy)
    const float3 shift_c = lvec_c.xyz + lvec_b.xyz*(nPBC.y*-2.f-1.f);      //  shift in scan(iz) 

    const int nMax = nab*nGrid.z;
    if(iG>=nMax) return;

    const float3 pos    = grid_p0.xyz  + dGrid_a.xyz*ia      + dGrid_b.xyz*ib      + dGrid_c.xyz*ic       // grid point within cell
                                       +  lvec_a.xyz*-nPBC.x + lvec_b.xyz*-nPBC.y + lvec_c.xyz*-nPBC.z;  // most negative PBC-cell

    //const float3  shift0 = lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;
    float4 fe_Paul = float4Zero;
    float4 fe_Lond = float4Zero;
    //float4 fe_Coul = float4Zero;
    for (int j0=0; j0<nAtoms; j0+= nL ){
        const int i = j0 + iL;
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = REQs [i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int jl=0; jl<nL; jl++){
            const int ja=jl+j0;
            if( ja<nAtoms ){ 
                const float4 REQK =       LCLJS [jl];
                float3       dp   = pos - LATOMS[jl].xyz;
            
                //if( (i0==0)&&(j==0)&&(iG==0) )printf( "pbc NONE dp(%g,%g,%g)\n", dp.x,dp.y,dp.z ); 
                //dp+=lvec.a.xyz*-nPBC.x + lvec.b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;

                //float3 shift=shift0; 
                for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
                    for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                        for(int ix=-nPBC.x; ix<=nPBC.x; ix++){

                            //if( (i0==0)&&(j==0)&&(iG==0) )printf( "pbc[%i,%i,%i] dp(%g,%g,%g)\n", ix,iy,iz, dp.x,dp.y,dp.z );   
                            float  r2  = dot(dp,dp);
                            float  r   = sqrt(r2+1e-32f );
                            // ---- Electrostatic
                            //float ir2  = 1.f/(r2+R2damp); 
                            //float   E  = COULOMB_CONST*REQK.z*sqrt(ir2);
                            //fe_Coul   += (float4)(dp*(E*ir2), E );
                            // ---- Morse ( Pauli + Dispersion )
                            float    e = exp( -alphaMorse*(r-REQK.x) );
                            float   eM = REQK.y*e;
                            float   de = 2.f*alphaMorse*eM/r;
                            float4  fe = (float4)( dp*de, eM );
                            fe_Paul += fe * e;
                            fe_Lond += fe * (float4)( -1.0f,-1.0f,-1.0f, -2.0f );

                            // if((iG==0)&&(j==0)){
                            //     //float3 sh = dp - pos + LCLJS[j].xyz + lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;
                            //     float3 sh = shift;
                            //     printf( "GPU(%2i,%2i,%2i) sh(%7.3f,%7.3f,%7.3f)\n", ix,iy,iz, sh.x,sh.y,sh.z  );
                            // }
                            //ipbc++; 
                            
                            dp   +=lvec_a.xyz;
                            //shift+=lvec.a.xyz;
                        }
                        dp   +=shift_b;
                        //shift+=shift_b;
                        //dp+=lvec.a.xyz*(nPBC.x*-2.f-1.f);
                        //dp+=lvec.b.xyz;
                    }
                    dp   +=shift_c;
                    //shift+=shift_c;
                    //dp+=lvec.b.xyz*(nPBC.y*-2.f-1.f);
                    //dp+=lvec.c.xyz;
                }

            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    FE_Paul[iG] = fe_Paul;
    FE_Lond[iG] = fe_Lond;
    //FE_Coul[iG] = fe_Coul;

    //int4 coord = (int4){ia,ib,ic,0};
    //write_imagef( FE_Paul, coord, (float4){pos,(float)iG} );
    //write_imagef( FE_Paul, coord, fe_Paul );
    //write_imagef( FE_Lond, coord, fe_Lond );
    //write_imagef( FE_Coul, coord, fe_Coul );
}

//>>>function make_Coulomb_points (__kernel void make_Coulomb_points()
__kernel void make_Coulomb_points(
    const int nAtoms,                // 1
    const int np,                    // 2
    __global const float4*  atoms,   // 3
    __global const float4*  ps,      // 4
    __global       float4*  FE_Coul, // 5
    const int4     nPBC,             // 6
    const float4   lvec_a,            // 8
    const float4   lvec_b,            // 9
    const float4   lvec_c,            // 10
    const float4   GFFParams         // 9
){
    __local float4 LATOMS[32];
    const int iG = get_global_id (0);
    //const int nG = get_global_size(0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);

    //const float  alphaMorse = GFFParams.y;
    const float  R2damp     = GFFParams.x*GFFParams.x;
    const float3 shift_b = lvec_b.xyz + lvec_a.xyz*(nPBC.x*-2.f-1.f);      //  shift in scan(iy)
    const float3 shift_c = lvec_c.xyz + lvec_b.xyz*(nPBC.y*-2.f-1.f);      //  shift in scan(iz) 
    
    if(iG>=np) return;

    // if( iG==0 ){
    //     printf( "GPU make_Coulomb_points() nAtoms=%i np=%i nPBC(%i,%i,%i)\n", nAtoms, np, nPBC.x,nPBC.y,nPBC.z );
    //     printf( "GPU make_Coulomb_points() lvec_a(%8.4f,%8.4f,%8.4f) lvec_b(%8.4f,%8.4f,%8.4f) lvec_c(%8.4f,%8.4f,%8.4f)\n", lvec_a.x,lvec_a.y,lvec_a.z,   lvec_b.x,lvec_b.y,lvec_b.z,   lvec_c.x,lvec_c.y,lvec_c.z  );
    //     for(int i=0; i<nAtoms; i++){ printf( "GPU atom[%i] (%8.4f,%8.4f,%8.4f|%8.4f)\n", i, atoms[i].x,atoms[i].y,atoms[i].z,atoms[i].w ); }
    //     //for(int i=0; i<np; i++){ printf( "GPU ps[%i] (%8.4f,%8.4f,%8.4f)\n", i, ps[i].x,ps[i].y,ps[i].z ); }
    // }

    const float3 pos    = ps[iG].xyz +  lvec_a.xyz*-nPBC.x + lvec_b.xyz*-nPBC.y + lvec_c.xyz*-nPBC.z;  // most negative PBC-cell

    float4 fe_Coul = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 c       = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int j0=0; j0<nAtoms; j0+= nL ){
        const int i = j0 + iL;
        LATOMS[iL] = atoms[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int jl=0; jl<nL; jl++){
            const int ja=jl+j0;
            if( ja<nAtoms ){ 
                const float4 atom = LATOMS[jl];
                float3       dp   = pos - atom.xyz;
        
                //float3 shift=shift0; 
                for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
                    for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                        for(int ix=-nPBC.x; ix<=nPBC.x; ix++){

                            //if( (i0==0)&&(j==0)&&(iG==0) )printf( "pbc[%i,%i,%i] dp(%g,%g,%g)\n", ix,iy,iz, dp.x,dp.y,dp.z );   
                            const float  r2  = dot(dp,dp);
                            const float ir2  = 1.f/(r2+R2damp); 
                            const float ir   = sqrt(ir2 );
                            const float   E  = COULOMB_CONST*atom.w*ir;

                            const float4 fei = (float4)(dp*(E*ir2), E );   

                            // Kahan Summation to reduce numerical iaccuracy ( https://en.wikipedia.org/wiki/Kahan_summation_algorithm )
                            const float4 y = fei - c;
                            const float4 t = fe_Coul + y;
                            c              = t - fe_Coul - y;
                            fe_Coul        = t;

                            dp   +=lvec_a.xyz;
                        }
                        dp   +=shift_b;
                    }
                    dp   +=shift_c;
                }

            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //FE_Paul[iG] = fe_Paul;
    //FE_Lond[iG] = fe_Lond;
    FE_Coul[iG] = fe_Coul;
}

//>>>function project_atom_on_grid_cubic_pbc (__kernel void project_atom_on_grid_cubic_pbc()
__kernel void project_atom_on_grid_cubic_pbc(
    const int na,                   // 1 number of atoms
    __global const float4* atoms,   // 2 Atom positions and charges
    __global       float*  Qgrid,   // 3 Output grid
    const int4 ng,                  // 4 grid size
    const float3 g0,                // 5 grid orgin
    const float3 dg                 // 6 grid dimensions
) {
    int iG = get_global_id(0);
    const int iL = get_local_id(0);
    if (iG >= na) return;

    __local int4 xqs[4];
    __local int4 yqs[4];
    __local int4 zqs[4];
    if      (iL<4 ){             xqs[iL]=make_inds_pbc(ng.x,iL); }
    else if (iL<8 ){ int i=iL-4; yqs[i ]=make_inds_pbc(ng.y,i ); }
    else if (iL<12){ int i=iL-8; yqs[i ]=make_inds_pbc(ng.y,i ); };
    barrier(CLK_LOCAL_MEM_FENCE);


    // Load atom position and charge
    float4 atom = atoms[iG];
    //float3 pos  = (float3)(atom_data.x, atom_data.y, atom_data.z);
    //float charge = atom_data.w;

    // Convert to grid coordinates
    float3      g = (atom.xyz - g0) / dg;
    int3       gi = (int3  ){(int)g.x, (int)g.y, (int)g.z};
    if(g.x<0) gi.x--;
    if(g.y<0) gi.y--;
    if(g.z<0) gi.z--;
    float3 t      = (float3){     g.x - gi.x, g.y - gi.y, g.z - gi.z};

    // Compute weights for cubic B-spline interpolation
    float wx[4], wy[4], wz[4];
    Bspline_basis(t.x, wx);
    Bspline_basis(t.y, wy);
    Bspline_basis(t.z, wz);

    const int nxy = ng.x * ng.y;
    // Pre-calculate periodic boundary condition indices for each dimension
    gi.x=modulo(gi.x-1,ng.x); const int4 xq = choose_inds_pbc_3(gi.x, ng.x, xqs );  const int* xq_ = (int*)&xq;
    gi.y=modulo(gi.y-1,ng.y); const int4 yq = choose_inds_pbc_3(gi.y, ng.y, yqs );  const int* yq_ = (int*)&xq;
    gi.z=modulo(gi.z-1,ng.z); const int4 zq = choose_inds_pbc_3(gi.z, ng.z, zqs );  const int* zq_ = (int*)&xq;

    //float4 Bspline_dbasis();

    for (int dz = 0; dz < 4; dz++) {
        const int gz  = zq_[dz];
        const int iiz = gz * nxy;
        for (int dy = 0; dy < 4; dy++) {
            const int gy = yq_[dy];
            const int iiy = iiz + gy * ng.x;
            const float qbyz = atom.w * wy[dy] * wz[dz];
            for (int dx = 0; dx < 4; dx++) {
                const int gx = xq_[dx];
                const int ig = gx + iiy;
                float qi = qbyz * wx[dx];
                Qgrid[ig] += qi;
            }
        }
    }

}

//>>>function project_atoms_on_grid_quintic_pbc (__kernel void project_atoms_on_grid_quintic_pbc()
__kernel void project_atoms_on_grid_quintic_pbc(
    const int na,                   // 1 number of atoms
    __global const float4* atoms,   // 2 Atom positions and charges
    __global       float2* Qgrid,   // 3 Output grid (complex, in order to be compatible with poisson)
    const int4   ng,                // 4 Grid size
    const float4 g0,                // 5 Grid origin
    const float4 dg                 // 6 Grid dimensions
) {
    int       iG = get_global_id(0);
    const int iL = get_local_id(0);
    
    // Declare and initialize shared memory for periodic boundary condition indices
    __local int xqs[6][6];
    __local int yqs[6][6];
    __local int zqs[6][6];
    if      (iL<6 ) { const int i=iL;    make_inds_pbc_5(ng.x,i,xqs[i]); }
    else if (iL<12) { const int i=iL-6;  make_inds_pbc_5(ng.y,i,yqs[i]); }
    else if (iL<18) { const int i=iL-12; make_inds_pbc_5(ng.z,i,zqs[i]); }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (iG >= na) return;

    // if( iG==0 ){
    //     printf("GPU project_atoms_on_grid_quintic_pbc() ng(%i,%i,%i) g0(%g,%g,%g) dg(%g,%g,%g) \n", ng.x,ng.y,ng.z,   g0.x,g0.y,g0.z,   dg.x,dg.y,dg.z );
    //     for(int i=0; i<6; i++){ int* q=xqs[i]; printf("GPU xqs[0](%4i,%4i,%4i,%4i,%4i,%4i) \n", q[0],  q[1], q[2], q[3], q[4], q[5] ); }
    //     for(int i=0; i<6; i++){ int* q=yqs[i]; printf("GPU yqs[0](%4i,%4i,%4i,%4i,%4i,%4i) \n", q[0],  q[1], q[2], q[3], q[4], q[5] ); }
    //     for(int i=0; i<6; i++){ int* q=zqs[i]; printf("GPU zqs[0](%4i,%4i,%4i,%4i,%4i,%4i) \n", q[0],  q[1], q[2], q[3], q[4], q[5] ); }
    //     for(int ia=0; ia<na; ia++){ 
    //         float4 atom = atoms[ia];
    //         float3 g    = (atom.xyz - g0.xyz) / dg.xyz;
    //         int3   gi   = (int3  ){(int)g.x,(int)g.y,(int)g.z};
    //         if(g.x<0) gi.x--;
    //         if(g.y<0) gi.y--;
    //         if(g.z<0) gi.z--;
    //         printf("GPU atom[%i]  gi(%3i,%3i,%3i) (%8.4f,%8.4f,%8.4f |%8.4f) \n", ia, gi.x,gi.y,gi.z,  atoms[ia].x, atoms[ia].y, atoms[ia].z, atoms[ia].w ); 
    //     }
    //     int ia = 0;
    //     float4 atom = atoms[ia];
    //     float3 g    = (atom.xyz - g0.xyz) / dg.xyz;
    //     int3   gi   = (int3  ){(int)g.x,(int)g.y,(int)g.z};
    //     if(g.x<0) gi.x--;
    //     if(g.y<0) gi.y--;
    //     if(g.z<0) gi.z--;
    //     float3 t    = (float3){g.x-gi.x, g.y-gi.y, g.z-gi.z};
    //     printf( "GPU g(%g,%g,%g) gi(%i,%i,%i) t(%g,%g,%g)\n", g.x,g.y,g.z, gi.x,gi.y,gi.z, t.x,t.y,t.z );
    //     // Compute weights for quintic B-spline interpolation
    //     float bx[6], by[6], bz[6];
    //     Bspline_basis5(t.x, bx);
    //     Bspline_basis5(t.y, by);
    //     Bspline_basis5(t.z, bz);
    //     const int nxy = ng.x * ng.y;
    //     int xq[6];
    //     int yq[6];
    //     int zq[6];
    //     // Pre-calculate periodic boundary condition indices for each dimension
    //     gi.x = modulo( gi.x-2, ng.x ); choose_inds_pbc_5(gi.x,ng.x, xqs, xq );
    //     gi.y = modulo( gi.y-2, ng.y ); choose_inds_pbc_5(gi.y,ng.y, yqs, yq );
    //     gi.z = modulo( gi.z-2, ng.z ); choose_inds_pbc_5(gi.z,ng.z, zqs, zq );
    //     for (int dz = 0; dz < 6; dz++) {
    //         const int gz    = zq[dz];
    //         const int iiz   = gz * nxy;
    //         const float qbz = atom.w * bz[dz];
    //         printf( "GPU dz[%i] gz[%i] qbz %g t(%g,%g,%g)\n", dz, gz, qbz, t.x,t.y,t.z );
    //     }
    // }

    // Load atom position and charge
    float4 atom = atoms[iG];
    float3 g    = (atom.xyz - g0.xyz) / dg.xyz;
    int3   gi   = (int3  ){(int)g.x,(int)g.y,(int)g.z};
    if(g.x<0) gi.x--;
    if(g.y<0) gi.y--;
    if(g.z<0) gi.z--;
    float3 t    = (float3){g.x-gi.x, g.y-gi.y, g.z-gi.z};

    // Compute weights for quintic B-spline interpolation
    float bx[6], by[6], bz[6];
    Bspline_basis5(t.x, bx);
    Bspline_basis5(t.y, by);
    Bspline_basis5(t.z, bz);

    const int nxy = ng.x * ng.y;
    
    int xq[6];
    int yq[6];
    int zq[6];
    // Pre-calculate periodic boundary condition indices for each dimension
    gi.x = modulo( gi.x-2, ng.x ); choose_inds_pbc_5(gi.x,ng.x, xqs, xq );
    gi.y = modulo( gi.y-2, ng.y ); choose_inds_pbc_5(gi.y,ng.y, yqs, yq );
    gi.z = modulo( gi.z-2, ng.z ); choose_inds_pbc_5(gi.z,ng.z, zqs, zq );

    for (int dz = 0; dz < 6; dz++) {
        const int gz    = zq[dz];
        const int iiz   = gz * nxy;
        const float qbz = atom.w * bz[dz];
        for (int dy = 0; dy < 6; dy++) {
            const int gy  = yq[dy];
            const int iiy = iiz + gy * ng.x;
            const float qbyz =  by[dy] * qbz;
            for (int dx = 0; dx < 6; dx++) {
                const int gx = xq[dx];
                const int ig = gx + iiy;
                float qi = qbyz * bx[dx];
                //Qgrid[ig].x += qi;
                Qgrid[ig] = (float2){qi,0.0f};
            }
        }
    }
    //const int ig = gi.z*nxy + gi.y*ng.x + gi.x;
    //Qgrid[ig] = (float2){gi.y*1.0f,0.0f};
}

//>>>function poissonW_old (__kernel void poissonW_old()
__kernel void poissonW_old(
    const int4   ns,         // (nx,ny,nz,nxyz)
    __global float2* rho_k,  // input array  rho(k) - fourier coefficients (complex)
    __global float2* V_k,    // output array V(k)   - fourier coefficients (complex)
    const float4 coefs       // (0,0,0, 4*pi*eps0*dV)
){    
    const int iG = get_global_id (0);
    //if(iG==0){  printf("GPU poissonW() ns(%i,%i,%i,%i) coefs(%g,%g,%g,%g) \n", ns.x,ns.y,ns.z,ns.w, coefs.x,coefs.y,coefs.z,coefs.w ); }
    if(iG>=ns.w) return;
    const int nab = ns.x*ns.y;
    const int ix  =  iG%ns.x; 
    const int iy  = (iG%nab)/ns.x;
    const int iz  =  iG/nab; 
    float4 k = (float4){ ix/(0.5f*ns.x), iy/(0.5f*ns.y), iz/(0.5f*ns.z), 0};
    k = 1.0f-fabs(k-1.0f); 
    float  f = coefs.w/dot( k, k );    // dCell.w = 4*pi*eps0*dV - rescaling constant
    if(iG==0)f=0;
    if(iG<ns.w){ 
        V_k[iG] = rho_k[iG]*f;
    }
};

//>>>function poissonW (__kernel void poissonW()
__kernel void poissonW(
    const int4   ns,         // (nx, ny, nz, nxyz)
    __global float2* rho_k,  // input array  rho(k) - Fourier coefficients (complex)
    __global float2* V_k,    // output array V(k)   - Fourier coefficients (complex)
    const float4 coefs,      // (freq_x, freq_y, freq_z, amp)
    const float4 params      // (gauss_a, bDivideByK2, bNormalizeGauss, unused)
){
    const int iG = get_global_id(0);
    if (iG >= ns.w) return;
    const int nx = ns.x;
    const int ny = ns.y;
    const int nz = ns.z;
    const int nab = nx * ny;
    const int ix = iG % nx;
    const int iy = (iG % nab) / nx;
    const int iz = iG / nab;

    const int nx2 = nx / 2;
    const int ny2 = ny / 2;
    const int nz2 = nz / 2;

    const float freq_x = coefs.x;
    const float freq_y = coefs.y;
    const float freq_z = coefs.z;

    const float kx = ((ix <= nx2) ? ix : ix - nx) * freq_x;
    const float ky = ((iy <= ny2) ? iy : iy - ny) * freq_y;
    const float kz = ((iz <= nz2) ? iz : iz - nz) * freq_z;

    const float k2 = kx * kx + ky * ky + kz * kz;

    float f = coefs.w;
    if (params.x > 0.0f) {
        f *= exp(-params.x * k2);
    }
    if (params.y > 0.5f) {
        f = (k2 > 1e-32f) ? (f / k2) : 0.0f;
    } else if ((k2 <= 1e-32f) && (params.x <= 0.0f) && (fabs(coefs.w - 1.0f) < 1e-8f)) {
        f = 1.0f;
    }

    V_k[iG] = rho_k[iG] * f;
}

//>>>function laplace_real_pbc (__kernel void laplace_real_pbc()
__kernel void laplace_real_pbc( 
    int4 ng,
    __global const float* Vin, 
    __global       float* Vout, 
    __global       float* vV, 
    float cSOR, 
    float cV
){
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    if( (ix>=ng.x) || (iy>=ng.y) || (iz>=ng.z) ) return;

    //if( (ix==0) && (iy==0) && (iz==0) ){ printf( "GPU laplace_real_pbc() global_sz(%i,%i,%i) ns(%i,%i,%i) cSOR=%g cV=%g @vV=%li \n ",  (int)get_global_size(0), (int)get_global_size(1), (int)get_global_size(2), ng.x, ng.y, ng.z, cSOR, cV, (long)vV  ); }

    int nxy = ng.x * ng.y;

    const int iiz =          iz       *nxy;
    const int ifz =  pbc_ifw(iz, ng.z)*nxy;
    const int ibz =  pbc_ibk(iz, ng.z)*nxy;
    
    const int iiy =          iy       *ng.x;
    const int ify =  pbc_ifw(iy, ng.y)*ng.x;
    const int iby =  pbc_ibk(iy, ng.y)*ng.x;
    const int ifx =  pbc_ifw(ix, ng.x);
    const int ibx =  pbc_ibk(ix, ng.x);

    float vi = 
    Vin[ ibx + iiy + iiz ] + Vin[ ifx + iiy + iiz ] + 
    Vin[ ix  + iby + iiz ] + Vin[ ix  + ify + iiz ] + 
    Vin[ ix  + iiy + ibz ] + Vin[ ix  + iiy + ifz ];

    const float fac = 1.0f/6.0f;
    vi *= fac;
    
    const int i = ix + iiy + iiz;

    const float vo = Vin[ i ];
    vi += (vi-vo)*cSOR; 
    if(vV != 0){   // inertia
        //if( (ix==0) && (iy==0) && (iz==0) ){ printf( "GPU laplace_real_pbc() @vV=%li \n ", (long)vV );}
        float v = vi - vo;                 // velocity ( change between new and old potential )
        v       = v*cV + vV[i]*(1.0f-cV);  // inertia ( mixing of new and old change )
        vV[i]   = v;                       // store updated velocity ( change )
        vi      = v + vo;                  // new potantial corrected by intertia
    }

    Vout[i] = vi;
    //Vout[i] = vo;
    // double v = V_[i]-V[i];
    // if(iter>0){ v = v*cV + vV[i]*(1-cV); }
    // vV[i] = v; 
    // V_[i] = V[i] + v;

}

//>>>function slabPotential (__kernel void slabPotential()
__kernel void slabPotential( 
    int4 ng,
    __global const float*  Vin,   // 1
    __global       float*  Vout,  // 2
    float4 params                 // 3 (dz, Vol, dVcor, Vcor0)          
){
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    if( (ix>=ng.x) || (iy>=ng.y) || (iz>=ng.w) ) return;

    const float dz    = params.x;
    const float dVcor = params.z;
    const float Vcor0 = params.w;
    const float Vcor_z = Vcor0 + dVcor * (iz*dz);

    const int nz_ = ng[2] + ng[3];
    //const int j = ix + ng.x*(iy + ng.y*(nz_-iz) );   // We found that the potential is inverted in z-direction ( maybe also x,y ? )
    const int j = (ng[0]-ix-1) + ng.x*( (ng[1]-iy-1) + ng.y*(nz_-iz-1) );  // maybe is is inverted also x,y ?

    const int i = ix + ng.x*(iy + ng.y*iz);

    Vout[i] = Vin[j] + Vcor_z;
    //Vout[i] = Vin[i] + Vcor_z;
}

//>>>function slabPotential_zyx (__kernel void slabPotential_zyx()
__kernel void slabPotential_zyx( 
    int4 ng,
    __global const float*  Vin,   // 1
    __global       float*  Vout,  // 2
    float4 params                 // 3 (dz, Vol, dVcor, Vcor0)          
){
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    if( (ix>=ng.x) || (iy>=ng.y) || (iz>=ng.w) ) return;

    const float dz    = params.x;
    const float dVcor = params.z;
    const float Vcor0 = params.w;
    const float Vcor_z = Vcor0 + dVcor * (iz*dz);

    const int nz_ = ng[2] + ng[3];
    //const int j = ix + ng.x*(iy + ng.y*(nz_-iz) );   // We found that the potential is inverted in z-direction ( maybe also x,y ? )
    const int j = (ng[0]-ix-1) + ng.x*( (ng[1]-iy-1) + ng.y*(nz_-iz-1) );  // maybe is is inverted also x,y ?

    //const int i = ix + ng.x*(iy + ng.y*iz);
    const int i = iz + ng.z*(iy + ng.y*ix);

    Vout[i] = Vin[j] + Vcor_z;
    //Vout[i] = Vin[i] + Vcor_z;
}

//>>>function make_GridFF (__kernel void make_GridFF()
__kernel void make_GridFF(
    const int nAtoms,                // 1
    __global float4*  atoms,         // 2
    __global float4*  REQs,          // 3
    __write_only image3d_t  FE_Paul, // 4
    __write_only image3d_t  FE_Lond, // 5
    __write_only image3d_t  FE_Coul, // 6
    const int4     nPBC,             // 7
    const int4     nGrid,            // 8
    const cl_Mat3  lvec,             // 9
    const float4   grid_p0,          // 10
    const float4   GFFParams         // 11
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

    const float  alphaMorse = GFFParams.y;
    const float  R2damp     = GFFParams.x*GFFParams.x;
    const float3 dGrid_a = lvec.a.xyz*(1.f/(float)nGrid.x);
    const float3 dGrid_b = lvec.b.xyz*(1.f/(float)nGrid.y);
    const float3 dGrid_c = lvec.c.xyz*(1.f/(float)nGrid.z);
    const float3 shift_b = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);      //  shift in scan(iy)
    const float3 shift_c = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);      //  shift in scan(iz)

    /*
    if(iG==0){printf("GPU:make_GridFF() nL=%i,nG=%i,nAtoms=%i,nPBC(%i,%i,%i) Rdamp %g alphaMorse %g \n", nL, nG, nAtoms, nPBC.x,nPBC.y,nPBC.z, GFFParams.x, alphaMorse );}
    if(iG==0){printf("GPU:make_GridFF() p0{%6.3f,%6.3f,%6.3f} lvec{{%6.3f,%6.3f,%6.3f},{%6.3f,%6.3f,%6.3f},{%6.3f,%6.3f,%6.3f}} \n", grid_p0.x,grid_p0.y,grid_p0.z,  lvec.a.x,lvec.a.y,lvec.a.z, lvec.b.x,lvec.b.y,lvec.b.z, lvec.c.x,lvec.c.y,lvec.c.z );}
    //if(iG==0){printf("GPU::make_GridFF(nAtoms=%i) \n", nAtoms );}
    if(iG==0){
        printf( "GPU_GFF_z #i   z  Ep_Paul Fz_Paul   Ep_Lond Fz_Lond  E_Coul Fz_Coul\n");
        for(int ic=0; ic<nGrid.z; ic++){
            const float3 pos_    = grid_p0.xyz  + dGrid_a.xyz*ia      + dGrid_b.xyz*ib      + dGrid_c.xyz*ic;  // grid point within cell
            const float3 pos     = pos_ + lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;       // most negative PBC-cell
            //const float3  shift0 = lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;
            float4 fe_Paul = float4Zero;
            float4 fe_Lond = float4Zero;
            float4 fe_Coul = float4Zero;
            for (int ja=0; ja<nAtoms; ja++ ){
                const float4 REQ  =       REQs[ja];
                float3       dp   = pos - atoms[ja].xyz;
                for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
                    for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                        for(int ix=-nPBC.x; ix<=nPBC.x; ix++){
                            float  r2  = dot(dp,dp);
                            float  r   = sqrt(r2 + 1e-32 );
                            float ir2  = 1.f/(r2+R2damp  );
                            // ---- Electrostatic
                            float   E  = COULOMB_CONST*REQ.z*sqrt(ir2);
                            fe_Coul   += (float4)(dp*(E*ir2), E );
                            // ---- Morse ( Pauli + Dispersion )
                            float    e = exp( -alphaMorse*(r-REQ.x) );
                            float   eM = REQ.y*e;
                            float   de = 2.f*alphaMorse*eM/r;
                            float4  fe = (float4)( dp*de, eM );
                            fe_Paul += fe * e;
                            fe_Lond += fe * (float4)( -1.0f,-1.0f,-1.0f, -2.0f );
                            dp   +=lvec.a.xyz;
                        }
                        dp   +=shift_b;
                    }
                    dp   +=shift_c;
                }
            }
            //printf(  "FE(RvdW[%i]) Paul(%g,%g,%g|%g) Lond(%g,%g,%g|%g) Coul(%g,%g,%g|%g)  \n", ia0, fe_Paul.x,fe_Paul.y,fe_Paul.z,fe_Paul.w,   fe_Lond.x,fe_Lond.y,fe_Lond.z,fe_Lond.w,    fe_Coul.x,fe_Coul.y,fe_Coul.z,fe_Coul.w  );
            //printf(  "%i %8.3f  %g %g    %g %g    %g %g  \n", ia0, dp.x, fe_Paul.x,fe_Paul.w,   fe_Lond.x,fe_Lond.w,    fe_Coul.x,fe_Coul.w  );
            //printf(  "%i %8.3f  %g %g %g %g %g   %g %g %g %g %g  \n", ia0, dp.x,  ELJ, fetot.w, fe_Paul.w,fe_Lond.w,fe_Coul.w*REQK.z,   FLJ, fetot.x, fe_Paul.x,fe_Lond.x,fe_Coul.x*REQK.z  );
            printf(  "GPU_GFF_z %3i %8.3f    %14.6f %14.6f    %14.6f %14.6f    %14.6f %14.6f\n", ic, pos.z, fe_Paul.w,fe_Paul.z, fe_Lond.w,fe_Lond.z,  fe_Coul.w,fe_Coul.z  );
        }
    }
    */

    const int nMax = nab*nGrid.z;
    if(iG>=nMax) return;

    const float3 pos    = grid_p0.xyz  + dGrid_a.xyz*ia      + dGrid_b.xyz*ib      + dGrid_c.xyz*ic       // grid point within cell
                                       +  lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;  // most negative PBC-cell

    //const float3  shift0 = lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;
    float4 fe_Paul = float4Zero;
    float4 fe_Lond = float4Zero;
    float4 fe_Coul = float4Zero;
    for (int j0=0; j0<nAtoms; j0+= nL ){
        const int i = j0 + iL;
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = REQs [i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int jl=0; jl<nL; jl++){
            const int ja=jl+j0;
            if( ja<nAtoms ){
                const float4 REQK =       LCLJS [jl];
                float3       dp   = pos - LATOMS[jl].xyz;

                //if( (i0==0)&&(j==0)&&(iG==0) )printf( "pbc NONE dp(%g,%g,%g)\n", dp.x,dp.y,dp.z );
                //dp+=lvec.a.xyz*-nPBC.x + lvec.b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;

                //float3 shift=shift0;
                for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
                    for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                        for(int ix=-nPBC.x; ix<=nPBC.x; ix++){

                            //if( (i0==0)&&(j==0)&&(iG==0) )printf( "pbc[%i,%i,%i] dp(%g,%g,%g)\n", ix,iy,iz, dp.x,dp.y,dp.z );
                            float  r2  = dot(dp,dp);
                            float  r   = sqrt(r2+1e-32 );
                            float ir2  = 1.f/(r2+R2damp);
                            // ---- Electrostatic
                            float   E  = COULOMB_CONST*REQK.z*sqrt(ir2);
                            fe_Coul   += (float4)(dp*(E*ir2), E );
                            // ---- Morse ( Pauli + Dispersion )
                            float    e = exp( -alphaMorse*(r-REQK.x) );
                            float   eM = REQK.y*e;
                            float   de = 2.f*alphaMorse*eM/r;
                            float4  fe = (float4)( dp*de, eM );
                            fe_Paul += fe * e;
                            fe_Lond += fe * (float4)( -1.0f,-1.0f,-1.0f, -2.0f );

                            // if((iG==0)&&(j==0)){
                            //     //float3 sh = dp - pos + LCLJS[j].xyz + lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;
                            //     float3 sh = shift;
                            //     printf( "GPU(%2i,%2i,%2i) sh(%7.3f,%7.3f,%7.3f)\n", ix,iy,iz, sh.x,sh.y,sh.z  );
                            // }
                            //ipbc++;

                            dp   +=lvec.a.xyz;
                            //shift+=lvec.a.xyz;
                        }
                        dp   +=shift_b;
                        //shift+=shift_b;
                        //dp+=lvec.a.xyz*(nPBC.x*-2.f-1.f);
                        //dp+=lvec.b.xyz;
                    }
                    dp   +=shift_c;
                    //shift+=shift_c;
                    //dp+=lvec.b.xyz*(nPBC.y*-2.f-1.f);
                    //dp+=lvec.c.xyz;
                }

            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(iG>=nMax) return;
    int4 coord = (int4){ia,ib,ic,0};
    //write_imagef( FE_Paul, coord, (float4){pos,(float)iG} );
    write_imagef( FE_Paul, coord, fe_Paul );
    write_imagef( FE_Lond, coord, fe_Lond );
    write_imagef( FE_Coul, coord, fe_Coul );
}

