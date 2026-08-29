// gridff_eval.cl — GridFF sampling macros (macro library for injection).
//
// Extracted from gridff_spammm.cl. Contains B-spline sampling kernels as
// //>>>macro blocks, plus helper inline functions at the top (always emitted).
// The macros are injected into getNonBonded of UFF.cl/SPFF.cl/RAFF.cl/RigidMolFF.cl
// via //<<<macro NAME. See doc/topical_audit/gridff_faf.md.
// Requires common.cl + Forces.cl concatenated first.

// ---- Samplers for GridFF ----
__constant sampler_t sampler_gff_norm =  CLK_NORMALIZED_COORDS_TRUE  | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

#ifndef MAKE_INDS_PBC_DEF
#define MAKE_INDS_PBC_DEF
inline int4 make_inds_pbc(const int n, const int iG) {
    switch( iG ){
        case 0: { return (int4)(0, 1,   2,   3  ); }
        case 1: { return (int4)(0, 1,   2,   3-n); }
        case 2: { return (int4)(0, 1,   2-n, 3-n); }
        case 3: { return (int4)(0, 1-n, 2-n, 3-n); }
    }
    return (int4)(-100, -100, -100, -100);
    // iqs[0] = (int4)(0, 1,   2,   3  );
    // iqs[1] = (int4)(0, 1,   2,   3-n);
    // iqs[2] = (int4)(0, 1,   2-n, 3-n);
    // iqs[3] = (int4)(0, 1-n, 2-n, 3-n);
}
#endif

inline int4 choose_inds_pbc(const int i, const int n, const int4* iqs) {
    if (i >= (n-3)) {
        const int ii = i + 4 - n;
        return iqs[ii];
    }
    return (int4)(0, +1, +2, +3);
}

inline int4 choose_inds_pbc_3( const int i, const int n, const int4* iqs ){
    if(i>=(n-3)){ 
        const int ii = i+4-n;
        //printf( "choose_inds_pbc() ii=%i i=%i n=%i \n", ii, i, n );
        const int4 d = iqs[ii];
        return (int4){ i+d.x, i+d.y, i+d.z, i+d.w }; 
    }
    return (int4){ i, i+1, i+2, i+3 };
}


inline float4 basis(float u) {
    const float inv6 = 1.0f / 6.0f;
    const float u2 = u * u;
    const float t = 1.0f - u;
    return (float4)(
        inv6 * t * t * t,
        inv6 * (3.0f * u2 * (u - 2.0f) + 4.0f),
        inv6 * (3.0f * u * (1.0f + u - u2) + 1.0f),
        inv6 * u2 * u
    );
}

inline float4 dbasis(float u) {
    const float u2 = u * u;
    const float t = 1.0f - u;
    return (float4)(
        -0.5f * t * t,
        0.5f * (3.0f * u2 - 4.0f * u),
        0.5f * (-3.0f * u2 + 2.0f * u + 1.0f),
        0.5f * u2
    );
}

// =================== 3D Interpolation - scalar ========================== 

inline float2 fe1D(__global const float* E, const float4 p, const float4 d) {
    const float4 cs = (float4)(E[0], E[1], E[2], E[3]); // ToDo: may be more efficient if we use float4* directly ?
    return (float2)(dot(p, cs), dot(d, cs));
}

inline float3 fe2d(int nz, __global const float* E, int4 di, const float4 pz, const float4 dz, const float4 by, const float4 dy) {
    const float2 fe0 = fe1D(E + di.x, pz, dz);
    const float2 fe1 = fe1D(E + di.y, pz, dz);
    const float2 fe2 = fe1D(E + di.z, pz, dz);
    const float2 fe3 = fe1D(E + di.w, pz, dz);
    return (float3)(
        fe0.x * dy.x + fe1.x * dy.y + fe2.x * dy.z + fe3.x * dy.w,
        fe0.y * by.x + fe1.y * by.y + fe2.y * by.z + fe3.y * by.w,
        fe0.x * by.x + fe1.x * by.y + fe2.x * by.z + fe3.x * by.w
    );
}

inline float4 fe3d_pbc(const float3 u, const int3 n, __global const float* Es, __local const int4* xqis, __local int4* yqis) {
    int ix = (int)u.x;
    int iy = (int)u.y;
    int iz = (int)u.z;
    if (u.x < 0) ix--;
    if (u.y < 0) iy--;
    const float tx = u.x - ix;
    const float ty = u.y - iy;
    const float tz = u.z - iz;

    if ((iz < 1) || (iz >= n.z - 2)) {
        return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }

    ix = modulo(ix-1, n.x);
    iy = modulo(iy-1, n.y);

    const int nyz = n.z * n.y;
    // int4 qx = xqis[ix%4] * nyz;
    // int4 qy = yqis[iy%4] * n.z;

    int4 qx = choose_inds_pbc( ix, n.x, xqis );
    //const int4 qx = choose_inds_pbc( ix, n.x, xqis )*nyz;
    const int4 qy = choose_inds_pbc( iy, n.y, yqis )*n.z;

    const float4 bz = basis(tz);
    const float4 dz = dbasis(tz);
    const float4 by = basis(ty);
    const float4 dy = dbasis(ty);
    
    const int i0 = (iz - 1) + n.z * (iy + n.y * ix);

    //printf( "GPU fe3d_pbc_comb() u(%8.4f,%8.4f,%8.4f) ixyz(%i,%i,%i) n(%i,%i,%i) \n", u.x,u.y,u.z, ix,iy,iz, n.x,n.y,n.z );
    //printf( "GPU fe3d_pbc_comb() u(%8.4f,%8.4f,%8.4f) ixyz(%i,%i,%i) qx(%i,%i,%i,%i) nyz=%i\n", u.x,u.y,u.z, ix,iy,iz, qx.x,qx.y,qx.z,qx.w, nyz );
    qx*=nyz;
    
    //return (float4){ 0.0f, 0.0f, 0.0f, dot(PLQH, Es[ i0 ])  };

    float3 E1 = fe2d(n.z, Es + (i0 + qx.x), qy, bz, dz, by, dy);
    float3 E2 = fe2d(n.z, Es + (i0 + qx.y), qy, bz, dz, by, dy);
    float3 E3 = fe2d(n.z, Es + (i0 + qx.z), qy, bz, dz, by, dy);
    float3 E4 = fe2d(n.z, Es + (i0 + qx.w), qy, bz, dz, by, dy);
    
    const float4 bx = basis(tx);
    const float4 dx = dbasis(tx);
    
    return (float4)(
        dot(dx, (float4)(E1.z, E2.z, E3.z, E4.z)),
        dot(bx, (float4)(E1.x, E2.x, E3.x, E4.x)),
        dot(bx, (float4)(E1.y, E2.y, E3.y, E4.y)),
        dot(bx, (float4)(E1.z, E2.z, E3.z, E4.z))
    );
}

inline float2 fe1Dcomb2(__global const float2* E, const float2 C, const float4 p, const float4 d) {
    const float4 cs = (float4)(dot(C, E[0]), dot(C, E[1]), dot(C, E[2]), dot(C, E[3]));
    return (float2)(dot(p, cs), dot(d, cs));
}

inline float3 fe2d_comb2(int nz, __global const float2* E, int4 di, const float2 C, const float4 pz, const float4 dz, const float4 by, const float4 dy) {
    const float2 fe0 = fe1Dcomb2(E + di.x, C, pz, dz);
    const float2 fe1 = fe1Dcomb2(E + di.y, C, pz, dz);
    const float2 fe2 = fe1Dcomb2(E + di.z, C, pz, dz);
    const float2 fe3 = fe1Dcomb2(E + di.w, C, pz, dz);
    
    return (float3)(
        fe0.x * dy.x + fe1.x * dy.y + fe2.x * dy.z + fe3.x * dy.w,
        fe0.y * by.x + fe1.y * by.y + fe2.y * by.z + fe3.y * by.w,
        fe0.x * by.x + fe1.x * by.y + fe2.x * by.z + fe3.x * by.w
    );
}

inline float4 fe3d_pbc_comb2(const float3 u, const int3 n, __global const float2* Es, const float2 PL, __local const int4* xqis, __local int4* yqis) {
    int ix = (int)u.x;
    int iy = (int)u.y;
    int iz = (int)u.z;
    if (u.x < 0) ix--;
    if (u.y < 0) iy--;
    const float tx = u.x - ix;
    const float ty = u.y - iy;
    const float tz = u.z - iz;

    if ((iz < 1) || (iz >= n.z - 2)) {
        return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }

    ix = modulo(ix-1, n.x);
    iy = modulo(iy-1, n.y);

    const int nyz = n.z * n.y;
    // int4 qx = xqis[ix%4] * nyz;
    // int4 qy = yqis[iy%4] * n.z;

    int4 qx = choose_inds_pbc( ix, n.x, xqis );
    //const int4 qx = choose_inds_pbc( ix, n.x, xqis )*nyz;
    const int4 qy = choose_inds_pbc( iy, n.y, yqis )*n.z;

    const float4 bz = basis(tz);
    const float4 dz = dbasis(tz);
    const float4 by = basis(ty);
    const float4 dy = dbasis(ty);
    
    const int i0 = (iz - 1) + n.z * (iy + n.y * ix);

    //printf( "GPU fe3d_pbc_comb() u(%8.4f,%8.4f,%8.4f) ixyz(%i,%i,%i) n(%i,%i,%i) \n", u.x,u.y,u.z, ix,iy,iz, n.x,n.y,n.z );
    //printf( "GPU fe3d_pbc_comb() u(%8.4f,%8.4f,%8.4f) ixyz(%i,%i,%i) qx(%i,%i,%i,%i) nyz=%i\n", u.x,u.y,u.z, ix,iy,iz, qx.x,qx.y,qx.z,qx.w, nyz );
    qx*=nyz;
    
    //return (float4){ 0.0f, 0.0f, 0.0f, dot(PLQH, Es[ i0 ])  };

    float3 E1 = fe2d_comb2(n.z, Es + (i0 + qx.x), qy, PL, bz, dz, by, dy);
    float3 E2 = fe2d_comb2(n.z, Es + (i0 + qx.y), qy, PL, bz, dz, by, dy);
    float3 E3 = fe2d_comb2(n.z, Es + (i0 + qx.z), qy, PL, bz, dz, by, dy);
    float3 E4 = fe2d_comb2(n.z, Es + (i0 + qx.w), qy, PL, bz, dz, by, dy);
    
    const float4 bx = basis(tx);
    const float4 dx = dbasis(tx);
    
    return (float4)(
        dot(dx, (float4)(E1.z, E2.z, E3.z, E4.z)),
        dot(bx, (float4)(E1.x, E2.x, E3.x, E4.x)),
        dot(bx, (float4)(E1.y, E2.y, E3.y, E4.y)),
        dot(bx, (float4)(E1.z, E2.z, E3.z, E4.z))
    );
}

inline float2 fe1Dcomb(__global const float4* E, const float4 C, const float4 p, const float4 d) {
    const float4 cs = (float4)(dot(C, E[0]), dot(C, E[1]), dot(C, E[2]), dot(C, E[3]));
    return (float2)(dot(p, cs), dot(d, cs));
}

inline float3 fe2d_comb(int nz, __global const float4* E, int4 di, const float4 C, const float4 pz, const float4 dz, const float4 by, const float4 dy) {
    const float2 fe0 = fe1Dcomb(E + di.x, C, pz, dz);
    const float2 fe1 = fe1Dcomb(E + di.y, C, pz, dz);
    const float2 fe2 = fe1Dcomb(E + di.z, C, pz, dz);
    const float2 fe3 = fe1Dcomb(E + di.w, C, pz, dz);
    
    return (float3)(
        fe0.x * dy.x + fe1.x * dy.y + fe2.x * dy.z + fe3.x * dy.w,
        fe0.y * by.x + fe1.y * by.y + fe2.y * by.z + fe3.y * by.w,
        fe0.x * by.x + fe1.x * by.y + fe2.x * by.z + fe3.x * by.w
    );
}

inline float4 fe3d_pbc_comb(const float3 u, const int3 n, __global const float4* Es, const float4 PLQH, __local const int4* xqis, __local int4* yqis) {
    int ix = (int)u.x;
    int iy = (int)u.y;
    int iz = (int)u.z;
    if (u.x < 0) ix--;
    if (u.y < 0) iy--;
    const float tx = u.x - ix;
    const float ty = u.y - iy;
    const float tz = u.z - iz;

    if ((iz < 1) || (iz >= n.z - 2)) {
        return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }

    ix = modulo(ix-1, n.x);
    iy = modulo(iy-1, n.y);

    const int nyz = n.z * n.y;
    // int4 qx = xqis[ix%4] * nyz;
    // int4 qy = yqis[iy%4] * n.z;

    int4 qx = choose_inds_pbc( ix, n.x, xqis );
    //const int4 qx = choose_inds_pbc( ix, n.x, xqis )*nyz;
    const int4 qy = choose_inds_pbc( iy, n.y, yqis )*n.z;

    const float4 bz = basis(tz);
    const float4 dz = dbasis(tz);
    const float4 by = basis(ty);
    const float4 dy = dbasis(ty);
    
    const int i0 = (iz - 1) + n.z * (iy + n.y * ix);

    //printf( "GPU fe3d_pbc_comb() u(%8.4f,%8.4f,%8.4f) ixyz(%i,%i,%i) n(%i,%i,%i) \n", u.x,u.y,u.z, ix,iy,iz, n.x,n.y,n.z );
    //printf( "GPU fe3d_pbc_comb() u(%8.4f,%8.4f,%8.4f) ixyz(%i,%i,%i) qx(%i,%i,%i,%i) nyz=%i\n", u.x,u.y,u.z, ix,iy,iz, qx.x,qx.y,qx.z,qx.w, nyz );
    qx*=nyz;
    
    //return (float4){ 0.0f, 0.0f, 0.0f, dot(PLQH, Es[ i0 ])  };

    float3 E1 = fe2d_comb(n.z, Es + (i0 + qx.x), qy, PLQH, bz, dz, by, dy);
    float3 E2 = fe2d_comb(n.z, Es + (i0 + qx.y), qy, PLQH, bz, dz, by, dy);
    float3 E3 = fe2d_comb(n.z, Es + (i0 + qx.z), qy, PLQH, bz, dz, by, dy);
    float3 E4 = fe2d_comb(n.z, Es + (i0 + qx.w), qy, PLQH, bz, dz, by, dy);
    
    const float4 bx = basis(tx);
    const float4 dx = dbasis(tx);
    
    return (float4)(
        dot(dx, (float4)(E1.z, E2.z, E3.z, E4.z)),
        dot(bx, (float4)(E1.x, E2.x, E3.x, E4.x)),
        dot(bx, (float4)(E1.y, E2.y, E3.y, E4.y)),
        dot(bx, (float4)(E1.z, E2.z, E3.z, E4.z))
    );
}

//>>>macro SAMPLE_3D
// Original kernel: __kernel void sample3D(...)
// Converted to macro for injection into getNonBonded.
__kernel void sample3D(
    const float4 g0,
    const float4 dg,
    const int4 ng,
    __global const float* Eg,
    const int n,
    __global const float4* ps,
    __global float4* fes
) {
    const int iG = get_global_id(0);
    const int iL = get_local_id(0);
    if (iG >= n) return;

    __local int4 xqs[4];
    __local int4 yqs[4];
    if      (iL<4){             xqs[iL]=make_inds_pbc(ng.x,iL); }
    else if (iL<8){ int i=iL-4; yqs[i ]=make_inds_pbc(ng.y,i ); };
    const float3 inv_dg = 1.0f / dg.xyz;
    barrier(CLK_LOCAL_MEM_FENCE);

    float3 p = ps[iG].xyz;
    float3 u = (p - g0.xyz) * inv_dg;
    float4 fe = fe3d_pbc(u, ng.xyz, Eg, xqs, yqs);
    fe.xyz *= -inv_dg;
    fes[iG] = fe;
}

//>>>macro SAMPLE_3D_GRID
// Original kernel: __kernel void sample3D_grid(...)
// Converted to macro for injection into getNonBonded.
__kernel void sample3D_grid(
    const float4 g0,
    const float4 dg,
    const int4   ng,
    __global const float* Eg,
    const float4 samp_g0,
    const float4 samp_dg,
    const int4   samp_ng,
    __global float4* fes
) {
    const int iG = get_global_id(0);
    const int iL = get_local_id(0);
    const int nxyz = samp_ng.w; 
    if (iG >= nxyz ) return;

    __local int4 xqs[4];
    __local int4 yqs[4];
    if      (iL<4){             xqs[iL]=make_inds_pbc(ng.x,iL); }
    else if (iL<8){ int i=iL-4; yqs[i ]=make_inds_pbc(ng.y,i ); };
    const float3 inv_dg = 1.0f / dg.xyz;
    barrier(CLK_LOCAL_MEM_FENCE);

    // if(iG==0){ 
    //     printf( "GPU sample3D_grid() g0(%8.4f,%8.4f,%8.4f) dg(%8.4f,%8.4f,%8.4f) ng(%i,%i,%i) \n", g0.x,g0.y,g0.z, dg.x,dg.y,dg.z, ng.x,ng.y,ng.z );
    //     printf( "GPU sample3D_grid() samp_g0(%8.4f,%8.4f,%8.4f) samp_dg(%8.4f,%8.4f,%8.4f) samp_ng(%i,%i,%i) \n", samp_g0.x,samp_g0.y,samp_g0.z, samp_dg.x,samp_dg.y,samp_dg.z, samp_ng.x,samp_ng.y,samp_ng.z ); 
        
    // }

    // if( iG==0 ){
    //     printf( "GPU sample3D_grid() samp_g0(%8.4f,%8.4f,%8.4f) samp_dg(%8.4f,%8.4f,%8.4f) samp_ng(%i,%i,%i|%i) \n", samp_g0.x,samp_g0.y,samp_g0.z, samp_dg.x,samp_dg.y,samp_dg.z, samp_ng.x,samp_ng.y,samp_ng.z,samp_ng.w );
    //     printf("GPU sample3D_comb() ng(%i,%i,%i) g0(%g,%g,%g) dg(%g,%g,%g) \n", ng.x,ng.y,ng.z,   g0.x,g0.y,g0.z,   dg.x,dg.y,dg.z );
    //     //printf("GPU xqs[0](%i,%i,%i,%i) xqs[1](%i,%i,%i,%i) xqs[2](%i,%i,%i,%i) xqs[3](%i,%i,%i,%i)\n", xqs[0].x, xqs[0].y, xqs[0].z, xqs[0].w,   xqs[1].x, xqs[1].y, xqs[1].z, xqs[1].w,   xqs[2].x, xqs[2].y, xqs[2].z, xqs[2].w,  xqs[3].x, xqs[3].y, xqs[3].z, xqs[3].w   );
    //     //for(int i=0; i<ng; i++){  printf("Gs[%i]=%f\n", i, Gs[i]); }
    //     for(int i=0; i<10; i++){
    //         //float3 p = ps[i].xyz;
    //         int ii = i +   samp_ng.x*10 +    10*samp_ng.x*samp_ng.y;
    //         const float3 g = (float3)( ii % samp_ng.x, (ii / samp_ng.x) % samp_ng.y, ii / (samp_ng.x * samp_ng.y));
    //         const float3 p = samp_g0.xyz + samp_dg.xyz * g;
    //         float3 u = (p - g0.xyz) * inv_dg;
    //         float4 fe = fe3d_pbc(u, ng.xyz, Eg, xqs, yqs);
    //         fe.xyz *= -inv_dg;
    //         printf( "GPU sample3D_comb()[%i|%i] g(%8.4f,%8.4f,%8.4f) p(%8.4f,%8.4f,%8.4f) u(%8.4f,%8.4f,%8.4f)   fe(%g,%g,%g | %g) \n",  i, ii,   g.x,g.y,g.z,   p.x,p.y,p.z,  u.x,u.y,u.z,   fe.x, fe.y, fe.z, fe.w );
    //         fes[i] = fe;
    //     }
    // }

    const int ix = iG % samp_ng.x;
    const int iy = (iG / samp_ng.x) % samp_ng.y;
    const int iz = iG / (samp_ng.x * samp_ng.y);

    const float3 g = (float3)(ix, iy, iz );
    const float3 p = samp_g0.xyz + samp_dg.xyz * g;
    const float3 u = (p - g0.xyz) * inv_dg;
    float4 fe = fe3d_pbc(u, ng.xyz, Eg, xqs, yqs);
    fe.xyz *= -inv_dg;
    fes[iG] = fe;

    //if( (ix==10) && (iy==10) ){     printf( "GPU sample3D_comb()[%i|%i,%i,%i] p(%8.4f,%8.4f,%8.4f) u(%8.4f,%8.4f,%8.4f)   fe(%g,%g,%g | %g) \n",  iG, ix,iy,iz,   p.x,p.y,p.z,  u.x,u.y,u.z,   fe.x, fe.y, fe.z, fe.w ); }
}

//>>>macro SAMPLE_3D_COMB2
// Original kernel: __kernel void sample3D_comb2(...)
// Converted to macro for injection into getNonBonded.
__kernel void sample3D_comb2(
    const float4 g0,
    const float4 dg,
    const int4 ng,
    __global const float2* Eg,
    const int n,
    __global const float4* ps,
    __global float4* fes,
    const float2 C
) {
    const int iG = get_global_id(0);
    const int iL = get_local_id(0);
    if (iG >= n) return;

    __local int4 xqs[4];
    __local int4 yqs[4];
    if      (iL<4){             xqs[iL]=make_inds_pbc(ng.x,iL); }
    else if (iL<8){ int i=iL-4; yqs[i ]=make_inds_pbc(ng.y,i ); };
    const float3 inv_dg = 1.0f / dg.xyz;
    barrier(CLK_LOCAL_MEM_FENCE);

    float3 p = ps[iG].xyz;
    float3 u = (p - g0.xyz) * inv_dg;
    float4 fe = fe3d_pbc_comb2(u, ng.xyz, Eg, C, xqs, yqs);
    fe.xyz *= -inv_dg;
    fes[iG] = fe;
}

//>>>macro SAMPLE_3D_COMB
// Original kernel: __kernel void sample3D_comb(...)
// Converted to macro for injection into getNonBonded.
__kernel void sample3D_comb(
    const float4 g0,
    const float4 dg,
    const int4 ng,
    __global const float4* Eg,
    const int n,
    __global const float4* ps,
    __global float4* fes,
    const float4 C
    //__global int4* xqs,
    //__global int4* yqs
) {
    const int iG = get_global_id(0);
    const int iL = get_local_id(0);
    if (iG >= n) return;

    __local int4 xqs[4];
    __local int4 yqs[4];
    if      (iL<4){             xqs[iL]=make_inds_pbc(ng.x,iL); }
    else if (iL<8){ int i=iL-4; yqs[i ]=make_inds_pbc(ng.y,i ); };
    const float3 inv_dg = 1.0f / dg.xyz;
    barrier(CLK_LOCAL_MEM_FENCE);

    // if( iG==0 ){
    //     printf("GPU sample3D_comb() ng(%i,%i,%i) g0(%g,%g,%g) dg(%g,%g,%g) C(%g,%g,%g) \n", ng.x,ng.y,ng.z,   g0.x,g0.y,g0.z,   dg.x,dg.y,dg.z,   C.x,C.y,C.z );
    //     printf("GPU xqs[0](%i,%i,%i,%i) xqs[1](%i,%i,%i,%i) xqs[2](%i,%i,%i,%i) xqs[3](%i,%i,%i,%i)\n", xqs[0].x, xqs[0].y, xqs[0].z, xqs[0].w,   xqs[1].x, xqs[1].y, xqs[1].z, xqs[1].w,   xqs[2].x, xqs[2].y, xqs[2].z, xqs[2].w,  xqs[3].x, xqs[3].y, xqs[3].z, xqs[3].w   );
    //     //for(int i=0; i<ng; i++){  printf("Gs[%i]=%f\n", i, Gs[i]); }
    //     for(int i=0; i<n; i++){
    //         float3 p = ps[i].xyz;
    //         //printf( "ps[%3i] ( %8.4f, %8.4f, %8.4f,) \n", i, p.x,p.y,p.z );
    //         float3 u = (p - g0.xyz) * inv_dg;
    //         // int ix = (int)u.x; 
    //         // int iy = (int)u.y;
    //         // int iz = (int)u.z;
    //         // int ixyz = iz + ng.z*( iy + ng.y*ix);
    //         // float4 Es = Eg[ixyz];
    //         // //printf( "Eg[%3i,%3i,%3i]=(%g,%g,%g,%g) \n", ix,iy,iz, Es.x,Es.y,Es.z,Es.w );
    //         // float E = dot(Es,C);
    //         // float4 fe  = (float4){E,E,E,E};
    //         float4 fe = fe3d_pbc_comb(u, ng.xyz, Eg, C, xqs, yqs);
    //         fe.xyz *= -inv_dg;
    //         //printf( "GPU sample3D_comb()[%i] fe(%g,%g,%g | %g) \n",i, fe.x, fe.y, fe.z, fe.w );
    //         fes[i] = fe;
    //     }
    // }

    float3 p = ps[iG].xyz;
    float3 u = (p - g0.xyz) * inv_dg;
    float4 fe = fe3d_pbc_comb(u, ng.xyz, Eg, C, xqs, yqs);
    fe.xyz *= -inv_dg;
    fes[iG] = fe;
    
}

//>>>macro SAMPLE_1D_PBC
// Original kernel: __kernel void sample1D_pbc(...)
// Converted to macro for injection into getNonBonded.
__kernel void sample1D_pbc(
    const float g0,
    const float dg,
    const int ng,
    __global const float* Gs,
    const int n,
    __global const float* ps,
    __global float2* fes
    //__global int4* xqs
) {
    const int iG = get_global_id(0);
    if (iG >= n) return;

    
    __local int4 xqs[4];
    const int iL = get_local_id(0);
    if      (iL<4){ xqs[iL]=make_inds_pbc(ng,iL); }
    barrier(CLK_LOCAL_MEM_FENCE);

    // if( (iG==0) ){
    //     printf("xqs[0](%i,%i,%i,%i)\n xqs[1](%i,%i,%i,%i)\n xqs[2](%i,%i,%i,%i)\n xqs[3](%i,%i,%i,%i)\n", xqs[0].x, xqs[0].y, xqs[0].z, xqs[0].w,   xqs[1].x, xqs[1].y, xqs[1].z, xqs[1].w,   xqs[2].x, xqs[2].y, xqs[2].z, xqs[2].w,  xqs[3].x, xqs[3].y, xqs[3].z, xqs[3].w   );
    //     for(int i=0; i<ng; i++){  printf("Gs[%i]=%f\n", i, Gs[i]); }
    // }

    // local memory barrire
    //int4 xqis[4]; make_inds_pbc(ng, xqis);   // this should be pre-calculated globaly

    float inv_dg = 1.0f / dg;
    float p = ps[iG];
    float2 fe = fe1d_pbc_macro(  (p - g0) * inv_dg, ng, Gs, xqs);
    fe.y *= inv_dg;
    fes[iG] = fe;
}

//>>>macro SAMPLE_GRIDFF_BSPLINE_POINTS
// Original kernel: __kernel void sampleGridFF_Bspline_points(...)
// Converted to macro for injection into getNonBonded.
__kernel void sampleGridFF_Bspline_points(
    const int4 ns,                  // 1  (natoms,nnode,nvec,0)
    __global float4*  atoms,        // 2
    __global float4*  forces,       // 3
    __global float4*  BsplinePLQ,   // 4
    const int4        grid_ns,      // 5
    const float4      grid_invStep, // 6
    const float4      grid_p0,      // 7
    const float4      PLQH          // 8
){
    __local int4 xqs[4];
    __local int4 yqs[4];
    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    const int iL = get_local_id(0);
    const int natoms = ns.x;
    const int nnode  = ns.y;
    const int nvec   = natoms + nnode;
    const int i0v    = iS*nvec;
    const int iav    = iG + i0v;
    if(iL<4){ xqs[iL] = make_inds_pbc(grid_ns.x, iL); }
    else if(iL<8){ int i=iL-4; yqs[i] = make_inds_pbc(grid_ns.y, i); }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(iG>=natoms) return;
    const float3 pos = atoms[iav].xyz;
    const float3 u = (pos - grid_p0.xyz) * grid_invStep.xyz;
    float4 fg = fe3d_pbc_comb(u, grid_ns.xyz, BsplinePLQ, PLQH, xqs, yqs);
    fg.xyz *= -grid_invStep.xyz;
    forces[iav] = (float4)(fg.x, fg.y, fg.z, -fg.w);
}

//>>>macro SAMPLE_GRIDFF
// Original kernel: __kernel void sampleGridFF(...)
// Converted to macro for injection into getNonBonded.
__kernel void sampleGridFF(
    const int4 ns,                  // 1
    __global float4*  atoms,        // 2
    __global float4*  forces,       // 3
    __global float4*  REQs,         // 4
    const float4  GFFParams,        // 5
    __read_only image3d_t  FE_Paul, // 6
    __read_only image3d_t  FE_Lond, // 7
    __read_only image3d_t  FE_Coul, // 8
    const cl_Mat3  diGrid,          // 9
    const float4   grid_p0          // 10
){
    const int iG = get_global_id  (0);
    const int nG = get_global_size(0);
    const int np = ns.x;

    float3 dz = (float3){ 0.0f, 0.0f, 0.1f };

    //const bool   bNode = iG<nnode;   // All atoms need to have neighbors !!!!
    const float4 REQ        = REQs[iG];
    const float3 posi       = atoms[iG].xyz;
    const float  R2damp     = GFFParams.x*GFFParams.x;
    const float  alphaMorse = GFFParams.y;

    const float ej   = exp( alphaMorse* REQ.x );
    const float cL   = ej*REQ.y;
    const float cP   = ej*cL;

    /*
    if(iG==0){ printf( "GPU::sampleGridFF() np=%i R2damp=%g aMorse=%g p(%g,%g,%g) REQ(%g,%g,%g)  cP=%g cL=%g ej=%g \n", np, R2damp, alphaMorse, posi.x,posi.y,posi.z, REQ.x,REQ.y,REQ.z, cP,cL,ej  ); }
    if(iG==0){
        printf( "GPU_sGFF #i  z  E_Paul Fz_Paul   E_Lond Fz_Lond   E_Coul Fz_Coul  \n" );
        for(int i=0; i<np; i++){
            const float4 REQ  = REQs[i];
            //const float3 posi = atoms[i].xyz;
            const float3 posi = grid_p0.xyz + dz*i;
            const float ej   = exp( alphaMorse* REQ.x );
            const float cL   = ej*REQ.y;
            const float cP   = ej*cL;

            float4 fe          = float4Zero;
            const float3 posg  = posi - grid_p0.xyz;
            const float4 coord = (float4)( dot(posg, diGrid.a.xyz),   dot(posg,diGrid.b.xyz), dot(posg,diGrid.c.xyz), 0.0f );
            #if 0
                //coord +=(float4){0.5f,0.5f,0.5f,0.0f}; // shift 0.5 voxel when using native texture interpolation
                const float4 fe_Paul = read_imagef( FE_Paul, sampler_gff_norm, coord );
                const float4 fe_Lond = read_imagef( FE_Lond, sampler_gff_norm, coord );
                const float4 fe_Coul = read_imagef( FE_Coul, sampler_gff_norm, coord );
            #else
                const float4 fe_Paul = read_imagef_trilin_norm( FE_Paul, coord );
                const float4 fe_Lond = read_imagef_trilin_norm( FE_Lond, coord );
                const float4 fe_Coul = read_imagef_trilin_norm( FE_Coul, coord );
            #endif
            //read_imagef_trilin( imgIn, coord );  // This is for higher accuracy (not using GPU hw texture interpolation)
            fe  += fe_Paul*cP  + fe_Lond*cL  +  fe_Coul*REQ.z;
            //printf( "GPU[%i] z(%g) E,fz(%g,%g)  PLQ(%g,%g,%g) REQ(%g,%g) \n", i, posi.z,  fe.w,fe.z,  cP,cL,REQ.z,  REQ.x,REQ.y  );
            printf(  "GPU_sGFF %3i %8.3f    %14.6f %14.6f    %14.6f %14.6f    %14.6f %14.6f\n", i, posi.z, fe_Paul.w,fe_Paul.z, fe_Lond.w,fe_Lond.z,  fe_Coul.w,fe_Coul.z  );
        }
    }
    */


// NOTE: https://registry.khronos.org/OpenCL/sdk/1.1/docs/man/xhtml/sampler_t.html
// CLK_ADDRESS_REPEAT - out-of-range image coordinates are wrapped to the valid range. This address mode can only be used with normalized coordinates. If normalized coordinates are not used, this addressing mode may generate image coordinates that are undefined.

    // ========== Interaction with grid
    float4 fe               = float4Zero;
    const float3 posg  = posi - grid_p0.xyz;
    float4 coord = (float4)( dot(posg, diGrid.a.xyz),   dot(posg,diGrid.b.xyz), dot(posg,diGrid.c.xyz), 0.0f );
    if(iG==0){ printf( "coord(%g,%g,%g) pos(%g,%g,%g) diGrid.a(%g,%g,%g)\n", coord.x,coord.y,coord.z,  posi.x,posi.y,posi.z, diGrid.a.x,diGrid.a.y,diGrid.a.z ); }
    //#if 0
        //coord +=(float4){0.5f,0.5f,0.5f,0.0f}; // shift 0.5 voxel when using native texture interpolation
        const float4 fe_Paul = read_imagef( FE_Paul, sampler_gff_norm, coord );
        const float4 fe_Lond = read_imagef( FE_Lond, sampler_gff_norm, coord );
        const float4 fe_Coul = read_imagef( FE_Coul, sampler_gff_norm, coord );
    // #else
    //     const float4 fe_Paul = read_imagef_trilin_norm( FE_Paul, coord );
    //     const float4 fe_Lond = read_imagef_trilin_norm( FE_Lond, coord );
    //     const float4 fe_Coul = read_imagef_trilin_norm( FE_Coul, coord );
    //#endif
    //read_imagef_trilin( imgIn, coord );  // This is for higher accuracy (not using GPU hw texture interpolation)
    forces[iG] = fe_Paul*cP  + fe_Lond*cL  +  fe_Coul*REQ.z;
}

