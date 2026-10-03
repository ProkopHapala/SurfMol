// gridff.cl - Grid force field construction, B-spline interpolation, and Poisson solver
//
// Provides kernels for building 3D force-field grids from atom positions and
// sampling them at arbitrary points via B-spline interpolation. Used to
// precompute molecule-substrate interaction fields for fast MD evaluation.
//
// Kernels:
//   - sample3D / sample3D_grid / sample3D_comb / sample3D_comb2: B-spline
//     interpolation of a 3D scalar/vector field at grid points or arbitrary
//     positions, with optional PBC wrapping.
//   - sample1D_pbc: 1D B-spline interpolation with PBC.
//   - BsplineConv3D / BsplineConv3D_tex: 3D convolution with B-spline kernel
//     (separable along x,y,z). Used for grid smoothing/interpolation.
//   - Convolution3D_General: General 3D convolution with arbitrary kernel.
//   - make_MorseFF / make_MorseFF_f4: Build Pauli/London/Coulomb force-field
//     grids from substrate atoms using Morse or LJ potentials.
//   - make_Coulomb_points: Build Coulomb potential grid from point charges.
//   - project_atom_on_grid_cubic_pbc / project_atoms_on_grid_quintic_pbc:
//     Project atom density onto grid using cubic/quintic splines with PBC.
//   - poissonW / poissonW_old: Poisson solver in Fourier space (spectral).
//   - laplace_real_pbc: Laplacian in real space with PBC.
//   - slabPotential / slabPotential_zyx: Apply slab correction potential.
//   - sampleGridFF_Bspline_points: Sample GridFF at arbitrary points via
//     B-spline interpolation (buffer-based, not texture).
//   - sampleGridFF: Sample GridFF at grid points (for validation/debugging).
//   - make_GridFF: Build Pauli/London/Coulomb grids from substrate atoms
//     (high-level wrapper combining projection + interpolation).
//   - addMul, dot_wg, setLinear, move, setMul, setCMul, set: Utility kernels
//     for array operations (element-wise multiply, dot product, fill, copy).
//
// Requires: common.cl + Forces.cl to be concatenated before this file.

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

// =================== 3D Interpolation - float2 ========================== 

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


// =================== 3D Interpolation - float4 ========================== 


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


// =================== 3D Interpolation - float4 ========================== 


inline float2 fe1d_pbc_macro(float x, int n, __global const float* Es, __local const int4* xqis ){
    int i = (int)x;
    if (x < 0) i--;
    float t = x - i;
    i = modulo(i - 1, n);
    int4 q = choose_inds_pbc_3(i, n, xqis);
    //printf( "fe1d_pbc_macro(x=%8.4f) %3i/%3i qi(%i,%i,%i,%i)     q0(%i,%i,%i,%i) q1(%i,%i,%i,%i) q2(%i,%i,%i,%i) q3(%i,%i,%i,%i) \n", x, i, n,  q.x,q.y,q.z,q.w,   xqis[0].x,xqis[0].y,xqis[0].z,xqis[0].w,   xqis[1].x,xqis[1].y,xqis[1].z,xqis[1].w,  xqis[2].x,xqis[2].y,xqis[2].z,xqis[2].w,  xqis[3].x,xqis[3].y,xqis[3].z,xqis[3].w     );
    float4 b = basis(t);
    float4 d = dbasis(t);
    float4 cs = (float4)(Es[q.x], Es[q.y], Es[q.z], Es[q.w]);
    
    return (float2)(dot(b, cs), dot(d, cs));
}

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


// =============================================
// ===================== Fitting
// =============================================

float conv3x3_pbc( __global const float* Gs, const float3 B, const int iiz, const int3 ix, const int3 iy ){
    return  Gs[ix.x+iy.x+iiz]*B.z + Gs[ix.y+iy.x+iiz]*B.y + Gs[ix.z+iy.x+iiz]*B.z  +
            Gs[ix.x+iy.y+iiz]*B.y + Gs[ix.y+iy.y+iiz]*B.x + Gs[ix.z+iy.y+iiz]*B.y  +
            Gs[ix.x+iy.z+iiz]*B.z + Gs[ix.y+iy.z+iiz]*B.y + Gs[ix.z+iy.z+iiz]*B.z  ;
}

float conv_3x3_tex( sampler_t samp, __read_only image3d_t tex, float3 B, int4 coord ){
    return
      read_imagef(tex, samp, coord + (int4)(-1,-1,0,0) ).x * B.z
    + read_imagef(tex, samp, coord + (int4)( 0,-1,0,0) ).x * B.y
    + read_imagef(tex, samp, coord + (int4)( 1,-1,0,0) ).x * B.z

    + read_imagef(tex, samp, coord + (int4)(-1, 0,0,0) ).x * B.y
    + read_imagef(tex, samp, coord                     ).x * B.x
    + read_imagef(tex, samp, coord + (int4)( 1, 0,0,0) ).x * B.y

    + read_imagef(tex, samp, coord + (int4)(-1, 1,0,0) ).x * B.z
    + read_imagef(tex, samp, coord + (int4)( 0, 1,0,0) ).x * B.y
    + read_imagef(tex, samp, coord + (int4)( 1, 1,0,0) ).x * B.z;

}

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


// =============================================
// ===================== Math Utilities
// =============================================

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

__attribute__((reqd_work_group_size(64,1,1)))
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

__constant sampler_t samp_pbc = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;

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

// Defaults
#ifndef LS_X
#define LS_X 8
#endif
#ifndef LS_Y
#define LS_Y 8
#endif
#ifndef LS_Z
#define LS_Z 8
#endif

#define TW (LS_X + 2)
#define TH (LS_Y + 2)
#define TD (LS_Z + 2)

inline int pmod(int i, int n) {
    int r = i % n;
    return (r < 0) ? r + n : r;
}

__kernel void Convolution3D_General(
    const int4 ns,            // {nx, ny, nz, 0}
    __global const float* Gs, // Input
    __global const float* G0, // Optional Additive Input
    __global       float* out,// Output
    const float4   weights,   // x=Center, y=Face, z=Edge, w=Corner
    const int4     bPBC,      // Boundary: 1=Periodic, 0=Zero
    const float2   coefs      // x=Multiplicative, y=Additive G0 scale
) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lz = get_local_id(2);
    const int g_start_x = get_group_id(0) * LS_X;
    const int g_start_y = get_group_id(1) * LS_Y;
    const int g_start_z = get_group_id(2) * LS_Z;
    // DEBUG: print kernel parameters once for the first work-item
    // if((g_start_x==0) && (g_start_y==0) && (g_start_z==0) && (lx==0) && (ly==0) && (lz==0)){
    //     int4 ls=(int4){get_local_size(0), get_local_size(1), get_local_size(2),0};
    //     int4 gs=(int4){get_global_size(0), get_global_size(1), get_global_size(2),0};
    //     printf("GPU Convolution3D_General ns{%i,%i,%i,%i} weights{%f,%f,%f,%f} bPBC{%i,%i,%i,%i} coefs{%f,%f} G0=%p ls(%i,%i,%i) gs(%i,%i,%i)\n",
    //         ns.x, ns.y, ns.z, ns.w,
    //         weights.x, weights.y, weights.z, weights.w,
    //         bPBC.x, bPBC.y, bPBC.z, bPBC.w,
    //         coefs.x, coefs.y,
    //         G0,
    //         ls.x, ls.y,ls.z,
    //         gs.x, gs.y,gs.z
    //     );
    // }
    __local float tile[TD][TH][TW];
    const int tid = lz * (LS_Y * LS_X) + ly * LS_X + lx; 
    const int group_nthreads = LS_X * LS_Y * LS_Z;
    const int tile_nelements = TW * TH * TD;
    // --- 1. LOAD TILE ---
    for (int i = tid; i < tile_nelements; i += group_nthreads) {
        int r = i;
        int tx = r % TW; r /= TW;
        int ty = r % TH; r /= TH;
        int tz = r;
        int gx = g_start_x + tx - 1;
        int gy = g_start_y + ty - 1;
        int gz = g_start_z + tz - 1;
        bool inside = true;
        if (gx < 0 || gx >= ns.x) { if (bPBC.x) gx = pmod(gx, ns.x); else inside = false; }
        if (gy < 0 || gy >= ns.y) { if (bPBC.y) gy = pmod(gy, ns.y); else inside = false; }
        if (gz < 0 || gz >= ns.z) { if (bPBC.z) gz = pmod(gz, ns.z); else inside = false; }
        float val = 0.0f;
        if (gx >= 0 && gx < ns.x && gy >= 0 && gy < ns.y && gz >= 0 && gz < ns.z) {
            val = Gs[gz * (ns.x * ns.y) + gy * ns.x + gx];
        }
        tile[tz][ty][tx] = val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // --- 2. CHECK OUTPUT BOUNDS ---
    int out_x = g_start_x + lx;
    int out_y = g_start_y + ly;
    int out_z = g_start_z + lz;
    if (out_x >= ns.x || out_y >= ns.y || out_z >= ns.z) return;
    // --- 3. CONVOLUTION ---
    int tx = lx + 1;
    int ty = ly + 1;
    int tz = lz + 1;
    float sum = 0.0f;
    // Center
    sum += tile[tz][ty][tx] * weights.x;
    // Faces (6)
    sum += (tile[tz][ty][tx-1] + tile[tz][ty][tx+1] +
            tile[tz][ty-1][tx] + tile[tz][ty+1][tx] +
            tile[tz-1][ty][tx] + tile[tz+1][ty][tx]) * weights.y;
    // Edges (12)
    sum += (tile[tz][ty-1][tx-1] + tile[tz][ty-1][tx+1] +
            tile[tz][ty+1][tx-1] + tile[tz][ty+1][tx+1] +
            tile[tz-1][ty][tx-1] + tile[tz-1][ty][tx+1] +
            tile[tz+1][ty][tx-1] + tile[tz+1][ty][tx+1] +
            tile[tz-1][ty-1][tx] + tile[tz-1][ty+1][tx] +
            tile[tz+1][ty-1][tx] + tile[tz+1][ty+1][tx]) * weights.z;
    // Corners (8)
    sum += (tile[tz-1][ty-1][tx-1] + tile[tz-1][ty-1][tx+1] +
            tile[tz-1][ty+1][tx-1] + tile[tz-1][ty+1][tx+1] +
            tile[tz+1][ty-1][tx-1] + tile[tz+1][ty-1][tx+1] +
            tile[tz+1][ty+1][tx-1] + tile[tz+1][ty+1][tx+1]) * weights.w;
    // --- 4. WRITE ---
    int g_idx = out_z * (ns.x * ns.y) + out_y * ns.x + out_x;
    sum *= coefs.x;
    if (G0) { sum += G0[g_idx] * coefs.y; }
    out[g_idx] = sum;
}

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

__kernel void set(
    const int  ntot,
    __global float* out,  
    float c
) {
    const int i = get_global_id(0);
    if (i > ntot ) return;
    out[i] = c;
}

__attribute__((reqd_work_group_size(32,1,1)))
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

__attribute__((reqd_work_group_size(32,1,1)))
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


__attribute__((reqd_work_group_size(32,1,1)))
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



int pbc_ifw(int i, int n){ i++; return (i<n )?  i :  i-n; };
int pbc_ibk(int i, int n){ i--; return (i>=0)?  i :  i+n; };


// float4 Bspline_basis(const float u) {
//     const float inv6 = 1.0f / 6.0f;
//     const float u2 = u * u;
//     const float t = 1.0f - u;
//     return (float4)(
//         inv6 * t * t * t,
//         inv6 * (3.0f * u2 * (u - 2.0f) + 4.0f),
//         inv6 * (3.0f * u * (1.0f + u - u2) + 1.0f),
//         inv6 * u2 * u
//     );
// }
// float4 Bspline_dbasis(const float u) {
//     const float u2 = u * u;
//     const float t = 1.0f - u;
//     return (float4)(
//         -0.5f * t * t,
//         0.5f * (3.0f * u2 - 4.0f * u),
//         0.5f * (-3.0f * u2 + 2.0f * u + 1.0f),
//         0.5f * u2
//     );
// }

void Bspline_basis(const float u, float * ws) {
    const float inv6 = 1.0f / 6.0f;
    const float u2 = u * u;
    const float t = 1.0f - u;
    //return (float4)(
    ws[0]=    inv6 * t * t * t;
    ws[1]=    inv6 * (3.0f * u2 * (u - 2.0f) + 4.0f);
    ws[2]=    inv6 * (3.0f * u * (1.0f + u - u2) + 1.0f);
    ws[3]=    inv6 * u2 * u;
    //);
}

void Bspline_dbasis(const float u, float * ws) {
    const float u2 = u * u;
    const float t = 1.0f - u;
    //return (float4)(
    ws[0]=    -0.5f * t * t;
    ws[1]=     0.5f * ( 3.0f * u2 - 4.0f * u);
    ws[2]=     0.5f * (-3.0f * u2 + 2.0f * u + 1.0f);
    ws[3]=     0.5f * u2;
    //);
}


void Bspline_basis5(const float t, float * ws){
    const float inv6 = 1.f/6.f;
    const float t2 = t*t;
    const float t3 = t2*t;
    const float t4 = t2*t2;
    const float t5 = t3*t2;
    //return (float8){                                                  
    ws[0]=  -0.008333333333333333*t5  +0.041666666666666666*t4  -0.08333333333333333*t3 +0.08333333333333333*t2  -0.041666666666666666*t   +0.008333333333333333;
    ws[1]=   0.041666666666666666*t5  -0.166666666666666666*t4  +0.16666666666666666*t3 +0.16666666666666666*t2  -0.416666666666666666*t   +0.216666666666666666;        
    ws[2]=  -0.083333333333333333*t5  +0.250000000000000000*t4                          -0.50000000000000000*t2                            +0.550000000000000000;  
    ws[3]=   0.083333333333333333*t5  -0.166666666666666666*t4  -0.16666666666666666*t3 +0.16666666666666666*t2  +0.416666666666666666*t   +0.216666666666666666;
    ws[4]=  -0.041666666666666666*t5  +0.041666666666666666*t4  +0.08333333333333333*t3 +0.08333333333333333*t2  +0.041666666666666666*t   +0.008333333333333333; 
    ws[5]=   0.008333333333333333*t5;
    //     0.f,0.f,
    //};
}


void Bspline_dbasis5(const float t, float * ws){
    const float inv6 = 1.f/6.f;
    const float t2 = t*t;
    const float t3 = t2*t;
    const float t4 = t2*t2;
    //return (float8){           
    ws[0]=    -0.0416666666666667*t4	+0.166666666666667*t3	-0.25*t2   +0.166666666666667*t	-0.041666666666666666;	
    ws[1]=     0.2083333333333333*t4	-0.666666666666667*t3	+0.50*t2   +0.333333333333333*t -0.416666666666666666;	
    ws[2]=    -0.4166666666666667*t4	+1.000000000000000*t3	           -1.000000000000000*t                      ;	
    ws[3]=     0.4166666666666667*t4	-0.666666666666667*t3	-0.50*t2   +0.333333333333333*t	+0.416666666666666666;	
    ws[4]=    -0.2083333333333333*t4	+0.166666666666667*t3	+0.25*t2   +0.166666666666667*t	+0.041666666666666666;	
    ws[5]=     0.0416666666666667*t4;
    //};
}


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

inline void make_inds_pbc_5(const int n, const int iG, __local int inds[6]) {
    switch (iG) {
        case 0:  inds[0]=0;    inds[1]=1;    inds[2]=2;    inds[3]=3;    inds[4]=4;    inds[5]=5;    break;
        case 1:  inds[0]=0;    inds[1]=1;    inds[2]=2;    inds[3]=3;    inds[4]=4;    inds[5]=5-n;  break;
        case 2:  inds[0]=0;    inds[1]=1;    inds[2]=2;    inds[3]=3;    inds[4]=4-n;  inds[5]=5-n;  break;
        case 3:  inds[0]=0;    inds[1]=1;    inds[2]=2;    inds[3]=3-n;  inds[4]=4-n;  inds[5]=5-n;  break;
        case 4:  inds[0]=0;    inds[1]=1;    inds[2]=2-n;  inds[3]=3-n;  inds[4]=4-n;  inds[5]=5-n;  break;
        case 5:  inds[0]=0;    inds[1]=1-n;  inds[2]=2-n;  inds[3]=3-n;  inds[4]=4-n;  inds[5]=5-n;  break;
        default: inds[0]=-100; inds[1]=-100; inds[2]=-100; inds[3]=-100; inds[4]=-100; inds[5]=-100; break;
    }
}

inline void choose_inds_pbc_5(const int i, const int n, __local const int iqs[6][6], int out[6]) {
    if (i >= (n - 5)) {
        const int ii  = i+6-n;
        const int* qi = iqs[ii];
              out[0]=i+qi[0];    out[1]=i+qi[1];    out[2]=i+qi[2];    out[3]=i+qi[3];    out[4]=i+qi[4];    out[5]=i+qi[5];
    } else {  out[0]=i;          out[1]=i+1;        out[2]=i+2;        out[3]=i+3;        out[4]=i+4;        out[5]=i+5;      }
}


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

// ---- From relax_multi.cl: GridFF sampling helpers ----
// ======================================================================
//                    sampleGridFF_Bspline_points()
// ======================================================================
// Sample GridFF B-spline at arbitrary points. Intended for rigid single-probe
// batched surface scans. X/Y are periodic, Z is non-periodic and returns zero
// outside valid cubic support. Output layout matches other rigid surface
// kernels: forces.xyz = -dE/dxyz, forces.w = -E.
__attribute__((reqd_work_group_size(32,1,1)))
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


inline float evalGridFFEnergy3D(
    const float3 pos,
    __global float4*  BsplinePLQ,
    const int4        grid_ns,
    const float4      grid_invStep,
    const float4      grid_p0,
    const float4      PLQH,
    __local const int4* xqs,
    __local int4* yqs
){
    const float3 u = (pos - grid_p0.xyz) * grid_invStep.xyz;
    float4 fg = fe3d_pbc_comb(u, grid_ns.xyz, BsplinePLQ, PLQH, xqs, yqs);
    return fg.w;
}


inline float2 fe1Dcomb_tex(__read_only image3d_t img,
                           sampler_t smp,
                           const float4 C,             // Coefficients for combining P,L,Q,H
                           const float4 pz,            // B-spline basis functions for z (B0, B1, B2, B3)
                           const float4 dz,            // B-spline derivative basis functions for z (B'0, B'1, B'2, B'3)
                           const int ix, const int iy, // X and Y integer coordinates
                           const int4 qz)    // 4 integer Z coordinates
{
    // Combine Pauli, London, Coulomb, H-bond components using C = (P,L,Q,H) for each grid point
    const float4 cs = (float4)(
        dot(C, read_imagef(img, smp, (int4)(ix, iy, qz.x, 0))),
        dot(C, read_imagef(img, smp, (int4)(ix, iy, qz.y, 0))),
        dot(C, read_imagef(img, smp, (int4)(ix, iy, qz.z, 0))),
        dot(C, read_imagef(img, smp, (int4)(ix, iy, qz.w, 0)))
    );
    // Interpolate energy and its derivative w.r.t. u_z using B-spline basis
    return (float2)(
        dot(pz, cs), // Energy
        dot(dz, cs)  // dEnergy/du_z
    );
}

// 2D B-spline interpolation in YZ plane, for a given X-coordinate slice
// Calls fe1Dcomb_tex four times.
// Returns (dEnergy/duy, dEnergy/duz, Energy)
inline float3 fe2d_comb_tex(__read_only image3d_t img,
                            sampler_t smp,
                            const int  ix,         // Single X integer coordinate for this slice
                            const int4 qy,         // 4 integer Y coordinates
                            const int4 qz,         // 4 integer Z coordinates
                            const float4 C,        // Coefficients for combining P,L,Q,H
                            const float4 pz, const float4 dz,  // Basis for Z
                            const float4 py, const float4 dy)  // Basis for Y
{
    // Interpolate along Z for 4 different Y lines (at the given ix)
    const float2 fe0 = fe1Dcomb_tex(img, smp, C, pz, dz, ix, qy.x, qz);
    const float2 fe1 = fe1Dcomb_tex(img, smp, C, pz, dz, ix, qy.y, qz);
    const float2 fe2 = fe1Dcomb_tex(img, smp, C, pz, dz, ix, qy.z, qz);
    const float2 fe3 = fe1Dcomb_tex(img, smp, C, pz, dz, ix, qy.w, qz);

    // feN.x is Energy(yN,      uz_interp)
    // feN.y is dEnergy/duz(yN, uz_interp)

    // Interpolate along Y using results from 1D Z-interpolation
    return (float3)(
        dot(dy, (float4)(fe0.x, fe1.x, fe2.x, fe3.x)),     // dEnergy/du_y = sum_j (B'_j(u_y) * Energy(y_j, u_z_interp))
        dot(py, (float4)(fe0.y, fe1.y, fe2.y, fe3.y)),     // dEnergy/du_z = sum_j (B_j(u_y) * dEnergy/du_z(y_j, u_z_interp))
        dot(py, (float4)(fe0.x, fe1.x, fe2.x, fe3.x))      // Energy       = sum_j (B_j(u_y) * Energy(y_j, u_z_interp))
    );
}

// 3D B-spline interpolation for force and energy
// u: normalized coordinates (fractional cell coordinates)
// n: dimensions of the B-spline grid (texture dimensions)
// img: 3D texture storing (Pauli, London, Coulomb, H-bond_correction) potential values
// PLQH: coefficients to combine the 4 potential components
// xqis, yqis: precomputed PBC index patterns for X and Y dimensions
// Returns (dEnergy/dux, dEnergy/duy, dEnergy/duz, Energy)
inline float4 fe3d_pbc_comb_tex(const float3 u,
                                const int3 n,
                                __read_only image3d_t img,
                                sampler_t smp,
                                const float4 PLQH,
                                __local const int4* xqis, // Patterns from make_inds_pbc for x-dim
                                __local const int4* yqis) // Patterns from make_inds_pbc for y-dim
{
    // Integer part of u (knot index preceding the point)
    // Matches original code's floor logic for ix, iy, iz
    int ix = (int)u.x;
    int iy = (int)u.y;
    int iz = (int)u.z;
    if (u.x < 0) ix--;
    if (u.y < 0) iy--;
    if (u.z < 0) iz--; // Also apply floor logic to z

    // Fractional part of u (position within the cell defined by knot ix, iy, iz)
    const float tx = u.x - ix;
    const float ty = u.y - iy;
    const float tz = u.z - iz;

    // B-spline interpolation requires 4 knots starting from index (i-1).
    // The indices needed are (i-1, i, i+1, i+2).
    const int ix_knot_start = ix - 1;
    const int iy_knot_start = iy - 1;
    const int iz_knot_start = iz - 1;

    // Boundary condition for Z: if iz_raw_knot is too close to edge, return zero.
    // iz_raw_knot must be in [1, n.z - 3] for full 4-knot support (iz_knot_start must be >=0, iz_knot_start+3 must be < n.z).
    if ((iz < 1) || (iz >= n.z - 2)) {
        return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
    // Absolute Z indices (no PBC for Z based on this check and index range)
    const int4 qz = (int4)(iz_knot_start, iz_knot_start + 1, iz_knot_start + 2, iz_knot_start + 3);

    // Apply PBC for X and Y dimensions to get base knot index for choose_inds_pbc_3
    // The base knot index for choose_inds_pbc_3 should be the starting index *after* modulo,
    // i.e., (ix-1) % n.x.
    const int ix_pbc_base = modulo(ix_knot_start, n.x);
    const int iy_pbc_base = modulo(iy_knot_start, n.y);

    // Get the 4 absolute integer grid indices for X and Y using PBC logic
    // These indices will be used directly in read_imagef
    const int4 qx = choose_inds_pbc_3(ix_pbc_base, n.x, xqis);
    const int4 qy = choose_inds_pbc_3(iy_pbc_base, n.y, yqis);

    // Calculate B-spline basis functions and their derivatives
    const float4 bz = basis(tz);  const float4 dz = dbasis(tz);
    const float4 by = basis(ty);  const float4 dy = dbasis(ty);
    const float4 bx = basis(tx);  const float4 dx = dbasis(tx);

    // Interpolate along YZ for 4 different X planes
    // E#.x = dE/duy, E#.y = dE/duz, E#.z = E, all at (qx.#, u_y_interp, u_z_interp)
    const float3 E1 = fe2d_comb_tex(img, smp, qx.x, qy, qz, PLQH, bz, dz, by, dy);
    const float3 E2 = fe2d_comb_tex(img, smp, qx.y, qy, qz, PLQH, bz, dz, by, dy);
    const float3 E3 = fe2d_comb_tex(img, smp, qx.z, qy, qz, PLQH, bz, dz, by, dy);
    const float3 E4 = fe2d_comb_tex(img, smp, qx.w, qy, qz, PLQH, bz, dz, by, dy);

    // Interpolate along X using results from 2D YZ-interpolation
    // Result is (dE/dux, dE/duy, dE/duz, E_total)
    return (float4)(
        dot(dx, (float4)(E1.z, E2.z, E3.z, E4.z)),     // dEnergy/du_x = sum_i (B'_i(u_x) * Energy(x_i, u_y_interp, u_z_interp))
        dot(bx, (float4)(E1.x, E2.x, E3.x, E4.x)),     // dEnergy/du_y = sum_i (B_i(u_x) * dEnergy/du_y(x_i, u_y_interp, u_z_interp))
        dot(bx, (float4)(E1.y, E2.y, E3.y, E4.y)),     // dEnergy/du_z = sum_i (B_i(u_x) * dEnergy/du_z(x_i, u_y_interp, u_z_interp))
        dot(bx, (float4)(E1.z, E2.z, E3.z, E4.z))      // Energy       = sum_i (B_i(u_x) * Energy(x_i, u_y_interp, u_z_interp))
    );
}



// ======================================================================
//                           sampleGridFF()
// ======================================================================
// this is just to test interpolation of Grid-Force-Field (GFF) on GPU
//__attribute__((reqd_work_group_size(1,1,1)))
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

