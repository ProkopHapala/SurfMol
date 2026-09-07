// ported from SPAMMM/kernels/common.cl (shared helpers, concatenated FIRST before SPFF.cl/Forces.cl)
// common.cl - Shared types, macros, and helper functions for all kernel modules
//
// This file is always concatenated FIRST before any other .cl file.
// It provides:
//   - cl_Mat3: 3x3 matrix type (lattice vectors, rotation matrices)
//   - Physical constants: COULOMB_CONST, const_kB, R2SAFE
//   - Macros: float4Zero, float3Zero, EXCL_MAX (max exclusions per atom)
//   - Math helpers: modulo, udiv_cmplx (unit complex division), rotMat/rotMatT
//   - Mixing rules: mixREQ_arithmetic (LJ combining rules: R_ij=R_i+R_j, E_ij=E_i*E_j)
//   - clampForce: limit force magnitude to prevent numerical instability
//
// No __kernel functions here — only inline helpers and definitions.

#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

typedef struct __attribute__ ((packed)){
    float4 a;
    float4 b;
    float4 c;
} cl_Mat3;

#define  float4Zero  (float4){0.f,0.f,0.f,0.f}
#define  float3Zero  (float3){0.f,0.f,0.f}
#define  float2Zero  (float3){0.f,0.f,0.f}

#define EXCL_MAX 16

#define R2SAFE          1e-4f
#define COULOMB_CONST   14.3996448915f       // [ eV*Ang/e^2 ]
#define const_kB        8.617333262145e-5f   // [ eV/K ]

#ifndef DBG_UFF
#define DBG_UFF 0
#endif

// ---- Math helpers ----
inline int modulo(const int i, const int m) {
    int result = i % m;
    if (result < 0) { result += m; }
    return result;
}

inline float2 udiv_cmplx( float2 a, float2 b ){ return (float2)( a.x*b.x + a.y*b.y,  a.y*b.x - a.x*b.y ); }

inline float3 rotMat ( float3 v, float3 a, float3 b, float3 c ){ return (float3)(dot(v,a),dot(v,b),dot(v,c)); }
inline float3 rotMatT( float3 v, float3 a, float3 b, float3 c ){ return a*v.x + b*v.y + c*v.z; }

// ---- Mixing rules ----
inline float4 mixREQ_arithmetic( float4 REQi, float4 REQj ){
    float R0 = REQi.x + REQj.x;
    float E0 = REQi.y * REQj.y;
    float Q  = REQi.z * REQj.z;
    float H  = REQi.w * REQj.w; if(H>0){H=0;}
    return (float4)(R0, E0, Q, H);
}

inline float3 clampForce( float3 f, float Fmax2 ){
    float f2 = dot(f,f);
    if(f2 > Fmax2){ f *= sqrt(Fmax2/f2); }
    return f;
}
