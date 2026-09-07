// ported from SPAMMM/kernels/Forces.cl (pairwise LJ/Morse/Coulomb helpers, requires common.cl first)
// Forces.cl - Inline pairwise potential functions (not __kernel)
//
// Provides force+energy evaluation for pairwise interactions used by
// nonbonded.cl, spff.cl, uff.cl, and other modules:
//   - getLJQH: Lennard-Jones 12-6 + damped Coulomb + H-bond correction
//   - getMorseQH: Morse potential + damped Coulomb + H-bond correction
//   - getMorsePLQH: Morse with Pauli/London decomposition (for GridFF)
//   - getCoulomb: damped Coulomb only
//   - Energy/decomposition macros: MODEL_LJQH2_PAIR, MODEL_MorseQ_PAIR, etc.
//
// All functions return float4 = (Fx, Fy, Fz, E) — force vector + energy.
// Requires: common.cl to be concatenated before this file.

//>>>function  getLJQH (dp,REQ,ffpars)

inline float4 getLJQH( float3 dp, float4 REQ, float R2damp ){
    // ---- Electrostatic (damped Coulomb potential)
    float   r2    = dot(dp,dp);
    float   ir2_  = 1.f/(  r2 +  R2damp);              // inverse distance squared and damped
    float   Ec    =  COULOMB_CONST*REQ.z*sqrt( ir2_ ); // Ec = Q1*Q2/sqrt(r^2+R2damp)
    // --- Lennard-Jones and Hydrogen bond correction
    float  ir2 = 1.f/r2;          // inverse distance squared
    float  u2  = REQ.x*REQ.x*ir2; // u2 = (R0/r)^2
    float  u6  = u2*u2*u2;        // u6 = (R0/r)^6
    float vdW  = u6*REQ.y;        // vdW = E0*(R0/r)^6
    float E    =       (u6-2.f)*vdW     + Ec  ;     // E = E0*(R0/r)^6 - E0*(R0/r)^12 + Q1*Q2/sqrt(r^2+R2damp)
    float fr   = -12.f*(u6-1.f)*vdW*ir2 - Ec*ir2_;  // fr = -12*E0*( (R0/r)^8/r + 12*E0*(R0/r)^14) - Q1*Q2/(r^2+R2damp)^1.5
    return  (float4){ dp*fr, E };
}

//>>>function getMorseQH (dp,REQH,ffpars )

inline float4 getMorseQH( float3 dp,  float4 REQH, float K, float R2damp ){
    float r2    = dot(dp,dp);
    float ir2_  = 1/(r2+R2damp);
    float r     = sqrt( r2   );
    float ir_   = sqrt( ir2_ );     // ToDo: we can save some cost if we approximate r^2 = r^2 + R2damp;
    float e     = exp ( K*(r-REQH.x));
    //double e2    = e*e;
    //double fMors =  E0*  2*K*( e2 -   e ); // Morse
    //double EMors =  E0*      ( e2 - 2*e );
    float   Ae  = REQH.y*e;
    float fMors = Ae*  2*K*(e - 1); // Morse
    float EMors = Ae*      (e - 2);
    float Eel   = COULOMB_CONST*REQH.z*ir_;
    float fr    = fMors/r - Eel*ir2_ ;
    return  (float4){ dp*fr, EMors+Eel };
}

#if 0  // --- macro templates below are not compiled directly; used via preprocess_opencl_source ---
//>>>macro MODEL_LJQH2_PAIR
{
    // Distance safeguards
    float r_safe  = fmax(r, R2SAFE);
    float inv_r = 1.0f / r_safe;

    // Electrostatic
    float dE_dQ = inv_r * COULOMB_CONST;
    float Eel   = Q * dE_dQ;

    // Lennard-Jones 12-6 with H2 scaling of attraction
    float u   = R0 * inv_r;
    float u3  = u*u*u;
    float u6  = u3*u3;
    float u6p = (1.f + H) * u6;
    float dE_dE0 = u6 * (u6p - 2.f);
    float ELJ    =  E0 * dE_dE0;

    // Accumulate derivatives for atom i
    float dE_dR0 = 12.f * (E0/R0) * u6 * (u6p - 1.f);
    float dE_dH  = -E0 * u6 * u6;
    fREQi.x +=  -dE_dR0;
    fREQi.y +=  -dE_dE0 *        REQj.y;
    fREQi.z +=  -dE_dQ  *        REQj.z;
    fREQi.w +=  dE_dH  *        REQj.w * sH;

    // Accumulate energy
    Ei += ELJ + Eel;
}

//>>>macro MODEL_MorseQ_PAIR_DECOMP
{
    // Morse potential decomposition
    // E_morse = E0 * [(1+H)*e^2 - 2*e] + Q/r
    // where e = exp(-alpha*(r-R0))
    
    const float alpha = 1.8f;
    float e    = exp( -alpha * ( r - R0 ) );
    float e2   = e * e;
    
    // Pauli: repulsive part (e^2 term without H)
    pauli  += E0 * e2;
    
    // London: attractive part (-2*e term)
    london += -2.f * E0 * e;
    
    // H-bond: H-dependent correction (H*e^2 term)
    hbond  += E0 * H * e2;
    
    // Electrostatic: Coulomb term
    electro += Q * inv_r * COULOMB_CONST;
}

//>>>macro MODEL_LJQH2_PAIR_DECOMP
{
    // Distance safeguards
    float r_safe  = fmax(r, R2SAFE);
    float inv_r = 1.0f / r_safe;
    
    // Electrostatic
    electro += Q * inv_r * COULOMB_CONST;
    
    // Lennard-Jones 12-6 with H2 scaling
    float u   = R0 * inv_r;
    float u3  = u*u*u;
    float u6  = u3*u3;
    float u12 = u6*u6;
    
    // Decompose LJ energy:
    // ELJ = E0 * u6 * ((1+H)*u6 - 2.f)
    //     = E0 * (1+H) * u12 - 2*E0 * u6
    //     = E0*u12 + E0*H*u12 - 2*E0*u6
    
    pauli  += E0 * u12;           // Repulsive r^-12 term
    london += -2.f * E0 * u6;     // Attractive r^-6 term (without H)
    hbond  += E0 * H * u12;       // H-bond correction (H-dependent)
}

//>>>macro MODEL_LJr8QH2_PAIR
{
    // Electrostatic
    float dE_dQ = inv_r * COULOMB_CONST;
    float Eel   = Q * dE_dQ;
    // r^-8 variant (8-6 like) with H2 scaling of r^-2 factor
    float u   = R0 * inv_r;
    float u2  = u*u;
    float u4  = u2*u2;
    float u6  = u4*u2;
    float u2p = (1.f + H) * u2;
    float dE_dE0 = u6 * (3.f * u2p - 4.f);
    float ELJ    =  E0 * dE_dE0;
    // Accumulate energy
    Ei += ELJ + Eel;

    // Accumulate derivatives for atom i
    float dE_dR0 = 24.f * (E0/R0) * u6 * (u2p - 1.f);
    float dE_dH  = -3.f * E0 * u6 * u2;
    fREQi.x +=  -dE_dR0;
    fREQi.y +=  -dE_dE0 *        REQj.y;
    fREQi.z +=  -dE_dQ  *        REQj.z;
    fREQi.w +=  dE_dH  *        REQj.w * sH;
}

//>>>macro MODEL_MorseQ_PAIR
{
    // Electrostatic
    float dE_dQ = inv_r * COULOMB_CONST;
    float Eel   = Q * dE_dQ;
    // Morse with alpha matching CPU kMorse = 1.8
    const float alpha = 1.8f;
    float e    = exp( -alpha * ( r - R0 ) );
    float e2   = e * e;
    float e2p  = (1.f + H) * e2;
    float dE_dE0 = e2p - 2.f * e;
    float ELJ    =  E0 * dE_dE0;
    Eij = ELJ + Eel;

    // Accumulate derivatives for atom i
    float dE_dR0 = 2.f * alpha * E0 * ( e2p - e );
    float dE_dH = - E0 * e2;
    // fREQi.x +=  dE_dR0;
    // fREQi.y +=  dE_dE0 *        REQj.y;
    // fREQi.z +=  dE_dQ  *        REQj.z;
    // fREQi.w +=  dE_dH *        REQj.w * sH;
    fij.x = -dE_dR0;                   // dEtot/dR0_i (match CPU)
    fij.y = -dE_dE0 * REQj.y;          // dEtot/dE0_i (match CPU)
    fij.z = -dE_dQ  * REQj.z;          // dEtot/dQi   (match CPU)
    fij.w =  dE_dH  * REQj.w * sH;     // dEtot/dH2i  (match CPU)
    // Accumulate energy
    
}

// Energy-only variants of the above models. These only accumulate Ei.
//>>>macro ENERGY_LJQH2_PAIR
{
    // Electrostatic
    float Eel   = Q * (inv_r * COULOMB_CONST);
    // Lennard-Jones 12-6 with H2 scaling on attraction
    float u   = R0 * inv_r;
    float u3  = u*u*u;
    float u6  = u3*u3;
    float u6p = (1.f + H)  *  u6;
    float ELJ =  E0 * ( u6 * (u6p - 2.f) );
    Ei += ELJ + Eel;
}

//>>>macro ENERGY_LJr8QH2_PAIR
{
    float Eel   = Q * (inv_r * COULOMB_CONST);
    float u   = R0 *   inv_r;
    float u2  = u*u;
    float u4  = u2*u2;
    float u6  = u4*u2;
    float u2p = (1.f + H) * u2;
    float ELJ =  E0 * ( u6 * (3.f * u2p - 4.f) );
    Ei += ELJ + Eel;
}

//>>>macro ENERGY_MorseQ_PAIR
{
    //float Eel   = Q*2.0 * (inv_r * COULOMB_CONST);
    float Eel   = Q * (inv_r * COULOMB_CONST);
    // Morse with alpha matching CPU kMorse = 1.8
    const float alpha = 1.8f;
    float e    = exp( -alpha * ( r - R0 ) );
    float e2   = e * e;
    float e2p  = (1.f + H) * e2;
    float ELJ  =  E0 * ( e2p - 2.f * e );
    Ei += ELJ + Eel;
}

#endif  // --- end of macro templates ---

// ---- Damped Coulomb potential ----
// evaluate damped Coulomb potential and force
inline float4 getCoulomb( float3 dp, float R2damp ){
    // ---- Electrostatic
    float   r2    = dot(dp,dp);
    float   ir2_  = 1.f/(  r2 + R2damp);
    float   E    = COULOMB_CONST*sqrt( ir2_ );
    return  (float4){ dp*-E*ir2_, E };
}

// ---- Morse with Pauli/London decomposition ----
inline float4 getMorsePLQH( float3 dp, float4 REQH, float4 PLQH, float K, float R2damp ){
    float r2    = dot(dp,dp);
    float ir2_  = 1/(r2+R2damp);
    float r     = sqrt( r2   );
    float ir_   = sqrt( ir2_ );
    float e     = exp ( K*(r-REQH.x));
    float Ee    = REQH.y*e;
    float EP    = Ee*e      * PLQH.x;
    float EL    = -2.0f*Ee  * PLQH.y;
    float EQ    = COULOMB_CONST*REQH.z*ir_ * PLQH.z;
    float frP   = (2.0f*K*EP)/r;
    float frL   = (-2.0f*K*Ee*PLQH.y)/r;
    float frQ   = -EQ*ir2_;
    return (float4){ dp*(frP+frL+frQ), EP+EL+EQ };
}

// ---- Unified compact exponential (power n=8) + soft radius ----
// Mirrors topics/NonBondingFFs/fit_radial.py::compact_exp_force_over_r
//
//   y = max(0, 1 - beta*(rho-R0)/8)^8
//   V = E0 * y * (alpha*y - (1+alpha))
//   rho = r^2 / (sqrt(r^2+w^2) + w)     // one sqrt; no r=sqrt(r2)
//
// Returns float2(E, f_over_r) with F_vec = f_over_r * dr.
// alpha=1,w=0 → compact Morse; alpha=0,w>0 → blunt attractive well.
inline float2 compact_exp_pair_EF( float3 dr, float R0, float E0, float alpha, float w, float beta ){
    const float eps = 1e-12f;
    float r2  = dot(dr, dr);
    float rw  = sqrt(r2 + w*w);
    float rho = r2 / fmax(rw + w, eps);
    float u   = fmax(0.0f, 1.0f - (beta * 0.125f) * (rho - R0)); // /8
    float u2  = u*u;
    float u4  = u2*u2;
    float y   = u4*u4;          // u^8
    float u7  = u4*u2*u;        // u^7
    float E   = E0 * y * (alpha*y - (1.0f + alpha));
    float f_over_r = E0 * beta * (2.0f*alpha*y - (1.0f + alpha)) * u7 / fmax(rw, eps);
    return (float2)(E, f_over_r);
}

// Unified PairFF site-pair primitive used by rigid dynamics and probe maps.
// REQ.x/y/z/w = radius, sqrt(E), charge/pseudo-charge, blunt width;
// type 0 is a real atom, types 1/2 are directional dummy sites.
// Returns (Fx,Fy,Fz,E) with the same operation order as the unified kernels.
inline float4 pairff_unified_site_EF(float3 dp, float4 REQ_i, int type_i,
                                     float4 REQ_j, int type_j, float beta){
    const float gi = (type_i == 0) ? 1.0f : 0.0f;
    const float gj = (type_j == 0) ? 1.0f : 0.0f;
    const float gij = gi * gj;
    const float R0 = gij * (REQ_i.x + REQ_j.x);
    const float w = REQ_i.w + REQ_j.w;
    const float alpha = gij;
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);
    const float rho_c = R0 + inv_beta_n;
    const float rc2 = rho_c * (rho_c + 2.0f * w);
    const float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
    const float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
    const float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
    const float r2 = dot(dp, dp);
    float3 f = (float3)(0.0f);
    float E = 0.0f;
    if (E0 != 0.0f && r2 <= rc2){
        const float2 ev = compact_exp_pair_EF(dp, R0, E0, alpha, w, beta);
        E += ev.x;
        f += dp * ev.y;
    }
    if (gij > 0.5f){
        const float Q = REQ_i.z * REQ_j.z;
        const float r2d = r2 + R2SAFE;
        const float ir2d = 1.0f / r2d;
        const float sqr_ir2d = sqrt(ir2d);
        E += COULOMB_CONST * Q * sqr_ir2d;
        f += dp * (COULOMB_CONST * Q * ir2d * sqr_ir2d);
    }
    return (float4)(f.x, f.y, f.z, E);
}
