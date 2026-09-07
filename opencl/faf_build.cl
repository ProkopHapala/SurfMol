// faf_build.cl — FAF construction kernels (macro/fragment library).
//
// Extracted from surface_spammm.cl. Contains surface potential / isosurface /
// Ewald2D build kernels as //>>>function blocks. Requires common.cl + Forces.cl
// concatenated first. Assembled by oclff::ClAssembler.
// See doc/topical_audit/gridff_faf.md.

//>>>function getSurfMorse (__kernel void getSurfMorse()
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

//>>>function compute_ewald_coefficients (__kernel void compute_ewald_coefficients()
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

//>>>function eval_potential_vacuum (__kernel void eval_potential_vacuum()
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

//>>>function eval_potential_full (__kernel void eval_potential_full()
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

//>>>function eval_potential_brute (__kernel void eval_potential_brute()
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

//>>>function eval_potential_cluster (__kernel void eval_potential_cluster()
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

//>>>function getSurfFlat (__kernel void getSurfFlat()
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

//>>>function getSurfaceIsoSurfMorse (__kernel void getSurfaceIsoSurfMorse()
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

//>>>function getSurfaceIsoGridFF (__kernel void getSurfaceIsoGridFF()
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

//>>>function addDipoleField (__kernel void addDipoleField()
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

