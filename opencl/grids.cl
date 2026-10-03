// grids.cl — Common 3D grid ops for density / FDBM (downsample, Gaussians, axpy)
//
// Load with: common.cl + grids.cl  (or grids.cl alone — self-contained atomics)
//
// Design: SOURCE-DRIVEN projection (scatter). Each source voxel is treated as a
// point charge Q = ρ·dV at its cell center. World position may be affinely mapped
// into the destination frame (shift + rotation), then deposited onto the 8
// neighboring dest voxels with trilinear (bilinear³) weights.
//
// Dipole preservation: trilinear weights satisfy Σ w_c r_c = r_point, so
//   Σ_src Q_i r_i  =  Σ_dst Q_j r_j
// (charge and dipole about the same origin) when all deposits land in-bounds.
//
// Density units: dest receives Δρ = (Q · w) / dV_dst  so ∫ρ is conserved.

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

inline void grids_atomic_add_f(__global float* addr, float val) {
    union { uint u; float f; } old, neu;
    old.u = as_uint(*addr);
    while (1) {
        neu.f = old.f + val;
        uint prev = atomic_cmpxchg((__global uint*)addr, old.u, neu.u);
        if (prev == old.u) break;
        old.u = prev;
        old.f = as_float(prev);
    }
}

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

inline int grids_flat(int ix, int iy, int iz, int nx, int ny, int nz) {
    // layout (nx,ny,nz) C-order: ix major
    return (ix * ny + iy) * nz + iz;
}

inline float3 grids_rot(float3 v, float3 R0, float3 R1, float3 R2) {
    return (float3)(dot(R0, v), dot(R1, v), dot(R2, v));
}

// Deposit charge Q at fractional dest coords (gx,gy,gz) onto 8 neighbors.
// gx = (x - origin_x) / step_x  (can be non-integer).
inline void grids_deposit_trilinear(
    __global float* dst, int nx, int ny, int nz,
    float gx, float gy, float gz, float dQ_over_dV
) {
    int i0 = (int)floor(gx);
    int j0 = (int)floor(gy);
    int k0 = (int)floor(gz);
    float fx = gx - (float)i0;
    float fy = gy - (float)j0;
    float fz = gz - (float)k0;
    float wx0 = 1.f - fx, wx1 = fx;
    float wy0 = 1.f - fy, wy1 = fy;
    float wz0 = 1.f - fz, wz1 = fz;

    // 8 corners
    int is[2] = {i0, i0 + 1};
    int js[2] = {j0, j0 + 1};
    int ks[2] = {k0, k0 + 1};
    float wxs[2] = {wx0, wx1};
    float wys[2] = {wy0, wy1};
    float wzs[2] = {wz0, wz1};

    for (int a = 0; a < 2; a++) {
        int ix = is[a];
        if (ix < 0 || ix >= nx) continue;
        for (int b = 0; b < 2; b++) {
            int iy = js[b];
            if (iy < 0 || iy >= ny) continue;
            for (int c = 0; c < 2; c++) {
                int iz = ks[c];
                if (iz < 0 || iz >= nz) continue;
                float w = wxs[a] * wys[b] * wzs[c];
                if (w == 0.f) continue;
                grids_atomic_add_f(&dst[grids_flat(ix, iy, iz, nx, ny, nz)], dQ_over_dV * w);
            }
        }
    }
}

// ── Clear / fill ─────────────────────────────────────────────────────────────
__kernel void grid_fill(const int n, __global float* a, const float val) {
    int i = get_global_id(0);
    if (i >= n) return;
    a[i] = val;
}

__kernel void grid_axpy(
    const int n,
    const float alpha, __global const float* x,
    const float beta,  __global float* y
) {
    // y := alpha * x + beta * y
    int i = get_global_id(0);
    if (i >= n) return;
    y[i] = alpha * x[i] + beta * y[i];
}

__kernel void grid_scale(const int n, __global float* a, const float s) {
    int i = get_global_id(0);
    if (i >= n) return;
    a[i] *= s;
}

__kernel void grid_subtract(
    const int n,
    __global const float* a,
    __global const float* b,
    __global float* out
) {
    // out = a - b
    int i = get_global_id(0);
    if (i >= n) return;
    out[i] = a[i] - b[i];
}

// ── Project / downsample (scatter, dipole-preserving trilinear) ──────────────
// One work-item per SOURCE voxel.
// p_dst_world = R * p_src_world + t
// Then gx = (p_dst_world - origin_dst) / step_dst  (component-wise)
//
// R rows: R0,R1,R2 (identity if no rotation). t in Å (dest world).
// vol_scale = dV_src / dV_dst  → density conservation when depositing ρ.
__kernel void project_density_trilinear(
    __global const float* src,
    const int   nx_s, const int ny_s, const int nz_s,
    const float ox_s, const float oy_s, const float oz_s,
    const float sx_s, const float sy_s, const float sz_s,
    const float3 R0, const float3 R1, const float3 R2,
    const float3 t,
    __global float* dst,
    const int   nx_d, const int ny_d, const int nz_d,
    const float ox_d, const float oy_d, const float oz_d,
    const float sx_d, const float sy_d, const float sz_d,
    const float vol_scale
) {
    int gid = get_global_id(0);
    int nsrc = nx_s * ny_s * nz_s;
    if (gid >= nsrc) return;

    float rho = src[gid];
    if (rho == 0.f) return;

    int iz = gid % nz_s;
    int t1 = gid / nz_s;
    int iy = t1 % ny_s;
    int ix = t1 / ny_s;

    // source voxel center (Å)
    float3 p_s = (float3)(
        ox_s + ((float)ix + 0.5f) * sx_s,
        oy_s + ((float)iy + 0.5f) * sy_s,
        oz_s + ((float)iz + 0.5f) * sz_s
    );
    float3 p_d = grids_rot(p_s, R0, R1, R2) + t;

    // continuous dest index of the point (cell-centered dest uses same center convention:
    // corner indices; deposit at (p - origin)/step which may be .5-aligned)
    float gx = (p_d.x - ox_d) / sx_d;
    float gy = (p_d.y - oy_d) / sy_d;
    float gz = (p_d.z - oz_d) / sz_d;

    // Q/dV_dst = rho * (dV_src/dV_dst) = rho * vol_scale
    grids_deposit_trilinear(dst, nx_d, ny_d, nz_d, gx, gy, gz, rho * vol_scale);
}

// ── Neutral-atom Gaussians: ρ += sign * Z * N(r; σ)  (grid-parallel) ─────────
// atoms: float4 (x,y,z,Z) in Å / electrons. sign=+1 add, -1 subtract.
__kernel void add_gaussian_atoms(
    __global float* grid,
    const int nx, const int ny, const int nz,
    const float ox, const float oy, const float oz,
    const float sx, const float sy, const float sz,
    __global const float4* atoms,
    const int natoms,
    const float sigma,
    const float sign
) {
    int gid = get_global_id(0);
    int n = nx * ny * nz;
    if (gid >= n) return;

    int iz = gid % nz;
    int t1 = gid / nz;
    int iy = t1 % ny;
    int ix = t1 / ny;

    float3 p = (float3)(
        ox + (float)ix * sx,
        oy + (float)iy * sy,
        oz + (float)iz * sz
    );

    float inv2s2 = 1.f / (2.f * sigma * sigma);
    float norm = native_recip(pow(2.f * (float)M_PI_F * sigma * sigma, 1.5f));
    float acc = 0.f;
    for (int ia = 0; ia < natoms; ia++) {
        float4 a = atoms[ia];
        float3 d = p - a.xyz;
        float r2 = dot(d, d);
        acc += a.w * norm * native_exp(-r2 * inv2s2);
    }
    grid[gid] += sign * acc;
}

// Faster splat from atoms (few atoms): each work-item = one atom voxel neighborhood.
// Bounding box: ±nsig * sigma in each direction (nsig typically 4–5).
__kernel void splat_gaussian_atoms(
    __global float* grid,
    const int nx, const int ny, const int nz,
    const float ox, const float oy, const float oz,
    const float sx, const float sy, const float sz,
    __global const float4* atoms,
    const int natoms,
    const float sigma,
    const float sign,
    const float nsig
) {
    int ia = get_global_id(0);
    if (ia >= natoms) return;
    float4 a = atoms[ia];
    if (a.w == 0.f) return;

    float inv2s2 = 1.f / (2.f * sigma * sigma);
    float norm = native_recip(pow(2.f * (float)M_PI_F * sigma * sigma, 1.5f));
    float R = nsig * sigma;
    int ix0 = max(0, (int)floor((a.x - R - ox) / sx));
    int iy0 = max(0, (int)floor((a.y - R - oy) / sy));
    int iz0 = max(0, (int)floor((a.z - R - oz) / sz));
    int ix1 = min(nx - 1, (int)ceil ((a.x + R - ox) / sx));
    int iy1 = min(ny - 1, (int)ceil ((a.y + R - oy) / sy));
    int iz1 = min(nz - 1, (int)ceil ((a.z + R - oz) / sz));

    for (int ix = ix0; ix <= ix1; ix++) {
        float x = ox + (float)ix * sx;
        float dx = x - a.x;
        for (int iy = iy0; iy <= iy1; iy++) {
            float y = oy + (float)iy * sy;
            float dy = y - a.y;
            for (int iz = iz0; iz <= iz1; iz++) {
                float z = oz + (float)iz * sz;
                float dz = z - a.z;
                float r2 = dx*dx + dy*dy + dz*dz;
                float val = sign * a.w * norm * native_exp(-r2 * inv2s2);
                grids_atomic_add_f(&grid[grids_flat(ix, iy, iz, nx, ny, nz)], val);
            }
        }
    }
}

// Strip uniform monopole: ρ -= (Σρ · dV) / V   (pass q_over_V = mean density to subtract)
__kernel void grid_add_const(const int n, __global float* a, const float c) {
    int i = get_global_id(0);
    if (i >= n) return;
    a[i] += c;
}
