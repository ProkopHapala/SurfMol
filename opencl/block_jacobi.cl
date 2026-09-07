// ============================================================================
// block_jacobi.cl
// Block Jacobi GPU kernel for truss elasticity using local memory.
//
// Copied verbatim 2026-08-29 from NumericalMathPlayground/topics/LinarElasticity/kernels_block_jacobi.cl
// for later adaptation to SurfMol buffer conventions. See notes/designs/2026-08-29_modal_relaxation_design_spec.md.
// Original: truss-elasticity block-Jacobi smoother (axial-spring stiffness A = M/dt² + K).
//
// One workgroup = one patch/cluster.  Cluster size <= WG_SIZE (compile-time).
// If cluster_size < WG_SIZE, remaining threads are idle (padded design).
//
// Compile-time defines:
//   WG_SIZE, CLUSTER_SIZE, NMAX_NEIGH, DIM, N_STEPS, OMEGA
// ============================================================================

#ifndef DIM
#define DIM 3
#endif

#if DIM == 2
typedef float2 Vec;
#else
typedef float3 Vec;
#endif

// ---- Address-space-generic vector load/store using inline functions ----

inline Vec load_vec_g(__global const float *x, int n) {
    Vec v;
    v.x = x[n * DIM + 0];
    v.y = x[n * DIM + 1];
    #if DIM == 3
    v.z = x[n * DIM + 2];
    #endif
    return v;
}

inline void store_vec_g(__global float *x, int n, Vec v) {
    x[n * DIM + 0] = v.x;
    x[n * DIM + 1] = v.y;
    #if DIM == 3
    x[n * DIM + 2] = v.z;
    #endif
}

inline Vec load_vec_l(__local const float *x, int n) {
    Vec v;
    v.x = x[n * DIM + 0];
    v.y = x[n * DIM + 1];
    #if DIM == 3
    v.z = x[n * DIM + 2];
    #endif
    return v;
}

inline void store_vec_l(__local float *x, int n, Vec v) {
    x[n * DIM + 0] = v.x;
    x[n * DIM + 1] = v.y;
    #if DIM == 3
    x[n * DIM + 2] = v.z;
    #endif
}

inline float dotv(Vec a, Vec b) {
    #if DIM == 2
    return a.x * b.x + a.y * b.y;
    #else
    return a.x * b.x + a.y * b.y + a.z * b.z;
    #endif
}

// ============================================================================
// Kernel: block_jacobi_step
// ============================================================================

__kernel void block_jacobi_step(
    __global const float *x,
    __global const float *b,
    __global float *x_out,
    __global const int *patch_node_ids,
    __global const int *patch_nneigh,
    __global const char *patch_core_mask,
    __global const float *k_vals,
    __global const float *n_dirs,
    __global const int *nneigh,
    __global const float *mass_dt2,
    __global const float *Dinv_global,
    __global const int *neighs,
    int n_patches
) {
    int patch_id = get_group_id(0);
    int lid = get_local_id(0);
    if (patch_id >= n_patches) return;

    __local float x_loc[CLUSTER_SIZE * DIM];
    __local float xnext_loc[CLUSTER_SIZE * DIM];
    __local float Dinv_loc[CLUSTER_SIZE * DIM * DIM];
    __local float b_loc[CLUSTER_SIZE * DIM];

    int node_global = -1;
    bool is_active = (lid < CLUSTER_SIZE);
    if (is_active) {
        node_global = patch_node_ids[patch_id * CLUSTER_SIZE + lid];
        is_active = (node_global >= 0);
    }

    if (is_active) {
        store_vec_l(x_loc, lid, load_vec_g(x, node_global));
        store_vec_l(b_loc, lid, load_vec_g(b, node_global));
        for (int d = 0; d < DIM * DIM; d++)
            Dinv_loc[lid * DIM * DIM + d] =
                Dinv_global[node_global * DIM * DIM + d];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 0; step < N_STEPS; step++) {
        if (is_active) {
            Vec xi = load_vec_l(x_loc, lid);
            float mi = mass_dt2[node_global];
            Vec Axi;
            Axi.x = mi * xi.x;
            Axi.y = mi * xi.y;
            #if DIM == 3
            Axi.z = mi * xi.z;
            #endif

            int nn = nneigh[node_global];
            for (int e = 0; e < nn; e++) {
                int local_neigh = patch_nneigh[patch_id * CLUSTER_SIZE * NMAX_NEIGH + lid * NMAX_NEIGH + e];
                float ke = k_vals[node_global * NMAX_NEIGH + e];
                int dir_base = (node_global * NMAX_NEIGH + e) * DIM;
                Vec n_dir;
                n_dir.x = n_dirs[dir_base + 0];
                n_dir.y = n_dirs[dir_base + 1];
                #if DIM == 3
                n_dir.z = n_dirs[dir_base + 2];
                #endif

                Vec xj;
                if (local_neigh >= 0 && local_neigh < CLUSTER_SIZE) {
                    xj = load_vec_l(x_loc, local_neigh);
                } else {
                    int global_neigh = neighs[node_global * NMAX_NEIGH + e];
                    xj = load_vec_g(x, global_neigh);
                }

                Vec diff;
                diff.x = xi.x - xj.x;
                diff.y = xi.y - xj.y;
                #if DIM == 3
                diff.z = xi.z - xj.z;
                #endif
                float d = ke * dotv(diff, n_dir);
                Axi.x += d * n_dir.x;
                Axi.y += d * n_dir.y;
                #if DIM == 3
                Axi.z += d * n_dir.z;
                #endif
            }

            Vec bi = load_vec_l(b_loc, lid);
            Vec r;
            r.x = bi.x - Axi.x;
            r.y = bi.y - Axi.y;
            #if DIM == 3
            r.z = bi.z - Axi.z;
            #endif

            // dx = Dinv * r
            int didx = lid * DIM * DIM;
            Vec dx;
            dx.x = Dinv_loc[didx + 0] * r.x + Dinv_loc[didx + 1] * r.y;
            dx.y = Dinv_loc[didx + DIM] * r.x + Dinv_loc[didx + DIM + 1] * r.y;
            #if DIM == 3
            dx.x += Dinv_loc[didx + 2] * r.z;
            dx.y += Dinv_loc[didx + DIM + 2] * r.z;
            dx.z = Dinv_loc[didx + 2*DIM] * r.x + Dinv_loc[didx + 2*DIM + 1] * r.y + Dinv_loc[didx + 2*DIM + 2] * r.z;
            #endif

            Vec x_next;
            x_next.x = xi.x + OMEGA * dx.x;
            x_next.y = xi.y + OMEGA * dx.y;
            #if DIM == 3
            x_next.z = xi.z + OMEGA * dx.z;
            #endif
            store_vec_l(xnext_loc, lid, x_next);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (is_active) {
            for (int d = 0; d < DIM; d++)  x_loc[lid * DIM + d] = xnext_loc[lid * DIM + d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (is_active) {
        char is_core = patch_core_mask[patch_id * CLUSTER_SIZE + lid];
        if (is_core) {
            store_vec_g(x_out, node_global, load_vec_l(x_loc, lid));
        }
    }
}

// ============================================================================
// Kernel: compute_residual  (r = b - A*x, one thread per node)
// ============================================================================

__kernel void compute_residual(
    __global const float *x,
    __global const float *b,
    __global float *r_out,
    __global const int *neighs,
    __global const float *k_vals,
    __global const float *n_dirs,
    __global const int *nneigh,
    __global const float *mass_dt2,
    int n_nodes
) {
    int i = get_global_id(0);
    if (i >= n_nodes) return;

    Vec xi = load_vec_g(x, i);
    float mi = mass_dt2[i];
    Vec Axi;
    Axi.x = mi * xi.x;
    Axi.y = mi * xi.y;
    #if DIM == 3
    Axi.z = mi * xi.z;
    #endif

    int nn = nneigh[i];
    for (int e = 0; e < nn; e++) {
        int j = neighs[i * NMAX_NEIGH + e];
        float ke = k_vals[i * NMAX_NEIGH + e];
        int dir_base = (i * NMAX_NEIGH + e) * DIM;
        Vec n_dir;
        n_dir.x = n_dirs[dir_base + 0];
        n_dir.y = n_dirs[dir_base + 1];
        #if DIM == 3
        n_dir.z = n_dirs[dir_base + 2];
        #endif

        Vec xj = load_vec_g(x, j);
        Vec diff;
        diff.x = xi.x - xj.x;
        diff.y = xi.y - xj.y;
        #if DIM == 3
        diff.z = xi.z - xj.z;
        #endif
        float d = ke * dotv(diff, n_dir);
        Axi.x += d * n_dir.x;
        Axi.y += d * n_dir.y;
        #if DIM == 3
        Axi.z += d * n_dir.z;
        #endif
    }

    Vec bi = load_vec_g(b, i);
    Vec r;
    r.x = bi.x - Axi.x;
    r.y = bi.y - Axi.y;
    #if DIM == 3
    r.z = bi.z - Axi.z;
    #endif
    store_vec_g(r_out, i, r);
}

// ============================================================================
// Kernel: compute_diagonal_dinv  (D_i^{-1} for each node)
// ============================================================================

__kernel void compute_diagonal_dinv(
    __global float *Dinv_out,
    __global const int *neighs,
    __global const float *k_vals,
    __global const float *n_dirs,
    __global const int *nneigh,
    __global const float *mass_dt2,
    int n_nodes
) {
    int i = get_global_id(0);
    if (i >= n_nodes) return;

    float D[DIM * DIM];
    for (int d = 0; d < DIM * DIM; d++) D[d] = 0.0f;

    float mi = mass_dt2[i];
    for (int d = 0; d < DIM; d++) D[d * DIM + d] = mi;

    int nn = nneigh[i];
    for (int e = 0; e < nn; e++) {
        float ke = k_vals[i * NMAX_NEIGH + e];
        int dir_base = (i * NMAX_NEIGH + e) * DIM;
        float n_vec[DIM];
        for (int d = 0; d < DIM; d++) n_vec[d] = n_dirs[dir_base + d];
        for (int d = 0; d < DIM; d++)
            for (int f = 0; f < DIM; f++)
                D[d * DIM + f] += ke * n_vec[d] * n_vec[f];
    }

    float inv[DIM * DIM];
    for (int d = 0; d < DIM; d++)
        for (int f = 0; f < DIM; f++)
            inv[d * DIM + f] = (d == f) ? 1.0f : 0.0f;

    for (int d = 0; d < DIM; d++) {
        float pivot = D[d * DIM + d];
        if (fabs(pivot) < 1e-20f) pivot = 1e-20f;
        float ip = 1.0f / pivot;
        for (int f = 0; f < DIM; f++) { D[d * DIM + f] *= ip; inv[d * DIM + f] *= ip; }
        for (int row = 0; row < DIM; row++) {
            if (row == d) continue;
            float factor = D[row * DIM + d];
            for (int f = 0; f < DIM; f++) {
                D[row * DIM + f] -= factor * D[d * DIM + f];
                inv[row * DIM + f] -= factor * inv[d * DIM + f];
            }
        }
    }

    for (int d = 0; d < DIM * DIM; d++)
        Dinv_out[i * DIM * DIM + d] = inv[d];
}
