// ============================================================================
// multigrid.cl
// GPU kernels for multigrid restriction (R = P^T) and prolongation (P).
//
// Copied verbatim 2026-08-29 from NumericalMathPlayground/topics/LinarElasticity/kernels_multigrid.cl
// for later adaptation to SurfMol buffer conventions. See notes/designs/2026-08-29_modal_relaxation_design_spec.md.
// Original: truss-elasticity multigrid (axial-spring stiffness A = M/dt² + K).
//
// Compile-time defines:
//   WG_SIZE, CLUSTER_SIZE, DIM, L_COARSE
// ============================================================================

#ifndef DIM
#define DIM 3
#endif

#if DIM == 2
typedef float2 Vec;
#else
typedef float3 Vec;
#endif

// ---- Float atomic add (OpenCL 1.2 doesn't have atomic_add for float) ----
inline void atomic_add_f(__global volatile float *ptr, float value) {
    union {
        unsigned int u;
        float f;
    } next, expected, current;
    current.f = *ptr;
    do {
        expected.f = current.f;
        next.f = expected.f + value;
        current.u = atomic_cmpxchg((__global volatile unsigned int *)ptr,
                                   expected.u, next.u);
    } while (current.u != expected.u);
}

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

// ============================================================================
// Kernel: prolongate  (x += P * e_c)
// One thread per local node in each patch.
// ============================================================================

__kernel void prolongate(
    __global float *x,
    __global const float *e_c,
    __global const float *P_global,
    __global const int *patch_node_ids,
    int n_patches
) {
    int patch_id = get_group_id(0);
    int a = get_local_id(0);
    if (patch_id >= n_patches) return;
    if (a >= CLUSTER_SIZE) return;

    int node = patch_node_ids[patch_id * CLUSTER_SIZE + a];

    // Load e_c into local memory (all threads must participate)
    __local float e_c_loc[L_COARSE];
    if (a < L_COARSE) e_c_loc[a] = e_c[a];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (node < 0) return;

    // Accumulate: x[node] += sum_j P[p, j, a, :] * e_c[j]
    float sum_x = 0.0f, sum_y = 0.0f;
    #if DIM == 3
    float sum_z = 0.0f;
    #endif

    int P_base = patch_id * L_COARSE * CLUSTER_SIZE * DIM + a * DIM;
    for (int j = 0; j < L_COARSE; j++) {
        float ec = e_c_loc[j];
        int P_off = P_base + j * CLUSTER_SIZE * DIM;
        sum_x += P_global[P_off + 0] * ec;
        sum_y += P_global[P_off + 1] * ec;
        #if DIM == 3
        sum_z += P_global[P_off + 2] * ec;
        #endif
    }

    Vec xi = load_vec_g(x, node);
    xi.x += sum_x;
    xi.y += sum_y;
    #if DIM == 3
    xi.z += sum_z;
    #endif
    store_vec_g(x, node, xi);
}

// ============================================================================
// Kernel: restrict_residual_tree  (r_c = P^T * r, with tree reduction)
// One workgroup per patch, tree-reduce across CLUSTER_SIZE threads.
// ============================================================================

__kernel void restrict_residual_tree(
    __global float *r_c,
    __global const float *r,
    __global const float *P_global,
    __global const int *patch_node_ids,
    int n_patches
) {
    int patch_id = get_group_id(0);
    int a = get_local_id(0);
    if (patch_id >= n_patches) return;

    float my_sum[L_COARSE];
    for (int j = 0; j < L_COARSE; j++) my_sum[j] = 0.0f;

    bool active = (a < CLUSTER_SIZE);
    int node = -1;
    if (active) node = patch_node_ids[patch_id * CLUSTER_SIZE + a];

    if (node >= 0) {
        Vec ri = load_vec_g(r, node);
        int P_base = patch_id * L_COARSE * CLUSTER_SIZE * DIM + a * DIM;
        for (int j = 0; j < L_COARSE; j++) {
            int P_off = P_base + j * CLUSTER_SIZE * DIM;
            my_sum[j] += P_global[P_off + 0] * ri.x + P_global[P_off + 1] * ri.y;
            #if DIM == 3
            my_sum[j] += P_global[P_off + 2] * ri.z;
            #endif
        }
    }

    __local float reduce_buf[CLUSTER_SIZE * L_COARSE];
    for (int j = 0; j < L_COARSE; j++)
        reduce_buf[a * L_COARSE + j] = my_sum[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = CLUSTER_SIZE / 2; stride > 0; stride >>= 1) {
        if (a < stride) {
            for (int j = 0; j < L_COARSE; j++)
                reduce_buf[a * L_COARSE + j] += reduce_buf[(a + stride) * L_COARSE + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (a == 0) {
        for (int j = 0; j < L_COARSE; j++)
            atomic_add_f(&r_c[j], reduce_buf[j]);
    }
}

// ============================================================================
// Kernel: zero_buffer
// ============================================================================

__kernel void zero_buffer(__global float *buf, int n) {
    int i = get_global_id(0);
    if (i < n) buf[i] = 0.0f;
}

// ============================================================================
// Kernel: coarse_solve_cholesky
// Solve A_c * e_c = r_c using precomputed Cholesky L (lower triangular).
// One workgroup, L_COARSE threads (or fewer).
// ============================================================================

__kernel void coarse_solve_cholesky(
    __global float *e_c,
    __global const float *r_c,
    __global const float *L_c,
    int l_coarse
) {
    int lid = get_local_id(0);

    __local float y_loc[64];
    __local float e_loc[64];

    // Forward solve: L * y = r
    for (int i = 0; i < l_coarse; i++) {
        if (lid == i) {
            float sum = r_c[i];
            for (int k = 0; k < i; k++)
                sum -= L_c[i * l_coarse + k] * y_loc[k];
            y_loc[i] = sum / L_c[i * l_coarse + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Backward solve: L^T * e = y
    for (int i = l_coarse - 1; i >= 0; i--) {
        if (lid == i) {
            float sum = y_loc[i];
            for (int k = i + 1; k < l_coarse; k++)
                sum -= L_c[k * l_coarse + i] * e_loc[k];
            e_loc[i] = sum / L_c[i * l_coarse + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid < l_coarse) e_c[lid] = e_loc[lid];
}
