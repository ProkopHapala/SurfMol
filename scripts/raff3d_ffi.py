"""raff3d_ffi.py — ctypes binding to SurfMol libmolff.so reactive-3D FF (raff_reactive).

Contract (ffi_raff_reactive.rs): all numpy float64 C-contiguous.
types (n,) uint8: 0=sp3 (4 ports tetra), 1=sp2 (3 ports planar), 2=sp1 (2 ports
linear), 3=H (1-port atom: single directed port, bonds only along it). quat
(n,4) is (x,y,z,w), renormalized on ingest.
pos/f/tau are (n,3). Energy eval is O(N²) CPU (Grid3 after set_grid) — safe
inside CUDA training loops.

Build: `cargo build --release -p molff` in SurfMol -> libmolff.so.
"""
import os
import ctypes
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# locate libmolff.so: MOLFF_SO env > CARGO_TARGET_DIR/release > repo target/release
_CANDS = [os.environ.get('MOLFF_SO'),
          os.path.join(os.environ.get('CARGO_TARGET_DIR', ''), 'release/libmolff.so') if os.environ.get('CARGO_TARGET_DIR') else None,
          '/home/prokop/.cargo/shared_target/release/libmolff.so',
          os.path.join(REPO, 'target/release/libmolff.so')]
MOLFF_SO = next((c for c in _CANDS if c and os.path.isfile(c)), None)
assert MOLFF_SO, f'libmolff.so not found in {_CANDS} (cargo build --release -p molff in SurfMol)'
L = ctypes.CDLL(MOLFF_SO)

_d = ctypes.POINTER(ctypes.c_double)
_u8 = ctypes.POINTER(ctypes.c_uint8)
_i32 = ctypes.POINTER(ctypes.c_int32)
_p = ctypes.c_void_p
L.raff3d_new.argtypes = [ctypes.c_int32, _u8]; L.raff3d_new.restype = _p
L.raff3d_free.argtypes = [_p]
L.raff3d_set_state.argtypes = [_p, _d, _d]
L.raff3d_get_state.argtypes = [_p, _d, _d]
L.raff3d_eval.argtypes = [_p, _d, _d, _d]; L.raff3d_eval.restype = ctypes.c_double
L.raff3d_step.argtypes = [_p, ctypes.c_double, ctypes.c_double]; L.raff3d_step.restype = ctypes.c_double
L.raff3d_relax.argtypes = [_p, ctypes.c_int32, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, _p, _p]; L.raff3d_relax.restype = ctypes.c_double
L.raff3d_set_traj.argtypes = [_p, ctypes.c_char_p, ctypes.c_int32]; L.raff3d_set_traj.restype = ctypes.c_int32
L.raff3d_clear_traj.argtypes = [_p]
L.raff3d_pairs_within.argtypes = [_p, ctypes.c_double, _i32, _d, ctypes.c_int32]; L.raff3d_pairs_within.restype = ctypes.c_int32
L.raff3d_get_ports.argtypes = [_p, _d]
L.raff3d_nports.argtypes = [_p, _i32]
L.raff3d_set_grid.argtypes = [_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
L.raff3d_clear_grid.argtypes = [_p]
L.raff3d_field_at.argtypes = [_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int32, ctypes.c_double, ctypes.c_double, ctypes.c_double]; L.raff3d_field_at.restype = ctypes.c_double
L.raff3d_emerged_bonds.argtypes = [_p, ctypes.c_double, _i32, ctypes.c_int32]; L.raff3d_emerged_bonds.restype = ctypes.c_int32
L.raff3d_set_param.argtypes = [_p, ctypes.c_int32, ctypes.c_double, ctypes.c_double, ctypes.c_double]
L.raff3d_set_pex_m.argtypes = [_p, ctypes.c_int32]
L.raff3d_set_cij.argtypes = [_p, ctypes.c_int32]
L.raff3d_set_box.argtypes = [_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]


def _ptr(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a, a.ctypes.data_as(_d)


TYPE = {'sp3': 0, 'sp2': 1, 'sp1': 2, 'h': 3}       # type name -> u8 code
# covalent radii per type (mirror of RCOV in ffi_raff_reactive.rs)
RCOV = {0: 0.77, 1: 0.73, 2: 0.66, 3: 0.37}


# ---------- quaternion helpers (xyzw, mirror of raff.rs) ----------

def quat_normalize(q):
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q)

def quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])

def quat_axis_angle(axis, angle):
    """Rotation by `angle` [rad] about unit axis -> (x,y,z,w)."""
    axis = np.asarray(axis, np.float64); axis = axis / np.linalg.norm(axis)
    s = np.sin(0.5 * angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(0.5 * angle)])

def quat_align(a, b):
    """Shortest-arc quaternion rotating unit vec a onto unit vec b."""
    a = np.asarray(a, np.float64); a = a / np.linalg.norm(a)
    b = np.asarray(b, np.float64); b = b / np.linalg.norm(b)
    ax = np.cross(a, b); s = np.linalg.norm(ax); c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0: return np.array([0., 0., 0., 1.])
        # a ≈ -b: 180° about any axis ⊥ a (a 180°-about-a quat leaves a invariant!)
        ax = np.cross(a, [1., 0., 0.] if abs(a[0]) < 0.9 else [0., 1., 0.])
        return np.array([ax[0] / np.linalg.norm(ax), ax[1] / np.linalg.norm(ax), ax[2] / np.linalg.norm(ax), 0.])
    return quat_axis_angle(ax / s, np.arctan2(s, c))

def quat_rotate(q, v):
    qv = np.array([v[0], v[1], v[2], 0.0])
    r = quat_mul(quat_mul(q, qv), np.array([-q[0], -q[1], -q[2], q[3]]))
    return r[:3]


class Raff3d:
    """One reactive-3D system. types (n,) uint8 or list of 'sp3'/'sp2'/'sp1'/'h'.
    pos (n,3) A; quat (n,4) xyzw (default identity)."""

    def __init__(self, n, types):
        t = np.array([TYPE[x] if isinstance(x, str) else int(x) for x in types], np.uint8)
        assert len(t) == n, f'len(types)={len(t)} != n={n}'
        self.n = n
        self.h = L.raff3d_new(n, t.ctypes.data_as(_u8))
        assert self.h, 'raff3d_new returned null'

    def set_state(self, pos, quat=None):
        """pos (n,3); quat (n,4) xyzw or None -> identity. Velocities reset."""
        (pos, pp) = _ptr(np.asarray(pos, np.float64).reshape(self.n, 3))
        if quat is None:
            quat = np.tile([0.0, 0.0, 0.0, 1.0], (self.n, 1))
        (quat, qp) = _ptr(np.asarray(quat, np.float64).reshape(self.n, 4))
        L.raff3d_set_state(self.h, pp, qp)

    def get_state(self):
        pos = np.zeros((self.n, 3)); quat = np.zeros((self.n, 4))
        L.raff3d_get_state(self.h, pos.ctypes.data_as(_d), quat.ctypes.data_as(_d))
        return pos, quat

    def eval(self):
        """-> (E, F (n,3) = -grad E, tau (n,3), eatom (n,))."""
        f = np.zeros((self.n, 3)); t = np.zeros((self.n, 3)); ea = np.zeros(self.n)
        e = L.raff3d_eval(self.h, f.ctypes.data_as(_d), t.ctypes.data_as(_d), ea.ctypes.data_as(_d))
        return e, f, t, ea

    def step(self, dt=0.02, cdamp=0.9):
        """One damped-MD step -> E."""
        return L.raff3d_step(self.h, dt, cdamp)

    def relax(self, fconv=1e-4, tconv=None, dt=0.02, cdamp=0.9, n_max_iter=10000, traj=None, stride=10):
        """Damped-MD relax to max|F| < fconv && max|tau| < tconv (tconv defaults
        to fconv). traj: optional multi-frame XYZ path written every `stride`
        steps. -> (E_final, steps, converged)."""
        if tconv is None:
            tconv = fconv
        steps = ctypes.c_int32(0); conv = ctypes.c_int32(0)
        if traj is not None:
            self.set_traj(traj, stride)
        try:
            e = L.raff3d_relax(self.h, n_max_iter, dt, cdamp, fconv, tconv, ctypes.byref(steps), ctypes.byref(conv))
        finally:
            if traj is not None:
                self.clear_traj()
        return e, steps.value, bool(conv.value)

    def set_traj(self, path, stride=10):
        """Attach multi-frame XYZ writer ({n}\nstep {s} E={e}\n + El x y z per
        atom); a frame every `stride` steps of step()/relax()."""
        r = L.raff3d_set_traj(self.h, os.fspath(path).encode(), int(stride))
        assert r == 0, f'raff3d_set_traj({path!r}) returned {r}'

    def clear_traj(self):
        """Flush + close the trajectory writer (no-op if none attached)."""
        L.raff3d_clear_traj(self.h)

    def pairs_within(self, rc):
        """All pairs (i,j) with |x_j - x_i| < rc -> (pairs (m,2) int32, dists
        (m,) f64). Two-call cap/retry; uses the grid when set, else auto-fit."""
        m = L.raff3d_pairs_within(self.h, rc, None, None, 0)
        pairs = np.zeros((m, 2), np.int32); dists = np.zeros(m, np.float64)
        if m == 0:
            return pairs, dists
        k = L.raff3d_pairs_within(self.h, rc, pairs.ctypes.data_as(_i32), dists.ctypes.data_as(_d), m)
        assert k == m, f'pairs_within count changed between calls: {m} -> {k}'
        return pairs, dists

    def get_ports(self):
        """-> (dirs (n,4,3) world port dirs (unused slots = 0), nport (n,) i32)."""
        dirs = np.zeros((self.n, 4, 3)); np_ = np.zeros(self.n, np.int32)
        L.raff3d_get_ports(self.h, dirs.ctypes.data_as(_d))
        L.raff3d_nports(self.h, np_.ctypes.data_as(_i32))
        return dirs, np_

    def field_at(self, x, y, z, skip=-1, r_cov=0.4, a=1.0, b=0.9):
        """Point-probe energy at (x,y,z): E a portless atom (r_cov,a,b) would
        feel. skip = atom index excluded (-1 none)."""
        return L.raff3d_field_at(self.h, x, y, z, skip, r_cov, a, b)

    def emerged_bonds(self, e_thr=-0.3, cap=4096):
        """Atom pairs with pair energy < e_thr -> (K,2) index pairs."""
        pairs = np.zeros((cap, 2), np.int32)
        k = L.raff3d_emerged_bonds(self.h, e_thr, pairs.ctypes.data_as(_i32), cap)
        assert k <= cap, f'{k} emerged bonds > cap {cap} — raise cap'
        return pairs[:k]

    def set_grid(self, dx, origin, nxyz):
        """Uniform neighbor grid: cell dx [A], origin (ox,oy,oz), dims (nx,ny,nz)."""
        L.raff3d_set_grid(self.h, dx, float(origin[0]), float(origin[1]), float(origin[2]), int(nxyz[0]), int(nxyz[1]), int(nxyz[2]))

    def clear_grid(self):
        L.raff3d_clear_grid(self.h)

    def set_box(self, bmin, bmax, k=50.0):
        """Harmonic AABB confinement F=k*(limit-x) outside box. k<=0 or
        bmin==bmax disables."""
        L.raff3d_set_box(self.h, float(bmin[0]), float(bmin[1]), float(bmin[2]), float(bmax[0]), float(bmax[1]), float(bmax[2]), float(k))

    def set_param(self, i, r_cov, a=1.0, b=0.9):
        """Override atom i params. NOTE: drops the grid — re-set if needed."""
        L.raff3d_set_param(self.h, i, r_cov, a, b)

    def set_pex_m(self, m):
        """pex squaring count (support r0 + 2^m/(b_i+b_j)). Drops the grid."""
        L.raff3d_set_pex_m(self.h, int(m))

    def set_cij(self, flag):
        """Port-pair gate h_i·h_j factor: 1 on (default), 0 off (ablation)."""
        L.raff3d_set_cij(self.h, int(flag))

    def __del__(self):
        if getattr(self, 'h', None):
            L.raff3d_free(self.h)
            self.h = None


if __name__ == '__main__':
    # sanity: methane-like assembly — C sp3 at origin + 4 H in-cone (tetrahedral
    # sites + jitter); random directions can land out-of-cone and are repelled
    # by design (see test_methane_random_start --ignored diagnostic).
    rng = np.random.default_rng(0xB5297A4D)
    s = 1.0 / np.sqrt(3.0)
    d = np.array([[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]])
    d = d * 2.0 + rng.uniform(-0.15, 0.15, size=(4, 3))
    ff = Raff3d(5, [0, 3, 3, 3, 3])
    # 1-port H must arrive port-first: quat rotates port_local +x onto -d.
    qs = np.vstack([[0.0, 0.0, 0.0, 1.0]] + [quat_align([1, 0, 0], -di) for di in d])
    ff.set_state(np.vstack([np.zeros(3), d]), qs)
    e0, _, _, _ = ff.eval()
    dbg = os.path.join(REPO, 'debug', 'raff3d_ffi_test')
    os.makedirs(dbg, exist_ok=True)
    tpath = os.path.join(dbg, 'traj.xyz')
    e, ns, conv = ff.relax(n_max_iter=5000, traj=tpath, stride=10)
    pos, _ = ff.get_state()
    bonds = ff.emerged_bonds(-0.3)
    print(f'methane-like: E {e0:.3f} -> {e:.3f} after {ns} steps (conv={conv}); emerged bonds: {bonds.tolist()}')
    for i, j in bonds:
        r = np.linalg.norm(pos[j] - pos[i])
        print(f'  bond {i}-{j}: r={r:.4f} (expect ~1.14)')
        assert abs(r - 1.14) < 0.05, f'bond {i}-{j} length {r}'
        assert i == 0 or j == 0, f'bond {i}-{j} is not C-H'
    assert len(bonds) == 4, f'expected 4 C-H bonds, got {bonds.tolist()}'
    print('PASS emerged_bonds')
    # (a) traj file: exists, 5 atoms per frame, >= 2 frames
    assert os.path.isfile(tpath), f'traj file missing: {tpath}'
    lines = open(tpath).read().splitlines()
    (k, nfr) = (0, 0)
    while k < len(lines):
        nat = int(lines[k]); assert nat == 5, f'traj frame at line {k}: natoms={nat} != 5'
        assert lines[k + 1].startswith('step '), f'traj frame at line {k}: bad comment {lines[k + 1]!r}'
        nfr += 1; k += nat + 2
    assert nfr >= 2, f'traj {tpath} has only {nfr} frames (< 2)'
    print(f'PASS traj: {nfr} frames x 5 atoms -> {tpath}')
    # (b) pairs_within: parity vs numpy brute force. NOTE H-H = r_CH*sqrt(8/3)
    # ~1.89 < 2.0, so rc=2.0 sees all 10 pairs; rc=1.5 isolates the 4 C-H.
    (bi, bj) = np.triu_indices(5, 1)
    rdm = np.linalg.norm(pos[bi] - pos[bj], axis=1)
    pairs, dists = ff.pairs_within(2.0)
    brute = sorted(zip(bi[rdm < 2.0].tolist(), bj[rdm < 2.0].tolist()))
    got = sorted(map(tuple, pairs.tolist()))
    assert got == brute, f'pairs_within(2.0) {got} != brute-force {brute}'
    print(f'PASS pairs_within(2.0): {len(pairs)} pairs == brute force (4 C-H + 6 H-H)')
    pairs, dists = ff.pairs_within(1.5)
    assert sorted(map(tuple, pairs.tolist())) == sorted(zip(bi[rdm < 1.5].tolist(), bj[rdm < 1.5].tolist()))
    assert len(pairs) == 4 and all(0 in p for p in pairs), f'expected 4 C-H pairs, got {pairs.tolist()}'
    assert np.allclose(dists, 1.156, atol=0.05), f'C-H dists {dists.tolist()} not ~1.156'
    print(f'PASS pairs_within(1.5): {pairs.tolist()} dists {np.round(dists, 4).tolist()} (4 C-H ~1.156)')
    print('raff3d_ffi: OK')
