#!/usr/bin/env python3
"""RAFF-reactive-3D pair potential — selected cuts through the FFI.

Drives libmolff.so via raff3d_ffi.Raff3d (ctypes). Three figure groups:
  a) single-atom field maps E on xy/yz/xz slices for sp3 and sp2 C
     (probe = point atom r_cov=0.4); port dirs drawn as projections.
  b) pair E(r) curves for aligned sp3-sp3 / sp2-sp2 / sp3-H +
     E vs z-rotation of atom j at fixed r0 (gate angular width).
  c) force cut: quiver of (Fx,Fy) + |F| on the xy slice, sp3 atom vs H probe.

Usage: python3 scripts/plot_raff3d_potential.py
Output: debug/raff3d_potential/*.png  (REVIEW lines printed)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raff3d_ffi import Raff3d, quat_align, quat_mul, quat_axis_angle   # noqa: E402

OUTDIR = os.path.join(REPO, 'debug', 'raff3d_potential')
os.makedirs(OUTDIR, exist_ok=True)

INV_SQRT3 = 1.0 / np.sqrt(3.0)
SP3_PORT0 = np.array([INV_SQRT3] * 3)        # body-frame port 0 of set_sp3
QID = np.array([0.0, 0.0, 0.0, 1.0])         # identity quat (xyzw)

# ---------- Fig 1: single-atom field maps on axis slices ----------
print("Fig1: single-atom field maps (sp3, sp2) ...", flush=True)
ext, ng = 2.4, 241
g = np.linspace(-ext, ext, ng)
SLICES = [('xy', 2), ('yz', 0), ('xz', 1)]   # (label, fixed-axis index)


def field_slice(ff, axis, val, g, probe_rc=0.4):
    """E on the plane {axis: val}: probe-point field_at over a 2D grid."""
    E = np.empty((len(g), len(g)))
    for iy, u in enumerate(g):                        # second slice axis = rows
        for ix, v in enumerate(g):                    # first slice axis = cols
            p = [0.0, 0.0, 0.0]
            axs = [a for a in range(3) if a != axis]
            p[axis], p[axs[0]], p[axs[1]] = val, v, u
            E[iy, ix] = ff.field_at(p[0], p[1], p[2], skip=-1, r_cov=probe_rc)
    return E


def port_proj(dirs, nport, axis):
    """Port dirs projected onto the slice plane (drop `axis` coord)."""
    axs = [a for a in range(3) if a != axis]
    return [(d[axs[0]], d[axs[1]]) for d in dirs[:nport]]


for tname in ('sp3', 'sp2'):
    ff = Raff3d(1, [tname])
    ff.set_state(np.zeros((1, 3)))                    # origin, identity quat
    dirs, nport = ff.get_ports()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (slab, axis) in zip(axes, SLICES):
        E = field_slice(ff, axis, 0.0, g)
        im = ax.imshow(np.clip(E, -2.0, 2.0), extent=[-ext, ext, -ext, ext],
                       origin='lower', cmap='RdBu_r', vmin=-2.0, vmax=2.0)
        ax.contour(g, g, E, levels=[-0.1], colors='k', linewidths=0.6)
        for (u, v) in port_proj(dirs[0], nport[0], axis):
            ax.arrow(0, 0, 0.5 * u, 0.5 * v, head_width=0.08, color='lime',
                     lw=1.2, length_includes_head=True)
        ax.plot(0, 0, 'k+', ms=10, mew=2)
        ax.set_title(f'{slab} slice'); ax.set_aspect('equal')
        ax.set_xlabel(f'{slab[0]} [A]'); ax.set_ylabel(f'{slab[1]} [A]')
    fig.colorbar(im, ax=axes[-1], label='E [eV]')
    fig.suptitle(f'{tname} atom at origin; point probe r_cov=0.4 (green = port projections, contour E=-0.1)')
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f'field_{tname}.png'), dpi=150); plt.close(fig)
    print(f"  wrote field_{tname}.png", flush=True)

# ---------- Fig 2: pair E(r) + E(theta_z) ----------
print("Fig2: pair E(r) curves + angular gate cuts ...", flush=True)
q_sp3_i = quat_align(SP3_PORT0, [1, 0, 0])            # i port0 -> +x (faces j)
q_sp3_j = quat_align(SP3_PORT0, [-1, 0, 0])           # j port0 -> -x (faces i)
q_sp2_j = quat_axis_angle([0, 0, 1], np.pi)           # sp2 j: +x -> -x in-plane
PAIRS = [('sp3-sp3', [0, 0], q_sp3_i, q_sp3_j, 1.54),
         ('sp2-sp2', [1, 1], QID, q_sp2_j, 1.46),
         ('sp3-H',   [0, 3], q_sp3_i, quat_align([1, 0, 0], [-1, 0, 0]), 1.14)]   # H is 1-port (port_local +x) — must face C

rs = np.linspace(0.8, 5.0, 300)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
table = []
for name, types, qi, qj, r0 in PAIRS:
    ff = Raff3d(2, types)
    E = np.empty(len(rs))
    for k, r in enumerate(rs):
        ff.set_state(np.array([[0, 0, 0], [r, 0, 0]]), [qi, qj])
        E[k] = ff.eval()[0]
    imin = E.argmin()
    table.append((name, rs[imin], E[imin], r0))
    ax.plot(rs, E, label=f'{name} (r0={r0})')
    ax.axvline(r0, color='k', ls=':', lw=0.7)
ax.axhline(0, color='k', lw=0.5)
ax.set_title('Aligned pair E(r) — facing ports'); ax.set_xlabel('r [A]')
ax.set_ylabel('E [eV]'); ax.set_ylim(-1.6, 4); ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1]                                   # E vs z-rotation of atom j at r0
ths = np.linspace(0.0, np.pi, 121)
for name, types, qi, qj, r0 in PAIRS[:2]:      # sp3 and sp2 only
    ff = Raff3d(2, types)
    E = np.empty(len(ths))
    for k, th in enumerate(ths):
        qj2 = quat_mul(quat_axis_angle([0, 0, 1], th), qj)
        ff.set_state(np.array([[0, 0, 0], [r0, 0, 0]]), [qi, qj2])
        E[k] = ff.eval()[0]
    ax.plot(np.degrees(ths), E, label=name)
ax.axhline(0, color='k', lw=0.5)
ax.set_title('E vs j rotation about z at r=r0 (gate width)')
ax.set_xlabel('theta [deg]'); ax.set_ylabel('E [eV]'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'pair_Er.png'), dpi=150); plt.close(fig)
print("  wrote pair_Er.png", flush=True)

print('\nE_min table (expect ~-1.0 at r0 = r_cov_i + r_cov_j):')
print(f'  {"pair":8} {"r_min":>7} {"E_min":>9} {"r0":>6}')
for name, rmin, emin, r0 in table:
    print(f'  {name:8} {rmin:7.3f} {emin:9.4f} {r0:6.2f}')

# ---------- Fig 3: force cut — sp3 atom vs H probe on xy slice ----------
print("Fig3: force quiver on xy slice (sp3 + H probe) ...", flush=True)
ff = Raff3d(2, [0, 3])
gf = np.linspace(-2.0, 2.0, 61)
Fm = np.empty((len(gf), len(gf))); Fx = np.empty_like(Fm); Fy = np.empty_like(Fm)
for iy, y in enumerate(gf):
    for ix, x in enumerate(gf):
        ff.set_state(np.array([[0, 0, 0], [x, y, 0.0]]), [q_sp3_i, QID])
        _, f, _, _ = ff.eval()
        Fx[iy, ix], Fy[iy, ix] = f[1, 0], f[1, 1]
        Fm[iy, ix] = np.hypot(f[1, 0], f[1, 1])
fig, ax = plt.subplots(figsize=(6.4, 5.6))
im = ax.imshow(np.log10(Fm + 1e-9), extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
st = 3                                          # quiver decimation (~20x20)
ax.quiver(gf[::st], gf[::st], Fx[::st, ::st], Fy[::st, ::st], color='w', scale=None)
ff.set_state(np.array([[0, 0, 0], [10, 0, 0]]), [q_sp3_i, QID])   # restore aligned quat for port arrows
dirs, nport = ff.get_ports()
for d in dirs[0, :nport[0]]:
    ax.arrow(0, 0, 0.5 * d[0], 0.5 * d[1], head_width=0.08, color='red', lw=1.2, length_includes_head=True)
ax.plot(0, 0, 'r+', ms=10, mew=2)
fig.colorbar(im, ax=ax, label='log10|F| [eV/A]')
ax.set_title('Force on 1-port H probe (H port fixed +x, xy slice); sp3 at origin, port0->+x (red = C port dirs)')
ax.set_xlabel('x [A]'); ax.set_ylabel('y [A]'); ax.set_aspect('equal')
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'force_xy.png'), dpi=150); plt.close(fig)
print("  wrote force_xy.png", flush=True)

print(f"\nDone.\nREVIEW: {OUTDIR}/field_sp3.png\nREVIEW: {OUTDIR}/field_sp2.png\nREVIEW: {OUTDIR}/pair_Er.png\nREVIEW: {OUTDIR}/force_xy.png", flush=True)
