#!/usr/bin/env python3
"""RARFF-2D mutual-gate verification + angular width w sweep.

Issue 2 (mutual gate): E_ij = a(e^2 - 2 e g_i g_j) — attraction requires BOTH
atoms port-aligned. Here atom i is FIXED at origin, phi_i=0 (ports at 0,120,240
deg), atom j is a probe — no ring entities (rings would obscure the pair term).

Issue 1 (w tuning): gate g = 1/(1+(sin(n*dphi/2)/w)^2); w controls port width.

Writes debug/rarff2d_mutual/{mutual_gate.png,w_sweep.png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.normpath(os.path.join(HERE, '..', 'debug', 'rarff2d_mutual'))
os.makedirs(OUTDIR, exist_ok=True)

R0, A, B, N = 1.3, 1.0, 1.8, 3

def pex(x, n=32.0):
    return (1.0 + x / n) ** n

def morse(r, a=A, b=B, r0=R0):
    e = pex(-b * (r - r0))
    return a * e * e, -2.0 * a * e            # rep, att (ungated)

def gate(dphi, w=0.3, n=N):
    return 1.0 / (1.0 + (np.sin(0.5 * n * dphi) / w) ** 2)

def pair_E_xy(X, Y, phi_i, phi_j_fn, w=0.3):
    """E(x_j,y_j): i at origin w/ fixed phi_i; j oriented by phi_j_fn(th_ij)."""
    R = np.sqrt(X * X + Y * Y)
    th = np.arctan2(Y, X)                      # i->j direction (source i port test)
    erep, eatt = morse(R)
    gi = gate(th - phi_i, w=w)                 # source i sees probe at angle th
    gj = gate(th + np.pi - phi_j_fn(th), w=w)  # probe j sees source at th+pi
    return erep + gi * gj * eatt

# ---------- Fig 1: mutual gate verification ----------
print("Fig1: mutual gate verification ...", flush=True)
ext = 3.0
xx = np.linspace(-ext, ext, 400)
X, Yg = np.meshgrid(xx, xx)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# A: cooperative probe (phi_j always faces i) -> pure source-port basins
E = pair_E_xy(X, Yg, 0.0, lambda th: th + np.pi)
im = axes[0, 0].imshow(np.clip(E, -1.6, 1.6), extent=[-ext, ext] * 2, origin='lower', cmap='RdBu_r')
axes[0, 0].set_title('A) j cooperative (faces i): i-port basins\n(probe gate = 1)')
fig.colorbar(im, ax=axes[0, 0], label='E [eV]')

# B: j orientation FIXED phi_j=pi (faces -x only) -> attraction survives only where BOTH aligned
E = pair_E_xy(X, Yg, 0.0, lambda th: np.pi + 0.0 * th)
im = axes[0, 1].imshow(np.clip(E, -1.6, 1.6), extent=[-ext, ext] * 2, origin='lower', cmap='RdBu_r')
axes[0, 1].set_title('B) j fixed phi=pi (faces -x): attraction only\nwhere j ALSO points at i (right side, port at 0deg)')
fig.colorbar(im, ax=axes[0, 1], label='E [eV]')

# C: (dphi_i, dphi_j) map at r=r0, th=0 — product structure
di = np.linspace(-np.pi / 3, np.pi / 3, 240)
DI, DJ = np.meshgrid(di, di)
erep, eatt = morse(R0)
E = erep + gate(DI) * gate(DJ) * eatt
im = axes[0, 2].imshow(E, extent=np.degrees([di[0], di[-1], di[0], di[-1]]), origin='lower', cmap='RdBu_r', vmin=-1.1, vmax=1.1)
axes[0, 2].set_xlabel('dphi_i [deg]'); axes[0, 2].set_ylabel('dphi_j [deg]')
axes[0, 2].set_title('C) E(dphi_i, dphi_j) at r=r0: product gate\nwell only when BOTH aligned')
fig.colorbar(im, ax=axes[0, 2], label='E [eV]')

# D: E(phi_j) at th=0, r=r0, i aligned — one well at dphi_j=0 (j faces i)
ph = np.linspace(-np.pi, np.pi, 400)
ax = axes[1, 0]
for w in [0.3, 0.6]:
    e = erep + gate(0 * ph, w=w) * gate(ph, w=w) * eatt
    ax.plot(np.degrees(ph), e, label=f'w={w}')
ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('dphi_j [deg]'); ax.set_ylabel('E [eV]')
ax.set_title('D) rotate j at r=r0 on i port:\nattraction only near alignment')
ax.legend(); ax.grid(alpha=0.3)

# E: radial cuts at th=0 — aligned vs j rotated 30/60 deg
r = np.linspace(0.7, 2.6, 300)
ax = axes[1, 1]
erep_r, eatt_r = morse(r)
ax.plot(r, erep_r + eatt_r, 'k--', lw=2, label='both aligned (Y=1)')
for deg, c in [(20, 'C1'), (30, 'C2'), (45, 'C3'), (60, 'C4')]:
    gj = gate(np.radians(deg))
    ax.plot(r, erep_r + gj * eatt_r, color=c, label=f'j off by {deg}deg (g_j={gj:.2f})')
ax.set_xlabel('r [A]'); ax.set_ylabel('E [eV]'); ax.set_ylim(-1.4, 2.0)
ax.set_title('E) E(r) at i port, i aligned:\nj misorientation weakens well only')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# F: force direction sanity — torque on j vs dphi_j (sign = rotates toward alignment)
ax = axes[1, 2]
w = 0.3
dg = gate(ph, w=w)
ddg = np.gradient(dg, ph)                    # numeric for readability
tau = -2.0 * 1.0 * 1.0 * ddg                 # ~ -dE/dphi (at r0, e≈1, g_i=1)
ax.plot(np.degrees(ph), tau)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('dphi_j [deg]'); ax.set_ylabel('tau_j ~ -dE/dphi')
ax.set_title('F) torque on j: pushes back toward 0\n(stable at alignment, anti-stable at anti-node)')
ax.grid(alpha=0.3)

for axx in axes[0, :2]:
    axx.plot(0, 0, 'w+', ms=14, mew=2)
    for k in range(3):
        t = k * 2 * np.pi / 3
        axx.plot(R0 * np.cos(t), R0 * np.sin(t), 'g^', ms=8)
fig.suptitle('Mutual gate: E_ij = a(e² - 2 e g_i g_j) — bond needs BOTH atoms aligned. Atom i (white +) fixed, ports = green triangles')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'mutual_gate.png'), dpi=150)
print("  wrote mutual_gate.png", flush=True)

# ---------- Fig 2: w sweep ----------
print("Fig2: w sweep ...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ws = [0.3, 0.45, 0.6, 0.8, 1.0]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ws)))

ax = axes[0]                                  # gate shapes
for w, c in zip(ws, colors):
    ax.plot(np.degrees(ph), gate(ph, w=w), color=c, label=f'w={w}')
ax.axvline(60, color='k', ls=':', lw=0.5); ax.axvline(-60, color='k', ls=':', lw=0.5)
ax.set_xlabel('dphi [deg]'); ax.set_ylabel('g'); ax.set_ylim(-0.02, 1.05)
ax.set_title('gate g(dphi), n=3'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]                                  # E(dphi_j) at r0 — capture basin
for w, c in zip(ws, colors):
    e = erep + gate(np.zeros_like(ph), w=w) * gate(ph, w=w) * eatt
    ax.plot(np.degrees(ph), e, color=c, label=f'w={w}')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('dphi_j [deg]'); ax.set_ylabel('E [eV]'); ax.set_ylim(-1.2, 1.2); ax.set_xlim(-70, 70)
ax.set_title('pair E(dphi_j) at r=r0 (i aligned)\ncapture basin = E<0 range')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]                                  # half-width where E(r0)<0 vs w
halfw = []
for w in ws:
    e = erep + gate(np.zeros_like(ph), w=w) * gate(ph, w=w) * eatt
    m = e < 0
    idx = np.where(m & (np.abs(ph) < np.pi / 3))[0]
    halfw.append(np.degrees(np.abs(ph[idx]).max()) if len(idx) else 0.0)
ax.plot(ws, halfw, 'o-')
ax.set_xlabel('w'); ax.set_ylabel('capture half-width [deg]')
ax.set_title('attraction half-width vs w\n(= how forgiving the port is)')
ax.grid(alpha=0.3)

fig.suptitle('Angular width sweep — w is the ONLY handle on port width; repulsion unaffected (ungated)')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'w_sweep.png'), dpi=150)
print("  wrote w_sweep.png", flush=True)
print(f"\nREVIEW: {OUTDIR}/mutual_gate.png\nREVIEW: {OUTDIR}/w_sweep.png", flush=True)
