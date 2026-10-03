#!/usr/bin/env python3
"""RARFF-2D angular-gated Morse pair potential — prototype plot.

Rod-free variant: atom i sits at origin with orientation phi_i=0; probe atom j
scans (x,y). Pair energy = ungated repulsive Morse wall + attractive Morse well
gated by an n-fold angular function of (theta_ij - phi_i):

    e    = exp(b*(r - r0))
    E    = a*e^2  +  g(dphi) * (-2*a*e)

Three gate shapes g(dphi) compared:
  V1 'literal'  : 1/(1 + (cos(n*dphi)/w)^2)      — PEAKS BETWEEN ports (probably wrong)
  V2 'lorentz'  : 1/(1 + ((1-cos(n*dphi))/w)^2)  — peaks AT ports, half-width ~w
  V3 'power4'   : ((1+cos(n*dphi))/2)^4          — RARFF-like (cos)^4, no width param

Usage: python3 scripts/plot_rarff2d_potential.py
Output: debug/rarff2d/*.png   (REVIEW: lines printed)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, 'debug', 'rarff2d')
os.makedirs(OUTDIR, exist_ok=True)

# Morse params — per-atom covalent radii; pair r0 = r_i + r_j.
# Aromatic carbon: r_i = 0.65 A  ->  C-C_ar bond = 1.3 A.
# Pair Morse decay b = b_i + b_j = 0.9 + 0.9 = 1.8 (user: typical e=exp(-b*r), b~1.7-1.8)
RI, RJ = 0.65, 0.65
R0 = RI + RJ                  # 1.3 A  = aromatic C-C
A, B = 1.0, 1.8               # pair aMorse [eV], pair bMorse [1/A]
N = 3                         # n-fold (3 = sp2-like trigonal)

def pex(x, n=32.0):                          # polynomial exp: (1+x/n)^n -> e^x (squaring)
    s = 1.0 + x / n
    return np.where(s > 0.0, s ** n, 0.0)

def morse_terms(r, a=A, b=B, r0=R0, poly=False):
    e = pex(-b * (r - r0)) if poly else np.exp(-b * (r - r0))
    return a * e * e, -2.0 * a * e        # repulsive E_rep, attractive E_att (<0)

def gate(phi, n=N, w=0.3, mode='lorentz'):
    c = np.cos(n * phi)
    if mode == 'literal':
        return 1.0 / (1.0 + (c / w) ** 2)
    if mode == 'lorentz':
        return 1.0 / (1.0 + ((1.0 - c) / w) ** 2)
    if mode == 'power4':
        return ((1.0 + c) * 0.5) ** 4
    if mode == 'sine':                                    # quadratic Lorentzian, peaks AT ports
        return 1.0 / (1.0 + (np.sin(0.5 * n * phi) / w) ** 2)
    if mode == 'cos15':                                   # user's cos(1.5f) — peaks at anti-nodes
        return 1.0 / (1.0 + (np.cos(0.5 * n * phi) / w) ** 2)
    raise ValueError(f"unknown gate mode {mode}")

def pair_E(r, dphi, w=0.3, mode='lorentz', n=N):
    erep, eatt = morse_terms(r)
    return erep + gate(dphi, n=n, w=w, mode=mode) * eatt

# ---------- Fig 1: gate shapes ----------
print("Fig1: gate shapes g(dphi) ...", flush=True)
phi = np.linspace(0, 2 * np.pi, 721)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, w in zip(axes, [0.15, 0.3, 0.6]):
    ax.plot(np.degrees(phi), gate(phi, w=w, mode='literal'), label='V1 literal 1/(1+(cos3f/w)^2)')
    ax.plot(np.degrees(phi), gate(phi, w=w, mode='lorentz'), label='V2 lorentz 1/(1+((1-cos3f)/w)^2)')
    ax.plot(np.degrees(phi), gate(phi, w=w, mode='power4'), label='V3 ((1+cos3f)/2)^4')
    ax.plot(np.degrees(phi), gate(phi, w=w, mode='sine'), label='V4 sine 1/(1+(sin1.5f/w)^2)', lw=2)
    ax.plot(np.degrees(phi), gate(phi, w=w, mode='cos15'), label='V5 cos(1.5f) — anti-node!', ls='--')
    for k in range(3):
        ax.axvline(k * 120.0, color='k', ls=':', lw=0.7)
        ax.axvline(k * 120.0 + 60.0, color='r', ls=':', lw=0.7)
    ax.set_title(f'w = {w}'); ax.set_xlabel('dphi [deg]'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
axes[0].set_ylabel('g(dphi)')
fig.suptitle('Angular gate variants (n=3; black dotted = port dir, red dotted = anti-node)')
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'gate_shapes.png'), dpi=150); plt.close(fig)
print("  wrote gate_shapes.png", flush=True)

# ---------- Fig 2: E(x,y) imshow ----------
print("Fig2: E(x,y) maps ...", flush=True)
ext = 3.4
gx = np.linspace(-ext, ext, 400)
X, Y = np.meshgrid(gx, gx)
R = np.hypot(X, Y) + 1e-9
TH = np.arctan2(Y, X)
panels = [('V1 literal', 'literal', 0.3), ('V2 lorentz w=0.15', 'lorentz', 0.15),
          ('V2 lorentz w=0.3', 'lorentz', 0.3), ('V2 lorentz w=0.6', 'lorentz', 0.6),
          ('V4 sine w=0.3', 'sine', 0.3), ('V4 sine w=0.5', 'sine', 0.5)]
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
for ax, (title, mode, w) in zip(axes, panels):
    E = pair_E(R, TH, w=w, mode=mode)
    im = ax.imshow(np.clip(E, -1.6, 3.0), extent=[-ext, ext, -ext, ext], origin='lower',
                   cmap='RdBu_r', vmin=-1.6, vmax=3.0)
    ax.contour(X, Y, E, levels=[-0.5, -0.2, 0.0], colors='k', linewidths=0.4)
    for k in range(N):
        d = k * 2 * np.pi / N
        ax.arrow(0, 0, 0.85 * ext * np.cos(d), 0.85 * ext * np.sin(d), head_width=0.12,
                 color='lime', lw=1.2, length_includes_head=True)
    ax.plot(0, 0, 'k+', ms=10, mew=2)
    ax.set_title(title, fontsize=9); ax.set_xlabel('x [A]'); ax.set_aspect('equal')
axes[0].set_ylabel('y [A]')
fig.colorbar(im, ax=axes[-1], label='E [eV]')
fig.suptitle('Pair E(x,y): atom i at origin, orientation 0 (green = port dirs); probe j at (x,y)')
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'pair_E_xy.png'), dpi=150); plt.close(fig)
print("  wrote pair_E_xy.png", flush=True)

# ---------- Fig 3: radial cuts — pure Morse wall comparison + gated ----------
print("Fig3: radial E(r) cuts ...", flush=True)
r = np.linspace(0.5, 3.5, 500)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]                                    # LEFT: pure Morse E = a(e^2 - 2e), different b + poly-exp
for b in [0.7, 1.4, 1.8]:
    erep, eatt = morse_terms(r, b=b)
    ax.plot(r, erep + eatt, label=f'pure Morse b={b}')
    ax.plot(r, erep, ls=':', lw=0.8, color=ax.lines[-1].get_color())
for n in [8, 32]:
    ep = pex(-1.8 * (r - R0), n)
    ax.plot(r, ep * ep - 2.0 * ep, '--', label=f'poly-exp n={n} b=1.8')
ax.axvline(R0, color='k', ls=':', lw=0.7); ax.axhline(0, color='k', lw=0.5)
ax.set_title('Pure Morse (dotted = rep wall a*e^2) — steeper b = harder wall')
ax.set_xlabel('r [A]'); ax.set_ylabel('E [eV]'); ax.set_ylim(-1.6, 6); ax.set_xlim(0.5, 3.5)
ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax = axes[1]                                    # RIGHT: gated, b=1.8 — port cut == pure Morse
w = 0.3
erep, eatt = morse_terms(r)
ax.plot(r, erep + eatt, 'k--', lw=2, label='pure Morse b=1.8 (= gated along port)')
for deg, c in [(0, 'g'), (15, 'c'), (30, 'b'), (45, 'm'), (60, 'r')]:
    d = np.radians(deg)
    ax.plot(r, erep + gate(d, w=w) * eatt, color=c, label=f'dphi={deg}deg (Y={gate(d,w=w):.3f})')
ax.axvline(R0, color='k', ls=':', lw=0.7); ax.axhline(0, color='k', lw=0.5)
ax.set_title(f'Gated: E = a(e^2 - 2e*Y), lorentz w={w}, b=1.8')
ax.set_xlabel('r [A]'); ax.set_ylim(-1.6, 6); ax.set_xlim(0.5, 3.5)
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle(f'Radial cuts (a=1 eV, r0=r_i+r_j={R0} A)')
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'radial_cuts.png'), dpi=150); plt.close(fig)
print("  wrote radial_cuts.png", flush=True)

# ---------- Fig 4: bonded pair i-j; where can a third atom attach? ----------
print("Fig4: field of bonded pair i-j seen by bare probe k ...", flush=True)
# i at (0,0) phi_i=0 (ports at 0/120/240); j at (r0,0) phi_j=pi (ports at 180/300/60 -> faces i)
# probe k has NO orientation (bare point): each pair attraction gated only by the source atom's ports.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, w in zip(axes, [0.15, 0.3, 0.6]):
    E = pair_E(R, TH, w=w, mode='lorentz')                       # k vs i: rep + g_i att
    dx, dy = X - R0, Y
    r_jk = np.hypot(dx, dy) + 1e-9
    th_jk = np.arctan2(dy, dx) - np.pi                           # direction j->k rel to j's frame (phi_j=pi)
    erep_jk, eatt_jk = morse_terms(r_jk)
    E += erep_jk + gate(th_jk, w=w, mode='lorentz') * eatt_jk    # k vs j: rep + g_j att
    im = ax.imshow(np.clip(E, -2.2, 3.0), extent=[-ext, ext, -ext, ext], origin='lower',
                   cmap='RdBu_r', vmin=-2.2, vmax=3.0)
    ax.contour(X, Y, E, levels=[-1.0, -0.5, 0.0], colors='k', linewidths=0.4)
    for k in range(N):
        for (ox, ph) in [(0.0, 0.0), (R0, np.pi)]:
            d = ph + k * 2 * np.pi / N
            ax.arrow(ox, 0, 0.5 * np.cos(d), 0.5 * np.sin(d), head_width=0.1, color='lime', lw=1.0, length_includes_head=True)
    ax.plot([0, R0], [0, 0], 'k+', ms=10, mew=2)
    ax.set_title(f'V2 w={w}', fontsize=9); ax.set_aspect('equal'); ax.set_xlabel('x [A]')
axes[0].set_ylabel('y [A]')
fig.suptitle('Bonded pair i(0,phi=0)-j(r0,0,phi=pi); probe k scans. Minima at open port dirs = attachment sites')
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'pair2_E_xy.png'), dpi=150); plt.close(fig)
print("  wrote pair2_E_xy.png", flush=True)

# ---------- Fig 5: ring potential — smootherstep-derivative bump x n-fold sites ----------
# B(rho) = 16 t^2 (1-t)^2, t=(rho+h)/2h, support |rho|<=h  (quintic smootherstep deriv, C^2 bump)
# E_ring = -a * B(|x-pc|-R) * (1+cos(n*(phi-phi_c)))/2   -> n wells on circle of radius R
print("Fig5: ring potential (smootherstep bump x n-fold sites) ...", flush=True)

def bump(rho, h=0.35):                               # (1 - rho^2/h^2)^4 — squaring only, C^3
    q = 1.0 - (rho / h) ** 2
    return np.where(q > 0.0, q ** 4, 0.0)

def bump_smootherstep(rho, h=0.35):                  # old: 16 t^2 (1-t)^2, C^2, for comparison
    t = rho / (2.0 * h) + 0.5
    b = 16.0 * t * t * (1.0 - t) ** 2
    return np.where((t > 0.0) & (t < 1.0), b, 0.0)

def ring_E(X, Y, n, Rr, a=1.0, h=0.5, phi_c=0.0, a_core=1.5, r0_core=None, b_core=2.5):
    r = np.hypot(X, Y) + 1e-9
    ph = np.arctan2(Y, X) - phi_c
    if r0_core is None:
        r0_core = Rr - 0.6                      # core wall ends just inside the annulus
    e_core = a_core * pex(-b_core * (r - r0_core)) ** 2   # poly-Morse repulsive wall in the hole
    return -a * bump(r - Rr, h) * 0.5 * (1.0 + np.cos(n * ph)) + e_core

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
ax = axes[0]                                     # bump profiles vs gaussian
rr = np.linspace(-1.0, 1.0, 400)
ax.plot(rr, bump(rr, h=0.5), 'r-', lw=2, label='(1-rho^2/h^2)^4  [used]')
ax.plot(rr, bump_smootherstep(rr, h=0.5), 'b-', label='16t^2(1-t)^2 smootherstep')
ax.plot(rr, np.exp(-0.5 * (rr / 0.25) ** 2), 'k--', label='gaussian s=0.25')
ax.set_title('radial bump (h=0.5 A)'); ax.set_xlabel('rho = r - R_ring [A]')
ax.grid(alpha=0.3); ax.legend(fontsize=8)
for ax, (n, Rr, name) in zip(axes[1:], [(6, 1.3, 'hexagon'), (5, 1.3 / (2 * np.sin(np.pi / 5)), 'pentagon')]):
    E = ring_E(X, Y, n, Rr)
    im = ax.imshow(np.clip(E, -1.05, 1.5), extent=[-ext, ext, -ext, ext], origin='lower', cmap='RdBu_r', vmin=-1.05, vmax=1.5)
    ax.contour(X, Y, E, levels=[-0.5, -0.2, 0.0], colors='k', linewidths=0.4)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(Rr * np.cos(th), Rr * np.sin(th), 'g--', lw=0.8)
    ax.plot(0, 0, 'k+', ms=10, mew=2)
    ax.set_title(f'{name}: n={n}, R={Rr:.3f} A, h=0.5, core'); ax.set_xlabel('x [A]'); ax.set_aspect('equal')
axes[1].set_ylabel('y [A]')
fig.colorbar(im, ax=axes[-1], label='E [eV]')
fig.suptitle('Ring: E = -a*B(|x|-R)*(1+cos n phi)/2 + a_core*pexp(-b_core*(r-r0c))^2  (sites + core wall)')
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'ring_E_xy.png'), dpi=150); plt.close(fig)
print("  wrote ring_E_xy.png", flush=True)

# ---------- Fig 6: polynomial approximations of exp and gaussian (1D inventory) ----------
# pex(x,n) = (1+x/n)^n -> e^x  (n squaring-friendly powers of two)
# bump_m(u) = (1-u^2)^m on |u|<1 ; curvature-matched to gaussian sigma: h = sqrt(2m)*sigma
print("Fig6: poly approximations of exp/gaussian ...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
x = np.linspace(-3.0, 4.0, 600)
ax = axes[0]
ax.plot(x, np.exp(-x), 'k-', lw=2.5, label='exp(-x) exact')
for n in [4, 8, 16, 32]:
    ax.plot(x, pex(-x, n), '--', label=f'(1-x/{n})^{n}')
ax.set_ylim(-0.1, 6); ax.set_title('poly-exp: (1-x/n)^n -> exp(-x)'); ax.set_xlabel('x = b*(r-r0)')
ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax = axes[1]                                       # rel error, log scale
for n in [4, 8, 16, 32]:
    ex, ap = np.exp(-x), pex(-x, n) + 1e-300
    ax.semilogy(x, np.abs(ap - ex) / ex, label=f'n={n}')
ax.set_ylim(1e-6, 1e1); ax.set_title('|pex - exp| / exp (rel err)'); ax.set_xlabel('x')
ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax = axes[2]                                       # bumps vs gaussians (curvature-matched h)
u = np.linspace(-1.6, 1.6, 600)
for sig, m in [(0.25, 4), (0.35, 4), (0.25, 8)]:
    h = np.sqrt(2 * m) * sig
    ax.plot(u, np.exp(-0.5 * (u / sig) ** 2), '--', label=f'gauss s={sig}')
    ax.plot(u, np.where(np.abs(u / h) < 1, (1 - (u / h) ** 2) ** m, 0), '-', label=f'(1-(u/{h:.2f})^2)^{m}  h=sqrt(2m)s')
ax.plot(u, np.where(np.abs(u) < 1, (1 - u ** 2) ** 4, 0), 'k-', lw=2, label='(1-u^2)^4 ref')
ax.set_title('compact poly bumps vs gaussians'); ax.set_xlabel('rho [A]')
ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.suptitle('Polynomial approximations: exp via squaring, gaussian via (1-u^2)^m')
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'poly_approx.png'), dpi=150); plt.close(fig)
print("  wrote poly_approx.png", flush=True)

print(f"\nDone.\nREVIEW: {OUTDIR}/gate_shapes.png\nREVIEW: {OUTDIR}/pair_E_xy.png\nREVIEW: {OUTDIR}/radial_cuts.png\nREVIEW: {OUTDIR}/pair2_E_xy.png\nREVIEW: {OUTDIR}/ring_E_xy.png\nREVIEW: {OUTDIR}/poly_approx.png", flush=True)
