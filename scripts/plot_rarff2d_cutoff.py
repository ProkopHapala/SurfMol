#!/usr/bin/env python3
"""Show how to get a SHORT, SMOOTH cutoff for the rarff2d pair potential.
Option A (chosen): integer squaring count m of pex — pex(x)=(1+x/2^m)^(2^m),
          exact 0 at x=-2^m -> support r0+2^m/b (m=3->5.7, 4->10.2, 5->19.1 A).
          No taper needed — that IS the point of the pexp design.
Option B: keep m=5, multiply e by smootherstep taper w(u)=10u^3-15u^4+6u^5
          on u=(r-r0)/(R-r0) — C2 at r0 and R seams, exact 0 at R.
Params (TYPE_SP2 x2): r0=1.3, pair b=1.8, a=1.0.
-> debug/rarff2d_cutoff/cutoff.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R0, B, A = 1.3, 1.8, 1.0
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'debug', 'rarff2d_cutoff'))
os.makedirs(OUT, exist_ok=True)

def pex(x, N):
    s = 1.0 + x / N
    return np.where(s > 0, np.maximum(s, 0) ** int(N), 0.0)

def w_smooth(u):  # taper = 1 - smootherstep(u): 1 at r0, 0 at R, C2 at both ends
    u = np.clip(u, 0.0, 1.0)
    return 1.0 - u**3 * (6*u*u - 15*u + 10)

r = np.linspace(0.2, 7.5, 1200)
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

# --- panel 1: radial factor e(r) for PEXP_N variants vs true exp
ax[0].plot(r, np.exp(-B * (r - R0)), 'k--', lw=2, label='true exp(-bΔr)')
for N, c in [(32, 'C0'), (16, 'C1'), (8, 'C2'), (4, 'C3')]:
    ax[0].plot(r, pex(-B * (r - R0), N), c, lw=1.5, label=f'pex N={N} (supp {R0 + N / B:.1f})')
    ax[0].axvline(R0 + N / B, color=c, ls=':', lw=0.8)
ax[0].axvline(6.0, color='r', ls='--', lw=1.5, label='rcut=6 (current)')
ax[0].set_yscale('log'); ax[0].set_ylim(1e-4, 50); ax[0].set_xlim(0, 7.5)
ax[0].set_title('radial factor e(r) — support = r0+N/b'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[0].set_xlabel('r [A]'); ax[0].set_ylabel('e(r)')

# --- panel 2: full pair energy E = a(e^2 - 2e) at aligned gate (g=1)
for N, c in [(32, 'C0'), (8, 'C2'), (4, 'C3')]:
    e = pex(-B * (r - R0), N)
    ax[1].plot(r, A * (e**2 - 2 * e), c, lw=1.5, label=f'N={N}')
    ax[1].axvline(R0 + N / B, color=c, ls=':', lw=0.8)
ax[1].axvline(6.0, color='r', ls='--', lw=1.5)
e32 = pex(-B * (r - R0), 32)
ax[1].plot(r, A * (e32**2 - 2 * e32), 'k', lw=0.5, alpha=0.4)
ax[1].set_title('pair E(r) at g=1 — attraction tail length set by N'); ax[1].legend(fontsize=8)
ax[1].set_xlabel('r [A]'); ax[1].set_ylabel('E [eV]'); ax[1].grid(alpha=0.3); ax[1].set_ylim(-1.3, 2.5)

# --- panel 3: hard cut at 6 vs smootherstep taper at R=4 and R=3
for R, c in [(6.0, 'C1'), (4.0, 'C2'), (3.0, 'C3')]:
    u = (r - R0) / (R - R0)
    w = w_smooth(u)
    e_t = e32 * w
    ax[2].plot(r, A * (e_t**2 - 2 * e_t), c, lw=1.8, label=f'tapered R={R} A')
# current behavior: hard chop at rcut=6 (discontinuity inset)
e_hard = np.where(r <= 6.0, e32, 0.0)
ax[2].plot(r, A * (e_hard**2 - 2 * e_hard), 'r--', lw=1.2, label='hard cut @6 (current)')
ax[2].set_title('PEXPN=32 + smootherstep taper -> C2-smooth cutoff anywhere')
ax[2].legend(fontsize=8); ax[2].set_xlabel('r [A]'); ax[2].set_ylabel('E [eV]')
ax[2].grid(alpha=0.3); ax[2].set_ylim(-1.3, 2.5); ax[2].set_xlim(0, 7.5)

fig.suptitle('RARFF-2D cutoff: support r0+N/b vs explicit smooth taper — hard rcut chops E≠0 (r=6: e≈2e-4)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'cutoff.png'), dpi=150)
print(f"wrote {OUT}/cutoff.png")
