#!/usr/bin/env python3
"""Plot the rarff2d_bench scenario reality-check: atoms + dx=6 grid cells +
atoms-per-cell histogram. Verifies the benchmark scene is physically sensible
(molecular fragments at bond spacing, not uniform noise or pathological clumps).
Reads debug/rarff2d_bench/scenario.csv -> debug/rarff2d_bench/scenario.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.normpath(os.path.join(HERE, '..', 'debug', 'rarff2d_bench'))
d = np.genfromtxt(os.path.join(OUTDIR, 'scenario.csv'), delimiter=',', names=True)
x, y, cell = d['x'], d['y'], d['cell'].astype(int)

DX = 6.0
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

ax = axes[0]
ax.scatter(x, y, s=12, c='k')
# bond-length circle per atom shows molecular spacing
for xi, yi in zip(x[::3], y[::3]):
    ax.add_patch(plt.Circle((xi, yi), 1.3, fill=False, color='C1', alpha=0.15, lw=0.5))
# grid lines — TRUE boundaries: bench uses origin=-(half+4), half = sqrt(n/(0.7*0.68))/2
half = np.sqrt(len(x) / (0.7 * 0.68)) * 0.5
ox = -(half + 4.0)
x0, x1 = x.min() - 2, x.max() + 2
y0, y1 = y.min() - 2, y.max() + 2
gx = ox + DX * np.arange(0, int((x1 - ox) / DX) + 1)
gy = ox + DX * np.arange(0, int((y1 - ox) / DX) + 1)
for v in gx: ax.axvline(v, color='C0', alpha=0.3, lw=0.7)
for v in gy: ax.axhline(v, color='C0', alpha=0.3, lw=0.7)
ax.set_aspect('equal'); ax.set_title(f'{len(x)} atoms in dx=rcut={DX} A cells (orange r=1.3 circles)')
ax.set_xlabel('x [A]'); ax.set_ylabel('y [A]')

# group AABBs: solid = position-fit box, dashed = +rcut halo (interaction reach)
try:
    g = np.genfromtxt(os.path.join(OUTDIR, 'groups.csv'), delimiter=',', names=True)
    import matplotlib.patches as mpatches
    if g.ndim == 0: g = g.reshape(1)
    for row in g:
        (x0, y0, x1, y1) = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='red', lw=1.2))
        ax.add_patch(mpatches.Rectangle((x0 - DX, y0 - DX), x1 - x0 + 2 * DX, y1 - y0 + 2 * DX,
                                        fill=False, edgecolor='red', lw=0.7, ls='--', alpha=0.5))
    ax.set_title(ax.get_title() + '\nred = group AABB fit (solid) + rcut halo (dashed)')
except FileNotFoundError:
    pass

ax = axes[1]
vals, counts = np.unique(cell, return_counts=True)
ax.hist(counts, bins=np.arange(0.5, counts.max() + 1.5), color='C2', edgecolor='k')
ax.set_xlabel('atoms per cell'); ax.set_ylabel('# cells')
ax.set_title(f'occupancy hist — {len(vals)} non-empty cells of ~{len(gx)*len(gy)} total\nmean {counts.mean():.1f} atoms/cell, max {counts.max()}')
ax.grid(alpha=0.3)

fig.suptitle('rarff2d_bench scenario: hexagonal lattice a=1.3 A, ~70% fill, jitter ±0.3 — realistic dense molecular sheet (non-commensurate w/ accel grid)')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'scenario.png'), dpi=150)
print(f"wrote {OUTDIR}/scenario.png")
