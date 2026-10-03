#!/usr/bin/env python3
"""Plot rarff2d_assemble artifacts: random atoms -> self-assembled molecule.
Reads debug/rarff2d_assemble{,_seedN}/{relax,atoms,bonds}.csv written by
`cargo run --release --bin rarff2d_assemble --seed N`.
Writes <dir>/assemble.png per directory.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DBG = os.path.normpath(os.path.join(HERE, '..', 'debug'))
dirs = sys.argv[1:] or ['rarff2d_assemble', 'rarff2d_assemble_seed2', 'rarff2d_assemble_seed3']
dirs = [d if os.path.isabs(d) else os.path.join(DBG, d) for d in dirs]

fig, axes = plt.subplots(len(dirs), 2, figsize=(11, 4.6 * len(dirs)))
if len(dirs) == 1:
    axes = axes.reshape(1, 2)

for row, D in enumerate(dirs):
    tag = os.path.basename(D)
    rel = np.genfromtxt(f'{D}/relax.csv', delimiter=',', names=True)
    at = np.genfromtxt(f'{D}/atoms.csv', delimiter=',', names=True)
    bd = np.genfromtxt(f'{D}/bonds.csv', delimiter=',', names=True)
    if bd.ndim == 0:
        bd = bd.reshape(1)
    init = np.genfromtxt(f'{D}/init.csv', delimiter=',', names=True) if os.path.exists(f'{D}/init.csv') else None

    ax = axes[row, 0]
    ax.plot(rel['step'], rel['E'], 'b-', lw=1.5)
    ax.set_xlabel('MD step'); ax.set_ylabel('E [eV]')
    ax2 = ax.twinx()
    ax2.semilogy(rel['step'], np.maximum(rel['maxF'], 1e-12), 'r-', lw=0.8)
    ax2.set_ylabel('max|F|', color='r')
    ax.set_title(f'{tag} — assembly convergence'); ax.grid(alpha=0.3)

    ax = axes[row, 1]
    for b in bd:
        i, j = int(b['i']), int(b['j'])
        ax.plot([at['x'][i], at['x'][j]], [at['y'][i], at['y'][j]], 'k-', lw=2.2, zorder=3)
    if init is not None:
        ax.scatter(init['x'], init['y'], facecolors='none', edgecolors='k', s=50, alpha=0.4, label='initial')
        for i in range(len(at)):
            ax.annotate('', xy=(at['x'][i], at['y'][i]), xytext=(init['x'][i], init['y'][i]),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5))
    sc = ax.scatter(at['x'], at['y'], c=at['eatom'], cmap='RdYlGn_r', s=90, zorder=5, vmin=-1.8, vmax=0)
    for i in range(len(at)):
        ax.text(at['x'][i] + 0.08, at['y'][i] + 0.08, str(int(at['i'][i])), fontsize=7)
    ax.set_aspect('equal'); ax.set_title('assembled graph (color = E_atom)')
    ax.set_xlabel('x [A]'); ax.set_ylabel('y [A]')
    fig.colorbar(sc, ax=ax, label='E_atom [eV]')

fig.suptitle('rarff2d self-assembly: random sp2 atoms -> bonded molecule')
fig.tight_layout()
out = os.path.join(DBG, 'rarff2d_assemble', 'assemble.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}', flush=True)
