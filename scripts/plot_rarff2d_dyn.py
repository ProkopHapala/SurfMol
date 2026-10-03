#!/usr/bin/env python3
"""Plot rarff2d_demo artifacts: corrupted-naphthalene repair dynamics.
Reads debug/rarff2d_dyn/{clean,bad}_{relax,atoms}.csv + rings.csv written by
`cargo run --release --bin rarff2d_demo`. Writes debug/rarff2d_dyn/dyn_repair.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.normpath(os.path.join(HERE, '..', 'debug', 'rarff2d_dyn'))

rg = np.genfromtxt(f'{D}/rings.csv', delimiter=',', names=True)
if rg.ndim == 0:
    rg = rg.reshape(1)

scenes = [('clean', 'A) displaced atom only'), ('bad', 'B) +spurious atom in hole')]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
th = np.linspace(0, 2 * np.pi, 100)

for row, (tag, label) in enumerate(scenes):
    rel = np.genfromtxt(f'{D}/{tag}_relax.csv', delimiter=',', names=True)
    at = np.genfromtxt(f'{D}/{tag}_atoms.csv', delimiter=',', names=True)

    ax = axes[row, 0]                                    # convergence
    ax.plot(rel['step'], rel['E'], 'b-', lw=1.5, label='E_tot')
    ax.set_xlabel('GD step'); ax.set_ylabel('E [eV]', color='b')
    ax2 = ax.twinx()
    ax2.semilogy(rel['step'], np.maximum(rel['maxF'], 1e-12), 'r-', lw=0.8, label='max|F|')
    ax2.semilogy(rel['step'], np.maximum(rel['maxT'], 1e-12), 'm--', lw=0.8, label='max|T|')
    ax2.set_ylabel('residual', color='r')
    ax.set_title(f'{label} — convergence'); ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')

    ax = axes[row, 1]                                    # geometry before/after
    for r in rg:
        ax.plot(r['x'] + r['R'] * np.cos(th), r['y'] + r['R'] * np.sin(th), 'g--', lw=1)
        for k in range(int(r['nfold'])):
            t = np.pi / 2 + k * 2 * np.pi / r['nfold']
            ax.plot(r['x'] + r['R'] * np.cos(t), r['y'] + r['R'] * np.sin(t), 'g.', ms=8)
    ax.scatter(at['x0'], at['y0'], facecolors='none', edgecolors='k', s=60, label='initial (corrupted)')
    sc = ax.scatter(at['xf'], at['yf'], c=at['eatomf'], cmap='RdYlGn_r', s=70, zorder=5, vmin=-2.2, vmax=0, label='final')
    for i in range(len(at)):
        ax.annotate('', xy=(at['xf'][i], at['yf'][i]), xytext=(at['x0'][i], at['y0'][i]),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
        ax.text(at['xf'][i] + 0.05, at['yf'][i] + 0.05, str(int(at['i'][i])), fontsize=7)
    ax.set_aspect('equal'); ax.set_title('repair (arrows = move, color = E_atom)')
    ax.set_xlabel('x [A]'); ax.set_ylabel('y [A]'); ax.legend(fontsize=8, loc='upper left')
    fig.colorbar(sc, ax=ax, label='E_atom [eV]')

    ax = axes[row, 2]                                    # per-atom consistency
    x = np.arange(len(at))
    ax.bar(x - 0.2, at['eatom0'], 0.4, label='initial', color='salmon')
    ax.bar(x + 0.2, at['eatomf'], 0.4, label='final', color='seagreen')
    ax.set_xticks(x); ax.set_xlabel('atom index'); ax.set_ylabel('E_atom [eV]')
    ax.set_title('per-atom consistency'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle('rarff2d repair dynamics: corrupted naphthalene + ring entities')
fig.tight_layout()
out = f'{D}/dyn_repair.png'
fig.savefig(out, dpi=150)
print(f'wrote {out}', flush=True)
