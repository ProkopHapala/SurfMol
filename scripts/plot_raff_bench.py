#!/usr/bin/env python3
"""Plot RAFF benchmark convergence curves from CSVs.

Reads debug/raff_bench/{molecule}_{distortion}_{solver}.csv
Generates log-scale PNG plots:
  - RMSD vs macrostep (log Y)
  - max|F| vs macrostep (log Y)
  - RMSD vs n_evals (log Y) — cross-solver performance objective
  - Summary bar chart: steps-to-T2 and steps-to-T1

Usage: python3 scripts/plot_raff_bench.py
Output: debug/raff_bench/*.png
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, 'debug', 'raff_bench')

# Solver display config: (label, color, linestyle)
SOLVER_STYLE = {
    'FIRE':                   ('FIRE',                  '#e41a1c', '-'),
    'Inertial-dt0.005':       ('Inertial dt=0.005',    '#ff7f00', ':'),
    'Inertial-dt0.01':        ('Inertial dt=0.01',     '#ff7f00', '--'),
    'Inertial-dt0.02':        ('Inertial dt=0.02',     '#ff7f00', '-'),
    'DampedMD':               ('Damped MD',             '#f781bf', ':'),
    # Legacy projection-only (no inertia)
    'PBD-or1.9':              ('PBD or=1.9 (no inertia)',   '#377eb8', '-'),
    'XPBD-dt0.05':            ('XPBD dt=0.05 (no inertia)', '#4daf4a', '--'),
    'XPBD-dt0.2':             ('XPBD dt=0.2 (no inertia)',  '#4daf4a', '-.'),
    'Projective-dt0.05':      ('Proj dt=0.05 (no inertia)', '#984ea3', '--'),
    'Projective-dt0.2':       ('Proj dt=0.2 (no inertia)',  '#984ea3', '-.'),
    # Proper PD with outer inertia; reset/free and active/no heavy-ball are explicit
    'PD-Proj-dt0.05-i4-reset':      ('PD-Proj dt=.05 i4 HB reset', '#a65628', '--'),
    'PD-Proj-dt0.1-i4-reset':       ('PD-Proj dt=.1 i4 HB reset',  '#a65628', '-'),
    'PD-Proj-dt0.1-i4-nohb-reset':  ('PD-Proj adiabatic dt=.1 i4', '#1b9e77', '-'),
    'PD-Proj-dt0.1-i3-nohb-reset':  ('PD-Proj adiabatic dt=.1 i3', '#1b9e77', '--'),
    'PD-Proj-dt0.15-i3-nohb-reset': ('PD-Proj adiabatic dt=.15 i3','#1b9e77', ':'),
    'PD-Proj-dt0.2-i4-reset':       ('PD-Proj dt=.2 i4 HB reset',  '#a65628', '-.'),
    'PD-Proj-dt0.2-i4-free':        ('PD-Proj dt=.2 i4 HB free',   '#d95f02', '-'),
    'PD-Proj-dt0.2-i4-nohb':        ('PD-Proj dt=.2 i4 free',      '#7570b3', '-'),
    'PD-Proj-dt0.1-i2-reset':       ('PD-Proj dt=.1 i2 reset',     '#f781bf', '--'),
    'PD-Proj-dt0.1-i8-reset':       ('PD-Proj dt=.1 i8 HB reset',  '#f781bf', '-'),
    'PD-XPBD-dt0.1-i4-reset':       ('PD-XPBD dt=.1 i4 reset',     '#999999', '--'),
    'PD-XPBD-dt0.2-i4-reset':       ('PD-XPBD dt=.2 i4 reset',     '#999999', '-'),
    'PD-ProjDyn-dt0.005-i4-reset':  ('PD-Proj dynamic dt=.005 i4', '#e6ab02', ':'),
    'PD-ProjDyn-dt0.01-i4-reset':   ('PD-Proj dynamic dt=.01 i4',  '#e6ab02', '--'),
    'PD-ProjDyn-dt0.02-i4-reset':   ('PD-Proj dynamic dt=.02 i4',  '#e6ab02', '-.'),
    'PD-ProjDyn-dt0.05-i4-reset':   ('PD-Proj dynamic dt=.05 i4',  '#e6ab02', '-'),
    'PD-ProjDyn-dt0.1-i4-reset':    ('PD-Proj dynamic dt=.1 i4',   '#d95f02', '-'),
}

def load_csv(path):
    """Load CSV -> structured array with step, rmsd, max_f, optional max_t, n_evals."""
    return np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')

def plot_convergence(traces, xlabel, xkey, ylabel, ykey, title, fname, ref_line=None):
    """Plot one figure: y vs x for all traces (log Y)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data in traces.items():
        if xkey not in data.dtype.names or ykey not in data.dtype.names:
            continue
        x = data[xkey]
        y = np.maximum(data[ykey], 1e-16)  # clamp for log scale
        style = SOLVER_STYLE.get(label, (label, '#999', '-'))
        ax.plot(x, y, label=style[0], color=style[1], linestyle=style[2], linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    if ref_line is not None:
        ax.axhline(ref_line[0], color='gray', linestyle=':', alpha=0.5, label=ref_line[1])
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")

def main():
    csvs = sorted(glob.glob(os.path.join(OUTDIR, '*.csv')))
    if not csvs:
        print(f"No CSVs found in {OUTDIR}. Run raff_bench first:")
        print("  cargo run --release -p molff --bin raff_bench")
        sys.exit(1)

    # Group CSVs by (molecule, distortion)
    groups = {}
    for path in csvs:
        base = os.path.basename(path).replace('.csv', '')
        parts = base.split('_')
        if len(parts) < 3:
            continue
        # Format: {molecule}_{distortion}_{solver}
        # distortion: D1_random, D2_stretch, D3a_dihedral
        # solver: ForceMD, PBD-or1.9, etc.
        dist_idx = None
        for i, p in enumerate(parts):
            if p.startswith('D1') or p.startswith('D2') or p.startswith('D3a'):
                dist_idx = i
                break
        if dist_idx is None:
            print(f"  skip {base}: no distortion prefix")
            continue
        mol = '_'.join(parts[:dist_idx])
        dist = '_'.join(parts[dist_idx:dist_idx+2])
        solver = '_'.join(parts[dist_idx+2:])
        key = (mol, dist)
        if key not in groups:
            groups[key] = {}
        data = load_csv(path)
        groups[key][solver] = data

    print(f"Loaded {len(csvs)} CSVs, {len(groups)} (molecule, distortion) groups")

    # Per-group plots: max|F| vs step (PRIMARY), max|F| vs evals (PRIMARY), RMSD vs step (secondary)
    for (mol, dist), traces in sorted(groups.items()):
        print(f"\n=== {mol} / {dist} ({len(traces)} solvers) ===")
        # PRIMARY: max|F| vs macrostep (DOF-independent)
        plot_convergence(traces, 'macrostep', 'step', 'max|F| [eV/A]', 'max_f',
                         f'{mol} / {dist}: max force vs macrostep (PRIMARY)',
                         f'{mol}_{dist}_force_vs_step.png',
                         ref_line=(0.1, 'T2 (rough)'))
        # Rotational residual for new-format traces; required for dynamic-orientation convergence.
        if any('max_t' in data.dtype.names for data in traces.values()):
            plot_convergence(traces, 'macrostep', 'step', 'max|tau| [eV]', 'max_t',
                             f'{mol} / {dist}: max torque vs macrostep',
                             f'{mol}_{dist}_torque_vs_step.png',
                             ref_line=(0.1, 'T2 (rough)'))
        # PRIMARY: max|F| vs n_evals (cross-solver perf, DOF-independent)
        plot_convergence(traces, 'n_evals (port-force evals)', 'n_evals', 'max|F| [eV/A]', 'max_f',
                         f'{mol} / {dist}: max force vs soft evals (PRIMARY)',
                         f'{mol}_{dist}_force_vs_evals.png',
                         ref_line=(0.1, 'T2 (rough)'))
        # SECONDARY: RMSD vs macrostep (meaningless for free-DOF molecules — dihedrals, branch rotations)
        plot_convergence(traces, 'macrostep', 'step', 'RMSD [A] (secondary)', 'rmsd',
                         f'{mol} / {dist}: residual RMSD vs macrostep (secondary — may plateau for free DOFs)',
                         f'{mol}_{dist}_rmsd_vs_step.png',
                         ref_line=(0.05, 'T2 (rough)'))

    # Summary bar chart: steps-to-T2 and steps-to-T1 (FORCE-BASED thresholds, DOF-independent)
    print("\n=== Summary bar chart ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, target, thresh in zip(axes, ['T2 (rough: |F|<0.1 eV/A)', 'T1 (accurate: |F|<1e-3 = 1 meV/A)'], [0.1, 1e-3]):
        all_solvers = sorted(set(s for traces in groups.values() for s in traces.keys()))
        all_moldist = sorted(groups.keys())
        x = np.arange(len(all_moldist))
        width = 0.8 / max(len(all_solvers), 1)
        for si, solver in enumerate(all_solvers):
            vals = []
            for md in all_moldist:
                traces = groups[md]
                if solver not in traces:
                    vals.append(0)
                    continue
                data = traces[solver]
                max_f = np.atleast_1d(data['max_f'])
                steps = np.atleast_1d(data['step'])
                converged = max_f < thresh
                if 'max_t' in data.dtype.names:
                    converged &= np.atleast_1d(data['max_t']) < thresh
                hit = np.where(converged)[0]
                vals.append(int(steps[hit[0]]) if len(hit) > 0 else int(max(steps) * 2))
            style = SOLVER_STYLE.get(solver, (solver, '#999', '-'))
            ax.bar(x + si * width, vals, width, label=style[0], color=style[1])
        ax.set_xticks(x + width * (len(all_solvers)-1) / 2)
        ax.set_xticklabels([f'{m}\n{d}' for m, d in all_moldist], fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('steps to convergence')
        ax.set_title(f'Steps to {target}')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'summary_steps_to_target.png'), dpi=150)
    plt.close(fig)
    print("  wrote summary_steps_to_target.png")

    print(f"\nDone. PNGs in {OUTDIR}/")

if __name__ == '__main__':
    main()
