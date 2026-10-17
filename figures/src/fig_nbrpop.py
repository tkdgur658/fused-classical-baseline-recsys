"""Neighbor x popularity heatmap (internal review 2026-10-07, Table 6 proposal 4): change of the top-20 hit rate of
held-out test targets from MSCA to the compared model, crossed by the relation of the target to the training history
(co-occurrence neighbor of >= 1 history item or not; rows) and the popularity quintile of its training degree (Q1 = least
popular fifth .. Q5; columns), one panel per dataset.

Data: tables/hyp_nbrpop.csv (tables/src/collect_nbrpop.py from the per-target cells of P04_hyp_strat, graph 'cooc' =
the MSCA co-occurrence graph, quintiles = the rule of fig_h_strat_pop). Compared model = column model_b of the CSV:
'MES' since 2026-10-10 (collect_nbrpop.py --arm mes, MAIN-table MES checkpoints; the earlier design E + R + A = --arm xre).
A dataset absent from the CSV (collector run with --partial) is drawn as a grey 'pending' panel (interim only).
Cell colour and number = mean over seeds 1-5 of 100 * (hit20_b - hit20_a) on the same seed (percentage points, signed,
diverging map centred at zero, clipped at +-VMAX with the true value printed); small number = targets in the cell
(identical in every seed); corner dot = all five seeds have the same sign.
Usage: python -s figures/src/fig_nbrpop.py [--png <preview.png>]
"""
import csv
import statistics as st
import sys

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from style import COL, TABLES, plt, save, setup

SRC = TABLES / 'hyp_nbrpop.csv'
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
SEEDS = ['1', '2', '3', '4', '5']
LINKS = [('nbr', 'Neighbor'), ('non', 'Other')]
Q = [f'q{q}' for q in range(1, 6)]
VMAX = 8.0


def main():
    setup()
    plt.rcParams.update({'mathtext.fontset': 'stix'})
    rows = [r for r in csv.DictReader(open(SRC, newline='', encoding='utf-8')) if r['graph'] == 'cooc']
    label = {r['model_b'] for r in rows}
    assert len(label) == 1, label
    label = label.pop()
    g, n = {}, {}
    for r in rows:
        g.setdefault((r['dataset'], r['link'], r['quintile']), {})[r['seed']] = float(r['gain_pp'])
        n.setdefault((r['dataset'], r['link'], r['quintile']), set()).add(int(r['n_targets']))
    norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0.0, vmax=VMAX)
    cmap = plt.get_cmap('RdBu')
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.05), sharex=True, sharey=True)
    for k, (ds, D) in enumerate(DATASETS):
        ax = axes[k // 2][k % 2]
        if (ds, 'nbr', 'q1') not in g:   # interim only (collect_nbrpop.py --partial): cells of this dataset not ready
            ax.imshow(np.zeros((2, 5)), cmap='Greys', vmin=0, vmax=1, aspect='auto')
            ax.text(2, 0.5, f'{label} seeds pending', ha='center', va='center', fontsize=6.2, color='#7f7f7f')
            ax.set_title(D, pad=2, fontsize=7)
            ax.set_xticks(range(5)); ax.set_xticklabels([f'Q{q}' for q in range(1, 6)])
            ax.set_yticks(range(2)); ax.set_yticklabels([lab for _, lab in LINKS])
            ax.tick_params(length=0, pad=1.5)
            continue
        M = np.zeros((2, 5))
        for i, (lk, _) in enumerate(LINKS):
            for j, q in enumerate(Q):
                xs = [g[(ds, lk, q)][s] for s in SEEDS]
                M[i, j] = st.mean(xs)
                assert len(n[(ds, lk, q)]) == 1
                rgba = cmap(norm(np.clip(M[i, j], -VMAX, VMAX)))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                tc = 'white' if lum < 0.5 else 'black'
                txt = f'{M[i, j]:+.1f}'
                txt = '0.0' if txt in ('+0.0', '-0.0') else txt.replace('-', '−')   # no sign on a rounded zero
                ax.text(j, i - 0.11, txt, ha='center', va='center', fontsize=6.2, color=tc)
                ax.text(j, i + 0.25, f'{n[(ds, lk, q)].pop()}', ha='center', va='center', fontsize=4.8, color=tc)
                if all(x > 0 for x in xs) or all(x < 0 for x in xs):
                    ax.plot(j + 0.39, i - 0.36, 'o', ms=1.6, color=tc, mew=0)
        ax.imshow(np.clip(M, -VMAX, VMAX), cmap=cmap, norm=norm, aspect='auto')
        ax.set_title(D, pad=2, fontsize=7)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f'Q{q}' for q in range(1, 6)])
        ax.set_yticks(range(2))
        ax.set_yticklabels([lab for _, lab in LINKS])
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        ax.set_xticks(np.arange(-0.5, 5), minor=True)
        ax.set_yticks(np.arange(-0.5, 2), minor=True)
        ax.grid(which='minor', color='white', lw=0.6)
        ax.tick_params(which='minor', length=0)
    fig.tight_layout(h_pad=0.4, w_pad=0.6, rect=(0, 0.17, 1, 1))
    fig.text(0.56, 0.185, 'Popularity quintile of the target (training degree)', ha='center', va='center', fontsize=7)
    cax = fig.add_axes([0.3, 0.09, 0.5, 0.022])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='both', ticks=[-8, -4, 0, 4, 8])
    cb.ax.tick_params(labelsize=5.8, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label(f'Hit-rate change (points), {label} minus MSCA', fontsize=6.2, labelpad=1.5)
    save(fig, 'fig_nbrpop', [SRC])
    if '--png' in sys.argv:
        fig.savefig(sys.argv[sys.argv.index('--png') + 1], dpi=300)


if __name__ == '__main__':
    main()
