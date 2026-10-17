"""Figure 1 (teaser, page width): the credit protocol F / N' / C, with MES as one row.  [added at revision log 12]

Panel F  fused baseline: Recall@20 of every compared model before (x) and after (y) the training-free fusion F, one dot
         per model-dataset pair (50 pairs, 8 datasets); MSCA + MES highlighted. The margin of MSCA + MES (+) over MSCA (+)
         on the four datasets with both rows is printed from the same table (-0.0005 ... +0.0011).
         Sources: tables/src/data/fusion_summary_9ds_mes.csv (50 pairs), tables/src/data/fusion_summary.csv (margin, 4 ds).
         Seeds: 1-5 per pair, except COHESION Scientific (3), LGMRec Pantry (3), Scientific (2), Sports (4).
Panel N' per-user oracle headroom g_O (gain of the better of the learned ranker and the best classical ranker per user)
         against the N3 decoy-null interval (same-stratum decoys, 200 draws), 8 datasets. revision log 18-2: the learned ranker
         is MES (R2NULL3; MicroLens has no MES run). Source: tables/src/data/r2null3.json (copy of study/results/r2null3.json;
         per_ds.<ds>.g_O and per_ds.<ds>.N3.lo/hi/cls). Seeds 1-5. Before 18-2: teaser/r2null2.json (E + R + A design, 9 ds).
Panel C  targets with at most two training edges: hits of the learned model (grey) vs the classical fill (blue), in units
         of hits expected from a random list, 20 backbone-by-dataset cells (MSCA native 8, E + R + A 8, MES 4 datasets).
         Source: tables/src/data/teaser/d3_per_seed.json (copy of A_DESIGN/results/d3_per_seed.json; metric le2, arms
         base, q2_fcombo, q2_ext; mean over seeds 1-5), the file behind A_DESIGN/RESULT_C_REJUDGE.md (20 of 20 cells hold).
Usage: python -s figures/src/fig_teaser.py   (writes figures/fig_teaser.pdf, .png preview, .sources.json)
"""
import csv
import json
import statistics as st

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import C_CTRL, C_GREY, C_OURS, DCOL, FIG, ROOT, plt, save, setup

DATA = ROOT / 'tables' / 'src' / 'data'
SRC_F9 = DATA / 'fusion_summary_9ds_mes.csv'
SRC_F4 = DATA / 'fusion_summary.csv'
SRC_N = DATA / 'r2null3.json'   # revision log 18-2 (was teaser/r2null2.json, E + R + A design)
SRC_C = DATA / 'teaser' / 'd3_per_seed.json'
FS = 7.0
ZERO = 0.07
DSN = {'baby': 'Baby', 'pantry': 'Pantry', 'scientific': 'Scientific', 'sports': 'Sports', 'clothing': 'Clothing',
       'instruments': 'Instruments', 'arts': 'Arts', 'office': 'Office', 'microlens': 'MicroLens'}


def panel_f(ax):
    rows = list(csv.DictReader(open(SRC_F9, encoding='utf-8')))
    base = np.array([float(r['base_r20_mean']) for r in rows])
    fused = np.array([float(r['r20_mean']) for r in rows])
    mes = np.array([r['model'] == 'MSCA+MES' for r in rows])
    assert len(rows) == 50 and (fused > base).all(), 'F must lift all 50 pairs'
    lo, hi = 0.06, 0.16
    ax.plot([lo, hi], [lo, hi], color=C_GREY, lw=0.7, ls='--', zorder=1)
    ax.scatter(base[~mes], fused[~mes], s=11, color=C_GREY, alpha=0.8, lw=0, zorder=2)
    ax.scatter(base[mes], fused[mes], s=34, marker='*', color=C_OURS, lw=0, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [0.06, 0.10, 0.14]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(['.06', '.10', '.14'])
    ax.set_yticklabels(['.06', '.10', '.14'])
    ax.set_xlabel('Recall@20 without F', labelpad=1)
    ax.set_ylabel('Recall@20 with F', labelpad=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.97, 0.03, f'above the dashed line:\n{int((fused > base).sum())} of {len(rows)} pairs', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=FS)
    # margin MES (+) vs MSCA (+), four datasets, from fusion_summary.csv
    f4 = {(r['model'], r['dataset']): float(r['r20_mean']) for r in csv.DictReader(open(SRC_F4, encoding='utf-8'))}
    dm = [f4[('MSCA+MES', d)] - f4[('MSCA', d)] for d in ('baby', 'pantry', 'scientific', 'sports')]
    assert round(min(dm), 4) == -0.0005 and round(max(dm), 4) == 0.0011, dm
    h = [Line2D([], [], marker='*', color=C_OURS, ls='', ms=7, label='MSCA + MES'),
         Line2D([], [], marker='o', color=C_GREY, ls='', ms=3.2, label='seven other models')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False, fontsize=FS,
              handletextpad=0.2, columnspacing=0.8)
    ax.text(0.5, -0.36, 'MES $\\oplus$ minus MSCA $\\oplus$ on four datasets:\n$-$.0005 to $+$.0011', ha='center', va='top',
            transform=ax.transAxes, fontsize=FS)
    ax.set_title('F: fusion lifts every model', fontsize=FS + 0.5, pad=3)


def panel_n(ax):
    d = {k: v for k, v in json.load(open(SRC_N, encoding='utf-8'))['per_ds'].items() if v.get('status') == 'OK'}
    assert 'microlens' not in d and len(d) == 8
    names = list(d)
    names.sort(key=lambda k: -d[k]['g_O'])
    for i, k in enumerate(names):
        v = d[k]
        y = len(names) - 1 - i
        lo, hi = 100 * v['N3']['lo'], 100 * v['N3']['hi']
        ax.barh(y, hi - lo, left=lo, height=0.5, color='#c8c8c8', ec='none', zorder=1)
        above = v['N3']['cls'] == 'above'
        ax.scatter([100 * v['g_O']], [y], s=20, zorder=3, color=C_OURS if above else 'white', ec=C_OURS, lw=0.9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([DSN[k] for k in names][::-1])
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlim(0, 57)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel(r"Per-user headroom $g_O$ (%)", labelpad=1)
    ax.spines[['top', 'right']].set_visible(False)
    h = [Line2D([], [], marker='o', color=C_OURS, ls='', ms=4.5, label=r'$g_O$ of MES, above N3'),
         Line2D([], [], marker='o', color=C_OURS, mfc='white', ls='', ms=4.5, label=r'$g_O$ of MES, not above'),
         Patch(fc='#c8c8c8', label='decoy-null interval N3')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.42, -0.19), ncol=1, frameon=False, fontsize=FS,
              handletextpad=0.3, labelspacing=0.2)
    ax.set_title("N$'$ (exploratory): headroom vs. decoy null", fontsize=FS + 0.5, pad=3)
    return sum(d[k]['N3']['cls'] == 'above' for k in names), len(names)


def panel_c(ax):
    d = json.load(open(SRC_C, encoding='utf-8'))
    dss = sorted({k.split('__')[1] for k in d})
    mk = {'native': ('o', 'MSCA'), 'xre': ('^', 'E+R+A'), 'mes': ('D', 'MES')}
    off = {'native': 0.24, 'xre': 0.0, 'mes': -0.24}
    ncell = 0
    for k, v in d.items():
        mdl, ds = k.split('__')
        y = len(dss) - 1 - dss.index(ds) + off[mdl]
        learned = st.mean(v['per']['base']['le2'])
        fill = max(st.mean(v['per'][a]['le2']) for a in ('q2_fcombo', 'q2_ext'))
        learned = learned if learned > 0 else ZERO   # exact zeros are drawn at the left edge, tick labelled 0
        m = mk[mdl][0]
        ax.plot([learned, fill], [y, y], color='#bbbbbb', lw=0.6, zorder=1)
        ax.scatter([learned], [y], s=14, marker=m, color=C_GREY, lw=0, zorder=2)
        ax.scatter([fill], [y], s=14, marker=m, color=C_OURS, lw=0, zorder=3)
        ncell += 1
    ax.axvline(1, color='black', lw=0.5, ls=':', zorder=0)
    ax.text(1.12, -0.62, 'random', fontsize=FS, ha='left', va='center')
    ax.set_xscale('log')
    ax.set_xlim(0.05, 250)
    ax.set_xticks([ZERO, 1, 10, 100])
    ax.minorticks_off()
    ax.set_xticklabels(['0', '1', '10', '100'])
    ax.set_yticks(range(len(dss)))
    ax.set_yticklabels([DSN[k] for k in dss][::-1])
    ax.set_ylim(-0.8, len(dss) - 0.2)
    ax.set_xlabel(r'Hits at $\leq$2 training edges ($\times$ random)', labelpad=1)
    ax.spines[['top', 'right']].set_visible(False)
    h = [Line2D([], [], marker='s', color=C_GREY, ls='', ms=4.5, label='learned model'),
         Line2D([], [], marker='s', color=C_OURS, ls='', ms=4.5, label='classical fill'),
         Line2D([], [], marker='o', color='black', ls='', ms=3.5, label='MSCA'),
         Line2D([], [], marker='^', color='black', ls='', ms=3.5, label='E+R+A'),
         Line2D([], [], marker='D', color='black', ls='', ms=3.0, label='MES')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.45, -0.19), ncol=3, frameon=False, fontsize=FS,
              handletextpad=0.1, columnspacing=0.7, labelspacing=0.2)
    ax.set_title('C: classical fill serves low-degree targets', fontsize=FS + 0.5, pad=3)
    return ncell


def main():
    setup()
    plt.rcParams.update({'mathtext.fontset': 'stix', 'font.size': FS, 'axes.labelsize': FS + 0.5,
                         'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS, 'axes.titlesize': FS + 0.5})
    fig = plt.figure(figsize=(DCOL, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.42, left=0.065, right=0.995, top=0.90, bottom=0.33)
    panel_f(fig.add_subplot(gs[0]))
    nab, nn = panel_n(fig.add_subplot(gs[1]))
    nc = panel_c(fig.add_subplot(gs[2]))
    print('N3 above:', nab, 'of', nn, '; C cells:', nc)
    save(fig, 'fig_teaser', [SRC_F9, SRC_F4, SRC_N, SRC_C])
    fig.savefig(FIG / 'fig_teaser.png', dpi=450)


if __name__ == '__main__':
    main()
