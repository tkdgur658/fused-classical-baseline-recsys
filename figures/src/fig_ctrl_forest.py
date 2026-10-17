"""Forest plot of the controlled-component contrasts of MES on MSCA (fig:ctrl_forest), style of fig_h_forest.py.

Data: tables/ctrl_contrasts.csv rows kind == 'forest' (tables/src/build_ctrl.py; seeds 1-5, one run per seed). Each row is
one contrast a - b of test Recall@20 with the same seed in a and b (E = masked entry pairs, R = retention pairs,
A = anchor term):
  (1) (E + R) - (E + R, full view)            (2) (E + R + A) - (E + R + A, full view)      masking
  (3) (E + R + A) - (E + R)  (4) (E + R + A) - (E + A)  (5) (E + A) - (R + A)                parts
  (6) (E + R + A) - (E + R + A, matched DNS)  (7) (A + masked DNS) - (A + masked BPR)        partner
  (8) (BPR + mu best top-K loss) - (E + R + A), best = highest mean per dataset              top-K (own panel below,
      because its differences are several times larger than the others)
  (0) MES - MSCA, (9) (E + A) - MES, (10) (E + R) - MES   only once tables/mes_perseed.csv exists (group MES on top)
x = 1000 x the difference. Marker = mean over seeds, bar = unadjusted 95% paired t-interval (df 4), small grey dots = the
five per-seed differences. Filled marker = Holm-significant (alpha 0.05, families of build_ctrl.py), open = not.
Datasets differ by marker and vertical offset, one color (C_OURS) as fig_h_forest.py.
Usage: python -s figures/src/fig_ctrl_forest.py
"""
import csv

from style import C_GREY, C_OURS, COL, TABLES, plt, save, setup

SRC = TABLES / 'ctrl_contrasts.csv'
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
MARK = {'baby': 'o', 'pantry': 's', 'scientific': '^', 'sports': 'D'}
LABELS = {0: '(0) E $-$ MSCA', 9: '(9) E+A $-$ E', 10: '(10) E+R $-$ E',
          1: '(1) E+R: masked $-$ full', 2: '(2) E+R+A: masked $-$ full',
          3: '(3) E+R+A $-$ E+R', 4: '(4) E+R+A $-$ E+A', 5: '(5) E+A $-$ R+A',
          6: '(6) E+R+A $-$ matched DNS', 7: '(7) A+DNS $-$ A+BPR', 8: '(8) best top-$K$ $-$ E+R+A'}
ORDER = [0, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8]
GROUPS = ['MES', 'Masking', 'Parts', 'Competitor', 'Top-K']
STEP = 0.19


def draw(ax, rows, ids, ds):
    pos = {i: -j for j, i in enumerate(ids)}
    ax.axvline(0, color='black', lw=0.6, zorder=1)
    for r in rows:
        i = int(r['fid'])
        if i not in pos:
            continue
        k = [d for d, _ in ds].index(r['dataset'])
        y = pos[i] - (k - (len(ds) - 1) / 2) * STEP          # first dataset on top
        seeds = [1e3 * float(r[f'd_s{s}']) for s in range(1, 6)]
        m, lo, hi = (1e3 * float(r[c]) for c in ('diff', 'ci_lo', 'ci_hi'))
        full = r['holm_reject'] == '1'
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], color=C_OURS, elinewidth=0.8, capsize=1.3, capthick=0.6, ls='',
                    zorder=2)
        ax.scatter(seeds, [y] * 5, s=3.2, color=C_GREY, alpha=0.8, linewidths=0, zorder=3)
        ax.plot(m, y, color=C_OURS, marker=MARK[r['dataset']], ms=3.2, mfc=C_OURS if full else 'white', mew=0.7, ls='',
                zorder=4)
    grp = {int(r['fid']): r['group'] for r in rows}
    names = [g for g in GROUPS if any(grp.get(i) == g for i in ids)]
    for j, name in enumerate(names):
        ys = [pos[i] for i in ids if grp[i] == name]
        if j:
            ax.axhline(max(ys) + 0.5, color=C_GREY, lw=0.4, zorder=0)
        ax.annotate(name, xy=(1.015, (max(ys) + min(ys)) / 2), xycoords=('axes fraction', 'data'), rotation=270,
                    ha='left', va='center', fontsize=6.2, style='italic', annotation_clip=False)
    ax.set_yticks([pos[i] for i in ids])
    ax.set_yticklabels([LABELS[i] for i in ids])
    ax.set_ylim(-len(ids) + 0.5, 0.5)
    ax.grid(axis='x', lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=1.5)
    ax.tick_params(axis='y', length=0)


def main():
    setup()
    plt.rcParams['mathtext.fontset'] = 'stix'
    with open(SRC, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['kind'] == 'forest']
    ds = [(d, D) for d, D in DATASETS if any(r['dataset'] == d for r in rows)]
    have = {int(r['fid']) for r in rows}
    top = [i for i in ORDER if i in have and i != 8]
    bot = [8]
    h_top, h_bot = 0.235 * len(top), 0.235
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(COL, 0.75 + h_top + h_bot), layout='constrained',
                                  gridspec_kw=dict(height_ratios=[h_top, h_bot]))
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.04)
    draw(ax, rows, top, ds)
    draw(ax2, rows, bot, ds)
    ax2.set_xlabel(r'$10^3 \times \Delta$Recall@20 (same seed)', labelpad=1.5)
    # the two panels have their own x-range; the lower one keeps 0 visible so the sign reads at a glance
    lo = min(1e3 * float(r['ci_lo']) for r in rows if int(r['fid']) == 8) - 0.6
    lo = min(lo, min(1e3 * float(r[f'd_s{s}']) for r in rows if int(r['fid']) == 8 for s in range(1, 6)) - 0.4)
    ax2.set_xlim(lo, 1.0)
    hs = [plt.Line2D([], [], color=C_OURS, marker=MARK[d], ls='', ms=3.2, mew=0.7) for d, _ in ds]
    hs += [plt.Line2D([], [], color=C_OURS, marker='o', ls='', ms=3.2, mfc='white', mew=0.7),
           plt.Line2D([], [], color=C_GREY, marker='o', ls='', ms=2.0, mew=0, alpha=0.8)]
    labs = [D for _, D in ds] + ['Holm n.s.', 'One seed']
    ax.legend(hs, labs, loc='lower center', bbox_to_anchor=(0.42, 1.0), ncol=len(hs), frameon=False, handlelength=0.8,
              handletextpad=0.3, columnspacing=0.6, borderaxespad=0.2, fontsize=6.2)
    save(fig, 'fig_ctrl_forest', [SRC])


if __name__ == '__main__':
    main()
