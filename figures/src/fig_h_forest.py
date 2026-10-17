"""Hypothesis figure: forest plot of seed-paired contrasts that locate the gain of XRE over the backbone MSCA.

Data: tables/hyp_forest.csv (tables/src/collect_hyp_forest.py; the runs of Table III tab:xmask and tables/paired_ci.csv,
one run per seed, seeds 1-5). Each row is one contrast a - b of test Recall@20 with the same seed in a and b:
  (1) A - MSCA, (2) XRE - A                                  gain components
  (3) A + masked BPR - A, (4) A + masked DNS - A, (5) XRE - (A + masked DNS)    partner type
  (6) Exchange masked view - full view, (7) XRE masked view - XRE on full view, (8) = (7) - (6) per seed   masked view
  (9) A + masked DNS - A + masked BPR, (10) XRE - A + masked BPR     direct partner contrasts against a uniform partner
      (added 2026-10-07; appended after (8) so that the row numbers cited in the text stay valid)
  (11) A - A with a single probe: the anchor term with the strongest of the M = 50 probes against the same term with the
      first of the same 50 draws (reviewer control M = 1, tables/m1probe_perseed.csv; added 2026-10-07 after (10))
  (12) A + masked DNS - A + full-view DNS: masking with a sampled high-scoring partner (reviewer control E1,
      tables/e1_perseed.csv; added 2026-10-07). Rows (11) and (12) ran on all four datasets (Sports since 2026-10-07).
x = 1000 x the difference (absolute Recall@20; absolute rather than relative to MSCA because the CSV intervals equal the
macros of tables/macros_paired.tex digit for digit, and the contrasts of all datasets lie within about 0.006, so one
axis serves all). Marker = mean over seeds, bar = 95% paired t-interval (df 4), small grey dots = the five per-seed
differences. Datasets differ by marker and vertical offset, one color (C_OURS) as fig_h_sens.py. Filled marker = the
interval excludes zero, open marker = it contains zero. Thin grey lines separate the three groups named on the right.
All datasets are drawn alike (user decision 2026-10-07; the former grey marker style for mixed hardware is removed).
Usage: python -s figures/src/fig_h_forest.py
"""
import csv

from style import C_GREY, C_OURS, COL, TABLES, plt, save, setup

SRC = TABLES / 'hyp_forest.csv'
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
MARK = {'baby': 'o', 'pantry': 's', 'scientific': '^', 'sports': 'D'}
# 2026-10-07 (user decision): the figure shows six core contrasts, renumbered (1)-(6): gain decomposition, then partner type,
# then list occupants against a sampled high-scoring partner. Keys are the contrast ids of collect_hyp_forest.py, which
# still computes all twelve for the macros (the other six are cited in the text through Table III by name).
FIG = {1: 1, 2: 2, 3: 3, 4: 4, 9: 5, 5: 6}
LABELS = {1: '(1) A $-$ MSCA', 2: '(2) XRE $-$ A', 3: '(3) A+BPR $-$ A', 4: '(4) A+DNS $-$ A',
          5: '(5) A+DNS $-$ A+BPR', 6: '(6) XRE $-$ A+DNS'}
GROUP = {1: 'Components', 2: 'Components', 3: 'Partner', 4: 'Partner', 5: 'Partner', 6: 'Partner'}
GROUPS = ['Components', 'Partner']


def main():
    setup()
    plt.rcParams['mathtext.fontset'] = 'stix'   # Times-like math, as fig_h_sens.py
    with open(SRC, newline='', encoding='utf-8') as f:
        rows = [dict(r, id=str(FIG[int(r['id'])])) for r in csv.DictReader(f) if int(r['id']) in FIG]
    ds = [(d, D) for d, D in DATASETS if any(r['dataset'] == d for r in rows)]
    ids = sorted({int(r['id']) for r in rows})
    fig, ax = plt.subplots(figsize=(COL, 1.68))   # six contrasts (2026-10-07)
    ax.axvline(0, color='black', lw=0.6, zorder=1)
    step = 0.26 if len(ds) == 3 else 0.2
    for r in rows:
        i = int(r['id'])
        k = [d for d, _ in ds].index(r['dataset'])
        y = -i + (k - (len(ds) - 1) / 2) * step * -1          # first dataset on top
        seeds = [1e3 * float(r[f'd_s{s}']) for s in range(1, 6)]
        m, lo, hi = (1e3 * float(r[c]) for c in ('mean', 'ci_lo', 'ci_hi'))
        full = r['ci_excl_zero'] == '1'
        col = C_OURS
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], color=col, elinewidth=0.8, capsize=1.3, capthick=0.6, ls='',
                    zorder=2)
        ax.scatter(seeds, [y] * 5, s=3.2, color=C_GREY, alpha=0.8, linewidths=0, zorder=3)   # seeds above the bar
        ax.plot(m, y, color=col, marker=MARK[r['dataset']], ms=3.2, mfc=col if full else 'white', mew=0.7, ls='',
                zorder=4)
    # group separators and names
    gid = {g: sorted(i for i in ids if GROUP[i] == g) for g in GROUPS}
    for j, name in enumerate(GROUPS):
        top, bot = min(gid[name]), max(gid[name])
        if j:
            ax.axhline(-top + 0.5, color=C_GREY, lw=0.4, zorder=0)
        ax.annotate(name, xy=(1.015, (-top - bot) / 2), xycoords=('axes fraction', 'data'), rotation=270,
                    ha='left', va='center', fontsize=6.2, style='italic', annotation_clip=False)
    ax.set_yticks([-i for i in ids])
    ax.set_yticklabels([LABELS[i] for i in ids])
    ax.set_ylim(-max(ids) - 0.55, -min(ids) + 0.55)
    ax.set_xlabel(r'$10^3 \times \Delta$Recall@20 (same seed)', labelpad=1.5)
    ax.grid(axis='x', lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=1.5)
    ax.tick_params(axis='y', length=0)
    hs = [plt.Line2D([], [], color=C_OURS, marker=MARK[d], ls='', ms=3.2, mew=0.7) for d, _ in ds]
    hs += [plt.Line2D([], [], color=C_OURS, marker='o', ls='', ms=3.2, mfc='white', mew=0.7),
           plt.Line2D([], [], color=C_GREY, marker='o', ls='', ms=2.0, mew=0, alpha=0.8)]
    labs = [D for _, D in ds] + ['95% CI includes 0', 'One seed']
    ax.legend(hs, labs, loc='lower center', bbox_to_anchor=(0.45, 1.0),
              ncol=len(hs), frameon=False, handlelength=0.8, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.2,
              fontsize=6.2)
    fig.tight_layout()
    save(fig, 'fig_h_forest', [SRC])


if __name__ == '__main__':
    main()
