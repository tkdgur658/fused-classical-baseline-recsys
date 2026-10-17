"""Hypothesis figure: direction of the entry gradient (Pantry, Scientific), test Recall@20 over seeds 1-5.

Revival of fig_gradsplit.py with the manuscript row names. Data: tables/gradsplit_perseed.csv (per seed) and
tables/hyp_gradsplit_tests.csv (tables/src/collect_hyp_gradsplit.py; arm definitions there). MSCA, lambda 0.1, K 40, one
run per seed.
Marker = mean over seeds 1-5, horizontal bar = sample SD over the seeds, small dots = seeds. Dashed line = mean of
unmatched full-view XRE (E3, the base of the split; renamed from 'XRE on full view' in R2335 because Table III,
supplementary Table S6 and Fig. 4 use that name for the mass-matched variant). Grey band = the two single-direction arms. Filled triangle right of a row =
lower than unmatched full-view XRE under the pre-set rule (mean difference <= -0.0010 and lower on >= 4 of 5 seeds).
Masked-XRE arms (reviewer control E4, 2026-10-07): open C_OURS diamonds below the rows 'Raise positive only' and 'Lower
list occupant only' give the same split applied to the entry pair of the actual masked XRE (tables/e4_perseed.csv,
tables/src/collect_e4.py; mean +- SD over the finished seeds, dots for seeds); their reference is the row Proposed (XRE).
Colors: Proposed (XRE) = C_OURS, Backbone (MSCA) = C_GREY, every other arm = C_CTRL (as fig_ablation.py).
Usage: python -s figures/src/fig_h_gradsplit.py
"""
import csv
import statistics as st

from style import C_CTRL, C_GREY, C_OURS, COL, TABLES, plt, save, setup

SRC = TABLES / 'gradsplit_perseed.csv'
TESTS = TABLES / 'hyp_gradsplit_tests.csv'
E4 = TABLES / 'e4_perseed.csv'
E4ARM = {'pos': 'pos-only - XRE', 'disp': 'occ-only - XRE'}   # row of the full-view arm -> masked-XRE contrast
DATASETS = [('pantry', 'Pantry'), ('scientific', 'Scientific')]
ROWS = [('XRE', 'Proposed (XRE)'), ('e3', 'Unmatched full-view XRE'), ('pos', 'Raise positive only'),
        ('disp', 'Lower list occupant only'), ('bdrand', 'Lower random list item'), ('bdnoanc', 'Lower occupant, no anchor'),
        ('A1', 'Anchor term (A)'), ('Native', 'Backbone (MSCA)')]
SEEDS = [1, 2, 3, 4, 5]


def main():
    setup()
    plt.rcParams['mathtext.fontset'] = 'stix'   # Times-like math, as fig_method.py / fig_teaser.py
    v = {}
    with open(SRC, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            v[(r['dataset'], r['arm'], int(r['seed']))] = float(r['r20'])
    with open(TESTS, newline='', encoding='utf-8') as f:
        rule = {(t['dataset'], t['a']): t['rule'] for t in csv.DictReader(f) if t['family'] == 'vs_e3'}
    m4 = {}
    with open(E4, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            m4.setdefault((r['dataset'], r['contrast']), []).append(float(r['arm_r20']))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.72), sharey=True)
    y = {arm: len(ROWS) - 1 - i for i, (arm, _) in enumerate(ROWS)}
    for ax, (ds, D) in zip(axes, DATASETS):
        ax.axhspan(y['disp'] - 0.45, y['pos'] + 0.45, color='#ebebeb', lw=0, zorder=0)
        xmax = max([v[(ds, a, s)] for a, _ in ROWS for s in SEEDS] + [x for a in E4ARM.values() for x in m4[(ds, a)]])
        for arm, _ in ROWS:
            col = C_OURS if arm == 'XRE' else (C_GREY if arm == 'Native' else C_CTRL)
            x = [v[(ds, arm, s)] for s in SEEDS]
            m, sd = st.mean(x), st.stdev(x)
            ax.scatter(x, [y[arm]] * len(x), s=4, color=col, alpha=0.45, linewidths=0, zorder=2)
            ax.errorbar(m, y[arm], xerr=sd, fmt='o', ms=3, color=col, elinewidth=0.7, capsize=1.5, zorder=3)
            if rule.get((ds, arm)) == 'lower':
                ax.plot(xmax + 0.0006, y[arm], marker='v', ms=2.4, color=C_CTRL, ls='', zorder=3, clip_on=False)
        for arm, con in E4ARM.items():
            x = m4[(ds, con)]
            yy = y[arm] - 0.3
            ax.scatter(x, [yy] * len(x), s=4, color=C_OURS, alpha=0.45, linewidths=0, zorder=2)
            ax.errorbar(st.mean(x), yy, xerr=st.stdev(x) if len(x) > 1 else 0, fmt='D', ms=2.6, mfc='white', mec=C_OURS,
                        mew=0.7, color=C_OURS, elinewidth=0.7, capsize=1.5, zorder=4)
        ax.axvline(st.mean(v[(ds, 'e3', s)] for s in SEEDS), color=C_CTRL, lw=0.5, ls='--', zorder=1)
        ax.set_title(D, pad=3)
        ax.set_xlabel('Test Recall@20', labelpad=1)
        ax.grid(axis='x', lw=0.3, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(length=2, pad=1.5)
        ax.margins(x=0.1)
    axes[0].set_yticks([y[a] for a, _ in ROWS])
    axes[0].set_yticklabels([lab for _, lab in ROWS])
    axes[0].set_ylim(-0.6, len(ROWS) - 0.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig_h_gradsplit', [SRC, TESTS, E4])


if __name__ == '__main__':
    main()
