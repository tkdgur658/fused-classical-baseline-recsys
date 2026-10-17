"""Figure: gradient-direction split of the boundary exchange (Pantry, Scientific), test Recall@20 over seeds 1-5.

Data: tables/gradsplit_perseed.csv (tables/src/collect_analysis.py). Arms (MSCA, lambda 0.1, K 40, one run per seed):
  Native     MSCA without an auxiliary term (Table I Native runs)
  A1         anchor term alone (row A1 of tables/xre_perseed.csv)
  e3         E3 = the exchange term computed on the full graph plus the anchor term (base of the split)
  pos        E3 whose entry pair passes only the gradient that raises the held-out positive (stop-gradient on s_uD)
  disp       E3 whose entry pair passes only the gradient that lowers the displaced item D_u (stop-gradient on s_up)
  bdrand     disp with D_u replaced by a uniform random non-positive item of the list
  bdnoanc    disp without the anchor term
  XRE        the proposed objective (masked-view exchange plus anchor term, row XRE of tables/xre_perseed.csv)
Marker = mean over seeds 1-5, horizontal bar = sample SD over the seeds, small dots = seeds. Dashed line = E3 mean.
Also writes tables/gradsplit_cells.csv: per dataset and arm the mean and SD of Recall@20 / NDCG@20 and the same-seed
paired difference of Recall@20 to E3 and to XRE (mean, seeds higher, seeds lower, two-sided paired t-test p).
Colors as fig_ablation.py: XRE = C_OURS, Native = C_GREY, every other arm = C_CTRL.
Usage: python -s figures/src/fig_gradsplit.py
"""
import csv
import statistics as st

from scipy import stats

from style import C_CTRL, C_GREY, C_OURS, COL, TABLES, plt, save, setup

SRC = TABLES / 'gradsplit_perseed.csv'
DATASETS = [('pantry', 'Pantry'), ('scientific', 'Scientific')]
ROWS = [('XRE', 'XRE'), ('e3', 'Full-graph XRE'), ('pos', 'Raise positive only'),
        ('disp', 'Lower boundary only'), ('bdrand', 'Lower random list item'), ('bdnoanc', 'Lower boundary, no anchor'),
        ('A1', 'Anchor term alone'), ('Native', 'Native')]
SEEDS = [1, 2, 3, 4, 5]


def load():
    v = {}
    with open(SRC, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            v[(r['dataset'], r['arm'], int(r['seed']))] = (float(r['r20']), float(r['n20']))
    return v


def paired(v, ds, a, b):
    ss = [s for s in SEEDS if (ds, a, s) in v and (ds, b, s) in v]
    if len(ss) < 2:
        return None
    x = [v[(ds, a, s)][0] for s in ss]
    y = [v[(ds, b, s)][0] for s in ss]
    d = [p - q for p, q in zip(x, y)]
    p = float(stats.ttest_rel(x, y).pvalue) if any(abs(e) > 1e-12 for e in d) else 1.0
    m = st.mean(d)
    return dict(n=len(ss), d=m if abs(m) > 5e-9 else 0.0, hi=sum(e > 1e-12 for e in d), lo=sum(e < -1e-12 for e in d), p=p)


def cells(v):
    out = []
    for ds, _ in DATASETS:
        for arm, lab in ROWS:
            r = [v[(ds, arm, s)] for s in SEEDS if (ds, arm, s) in v]
            if not r:
                continue
            row = dict(dataset=ds, arm=arm, label=lab, n=len(r), r20_mean=f'{st.mean(x[0] for x in r):.6f}',
                       r20_sd=f'{st.stdev(x[0] for x in r):.6f}' if len(r) > 1 else '',
                       n20_mean=f'{st.mean(x[1] for x in r):.6f}', n20_sd=f'{st.stdev(x[1] for x in r):.6f}' if len(r) > 1 else '')
            for ref, tag in (('e3', 'e3'), ('XRE', 'xre')):
                c = paired(v, ds, arm, ref) if arm != ref else None
                row.update({f'd_{tag}': f"{c['d']:.6f}" if c else '', f'hi_{tag}': c['hi'] if c else '',
                            f'lo_{tag}': c['lo'] if c else '', f'p_{tag}': f"{c['p']:.4f}" if c else ''})
            out.append(row)
    return out


def main():
    setup()
    v = load()
    rows = cells(v)
    with open(TABLES / 'gradsplit_cells.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75), sharey=True)
    y = {arm: len(ROWS) - 1 - i for i, (arm, _) in enumerate(ROWS)}
    for ax, (ds, D) in zip(axes, DATASETS):
        cell = {r['arm']: r for r in rows if r['dataset'] == ds}
        for arm, _ in ROWS:
            if arm not in cell:
                continue
            col = C_OURS if arm == 'XRE' else (C_GREY if arm == 'Native' else C_CTRL)
            seeds = [v[(ds, arm, s)][0] for s in SEEDS if (ds, arm, s) in v]
            ax.scatter(seeds, [y[arm]] * len(seeds), s=4, color=col, alpha=0.45, linewidths=0, zorder=2)
            ax.errorbar(float(cell[arm]['r20_mean']), y[arm], xerr=float(cell[arm]['r20_sd']), fmt='o', ms=3, color=col,
                        elinewidth=0.7, capsize=1.5, zorder=3)
        ax.axvline(float(cell['e3']['r20_mean']), color=C_CTRL, lw=0.5, ls='--', zorder=0)
        ax.axhline(y['bdrand'] + 0.5, color='#cccccc', lw=0.4, zorder=0)
        ax.set_title(D, pad=3)
        ax.set_xlabel('Test Recall@20')
        ax.grid(axis='x', lw=0.3, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=6)
        ax.margins(x=0.08)
    axes[0].set_yticks([y[a] for a, _ in ROWS])
    axes[0].set_yticklabels([lab for _, lab in ROWS])
    axes[0].set_ylim(-0.6, len(ROWS) - 0.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig_gradsplit', [SRC])


if __name__ == '__main__':
    main()
