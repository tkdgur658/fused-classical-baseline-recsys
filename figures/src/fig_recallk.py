"""Figure: relative change of test Recall@K of XRE against Native on MSCA, K = 5, 10, 20, 50.

Data: tables/recallk_listana.csv (tables/src/collect_recallk.py: result.json best_test of the Native and XRE runs, seeds 1-5,
test split, full ranking; Baby / Pantry / Scientific = P04_listana reruns, Sports = the Table I runs of P04_sports).
Per dataset, K and seed: rel = 100 * (Recall@K of XRE - Recall@K of Native) / Recall@K of Native, same seed.
Marker and line = mean of rel over seeds 1-5, error bar = range (min to max) of rel over the five seed pairs.
Also writes tables/analysis_recallk_rel.csv (dataset, k, n, rel_mean, rel_min, rel_max, n_pos = seed pairs with rel > 0).
Color: XRE = C_OURS (style.py); datasets differ by marker and line style, as in fig_selfmsg.py.
Usage: python -s figures/src/fig_recallk.py
"""
import csv
import statistics as st

from style import C_OURS, COL, TABLES, plt, save, setup

SRC = TABLES / 'recallk_listana.csv'
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
STYLE = {'baby': ('o', '-'), 'pantry': ('s', '--'), 'scientific': ('^', ':'), 'sports': ('D', (0, (3, 1, 1, 1)))}
KS = [5, 10, 20, 50]


def rel_rows():
    v = {}
    with open(SRC, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            v[(r['dataset'], r['arm'], int(r['seed']), int(r['k']))] = float(r['recall'])
    out = []
    for ds, _ in DATASETS:
        for k in KS:
            seeds = sorted(s for (d, a, s, kk) in v if d == ds and a == 'Native' and kk == k and (ds, 'XRE', s, k) in v)
            rel = [100 * (v[(ds, 'XRE', s, k)] - v[(ds, 'Native', s, k)]) / v[(ds, 'Native', s, k)] for s in seeds]
            out.append(dict(dataset=ds, k=k, n=len(rel), rel_mean=f'{st.mean(rel):.4f}', rel_min=f'{min(rel):.4f}',
                            rel_max=f'{max(rel):.4f}', n_pos=sum(x > 0 for x in rel)))
    return out


def main():
    setup()
    rows = rel_rows()
    with open(TABLES / 'analysis_recallk_rel.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(COL, 1.5))
    for j, (ds, D) in enumerate(DATASETS):
        mk, ls = STYLE[ds]
        r = [x for x in rows if x['dataset'] == ds]
        xs = [i + (j - (len(DATASETS) - 1) / 2) * 0.09 for i in range(len(KS))]
        m = [float(x['rel_mean']) for x in r]
        lo = [a - float(x['rel_min']) for a, x in zip(m, r)]
        hi = [float(x['rel_max']) - a for a, x in zip(m, r)]
        ax.errorbar(xs, m, yerr=[lo, hi], color=C_OURS, marker=mk, ms=3, ls=ls, lw=0.8, elinewidth=0.6, capsize=1.2,
                    mfc='white', mew=0.7)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel('Cutoff $K$')
    ax.set_ylabel('Recall@$K$ vs MSCA (%)')
    ax.set_xlim(-0.35, len(KS) - 0.65)
    ax.grid(axis='y', lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    hs = [plt.Line2D([], [], color=C_OURS, marker=STYLE[ds][0], ls=STYLE[ds][1], ms=3, lw=0.8, mfc='white', mew=0.7)
          for ds, _ in DATASETS]
    ax.legend(hs, [D for _, D in DATASETS], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(DATASETS), frameon=False,
              handlelength=2.2, columnspacing=1.2, borderaxespad=0.1)
    fig.tight_layout()
    save(fig, 'fig_recallk', [SRC])


if __name__ == '__main__':
    main()
