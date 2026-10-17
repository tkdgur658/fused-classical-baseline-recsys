"""Hypothesis figure: sensitivity of MSCA + XRE to the weight lambda (K = 40) and the list size K (lambda = 0.1).

Data: tables/hyp_sens.csv (tables/src/collect_hyp_sens.py; the runs of supplementary Table tab:sensitivity, one run per
seed). y = change of test Recall@20 against the MSCA mean over seeds 1-5 (%), mean over seeds with a bar of one
SD. The dashed line at 0 is the MSCA mean of every dataset (one line, since each dataset is divided by its own MSCA mean).
Relative change is plotted instead of raw Recall@20 because the three datasets differ by 0.03 in Recall@20 while the
settings differ by about 0.003, so raw values would need three y axes. Filled marker = seeds 1-5 (every setting since
2026-10-07: lambda 0.3 and K 80 seeds 4-5 ran on HOST_A qxs3/A/hA_s45). SHOW_SEED_COUNT keeps the data-driven check: a
setting with n < 5 in the CSV would get an open marker and a 'fewer seeds' legend entry. Grey column = the adopted setting (lambda 0.1, K 40). Datasets differ by marker and line style,
one color (C_OURS), as fig_recallk.py.
Usage: python -s figures/src/fig_h_sens.py
"""
import csv

from style import C_OURS, COL, TABLES, plt, save, setup

SRC = TABLES / 'hyp_sens.csv'
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific')]
STYLE = {'baby': ('o', '-'), 'pantry': ('s', '--'), 'scientific': ('^', ':')}
PANELS = [('lambda', 'lam', r'(a) Weight $\lambda$ ($K=40$)', r'Weight $\lambda$', ['0.03', '0.1', '0.3']),
          ('K', 'K', r'(b) List size $K$ ($\lambda=0.1$)', r'List size $K$', ['20', '40', '80'])]
SHOW_SEED_COUNT = True   # data-driven: open markers only for settings with fewer than five seeds (none since 2026-10-07)


def main():
    setup()
    plt.rcParams['mathtext.fontset'] = 'stix'   # Times-like math, as fig_method.py / fig_teaser.py
    with open(SRC, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.98), sharey=True)
    for ax, (group, key, title, xlab, ticks) in zip(axes, PANELS):
        ax.axvspan(0.6, 1.4, color='#ebebeb', lw=0, zorder=0)
        ax.axhline(0, color='black', lw=0.6, ls='--', zorder=1)
        for k, (ds, D) in enumerate(DATASETS):
            mk, ls = STYLE[ds]
            rr = sorted((r for r in rows if r['dataset'] == ds and r['group'] == group), key=lambda r: float(r[key]))
            xs = [i + (k - 1) * 0.12 for i in range(len(rr))]
            m = [float(r['rel_mean']) for r in rr]
            ax.plot(xs, m, color=C_OURS, ls=ls, lw=0.8, zorder=2)
            for x, r, y in zip(xs, rr, m):
                full = int(r['n']) == 5 or not SHOW_SEED_COUNT
                ax.errorbar(x, y, yerr=float(r['rel_sd']), color=C_OURS, marker=mk, ms=3, elinewidth=0.6, capsize=1.2,
                            mfc=C_OURS if full else 'white', mew=0.7, zorder=3)
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.5, len(ticks) - 0.5)
        ax.set_xlabel(xlab, labelpad=1)
        ax.set_title(title, pad=3)
        ax.grid(axis='y', lw=0.3, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2, pad=1.5)
    axes[0].set_ylabel('Recall@20 vs MSCA (%)')
    axes[1].text(2.45, 0.25, 'MSCA', ha='right', va='bottom', fontsize=6)
    hs = [plt.Line2D([], [], color=C_OURS, marker=STYLE[ds][0], ls=STYLE[ds][1], ms=3, lw=0.8, mew=0.7) for ds, _ in DATASETS]
    labs = [D for _, D in DATASETS]
    if SHOW_SEED_COUNT and any(int(r['n']) < 5 for r in rows):
        hs.append(plt.Line2D([], [], color=C_OURS, marker='o', ls='', ms=3, mfc='white', mew=0.7))
        labs.append('fewer seeds')
    fig.legend(hs, labs, loc='upper center', ncol=len(hs), frameon=False, handlelength=1.8,
               columnspacing=1.0, bbox_to_anchor=(0.54, 1.1))
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig_h_sens', [SRC])


if __name__ == '__main__':
    main()
