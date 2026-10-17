"""Figure: the self-message of a training positive and the BPR positive gradient it removes (Proposition prop:atten).

2026-10-08 (method change XRE -> MES): MSCA only; the MSCA + XRE arm is dropped because MES has no checkpoint runs
yet (old script: figures/src/_old/fig_selfmsg_pre1009.py). The xre rows of the CSV are no longer read.
2026-10-10: MES restored in (b) as in the original design (MSCA vs the compared method, seed-paired): arm 'mes' of the CSV
= MAIN-table MES checkpoints (tables/src/collect_analysis.py --selfmsg-only), same training pairs and negatives as MSCA.
(a) stays MSCA only, as in the original design. A dataset without MES rows shows MSCA only (interim, Sports seeds).
Data: tables/analysis_selfmsg.csv (tables/src/collect_analysis.py; MSCA Native checkpoints of P04_listana for
Baby / Pantry / Scientific and the Table I runs of P04_sports for Sports, seeds 1-5).
  (a) mean share S_up / s_up of the self-message in the training score of a positive, by the training history
      length of the user (bins <=3, 4-7, 8-15, >=16). Marker = mean over seeds 1-5, error bar = range over the seeds.
  (b) kept BPR weight E[w_full] / E[w_cav], with w = sigma(s_un - s_up) for the full score and sigma(s_un - c_up)
      for the score without the self-message, MSCA (left bar) and MES (right bar) per dataset. Dots = seeds 1-5, a thin
      grey line joins the two arms of one seed, bar = mean over the seeds. 1 - ratio is the share of
      the positive gradient that the self-message removes.
Colors: MSCA = C_CTRL family (manuscript palette, style.py); datasets in (a) differ by marker, line style, and shade.
Math uses the stix font set, as fig_h_sens.py. All text >= 6.5 pt at the printed width (3.5 in).
Usage: python -s figures/src/fig_selfmsg.py
"""
import csv
import statistics as st

from style import C_CTRL, C_OURS, COL, TABLES, plt, save, setup

SRC = TABLES / 'analysis_selfmsg.csv'
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
BINS = [('le3', r'$\leq$3'), ('4to7', '4–7'), ('8to15', '8–15'), ('ge16', r'$\geq$16')]
STYLE = {'baby': ('o', '-'), 'pantry': ('s', '--'), 'scientific': ('^', ':'), 'sports': ('D', (0, (3, 1, 1, 1)))}
SHADE = {'baby': '#5c1714', 'pantry': '#c0504d', 'scientific': '#e6a19e', 'sports': '#8f2f2b'}   # shades of C_CTRL
BAR_FC = '#f4dad9'   # light fill of the mean bars (C_CTRL family)
BAR_FC_OURS = '#d6e0ef'   # light fill of the MES mean bars (C_OURS family)
ARMS = [('native', 'MSCA', C_CTRL, BAR_FC), ('mes', 'MES', C_OURS, BAR_FC_OURS)]


def load():
    with open(SRC, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    setup()
    plt.rcParams['mathtext.fontset'] = 'stix'   # Times-like math, as fig_h_sens.py
    rows = load()
    get = lambda ds, arm, b, key: [float(r[key]) for r in rows if r['dataset'] == ds and r['arm'] == arm and r['bin'] == b]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL + 0.19, 1.45), gridspec_kw=dict(width_ratios=[1.0, 1.15]))

    # (a) self-message share by history length, MSCA
    for k, (ds, D) in enumerate(DATASETS):
        mk, ls = STYLE[ds]
        xs = [i + (k - (len(DATASETS) - 1) / 2) * 0.11 for i in range(len(BINS))]
        v = [[100 * x for x in get(ds, 'native', b, 'share_mean')] for b, _ in BINS]
        m = [st.mean(x) for x in v]
        ax1.errorbar(xs, m, yerr=[[a - min(x) for a, x in zip(m, v)], [max(x) - a for a, x in zip(m, v)]], color=SHADE[ds],
                     marker=mk, ms=3, ls=ls, lw=0.8, elinewidth=0.6, capsize=1.2, mfc='white', mew=0.7, label=D)
    ax1.set_xticks(range(len(BINS)))
    ax1.set_xticklabels([lab for _, lab in BINS])
    ax1.set_xlabel('Training history length')
    ax1.set_ylabel('Share of score (%)')
    ax1.set_ylim(0, 112)   # headroom for the legend above the lines
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_title('(a) Self-message share', pad=3)
    hs = [plt.Line2D([], [], color=SHADE[ds], marker=STYLE[ds][0], ls=STYLE[ds][1], ms=3, lw=0.8, mfc='white', mew=0.7)
          for ds, _ in DATASETS]
    ax1.legend(hs, [D for _, D in DATASETS], loc='upper right', ncol=2, frameon=False, handlelength=1.5, borderaxespad=0.1, handletextpad=0.4,
               labelspacing=0.2, columnspacing=0.6, fontsize=6.5)

    # (b) kept BPR weight, MSCA vs MES, seed-paired: seeds as dots, mean as bar
    bw = 0.34
    for j, (ds, D) in enumerate(DATASETS):
        pts = {}
        for a, (arm, lab, col, fc) in enumerate(ARMS):
            v = get(ds, arm, 'all', 'wratio')
            if not v:
                continue
            x0 = j + (a - 0.5) * bw
            ax2.bar(x0, st.mean(v), width=bw * 0.92, color=fc, edgecolor=col, lw=0.6, zorder=2,
                    label=lab if j == 0 else None)
            xs = [x0 + (s - (len(v) - 1) / 2) * 0.045 for s in range(len(v))]
            ax2.scatter(xs, v, s=5, color=col, zorder=4, linewidths=0)
            pts[arm] = (xs, v)
        if len(pts) == 2:
            (xa, va), (xb, vb) = pts['native'], pts['mes']
            for p, q, r_, t in zip(xa, va, xb, vb):
                ax2.plot([p, r_], [q, t], color='#9a9a9a', lw=0.35, zorder=3)
    ax2.set_xticks(range(len(DATASETS)))
    ax2.set_xticklabels([D for _, D in DATASETS])
    ax2.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax2.set_ylim(0, 0.75)
    ax2.set_yticks([0, 0.2, 0.4, 0.6])
    ax2.set_ylabel(r'$\mathbb{E}[w]\,/\,\mathbb{E}[\bar{w}]$')   # w = w(u,p,n), bar w without the self-message
    ax2.set_title('(b) Kept BPR weight', pad=3)
    ax2.legend(loc='upper right', ncol=2, frameon=False, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1,
               columnspacing=0.6, fontsize=6.5)
    for ax in (ax1, ax2):
        ax.grid(axis='y', lw=0.3, alpha=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig_selfmsg', [SRC])


if __name__ == '__main__':
    main()
