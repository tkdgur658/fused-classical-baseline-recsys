"""Hypothesis figure: exchange events during training, masked view versus full view (Lemma lem:masked).

Revival of fig_events.py under the one-run-per-seed rule: seeds 1-5, replicate 1 only (no replicates are averaged).
Data: tables/hyp_events_epochs.csv and tables/hyp_events_summary.csv (tables/src/collect_hyp_events.py; MSCA, lambda 0.1,
K 40, exchange term alone without the anchor term, masked view = row wo_A1, full view = row wo_A1_masked of tab:xmask).
  top row     share of held-out positives already inside the top-K list of the view (no entry event), %
  bottom row  entry share of the exchange coefficient mass, mass_entry / (mass_entry + mass_retention), %
Line = mean over the runs still training (drawn while >= 3 of the 5 runs train), band = min-max over those runs,
open circle = the selected (best-validation) epoch, mean over seeds, placed on the mean curve.
Columns: Baby, Pantry, Scientific, plus Sports when collect_hyp_events.py found all ten Sports runs finished.
Colors: masked view = C_OURS, full view = C_CTRL (style.py).
Usage: python -s figures/src/fig_h_events.py
"""
import csv
from collections import defaultdict

from style import C_CTRL, C_OURS, COL, TABLES, plt, save, setup

SRC = TABLES / 'hyp_events_epochs.csv'
SUMM = TABLES / 'hyp_events_summary.csv'
NAMES = {'baby': 'Baby', 'pantry': 'Pantry', 'scientific': 'Scientific', 'sports': 'Sports'}
VIEWS = [('mask', 'masked view', C_OURS), ('full', 'full view', C_CTRL)]
PAIR = 'exonly'


def main():
    setup()
    plt.rcParams['mathtext.fontset'] = 'stix'   # Times-like math, as fig_method.py / fig_teaser.py
    cur = defaultdict(list)
    with open(SRC, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['pair'] == PAIR:
                cur[(r['dataset'], r['view'])].append(r)
    with open(SUMM, newline='', encoding='utf-8') as f:
        best = {(r['dataset'], r['view']): float(r['best_epoch_mean']) for r in csv.DictReader(f)
                if r['pair'] == PAIR and r['view'] != 'full-mask'}
    dss = [d for d in NAMES if (d, 'mask') in cur]
    fig, axes = plt.subplots(2, len(dss), figsize=(COL, 1.82), sharey='row')
    for j, ds in enumerate(dss):
        for view, lab, col in VIEWS:
            rr = sorted(cur[(ds, view)], key=lambda r: int(r['epoch']))
            ep = [int(r['epoch']) for r in rr]
            for i, q in enumerate(('listed', 'entry')):
                ax = axes[i, j]
                m = [100 * float(r[f'{q}_mean']) for r in rr]
                ax.fill_between(ep, [100 * float(r[f'{q}_min']) for r in rr], [100 * float(r[f'{q}_max']) for r in rr],
                                color=col, alpha=0.18, lw=0)
                ax.plot(ep, m, color=col, lw=0.9, label=lab)
                b = best[(ds, view)]
                k = min(range(len(ep)), key=lambda t: abs(ep[t] - b))
                ax.plot(ep[k], m[k], 'o', ms=3.2, mfc='white', mec=col, mew=0.8, zorder=4)
        axes[0, j].set_title(NAMES[ds], pad=2)
        axes[1, j].set_xlabel('Epoch', labelpad=1)
        for ax in axes[:, j]:
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 50, 100])
            ax.grid(lw=0.3, alpha=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(length=2, pad=1.5)
        axes[0, j].tick_params(labelbottom=False)
    axes[0, 0].set_ylabel('Positive listed (%)', fontsize=6.5)
    axes[1, 0].set_ylabel('Entry share (%)', fontsize=6.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls='', marker='o', ms=3.2, mfc='white', mec='#555555', mew=0.8))
    l.append('selected epoch')
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, handlelength=1.5, columnspacing=1.2,
               bbox_to_anchor=(0.54, 1.075))
    fig.tight_layout(h_pad=0.25, w_pad=0.35)
    save(fig, 'fig_h_events', [SRC, SUMM])


if __name__ == '__main__':
    main()
