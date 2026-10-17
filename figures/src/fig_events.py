"""Figure: exchange events during training on MSCA, exchange term alone (no anchor term) on the masked view (row wo_A1)
versus the same exchange term on the full view (row wo_A1_masked). This is the pair of the masked-view contrast of the
ablation figure (contribution (i)), so the anchor term is absent from both curves.

Input: tables/xre_events.tsv (tables/src/collect_xre.py): the monitor line that every exchange run prints every 500
training steps, computed on the batch of that step for the rows whose positive belongs to the held-out fold.
  p_in_list        fraction of those rows whose held-out positive p is already in the view's top-K list (no entry event; K of the adopted setting)
  mass_entry / mass_retention   summed sigmoid(gap) coefficients of the entry and retention terms (detached)
For XRE the view is the alternating-fold masked view. For the w/o-masked-view variant the events are formed on the full view
(the monitor reports the full-view events before the per-batch mass rescaling).
x axis: epoch = step / ceil(train interactions / 256). Each curve is the mean over the runs with a stored stdout.log (seeds
1-5, all replicates; the eur copies q12r/E were synced without stdout.log, so 8-9 of the 10 runs per cell enter) of the
values logged within each epoch; an epoch is drawn while at least 3 runs are still training.
Also writes tables/events_summary.csv (per dataset and view: mean p_in_list and entry share of the mass over the first
logged epoch and over the last 5 drawn epochs).
Usage: python -s figures/src/fig_events.py
"""
import csv
import math
from collections import defaultdict

import sys

from style import C_CTRL, C_OURS, COL, TABLES, plt, save, setup

sys.path.insert(0, str(TABLES / 'src'))
from collect_xre import setting_id  # noqa: E402

DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific')]
VIEWS = [('mask', 'masked view', C_OURS), ('full', 'full view', C_CTRL)]
VIEW_OF_ROW = {'wo_A1': 'mask', 'wo_A1_masked': 'full'}   # rows of xre_events.tsv (exchange term without the anchor term)
MIN_RUNS = 3


def main():
    setup()
    by = defaultdict(lambda: defaultdict(list))   # (ds, arm) -> epoch -> [(run, pin, share)]
    with open(TABLES / 'xre_events.tsv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['backbone'] != 'MSCA' or r['row'] not in VIEW_OF_ROW:
                continue
            if r['setting'] != setting_id():
                raise SystemExit('xre_events.tsv was collected under another XRE_SETTING: re-run collect_xre.py')
            arm = VIEW_OF_ROW[r['row']]
            ep = int(r['step']) // math.ceil(int(r['train_interactions']) / 256)
            me, mr = float(r['mass_entry']), float(r['mass_retention'])
            by[(r['dataset'], arm)][ep].append((r['run'], float(r['p_in_list']), me / (me + mr) if me + mr > 0 else 0))
    fig, axes = plt.subplots(2, 3, figsize=(COL, 2.3), sharex='col')
    summ = []
    for j, (ds, name) in enumerate(DATASETS):
        for arm, lab, col in VIEWS:
            eps = sorted(e for e, v in by[(ds, arm)].items() if len({x[0] for x in v}) >= MIN_RUNS)
            if not eps:
                continue
            pin = [sum(x[1] for x in by[(ds, arm)][e]) / len(by[(ds, arm)][e]) for e in eps]
            sh = [sum(x[2] for x in by[(ds, arm)][e]) / len(by[(ds, arm)][e]) for e in eps]
            axes[0, j].plot(eps, pin, color=col, label=lab)
            axes[1, j].plot(eps, sh, color=col, label=lab)
            summ.append(dict(dataset=ds, view=arm, first_epoch=eps[0], last_epochs=f'{eps[-5]}-{eps[-1]}',
                             p_in_list_first=f'{pin[0]:.4f}', p_in_list_last5=f'{sum(pin[-5:]) / 5:.4f}',
                             entry_share_first=f'{sh[0]:.4f}', entry_share_last5=f'{sum(sh[-5:]) / 5:.4f}',
                             runs=len({x[0] for e in eps for x in by[(ds, arm)][e]})))
        axes[0, j].set_title(name)
        axes[0, j].set_ylim(0, 1)
        axes[1, j].set_ylim(0, 1)
        axes[1, j].set_xlabel('Epoch')
        for a in axes[:, j]:
            a.grid(lw=0.3, alpha=0.5)
        if j:
            axes[0, j].set_yticklabels([])
            axes[1, j].set_yticklabels([])
    axes[0, 0].set_ylabel('Positive listed')
    axes[1, 0].set_ylabel('Entry mass share')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, handlelength=1.5, bbox_to_anchor=(0.55, 1.05))
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    save(fig, 'fig_events', [TABLES / 'xre_events.tsv'])
    with open(TABLES / 'events_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(summ)


if __name__ == '__main__':
    main()
