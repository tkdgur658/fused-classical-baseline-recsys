"""Figure: component ablation of XRE on MSCA (test Recall@20, seeds 1-5), and the Holm-corrected contrasts behind it.

Since 2026-10-06 (review plan B1) the figure is no longer input in the manuscript: its rows are merged into Table III
(tables/src/build_table_xmask.py), which prints the marks of the figure family below from tables/ablation_tests.csv, and
the appendix family marks Table tab:ablation_full (tables/src/build_table_xre_appendix.py). The PDF stays in the repository.
The tests cover every dataset on which all six variants and Native have seeds 1-5 (Sports once its ablation runs are
collected); the figure panels stay Baby / Pantry / Scientific.

Rows (top to bottom): XRE, w/o exchange term (anchor term alone, row A1), w/o anchor term (exchange term alone, row
wo_A1), w/o anchor term on the full view (row wo_A1_masked), Native (MSCA without an auxiliary term).
Inputs: tables/xre_cells.csv and tables/xre_perseed.csv (tables/src/collect_xre.py), tables/table1_cells.csv and
tables/table1_perseed.csv (Native = the MSCA Native row of Tables I and II, via tables/src/native_rows.py). Each dot is the
mean over seeds 1-5 of the per-seed mean (all eligible runs of a seed averaged first); bars are the SD over seeds.
Contrasts (a minus b, Recall@20), two-sided t-tests over seeds, paired by seed for two variants (all variants of one seed
start from identical initial parameters, identical first-batch losses at lambda = 0 in the package unit test), Welch for
Native (a different run set). wins = seeds with a > b (per-seed means).
  figure family (Holm over the three contrasts of a dataset; the marked row is b, or a for the Native contrast):
    exchange   XRE - w/o exchange term            (the exchange term added to the anchor term)
    masked     w/o anchor term - w/o anchor, full (masked versus full view, both without the anchor term)
    native     w/o anchor term - Native           (the exchange term alone versus the backbone)
  appendix family (Holm over its three contrasts; appendix table tab:ablation_full, no marker in the figure):
    anchor     XRE - w/o anchor term
    retention  XRE - w/o retention
    fullxre    XRE - XRE on full view
Writes tables/ablation_tests.csv (one row per contrast and dataset) and tables/ablation_rows.csv (mean and SD per row).
Usage: python -s figures/src/fig_ablation.py
"""
import csv
import sys

from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy import stats

from style import C_CTRL, C_GREY, C_OURS, COL, TABLES, plt, save, setup

sys.path.insert(0, str(TABLES / 'src'))
import native_rows  # noqa: E402
from collect_xre import check_setting  # noqa: E402

DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific')]
TEST_DATASETS = DATASETS + [('sports', 'Sports')]
VARIANTS = ['XRE', 'A1', 'wo_A1', 'wo_A1_masked', 'wo_retention', 'wo_masked', 'MSCA']
ROWS = [('XRE', 'MSCA + XRE'), ('A1', 'w/o exchange term'), ('wo_A1', 'w/o anchor term'),   # labels: 2026-10-06
        ('wo_A1_masked', 'w/o anchor term, full view'), ('MSCA', 'MSCA')]
# (name, a, b, family, marked row)
CONTRASTS = [('exchange', 'XRE', 'A1', 'figure', 'A1'), ('masked', 'wo_A1', 'wo_A1_masked', 'figure', 'wo_A1_masked'),
             ('native', 'wo_A1', 'MSCA', 'figure', 'MSCA'), ('anchor', 'XRE', 'wo_A1', 'appendix', None),
             ('retention', 'XRE', 'wo_retention', 'appendix', None), ('fullxre', 'XRE', 'wo_masked', 'appendix', None)]


def holm(p):
    order = sorted(range(len(p)), key=lambda i: p[i])
    rej = set()
    for k, i in enumerate(order):
        if p[i] <= 0.05 / (len(p) - k):
            rej.add(i)
        else:
            break
    return rej


def load():
    cell, per = {}, {}
    with open(TABLES / 'xre_cells.csv', newline='') as f:
        for r in csv.DictReader(f):
            if r['backbone'] == 'MSCA' and r['seedset'] == '12345':
                cell[(r['row'], r['dataset'])] = (float(r['r20_mean']), float(r['r20_std'] or 0), r['seeds'])
    with open(TABLES / 'xre_perseed.csv', newline='') as f:
        for r in csv.DictReader(f):
            if r['backbone'] == 'MSCA':
                per.setdefault((r['row'], r['dataset']), {})[int(r['seed'])] = float(r['r20'])
    # MSCA row = the MSCA Native runs of Tables I and II (tables/src/native_rows.py)
    nat_per = native_rows.perseed('MSCA')
    for d, _ in TEST_DATASETS:
        c = native_rows.native('MSCA', d)
        if c:
            cell[('MSCA', d)] = (float(c['r20_mean']), float(c['r20_std']), c['seeds'])
            per[('MSCA', d)] = nat_per[(d, 'r20')]
    return cell, per


def test(per, a, b, ds):
    x, y = per[(a, ds)], per[(b, ds)]
    s = sorted(set(x) & set(y))
    if 'MSCA' in (a, b):
        p = stats.ttest_ind([x[v] for v in x], [y[v] for v in y], equal_var=False).pvalue
    else:
        p = stats.ttest_rel([x[v] for v in s], [y[v] for v in s]).pvalue
    return p, sum(x[v] > y[v] for v in s), len(s)


def main():
    check_setting()
    setup()
    cell, per = load()
    tests = []
    for ds, _ in TEST_DATASETS:
        if not all((k, ds) in cell for k in VARIANTS):
            continue
        for fam in ('figure', 'appendix'):
            cs = [c for c in CONTRASTS if c[3] == fam and (c[1], ds) in cell and (c[2], ds) in cell]
            res = [test(per, a, b, ds) for _, a, b, _, _ in cs]
            rej = holm([r[0] for r in res])
            for n, ((name, a, b, _, mark), (p, wins, ns)) in enumerate(zip(cs, res)):
                tests.append(dict(dataset=ds, contrast=name, a=a, b=b, family=fam, marked=mark or '',
                                  diff=f"{cell[(a, ds)][0] - cell[(b, ds)][0]:+.6f}",
                                  diff_printed=f"{round(cell[(a, ds)][0], 4) - round(cell[(b, ds)][0], 4):+.4f}",
                                  wins=wins, n_seeds=ns, p=f'{p:.4g}', holm_rejected=int(n in rej)))
    marked = {(r['dataset'], r['marked']) for r in tests if r['family'] == 'figure' and r['holm_rejected']}
    rows_all = [(k, lab) for k, lab in ROWS if any((k, d) in cell for d, _ in DATASETS)]
    fig, axes = plt.subplots(1, 3, figsize=(COL, 1.45), sharey=True)
    for ax, (ds, name) in zip(axes, DATASETS):
        for k, lab in rows_all:
            if (k, ds) not in cell:
                continue
            m, sd, _ = cell[(k, ds)]
            y = len(rows_all) - 1 - [r[0] for r in rows_all].index(k)
            col = C_OURS if k == 'XRE' else (C_GREY if k == 'MSCA' else C_CTRL)
            ax.errorbar(m, y, xerr=sd, fmt='o', ms=3, color=col, elinewidth=0.7, capsize=1.5)
            if (ds, k) in marked:
                ax.annotate('*', (m, y), xytext=(0, 1.5), textcoords='offset points', ha='center', fontsize=7)
        if ('XRE', ds) in cell:
            ax.axvline(cell[('XRE', ds)][0], color=C_OURS, lw=0.5, ls='--', zorder=0)
        ax.set_title(name)
        ax.set_ylim(-0.6, len(rows_all) - 0.4)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(axis='x', lw=0.3, alpha=0.5)
    axes[0].set_yticks(range(len(rows_all)))
    axes[0].set_yticklabels([lab for _, lab in reversed(rows_all)])
    fig.supxlabel(r'Test Recall@20 (mean $\pm$ SD over seeds 1-5)', fontsize=7, y=0.1)
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig_ablation', [TABLES / 'xre_cells.csv', TABLES / 'xre_perseed.csv', TABLES / 'table1_cells.csv',
                               TABLES / 'table1_perseed.csv'])
    with open(TABLES / 'ablation_tests.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(tests)


if __name__ == '__main__':
    main()
