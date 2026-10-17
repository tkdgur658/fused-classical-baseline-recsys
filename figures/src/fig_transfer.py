"""Figure: XRE on the other graph backbones (MENTOR, FREEDOM, MMGCN), relative change of test Recall@20 against Native.

Per backbone (one panel, newest backbone first) and dataset (group), two bars:
  best plug-in  the plug-in objective with the highest mean Recall@20 in that backbone-dataset cell among the five plug-ins
                of Table I (tables/table1_cells.csv, collect_table1.py); plug-ins were run on FREEDOM and MMGCN only, so
                the MENTOR panel has the XRE bar alone
  XRE           row 'XRE' of tables/xre_cells.csv (collect_xre.py, adopted setting)
Native: FREEDOM / MMGCN = the Native row of Table-I cells (tables/src/native_rows.py), MENTOR = the MENTOR row of the
comparison table (tables/table2_cells.csv), i.e. the same Native values as the appendix transfer table (tab:transfer),
which prints every number of this figure. Relative change = 100 * (mean - Native mean) / Native mean on the unrounded
means over seeds 1-5. Also writes tables/transfer_summary.csv (one row per backbone and dataset).
Colors: the manuscript palette of style.py (XRE = C_OURS, the reference bar = C_GREY); the table view is tab:transfer.
Usage: python -s figures/src/fig_transfer.py
"""
import csv
import sys

from style import C_GREY, C_OURS, COL, TABLES, plt, save, setup

sys.path.insert(0, str(TABLES / 'src'))
import native_rows  # noqa: E402
from collect_xre import check_setting  # noqa: E402

DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific')]
BACKBONES = ['MENTOR', 'FREEDOM', 'MMGCN']   # newest backbone first
PLUGINS = [('INA', 'INA-style'), ('MDVT', 'MDVT'), ('MG', 'MG'), ('DA-MRS', 'DA-MRS'),
           ('UltraGCN-LI', 'UltraGCN L_I')]
N = 5


def load():
    with open(TABLES / 'xre_cells.csv', newline='', encoding='utf-8') as f:
        xre = {(r['backbone'], r['dataset']): r for r in csv.DictReader(f)
               if r['row'] == 'XRE' and r['seedset'] == '12345' and int(r['n']) == N}
    with open(TABLES / 'table2_cells.csv', newline='', encoding='utf-8') as f:
        t2 = {(r['model'], r['dataset']): r for r in csv.DictReader(f) if int(r['n']) == N}
    t1 = native_rows.cells()
    out = []
    for bb in BACKBONES:
        for d, D in DATASETS:
            nat = t2.get(('MENTOR', d)) if bb == 'MENTOR' else native_rows.native(bb, d)
            x = xre.get((bb, d))
            if not nat or not x:
                continue
            base = float(nat['r20_mean'])
            plugs = [(float(c['r20_mean']), lab) for k, lab in PLUGINS
                     for c in [native_rows.complete(t1.get((bb, k, d)))] if c]
            bp = max(plugs) if plugs else None
            out.append(dict(backbone=bb, dataset=d, native_r20=f'{base:.6f}', xre_r20=f"{float(x['r20_mean']):.6f}",
                            xre_rel=f"{100 * (float(x['r20_mean']) - base) / base:.2f}",
                            best_plugin=bp[1] if bp else '', best_plugin_r20=f'{bp[0]:.6f}' if bp else '',
                            best_plugin_rel=f'{100 * (bp[0] - base) / base:.2f}' if bp else ''))
    return out


def main():
    check_setting()
    setup()
    rows = load()
    fig, axes = plt.subplots(1, 3, figsize=(COL, 1.3), sharey=True)
    w = 0.36
    for ax, bb in zip(axes, BACKBONES):
        mine = {r['dataset']: r for r in rows if r['backbone'] == bb}
        for i, (d, _) in enumerate(DATASETS):
            r = mine.get(d)
            if not r:
                continue
            if r['best_plugin_rel']:
                ax.bar(i - w / 2 - 0.02, float(r['best_plugin_rel']), w, color=C_GREY, label='best plug-in' if not i else None)
                ax.bar(i + w / 2 + 0.02, float(r['xre_rel']), w, color=C_OURS, label='XRE' if not i else None)
            else:
                ax.bar(i, float(r['xre_rel']), w, color=C_OURS)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(['Baby', 'Pantry', 'Sci.'], fontsize=6)
        ax.set_title(bb)
        ax.grid(axis='y', lw=0.3, alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('R@20 vs Native (%)')
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=2, frameon=False, handlelength=1.2, bbox_to_anchor=(0.55, 1.08))
    fig.tight_layout(w_pad=0.3)
    save(fig, 'fig_transfer', [TABLES / 'xre_cells.csv', TABLES / 'table1_cells.csv', TABLES / 'table2_cells.csv'])
    with open(TABLES / 'transfer_summary.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        wr.writeheader()
        wr.writerows(rows)


if __name__ == '__main__':
    main()
