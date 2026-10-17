"""Build tables/table2_main.tex (Table 2: comparison with multimodal recommenders under the common protocol) from
tables/table2_cells.csv (baselines) and tables/src/mes_source.py (the Ours row "MSCA + MES"). Numbers are never typed by hand.

Revision 2026-10-08 (MES replaces XRE): the Ours row is MSCA trained with MES, per seed from mes_source.py, i.e.
tables/mes_perseed.csv once it holds seeds 1-5 of a dataset, otherwise the PROVISIONAL stand-in xre_perseed.csv row wo_A1
(E + R, rep1). Provisional cells print as \\tbd{...} (red), the caption says so, and the bold marks and the Welch/Holm marks
against the Ours row are computed from the printed (provisional) per-seed values. There is no MENTOR + XRE row (removed
2026-10-05). The paragraphs below describe the earlier XRE sources and are kept for the baselines.

table2_cells.csv is written by tables/src/collect_table2.py (see its docstring for sources and the duplicate rule);
xre_cells.csv is written by tables/src/collect_xre.py (Ours: every eligible run of each seed averaged first, then seeds).
Each cell shows the mean test metric over seeds 1-5 and the sample standard deviation over seeds. When every cell uses the
same five seeds the per-cell seed count is omitted and stated once in the caption; otherwise it is printed in parentheses.
Within each column, the best printed mean is set in bold (with its standard deviation) and the second best is underlined.
Ranks are computed on the printed (4-decimal) means over all rows including both Ours rows, so ours is not assumed to be
best. An Ours cell with fewer than five seeds prints \\todo{pending} and takes no rank.
Markers: a baseline cell is marked with $^{\\blacktriangledown}$ when it is significantly lower than Ours (MSCA + MES) and
with $^{\\blacktriangle}$ when significantly higher. Since revision log 18-3 (single test family, tables/src/sigtest.py) the test
is the two-sided paired t-test over seeds 1-5 (runs of the same seed paired) with Holm over the four datasets of one
(model, metric) contrast; before 18-3 it was a Welch t-test with Holm over the compared models of one column. Every test
is written to tables/sigtests_table2.csv.
MENTOR + XRE: its MENTOR baseline is the MENTOR row of this table (mentor package, config MENTOR, the Table-2 runs of
collect_table2.py). xret_n <ds>_native_l000 (the lambda = 0 path of the XRE package) has the same model block as
models/mentor/configs/MENTOR.yaml (text diff) and is bitwise the mentor package at lambda = 0 (xre_t unit test: state_dict,
loss, every gradient); both run mentor.py sha256 52996653 with identical overall.yaml and dataset yaml files.
Printed values and marks of every cell are also written to tables/table2_marks.csv (read by build_macros.py).
The MSCA row is not read from table2_cells.csv: it shows the MSCA Native runs of Table II (tables/src/native_rows.py,
objx_s <ds>_native seeds 1-5 = the official XRE baseline), which have the same model configuration as the msca-package
runs of collect_table2.py, so Table I, Table II and the prose print one MSCA value. The msca-package runs stay in
table2_runs.tsv / table2_cells.csv as information.
Usage: python tables/src/build_table2_main.py
"""
import csv
import hashlib
from pathlib import Path

from scipy import stats

import native_rows
from collect_xre import check_setting, setting_id
import mes_source
import sigtest   # revision log 18-3: one test family for every significance mark

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / 'table2_cells.csv'
RUNS = ROOT / 'table2_runs.tsv'
# baselines of the datasets added on HOST_A (tables/src/collect_table2_extra.py), read next to the two files above
CELLS_EXTRA = ROOT / 'table2_cells_extra.csv'
RUNS_EXTRA = ROOT / 'table2_runs_extra.tsv'
XRE_CELLS = ROOT / 'xre_cells.csv'
XRE_SEEDS = ROOT / 'xre_perseed.csv'
OUT = ROOT / 'table2_main.tex'
# S4R draft (2026-10-14, TABLE_FUSION): training-free post-hoc fusion block (backbone + z-scored RP3beta and ItemKNN-CBF scores, weight
# chosen on validation R@20 per seed). Source: server <RUN_ROOT>/TABLE_FUSION/fusion_summary.csv (marker TABLE_FUSION_DONE; the nine-dataset
# rerun writes fusion_summary_9ds.csv), copied here as tables/src/data/fusion_summary.csv. Columns: model (MSCA+MES, MSCA, COHESION,
# SMORE, PGL, DiffMM, LGMRec, FREEDOM), dataset, n, seeds, r20_mean, r20_std, n20_mean, n20_std, base_r20_mean, base_n20_mean; one row per
# (model, dataset) over the seeds that passed the base-reproduction gate. While the file is missing, or a (model, dataset) row is absent,
# the cell is 	bd{--}. A cell with fewer than five gated seeds prints with its seed count in parentheses (red since revision log 18-D removed). The block has no significance marks.
FUSION_OLD = Path(__file__).resolve().parent / 'data' / 'fusion_summary.csv'   # four datasets, 32 pairs (fallback)
# 2026-10-16 (revision log 13): MES9 merged file (study/results/fusion_summary_9ds_mes.csv, 50 pairs on eight datasets) is preferred when present;
# its rows for the four datasets of this table are identical to FUSION_OLD (checked), so the table does not change.
FUSION_MES9 = Path(__file__).resolve().parent / 'data' / 'fusion_summary_9ds_mes.csv'
FUSION = FUSION_MES9 if FUSION_MES9.exists() else FUSION_OLD
FUSION_NAME = {'Ours': 'MSCA+MES'}   # table row key -> model name in fusion_summary.csv

# (key, printed label, backbone in xre_cells.csv); the first row is the reference of the significance markers
OURS_ROWS = [('Ours', 'MSCA + MES', 'MSCA')]   # MSCA + MES (2026-10-08); the MENTOR + XRE row is removed (user decision 2026-10-05)
OURS = OURS_ROWS[0][:2]
MARKS = ROOT / 'table2_marks.csv'
MODELS = [('MSCA', 'MSCA~\\cite{msca2026}'), ('COHESION', 'COHESION~\\cite{xu2025cohesion}'),
          ('SMORE', 'SMORE~\\cite{ong2025smore}'), ('PGL', 'PGL~\\cite{yu2025pgl}'),
          ('DiffMM', 'DiffMM~\\cite{jiang2024diffmm}'),   # MENTOR removed from the compared models (2026-10-05)
          ('LGMRec', 'LGMRec~\\cite{guo2023lgmrec}'), ('FREEDOM', 'FREEDOM~\\cite{zhou2023freedom}')]
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]   # Instruments and Clothing removed (user decision 2026-10-05)
METRICS = [('r20', 'R@20'), ('n20', 'N@20')]
PANEL = 4   # datasets per stacked panel of the table (the table never uses \resizebox)
ALPHA = 0.05
NWORD = {5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}


def holm(pvals):
    """Holm step-down: returns the set of indices rejected at ALPHA"""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    rej = set()
    for k, i in enumerate(order):
        if pvals[i] <= ALPHA / (len(pvals) - k):
            rej.add(i)
        else:
            break
    return rej


def load_fusion():
    """(model, dataset) -> row of fusion_summary.csv; empty when the server job has not delivered the file yet"""
    out = {}
    if FUSION.exists():
        with open(FUSION, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                out[(row['model'], row['dataset'])] = row
    return out


def fusion_cell(fus, m, d, k):
    """printed cell of the fusion block; tbd{--} (red) unless the file holds the (model, dataset) row"""
    c = fus.get((FUSION_NAME.get(m, m), d))
    if not (c and c.get(f'{k}_mean')):
        return '\\tbd{--}'
    if int(c['n']) == 5:
        sd = c.get(f'{k}_std')
        return f"{float(c[f'{k}_mean']):.4f}" + (f'{{\\scriptsize$\\pm${float(sd):.4f}}}'.replace('0.', '.', 1) if sd else '')
    # revision log 18-D (decision 8, SOTAFILL): a cell with fewer than five gated seeds keeps its seed count in parentheses but is no longer red
    return f"{float(c[f'{k}_mean']):.4f}" + f"\\,({c['n']})"


def mix_cell(m, d, k):
    """revision log 29 (item 1): MSCA + MES or MSCA fused with the validation mix of the four rankers (CLASSIC cells, build_rev2.py)"""
    import statistics as _st
    import build_rev2 as rv
    s = [i for i in rv.bc.SEEDS if (m, d, i) in rv.bc.cell and 'mix' in rv.bc.cell[(m, d, i)]]
    if not s:
        return '--'
    x = [rv.val((m, d, i), 'mix', k) for i in s]
    v = f'{sum(x) / len(x):.4f}'
    if len(s) == 5:
        return v + f'{{\\scriptsize$\\pm${_st.stdev(x):.4f}}}'.replace('0.', '.', 1)
    return v + f'\\,({len(s)})'


def margin_cell(variant, d, k):
    """revision log 31: MSCA + MES minus MSCA after F (combo) or after the mix over seeds 1-10 (preregistered extension PREREG_MSCA10), with
    the mark of its paired t-test with Holm correction over the four tuned datasets (tables/src/data/msca10_margins.csv, written by
    build_macros_msca10.py).  Before (revision log 29): seeds 1-5, Holm over six datasets (rev2_margins.csv, build_rev2.py)."""
    src = Path(__file__).resolve().parent / 'data' / 'msca10_margins.csv'
    fus = {'combo': 'F', 'mix': 'mix'}[variant]
    rows = {(r['fusion'], r['metric'], r['dataset']): r for r in csv.DictReader(open(src, newline='', encoding='utf-8'))}
    r = rows.get((fus, k, d))
    if not r:
        return '--'
    assert r['family'] == 'tuned' and r['window'] == '1-10' and r['n'] == '10', r
    x = float(r['diff'])
    body = f'{x:+.4f}'.replace('-', '$-$').replace('+', '$+$')
    if r['reject'] == '1':
        body += '\\rlap{$^{\\blacktriangle}$}' if x > 0 else '\\rlap{$^{\\blacktriangledown}$}'
    return body


def main():
    cells, per = {}, {}
    fus = load_fusion()
    for path in (CELLS, CELLS_EXTRA):
        if not path.exists():
            continue
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                if (r['model'], r['dataset']) in cells:
                    raise SystemExit(f"duplicate cell {r['model']} {r['dataset']} in {path.name}")
                cells[(r['model'], r['dataset'])] = dict(n=int(r['n']), seeds=r['seeds'], r20=r['r20_mean'],
                                                         n20=r['n20_mean'], r20_std=r['r20_std'], n20_std=r['n20_std'])
    for path in (RUNS, RUNS_EXTRA):
        if not path.exists():
            continue
        with open(path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                if r['chosen'] == '1':
                    for m, _ in METRICS:
                        per.setdefault((r['model'], r['dataset'], m), {})[int(r['seed'])] = float(r[f'test_{m}'])
    # MSCA row = the MSCA Native runs of Table II (one run set for every MSCA value of the manuscript)
    msca_per = native_rows.perseed('MSCA')
    for d, _ in DATASETS:
        c = native_rows.native('MSCA', d)
        cells.pop(('MSCA', d), None)
        for m, _ in METRICS:
            per.pop(('MSCA', d, m), None)
        if c:
            cells[('MSCA', d)] = dict(n=int(c['n']), seeds=c['seeds'], r20=c['r20_mean'], n20=c['n20_mean'],
                                      r20_std=c['r20_std'], n20_std=c['n20_std'])
            for m, _ in METRICS:
                per[('MSCA', d, m)] = msca_per[(d, m)]
    check_setting()
    rps = {}   # Ours key -> set of runs-per-seed counts (MES: one run per seed, asserted in mes_source.py)
    bb_key = {bb: key for key, _, bb in OURS_ROWS}
    prov = set()   # (Ours key, dataset) printed from the provisional stand-in
    for bb, key in bb_key.items():
        mper = mes_source.perseed(bb, [d for d, _ in DATASETS])
        for d, _ in DATASETS:
            c = mes_source.cell(bb, d)
            if c:
                cells[(key, d)] = dict(n=int(c['n']), seeds=c['seeds'], r20=c['r20_mean'], n20=c['n20_mean'],
                                       r20_std=c['r20_std'], n20_std=c['n20_std'])
                rps.setdefault(key, set()).add('1')
                if c['provisional']:
                    prov.add((key, d))
            for m, _ in METRICS:
                if (d, m) in mper:
                    per[(key, d, m)] = mper[(d, m)]
    ours_keys = set(bb_key.values())
    rows = [(key, label) for key, label, _ in OURS_ROWS] + MODELS
    cols = [(d, m) for d, _ in DATASETS for m, _ in METRICS]
    printed = {}
    for m, _ in rows:
        for d, k in cols:
            c = cells.get((m, d))
            ok = c and c[k] and c['n'] > 0 and (m not in ours_keys or c['n'] == 5)
            printed[(m, d, k)] = f'{float(c[k]):.4f}' if ok else None
    marks, sig = {}, {}
    for d, k in cols:
        vals = sorted({float(v) for (m, dd, kk), v in printed.items() if dd == d and kk == k and v}, reverse=True)
        for (m, dd, kk), v in printed.items():
            if dd == d and kk == k and v:
                marks[(m, d, k)] = {0: 'best', 1: 'second'}.get(vals.index(float(v)))
    # revision log 18-3 (single test family, tables/src/sigtest.py): two-sided paired t-test over seeds 1-5 (runs of the same
    # seed paired), one family = one compared model against MSCA + MES in one metric across the datasets of the table,
    # Holm over these datasets. Replaces the Welch test with Holm over the seven models of one column (before 18-3).
    sig_rows = []
    for m, _ in MODELS:
        for k, _ in METRICS:
            ds_ok = [d for d, _ in DATASETS if len(per.get((OURS[0], d, k), {})) >= 2 and len(per.get((m, d, k), {})) >= 2]
            fam = sigtest.contrast({d: per[(m, d, k)] for d in ds_ok}, {d: per[(OURS[0], d, k)] for d in ds_ok}, ds_ok,
                                   fam='table2', name=f'{m} - MSCA + MES', a=m, b='MSCA + MES', metric=k)
            sig_rows += fam
            for r in fam:
                if r['reject']:
                    sig[(m, r['dataset'], k)] = '^{\\blacktriangle}' if r['diff'] > 0 else '^{\\blacktriangledown}'
    sigtest.write_csv(ROOT / 'sigtests_table2.csv', sig_rows)
    all5 = all(c['seeds'] == '1/2/3/4/5' for c in cells.values())
    with open(MARKS, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['row', 'label', 'dataset', 'metric', 'printed', 'mark', 'sig_vs_ours', 'setting', 'provisional'])
        for m, label in rows:
            for d, k in cols:
                s = sig.get((m, d, k), '')
                w.writerow([m, label, d, k, printed[(m, d, k)] or '', marks.get((m, d, k)) or '',
                            'higher' if s.endswith('blacktriangle}') else ('lower' if s else ''),
                            ('mes-provisional' if (m, d) in prov else 'mes') if m in ours_keys else setting_id(),
                            int((m, d) in prov)])

    def cell(m, d, k):
        v = printed[(m, d, k)]
        if v is None:
            return '\\todo{pending}' if m in ours_keys | {'MSCA'} else '--'
        c = cells[(m, d)]
        sd = c.get(f'{k}_std')
        body = v + (f'{{\\scriptsize$\\pm${float(sd):.4f}}}'.replace('0.', '.', 1) if sd else '')
        body = {'best': f'\\textbf{{{body}}}'}.get(marks[(m, d, k)], body)   # no second-best underline (2026-10-05)
        body = mes_source.tbd(body, (m, d) in prov)   # provisional MES cell: red
        # zero-width marker (\rlap) so that marked and unmarked cells keep the same alignment
        body += '' if (m, d, k) not in sig else f'\\rlap{{${sig[(m, d, k)]}$}}'
        return body + ('' if all5 else f"\\,({c['n']})")

    sha = hashlib.sha256(CELLS.read_bytes() + (CELLS_EXTRA.read_bytes() if CELLS_EXTRA.exists() else b'')).hexdigest()[:16]
    t1sha = hashlib.sha256(native_rows.CELLS.read_bytes() + native_rows.PERSEED.read_bytes()).hexdigest()[:16]
    seed_text = ('the mean over seeds 1--5 with the standard deviation over seeds' if all5 else
                 'the mean over the available seeds among seeds 1--5, with the number of seeds in parentheses')
    words = {'2': 'two', '3': 'three', '4': 'four'}
    runs_text = ''.join(f'For {label}, each seed first averages {words.get(min(rps[key]), min(rps[key]))} runs of that seed. '
                        if len(rps.get(key, set())) == 1 and min(rps[key]) != '1' else
                        (f'For {label}, each seed first averages all runs of that seed. ' if rps.get(key, set()) - {'1'}
                         else '')
                        for key, label, _ in OURS_ROWS)
    lines = [
        f'% Generated by tables/src/build_table2_main.py from tables/table2_cells.csv + table2_cells_extra.csv (sha256 {sha}), '
        f'tables/table2_runs.tsv, tables/src/mes_source.py (MSCA + MES) and, for the MSCA row, tables/table1_cells.csv '
        f'and table1_perseed.csv (sha256 {t1sha}). Do not edit numbers by hand.',
        mes_source.comment([bb for _, _, bb in OURS_ROWS], [d for d, _ in DATASETS]),
        '\\begin{table*}[t]',
        '\\caption{Recall@20 and NDCG@20 of multimodal recommenders under the common training protocol. '
        'Every compared model uses its author-derived configuration, to which MSCA + MES adds MES. '
        f'Each cell reports {seed_text}. {runs_text}'
        'In each column, the best mean over seeds among the unfused rows is in bold. '
        '$^{\\blacktriangledown}$/$^{\\blacktriangle}$: significantly lower/higher than MSCA + MES '
        f'({sigtest.RULE_TEXT}). '
        'The last block adds the same training-free fusion of RP3$\\beta$ and ItemKNN-CBF scores (F) to every model, with the fusion weight '
        'selected on validation recall for each seed; its rows carry no bold, and no model is claimed to rank first after fusion. '
        # revision log 29 (item 1): the mix rows and the two margin rows of the fused block
        'Its last four rows fuse the validation mix of RP3$\\beta$, ItemKNN-CBF, ItemKNN-CF, and EASE instead, and give the margin of MSCA + MES over MSCA after F '
        # revision log 31 (MSCA10): the two margin rows use seeds 1-10 of the preregistered extension, Holm over the four tuned datasets
        'and after the mix. The two margin rows ($^{\\dagger}$) use seeds 1--10 of the preregistered ten-seed extension of the fusion run set, whereas the other rows average seeds 1--5, '
        # revision log 32 (decision ): the marks of the two margin rows in the NDCG@20 columns are post hoc (the preregistered tests cover Recall@20)
        'and they are marked when significant with Holm correction over the four datasets of this table, where the marks in the NDCG@20 columns are post hoc. '
        # revision log 18-D (decision 8, SOTAFILL): the eight cells with fewer than five seeds are no longer red; sentence verbatim from the decision
        'A number in parentheses in this block gives the number of seeds that passed the re-evaluation gate, and such a cell shows the mean over these seeds without a standard deviation. '
        'Missing seeds in the fused block failed the re-evaluation gate because of stochastic evaluation (LGMRec Gumbel sampling, COHESION graph padding). '
        # revision log 18-4 (decision a): checkpoints of MSCA in the fused block
        # revision log 29 (items 2 and 5): run set and Baby cache build named in the caption
        'The fused MSCA rows come from the five-seed fusion run set of MSCA, which shares the checkpoints of the rows above on Sports alone, '
        'since only seed 1 of these checkpoints is stored on Baby, Pantry, and Scientific (Section~\\ref{sec:settings}). '
        'On Baby, MSCA, MSCA + MES, and the unfused COHESION row read the canonical build of the item kNN cache, the other rows of SMORE, PGL, FREEDOM, and COHESION read its second build, '
        'and DiffMM and LGMRec read neither (supplementary Table~\\ref{tab:babybuild}). '
        'Table~\\ref{tab:fuse_added} reports the fusion on five further datasets.'
        + mes_source.caption_note([bb for _, _, bb in OURS_ROWS], [d for d, _ in DATASETS]) + '}',
        '\\label{tab:sota}',
        '\\centering',
        '\\footnotesize',
        '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{tabular}{l' + 'cc' * PANEL + '}',   # no \resizebox (lab rule): datasets stacked in panels of PANEL
    ]
    for pi in range(0, len(DATASETS), PANEL):
        panel = DATASETS[pi:pi + PANEL]
        pad = ' &' * (2 * (PANEL - len(panel)))
        pcols = [(d, k) for d, _ in panel for k, _ in METRICS]
        lines += [
            '\\toprule' if pi == 0 else '\\midrule[\\heavyrulewidth]',
            ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{name}}}' for _, name in panel) + pad + ' \\\\',
            ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(panel))),
            'Model & ' + ' & '.join(' & '.join(name for _, name in METRICS) for _ in panel) + pad + ' \\\\',
            '\\midrule',
        ] + [f'{label} & ' + ' & '.join(cell(key, d, k) for d, k in pcols) + pad + ' \\\\' for key, label, _ in OURS_ROWS]
        lines.append('\\midrule')
        for m, label in MODELS:
            lines.append(f'{label} & ' + ' & '.join(cell(m, d, k) for d, k in pcols) + pad + ' \\\\')
        lines += ['\\midrule', '\\multicolumn{%d}{l}{\\textit{Training-free post-hoc fusion: backbone $\\oplus$ z-scored RP3$\\beta$ + ItemKNN-CBF '
                  '(weight chosen on validation per seed)}}\\\\' % (1 + 2 * PANEL)]
        for key, label in [(OURS[0], OURS[1])] + [(m, m) for m, _ in MODELS]:
            lines.append(f'{label} $\\oplus$ & ' + ' & '.join(fusion_cell(fus, key, d, k) for d, k in pcols) + pad + ' \\\\')
        # revision log 29 (item 1): the mix fusion of MSCA + MES and MSCA, then the two margins
        lines.append('\\midrule')
        for key, label in (('MSCA+MES', 'MSCA + MES'), ('MSCA', 'MSCA')):
            lines.append(f'{label} $\\oplus$ mix & ' + ' & '.join(mix_cell(key, d, k) for d, k in pcols) + pad + ' \\\\')
        for variant, label in (('combo', 'Margin of MES, F$^{\\dagger}$'), ('mix', 'Margin of MES, mix$^{\\dagger}$')):
            lines.append(f'{label} & ' + ' & '.join(margin_cell(variant, d, k) for d, k in pcols) + pad + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    OUT.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
