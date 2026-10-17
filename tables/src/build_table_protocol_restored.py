"""E6 (thesis B scope (i), CLASSIC, 2026-10-16): protocol-sensitivity tables rebuilt from EXISTING runs only.

The two appendix tables left the supplement on 2026-10-07 (supplement.tex line 124 and text/6_appendix.tex lines 14 and 71-73:
"\\input{tables/appendix_protocol} removed ... (user decision; file kept on disk)" and "\\input{tables/appendix_protocol_sota} removed ...
(user decision: cells mix 1-seed and 3-seed means)").  This generator restores their content as ONE new file with the seed
count of every cell printed and a footnote "mixed seed counts"; it neither trains nor re-collects anything and it does not
change appendix_protocol*.tex, build_table_protocol.py or any \\input.  New run sets (B and P seeds 1-5 for the compared models,
MSCA + MES under B and P) are not part of this scope: those cells print "--" or their present seed count.

Reads (all local, unchanged): tables/protocol_runs.tsv (collect_protocol_runs.py: MSCA and FREEDOM, Table-I rows, protocols B and
P, seeds 1-3), tables/protocol_sota_runs.tsv (collect_protocol_sota_runs.py: compared models, B seed 1 and P seeds 1-3),
tables/table1_cells.csv + table1_runs.tsv (protocol C of the plug-in block), tables/table2_runs.tsv (protocol C of the compared
models, chosen runs, seeds 1-5), tables/table1_perseed.csv (protocol C of MSCA = Native row), tables/mes_perseed.csv (MSCA + MES, C).
Writes tables/appendix_protocol_restored.tex (labels tab:protocol_rest_sota and tab:protocol_rest_plug; not \\input anywhere) and
tables/src/data/classic_protocol_cells.csv (every cell: n, seeds, mean, early-stopped runs).  macro_lines() is called by
build_table_classic.py (macros_classic.tex, prefix px).  Dataset panels are stacked (no \\resizebox).
Usage: python -s tables/src/build_table_protocol_restored.py
"""
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_table_protocol as bp  # noqa: E402  (read-only reuse of aggregate / read_tsv / common_from_table1 / COLLAPSE_EPOCH)
from numwords import num_word  # noqa: E402

T = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / 'data'
OUT = T / 'appendix_protocol_restored.tex'
DS3 = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific')]
MODELS = ['MSCA', 'COHESION', 'SMORE', 'PGL', 'DiffMM', 'LGMRec', 'FREEDOM']
PROT = [('common', 'C'), ('b2048', 'B'), ('plugin', 'P')]
BACK = ['MSCA', 'FREEDOM']
ROWS = bp.ROWS


def sota_cells():
    """{(protocol, model, dataset): cell} for the compared models and MSCA + MES"""
    cells = bp.aggregate(bp.read_tsv(bp.SOTA), ['protocol', 'model', 'dataset'])
    cells = {k: v for k, v in cells.items() if k[1] in MODELS}
    # protocol C = the chosen runs of Table sota (table2_runs.tsv), MSCA = the Native runs of Table I
    runs = [r for r in bp.read_tsv(T / 'table2_runs.tsv') if r['chosen'] == '1' and r['model'] in MODELS and r['model'] != 'MSCA'
            and r['dataset'] in dict(DS3)]
    by = {}
    for r in runs:
        by.setdefault((r['model'], r['dataset']), []).append(r)
    for (m, d), rr in by.items():
        cells[('common', m, d)] = dict(n=len(rr), r20=st.mean(float(r['test_r20']) for r in rr), n20=st.mean(float(r['test_n20']) for r in rr),
                                       collapsed=sum(int(r['best_epoch']) <= bp.COLLAPSE_EPOCH for r in rr),
                                       best_epochs='/'.join(r['best_epoch'] for r in rr), seeds='/'.join(r['seed'] for r in rr))
    ms = [r for r in csv.DictReader(open(T / 'table1_perseed.csv', newline='')) if r['backbone'] == 'MSCA' and r['row'] == 'Native'
          and r['dataset'] in dict(DS3)]
    for d, _ in DS3:
        rr = [r for r in ms if r['dataset'] == d]
        cells[('common', 'MSCA', d)] = dict(n=len(rr), r20=st.mean(float(r['r20']) for r in rr), n20=st.mean(float(r['n20']) for r in rr),
                                            collapsed=0, best_epochs='', seeds='/'.join(r['seed'] for r in rr))
    me = [r for r in csv.DictReader(open(T / 'mes_perseed.csv', newline='')) if r['backbone'] == 'MSCA' and r['row'] == 'MES'
          and r['dataset'] in dict(DS3)]
    for d, _ in DS3:
        rr = [r for r in me if r['dataset'] == d]
        cells[('common', 'MSCA+MES', d)] = dict(n=len(rr), r20=st.mean(float(r['r20']) for r in rr), n20=st.mean(float(r['n20']) for r in rr),
                                                collapsed=0, best_epochs='', seeds='/'.join(r['seed'] for r in rr))
    return cells


def plug_cells():
    cells = bp.aggregate(bp.read_tsv(bp.RUNS), ['protocol', 'backbone', 'row', 'dataset'])
    bp.common_from_table1(cells)
    return cells


def fmt(c):
    return '--' if c is None else bp.fmt(c)


def caption_sota():
    return ('Test Recall@20 of the compared recommenders under three training protocols. '
            'C: the common protocol of Section~\\ref{supp:settings}. B: batch 2048, 1000 epochs, patience 20, rest as C. '
            'P: MMRec defaults (batch 2048, Adam, constant learning rate, full precision, 1000 epochs, patience 20). '
            'Each model uses its author configuration under B and P. Each cell is the mean over the available seeds, with the number of seeds in '
            'parentheses (mixed seed counts: C uses seeds 1--5, B uses seed 1, and P uses seeds 1--3). '
            f'A superscript $k$: in $k$ of these runs, validation Recall@20 peaked within {bp.COLLAPSE_EPOCH} epochs and did not exceed it in '
            'the next 20 epochs, thus early stopping ended training. --: not run under this protocol. '
            # revision log 25 (decision e): rank agreement with C and the dataset without B and P runs
            'Kendall $\\tau$: agreement of the ranking of the seven compared models (MSCA + MES excluded) under B or P with their ranking '
            'under C. Sports has no runs under B and P.')


def build():
    sc, pc = sota_cells(), plug_cells()
    L = ['% Generated by tables/src/build_table_protocol_restored.py (CLASSIC E6, 2026-10-16) from existing runs only. Do not edit numbers by hand.',
         '% New file: not \\input anywhere. Restores the content of tab:protocol and tab:protocol_sota (removed 2026-10-07) with the seed count of each cell.',
         '\\begin{table}[t]', f'\\caption{{{caption_sota()}}}', '\\label{tab:protocol_rest_sota}', '\\centering', '\\footnotesize',
         '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{lccc}', '\\toprule', 'Model & C & B & P \\\\']
    for d, name in DS3:
        L += ['\\midrule', f'\\multicolumn{{4}}{{l}}{{\\textit{{{name}}}}} \\\\']
        for m in MODELS + ['MSCA+MES']:
            lab = 'MSCA + MES' if m == 'MSCA+MES' else m
            L.append(f'{lab} & ' + ' & '.join(fmt(sc.get((p, m, d))) for p, _ in PROT) + ' \\\\')
        # revision log 25 (decision e): Kendall tau of the seven compared models between C and B / P (same values as the px macros)
        c = [sc[('common', m, d)]['r20'] for m in MODELS]
        L.append('Kendall $\\tau$ with C & -- & '
                 + ' & '.join(f"{kendall(c, [sc[(p, m, d)]['r20'] for m in MODELS]):.2f}" for p, _ in PROT[1:]) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    cap2 = ('Test Recall@20 of the plug-in rows of Table~\\ref{tab:main} on MSCA and of Table~\\ref{tab:transfer} on FREEDOM under the protocols of '
            'Table~\\ref{tab:protocol_rest_sota}. Each cell is the mean over the available seeds, with the number of seeds in parentheses '
            '(mixed seed counts: C uses seeds 1--5, B and P use seeds 1--3). The superscript $k$ and the dash are defined as in '
            'Table~\\ref{tab:protocol_rest_sota}.')
    L += ['\\begin{table*}[t]', f'\\caption{{{cap2}}}', '\\label{tab:protocol_rest_plug}', '\\centering', '\\footnotesize',
          '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{l' + 'ccc' * len(BACK) + '}', '\\toprule',
          ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{b}}}' for b in BACK) + ' \\\\',
          ''.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(BACK))),
          'Objective & ' + ' & '.join(' & '.join(p for _, p in PROT) for _ in BACK) + ' \\\\']
    for d, name in DS3:
        L += ['\\midrule', f'\\multicolumn{{{1 + 3 * len(BACK)}}}{{l}}{{\\textit{{{name}}}}} \\\\']
        for row, label in ROWS:
            vals = [fmt(pc.get((p, b, row, d))) for b in BACK for p, _ in PROT]
            L.append(f"{label or 'Backbone'} & " + ' & '.join(vals) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    OUT.write_text('\n'.join(L), encoding='utf-8', newline='\n')
    # cell list
    rows = []
    for (p, m, d), c in sorted(sc.items()):
        rows.append(['compared', p, m, 'Native', d, c['n'], f"{c['r20']:.4f}", f"{c['n20']:.4f}", c['collapsed'], c['seeds']])
    for (p, b, r, d), c in sorted(pc.items()):
        rows.append(['plugin_block', p, b, r, d, c['n'], f"{c['r20']:.4f}", f"{c['n20']:.4f}", c['collapsed'], c['seeds']])
    with open(DATA / 'classic_protocol_cells.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['block', 'protocol', 'model', 'row', 'dataset', 'n', 'r20_mean', 'n20_mean', 'n_stopped_le2', 'seeds'])
        w.writerows(rows)
    return sc, pc, rows


def kendall(a, b):
    from scipy import stats
    return float(stats.kendalltau(a, b).statistic)


def macro_lines():
    sc, pc, rows = build()
    out = ['% --- E6 protocol sensitivity (build_table_protocol_restored.py), existing runs only; counts 0-10 spelled out']

    def mac(n, v):
        out.append(f'\\newcommand{{\\{n}}}{{{v}}}')
    f4 = lambda x: f'{x:.4f}'   # noqa: E731
    # MSCA and PGL on Pantry: range over the three protocols
    for key, cell in (('Msca', lambda p: pc.get((p, 'MSCA', 'Native', 'pantry'))), ('Pgl', lambda p: sc.get((p, 'PGL', 'pantry')))):
        vals = {p: cell(p) for p, _ in PROT}
        for p, tag in PROT:
            mac(f'px{key}Pantry{tag}', f4(vals[p]['r20']))
            mac(f'px{key}Pantry{tag}N', num_word(vals[p]['n']))
        mac(f'px{key}PantryLo', f4(min(v['r20'] for v in vals.values())))
        mac(f'px{key}PantryHi', f4(max(v['r20'] for v in vals.values())))
    # revision log 29 (item 5): MSCA on Pantry in the compared-model table (Table D1) next to the backbone row of the plug-in table (D2)
    for p, tag in PROT:
        c = sc.get((p, 'MSCA', 'pantry'))
        mac(f'pxMscaDPantry{tag}', f4(c['r20'])); mac(f'pxMscaDPantry{tag}N', num_word(c['n']))
    # coverage
    allc = [c for c in list(sc.values()) + list(pc.values())]
    mac('pxNCells', str(len(allc)))
    mac('pxNMixed', str(sum(c['n'] < 5 for c in allc)))
    mac('pxNOneSeed', str(sum(c['n'] == 1 for c in allc)))
    mac('pxNStopped', str(sum(c['collapsed'] > 0 for c in allc)))
    mac('pxNModels', num_word(len(MODELS)))
    mac('pxNDs', num_word(len(DS3)))
    # revision log 33 (review 3, minor): seeds per cell of the compared-model table behind the Kendall tau (most seeds in any cell: C five, B one, P three)
    nmax = {tag: max(sc[(p, m, d)]['n'] for m in MODELS for d, _ in DS3 if (p, m, d) in sc) for p, tag in PROT}
    assert nmax['B'] == 1, nmax   # the sentences read "one seed" under B
    for tag in ('C', 'B', 'P'):
        mac(f'pxSeeds{tag}', num_word(nmax[tag]))
    # ranking agreement of the seven compared models between C and B / P (Kendall tau of the mean Recall@20; B has one seed per cell)
    taus = []
    for d, name in DS3:
        c = [sc[('common', m, d)]['r20'] for m in MODELS]
        for p, tag in PROT[1:]:
            v = [sc[(p, m, d)]['r20'] for m in MODELS]
            mac(f'pxTau{tag}{name}', f'{kendall(c, v):.2f}')
            taus.append(float(f'{kendall(c, v):.2f}'))
        top = {p: max(MODELS, key=lambda m: sc[(p, m, d)]['r20']) for p, _ in PROT}
        mac(f'pxTop{name}C', top['common']); mac(f'pxTop{name}B', top['b2048']); mac(f'pxTop{name}P', top['plugin'])
    # revision log 25 (decision e): range of the six Kendall tau values (three datasets x B, P)
    mac('pxTauMin', f'{min(taus):.2f}'); mac('pxTauMax', f'{max(taus):.2f}')
    return out


if __name__ == '__main__':
    lines = macro_lines()
    print('\n'.join(lines))
    print('wrote', OUT, DATA / 'classic_protocol_cells.csv')
