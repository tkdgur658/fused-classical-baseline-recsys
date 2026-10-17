"""Build the appendix tables of XRE from tables/xre_cells.csv (tables/src/collect_xre.py). Numbers are never typed by hand.

  tables/appendix_sensitivity.tex  MSCA + XRE around the adopted setting (collect_xre.SET['sens']): three lambda values
                                   at the adopted K and three K values at the adopted lambda; the adopted row is XRE
                                   itself (l003_k20: lambda {0.01, 0.03, 0.1}, K {10, 20, 40}). Seeds 1-5.
  tables/appendix_transfer.tex     tab:transfer, MENTOR, FREEDOM and MMGCN: XRE, XRE on full view, Anchor term (A)
                                   (the anchor term alone, labelled as in Table III), the six plug-ins (FREEDOM / MMGCN, table1_cells.csv) and Native
                                   (table1_cells.csv for FREEDOM / MMGCN, the MENTOR row of table2_cells.csv). Plain means,
                                   seeds 1-5 (an incomplete cell is marked ^{\ast} and the caption gives its seed
                                   count; 2026-10-06: the former \rlap{(n)} superscript overlapped the next column).
  tables/appendix_ablation.tex     tab:ablation_full, every MSCA variant of fig_ablation plus w/o retention and XRE on
                                   full view, mean and SD over seeds 1-5, labelled as in Table III. ^{\blacktriangledown}:
                                   the appendix family of fig_ablation.py (tables/ablation_tests.csv, Holm over XRE
                                   against the exchange term alone, w/o retention and XRE on full view), Recall@20 only.
Cells without collected runs print a red \\todo{pending}; re-running collect_xre.py and this builder fills them.
Before the transfer table is used, check the MMGCN runs for NaN batches with tools/r16t_scan_add.py on HOST_A
(P04_xret_wt/results/xret/APPLICABILITY.md); a diverged run must be reported as diverged, not as a result.
Usage: python tables/src/build_table_xre_appendix.py
"""
import csv
import hashlib
from pathlib import Path

import native_rows
import mes_source   # MES rows of the transfer table (r20_R1/MANU 2026-10-10: mes_f / mes_m / mes_n runs)
from collect_xre import SET, check_setting

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / 'xre_cells.csv'
T2_CELLS = ROOT / 'table2_cells.csv'
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific')]


def load():
    with open(CELLS, newline='') as f:
        cells = {(r['backbone'], r['row'], r['dataset']): r for r in csv.DictReader(f) if r['seedset'] == '12345'}
    # Native rows = the Native runs printed in the main tables (one run set per Native value): FREEDOM and MMGCN from
    # Table II (tables/src/native_rows.py), MENTOR from Table I (tables/table2_cells.csv). The in-package lambda = 0
    # runs (xret_* <ds>_native_l000, bitwise identical to these configurations in the xre_t unit test) stay in
    # xre_cells.csv as information and are not printed.
    t2 = {}
    if T2_CELLS.exists():
        with open(T2_CELLS, newline='') as f:
            t2 = {(r['model'], r['dataset']): r for r in csv.DictReader(f)}
    for d, _ in DATASETS:
        for bb in ('FREEDOM', 'MMGCN'):
            cells.pop((bb, 'Native', d), None)
            if native_rows.native(bb, d):
                cells[(bb, 'Native', d)] = native_rows.native(bb, d)
        cells.pop(('MENTOR', 'Native', d), None)
        m = t2.get(('MENTOR', d))
        if m and m['n'] == '5':
            cells[('MENTOR', 'Native', d)] = m
    return cells


def cell(c, key):
    if not c:
        return '\\todo{pending}'
    v = f"{float(c[key]):.4f}"
    sd = c.get(key.replace('mean', 'std'))
    s = v + (f"{{\\tiny$\\pm${float(sd):.4f}}}".replace('0.', '.', 1) if sd else '')
    return s if c['n'] == '5' else s + f"\\,({c['n']})"


def header(caption, label, first, ds=None):
    ds = ds or DATASETS
    return ['\\begin{table*}[!tbp]', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\centering', '\\scriptsize',
            '\\setlength{\\tabcolsep}{2pt}', '\\begin{tabular}{l' + 'cc' * len(ds) + '}', '\\toprule',   # no \resizebox (lab rule)
            ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{n}}}' for _, n in ds) + ' \\\\',
            ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(ds))),
            f'{first} & ' + ' & '.join('R@20 & N@20' for _ in ds) + ' \\\\', '\\midrule']


def body(cells, rows, ds=None):
    ds = ds or DATASETS
    out = []
    for bb, key, label in rows:
        if bb is None:
            out.append(label)
            continue
        out.append(f'{label} & ' + ' & '.join(cell(cells.get((bb, key, d)), k) for d, _ in ds
                                             for k in ('r20_mean', 'n20_mean')) + ' \\\\')
    return out


def main():
    check_setting()
    cells = load()
    sha = hashlib.sha256(CELLS.read_bytes()).hexdigest()[:16]
    t1sha = hashlib.sha256(native_rows.CELLS.read_bytes()).hexdigest()[:16] if native_rows.CELLS.exists() else 'absent'
    gen = (f'% Generated by tables/src/build_table_xre_appendix.py from tables/xre_cells.csv (sha256 {sha}); Native rows '
           f'of the transfer table from tables/table1_cells.csv (sha256 {t1sha}) and tables/table2_cells.csv (MENTOR). '
           'Do not edit numbers by hand.')
    sens = []   # rows of the adopted setting (collect_xre.SET['sens']); the adopted row is XRE itself
    for i, (group, lam, k, row) in enumerate(SET['sens']):
        if i and group != SET['sens'][i - 1][0]:
            sens.append((None, None, '\\midrule'))
        sens.append(('MSCA', row, f'$\\lambda={lam}$, $K={k}$' + (' (XRE)' if row == 'XRE' else '')))
    words = {'1': 'one', '2': 'two', '3': 'three', '4': 'four'}

    def runs(rows_):
        got = {n for r in rows_ for d, _ in DATASETS if cells.get(('MSCA', r, d))
               for n in cells[('MSCA', r, d)]['runs_per_seed'].split('/')}
        return words.get(next(iter(got)), 'all') if len(got) == 1 else 'all'
    other = [r for _, _, _, r in SET['sens'] if r != 'XRE']
    lines = [gen] + header('Sensitivity of MSCA with XRE, the E + R + A design, to the weight $\\lambda$ and the list size $K$ of the exchange '
                           'term. Each cell reports the mean and standard deviation over seeds 1--5, or over the number of seeds in '
                           'parentheses. Every row uses one run per seed. '
                           # revision log 18-4: not the tuned MES of the main text (macro of build_macros.py, values read from the data)
                           '\\sensMesBabyNote{}',
                           'tab:sensitivity', 'Setting') + body(cells, sens) + ['\\bottomrule', '\\end{tabular}',
                                                                              '\\end{table*}', '']
    lines[1] = '\\begin{table*}[!t]'
    (ROOT / 'appendix_sensitivity.tex').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    # ---- transfer table (tab:transfer): FREEDOM and MMGCN with Native, the six plug-ins (Table-I cells of
    # tables/table1_cells.csv, collect_table1.py) and the XRE rows; MENTOR with Native and the XRE rows (no plug-in runs).
    # Plain means (as in Table I); the best printed mean of each block and column is in bold.
    t1 = native_rows.cells()
    # printed labels as Table I (user decision 2026-10-06): '+ ' for added terms, the backbone row prints the backbone name
    plug = [('INA', 'INA-style~\\cite{iimrec}'),   # VI-MMRec dropped (user decision 2026-10-03)
            ('MDVT', 'MDVT~\\cite{mdvt}'), ('MG', 'MG~\\cite{zhong2024mg}'), ('DA-MRS', 'DA-MRS~\\cite{xu2024damrs}'),
            ('UltraGCN-LI', 'UltraGCN $\\mathcal{L}_I$~\\cite{mao2021ultragcn}')]
    tr, short = [], []
    mes_any, mes_all = [], []
    for bb, lab in (('MENTOR', 'MENTOR (modified)~\\cite{xu2024mentor}'), ('FREEDOM', 'FREEDOM~\\cite{zhou2023freedom}'),
                    ('MMGCN', 'MMGCN~\\cite{wei2019mmgcn}')):  # newest backbone first
        mes = [mes_source.cell(bb, d) for d, _ in DATASETS]
        mes = [c if c and not c['provisional'] else None for c in mes]   # real MES runs only (no stand-in in this table)
        mes_any.append(any(mes)); mes_all.append(all(mes))
        block = ([('MES', mes)] if any(mes) else []) + [
                 ('E + R + A', [cells.get((bb, 'XRE', d)) for d, _ in DATASETS]),
                 ('E + R + A, full view', [cells.get((bb, 'wo_masked', d)) for d, _ in DATASETS]),
                 ('A', [cells.get((bb, 'A1', d)) for d, _ in DATASETS])]
        if bb != 'MENTOR':
            block += [(pl, [native_rows.complete(t1.get((bb, k, d))) for d, _ in DATASETS]) for k, pl in plug]
        block.append((f'Backbone ({bb})', [cells.get((bb, 'Native', d)) for d, _ in DATASETS]))
        best = {}
        for di in range(len(DATASETS)):
            for k in ('r20_mean', 'n20_mean'):
                v = [f"{float(c[di][k]):.4f}" for _, c in block if c[di] and c[di].get('n') == '5']
                best[(di, k)] = max(v, key=float) if v else None
        tr.append(f'\\midrule\n\\multicolumn{{7}}{{l}}{{\\textit{{Backbone: {lab}}}}} \\\\')
        for label, cs in block:
            vals, row_short = [], []
            for di, c in enumerate(cs):
                for k in ('r20_mean', 'n20_mean'):
                    if not c:
                        vals.append('\\todo{pending}')
                        continue
                    v = f"{float(c[k]):.4f}"
                    if c.get('n') != '5':   # incomplete cell: never bold; the row label is marked, the caption says where
                        if k == 'r20_mean':
                            row_short.append((DATASETS[di][1], c['n']))
                        vals.append(v)
                        continue
                    vals.append(f'\\textbf{{{v}}}' if v == best[(di, k)] else v)
            if row_short:   # marker on the row label (a cell marker overlapped the next column, 2026-10-06)
                short.append((bb, label, row_short))
            tr.append(f'{label}' + ('$^{\\ast}$' if row_short else '') + ' & ' + ' & '.join(vals) + ' \\\\')
    short_note = []
    for bb, label, rs in short:
        for n in sorted({n for _, n in rs}):
            dd = [D for D, m in rs if m == n]
            short_note.append(f'{label} of {bb} averages {words[n]} seeds on '
                              + (' and '.join(dd) if len(dd) < 3 else ', '.join(dd[:-1]) + ', and ' + dd[-1]))
    lines = [gen, '\\begin{table}[!t]'   # tall one-column table: top of a column of page 1 of the supplement (float before first mention),
             f"\\caption{{Recall@20 and NDCG@20 of the full design E + R + A (entry pairs, retention pairs, and anchor term) on MENTOR (modified), FREEDOM, and MMGCN with $\\lambda={SET['lam']}$ and "
             f"$K={SET['K']}$ unchanged, next to the five plug-ins of Table~\\ref{{tab:main}} on FREEDOM and MMGCN. E + R + A, full view computes the entry and retention "
             'pairs on the full view, while A adds the anchor term alone. '
             + ('MES uses the same $\\lambda$ and $K$. ' if all(mes_all) else '\\tbd{MES runs on these backbones are pending.} ')
             + 'The FREEDOM and MMGCN backbone rows come from the plug-in comparison runs of '
             'the same configurations, thus letting a FREEDOM mean differ from Table~\\ref{tab:sota} in the fourth decimal. '
             'MENTOR (modified) is our implementation, described in Section~\\ref{app:xre_extra}. Each cell reports '
             'the mean over seeds 1--5 with one run per seed'
             + (', except in the row marked with $^{\\ast}$' if short else '') + '. '
             + ('$^{\\ast}$: ' + '. '.join(short_note) + '. ' if short else '') +
             'In each block and column, the highest mean is shown in bold.}',   # R0525 (05 L-1): last line was "BOLD."
             '\\label{tab:transfer}', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{2.5pt}',
             '\\begin{tabular}{l' + 'cc' * len(DATASETS) + '}', '\\toprule',
             ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{n}}}' for _, n in DATASETS) + ' \\\\',
             ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(DATASETS))),
             'Objective & ' + ' & '.join('R@20 & N@20' for _ in DATASETS) + ' \\\\'] + tr + [
             '\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    (ROOT / 'appendix_transfer.tex').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    # ---- full ablation table (tab:ablation_full): every MSCA variant of the adopted setting, mean and SD over seeds
    # Sports (2026-10-07): printed once every variant has seeds 1-5 there (collect_xre.SPORTS_ABL_ROOT)
    abl_ds = DATASETS + [(d, D) for d, D in (('sports', 'Sports'),)
                         if all((cells.get(('MSCA', r, d)) or {}).get('n') == '5'
                                for r in ('XRE', 'A1', 'wo_A1', 'wo_A1_masked', 'wo_retention', 'wo_masked'))]
    for d, _ in abl_ds:
        c = native_rows.native('MSCA', d)
        if c:
            cells[('MSCA', 'Native', d)] = c
    abl = [('MSCA', 'XRE', 'MSCA + XRE'), ('MSCA', 'A1', 'Anchor term (A)'), ('MSCA', 'wo_A1', 'Exchange (masked view)'),
           ('MSCA', 'wo_A1_masked', 'Exchange (full view)'), ('MSCA', 'wo_retention', 'w/o retention'),
           ('MSCA', 'wo_masked', 'XRE on full view'), ('MSCA', 'Native', 'MSCA')]
    # appendix family of figures/src/fig_ablation.py (tables/ablation_tests.csv): Recall@20 of the row b significantly
    # lower than XRE after Holm correction over its three contrasts per dataset
    mark = set()
    tests = ROOT / 'ablation_tests.csv'
    if tests.exists():
        for r in csv.DictReader(open(tests, encoding='utf-8')):
            if r['family'] == 'appendix' and r['holm_rejected'] == '1' and float(r['diff']) > 0:
                mark.add((r['b'], r['dataset']))
    abl_body = body(cells, abl, abl_ds)
    for i, (bb, key, label) in enumerate(abl):
        parts = abl_body[i][:-3].split(' & ')
        for j, (d, _) in enumerate(abl_ds):
            if (key, d) in mark:
                parts[1 + 2 * j] += '$^{\\blacktriangledown}$'
        abl_body[i] = ' & '.join(parts) + ' \\\\'
    lines = [gen] + header('Component ablation of XRE on MSCA. Anchor term (A) and the two exchange rows train one term '
                           'of XRE alone, as in Table~\\ref{tab:xmask}. W/o retention keeps only the entry pairs, '
                           'whereas XRE on full view computes the exchange term on the full view. Each cell reports the '
                           'mean and standard deviation over seeds 1--5, with one run per seed. '
                           '$^{\\blacktriangledown}$: Recall@20 significantly lower than MSCA + XRE (paired $t$-test '
                           'over seeds, Holm correction over Exchange (masked view), w/o retention, and XRE on full '
                           'view, $p<0.05$).',   # 2026-10-07: run hardware no longer stated (user decision)
                           'tab:ablation_full', 'Variant', abl_ds) + abl_body + ['\\bottomrule', '\\end{tabular}',
                                                                       '\\end{table*}', '']
    lines[1] = '\\begin{table*}[!t]'
    (ROOT / 'appendix_ablation.tex').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print('wrote appendix_sensitivity.tex, appendix_transfer.tex, appendix_ablation.tex')


if __name__ == '__main__':
    main()
