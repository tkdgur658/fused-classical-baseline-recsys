"""Build tables/table_xmask.tex (Table III: sources of the gain of XRE on MSCA) and tables/macros_xmask.tex (numbers of
the prose). Numbers are never typed by hand.

Sources: tables/xmask_perseed.csv (collect_xmask.py: masked BPR / DNS / SL@K rows), tables/xre_perseed.csv (rows XRE,
A1 = anchor term alone, wo_A1 = exchange term alone on the masked view and wo_A1_masked = exchange term alone on the full
view, all of MSCA, collect_xre.py), tables/table1_perseed.csv (MSCA Native and the plug-in rows of Table I,
collect_table1.py), tables/topk_perseed.csv (top-K loss rows of Table I on MSCA, collect_topk.py) and
tables/ablation_tests.csv (Holm-corrected contrasts of figures/src/fig_ablation.py) and tables/m1probe_perseed.csv
(single-probe anchor term, collect_m1probe.py).
Rows (2026-10-06; the component ablation figure fig:ablation is merged into this table, review plan B1): XRE, the anchor
term plus one loss on the held-out positives of the masked view (DNS, SL@K, uniform BPR), the anchor term alone, the
exchange term alone on the masked view and on the full view, and Native. Sports is printed once every row has seeds 1-5.
Each cell: mean over seeds 1-5 of the test metric at the best-validation checkpoint. The best mean of each column is in
bold. Markers (no new test, the two existing families):
  $^{\\blacktriangledown}$  a row with the anchor term significantly lower than XRE in that column (paired t-test over seeds
                           1-5, Holm correction over the four rows that share the anchor term, namely masked BPR, masked
                           DNS, masked SL@K and the anchor term alone, alpha 0.05), computed here.
  $^{\\dagger}$             Recall@20 of the full-view exchange row significantly lower than the masked-view exchange row
  $^{\\ddagger}$            Recall@20 of the masked-view exchange row significantly higher than Native
                           (both from the figure family of fig_ablation.py: Holm over the three contrasts exchange,
                           masked and native per dataset; read from tables/ablation_tests.csv, never recomputed here).
Native is printed for reference and takes no ^{\\blacktriangledown} test.
Row A1first (added 2026-10-07, reviewer control M = 1; tables/m1probe_perseed.csv from tables/src/collect_m1probe.py): the
anchor term alone whose anchor is probe 0 of the same 50 uniform probes instead of the highest-scoring one, printed right
below "Anchor term (A)". It takes no ^{\\blacktriangledown} test (the Holm family stays the four rows above, so their marks
are unchanged) and enters the bold rule like every row; its paired contrast against row A1 is in tables/macros_m1probe.tex.
Row mdnsfv (added 2026-10-07, reviewer control E1; tables/e1_perseed.csv from tables/src/collect_e1.py): A + full-view DNS,
the DNS partner picked by the full-view score and the term computed on the full view with the coefficient mass of masked
DNS, printed right below "A + masked DNS". Like A1first it takes no ^{\\blacktriangledown} test and enters the bold rule;
its paired contrast against row mdns is in tables/macros_e1.tex and row (12) of the forest figure.
Sports (added 2026-10-07): the core rows (every row except the controls A1first and mdnsfv, which ran on Baby, Pantry and
Scientific only and print a dash on Sports) decide whether Sports is printed. All four datasets are treated alike
(user decision 2026-10-07: run hardware is not reported). Four datasets do not fit one column as eight numeric
columns (no \\resizebox, lab rule), so the table stacks two dataset panels of two datasets each.
Usage: python -s tables/src/build_table_xmask.py
"""
import csv
import statistics as st
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DS_ALL = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
DS = DS_ALL[:3]   # reset in main() to the datasets on which every row has seeds 1-5
SEEDS = [1, 2, 3, 4, 5]
ROWS = [('XRE', 'A + exchange (XRE)'), ('mdns', 'A + masked DNS'), ('mdnsfv', 'A + full-view DNS'),
        ('mslk', 'A + masked SL@$K$'),
        ('mbpr', 'A + masked BPR'), ('A1', 'Anchor term (A)'), ('A1first', 'Anchor term, single probe'),
        ('wo_A1', 'Exchange (masked view)'),
        ('wo_A1_masked', 'Exchange (full view)'), ('Native', 'Backbone (MSCA)')]   # Ours first (user, 2026-10-06)
OPTIONAL = ('A1first', 'mdnsfv')   # controls; dash where a dataset lacks their runs (Sports added 2026-10-07, none left)
ANCHOR_ROWS = ['XRE', 'mdns', 'mslk', 'mbpr', 'A1']
TESTED = ['mdns', 'mslk', 'mbpr', 'A1']
XRE_ROWS = ('XRE', 'A1', 'wo_A1', 'wo_A1_masked')
ALPHA = 0.05
WORDS = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}


def phrase(names):
    if not names:
        return 'none of the datasets'
    if len(names) == 1:
        return names[0]
    return ', '.join(names[:-1]) + (' and ' if len(names) == 2 else ', and ') + names[-1]


def load():
    V = {}
    for r in csv.DictReader(open(ROOT / 'xmask_perseed.csv', encoding='utf-8')):
        V[(r['row'], r['dataset'], int(r['seed']))] = (float(r['r20']), float(r['n20']))
    for r in csv.DictReader(open(ROOT / 'm1probe_perseed.csv', encoding='utf-8')):
        key = (r['row'], r['dataset'], int(r['seed']))
        assert key not in V, f'duplicate run {key}'   # one run per seed
        V[key] = (float(r['r20']), float(r['n20']))
    for r in csv.DictReader(open(ROOT / 'e1_perseed.csv', encoding='utf-8')):
        key = (r['row'], r['dataset'], int(r['seed']))
        assert key not in V, f'duplicate run {key}'   # one run per seed
        V[key] = (float(r['r20']), float(r['n20']))
    for r in csv.DictReader(open(ROOT / 'xre_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] in XRE_ROWS and r['dataset'] in dict(DS_ALL):
            V[(r['row'], r['dataset'], int(r['seed']))] = (float(r['r20']), float(r['n20']))
    T1 = {}
    for r in csv.DictReader(open(ROOT / 'table1_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['dataset'] in dict(DS_ALL):
            T1[(r['row'], r['dataset'], int(r['seed']))] = float(r['r20'])
            if r['row'] == 'Native':
                V[('Native', r['dataset'], int(r['seed']))] = (float(r['r20']), float(r['n20']))
    for r in csv.DictReader(open(ROOT / 'topk_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['dataset'] in dict(DS_ALL):
            T1[(r['row'], r['dataset'], int(r['seed']))] = float(r['r20'])
    global DS
    DS = [(d, D) for d, D in DS_ALL if all((k, d, s) in V for k, _ in ROWS if k not in OPTIONAL for s in SEEDS)]
    for d, _ in DS_ALL[:3]:
        assert all((k, d, s) in V for k, _ in ROWS for s in SEEDS), f'incomplete rows on {d}'
    return V, T1


def has(V, k, d):
    return all((k, d, s) in V for s in SEEDS)


def holm(p):
    order = sorted(range(len(p)), key=lambda i: p[i])
    rej = set()
    for k, i in enumerate(order):
        if p[i] <= ALPHA / (len(p) - k):
            rej.add(i)
        else:
            break
    return rej


def figure_family():
    """Recall@20 markers of the exchange-only rows, read from the figure family of fig_ablation.py."""
    marks, tests = {}, {}
    for r in csv.DictReader(open(ROOT / 'ablation_tests.csv', encoding='utf-8')):
        if r['family'] != 'figure':
            continue
        tests[(r['contrast'], r['dataset'])] = r
        if r['holm_rejected'] == '1' and float(r['diff']) > 0:
            if r['contrast'] == 'masked':
                marks[('wo_A1_masked', r['dataset'])] = '\\dagger'
            elif r['contrast'] == 'native':
                marks[('wo_A1', r['dataset'])] = '\\ddagger'
    for d, D in DS:
        for c in ('exchange', 'masked', 'native'):
            assert (c, d) in tests, f'ablation_tests.csv has no {c} contrast on {d}: run figures/src/fig_ablation.py'
    return marks


def main():
    V, T1 = load()
    mean = {(k, d, m): st.mean(V[(k, d, s)][m] for s in SEEDS) for k, _ in ROWS for d, _ in DS for m in (0, 1)
            if has(V, k, d)}
    sig, pvals, wins = {}, {}, {}
    for d, _ in DS:
        for m in (0, 1):
            x = [V[('XRE', d, s)][m] for s in SEEDS]
            ps = []
            for k in TESTED:
                y = [V[(k, d, s)][m] for s in SEEDS]
                ps.append(stats.ttest_rel(x, y).pvalue)
                wins[(k, d, m)] = sum(1 for a, b in zip(x, y) if a > b)
            for i in holm(ps):
                if mean[(TESTED[i], d, m)] < mean[('XRE', d, m)]:
                    sig[(TESTED[i], d, m)] = True
            for k, p in zip(TESTED, ps):
                pvals[(k, d, m)] = p
    fam = figure_family()
    best = {(d, m): max(f'{mean[(k, d, m)]:.4f}' for k, _ in ROWS if (k, d, m) in mean) for d, _ in DS for m in (0, 1)}
    best_anchor = {(d, m): max(f'{mean[(k, d, m)]:.4f}' for k in ANCHOR_ROWS) for d, _ in DS for m in (0, 1)}
    dashes = sorted({k for k in OPTIONAL for d, _ in DS if (k, d, 0) not in mean})

    def cell(k, d, m):
        if (k, d, m) not in mean:
            return '--'
        v = f'{mean[(k, d, m)]:.4f}'
        body = v.replace('0.', '.', 1)
        if v == best[(d, m)]:
            body = f'\\textbf{{{body}}}'
        if sig.get((k, d, m)):
            body += '\\rlap{$^{\\blacktriangledown}$}'
        if m == 0 and (k, d) in fam:
            body += f'\\rlap{{$^{{{fam[(k, d)]}}}$}}'
        return body

    ctrl_short = ((', and dashes mark controls run on '
                   + phrase([D for d, D in DS if all((k, d, 0) in mean for k in OPTIONAL)]) + ' only.')
                  if dashes else '.')
    lines = [
        '% Generated by tables/src/build_table_xmask.py from tables/xmask_perseed.csv, xre_perseed.csv, m1probe_perseed.csv, '
        'e1_perseed.csv, table1_perseed.csv and ablation_tests.csv.',
        '% Do not edit numbers by hand.',
        '\\begin{table}[!t]',   # [!t] (2026-10-07): may sit under Table II, on the page of its first mention
        # R0110 (2026-10-07): short caption; the row definitions, the coefficient-mass matching and the test families
        # moved to supplementary Section app:ablation (text/6_appendix.tex), which the caption cites.
        '\\caption{Ablation of XRE on MSCA, adding the anchor term (A) and one loss on the held-out positives, or training one term alone. Each cell '
        'reports the mean test Recall@20 (R) or NDCG@20 (N) over seeds 1--5, with the best mean of each column in bold. '
        # R0225 (03 P1-3, 06 N-3): the marker is tested on the four rows with A only, so the caption scopes it.
        # R0415 (verifiability): the caption names the Holm correction of every mark, so the significance words of the
        # abstract S7, contribution (ii) s2-s3, and conclusion S4-S5, which rest on these marks, are checkable in the PDF.
        'Holm-corrected $t$-tests give $^{\\blacktriangledown}$ for A and the three masked losses significantly lower than XRE and '
        # R0525 (03 P1-L1, P2-1): the marks name the exchange-alone rows they test; shorter, last line was 0.45
        '$^{\\dagger}$/$^{\\ddagger}$ for exchange-alone R significantly below the masked view/above MSCA. '
        'Supplementary Section~\\ref{app:ablation} defines these tests' + ctrl_short + '}',   # 2026-10-07: last caption line >= 0.8
        '\\label{tab:xmask}',
        '\\centering',
        '\\footnotesize',
    ]
    # one panel of all datasets if three fit the column; four datasets: two stacked panels of two datasets (no \resizebox)
    panels = [DS] if len(DS) <= 3 else [DS[:2], DS[2:]]
    lines += ['\\setlength{\\tabcolsep}{3.2pt}' if len(panels) == 1 else '\\setlength{\\tabcolsep}{5pt}',
              '\\begin{tabular}{l' + 'cc' * len(panels[0]) + '}', '\\toprule']
    for pi, P in enumerate(panels):
        if pi:
            lines.append('\\midrule[\\heavyrulewidth]')
        lines += [' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{n}}}' for _, n in P) + ' \\\\',
                  ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(P))),
                  ] + (['Objective & ' + ' & '.join('R & N' for _ in P) + ' \\\\'] + ['\\midrule'] if pi == 0 else [])
        for k, label in ROWS:
            if k in ('wo_A1', 'Native'):
                lines.append('\\midrule')
            lines.append(f'{label} & ' + ' & '.join(cell(k, d, m) for d, _ in P for m in (0, 1)) + ' \\\\')
            if k == 'XRE':
                lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    (ROOT / 'table_xmask.tex').write_text('\n'.join(lines), encoding='utf-8', newline='\n')

    # macros of the prose
    out = ['% Generated by tables/src/build_table_xmask.py. Do not edit by hand.']

    def mac(n, v):
        out.append(f'\\newcommand{{\\{n}}}{{{v}}}')

    def sgn(x):
        return f'{x:+.4f}'.replace('-', '$-$')
    # the exchange term among the rows with the anchor term (the partner comparison of the prose)
    bestx = [D for d, D in DS if f"{mean[('XRE', d, 0)]:.4f}" == best_anchor[(d, 0)]]
    mac('xmXreBestR', 'every dataset' if len(bestx) == len(DS) else phrase(bestx))
    for k, K in (('mbpr', 'Bpr'), ('mdns', 'Dns'), ('mslk', 'Slk'), ('A1', 'Aone')):
        sigd = [D for d, D in DS if sig.get((k, d, 0))]
        mac(f'xmSig{K}', 'every dataset' if len(sigd) == len(DS) else phrase(sigd))   # 'every dataset' since 2026-10-07
        mac(f'xmNSig{K}', WORDS[len(sigd)])
        for d, D in DS:
            mac(f'xmWin{K}{D}', str(wins[(k, d, 0)]))
            mac(f'xmDiff{K}{D}', sgn(mean[('XRE', d, 0)] - mean[(k, d, 0)]))
    # masked BPR / DNS / SL@K relative to the anchor term alone (R@20), the question whether masking alone helps
    for k, K in (('mbpr', 'Bpr'), ('mdns', 'Dns'), ('mslk', 'Slk')):
        for d, D in DS:
            mac(f'xmOverAone{K}{D}', sgn(mean[(k, d, 0)] - mean[('A1', d, 0)]))
    # share of the exchange gain over the anchor term that masked DNS recovers (R@20, unrounded means, percent)
    share = [100 * (mean[('mdns', d, 0)] - mean[('A1', d, 0)]) / (mean[('XRE', d, 0)] - mean[('A1', d, 0)])
             for d, _ in DS]
    mac('xmDnsShareMin', f'{min(share):.0f}')
    mac('xmDnsShareMax', f'{max(share):.0f}')
    # the ladder of the prose: anchor term over Native, and Ours over Native (R@20)
    for d, D in DS:
        mac(f'xmAoneOverNative{D}', sgn(mean[('A1', d, 0)] - mean[('Native', d, 0)]))
        mac(f'xmXreOverNative{D}', sgn(mean[('XRE', d, 0)] - mean[('Native', d, 0)]))
    # datasets on which both exchange-only rows (masked and full view, no anchor term) exceed XRE in mean R
    exa = [D for d, D in DS if all(mean[(k, d, 0)] > mean[('XRE', d, 0)] for k in ('wo_A1', 'wo_A1_masked'))]
    mac('xmExchAboveXreDs', phrase(exa) if exa else 'none of the datasets')
    mac('xmDsPhrase', phrase([D for _, D in DS]))
    mac('xmNDs', WORDS[len(DS)])
    mac('xmDnsPMin', f"{min(pvals[('mdns', d, 0)] for d, _ in DS):.2f}")
    mac('xmDnsPMax', f"{max(pvals[('mdns', d, 0)] for d, _ in DS):.2f}")

    # Table I (MSCA block) facts of the prose, Recall@20 means over seeds 1-5
    def t1(row, d):
        v = [T1[(row, d, s)] for s in SEEDS if (row, d, s) in T1]
        return st.mean(v) if len(v) == len(SEEDS) else None
    # datasets on which all three top-K losses stay below MSCA
    tk = [D for d, D in DS_ALL if all(t1(r, d) is not None and t1(r, d) < t1('Native', d) for r in ('SL@K', 'PSL', 'SSM'))]
    mac('tkBelowMscaDs', phrase(tk))
    # UltraGCN L_I and the INA-style term supervise co-occurrence neighbors with no partner or one uniform partner:
    # their change over MSCA (difference of the printed means) against the change of the anchor term alone
    nb_ds, nb_vals, nb_exc = [], [], []
    for d, D in DS:
        a_gain = round(mean[('A1', d, 0)], 4) - round(mean[('Native', d, 0)], 4)
        gains = {lab: round(t1(r, d), 4) - round(t1('Native', d), 4)
                 for r, lab in (('UltraGCN-LI', 'UltraGCN $\\mathcal{L}_I$'), ('INA', 'the INA-style term'))}
        if all(g < a_gain - 1e-9 for g in gains.values()):
            nb_ds.append(D)
            nb_vals += list(gains.values())
        else:
            nb_exc += [f'{lab} exceeds the anchor term alone on {D}' for lab, g in gains.items() if g > a_gain + 1e-9]
    mac('nbDs', phrase(nb_ds))
    mac('nbMin', sgn(min(nb_vals)) if nb_vals else '--')
    mac('nbMax', sgn(max(nb_vals)) if nb_vals else '--')
    exc = phrase(nb_exc) if nb_exc else ''
    mac('nbExceptSentence', (exc[0].upper() + exc[1:] + '.') if exc else '')
    (ROOT / 'macros_xmask.tex').write_text('\n'.join(out) + '\n', encoding='utf-8', newline='\n')
    # qualitative statements of the prose, appended to tables/claims_check.txt (lines 'xmask: ...' replaced on re-run)
    cc = ROOT / 'claims_check.txt'
    keep = [x for x in (cc.read_text(encoding='utf-8').splitlines() if cc.exists() else []) if not x.startswith('xmask:')]
    keep += [f'xmask: masked DNS recovers {min(share):.0f}-{max(share):.0f}% of the exchange gain over A (R@20); more than '
             f'half on every dataset: {min(share) > 50}',
             f'xmask: the exchange term has the highest R@20 among the rows with A on {bestx}; Exchange (masked view) above '
             f'XRE in R@20 on {[D for d, D in DS if mean[("wo_A1", d, 0)] > mean[("XRE", d, 0)]]}',
             f'xmask: top-K losses (SL@K, PSL, SSM) below MSCA in R@20 on {tk}',
             f'xmask: UltraGCN L_I and INA-style below the anchor-term gain on {nb_ds} (changes {nb_vals}); exceptions {nb_exc}']
    cc.write_text('\n'.join(keep) + '\n', encoding='utf-8', newline='\n')
    print('wrote table_xmask.tex and macros_xmask.tex; datasets', [D for _, D in DS])
    for k, _ in ROWS:
        print(k, [f"{mean[(k, d, 0)]:.4f}" if (k, d, 0) in mean else '--' for d, _ in DS], [bool(sig.get((k, d, 0))) for d, _ in DS],
              [fam.get((k, d), '') for d, _ in DS])
    print('masked DNS share of the exchange gain (%):', [f'{s:.0f}' for s in share])
    print('top-K losses below MSCA on', tk, '| neighbor plug-ins below A on', nb_ds, nb_vals, '| exceptions', nb_exc)


if __name__ == '__main__':
    main()
