"""revision log 18-2 (MES numbers only, author decision 8): per-user headroom N' with MES as the learned ranker.
Reads tables/src/data/r2null3.json (copy of study/results/r2null3.json, script sha 106df8e3, eight datasets, MicroLens has no
MES run) and tables/src/data/n3p_mes.json (one-sided N3 p per dataset, re-drawn on HOST_A by study/R2NULL3_P18/n3p_mes.py =
tables/src/data/n3p_mes.py, which reproduces the N3 intervals of r2null3.json exactly).
Writes tables/table_headroom.tex (Table tab:headroom) and tables/macros_nprime.tex (prefix nq, descriptive numbers of the
prose). Significance counts of N3 and W (Holm over datasets) are in tables/macros_sig.tex (build_sigcounts.py), and this
script recomputes them with the same rule to print the p columns of the table.
Before 18-2, tab:headroom was typed into text/4_ex/4f_cold_neardup.tex from r2null2.md (E + R + A design, nine datasets).
Usage: python -s tables/src/build_nprime_mes.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sigtest  # noqa: E402
from numwords import k_of_n, num_word  # noqa: E402

T = Path(__file__).resolve().parents[1]
DATA = T / 'src' / 'data'
NAME = {'baby': 'Baby', 'pantry': 'Pantry', 'scientific': 'Scientific', 'sports': 'Sports', 'clothing': 'Clothing',
        'instruments': 'Instruments', 'arts': 'Arts', 'office': 'Office', 'microlens': 'MicroLens'}
TUNED = ['baby', 'pantry', 'scientific', 'sports']
# history-length columns of the table: first training-degree edge of each column (r2null3 strata_edges 1, 2, 3, 4, 6, 10, 20)
COLS = [((1, 2), '1--2'), ((3,), '3'), ((4,), '4--5'), ((6,), '6--9'), ((10,), '10--19'), ((20,), '$\\geq 20$')]


def phrase(names, none='none'):
    if not names:
        return none
    if len(names) == 1:
        return names[0]
    return ', '.join(names[:-1]) + (' and ' if len(names) == 2 else ', and ') + names[-1]


def col_of(edge):
    return next(i for i, (es, _) in enumerate(COLS) if edge in es)


def pct(x, nd=1):
    return f'{100 * x:+.{nd}f}'


def users(n):
    s = f'{n:,}'
    return s.replace(',', '{,}')


def main():
    R = json.load(open(DATA / 'r2null3.json', encoding='utf-8'))
    P = json.load(open(DATA / 'n3p_mes.json', encoding='utf-8'))
    assert P['source_sha'] == R['code_sha256'] == '106df8e3ad91a4265398544d9e626f17e67d462a098eff7a435983a8bf122f6a'
    assert R['B']['N3'] == P['B'] == 200
    D = R['per_ds']
    dss = [d for d in D if D[d].get('status') == 'OK']
    assert 'microlens' not in dss and len(dss) == 8, dss
    for d in dss:
        p = P['per_ds'][d]
        assert p['reproduces_r2null3'] and abs(p['g_O'] - D[d]['g_O']) < 1e-12 and p['n3_cls'] == D[d]['N3']['cls'], d
    # tests (same rule as build_sigcounts.py): Holm over the datasets of each null
    n3_holm = dict(zip(dss, sigtest.holm([P['per_ds'][d]['p_n3'] for d in dss])))
    wds = [d for d in dss if D[d].get('W')]
    w_holm = dict(zip(wds, sigtest.holm([D[d]['W']['p'] for d in wds])))

    # ------------------------------------------------------------- table
    rows = []
    for d in dss:
        v = D[d]
        cells = []
        layers = sorted(v['layers'].values(), key=lambda x: x['edges'][0])
        occupied = {}
        for L in layers:
            cs = sorted({col_of(e) for e in L['edges']})
            assert cs == list(range(cs[0], cs[-1] + 1)), (d, L['edges'])
            occupied[cs[0]] = (len(cs), L)
            for c in cs[1:]:
                occupied[c] = None
        c = 0
        while c < len(COLS):
            if c not in occupied:
                cells.append('--')
                c += 1
                continue
            span, L = occupied[c]
            txt = f"${100 * L['g_O_seed1']:+.0f}$\\%\\,({users(L['n'])})"
            cells.append(txt if span == 1 else f'\\multicolumn{{{span}}}{{c}}{{{txt}}}')
            c += span
        n3 = v['N3']
        p3 = P['per_ds'][d]['p_n3']
        pw = f"{v['W']['p']:.3f}" if v.get('W') else '--'
        mark3 = '$^{\\ast}$' if n3_holm[d] <= sigtest.ALPHA else ''
        markw = '$^{\\ast}$' if d in w_holm and w_holm[d] <= sigtest.ALPHA else ''
        rows.append(f"{NAME[d]} & ${pct(v['g_O'])}$\\% & $[{pct(n3['lo'])}, {pct(n3['hi'])}]$\\% & {p3:.3f}{mark3} & {pw}{markw} & "
                    + ' & '.join(cells) + ' \\\\')
    n_w = len(wds)
    untuned = [NAME[d] for d in dss if d not in TUNED]
    cap = ('Exploratory analysis N$\'$: headroom $g_O$ of the per-user oracle over MES as the learned ranker, by stratum of training history length. '
           'Each stratum cell reports $g_O$ of seed 1 and, in parentheses, its number of users. '
           'The column $g_O$ is the dataset-level headroom over seeds 1--5, and the next columns give the 95\\% interval of the '
           f"same-stratum decoy null N3 ({R['B']['N3']} draws), its one-sided $p$, and the $p$ of the winner-agreement test W "
           f"({R['B']['W']} permutations), with $^{{\\ast}}$ for significance after Holm correction over the datasets of each test. "
           f"W uses the {num_word(n_w)} datasets with validation dumps"
           + (', which are all datasets of this table. ' if n_w == len(dss) else '. ')
           + 'MES on Baby, Pantry, Scientific, and Sports uses validation-selected $\\lambda$ and $K$, whereas '
           + phrase(untuned) + ' use $\\lambda=0.1$ and $K=40$ without tuning. '
           'Strata with fewer than 200 users are merged with a neighbor and span their columns, and a dash marks a history '
           'length absent from the data.')
    ncol = 5 + len(COLS)
    lines = ['% Generated by tables/src/build_nprime_mes.py from tables/src/data/r2null3.json (sha 106df8e3) and n3p_mes.json.',
             '% Do not edit by hand. revision log 18-2: MES as the learned ranker, eight datasets (MicroLens has no MES run).',
             '\\begin{table*}[!t]', f'\\caption{{{cap}}}', '\\label{tab:headroom}', '\\centering', '\\scriptsize',
             '\\setlength{\\tabcolsep}{3.2pt}', '\\begin{tabular}{l' + 'c' * (ncol - 1) + '}', '\\toprule',
             f' & & & & & \\multicolumn{{{len(COLS)}}}{{c}}{{History length (training interactions)}} \\\\',
             f'\\cmidrule(lr){{6-{ncol}}}',
             'Dataset & $g_O$ & N3 interval & N3 $p$ & W $p$ & ' + ' & '.join(lab for _, lab in COLS) + ' \\\\',
             '\\midrule'] + rows + \
            ['\\bottomrule',
             f'\\multicolumn{{{ncol}}}{{l}}{{MicroLens, the ninth dataset of the pool of Section~\\ref{{sec:settings}}, has no MES run and is left out.}} \\\\',
             '\\end{tabular}', '\\end{table*}', '']
    (T / 'table_headroom.tex').write_text('\n'.join(lines), encoding='utf-8', newline='\n')

    # ------------------------------------------------------------- prose macros
    above = [d for d in dss if D[d]['N3']['cls'] == 'above']
    inside = [d for d in dss if D[d]['N3']['cls'] == 'inside']
    below = [d for d in dss if D[d]['N3']['cls'] == 'below']
    g = [D[d]['g_O'] for d in dss]
    r3 = [D[d]['diag']['N3_mean_over_gO'] for d in above]
    r2 = [D[d]['diag']['N2_mean_over_gO'] for d in dss]
    n2_above = [d for d in dss if D[d]['N2']['cls'] == 'above']
    n2_below = [d for d in dss if D[d]['N2']['cls'] == 'below']
    longer = []
    for d in dss:
        L = sorted(D[d]['layers'].values(), key=lambda x: x['edges'][0])
        if L[-1]['g_O_seed1'] > L[0]['g_O_seed1']:
            longer.append(d)
    M = [('nqNDs', num_word(len(dss))),
         ('nqGoMin', f'{100 * min(g):.1f}'), ('nqGoMax', f'{100 * max(g):.1f}'),
         ('nqNAbove', num_word(len(above))), ('nqAboveDs', phrase([NAME[d] for d in above])),
         ('nqNInside', num_word(len(inside))), ('nqInsideDs', phrase([NAME[d] for d in inside])),
         ('nqNBelow', num_word(len(below))), ('nqBelowDs', phrase([NAME[d] for d in below])),
         ('nqNotAboveDs', phrase([NAME[d] for d in dss if d not in above])),
         ('nqNotAboveN', num_word(len(dss) - len(above))),
         ('nqThreeRatioMin', f'{min(r3):.2f}'), ('nqThreeRatioMax', f'{max(r3):.2f}'),
         ('nqTwoRatioMin', f'{min(r2):.1f}'), ('nqTwoRatioMax', f'{max(r2):.1f}'),
         ('nqNTwoBelowOf', k_of_n(len(n2_below), len(dss))), ('nqNTwoAbove', num_word(len(n2_above))),
         ('nqNLonger', num_word(len(longer)) if len(longer) < len(dss) else f'all {num_word(len(dss))}'),
         ('nqLongerExceptDs', phrase([NAME[d] for d in dss if d not in longer])),
         ('nqNWDs', num_word(n_w))]
    out = ['% Generated by tables/src/build_nprime_mes.py from tables/src/data/r2null3.json (MES, eight datasets). Do not edit.',
           '% g_O in percent (one decimal); ratios = mean of a null over g_O (N3: datasets above the N3 interval; N2: all).']
    out += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M]
    (T / 'macros_nprime.tex').write_text('\n'.join(out) + '\n', encoding='utf-8', newline='\n')
    print('\n'.join(out))
    print('N3 holm', {d: round(x, 4) for d, x in n3_holm.items()})
    print('W holm', {d: round(x, 4) for d, x in w_holm.items()})


if __name__ == '__main__':
    main()
