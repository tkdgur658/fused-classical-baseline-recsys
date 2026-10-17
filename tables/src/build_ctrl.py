"""Controlled-component evidence of MES (TMM revision rev-selfmsg-1009): table tab:ctrl, its NDCG@20 twin tab:ctrl_ndcg,
the contrasts of the forest figure fig_ctrl_forest, the prose macros, and the weight-decay fairness macros.
Numbers are never typed by hand. Only reads tables/*.csv (and, with --collect-wd, the HOST_A wdcmp collector).

Component letters (BRIEF_rev1009): E = masked entry pairs (MES), R = retention pairs, A = anchor term (co-occurrence
neighbor supervision), mu = weight of a top-K loss added to BPR. Run names -> rows (one run per seed, seeds 1-5):
  MES          E (MES, proposed)        tables/mes_perseed.csv (backbone MSCA) if present; otherwise PROVISIONAL = wo_A1
                                        per seed, printed as \\tbd{...} (red), and every MES contrast is skipped
  wo_A1        E + R                    tables/xre_perseed.csv, MSCA, n_runs == 1 asserted (as collect_hyp_forest.py)
  xeo          E + A                    tables/novctrl_perseed.csv role new
  XRE          E + R + A                tables/xre_perseed.csv
  xto          R + A                    novctrl_perseed.csv
  xmd          E + R + A, matched DNS competitor     novctrl_perseed.csv
  xph          E + R + A, history-free probes        novctrl_perseed.csv
  wo_A1_masked E + R, full view         xre_perseed.csv
  wo_masked    E + R + A, full view     xre_perseed.csv
  A1           A                        xre_perseed.csv
  mbpr / mdns  A + masked BPR / DNS     tables/xmask_perseed.csv (mslk is loaded for the Holm family of Table III only)
  add_ssm/psl/slk  BPR + mu SSM / PSL / SL@K   novctrl_perseed.csv (mu selected on seed-1 valid R@20, column mu)
  Native       Backbone (MSCA)          tables/table1_perseed.csv
  rep_*        top-K replacement rows of Table I (tables/topk_perseed.csv), macros only
Statistics: paired t-test over seeds 1-5 (scipy ttest_rel), 95% paired t-interval (df 4, unadjusted), wins = seeds with
a > b. revision log 18-3 (single test family, tables/src/sigtest.py): Holm (alpha 0.05, step-down adjusted p) runs per metric
over the four datasets of ONE contrast (a, b) in every family below; the scopes listed below are the family names and the
pre-18-3 Holm scopes (parts 8 cells, additive 12 cells, xmask / mesrows the rows of one dataset), kept for reference:
  matched  XRE - xmd                     over the four datasets          (= novctrl_contrasts.csv family matched)
  parts    XRE - xeo, XRE - xto          over the 8 cells                (= novctrl family parts)
  probe    XRE - xph                     over the four datasets          (= novctrl family probe)
  additive XRE - add_<loss>              over the 12 cells               (= novctrl family additive)
  xmask    XRE - mdns / mslk / mbpr / A1 over the four rows of ONE dataset (= the black-triangle marks of Table III)
  addA     XRE - wo_A1                   over the four datasets
  maskA    XRE - wo_masked               over the four datasets
  maskX    wo_A1 - wo_A1_masked          over the four datasets
  dnsbpr   mdns - mbpr                   over the four datasets
  entret   xeo - xto                     over the four datasets
  (MES present) mesrows  MES - <each G2-G4 row>  over the rows of one dataset;  mesnat  MES - Native  over the datasets
Table marks: dagger / double dagger = significantly below / above E + R + A (families matched, parts, probe, additive,
xmask, addA, maskA); section = E + R, full view significantly below/above E + R (maskX); pilcrow = A + masked DNS
significantly above/below A + masked BPR (dnsbpr); asterisk / circle = significantly below / above E (MES rows; the
Backbone row carries the asterisk when MSCA is significantly below E). The caption names only the marks that occur.
Forest contrasts (a - b, figure ids): (1) maskX, (2) maskA, (3) addA, (4) XRE - xeo (parts), (5) entret, (6) matched,
(7) dnsbpr, (8) best additive - XRE (best = highest mean R@20 per dataset among the three; p and Holm of the additive
family, sign flipped), and with MES data (0) MES - Native, (9) xeo - MES, (10) wo_A1 - MES (mesrows family, sign flipped).
Cross-checks (exit 1 on mismatch): novctrl_contrasts.csv (diff, CI, p, Holm p, reject) for the four novctrl families,
macros_hyp_forest.tex hf<MaskX|MaskA|DnsVsBpr><Mean|CI|P|Pos><DS> for forest (1), (2), (7), the p of (1)/(2) against
ablation_tests.csv, the xmask family against \\xmSig<Bpr|Dns|Aone> of macros_xmask.tex, the ref rows of novctrl_perseed.csv,
and the wd 0 arms of wdcmp_perseed.csv against xre_perseed.csv / table1_perseed.csv.
Writes tables/ctrl_perseed.csv, tables/ctrl_contrasts.csv, tables/table_ctrl.tex, tables/appendix_ctrl_ndcg.tex,
tables/macros_ctrl.tex, tables/macros_wdcmp.tex.
Usage (repository root):  python -s tables/src/build_ctrl.py               (then python -s figures/src/fig_ctrl_forest.py)
                          python -s tables/src/build_ctrl.py --collect-wd  (first refresh tables/wdcmp_perseed.csv on HOST_A)
"""
import csv
import math
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numwords import k_of_n, num_word  # noqa: E402
import sigtest  # noqa: E402  (revision log 18-3: one Holm family = one contrast over the four datasets)

T = Path(__file__).resolve().parents[1]
DS_ALL = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
DS = [d for d, _ in DS_ALL]
NAME = dict(DS_ALL)
SEEDS = [1, 2, 3, 4, 5]
ALPHA = 0.05
METRICS = [(0, 'R@20'), (1, 'N@20')]
LOSSES = [('ssm', 'SSM'), ('psl', 'PSL'), ('slk', 'SL@$K$')]
GROUPS = [
    # revision log 18-2 (author decision 7): the MES row = the validation-selected MES run of Table tab:sota (row MES of
    # mes_perseed.csv, no marks); row E = MES at the fixed lambda 0.1, K 40 of the E + R + A design (MES_k40), the reference
    # of the asterisk / circle marks. Before 18-2 the table had row E alone, labelled 'E (MES, proposed)'.
    [('MESsel', 'MES (validation-selected)'), ('MES', 'E (MES, $\\lambda=0.1$, $K=40$)')],
    [('wo_A1', 'E + R'), ('xeo', 'E + A'), ('XRE', 'E + R + A'), ('xto', 'R + A'),
     ('xmd', 'E + R + A, matched DNS competitor'), ('xph', 'E + R + A, history-free probes')],
    [('wo_A1_masked', 'E + R, full view'), ('wo_masked', 'E + R + A, full view'), ('A1', 'A'),
     ('mbpr', 'A + masked BPR'), ('mdns', 'A + masked DNS')],
    [(f'add_{k}', f'BPR + $\\mu$ {L}') for k, L in LOSSES],
    [('Native', 'Backbone (MSCA)')],
]
ROWS = [r for g in GROUPS for r in g]
LABEL = dict(ROWS)
MES_CMP = [k for g in GROUPS[1:4] for k, _ in g]        # rows tested against MES once MES data exist
WD_CMD = ('cd <RUN_ROOT_B>/P04_wdcmp && CUDA_VISIBLE_DEVICES= nice -n 15 python3 collect.py --metric both '
          '--csv /tmp/wdcmp_ctrl_$$.csv >/dev/null && cat /tmp/wdcmp_ctrl_$$.csv && rm -f /tmp/wdcmp_ctrl_$$.csv')


# ------------------------------------------------------------------ data
def load():
    V, SRC = {}, {}

    def put(row, d, s, r20, n20, src):
        key = (row, d, int(s))
        assert key not in V, f'duplicate run {key} (one run per seed, no replicates)'
        V[key] = (float(r20), float(n20))
        SRC[key] = src
    for r in csv.DictReader(open(T / 'xre_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] in ('XRE', 'A1', 'wo_A1', 'wo_A1_masked', 'wo_masked') and r['dataset'] in DS:
            assert int(r['n_runs']) == 1, r
            put(r['row'], r['dataset'], r['seed'], r['r20'], r['n20'], 'xre_perseed.csv')
    for r in csv.DictReader(open(T / 'xmask_perseed.csv', encoding='utf-8')):
        if r['row'] in ('mbpr', 'mdns', 'mslk') and r['dataset'] in DS:
            put(r['row'], r['dataset'], r['seed'], r['r20'], r['n20'], 'xmask_perseed.csv')
    for r in csv.DictReader(open(T / 'table1_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] == 'Native' and r['dataset'] in DS:
            put('Native', r['dataset'], r['seed'], r['r20'], r['n20'], 'table1_perseed.csv')
    inv = {'SSM': 'ssm', 'PSL': 'psl', 'SL@K': 'slk'}
    for r in csv.DictReader(open(T / 'topk_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['dataset'] in DS:
            put('rep_' + inv[r['row']], r['dataset'], r['seed'], r['r20'], r['n20'], 'topk_perseed.csv')
    MU, refs = {}, []
    for r in csv.DictReader(open(T / 'novctrl_perseed.csv', encoding='utf-8')):
        if r['role'] == 'new':
            put(r['row'], r['dataset'], r['seed'], r['test_r20'], r['test_n20'], 'novctrl_perseed.csv')
            if r['row'].startswith('add_'):
                MU.setdefault((r['row'], r['dataset']), set()).add(float(r['mu']))
        elif r['role'] == 'ref':
            refs.append(r)
    # the reference rows that collect_novctrl.py copied must equal the paper CSVs read here
    for r in refs:
        key = (r['row'] if not r['row'].startswith('rep_') else r['row'], r['dataset'], int(r['seed']))
        if key[0] == 'wo_retention':
            continue
        assert key in V and abs(V[key][0] - float(r['test_r20'])) < 1e-9 and abs(V[key][1] - float(r['test_n20'])) < 1e-9, \
            f'novctrl ref row differs from the paper CSV: {key}'
    for k, v in MU.items():
        assert len(v) == 1, (k, v)
    MU = {k: v.pop() for k, v in MU.items()}
    mes_file = T / 'mes_perseed.csv'
    provisional = not mes_file.exists()
    if provisional:
        for d in DS:
            for s in SEEDS:
                V[('MES', d, s)] = V[('wo_A1', d, s)]
                SRC[('MES', d, s)] = 'PROVISIONAL = wo_A1 (E + R)'
    else:
        rs = [r for r in csv.DictReader(open(mes_file, encoding='utf-8')) if r['backbone'] == 'MSCA' and r['dataset'] in DS]
        names = {r['row'] for r in rs}
        # 2026-10-10 (r20_R1/MANU): every row of this table uses lambda 0.1, K 40, so the MES row is MES_k40 (P04_xen0)
        # when present; the main-table MES (row MES) may use the valid-selected lambda / K per dataset.
        keep = 'MES_k40' if 'MES_k40' in names else 'MES'
        assert keep in names, f'mes_perseed.csv: no MES row among {names}'
        for r in rs:
            if r['row'] == keep:
                put('MES', r['dataset'], r['seed'], r['r20'], r['n20'], f'mes_perseed.csv row {keep}')
            if r['row'] == 'MES':   # 18-2: validation-selected MES (Table tab:sota), shown without marks
                put('MESsel', r['dataset'], r['seed'], r['r20'], r['n20'], 'mes_perseed.csv row MES')
        # control "E, full view" (xres0f, MES with entry pairs on the full view): a row of the full-view group once complete
        fv = [r for r in rs if r['row'] == 'MES_fv']
        if {(r['dataset'], int(r['seed'])) for r in fv} == {(d, s) for d in DS for s in SEEDS}:
            for r in fv:
                put('MES_fv', r['dataset'], r['seed'], r['r20'], r['n20'], 'mes_perseed.csv row MES_fv')
            if ('MES_fv', 'E, full view') not in GROUPS[2]:
                GROUPS[2].insert(0, ('MES_fv', 'E, full view'))
                ROWS[:] = [x for g in GROUPS for x in g]
                LABEL.update(dict(ROWS))
                MES_CMP[:] = [k for g in GROUPS[1:4] for k, _ in g]
    for k, _ in ROWS + [('mslk', '')]:
        for d in DS:
            for s in SEEDS:
                assert (k, d, s) in V, f'missing {k} {d} seed {s}'
    return V, SRC, MU, provisional


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def paired(V, a, b, d, m):
    x = [V[(a, d, s)][m] for s in SEEDS]
    y = [V[(b, d, s)][m] for s in SEEDS]
    dd = [p - q for p, q in zip(x, y)]
    mu, sd = st.mean(dd), st.stdev(dd)
    se = sd / math.sqrt(len(dd))
    h = stats.t.ppf(0.975, len(dd) - 1) * se
    p = float(stats.ttest_rel(x, y).pvalue) if sd > 0 else float(mu == 0)
    return dict(n=len(dd), mean_a=st.mean(x), mean_b=st.mean(y), diff=mu, sd=sd, ci_lo=mu - h, ci_hi=mu + h,
                t=mu / se if se > 0 else 0.0, p=p, wins_a=sum(1 for p_, q_ in zip(x, y) if p_ > q_), d=dd)


def families(provisional):
    """family -> (list of (a, b) pairs, Holm scope: 'all' = every (pair, dataset) cell, 'dataset' = pairs of one dataset)."""
    F = {'matched': ([('XRE', 'xmd')], 'all'), 'parts': ([('XRE', 'xeo'), ('XRE', 'xto')], 'all'),
         'probe': ([('XRE', 'xph')], 'all'), 'additive': ([('XRE', f'add_{k}') for k, _ in LOSSES], 'all'),
         'xmask': ([('XRE', 'mdns'), ('XRE', 'mslk'), ('XRE', 'mbpr'), ('XRE', 'A1')], 'dataset'),
         'addA': ([('XRE', 'wo_A1')], 'all'), 'maskA': ([('XRE', 'wo_masked')], 'all'),
         'maskX': ([('wo_A1', 'wo_A1_masked')], 'all'), 'dnsbpr': ([('mdns', 'mbpr')], 'all'),
         'entret': ([('xeo', 'xto')], 'all')}
    if not provisional:
        F['mesrows'] = ([('MES', k) for k in MES_CMP], 'dataset')
        F['mesnat'] = ([('MES', 'Native')], 'all')
    return F


def contrasts(V, provisional):
    """revision log 18-3 (single test family, tables/src/sigtest.py): every (a, b) pair is its own Holm family over the four
    datasets of one metric, whatever the scope of families() says. Before 18-3, 'all' pooled every pair of a family
    (parts: 8 cells, additive: 12 cells) and 'dataset' pooled the pairs of one dataset (xmask, mesrows)."""
    C = {}   # (family, a, b, dataset, metric index) -> result with p_holm
    for fam, (pairs, scope) in families(provisional).items():
        for m, _ in METRICS:
            for a, b in pairs:
                g = [(a, b, d) for d in DS]
                res = [paired(V, a, b, d, m) for a, b, d in g]
                for (a, b, d), r, pa in zip(g, res, sigtest.holm([r['p'] for r in res])):
                    r['p_holm'] = pa
                    r['reject'] = int(pa <= ALPHA)
                    C[(fam, a, b, d, m)] = r
    return C


def flip(r):
    return dict(r, mean_a=r['mean_b'], mean_b=r['mean_a'], diff=-r['diff'], ci_lo=-r['ci_hi'], ci_hi=-r['ci_lo'],
                t=-r['t'], wins_a=r['n'] - r['wins_a'] - sum(1 for x in r['d'] if x == 0), d=[-x for x in r['d']])


def forest(V, C, provisional):
    """Forest rows: list of (fid, group, label, a, b, family, dataset, result) for Recall@20."""
    out = []
    spec = [(1, 'Masking', 'maskX', 'wo_A1', 'wo_A1_masked', False), (2, 'Masking', 'maskA', 'XRE', 'wo_masked', False),
            (3, 'Parts', 'addA', 'XRE', 'wo_A1', False), (4, 'Parts', 'parts', 'XRE', 'xeo', False),
            (5, 'Parts', 'entret', 'xeo', 'xto', False), (6, 'Competitor', 'matched', 'XRE', 'xmd', False),
            (7, 'Competitor', 'dnsbpr', 'mdns', 'mbpr', False)]
    if not provisional:
        spec = [(0, 'MES', 'mesnat', 'MES', 'Native', False), (9, 'MES', 'mesrows', 'MES', 'xeo', True),
                (10, 'MES', 'mesrows', 'MES', 'wo_A1', True)] + spec
    for fid, grp, fam, a, b, fl in spec:
        for d in DS:
            r = C[(fam, a, b, d, 0)]
            out.append((fid, grp, (b, a) if fl else (a, b), fam, d, flip(r) if fl else r))
    for d in DS:
        best = max((f'add_{k}' for k, _ in LOSSES), key=lambda k: st.mean(V[(k, d, s)][0] for s in SEEDS))
        out.append((8, 'Top-K', (best, 'XRE'), 'additive', d, flip(C[('additive', 'XRE', best, d, 0)])))
    return out


# ------------------------------------------------------------------ formatting helpers
def sgn(x):
    s = f'{x:+.4f}'
    if s in ('+0.0000', '-0.0000'):
        return '0.0000'
    return s.replace('-', '$-$')


def phrase(names, none='none of the datasets'):
    if not names:
        return none
    if len(names) == 1:
        return names[0]
    return ', '.join(names[:-1]) + (' and ' if len(names) == 2 else ', and ') + names[-1]


def pct(x, signed=False):
    s = f'{x:+.1f}' if signed else f'{x:.1f}'
    return s.replace('-', '$-$') + '\\%'


def read_macros(path):
    return dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}', path.read_text(encoding='utf-8')))


# ------------------------------------------------------------------ table
MARKS = {'below_xre': '\\dagger', 'above_xre': '\\ddagger', 'maskx': '\\S', 'dnsbpr': '\\P',
         'below_mes': '\\ast', 'above_mes': '\\circ'}
HOLM_NOTES = []   # revision log 18-3: Holm decisions that differ from the pre-18-3 family files (printed, never an error)


def marks_for(C, m, provisional):
    """(row, dataset) -> list of mark keys for metric m."""
    M = {}

    def add(k, d, key):
        M.setdefault((k, d), []).append(key)
    vs_xre = [('matched', 'xmd'), ('parts', 'xeo'), ('parts', 'xto'), ('probe', 'xph'), ('xmask', 'mdns'),
              ('xmask', 'mbpr'), ('xmask', 'A1'), ('addA', 'wo_A1'), ('maskA', 'wo_masked')] + \
             [('additive', f'add_{k}') for k, _ in LOSSES]
    for d in DS:
        for fam, k in vs_xre:
            r = C[(fam, 'XRE', k, d, m)]
            if r['reject']:
                add(k, d, 'below_xre' if r['diff'] > 0 else 'above_xre')
        r = C[('maskX', 'wo_A1', 'wo_A1_masked', d, m)]
        if r['reject']:
            add('wo_A1_masked', d, ('maskx', 'below' if r['diff'] > 0 else 'above'))
        r = C[('dnsbpr', 'mdns', 'mbpr', d, m)]
        if r['reject']:
            add('mdns', d, ('dnsbpr', 'above' if r['diff'] > 0 else 'below'))
        if not provisional:
            for k in MES_CMP:
                r = C[('mesrows', 'MES', k, d, m)]
                if r['reject']:
                    add(k, d, 'below_mes' if r['diff'] > 0 else 'above_mes')
            r = C[('mesnat', 'MES', 'Native', d, m)]
            if r['reject']:
                add('Native', d, 'below_mes' if r['diff'] > 0 else 'above_mes')
    return M


def table(V, C, provisional, m, label, place):
    metric = 'Recall@20' if m == 0 else 'NDCG@20'
    mean = {(k, d): st.mean(V[(k, d, s)][m] for s in SEEDS) for k, _ in ROWS for d in DS}
    best = {d: max(f'{mean[(k, d)]:.4f}' for k, _ in ROWS if not (k == 'MES' and provisional)) for d in DS}
    M = marks_for(C, m, provisional)
    used = set()

    def cell(k, d):
        v = f'{mean[(k, d)]:.4f}'
        body = v.replace('0.', '.', 1)
        if k == 'MES' and provisional:
            return f'\\tbd{{{body}}}'
        if v == best[d]:
            body = f'\\textbf{{{body}}}'
        ms = []
        for key in M.get((k, d), []):
            sym = MARKS[key[0] if isinstance(key, tuple) else key]
            used.add(key)
            ms.append(sym)
        if ms:
            body += '\\rlap{$^{' + ''.join(ms) + '}$}'
        return body
    body = []
    for gi, g in enumerate(GROUPS):
        if gi:
            body.append('\\midrule')
        for k, lab in g:
            body.append(f'{lab} & ' + ' & '.join(cell(k, d) for d in DS) + ' \\\\')
    # caption: letters, cell content, marks that occur, Holm families, provisional note
    leg = []
    if used & {'below_xre', 'above_xre'}:
        leg.append('$^{\\dagger}$/$^{\\ddagger}$ for a row significantly below/above E + R + A')
    for kind, sym, what in (('maskx', '\\S', 'E + R, full view significantly {} E + R'),
                            ('dnsbpr', '\\P', 'A + masked DNS significantly {} A + masked BPR')):
        dirs = sorted({key[1] for key in used if isinstance(key, tuple) and key[0] == kind})
        if dirs:
            leg.append(f'$^{{{sym}}}$ for ' + what.format('/'.join(dirs)))
    if used & {'below_mes', 'above_mes'}:
        leg.append('$^{\\ast}$/$^{\\circ}$ for a row significantly below/above E')
    # revision log 18-2 (author decision 7): caption begins with the scope; before: 'Controlled variants of MES on MSCA. '
    cap = ('Controlled variants of the E + R + A design; the MES row is the validation-selected MES run of '
           'Table~\\ref{tab:sota}. '
           'E marks the masked entry pairs, R the retention pairs, A the anchor term, and $\\mu$ the weight of a top-$K$ '
           'loss added to BPR, all on MSCA. ')
    cap += (f'Each cell reports the mean test {metric} over seeds 1--5, one run per seed, with the best mean of each '
            'column in bold. ' if m == 0 else
            f'Each cell reports the mean test {metric} over seeds 1--5 for the runs of Table~\\ref{{tab:ctrl}}, with '
            'the best mean of each column in bold. ')
    if m == 0:   # revision log 14-c / 18-2: row E is not the validation-selected MES row
        cap += ('Row E applies MES with the fixed $\\lambda$ and $K$ of the E + R + A design and reuses those runs, thus it is '
                'not interchangeable with the MES row. ')
    if leg:
        # revision log 18-3: the rule of tables/src/sigtest.py, stated once per caption
        cap += ('Two-sided paired $t$-tests over seeds 1--5, Holm-corrected over the four datasets of each contrast, give '
                + phrase(leg, '') + '. ')
    cap += 'Supplementary Section~\\ref{supp:ctrl} defines the variants.'
    if provisional:
        cap += (' \\tbd{Red E values are provisional copies of E + R until the MES runs of seeds 1--5 are '
                'collected.}')
    lines = [f'% Generated by tables/src/build_ctrl.py ({metric}). Do not edit numbers by hand.',
             f'% Sources: xre_perseed.csv, novctrl_perseed.csv, xmask_perseed.csv, table1_perseed.csv'
             + ('' if provisional else ', mes_perseed.csv') + '.',
             f'\\begin{{table}}[{place}]', f'\\caption{{{cap}}}', f'\\label{{{label}}}', '\\centering', '\\footnotesize',
             '\\setlength{\\tabcolsep}{3.3pt}', '\\begin{tabular}{lcccc}', '\\toprule',   # 4pt was 1.8pt too wide
             'Objective & ' + ' & '.join(NAME[d] for d in DS) + ' \\\\', '\\midrule'] + body + \
            ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(lines), mean, M


# ------------------------------------------------------------------ macros
def macros_ctrl(V, C, MU, provisional, mean):
    out = ['% Generated by tables/src/build_ctrl.py. Do not edit by hand. Differences are a - b of mean test Recall@20',
           '% over seeds 1-5 (one run per seed); "Sig" = Holm-corrected paired t-test, alpha 0.05 (families in build_ctrl.py).']

    def mac(n, v):
        assert re.fullmatch(r'[A-Za-z]+', n), n
        out.append(f'\\newcommand{{\\{n}}}{{{v}}}')
    R = lambda fam, a, b, d: C[(fam, a, b, d, 0)]   # noqa: E731
    # matched DNS competitor
    rs = {d: R('matched', 'XRE', 'xmd', d) for d in DS}
    sig = [NAME[d] for d in DS if rs[d]['reject'] and rs[d]['diff'] > 0]
    mac('ctMatchedDsSig', phrase(sig)); mac('ctMatchedNSig', num_word(len(sig)))
    mac('ctMatchedMin', sgn(min(r['diff'] for r in rs.values()))); mac('ctMatchedMax', sgn(max(r['diff'] for r in rs.values())))
    # adding R: (E + R + A) - (E + A)
    rs = {d: R('parts', 'XRE', 'xeo', d) for d in DS}
    mac('ctAddRMin', sgn(min(r['diff'] for r in rs.values()))); mac('ctAddRMax', sgn(max(r['diff'] for r in rs.values())))
    mac('ctAddRNSig', num_word(sum(r['reject'] for r in rs.values())))
    # entry vs retention: (E + A) - (R + A)
    rs = {d: R('entret', 'xeo', 'xto', d) for d in DS}
    mac('ctEntryRetMin', sgn(min(r['diff'] for r in rs.values()))); mac('ctEntryRetMax', sgn(max(r['diff'] for r in rs.values())))
    mac('ctEntryRetNGe', num_word(sum(mean[('xeo', d)] >= mean[('xto', d)] for d in DS)))
    rs = {d: R('parts', 'XRE', 'xto', d) for d in DS}
    mac('ctXreRetDsSig', phrase([NAME[d] for d in DS if rs[d]['reject']]))
    # adding A: (E + R + A) - (E + R)
    rs = {d: R('addA', 'XRE', 'wo_A1', d) for d in DS}
    mac('ctAddAMin', sgn(min(r['diff'] for r in rs.values()))); mac('ctAddAMax', sgn(max(r['diff'] for r in rs.values())))
    mac('ctAddADsUp', phrase([NAME[d] for d in DS if rs[d]['reject'] and rs[d]['diff'] > 0]))
    mac('ctAddADsDown', phrase([NAME[d] for d in DS if rs[d]['reject'] and rs[d]['diff'] < 0]))
    # additive top-K losses
    cells = [(k, d) for k, _ in LOSSES for d in DS]
    tk = [R('additive', 'XRE', f'add_{k}', d) for k, d in cells]
    mac('ctTopkNCells', k_of_n(sum(r['diff'] > 0 for r in tk), len(tk)))
    mac('ctTopkMin', sgn(min(r['diff'] for r in tk))); mac('ctTopkMax', sgn(max(r['diff'] for r in tk)))
    mac('ctTopkBelowNativeN', num_word(sum(mean[(f'add_{k}', d)] <= mean[('Native', d)] for k, d in cells)))
    rep = {(k, d): st.mean(V[(f'rep_{k}', d, s)][0] for s in SEEDS) for k, d in cells}
    mac('ctTopkVsRepUpN', num_word(sum(mean[(f'add_{k}', d)] > rep[(k, d)] for k, d in cells)))
    # history-free probes: (xph) - (E + R + A)
    rs = {d: R('probe', 'XRE', 'xph', d) for d in DS}
    mac('ctProbeMin', sgn(min(-r['diff'] for r in rs.values()))); mac('ctProbeMax', sgn(max(-r['diff'] for r in rs.values())))
    mac('ctProbeNSig', num_word(sum(r['reject'] for r in rs.values())))
    # anchor collision (training-free probe 5a): seed means
    p5 = {r['dataset']: r for r in csv.DictReader(open(T / 'novctrl_probe5a.csv', encoding='utf-8')) if r['seed'] == 'mean'}
    hist = [100 * float(p5[d]['hist']) for d in DS]
    base = [100 * float(p5[d]['probe_hist']) for d in DS]
    mac('ctCollisionMin', f'{min(hist):.1f}'); mac('ctCollisionMax', f'{max(hist):.1f}')
    mac('ctCollisionBaseMin', f'{min(base):.2f}'); mac('ctCollisionBaseMax', f'{max(base):.2f}')
    # selected mu per loss (Baby/Pantry/Scientific/Sports)
    mus = [('/'.join(f"{MU[(f'add_{k}', d)]:.1f}" for d in DS) + f' for {L}') for k, L in LOSSES]
    mac('ctMuList', phrase(mus, ''))
    # MES
    mac('ctMesProvisional', '\\tbd{provisional}' if provisional else '')
    wrap = (lambda s: f'\\tbd{{{s}}}') if provisional else (lambda s: s)   # noqa: E731
    gain = [100 * (mean[('MES', d)] - mean[('Native', d)]) / mean[('Native', d)] for d in DS]
    up = [sum(V[('MES', d, s)][0] > V[('Native', d, s)][0] for s in SEEDS) for d in DS]
    mac('ctMesGainMin', wrap(pct(min(gain)))); mac('ctMesGainMax', wrap(pct(max(gain))))
    mac('ctMesSeedsUp', wrap(phrase([num_word(u) for u in up], '') + f' of {num_word(len(SEEDS))} seeds'))
    # MES on the masked vs the full view (row MES_fv, xres0f; r20_R1/MANU 2026-10-10): R@20 difference x 1e3 per dataset,
    # Holm over the rows of one dataset (family mesrows); tbd placeholders until the 20 runs exist
    if ('mesrows', 'MES', 'MES_fv', DS[0], 0) in C:
        fvr = [C[('mesrows', 'MES', 'MES_fv', d, 0)] for d in DS]
        mac('ctMesFvDiffList', phrase([f"{1e3 * r['diff']:+.1f}".replace('-', '$-$') for r in fvr], ''))
        mac('ctMesFvWinsList', phrase([num_word(r['wins_a']) for r in fvr], '') + f' of {num_word(len(SEEDS))} seeds')
        sig = [NAME[d] for d, r in zip(DS, fvr) if r['reject'] and r['diff'] > 0]
        mac('ctMesFvSigDs', phrase(sig, 'none') if sig else 'none')
        mac('ctMesFvNSig', num_word(len(sig)))
    else:
        for n in ('ctMesFvDiffList', 'ctMesFvWinsList', 'ctMesFvSigDs', 'ctMesFvNSig'):
            mac(n, '\\tbd{pending}')
    return '\n'.join(out) + '\n', dict(gain=gain, up=up)


def macros_wd():
    W = {}
    for r in csv.DictReader(open(T / 'wdcmp_perseed.csv', encoding='utf-8')):
        key = (r['arm'], r['dataset'], int(r['seed']))
        assert key not in W, f'duplicate wdcmp run {key} (one run per seed)'
        W[key] = (float(r['r20']), float(r['n20']))
    for a in ('MSCA_wd0', 'MSCA_wd', 'XRE_wd0', 'XRE_wd'):
        for d in DS:
            for s in SEEDS:
                assert (a, d, s) in W, f'wdcmp missing {a} {d} s{s}'
    m = lambda a, d: st.mean(W[(a, d, s)][0] for s in SEEDS)   # noqa: E731
    rel = lambda a, b, d: 100 * (m(a, d) - m(b, d)) / m(b, d)  # noqa: E731
    seeds = lambda a, b, d: sum(W[(a, d, s)][0] > W[(b, d, s)][0] for s in SEEDS)   # noqa: E731
    out = ['% Generated by tables/src/build_ctrl.py from tables/wdcmp_perseed.csv (HOST_A P04_wdcmp/collect.py; one run',
           '% per seed, seeds 1-5). Relative change = 100 (mean a - mean b) / mean b of test Recall@20, lists in the order',
           '% Baby, Pantry, Scientific, Sports. _wd = decoupled AdamW weight decay 0.1, _wd0 = the official runs (wd 0).']

    def mac(n, v):
        out.append(f'\\newcommand{{\\{n}}}{{{v}}}')
    g = [rel('XRE_wd', 'MSCA_wd', d) for d in DS]
    mac('wdGainList', phrase([pct(x, signed=min(g) < 0) for x in g], ''))
    mac('wdSeedsList', phrase([num_word(seeds('XRE_wd', 'MSCA_wd', d)) for d in DS], '') + ' of five seeds')
    nat = [rel('MSCA_wd', 'MSCA_wd0', d) for d in DS]
    mac('wdNativeList', phrase([pct(x, signed=True) for x in nat], ''))
    ovt = [rel('XRE_wd0', 'MSCA_wd', d) for d in DS]
    mac('wdOursVsTunedList', phrase([pct(x, signed=min(ovt) < 0) for x in ovt], ''))
    mac('wdOursVsTunedSeedsList', phrase([num_word(seeds('XRE_wd0', 'MSCA_wd', d)) for d in DS], '') + ' of five seeds')
    # MES with weight decay (r20_R1/MANU 2026-10-10): rows MES_wd (xres0 <ds>_xres_xeoa0_l010_k40_wd01) and MES (wd 0) of
    # tables/mes_perseed.csv (collect_mes.py), against MSCA_wd of wdcmp_perseed.csv; \tbd{pending} until seeds 1-5 exist
    # on all four datasets. wdMesGainList = MES_wd vs MSCA_wd (same wd 0.1), wdMesVsTunedList = MES (wd 0) vs MSCA_wd,
    # wdMesSelfList = MES_wd vs MES (effect of weight decay on MES itself).
    mp = T / 'mes_perseed.csv'
    for r in (csv.DictReader(open(mp, encoding='utf-8')) if mp.exists() else []):
        if r['backbone'] == 'MSCA' and r['row'] in ('MES', 'MES_wd'):
            key = (r['row'], r['dataset'], int(r['seed']))
            assert key not in W, f'duplicate MES run {key} (one run per seed)'
            assert int(r['n_runs']) == 1, r
            W[key] = (float(r['r20']), float(r['n20']))
    mes_ok = all((a, d, s) in W for a in ('MES', 'MES_wd') for d in DS for s in SEEDS)
    out.append('% MES rows: tables/mes_perseed.csv (MES = P04_xen0, MES_wd = r20/MANU runs): '
               + ('complete' if mes_ok else 'PENDING (tbd placeholders)'))
    mw = {}
    if mes_ok:
        mw['gain'] = [rel('MES_wd', 'MSCA_wd', d) for d in DS]
        mw['ovt'] = [rel('MES', 'MSCA_wd', d) for d in DS]
        mw['self'] = [rel('MES_wd', 'MES', d) for d in DS]
        mw['seeds'] = [seeds('MES_wd', 'MSCA_wd', d) for d in DS]
        mw['ovt_seeds'] = [seeds('MES', 'MSCA_wd', d) for d in DS]
        mac('wdMesGainList', phrase([pct(x, signed=min(mw['gain']) < 0) for x in mw['gain']], ''))
        mac('wdMesSeedsList', phrase([num_word(x) for x in mw['seeds']], '') + ' of five seeds')
        mac('wdMesVsTunedList', phrase([pct(x, signed=min(mw['ovt']) < 0) for x in mw['ovt']], ''))
        mac('wdMesVsTunedSeedsList', phrase([num_word(x) for x in mw['ovt_seeds']], '') + ' of five seeds')
        mac('wdMesSelfList', phrase([pct(x, signed=True) for x in mw['self']], ''))
    else:
        for n in ('wdMesGainList', 'wdMesSeedsList', 'wdMesVsTunedList', 'wdMesVsTunedSeedsList', 'wdMesSelfList'):
            mac(n, '\\tbd{pending}')
    return '\n'.join(out) + '\n', W, dict(mes=mw, gain=g, nat=nat, ovt=ovt,
                                           seeds=[seeds('XRE_wd', 'MSCA_wd', d) for d in DS],
                                           ovt_seeds=[seeds('XRE_wd0', 'MSCA_wd', d) for d in DS])


# ------------------------------------------------------------------ cross-checks
def checks(V, C, W):
    bad = []
    nov = {}
    for r in csv.DictReader(open(T / 'novctrl_contrasts.csv', encoding='utf-8')):
        nov[(r['family'], r['contrast'], r['dataset'], r['metric'])] = r
    n_ok = 0
    for fam, pairs in (('matched', [('XRE', 'xmd')]), ('parts', [('XRE', 'xeo'), ('XRE', 'xto')]),
                       ('probe', [('XRE', 'xph')]), ('additive', [('XRE', f'add_{k}') for k, _ in LOSSES])):
        for a, b in pairs:
            for d in DS:
                for m, mn in METRICS:
                    ref = nov.get((fam, f'{a} - {b}', d, mn))
                    me = C[(fam, a, b, d, m)]
                    if ref is None:
                        bad.append(f'novctrl_contrasts.csv lacks {fam} {a} - {b} {d} {mn}')
                        continue
                    for k in ('diff', 'ci_lo', 'ci_hi', 'p'):
                        if abs(round(me[k], 6) - float(ref[k])) > 1.5e-6:
                            bad.append(f'{fam} {a}-{b} {d} {mn} {k}: {me[k]:.6g} vs novctrl {ref[k]}')
                    if me['wins_a'] != int(ref['wins_a']):
                        bad.append(f'{fam} {a}-{b} {d} {mn}: wins {me["wins_a"]} vs {ref["wins_a"]}')
                    # revision log 18-3: novctrl_contrasts.csv keeps the pre-18-3 Holm families (parts 8 cells, additive 12
                    # cells), so a different Holm p or decision is reported as a note, not as a mismatch
                    if abs(round(me['p_holm'], 6) - float(ref['p_holm'])) > 1.5e-6 or me['reject'] != int(ref['holm_reject']):
                        HOLM_NOTES.append(f'{fam} {a}-{b} {d} {mn}: p_holm {float(ref["p_holm"]):.4g} -> {me["p_holm"]:.4g}, '
                                          f'reject {ref["holm_reject"]} -> {me["reject"]} (novctrl family -> one contrast over datasets)')
                    n_ok += 1
    print(f'check novctrl_contrasts.csv: {n_ok} cells compared (diff, CI, p, Holm p, reject, wins)')
    hf = read_macros(T / 'macros_hyp_forest.tex')
    fmt = lambda v: f'{v:+.4f}'.replace('-', '$-$')   # noqa: E731  (the format of collect_hyp_forest.py)
    for name, fam, a, b in (('MaskX', 'maskX', 'wo_A1', 'wo_A1_masked'), ('MaskA', 'maskA', 'XRE', 'wo_masked'),
                            ('DnsVsBpr', 'dnsbpr', 'mdns', 'mbpr')):
        for d in DS:
            r, D = C[(fam, a, b, d, 0)], NAME[d]
            mine = {'Mean': fmt(r['diff']), 'CI': f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]",
                    'P': f"{r['p']:.3f}" if r['p'] >= 0.001 else '$<$0.001', 'Pos': f"{r['wins_a']}/5"}
            for k, v in mine.items():
                if hf.get(f'hf{name}{k}{D}') != v:
                    bad.append(f'hf{name}{k}{D}: {hf.get(f"hf{name}{k}{D}")} vs {v}')
    print('check macros_hyp_forest.tex: hfMaskX / hfMaskA / hfDnsVsBpr Mean, CI, P, Pos on four datasets')
    abl = {(r['contrast'], r['dataset']): r for r in csv.DictReader(open(T / 'ablation_tests.csv', encoding='utf-8'))}
    notes = []
    for fam, c, a, b in (('maskX', 'masked', 'wo_A1', 'wo_A1_masked'), ('maskA', 'fullxre', 'XRE', 'wo_masked'),
                         ('addA', 'anchor', 'XRE', 'wo_A1')):
        for d in DS:
            r, q = C[(fam, a, b, d, 0)], abl[(c, d)]
            if abs(r['p'] - float(q['p'])) > 1e-3 * max(float(q['p']), 1e-6):
                bad.append(f'p {fam} {d}: {r["p"]:.4g} vs ablation_tests {q["p"]}')
            if r['reject'] != int(q['holm_rejected']):
                notes.append(f'{fam} ({c}) {NAME[d]}: Holm over the four datasets p_holm {r["p_holm"]:.4f} -> reject '
                             f'{r["reject"]}; ablation_tests.csv ({q["family"]} family, Holm over three contrasts of one '
                             f'dataset) reject {q["holm_rejected"]}')
    xm = read_macros(T / 'macros_xmask.tex')
    for k, K in (('mbpr', 'Bpr'), ('mdns', 'Dns'), ('A1', 'Aone'), ('mslk', 'Slk')):
        sigd = [NAME[d] for d in DS if C[('xmask', 'XRE', k, d, 0)]['reject'] and C[('xmask', 'XRE', k, d, 0)]['diff'] > 0]
        v = 'every dataset' if len(sigd) == len(DS) else phrase(sigd)
        if xm.get(f'xmSig{K}') != v:
            # revision log 18-3: macros_xmask.tex (parked Table tab:xmask, not compiled) keeps Holm over the four rows of one
            # dataset; the xmask family here is now Holm over the four datasets of one contrast
            HOLM_NOTES.append(f'xmSig{K} (parked tab:xmask, rows-of-one-dataset Holm): {xm.get(f"xmSig{K}")} -> {v} (one contrast over datasets)')
    print('check macros_xmask.tex: xmSigBpr / Dns / Slk / Aone against the xmask family (differences are notes since 18-3)')
    for d in DS:
        for s in SEEDS:
            if abs(W[('XRE_wd0', d, s)][0] - V[('XRE', d, s)][0]) > 1e-9 or abs(W[('MSCA_wd0', d, s)][0] - V[('Native', d, s)][0]) > 1e-9:
                bad.append(f'wdcmp wd0 arm differs from the paper CSVs on {d} s{s}')
    print('check wdcmp_perseed.csv: wd 0 arms equal xre_perseed.csv XRE and table1_perseed.csv Native')
    return bad, notes


# ------------------------------------------------------------------ main
def main():
    if '--collect-wd' in sys.argv:
        txt = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', WD_CMD], capture_output=True, text=True,
                             check=True).stdout
        assert txt.startswith('arm,dataset,seed'), txt[:200]
        (T / 'wdcmp_perseed.csv').write_text(txt, encoding='utf-8', newline='\n')
        print('wrote tables/wdcmp_perseed.csv', txt.count('\n') - 1, 'rows')
    V, SRC, MU, provisional = load()
    C = contrasts(V, provisional)
    # per-seed table
    with open(T / 'ctrl_perseed.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['row', 'label', 'dataset', 'seed', 'r20', 'n20', 'source'])
        for k, lab in ROWS + [('mslk', 'A + masked SL@$K$ (Holm family only)')] + \
                [(f'rep_{k}', f'{L} replacement (macros only)') for k, L in LOSSES]:
            for d in DS:
                for s in SEEDS:
                    w.writerow([k, lab, d, s, f'{V[(k, d, s)][0]:.4f}', f'{V[(k, d, s)][1]:.4f}', SRC[(k, d, s)]])
    # contrasts CSV: every family cell, then the forest rows
    F = forest(V, C, provisional)
    fids = {}
    for fid, grp, (a, b), fam, d, r in F:
        fids.setdefault((fam, a, b, d), (fid, grp))
    with open(T / 'ctrl_contrasts.csv', 'w', newline='', encoding='utf-8') as f:
        cols = ['kind', 'fid', 'group', 'family', 'contrast', 'a', 'b', 'dataset', 'metric', 'n', 'mean_a', 'mean_b',
                'diff', 'ci_lo', 'ci_hi', 't', 'p', 'p_holm', 'holm_reject', 'wins_a'] + [f'd_s{s}' for s in SEEDS]
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()

        def row(kind, fid, grp, fam, a, b, d, mn, r):
            w.writerow(dict(kind=kind, fid=fid, group=grp, family=fam, contrast=f'({LABEL.get(a, a)}) - ({LABEL.get(b, b)})',
                            a=a, b=b, dataset=d, metric=mn, n=r['n'], mean_a=f"{r['mean_a']:.6f}",
                            mean_b=f"{r['mean_b']:.6f}", diff=f"{r['diff']:.6f}", ci_lo=f"{r['ci_lo']:.6f}",
                            ci_hi=f"{r['ci_hi']:.6f}", t=f"{r['t']:.4f}", p=f"{r['p']:.6g}", p_holm=f"{r['p_holm']:.6g}",
                            holm_reject=r['reject'], wins_a=r['wins_a'],
                            **{f'd_s{s}': f'{x:.6f}' for s, x in zip(SEEDS, r['d'])}))
        for (fam, a, b, d, m), r in C.items():
            row('family', '', '', fam, a, b, d, dict(METRICS)[m], r)
        for fid, grp, (a, b), fam, d, r in sorted(F, key=lambda x: (x[0], DS.index(x[4]))):
            row('forest', fid, grp, fam, a, b, d, 'R@20', r)
    # tables
    tex, mean, M = table(V, C, provisional, 0, 'tab:ctrl', '!t')
    (T / 'table_ctrl.tex').write_text(tex, encoding='utf-8', newline='\n')
    tex_n, _, _ = table(V, C, provisional, 1, 'tab:ctrl_ndcg', 't')
    (T / 'appendix_ctrl_ndcg.tex').write_text(tex_n, encoding='utf-8', newline='\n')
    mc, mes = macros_ctrl(V, C, MU, provisional, mean)
    (T / 'macros_ctrl.tex').write_text(mc, encoding='utf-8', newline='\n')
    mw, W, wd = macros_wd()
    (T / 'macros_wdcmp.tex').write_text(mw, encoding='utf-8', newline='\n')
    bad, notes = checks(V, C, W)
    # report
    print('MES', 'PROVISIONAL (= E + R per seed)' if provisional else 'from mes_perseed.csv')
    for k, lab in ROWS:
        print(f'{lab:36s}', ' '.join(f"{mean[(k, d)]:.4f}" for d in DS), M.get((k, 'baby'), ''), M.get((k, 'pantry'), ''),
              M.get((k, 'scientific'), ''), M.get((k, 'sports'), ''))
    for fid, grp, (a, b), fam, d, r in sorted(F, key=lambda x: (x[0], DS.index(x[4]))):
        print(f'({fid}) {a} - {b} {d:10s} {1e3 * r["diff"]:+.2f} [{1e3 * r["ci_lo"]:+.2f}, {1e3 * r["ci_hi"]:+.2f}] '
              f'p {r["p"]:.4f} holm {r["p_holm"]:.4f} rej {r["reject"]} wins {r["wins_a"]}/5 ({fam})')
    print('MES gain over MSCA (%):', [round(x, 2) for x in mes['gain']], 'seeds up', mes['up'])
    print('wd: XRE_wd - MSCA_wd (%)', [round(x, 2) for x in wd['gain']], 'seeds', wd['seeds'], '| MSCA wd0->0.1 (%)',
          [round(x, 2) for x in wd['nat']], '| XRE_wd0 vs MSCA_wd (%)', [round(x, 2) for x in wd['ovt']], wd['ovt_seeds'])
    if wd['mes']:
        print('wd: MES_wd - MSCA_wd (%)', [round(x, 2) for x in wd['mes']['gain']], 'seeds', wd['mes']['seeds'],
              '| MES (wd0) vs MSCA_wd (%)', [round(x, 2) for x in wd['mes']['ovt']], wd['mes']['ovt_seeds'],
              '| MES_wd vs MES (%)', [round(x, 2) for x in wd['mes']['self']])
    else:
        print('wd: MES_wd rows incomplete in tables/mes_perseed.csv: wdMes* macros are tbd placeholders')
    exp = [1.00, 1.64, 2.80, 3.13]
    assert all(abs(round(x, 2) - e) < 0.006 for x, e in zip(wd['gain'], exp)) and wd['seeds'] == [5, 4, 5, 5], \
        'wdcmp numbers differ from the expected XRE_wd - MSCA_wd +1.00/+1.64/+2.80/+3.13%, seeds 5/4/5/5'
    for n in notes + HOLM_NOTES:
        print('NOTE', n)
    if bad:
        print('MISMATCH:', *bad, sep='\n  ')
        sys.exit(1)
    print('all cross-checks pass')


if __name__ == '__main__':
    main()
