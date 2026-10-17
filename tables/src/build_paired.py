"""Paired seed differences with 95% t-intervals for the contrasts that carry significance claims in the manuscript.
Numbers are never typed by hand. Trains nothing and reads no remote file.

Sources (all one run per seed, seeds 1-5):
  tables/xmask_perseed.csv   masked BPR / DNS / SL@K rows of Table tab:xmask (collect_xmask.py)
  tables/xre_perseed.csv     MSCA rows XRE, A1 (anchor term alone = w/o exchange term), wo_A1 (w/o anchor term),
                             wo_A1_masked (w/o anchor term, full view), wo_retention, wo_masked (XRE on the full view)
  tables/table1_perseed.csv  MSCA Native (the run set of tables/src/native_rows.py)
  tables/cohsmo_perseed.csv  COHESION / SMORE blocks of Table tab:main (collect_cohsmo.py)
Families (a minus b, per dataset and metric; d_s = a_s - b_s over seeds 1-5):
  xmask      Table tab:xmask (gain ladder): XRE - {A + masked DNS, A + masked SL@K, A + masked BPR, anchor term A}, Holm
             over these four rows per dataset and metric (the rule of build_table_xmask.py).
  xmask_ref  row of Table tab:xmask without a mark (unadjusted p): A - Native (the first step of the ladder; the step
             XRE - Native of the ladder is the MSCA row of family main).
  abl_fig    figure family of fig_ablation.py (marks of the exchange rows of Table tab:xmask since 2026-10-06), Holm over three contrasts per dataset (the rule of figures/src/fig_ablation.py): exchange
             XRE - w/o exchange term, masked w/o anchor term - w/o anchor term full view, native w/o anchor term - Native.
             The figure tests the native contrast with the Welch t-test (Native is a different run set); its Holm p uses
             Welch here as well, and its interval columns hold the Welch interval (test = welch).
  abl_app    appendix family of fig_ablation.py: XRE - {w/o anchor term, w/o retention, XRE on the full view}, Holm over three.
  main       Table tab:main, MSCA, COHESION and SMORE blocks: XRE - Native, paired by seed, unadjusted p (Table tab:main
             carries no significance mark). MSCA XRE = xre_perseed.csv, COHESION / SMORE = cohsmo_perseed.csv.
  main_mes   (revision 2026-10-08, MES replaces XRE) Table tab:main: MES - Native for MSCA, COHESION and SMORE, paired by
             seed, unadjusted p. MES per seed from tables/src/mes_source.py (tables/mes_perseed.csv, or the PROVISIONAL
             stand-ins xre_perseed.csv wo_A1 for MSCA and cohsmo_perseed.csv XRE for COHESION / SMORE). The prose macros
             pc{Msca,Coh,Smo}*, pcN{,Excl}{Cs,CsR,Main}, pc{Coh,Smo}{Sig,Zero}Ds and the Table tab:main group of the
             appendix table read this family; their values are wrapped in \\tbd{} (red) while a MES cell is provisional.
             Family main (XRE - Native) stays in paired_ci.csv for build_stats_audit.py (E + R + A ablation row).
Interval: mean(d) +- t_{0.975, n-1} sd(d) / sqrt(n) (paired, df = 4); p = two-sided paired t-test. Holm-adjusted p (step-down):
p_(i) = max_{j<=i} min(1, (m-j+1) p_(j)). For every contrast whose b is Native the Welch interval and p are stored as well.
Datasets: a family enters a dataset once every row of the family has seeds 1-5 there (Sports joins automatically; Table
tab:main prints every complete XRE / Native cell, Sports included). The appendix table prints the union of datasets, '--'
where a family is absent.
Checks: the Holm marks of table_xmask.tex and the holm_rejected column of ablation_tests.csv are reproduced (assert).
Outputs: tables/paired_ci.csv (every contrast, both metrics), tables/macros_paired.tex, tables/appendix_paired.tex
(label tab:paired, Recall@20, values x 10^3).
Usage (repository root): python -s tables/src/build_paired.py
"""
import csv
import math
import re
import statistics as st
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from build_table_xmask import ROWS as XM_ROWS, OPTIONAL as XM_OPTIONAL  # noqa: E402  (labels of Table tab:xmask)
import mes_source  # noqa: E402  (proposed method MES, provisional switch)

DS_ALL = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
METRICS = ['r20', 'n20']
SEEDS = [1, 2, 3, 4, 5]
ALPHA = 0.05
XM_TESTED = ['mdns', 'mslk', 'mbpr', 'A1']
LABEL = {'XRE': 'XRE', 'Native': 'MSCA','A1': 'w/o exchange term', 'wo_A1': 'w/o anchor term',
         'wo_A1_masked': 'w/o anchor term, full view', 'wo_retention': 'w/o retention', 'wo_masked': 'XRE on the full view'}
XM_LABEL = dict(XM_ROWS)
SHORT = {'exchange': 'XRE $-$ w/o exchange term', 'masked': 'w/o anchor term: masked $-$ full view',
         'native': 'w/o anchor term $-$ MSCA$^{\\dagger}$', 'anchor': 'XRE $-$ w/o anchor term',
         'retention': 'XRE $-$ w/o retention', 'fullxre': 'XRE $-$ XRE on the full view'}
# (family, contrast, a, b, test); the names of fig_ablation.py CONTRASTS
ABL = [('abl_fig', 'exchange', 'XRE', 'A1', 'paired'), ('abl_fig', 'masked', 'wo_A1', 'wo_A1_masked', 'paired'),
       ('abl_fig', 'native', 'wo_A1', 'Native', 'welch'), ('abl_app', 'anchor', 'XRE', 'wo_A1', 'paired'),
       ('abl_app', 'retention', 'XRE', 'wo_retention', 'paired'), ('abl_app', 'fullxre', 'XRE', 'wo_masked', 'paired')]


def load():
    V = {}   # (backbone, row, dataset, metric) -> {seed: value}

    def put(bb, row, ds, seed, r):
        for m in METRICS:
            V.setdefault((bb, row, ds, m), {})[int(seed)] = float(r[m])
    for r in csv.DictReader(open(ROOT / 'xmask_perseed.csv', encoding='utf-8')):
        put('MSCA', r['row'], r['dataset'], r['seed'], r)
    for r in csv.DictReader(open(ROOT / 'xre_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA':
            assert int(r['n_runs']) == 1, r      # one run per seed
            put('MSCA', r['row'], r['dataset'], r['seed'], r)
    for r in csv.DictReader(open(ROOT / 'table1_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] == 'Native':
            put('MSCA', 'Native', r['dataset'], r['seed'], r)
    for r in csv.DictReader(open(ROOT / 'cohsmo_perseed.csv', encoding='utf-8')):
        put(r['backbone'], r['row'], r['dataset'], r['seed'], r)
    for bb in ('MSCA', 'COHESION', 'SMORE'):   # MES (mes_source.py; real or provisional stand-in)
        for (d, m), sv in mes_source.perseed(bb).items():
            if m in METRICS:
                V[(bb, 'MES', d, m)] = dict(sv)
    return V


def complete(V, bb, rows, d):
    return all(sorted(V.get((bb, k, d, m), {})) == SEEDS for k in rows for m in METRICS)


def paired(x, y):
    d = [x[k] - y[k] for k in SEEDS]
    n = len(d)
    mu, sd = st.mean(d), st.stdev(d)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    p = float(stats.ttest_rel([x[k] for k in SEEDS], [y[k] for k in SEEDS]).pvalue) if sd > 0 else float(mu == 0)
    return dict(n=n, df=n - 1, diff=mu, sd=sd, lo=mu - h, hi=mu + h, p=p, wins=sum(v > 0 for v in d))


def welch(x, y):
    a, b = list(x.values()), list(y.values())
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    mu = st.mean(a) - st.mean(b)
    h = stats.t.ppf(0.975, df) * math.sqrt(va + vb)
    return dict(diff=mu, lo=mu - h, hi=mu + h, df=df, p=float(stats.ttest_ind(a, b, equal_var=False).pvalue))


def holm_adj(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def holm_reject(ps):   # the sequential rule of build_table_xmask.py / fig_ablation.py
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej = set()
    for k, i in enumerate(order):
        if ps[i] <= ALPHA / (len(ps) - k):
            rej.add(i)
        else:
            break
    return rej


def contrast(V, fam, name, bb, a, b, ds, m, test):
    x, y = V[(bb, a, ds, m)], V[(bb, b, ds, m)]
    r = paired(x, y)
    row = dict(family=fam, contrast=name, backbone=bb, a=a, b=b, dataset=ds, metric=m, test=test, n=r['n'],
               diff=r['diff'], sd_diff=r['sd'], df=r['df'], ci_lo=r['lo'], ci_hi=r['hi'], p_paired=r['p'], wins=r['wins'],
               p_welch='', ci_lo_welch='', ci_hi_welch='', df_welch='', p_holm='', holm_m='', holm_rej='')
    if b == 'Native':
        w = welch(x, y)
        row.update(p_welch=w['p'], ci_lo_welch=w['lo'], ci_hi_welch=w['hi'], df_welch=w['df'])
        if test == 'welch':
            row.update(df=w['df'], ci_lo=w['lo'], ci_hi=w['hi'])
    row['p_raw'] = row['p_welch'] if test == 'welch' else row['p_paired']
    return row


def holm_family(fam):
    ps = [r['p_raw'] for r in fam]
    rej = holm_reject(ps)
    for i, (r, pa) in enumerate(zip(fam, holm_adj(ps))):
        r.update(p_holm=pa, holm_m=len(fam), holm_rej=int(i in rej))
    return fam


def build(V):
    out, dsets = [], {}
    xm_rows = [k for k, _ in XM_ROWS if k not in XM_OPTIONAL]   # controls (single probe, full-view DNS) untested here
    dsets['xmask'] = [d for d, _ in DS_ALL if complete(V, 'MSCA', xm_rows, d)]
    for d in dsets['xmask']:
        for m in METRICS:
            out += holm_family([contrast(V, 'xmask', k, 'MSCA', 'XRE', k, d, m, 'paired') for k in XM_TESTED])
            out.append(contrast(V, 'xmask_ref', 'anchor_native', 'MSCA', 'A1', 'Native', d, m, 'paired'))
    abl_rows = sorted({x for c in ABL for x in c[2:4]})
    dsets['abl'] = [d for d, _ in DS_ALL if complete(V, 'MSCA', abl_rows, d)]
    for d in dsets['abl']:
        for m in METRICS:
            for f in ('abl_fig', 'abl_app'):
                out += holm_family([contrast(V, f, n, 'MSCA', a, b, d, m, t) for ff, n, a, b, t in ABL if ff == f])
    dsets['main'] = []
    for bb in ('MSCA', 'COHESION', 'SMORE'):
        for d, _ in DS_ALL:
            if complete(V, bb, ['XRE', 'Native'], d):
                dsets['main'] += [d] if d not in dsets['main'] else []
                out += [contrast(V, 'main', 'xre_native', bb, 'XRE', 'Native', d, m, 'paired') for m in METRICS]
    dsets['main_mes'] = []
    for bb in ('MSCA', 'COHESION', 'SMORE'):
        fam = {m: [] for m in METRICS}
        for d, _ in DS_ALL:
            if complete(V, bb, ['MES', 'Native'], d):
                dsets['main_mes'] += [d] if d not in dsets['main_mes'] else []
                for m in METRICS:
                    r = contrast(V, 'main_mes', 'mes_native', bb, 'MES', 'Native', d, m, 'paired')
                    r['provisional'] = int(mes_source.is_provisional(bb, d))
                    fam[m].append(r)
        # revision log 18-3 (single test family, tables/src/sigtest.py): MES - backbone is Holm-corrected over the datasets of
        # one backbone and metric (before 18-3: unadjusted p)
        for m in METRICS:
            out += holm_family(fam[m])
    for r in out:
        r['ci_excl_zero'] = int(r['ci_lo'] > 0 or r['ci_hi'] < 0)
        r.setdefault('provisional', '')
    return out, dsets


def check_marks(rows):
    """Holm marks of table_xmask.tex (R and N columns) and ablation_tests.csv (R@20) are reproduced."""
    tex = (ROOT / 'table_xmask.tex').read_text(encoding='utf-8')
    names = re.findall(r'\\multicolumn\{2\}\{c\}\{(\w+)\}', tex)
    dmap = {D: d for d, D in DS_ALL}
    n_xm = 0
    for k in XM_TESTED:
        # one line per dataset panel (two stacked panels for four datasets), in the order of the panel headers
        cells = [c.strip() for ln in tex.splitlines() if ln.startswith(XM_LABEL[k] + ' &')
                 for c in ln.rstrip('\\ ').split('&')[1:]]
        assert len(cells) == 2 * len(names), (k, cells)
        for j, D in enumerate(names):
            for mi, m in enumerate(METRICS):
                marked = 'blacktriangledown' in cells[2 * j + mi]
                r = next(x for x in rows if x['family'] == 'xmask' and x['b'] == k and x['dataset'] == dmap[D]
                         and x['metric'] == m)
                assert marked == (bool(r['holm_rej']) and r['diff'] > 0), (k, D, m, marked, r)
                assert (r['p_holm'] < ALPHA) == bool(r['holm_rej']), r
                n_xm += 1
    n_ab = 0
    for t in csv.DictReader(open(ROOT / 'ablation_tests.csv', encoding='utf-8')):
        r = next(x for x in rows if x['family'] in ('abl_fig', 'abl_app') and x['contrast'] == t['contrast']
                 and x['dataset'] == t['dataset'] and x['metric'] == 'r20')
        assert int(t['holm_rejected']) == r['holm_rej'], (t, r)
        assert abs(float(t['p']) - r['p_raw']) <= 1e-3 * max(r['p_raw'], 1e-6), (t, r)
        assert (r['p_holm'] < ALPHA) == bool(r['holm_rej']), r
        n_ab += 1
    return n_xm, n_ab


def fmt(v, nd=4):
    return f'{v:+.{nd}f}'.replace('-', '$-$')


def k3(v, keep_sign=False):   # x 10^3, one decimal, signed; interval bounds keep the sign of a value that rounds to 0
    s = f'{v * 1000:+.1f}'
    if s in ('+0.0', '-0.0') and not (keep_sign and v != 0):
        s = '0.0'
    return s.replace('-', '$-$')


def pfmt(p):
    return '$<$.001' if p < 0.001 else f'{p:.3f}'.replace('0.', '.', 1)


def main():
    V = load()
    rows, dsets = build(V)
    n_xm, n_ab = check_marks(rows)
    cols = ['family', 'contrast', 'backbone', 'a', 'b', 'dataset', 'metric', 'test', 'n', 'df', 'diff', 'sd_diff',
            'ci_lo', 'ci_hi', 'ci_excl_zero', 'wins', 'p_paired', 'p_welch', 'ci_lo_welch', 'ci_hi_welch', 'df_welch',
            'p_raw', 'holm_m', 'p_holm', 'holm_rej', 'provisional']
    with open(ROOT / 'paired_ci.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{r[k]:.6g}' if isinstance(r[k], float) else r[k]) for k in cols})

    def find(**kw):
        return next((r for r in rows if all(r[k] == v for k, v in kw.items())), None)

    # ---------------- macros
    out = ['% Generated by tables/src/build_paired.py from tables/paired_ci.csv. Do not edit by hand.',
           '% Paired seed differences (a - b, seeds 1-5, df 4) with 95% t-intervals; Recall@20 unless the name has Ndcg.',
           mes_source.comment(['MSCA', 'COHESION', 'SMORE'], [d for d, _ in DS_ALL])]

    def mac(n, v, prov=False):
        assert re.fullmatch(r'[A-Za-z]+', n), n
        out.append(f'\\newcommand{{\\{n}}}{{{mes_source.tbd(v, prov)}}}')
    words = 'none one two three four five six seven eight nine ten eleven twelve'.split()

    def say(n):
        return words[n] if n < len(words) else str(n)

    def pm(p):
        return f'{p:.3f}' if p >= 0.001 else '$<$0.001'
    MS = (('r20', ''), ('n20', 'Ndcg'))
    for d, D in DS_ALL:
        for m, M in MS:
            for k, K in (('mdns', 'Dns'), ('mslk', 'Slk'), ('mbpr', 'Bpr'), ('A1', 'Aone')):
                r = find(family='xmask', b=k, dataset=d, metric=m)
                if r:
                    mac(f'pc{K}{M}Lo{D}', fmt(r['ci_lo']))
                    mac(f'pc{K}{M}Hi{D}', fmt(r['ci_hi']))
                    mac(f'pc{K}{M}CI{D}', f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]")
                    mac(f'pc{K}{M}PHolm{D}', pm(r['p_holm']))
            r = find(family='xmask_ref', contrast='anchor_native', dataset=d, metric=m)
            if r:
                mac(f'pcAoneNative{M}CI{D}', f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]")
                mac(f'pcAoneNative{M}P{D}', pm(r['p_raw']))
            for c, C in (('exchange', 'Exchange'), ('masked', 'Masked'), ('native', 'Xonly'), ('anchor', 'Anchor'),
                         ('retention', 'Retention'), ('fullxre', 'Fullxre')):
                r = find(contrast=c, dataset=d, metric=m)
                if r and m == 'r20':
                    mac(f'pcAbl{C}CI{D}', f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]")
                    mac(f'pcAbl{C}PHolm{D}', pm(r['p_holm']))
            for bb, B in (('MSCA', 'Msca'), ('COHESION', 'Coh'), ('SMORE', 'Smo')):
                r = find(family='main_mes', backbone=bb, dataset=d, metric=m)   # MES - backbone (2026-10-08)
                if r:
                    mac(f'pc{B}{M}CI{D}', f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]", r['provisional'] == 1)
                    mac(f'pc{B}{M}P{D}', pm(r['p_raw']), r['provisional'] == 1)

    fams = {'Xmask': lambda r: r['family'] == 'xmask',
            'XmaskR': lambda r: r['family'] == 'xmask' and r['metric'] == 'r20',
            'Dns': lambda r: r['family'] == 'xmask' and r['b'] == 'mdns',
            'AblFig': lambda r: r['family'] == 'abl_fig' and r['metric'] == 'r20',
            'AblApp': lambda r: r['family'] == 'abl_app' and r['metric'] == 'r20',
            'Cs': lambda r: r['family'] == 'main_mes' and r['backbone'] != 'MSCA',
            'CsR': lambda r: r['family'] == 'main_mes' and r['backbone'] != 'MSCA' and r['metric'] == 'r20',
            'Main': lambda r: r['family'] == 'main_mes'}

    def count(sel):
        s = [r for r in rows if sel(r)]
        return sum(r['ci_excl_zero'] for r in s), len(s)
    for n, sel in fams.items():
        e, t = count(sel)
        mac(f'pcNExcl{n}', say(e), any(r['provisional'] == 1 for r in rows if sel(r)))   # MES families: red if provisional
        mac(f'pcN{n}', say(t))
    hol = [r for r in rows if r['holm_m'] != '']
    mac('pcNHolm', str(len(hol)))
    mac('pcNHolmRej', str(sum(r['holm_rej'] for r in hol)))
    mac('pcNExclNotRej', say(sum(1 for r in hol if r['ci_excl_zero'] and not r['holm_rej'])))
    # R0415 (06 G1-3): Recall@20 datasets whose paired 95% interval of XRE - backbone excludes zero (positive side) or
    # contains zero, for COHESION and SMORE (contribution (i) and Section IV-A h)). 'all four datasets' when every
    # complete dataset qualifies.
    def dsphrase(xs, n_all):
        if not xs:
            return 'none of the datasets'
        if len(xs) == n_all and n_all > 2:
            return f'all {say(n_all)} datasets'
        return xs[0] if len(xs) == 1 else (' and '.join(xs) if len(xs) == 2 else ', '.join(xs[:-1]) + ', and ' + xs[-1])
    sig_lines = []
    for bb, B in (('COHESION', 'Coh'), ('SMORE', 'Smo')):
        rs = [(D, find(family='main_mes', backbone=bb, dataset=d, metric='r20')) for d, D in DS_ALL]
        rs = [(D, r) for D, r in rs if r]
        up = [D for D, r in rs if r['ci_excl_zero'] and r['ci_lo'] > 0]
        zero = [D for D, r in rs if not r['ci_excl_zero']]
        down = [D for D, r in rs if r['ci_excl_zero'] and r['ci_hi'] < 0]   # MES below the backbone (flagged, no assert)
        prov = any(r['provisional'] == 1 for _, r in rs)
        mac(f'pc{B}SigDs', dsphrase(up, len(rs)), prov)
        mac(f'pc{B}ZeroDs', dsphrase(zero, len(rs)), prov)
        sig_lines.append(f"paired: {bb} MES - backbone R@20 interval above zero on {up}, containing zero on {zero}, "
                         f"below zero on {down}{' (FLIP: an interval below zero)' if down else ''}; provisional={prov} "
                         f"(contribution (i) 'exclude zero on pc{B}SigDs'; h) 'contain zero on pcCohZeroDs')")
    (ROOT / 'macros_paired.tex').write_text('\n'.join(out) + '\n', encoding='utf-8', newline='\n')
    cc = ROOT / 'claims_check.txt'
    keep = [x for x in (cc.read_text(encoding='utf-8').splitlines() if cc.exists() else []) if not x.startswith('paired:')]
    cc.write_text('\n'.join(keep + sig_lines) + '\n', encoding='utf-8', newline='\n')

    # ---------------- appendix table (Recall@20, x 10^3)
    DS = [(d, D) for d, D in DS_ALL if any(d in v for v in dsets.values())]
    groups = [('Table~\\ref{tab:xmask}, MSCA: XRE minus each row',
               [dict(family='xmask', b=k) for k in XM_TESTED]),
              ('Exchange rows of Table~\\ref{tab:xmask} and Table~\\ref{tab:ablation_full}, MSCA', [dict(contrast=c) for c in
                                                   ('exchange', 'masked', 'native', 'anchor', 'retention', 'fullxre')]),
              ('Table~\\ref{tab:main}: MES minus the backbone row', [dict(family='main_mes', backbone=bb)
                                                          for bb in ('MSCA', 'COHESION', 'SMORE')])]

    def lab(r):
        if r['family'] in ('main', 'main_mes'):
            return r['backbone']
        if r['family'].startswith('xmask'):
            return XM_LABEL[r['b']]
        return SHORT[r['contrast']]

    lines = ['% Generated by tables/src/build_paired.py from tables/paired_ci.csv. Do not edit numbers by hand.',
             mes_source.comment(['MSCA', 'COHESION', 'SMORE'], [d for d, _ in DS_ALL]),
             '\\begin{table*}[t]',
             '\\caption{Paired seed differences of test Recall@20 ($\\times 10^{3}$) behind the comparisons of '
             'Tables~\\ref{tab:xmask}, \\ref{tab:ablation_full}, and~\\ref{tab:main}. $\\Delta$ is the mean over seeds '
             '1--5 of the per-seed difference, given with its 95\\% $t$-interval (4 degrees of freedom). $p$ is Holm-adjusted '
             'within the family that sets the marks of Table~\\ref{tab:xmask} or Table~\\ref{tab:ablation_full}, and over the four '
             'datasets for MES against the backbone row in Table~\\ref{tab:main}. $^{\\dagger}$: Welch interval and test, as '
             'for the contrast with MSCA in Table~\\ref{tab:xmask}. --: no runs with seeds 1--5 on that dataset.'
             + mes_source.caption_note(['MSCA', 'COHESION', 'SMORE'], [d for d, _ in DS_ALL]) + '}',
             '\\label{tab:paired}',
             '\\centering',
             '\\scriptsize',
             '\\setlength{\\tabcolsep}{2.3pt}',
             '\\begin{tabular}{l' + 'rcr' * len(DS) + '}',
             '\\toprule',
             ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{D}}}' for _, D in DS) + ' \\\\',
             ''.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(DS))),
             'Contrast & ' + ' & '.join('$\\Delta$ & 95\\% CI & $p$' for _ in DS) + ' \\\\']
    for title, items in groups:
        lines += ['\\midrule', f'\\multicolumn{{{1 + 3 * len(DS)}}}{{l}}{{\\emph{{{title}}}}} \\\\']
        for sel in items:
            cells, label = [], None
            for d, _ in DS:
                r = find(dataset=d, metric='r20', **sel)
                if r is None:
                    cells.append('\\multicolumn{3}{c}{--}')
                    continue
                label = lab(r)
                p = r['p_holm'] if r['p_holm'] != '' else r['p_raw']
                t = mes_source.tbd if r['provisional'] == 1 else (lambda x: x)   # provisional MES contrast: red
                ci = '[' + k3(r['ci_lo'], True) + ', ' + k3(r['ci_hi'], True) + ']'
                cells.append(t(k3(r['diff'])) + ' & ' + t(ci) + ' & ' + t(pfmt(p)))
            lines.append(f'\\quad {label} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    (ROOT / 'appendix_paired.tex').write_text('\n'.join(lines), encoding='utf-8', newline='\n')

    print(f'wrote paired_ci.csv ({len(rows)} rows), macros_paired.tex, appendix_paired.tex; datasets {dsets}; '
          f'Holm marks reproduced: table_xmask.tex {n_xm} cells, ablation_tests.csv {n_ab} rows')
    for r in rows:
        print(f"{r['family']:9s} {r['backbone']:8s} {r['a']:6s}-{r['b']:13s} {r['dataset']:10s} {r['metric']} "
              f"d={r['diff']:+.5f} [{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}] excl={r['ci_excl_zero']} wins={r['wins']} "
              f"p={r['p_raw']:.4f} holm={r['p_holm'] if r['p_holm'] == '' else round(r['p_holm'], 4)} rej={r['holm_rej']}")
    for n, sel in fams.items():
        print(n, 'CI excludes zero / total:', count(sel))


if __name__ == '__main__':
    main()
