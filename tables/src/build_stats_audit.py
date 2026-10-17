"""Statistics audit of the main contrasts (review R0305, AE 2). Numbers are never typed by hand. Trains nothing and reads
no remote file.

Contrasts (a - b of test Recall@20 and NDCG@20, the same seed in a and b, seeds 1-5, one run per seed):
  rows (2), (5), (6), (7), (9), (10), (11), (12) of Fig. h_forest (the numbering of tables/src/collect_hyp_forest.py) and
  TableII = MSCA + XRE - MSCA, the MSCA pair of Table tab:sota (= the MSCA block of Table tab:main, same runs).
Sources (per-seed run tables): tables/xmask_perseed.csv (mbpr, mdns, mslk), tables/xre_perseed.csv (MSCA rows XRE, A1,
  wo_A1, wo_A1_masked, wo_masked, wo_retention), tables/m1probe_perseed.csv (A1first), tables/e1_perseed.csv (mdnsfv),
  tables/table1_perseed.csv (MSCA Native).
Statistics per contrast, dataset, and metric: mean of d_s, 95% paired t-interval mean +- t_{0.975,4} sd / sqrt(5)
  (unadjusted, the interval of Fig. h_forest), nominal two-sided p (paired t-test, Welch for the one unpaired member of
  family T3-dagger), and the Holm-adjusted p within the family named in column holm_family:
  T3-down    XRE - {A + masked DNS, A + masked SL@K, A + masked BPR, A}, per dataset and metric (m = 4); the triangle marks
             of Table tab:xmask and the significance words of the abstract, contribution (ii), and the conclusion for
             rows (2), (5), (10).
  T3-dagger  {XRE - A, Exchange (masked) - Exchange (full view), Exchange (masked) - MSCA (Welch)}, per dataset and
             metric (m = 3); the dagger marks of Table tab:xmask (row (6)).
  S6         XRE - {w/o anchor term, w/o retention, XRE on full view}, per dataset and metric (m = 3); the marks of
             supplementary Table tab:ablation_full (row (7)).
  datasets   the datasets of the contrast, per metric (m = 4, or 3 for rows (11), (12)); rows (9), (11), (12), and TableII.
  Holm (step-down): p_(i) = max_{j<=i} min(1, (m-j+1) p_(j)); significant = adjusted p < 0.05.
Cross-checks (assert): Recall@20 means and intervals equal tables/hyp_forest.csv; p and Holm p of rows (2), (5), (10)
equal family xmask, row (6) family abl_fig, row (7) family abl_app, and TableII family main of tables/paired_ci.csv.
Output: tables/stats_audit.csv. Usage (repository root): python -s tables/src/build_stats_audit.py
"""
import csv
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

TABLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TABLES / 'src'))

OUT = TABLES / 'stats_audit.csv'
DS = ['baby', 'pantry', 'scientific', 'sports']
SEEDS = [1, 2, 3, 4, 5]
METRICS = ['r20', 'n20']
ALPHA = 0.05
XRE_ROWS = ('XRE', 'A1', 'wo_A1', 'wo_A1_masked', 'wo_masked', 'wo_retention')
LABEL = {'XRE': 'XRE', 'A1': 'A', 'mdns': 'A + masked DNS', 'mbpr': 'A + masked BPR', 'mslk': 'A + masked SL@K',
         'wo_A1': 'Exchange (masked view)', 'wo_A1_masked': 'Exchange (full view)', 'wo_masked': 'XRE on full view',
         'wo_retention': 'w/o retention', 'A1first': 'A, single probe', 'mdnsfv': 'A + full-view DNS', 'Native': 'MSCA'}
# (printed id, a, b) of every reported contrast; families list (id, a, b, test) with the reported members first
REPORTED = [('(2)', 'XRE', 'A1'), ('(5)', 'XRE', 'mdns'), ('(6)', 'wo_A1', 'wo_A1_masked'), ('(7)', 'XRE', 'wo_masked'),
            ('(9)', 'mdns', 'mbpr'), ('(10)', 'XRE', 'mbpr'), ('(11)', 'A1', 'A1first'), ('(12)', 'mdns', 'mdnsfv'),
            ('TableII', 'XRE', 'Native')]
PER_DATASET = {
    'T3-down': [('(5)', 'XRE', 'mdns', 'paired'), ('-', 'XRE', 'mslk', 'paired'), ('(10)', 'XRE', 'mbpr', 'paired'),
                ('(2)', 'XRE', 'A1', 'paired')],
    'T3-dagger': [('-', 'XRE', 'A1', 'paired'), ('(6)', 'wo_A1', 'wo_A1_masked', 'paired'),
                  ('-', 'wo_A1', 'Native', 'welch')],
    'S6': [('-', 'XRE', 'wo_A1', 'paired'), ('-', 'XRE', 'wo_retention', 'paired'), ('(7)', 'XRE', 'wo_masked', 'paired')],
}
ACROSS_DATASETS = ['(9)', '(11)', '(12)', 'TableII']


def load():
    V = {}

    def put(row, ds, seed, r):
        for m in METRICS:
            key = (row, ds, int(seed), m)
            assert key not in V, f'duplicate run {key}'   # one run per seed
            V[key] = float(r[m])
    for name in ('xmask', 'm1probe', 'e1'):
        for r in csv.DictReader(open(TABLES / f'{name}_perseed.csv', encoding='utf-8')):
            put(r['row'], r['dataset'], r['seed'], r)
    for r in csv.DictReader(open(TABLES / 'xre_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] in XRE_ROWS:
            assert int(r['n_runs']) == 1, r
            put(r['row'], r['dataset'], r['seed'], r)
    for r in csv.DictReader(open(TABLES / 'table1_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] == 'Native':
            put('Native', r['dataset'], r['seed'], r)
    return V


def has(V, rows, d, m):
    return all((x, d, s, m) in V for x in rows for s in SEEDS)


def contrast(V, a, b, d, m, test):
    xa = [V[(a, d, s, m)] for s in SEEDS]
    xb = [V[(b, d, s, m)] for s in SEEDS]
    dd = [p - q for p, q in zip(xa, xb)]
    mu, n = st.mean(dd), len(dd)
    if test == 'paired':
        sd = st.stdev(dd)
        h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        p = float(stats.ttest_rel(xa, xb).pvalue) if sd > 0 else float(mu == 0)
        lo, hi = mu - h, mu + h
    else:
        p = float(stats.ttest_ind(xa, xb, equal_var=False).pvalue)
        lo = hi = float('nan')
    return dict(mean=mu, ci_lo=lo, ci_hi=hi, p=p, pos=sum(x > 0 for x in dd))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - j) * ps[i]))
        adj[i] = run
    return adj


def main():
    V = load()
    out = []
    for fam, members in PER_DATASET.items():
        for d in DS:
            for m in METRICS:
                if not has(V, {x for _, a, b, _ in members for x in (a, b)}, d, m):
                    continue
                res = [contrast(V, a, b, d, m, t) for _, a, b, t in members]
                for (cid, a, b, t), r, pa in zip(members, res, holm([r['p'] for r in res])):
                    if cid != '-':
                        out.append(dict(id=cid, a=a, b=b, test=t, dataset=d, metric=m, family=fam, m=len(members),
                                        p_holm=pa, **r))
    for cid in ACROSS_DATASETS:
        a, b = next((a, b) for c, a, b in REPORTED if c == cid)
        for m in METRICS:
            dsets = [d for d in DS if has(V, (a, b), d, m)]
            res = [contrast(V, a, b, d, m, 'paired') for d in dsets]
            for d, r, pa in zip(dsets, res, holm([r['p'] for r in res])):
                out.append(dict(id=cid, a=a, b=b, test='paired', dataset=d, metric=m, family='datasets', m=len(dsets),
                                p_holm=pa, **r))
    rank = {c: i for i, (c, _, _) in enumerate(REPORTED)}
    out.sort(key=lambda r: (rank[r['id']], METRICS.index(r['metric']), DS.index(r['dataset'])))
    # cross-checks against the files that the manuscript already reads
    forest = {(r['id'], r['dataset']): r for r in csv.DictReader(open(TABLES / 'hyp_forest.csv', encoding='utf-8'))}
    paired = {(r['family'], r['contrast'], r['backbone'], r['dataset'], r['metric']): r
              for r in csv.DictReader(open(TABLES / 'paired_ci.csv', encoding='utf-8'))}
    pmap = {'(2)': ('xmask', 'A1'), '(5)': ('xmask', 'mdns'), '(10)': ('xmask', 'mbpr'), '(6)': ('abl_fig', 'masked'),
            '(7)': ('abl_app', 'fullxre'), 'TableII': ('main', 'xre_native')}
    for r in out:
        if r['metric'] == 'r20' and r['id'] != 'TableII':
            f = forest[(r['id'].strip('()'), r['dataset'])]
            assert abs(float(f['mean']) - r['mean']) < 1e-9 and abs(float(f['ci_lo']) - r['ci_lo']) < 1e-6, (r, f)
            assert abs(float(f['p_paired']) - r['p']) < 1e-5 * max(1, r['p']) + 1e-9, (r, f)
        if r['id'] in pmap:
            fam, c = pmap[r['id']]
            q = paired[(fam, c, 'MSCA', r['dataset'], r['metric'])]
            assert abs(float(q['p_paired']) - r['p']) < 1e-5, (r, q)
            if q['p_holm']:
                assert abs(float(q['p_holm']) - r['p_holm']) < 1e-5, (r, q)
    rows = []
    for r in out:
        rows.append({'contrast': r['id'], 'a': LABEL[r['a']], 'b': LABEL[r['b']], 'dataset': r['dataset'],
                     'metric': r['metric'], 'n_seeds': 5, 'mean_diff': f"{r['mean']:+.5f}",
                     'ci95_lo': f"{r['ci_lo']:+.5f}", 'ci95_hi': f"{r['ci_hi']:+.5f}", 'seeds_a_gt_b': r['pos'],
                     'test': r['test'], 'p_nominal': f"{r['p']:.4g}", 'holm_family': r['family'], 'holm_m': r['m'],
                     'p_holm': f"{r['p_holm']:.4g}", 'significant_holm': int(r['p_holm'] < ALPHA),
                     'ci_excludes_zero': int(r['ci_lo'] > 0 or r['ci_hi'] < 0)})
    # claims of the manuscript that rest on this file (printed; a mismatch stops the build)
    claims = [  # (contrast, metric, rule, text), rule: datasets significant after Holm == datasets with CI above zero
        ('(9)', 'r20', 'Section IV-A c): both datasets of row (9) keep the verdict after Holm over its datasets'),
        ('(2)', 'r20', 'abstract S7, (ii), conclusion: row (2) significant after Holm (T3-down) where the CI excludes zero'),
        ('(10)', 'r20', '(ii), conclusion: row (10) significant after Holm (T3-down) where the CI excludes zero'),
    ]
    for cid, m, text in claims:
        sel = [r for r in rows if r['contrast'] == cid and r['metric'] == m]
        holm_set = {r['dataset'] for r in sel if r['significant_holm']}
        ci_set = {r['dataset'] for r in sel if r['ci_excludes_zero'] and float(r['mean_diff']) > 0}
        print(f'claim: {text}: holm {sorted(holm_set)} == ci {sorted(ci_set)} -> {holm_set == ci_set}')
        assert holm_set == ci_set, (cid, holm_set, ci_set)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT, len(rows), 'rows')
    for r in rows:
        if r['metric'] == 'r20':
            print(f"{r['contrast']:5s} {r['dataset']:10s} {r['mean_diff']} p={r['p_nominal']:8s} holm={r['p_holm']:8s} "
                  f"[{r['holm_family']}/{r['holm_m']}] sig={r['significant_holm']} ci={r['ci_excludes_zero']}")


if __name__ == '__main__':
    main()
