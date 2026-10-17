"""Single test family of the manuscript (revision log 18-3, 2026-10-16). Every generator that prints a significance mark or a
significance count imports this module, so that one rule holds for every table, figure, and sentence:

  test    two-sided paired t-test over seeds 1-5, pairing the two runs of the same seed (scipy ttest_rel, df = n - 1)
  family  one contrast (a row against its reference, one metric) across the datasets of that table or figure
  rule    Holm step-down over the family, alpha 0.05; a cell is significant when its Holm-adjusted p is <= 0.05

Tests that cannot be written as a paired t-test (the permutation and decoy nulls of N') keep their own statistic, but their
p-values are Holm-corrected over datasets with holm() below. RULE_TEXT is the wording used by the captions.
Numbers are never typed by hand.
"""
import csv
import math
import statistics as st

from scipy import stats

ALPHA = 0.05
SEEDS = [1, 2, 3, 4, 5]
RULE_TEXT = ('two-sided paired $t$-test over seeds 1--5, Holm correction over the datasets of each contrast, $p<0.05$')
COLS = ['family', 'contrast', 'a', 'b', 'dataset', 'metric', 'n', 'mean_a', 'mean_b', 'diff', 'ci_lo', 'ci_hi', 't', 'p',
        'm', 'p_holm', 'reject', 'wins_a']


def paired(x, y):
    """x, y: per-seed values in the same seed order (lists) or {seed: value} dicts with the same seeds."""
    if isinstance(x, dict):
        assert sorted(x) == sorted(y), (sorted(x), sorted(y))
        ks = sorted(x)
        x, y = [x[k] for k in ks], [y[k] for k in ks]
    assert len(x) == len(y) >= 2
    d = [p - q for p, q in zip(x, y)]
    n = len(d)
    mu, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = float(stats.ttest_rel(x, y).pvalue) if sd > 0 else float(mu == 0)
    return dict(n=n, mean_a=st.mean(x), mean_b=st.mean(y), diff=mu, sd=sd, ci_lo=mu - h, ci_hi=mu + h,
                t=mu / se if se > 0 else 0.0, p=p, wins_a=sum(1 for a, b in zip(x, y) if a > b), d=d)


def holm(ps):
    """Holm step-down adjusted p-values (same order as ps)."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def family(results):
    """results: list of dicts with key 'p' (one per dataset of one contrast); adds m, p_holm and reject in place."""
    for r, pa in zip(results, holm([r['p'] for r in results])):
        r['m'] = len(results)
        r['p_holm'] = pa
        r['reject'] = int(pa <= ALPHA)
    return results


def contrast(per_a, per_b, datasets, fam='', name='', a='', b='', metric=''):
    """One family: per_a / per_b map dataset -> {seed: value}; returns the list of results over datasets with Holm."""
    out = []
    for d in datasets:
        r = paired(per_a[d], per_b[d])
        r.update(family=fam, contrast=name, a=a, b=b, dataset=d, metric=metric)
        out.append(r)
    return family(out)


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{r[k]:.6g}' if isinstance(r.get(k), float) else r.get(k, '')) for k in COLS})
