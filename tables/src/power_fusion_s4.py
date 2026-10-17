"""S4 step 4 (task 3), POST HOC: power, minimum detectable effect, and equivalence (TOST) of the post-fusion margin
MSCA + MES (+)F minus MSCA (+)F per dataset, on test Recall@20, paired over seeds 1-5.  Nothing here was pre-registered.

Sources
  tables/src/data/fusion_cells_s4.csv   REFUSE cells (study/REFUSE): tf_fuse.py re-run on the Table-II checkpoints; a row whose
                                        source starts with CKPT_ABSENT has no checkpoint on HOST_A (Table-II MSCA seeds 2-5 of Baby,
                                        Pantry, Scientific were trained on a pod; only result.json survives) and carries no metrics
  tables/src/data/fusion_cells.csv      original TABLE_FUSION cells (MSCA from separately trained P04_listana runs)
Three views of the same margin
  strict  both rows from REFUSE cells with Table-II checkpoints, paired seeds only (n < 5 where a checkpoint is absent)
  mixed   MSCA (+)F from REFUSE where it exists, else from the original TABLE_FUSION cell of that seed (flagged); MSCA+MES (+)F
          from REFUSE (its checkpoints are the Table-II ones)
  old     original TABLE_FUSION cells only (the numbers of the current manuscript)
Per dataset: mean paired difference d, SD, SE, 95% CI half-width t_{.975,n-1} SE, two-sided paired t (p), minimum detectable
effect at 80% power (two-sided paired t, alpha .05, same n, using the observed SD of the differences; noncentral t), and TOST
equivalence with margin +-0.002 Recall@20 (p_TOST = max of the two one-sided p; passes at alpha .05 iff p_TOST < .05).  The smallest
margin for which TOST passes at alpha .05 equals |d| + t_{.95,n-1} SE (the 90% CI limit nearest to the margin).
Usage: python -s tables/src/power_fusion_s4.py   ->  tables/src/data/power_fusion_s4.json
"""
import csv, json, math
from pathlib import Path
from scipy import stats, optimize

D = Path(__file__).resolve().parent / 'data'
DS = ['baby', 'pantry', 'scientific', 'sports']
MARGIN, ALPHA, POWER = 0.002, 0.05, 0.80


def load(p):
    out = {}
    for r in csv.DictReader(open(p, encoding='utf-8')):
        if r['model'] not in ('MSCA', 'MSCA+MES'):
            continue
        absent = r['source'].startswith('CKPT_ABSENT') or not r['fused_r20']
        out[(r['model'], r['dataset'], int(r['seed']))] = None if absent else dict(f=float(r['fused_r20']), b=float(r['base_r20']), gate=int(r['gate_ok']), src=r['source'])
    return out


def mde(sd, n, alpha=ALPHA, power=POWER):
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)
    g = lambda nc: (1 - stats.nct.cdf(tc, df, nc)) + stats.nct.cdf(-tc, df, nc) - power
    nc = optimize.brentq(g, 0.5, 7)
    return nc * sd / math.sqrt(n), nc


def analyse(d):
    n = len(d)
    if n < 3:
        return dict(n=n, note='n < 3: no inference', d=d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    hw = stats.t.ppf(0.975, df) * se
    t, p = (m / se, 2 * (1 - stats.t.cdf(abs(m / se), df))) if se > 0 else (float('nan'), float('nan'))
    mde_v, nc = mde(sd, n)
    p_lo = 1 - stats.t.cdf((m + MARGIN) / se, df)       # H0: mu <= -margin
    p_hi = stats.t.cdf((m - MARGIN) / se, df)           # H0: mu >= +margin
    p_tost = max(p_lo, p_hi)
    min_margin = abs(m) + stats.t.ppf(1 - ALPHA, df) * se
    return dict(n=n, d_seeds=d, mean=m, sd=sd, se=se, ci95_halfwidth=hw, ci95=[m - hw, m + hw], t=t, p_two_sided=p,
                mde_80=mde_v, mde_noncentrality=nc, observed_over_mde=(abs(m) / mde_v if mde_v else None),
                tost_margin=MARGIN, tost_p_lower=p_lo, tost_p_upper=p_hi, tost_p=p_tost, tost_pass=bool(p_tost < ALPHA),
                tost_min_margin_pass=min_margin)


def main():
    new, old = load(D / 'fusion_cells_s4.csv'), load(D / 'fusion_cells.csv')
    res = {'label': 'POST HOC, not pre-registered', 'measure': 'test Recall@20, MSCA+MES(+)F minus MSCA(+)F, paired over seeds', 'alpha': ALPHA,
           'power': POWER, 'margin': MARGIN, 'views': {}}
    for view in ('strict', 'mixed', 'old'):
        res['views'][view] = {}
        for ds in DS:
            d, flags = [], []
            for s in range(1, 6):
                if view == 'old':
                    a, b, fl = old[('MSCA+MES', ds, s)], old[('MSCA', ds, s)], ''
                elif view == 'strict':
                    a, b, fl = new[('MSCA+MES', ds, s)], new[('MSCA', ds, s)], ''
                    if a is None or b is None:
                        continue
                else:
                    a, b = new[('MSCA+MES', ds, s)], new[('MSCA', ds, s)]
                    fl = ''
                    if b is None:
                        b, fl = old[('MSCA', ds, s)], 'MSCA from original TABLE_FUSION cell'
                    if a is None:
                        a, fl = old[('MSCA+MES', ds, s)], fl + ' MES from original cell'
                d.append(a['f'] - b['f'])
                flags.append(fl)
            r = analyse(d)
            r['flags'] = flags
            res['views'][view][ds] = r
    json.dump(res, open(D / 'power_fusion_s4.json', 'w', encoding='utf-8'), indent=1)
    for view, v in res['views'].items():
        print(view)
        for ds, r in v.items():
            if 'mean' not in r:
                print('  ', ds, 'n', r['n'], r.get('note'))
                continue
            print(f"  {ds:11s} n={r['n']} d={r['mean']:+.5f} sd={r['sd']:.5f} CI95hw={r['ci95_halfwidth']:.5f} p={r['p_two_sided']:.3f} "
                  f"MDE80={r['mde_80']:.5f} TOST(.002) p={r['tost_p']:.3f} pass={r['tost_pass']} minmargin={r['tost_min_margin_pass']:.5f}")


if __name__ == '__main__':
    main()
