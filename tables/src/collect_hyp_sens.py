"""Hypothesis figure fig_h_sens: sensitivity of MSCA + XRE to the weight lambda and the list size K.

Hypothesis. The gain of XRE over MSCA does not hinge on tuning: around the adopted setting (lambda 0.1, K 40) every
setting keeps test Recall@20 above the MSCA backbone, unless the data below say otherwise (reported as they are).
Data (the runs of supplementary Table tab:sensitivity, tables/src/build_table_xre_appendix.py)
  tables/xre_perseed.csv   (tables/src/collect_xre.py, REPLICATES = 'rep1', one run per seed, package xres)
      rows XRE (lambda 0.1, K 40), sens_l003_k40 (lambda 0.03), sens_l030_k40 (lambda 0.3), sens_l010_k20 (K 20),
      sens_l010_k80 (K 80). Seeds 1-5 for every setting (lambda 0.3 and K 80 seeds 4-5 added 2026-10-07, HOST_A qxs3/A/hA_s45).
  tables/table1_perseed.csv (tables/src/collect_table1.py) row MSCA Native, seeds 1-5 (the Table I backbone runs).
For each dataset (Baby, Pantry, Scientific) and setting
  rel = 100 (R_s / mean_MSCA - 1) per seed s, where mean_MSCA = MSCA mean over seeds 1-5; reported as mean and sample SD
        over the seeds of the setting (the same scale as the raw Recall@20 divided by one constant per dataset).
  Welch t-test of the setting against MSCA (different run sets), seeds above the MSCA mean, and the difference to the
  adopted setting (XRE) in percent of mean_MSCA.
Writes tables/hyp_sens.csv and tables/macros_hyp_sens.tex:
  hySens<Setting><DS>  rel mean (%, 1 dp); Setting in LamLow (0.03), LamMid (0.1, K 40 = XRE), LamHigh (0.3), KLow (20),
                       KHigh (80); DS in Baby, Pantry, Scientific;  hySens<Setting><DS>P  Welch p (2 dp)
  hySensAbove          "<cells with rel mean > 0>/<cells>" over the 5 distinct settings x 3 datasets
  hySensMin / hySensMax  smallest / largest rel mean over those cells;  hySensBelowCells  list of the cells below MSCA
Usage: python -s tables/src/collect_hyp_sens.py      (then python -s figures/src/fig_h_sens.py)
"""
import csv
import statistics as st
import sys
from pathlib import Path

from scipy import stats

TABLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TABLES / 'src'))
from collect_xre import check_setting  # noqa: E402

XRE = TABLES / 'xre_perseed.csv'
T1 = TABLES / 'table1_perseed.csv'
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific')]
# (group, lambda, K, row of xre_perseed, macro name)
SETTINGS = [('lambda', 0.03, 40, 'sens_l003_k40', 'LamLow'), ('lambda', 0.1, 40, 'XRE', 'LamMid'),
            ('lambda', 0.3, 40, 'sens_l030_k40', 'LamHigh'),
            ('K', 0.1, 20, 'sens_l010_k20', 'KLow'), ('K', 0.1, 40, 'XRE', 'KMid'), ('K', 0.1, 80, 'sens_l010_k80', 'KHigh')]
SEEDS = [1, 2, 3, 4, 5]


def main():
    check_setting()
    xre, nat = {}, {}
    with open(XRE, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['backbone'] == 'MSCA':
                xre[(r['row'], r['dataset'], int(r['seed']))] = float(r['r20'])
    with open(T1, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['backbone'] == 'MSCA' and r['row'] == 'Native':
                nat[(r['dataset'], int(r['seed']))] = float(r['r20'])
    out = []
    for ds, D in DATASETS:
        base = [nat[(ds, s)] for s in SEEDS]
        mb = st.mean(base)
        ref = [xre[('XRE', ds, s)] for s in SEEDS]
        for group, lam, k, row, name in SETTINGS:
            x = [(s, xre[(row, ds, s)]) for s in SEEDS if (row, ds, s) in xre]
            v = [e for _, e in x]
            rel = [100 * (e / mb - 1) for e in v]
            p = float(stats.ttest_ind(v, base, equal_var=False).pvalue)
            out.append(dict(dataset=ds, group=group, lam=lam, K=k, row=row, macro=name, n=len(v),
                            seeds='/'.join(str(s) for s, _ in x), r20_mean=f'{st.mean(v):.6f}', r20_sd=f'{st.stdev(v):.6f}',
                            msca_mean=f'{mb:.6f}', msca_sd=f'{st.stdev(base):.6f}',
                            rel_mean=f'{st.mean(rel):.3f}', rel_sd=f'{st.stdev(rel):.3f}',
                            msca_rel_sd=f'{100 * st.stdev(base) / mb:.3f}',
                            seeds_above=sum(e > mb for e in v), welch_p=f'{p:.4f}',
                            vs_xre_pct=f'{100 * (st.mean(v) - st.mean(ref)) / mb:.3f}'))
    with open(TABLES / 'hyp_sens.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(out)
    mac = []
    cells = []
    for r in out:
        D = dict(DATASETS)[r['dataset']]
        rm, pv = float(r['rel_mean']), float(r['welch_p'])
        mac += [(f"hySens{r['macro']}{D}", ('$-$' if rm < 0 else '') + f'{abs(rm):.1f}'),
                (f"hySens{r['macro']}{D}P", '$<$0.01' if pv < 0.005 else f'{pv:.2f}'),
                (f"hySens{r['macro']}{D}N", str(r['n']))]
        if r['macro'] != 'KMid':   # KMid = LamMid = XRE: count the adopted setting once
            cells.append(r)
    rel = [float(r['rel_mean']) for r in cells]
    below = [f"{dict(DATASETS)[r['dataset']]} ($\\lambda={r['lam']}$, $K={r['K']}$)" for r in cells if float(r['rel_mean']) <= 0]
    from numwords import k_of_n   # prose ratio as 'k of n' (user rule 2026-10-07)
    mac += [('hySensAbove', k_of_n(sum(x > 0 for x in rel), len(rel))),
            ('hySensMin', ('$-$' if min(rel) < 0 else '') + f'{abs(min(rel)):.1f}'), ('hySensMax', f'{max(rel):.1f}'),
            ('hySensBelowCells', (', '.join(below) + ' in the E + R + A design') if below else 'none')]   # revision log 18-4: not the tuned MES
    lines = ['% Generated by tables/src/collect_hyp_sens.py from tables/xre_perseed.csv and tables/table1_perseed.csv.',
             '% Do not edit by hand. Test Recall@20 change of MSCA + XRE against the MSCA mean over seeds 1-5 (%, 1 dp);',
             '% P = Welch t-test against MSCA, N = seeds of the setting (five for every setting).']
    lines += [f'\\newcommand{{\\{k}}}{{{x}}}' for k, x in mac]
    (TABLES / 'macros_hyp_sens.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    for r in out:
        print(r['dataset'], r['lam'], r['K'], r['n'], r['rel_mean'], r['rel_sd'], r['seeds_above'], r['welch_p'], r['vs_xre_pct'])
    print('above', sum(x > 0 for x in rel), len(rel), 'below', below)


if __name__ == '__main__':
    main()
