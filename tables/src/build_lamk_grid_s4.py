"""S4 step 4 (task 1): lambda x K tuning grid of MES on MSCA (post-hoc documentation; nothing here selects anything).
Inputs
  tables/src/data/lamk_scan_s4.json   scan of every xres0 result.json under P04_mestune/runs_mestune, P04_xen0/runs_xen0 and
                                      run_data/r20/MANU/runs (HOST_A) made by tables/src/scan_tune_s4.py (key tb_r20 =
                                      trainer_best_valid recall@20, the selection metric of collect_mes.py; bt_r20 = test R@20)
  tables/mes_perseed.csv              rows MES (tuned, SELECTED of collect_mes.py) and MES_k40 (fixed lambda 0.1, K 40)
Outputs (tables/src/data/)
  lamk_grid_s4.csv           one row per (dataset, lambda, K) of seed 1: validation R@20 (selection metric), selected flag,
                             validation rank, and, where a five-seed set exists (selected / fixed row), mean test R@20
  lamk_fixed_vs_tuned_s4.csv one row per dataset: five-seed test Recall@20 of the tuned row (MES) and the fixed row (MES_k40),
                             difference, paired t-test over seeds (post hoc, unadjusted)
Usage: python -s tables/src/build_lamk_grid_s4.py
"""
import csv, json, re, statistics as st
from pathlib import Path
from scipy import stats

T = Path(__file__).resolve().parents[1]
D = T / 'src' / 'data'
SELECTED = {'baby': 'l030_k40', 'pantry': 'l010_k20', 'scientific': 'l020_k40', 'sports': 'l010_k80'}   # collect_mes.py
DS = ['baby', 'pantry', 'scientific', 'sports']

scan = json.load(open(D / 'lamk_scan_s4.json', encoding='utf-8'))
s1 = {}
for r in scan:
    if r['status'] != 'SUCCESS' or r['mode'] != 'train' or r['te'] != 'best_primed' or 'wd01' in r['cfg'] or int(r['seed']) != 1:
        continue
    m = re.search(r'_xeoa0_l(\d{3})_k(\d+)$', r['cfg'])
    if not m:
        continue
    key = (r['ds'], int(m.group(1)) / 100, int(m.group(2)))
    assert key not in s1, ('replicate', key)
    s1[key] = r

ps = {}
for r in csv.DictReader(open(T / 'mes_perseed.csv', encoding='utf-8')):
    if r['backbone'] == 'MSCA' and r['row'] in ('MES', 'MES_k40'):
        ps.setdefault((r['row'], r['dataset']), {})[int(r['seed'])] = float(r['r20'])


def mean5(row, ds):
    v = ps.get((row, ds), {})
    return (sum(v.values()) / 5, st.stdev(v.values())) if len(v) == 5 else (None, None)


rows = []
for ds in DS:
    cells = sorted((k for k in s1 if k[0] == ds), key=lambda k: (k[1], k[2]))
    ranked = sorted(cells, key=lambda k: -s1[k]['tb_r20'])
    sel = SELECTED[ds]
    lam_s, k_s = int(sel[1:4]) / 100, int(sel.split('_k')[1])
    best = ranked[0]
    assert (best[1], best[2]) == (lam_s, k_s), ('argmax differs from SELECTED', ds, best, sel)
    for k in cells:
        r = s1[k]
        is_sel = int((k[1], k[2]) == (lam_s, k_s))
        is_fix = int((k[1], k[2]) == (0.1, 40))
        row = 'MES' if is_sel else ('MES_k40' if is_fix else '')
        m5, s5 = mean5(row, ds) if row else (None, None)
        if is_sel and is_fix:
            m5, s5 = mean5('MES', ds)
        rows.append(dict(dataset=ds, lam=k[1], K=k[2], seed1_valid_r20=f"{r['tb_r20']:.4f}", valid_rank=ranked.index(k) + 1, selected=is_sel,
                         fixed_setting=is_fix, seed1_test_r20_not_used=f"{r['bt_r20']:.4f}",
                         mean5_test_r20='' if m5 is None else f'{m5:.5f}', sd5_test_r20='' if s5 is None else f'{s5:.5f}',
                         valid_gap_to_best=f"{best and (s1[best]['tb_r20'] - r['tb_r20']):.4f}", run_dir=r['run']))
cols = list(rows[0].keys())
with open(D / 'lamk_grid_s4.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, cols, lineterminator='\n'); w.writeheader(); w.writerows(rows)

out = []
for ds in DS:
    a = ps[('MES', ds)]; b = ps[('MES_k40', ds)]
    seeds = range(1, 6)
    x = [a[s] for s in seeds]; y = [b[s] for s in seeds]
    t, p = stats.ttest_rel(x, y) if x != y else (float('nan'), float('nan'))
    sel = SELECTED[ds]
    out.append(dict(dataset=ds, tuned_lam=int(sel[1:4]) / 100, tuned_K=int(sel.split('_k')[1]), fixed_lam=0.1, fixed_K=40,
                    tuned_is_fixed=int(sel == 'l010_k40'), tuned_mean5=f'{sum(x) / 5:.5f}', tuned_sd5=f'{st.stdev(x):.5f}',
                    fixed_mean5=f'{sum(y) / 5:.5f}', fixed_sd5=f'{st.stdev(y):.5f}', diff_tuned_minus_fixed=f'{(sum(x) - sum(y)) / 5:+.5f}',
                    wins_tuned=sum(i > j for i, j in zip(x, y)), paired_t=f'{t:.3f}', p_two_sided_post_hoc=f'{p:.4f}'))
with open(D / 'lamk_fixed_vs_tuned_s4.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(out[0].keys()), lineterminator='\n'); w.writeheader(); w.writerows(out)
print(open(D / 'lamk_grid_s4.csv', encoding='utf-8').read())
print(open(D / 'lamk_fixed_vs_tuned_s4.csv', encoding='utf-8').read())
